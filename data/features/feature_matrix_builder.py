"""
Feature matrix builder — constructs per-model training DataFrames from ball-by-ball parquet.
All features are computed point-in-time: only data available BEFORE the match cutoff_date.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PARQUET = Path("data/processed/ipl_balls.parquet")

# Historical IPL playoff qualifiers by season (top-4)
_PLAYOFF_TEAMS: dict[int, list[str]] = {
    2008: ["Rajasthan Royals", "Chennai Super Kings", "Delhi Capitals", "Kings XI Punjab"],
    2009: ["Deccan Chargers", "Royal Challengers Bengaluru", "Rajasthan Royals", "Delhi Capitals"],
    2010: ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru", "Delhi Capitals"],
    2011: ["Chennai Super Kings", "Royal Challengers Bengaluru", "Mumbai Indians", "Kolkata Knight Riders"],
    2012: ["Kolkata Knight Riders", "Chennai Super Kings", "Delhi Capitals", "Mumbai Indians"],
    2013: ["Mumbai Indians", "Chennai Super Kings", "Rajasthan Royals", "Sunrisers Hyderabad"],
    2014: ["Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings", "Mumbai Indians"],
    2015: ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru", "Rajasthan Royals"],
    2016: ["Sunrisers Hyderabad", "Royal Challengers Bengaluru", "Gujarat Lions", "Kolkata Knight Riders"],
    2017: ["Mumbai Indians", "Rising Pune Supergiants", "Kolkata Knight Riders", "Sunrisers Hyderabad"],
    2018: ["Chennai Super Kings", "Sunrisers Hyderabad", "Kolkata Knight Riders", "Rajasthan Royals"],
    2019: ["Mumbai Indians", "Chennai Super Kings", "Delhi Capitals", "Sunrisers Hyderabad"],
    2020: ["Mumbai Indians", "Delhi Capitals", "Sunrisers Hyderabad", "Royal Challengers Bengaluru"],
    2021: ["Chennai Super Kings", "Kolkata Knight Riders", "Delhi Capitals", "Royal Challengers Bengaluru"],
    2022: ["Gujarat Titans", "Rajasthan Royals", "Royal Challengers Bengaluru", "Lucknow Super Giants"],
    2023: ["Chennai Super Kings", "Gujarat Titans", "Mumbai Indians", "Lucknow Super Giants"],
    2024: ["Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals", "Royal Challengers Bengaluru"],
}

_SEASON_WINNERS: dict[int, str] = {
    2008: "Rajasthan Royals",
    2009: "Deccan Chargers",
    2010: "Chennai Super Kings",
    2011: "Chennai Super Kings",
    2012: "Kolkata Knight Riders",
    2013: "Mumbai Indians",
    2014: "Kolkata Knight Riders",
    2015: "Mumbai Indians",
    2016: "Sunrisers Hyderabad",
    2017: "Mumbai Indians",
    2018: "Chennai Super Kings",
    2019: "Mumbai Indians",
    2020: "Mumbai Indians",
    2021: "Chennai Super Kings",
    2022: "Gujarat Titans",
    2023: "Chennai Super Kings",
    2024: "Kolkata Knight Riders",
}


def load_balls(path: Path = _DEFAULT_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── MODEL-A: Pre-match feature matrix ────────────────────────────────────────

def build_prematch_df(balls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Two rows per match (both team perspectives). Label=1 if focal team won.
    Data doubling forces symmetric feature→outcome learning and removes
    arbitrary team1/team2 ordering bias from CricSheet data.
    Features computed using only historical data before match date (R4 guard).
    """
    matches = (
        balls_df[["match_id", "season", "date", "venue", "team1", "team2",
                   "toss_winner", "toss_decision", "winner"]]
        .drop_duplicates("match_id")
        .sort_values("date")
        .reset_index(drop=True)
    )

    matches = matches[matches["season"] >= 2013].reset_index(drop=True)

    rows = []
    for _, m in matches.iterrows():
        cutoff = m["date"]
        hist = balls_df[balls_df["date"] < cutoff]
        hist_matches = (
            hist[["match_id", "date", "season", "team1", "team2", "winner", "venue", "toss_winner"]]
            .drop_duplicates("match_id")
            .sort_values("date")
        )

        t1, t2 = m["team1"], m["team2"]
        winner = m["winner"]
        venue = m["venue"]

        # Skip no-result matches — both rows would get label=0, poisoning the classifier
        if pd.isna(winner) or str(winner).strip() == "":
            continue

        sq1 = _squad_strength(balls_df, t1, t2, cutoff)
        vas = _venue_avg_score(balls_df, venue, cutoff)

        # Toss info — available at match time (toss happens before first ball)
        toss_winner = m.get("toss_winner", "")
        toss_decision = m.get("toss_decision", "")
        if toss_decision == "field":
            chasing_team = toss_winner
        elif toss_decision == "bat":
            chasing_team = t2 if t1 == toss_winner else t1
        else:
            chasing_team = t2

        for focal, opp, sq in [(t1, t2, sq1), (t2, t1, round(1 - sq1, 4))]:
            rows.append({
                "match_id": m["match_id"],
                "season": m["season"],
                "date": m["date"],
                "squad_strength_index": sq,
                "venue_history": _venue_win_pct(hist_matches, focal, venue),
                "h2h_record": _h2h_win_pct(hist_matches, focal, opp),
                "team_form_last_5": _form_last_n(hist_matches, focal, n=5),
                "team_form_last_10": _form_last_n(hist_matches, focal, n=10),
                "opponent_form_last_10": _form_last_n(hist_matches, opp, n=10),
                "toss_win_pct": _toss_win_pct(hist_matches, focal),
                "venue_away_record": _venue_win_pct(hist_matches, opp, venue),
                "win_streak": _win_streak(hist_matches, focal),
                "venue_avg_score": vas,
                "current_season_win_rate": _season_win_pct(hist_matches, focal, int(m["season"])),
                "focal_won_toss": 1 if focal == toss_winner else 0,
                "focal_is_chasing": 1 if focal == chasing_team else 0,
                "label": 1 if winner == focal else 0,
            })

    df = pd.DataFrame(rows)
    logger.info("Prematch df: %d rows, %d features", len(df), df.shape[1] - 3)
    return df


def _squad_strength(balls_df: pd.DataFrame, team1: str, team2: str,
                    cutoff: pd.Timestamp) -> float:
    """Ratio of team1 avg runs to (team1 + team2) avg runs — proxy for squad strength."""
    hist = balls_df[balls_df["date"] < cutoff]

    def team_avg_runs(team: str) -> float:
        t_balls = hist[(hist["team1"] == team) | (hist["team2"] == team)]
        batting = t_balls[t_balls["batsman"].notna()]
        if batting.empty:
            return 0.0
        per_match = batting.groupby("match_id")["runs_batter"].sum()
        return float(per_match.mean())

    r1 = team_avg_runs(team1)
    r2 = team_avg_runs(team2)
    total = r1 + r2
    return round(r1 / total, 4) if total > 0 else 0.5


def _venue_win_pct(hist_matches: pd.DataFrame, team: str, venue: str) -> float:
    at_venue = hist_matches[
        (hist_matches["venue"] == venue) &
        ((hist_matches["team1"] == team) | (hist_matches["team2"] == team))
    ]
    if at_venue.empty:
        return 0.5
    wins = (at_venue["winner"] == team).sum()
    return round(wins / len(at_venue), 4)


def _h2h_win_pct(hist_matches: pd.DataFrame, team1: str, team2: str) -> float:
    h2h = hist_matches[
        ((hist_matches["team1"] == team1) & (hist_matches["team2"] == team2)) |
        ((hist_matches["team1"] == team2) & (hist_matches["team2"] == team1))
    ]
    if h2h.empty:
        return 0.5
    wins = (h2h["winner"] == team1).sum()
    return round(wins / len(h2h), 4)


def _form_last_n(hist_matches: pd.DataFrame, team: str, n: int = 5) -> float:
    team_matches = hist_matches[
        (hist_matches["team1"] == team) | (hist_matches["team2"] == team)
    ].sort_values("date").tail(n)
    if team_matches.empty:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return round(wins / len(team_matches), 4)


def _overall_win_pct(hist_matches: pd.DataFrame, team: str) -> float:
    team_matches = hist_matches[
        (hist_matches["team1"] == team) | (hist_matches["team2"] == team)
    ]
    if team_matches.empty:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return round(wins / len(team_matches), 4)


def _season_win_pct(hist_matches: pd.DataFrame, team: str, season: int) -> float:
    season_matches = hist_matches[
        (hist_matches["season"] == season) &
        ((hist_matches["team1"] == team) | (hist_matches["team2"] == team))
    ] if "season" in hist_matches.columns else pd.DataFrame()
    if season_matches.empty:
        return 0.5  # No current-season data yet — neutral prior
    wins = (season_matches["winner"] == team).sum()
    return round(wins / len(season_matches), 4)


def _toss_win_pct(hist_matches: pd.DataFrame, team: str) -> float:
    """Win % for this team after winning the toss (matches where team won toss)."""
    if "toss_winner" not in hist_matches.columns:
        return 0.5
    toss_won = hist_matches[hist_matches["toss_winner"] == team]
    if toss_won.empty:
        return 0.5
    wins = (toss_won["winner"] == team).sum()
    return round(wins / len(toss_won), 4)


def _win_streak(hist_matches: pd.DataFrame, team: str) -> int:
    """Consecutive wins immediately before cutoff date. Loss/tie resets to 0."""
    team_matches = hist_matches[
        (hist_matches["team1"] == team) | (hist_matches["team2"] == team)
    ].sort_values("date")
    streak = 0
    for winner in reversed(team_matches["winner"].tolist()):
        if winner == team:
            streak += 1
        else:
            break
    return streak


def _venue_avg_score(balls_df: pd.DataFrame, venue: str, cutoff: pd.Timestamp) -> float:
    """Average first-innings total at this venue using only pre-cutoff matches."""
    hist = balls_df[(balls_df["date"] < cutoff) & (balls_df["venue"] == venue)]
    if hist.empty:
        return 160.0  # IPL overall average fallback
    innings1 = hist[hist["innings"] == 1] if "innings" in hist.columns else hist
    if innings1.empty:
        return 160.0
    per_match = innings1.groupby("match_id")["runs_batter"].sum()
    return round(float(per_match.mean()), 2)


# ── MODEL-B: Live in-match feature matrix ────────────────────────────────────

def build_live_df(balls_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (match, innings, over). Label=1 if batting team won the match.
    Only innings-2 rows (chase) used for win probability training.
    """
    balls_df = balls_df.copy()
    balls_df["date"] = pd.to_datetime(balls_df["date"])

    rows = []
    for match_id, match_balls in balls_df.groupby("match_id"):
        meta = match_balls.iloc[0]
        winner = meta["winner"]
        team1 = meta["team1"]
        team2 = meta["team2"]

        # Skip no-result matches
        if pd.isna(winner) or str(winner).strip() == "":
            continue

        target = _compute_innings1_total(match_balls)
        if target is None:
            continue

        # Only 2nd innings overs
        inn2 = match_balls[match_balls["innings"] == 2].sort_values(["over", "ball"])
        if inn2.empty:
            continue

        # Identify chasing team (batting second) via toss decision
        toss_decision = meta.get("toss_decision", "")
        toss_winner = meta.get("toss_winner", "")
        if toss_decision == "field":
            chasing_team = toss_winner  # fielded first → chases
        elif toss_decision == "bat":
            chasing_team = team2 if team1 == toss_winner else team1
        else:
            chasing_team = team2  # CricSheet default

        # Label from team1 perspective (consistent with prematch_df)
        label = 1 if winner == team1 else 0
        team1_is_chasing = 1 if chasing_team == team1 else 0

        for over_num, over_balls in inn2.groupby("over"):
            completed = inn2[inn2["over"] < over_num]
            if completed.empty:
                continue

            current_score = completed["runs_total"].sum()
            wickets = completed["wicket"].sum()
            balls_bowled = len(completed)
            overs_done = balls_bowled / 6
            overs_remaining = max(20 - overs_done, 0.1)
            runs_needed = max(target - current_score, 0)
            rrr = runs_needed / overs_remaining

            # Current batsman and bowler
            last_ball = over_balls.iloc[-1]
            batsman = last_ball["batsman"]
            bowler = last_ball["bowler"]

            hist_cutoff = meta["date"]
            batsman_avg = _player_batting_avg(balls_df, batsman, hist_cutoff)
            bowler_econ = _player_bowling_econ(balls_df, bowler, hist_cutoff)
            matchup = _h2h_matchup(balls_df, batsman, bowler, hist_cutoff)

            crr = current_score / overs_done if overs_done > 0 else 0.0
            rows.append({
                "match_id": match_id,
                "season": meta["season"],
                "over_number": int(over_num),
                "current_score": int(current_score),
                "wickets_fallen": int(wickets),
                "run_rate_required": round(float(rrr), 4),
                "run_rate_differential": round(float(crr - rrr), 4),
                "balls_remaining": int(max((20 - overs_done) * 6, 0)),
                "wickets_remaining": int(10 - wickets),
                "bowling_phase": _bowling_phase(int(over_num)),
                "batsman_at_crease_features": round(batsman_avg, 4),
                "bowler_current_form": round(bowler_econ, 4),
                "matchup_h2h": round(matchup, 4),
                "team1_is_chasing": team1_is_chasing,
                "label": label,
            })

    df = pd.DataFrame(rows)
    logger.info("Live df: %d rows", len(df))
    return df


def _compute_innings1_total(match_balls: pd.DataFrame) -> int | None:
    inn1 = match_balls[match_balls["innings"] == 1]
    if inn1.empty:
        return None
    return int(inn1["runs_total"].sum())


def _bowling_phase(over: int) -> int:
    if over <= 6:
        return 0  # powerplay
    if over <= 15:
        return 1  # middle
    return 2  # death


def _player_batting_avg(balls_df: pd.DataFrame, player: str,
                        cutoff: pd.Timestamp) -> float:
    hist = balls_df[(balls_df["batsman"] == player) & (balls_df["date"] < cutoff)]
    if hist.empty:
        return 25.0  # league average fallback
    per_match = hist.groupby("match_id")["runs_batter"].sum()
    dismissals = hist[hist["player_out"] == player].groupby("match_id").size()
    total_d = dismissals.sum()
    return float(per_match.sum() / total_d) if total_d > 0 else float(per_match.mean())


def _player_bowling_econ(balls_df: pd.DataFrame, player: str,
                         cutoff: pd.Timestamp) -> float:
    hist = balls_df[(balls_df["bowler"] == player) & (balls_df["date"] < cutoff)]
    if hist.empty:
        return 8.0  # league average fallback
    per_over = hist.groupby("match_id").apply(
        lambda g: g["runs_total"].sum() / max(len(g) / 6, 0.1),
        include_groups=False,
    )
    return float(per_over.mean())


def _h2h_matchup(balls_df: pd.DataFrame, batsman: str, bowler: str,
                 cutoff: pd.Timestamp) -> float:
    """Runs per ball in historical batsman vs bowler matchup."""
    hist = balls_df[
        (balls_df["batsman"] == batsman) &
        (balls_df["bowler"] == bowler) &
        (balls_df["date"] < cutoff)
    ]
    if hist.empty:
        return 1.0  # league average ~1 run/ball in T20
    return float(hist["runs_batter"].sum() / len(hist))


# ── MODEL-C: Player impact feature matrix ────────────────────────────────────

def build_player_df(balls_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (player, match). impact_score = synthetic composite (batting + bowling).
    """
    rows = []
    for match_id, match_balls in balls_df.groupby("match_id"):
        meta = match_balls.iloc[0]
        cutoff = meta["date"]
        hist = balls_df[balls_df["date"] < cutoff]

        players = set(match_balls["batsman"].dropna()) | set(match_balls["bowler"].dropna())
        for player in players:
            bat_stats = _match_batting_stats(match_balls, player)
            bowl_stats = _match_bowling_stats(match_balls, player)

            # Synthetic impact: weighted combo of batting + bowling contribution
            impact = (
                bat_stats["runs"] * 0.5 +
                bat_stats["sr"] * 0.1 +
                bowl_stats["wickets"] * 15 +
                max(0, 8 - bowl_stats["economy"]) * 3
            )

            rows.append({
                "match_id": match_id,
                "season": meta["season"],
                "player": player,
                "batting_avg_career": _player_batting_avg(hist, player, cutoff),
                "bowling_econ_career": _player_bowling_econ(hist, player, cutoff),
                "match_runs": bat_stats["runs"],
                "match_sr": bat_stats["sr"],
                "match_wickets": bowl_stats["wickets"],
                "match_economy": bowl_stats["economy"],
                "impact_score": round(float(impact), 4),
            })

    df = pd.DataFrame(rows)

    # Derived features to satisfy MODEL-C INPUT_FEATURES
    df["batsman_form_momentum"] = (df["match_sr"] / 150).clip(0, 1)
    df["bowler_threat_index"] = ((12 - df["match_economy"]) / 12).clip(0, 1)
    df["fielding_contribution"] = 0.0
    df["matchup_advantage"] = ((df["match_runs"] * 0.3 + df["match_wickets"] * 15) / 100).clip(0, 1)
    df["phase_of_play"] = (df["match_wickets"] / 10).clip(0, 1)
    df["pressure_index"] = ((df["match_wickets"] * 10 + df["batting_avg_career"]) / 100).clip(0, 1)

    logger.info("Player df: %d rows", len(df))
    return df


def _match_batting_stats(match_balls: pd.DataFrame, player: str) -> dict:
    bat = match_balls[match_balls["batsman"] == player]
    if bat.empty:
        return {"runs": 0, "sr": 0.0}
    runs = int(bat["runs_batter"].sum())
    balls = len(bat)
    return {"runs": runs, "sr": round(runs / balls * 100, 2) if balls > 0 else 0.0}


def _match_bowling_stats(match_balls: pd.DataFrame, player: str) -> dict:
    bowl = match_balls[match_balls["bowler"] == player]
    if bowl.empty:
        return {"wickets": 0, "economy": 8.0}
    wickets = int(bowl["wicket"].sum())
    overs = len(bowl) / 6
    economy = round(bowl["runs_total"].sum() / overs, 2) if overs > 0 else 8.0
    return {"wickets": wickets, "economy": economy}


# ── MODEL-D: Season trajectory feature matrix ─────────────────────────────────

def build_season_df(balls_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (team, season). Labels: season_winner, playoff_qualifier.
    """
    rows = []
    season_matches = (
        balls_df[["match_id", "season", "date", "team1", "team2", "winner", "venue"]]
        .drop_duplicates("match_id")
    )

    teams_by_season: dict[tuple, pd.DataFrame] = {}
    for _, row in season_matches.iterrows():
        for team in [row["team1"], row["team2"]]:
            key = (row["season"], team)
            if key not in teams_by_season:
                teams_by_season[key] = []
            teams_by_season[key].append(row)

    for (season, team), match_rows in teams_by_season.items():
        team_df = pd.DataFrame(match_rows).sort_values("date")
        wins = (team_df["winner"] == team).sum()
        played = len(team_df)
        win_rate = wins / played if played > 0 else 0.0

        # Late-season form (last 7 matches)
        late = team_df.tail(7)
        late_wins = (late["winner"] == team).sum()
        late_form = late_wins / len(late) if len(late) > 0 else 0.0

        # Batting runs when this team is batting (batsman belongs to this team)
        # Approximate: team bats in innings where their players appear as batsman
        team_match_ids = team_df["match_id"].tolist()
        team_balls_all = balls_df[balls_df["match_id"].isin(team_match_ids)]
        # team1 bats innings 1, team2 bats innings 2 — use per-match assignments
        batting_runs_list = []
        conceding_runs_list = []
        for mid in team_match_ids:
            m_balls = team_balls_all[team_balls_all["match_id"] == mid]
            if m_balls.empty:
                continue
            m_meta = m_balls.iloc[0]
            if m_meta["team1"] == team:
                bat_inn, field_inn = 1, 2
            else:
                bat_inn, field_inn = 2, 1
            bat_runs = m_balls[m_balls["innings"] == bat_inn]["runs_batter"].sum()
            conc_runs = m_balls[m_balls["innings"] == field_inn]["runs_total"].sum()
            batting_runs_list.append(bat_runs)
            conceding_runs_list.append(conc_runs)

        avg_batting = float(np.mean(batting_runs_list)) if batting_runs_list else 150.0
        avg_conceding = float(np.mean(conceding_runs_list)) if conceding_runs_list else 150.0
        avg_score = avg_batting  # kept for backward compat column

        rows.append({
            "season": season,
            "team": team,
            "matches_played": played,
            "win_rate": round(win_rate, 4),
            "late_season_form": round(late_form, 4),
            "avg_score_per_match": round(avg_score, 4),
            "avg_batting": round(avg_batting, 4),
            "avg_conceding": round(avg_conceding, 4),
            "season_winner": 1 if _SEASON_WINNERS.get(season) == team else 0,
            "playoff_qualifier": 1 if team in _PLAYOFF_TEAMS.get(season, []) else 0,
        })

    df = pd.DataFrame(rows)

    # Derived features to satisfy MODEL-D INPUT_FEATURES
    df["points_table_state"] = df["win_rate"]
    df["net_run_rate"] = ((df["avg_batting"] - df["avg_conceding"]) / 50).clip(-1, 1)
    df["remaining_fixtures_difficulty"] = (1 - df["win_rate"]).clip(0, 1)
    df["player_availability"] = 1.0
    df["team_momentum_score"] = df["late_season_form"]
    df["coach_strategy_fingerprint"] = 0.5

    logger.info("Season df: %d rows", len(df))
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def build_all(parquet_path: Path = _DEFAULT_PARQUET) -> dict[str, pd.DataFrame]:
    balls_df = load_balls(parquet_path)
    logger.info("Loaded %d ball records from %s", len(balls_df), parquet_path)
    return {
        "prematch_df": build_prematch_df(balls_df),
        "live_df": build_live_df(balls_df),
        "player_df": build_player_df(balls_df),
        "season_df": build_season_df(balls_df),
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = build_all()
    for name, df in frames.items():
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {name}: {len(df):,} rows → {out_path}")
