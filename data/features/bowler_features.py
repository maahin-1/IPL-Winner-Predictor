"""
Bowler feature computations — all features from prd.feature_store.bowler.
MITIGATION: R4 — cutoff_date enforced on every function.
"""
from __future__ import annotations

import pandas as pd


def economy_rate_recent(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    window: str = "last_10",
) -> float:
    """Economy rate over last N matches (last_5 / last_10 / season)."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = _apply_window(df, window)
    balls = len(df)
    runs = df["runs_total"].sum()
    overs = balls / 6.0
    return float(runs / overs) if overs > 0 else 0.0


def wicket_prob_by_phase(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> dict[str, float]:
    """Wicket probability in powerplay / middle / death overs."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = df[df["season"] == season]
    phases = {
        "powerplay": df[df["over"] < 6],
        "middle": df[(df["over"] >= 6) & (df["over"] < 16)],
        "death": df[df["over"] >= 16],
    }
    result = {}
    for phase, sub in phases.items():
        balls = len(sub)
        wickets = sub["wicket"].sum()
        result[phase] = float(wickets / balls) if balls > 0 else 0.0
    return result


def dot_ball_pct(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Percentage of deliveries resulting in dot balls."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = df[df["season"] == season]
    if df.empty:
        return 0.0
    dots = (df["runs_total"] == 0).sum()
    return float(dots / len(df))


def bowling_at_venue(
    balls_df: pd.DataFrame,
    player: str,
    venue: str,
    cutoff_date: pd.Timestamp,
) -> dict[str, float]:
    """Economy and wicket rate at a specific venue (career)."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = df[df["venue"] == venue]
    balls = len(df)
    if balls == 0:
        return {"economy": 0.0, "wicket_rate": 0.0}
    overs = balls / 6.0
    return {
        "economy": float(df["runs_total"].sum() / overs),
        "wicket_rate": float(df["wicket"].sum() / balls),
    }


def vs_batsman_type(
    balls_df: pd.DataFrame,
    player: str,
    handedness_map: dict[str, str],
    cutoff_date: pd.Timestamp,
) -> dict[str, float]:
    """Economy vs left-hand (LHB) and right-hand (RHB) batsmen (career)."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = df.copy()
    df["hand"] = df["batsman"].map(handedness_map).fillna("unknown")
    result = {}
    for hand in ["LHB", "RHB"]:
        sub = df[df["hand"] == hand]
        balls = len(sub)
        overs = balls / 6.0
        result[hand] = float(sub["runs_total"].sum() / overs) if overs > 0 else 0.0
    return result


def death_specialist_score(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Composite: economy + wickets in overs 17-20 (lower economy + more wickets = higher score)."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = df[(df["season"] == season) & (df["over"] >= 16)]
    balls = len(df)
    if balls == 0:
        return 0.0
    overs = balls / 6.0
    economy = df["runs_total"].sum() / overs
    wicket_rate = df["wicket"].sum() / balls * 100
    score = max(0.0, (12.0 - economy) * 5 + wicket_rate)
    return float(score)


def pressure_wicket_rate(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Wickets taken when batting team urgently needs a breakthrough (death overs, chase)."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    pressure = df[(df["innings"] == 2) & (df["over"] >= 15)]
    balls = len(pressure)
    if balls == 0:
        return 0.0
    return float(pressure["wicket"].sum() / balls)


def h2h_wickets_vs_team(
    balls_df: pd.DataFrame,
    player: str,
    opponent_team: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Total wickets taken against a specific opponent (career)."""
    df = _filter_bowler(balls_df, player, cutoff_date)
    df = df[
        ((df["team1"] == opponent_team) | (df["team2"] == opponent_team))
    ]
    return float(df["wicket"].sum())


# ── helpers ──────────────────────────────────────────────────────────────────

def _filter_bowler(df: pd.DataFrame, player: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return df[(df["bowler"] == player) & (df["date"] < cutoff_date)]


def _apply_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    n_map = {"last_5": 5, "last_10": 10}
    if window in n_map:
        recent_matches = (
            df.sort_values("date", ascending=False)["match_id"]
            .drop_duplicates()
            .head(n_map[window])
        )
        return df[df["match_id"].isin(recent_matches)]
    return df
