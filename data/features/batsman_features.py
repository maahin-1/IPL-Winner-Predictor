"""
Batsman feature computations — all features from prd.feature_store.batsman.
MITIGATION: R4 — every function receives a cutoff_date; never uses future data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def batting_avg_recent(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    window: str = "last_10",
) -> float:
    """Batting average over last N matches (last_5 / last_10 / season)."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = _apply_window(df, window)
    innings = df.groupby("match_id")["runs_batter"].sum()
    dismissals = df[df["player_out"] == player].groupby("match_id").size()
    total_runs = innings.sum()
    total_dismissals = dismissals.sum()
    return float(total_runs / total_dismissals) if total_dismissals > 0 else float(total_runs)


def strike_rate_venue(
    balls_df: pd.DataFrame,
    player: str,
    venue: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Strike rate at the specific match venue (career window)."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = df[df["venue"] == venue]
    balls_faced = len(df)
    runs = df["runs_batter"].sum()
    return float(runs / balls_faced * 100) if balls_faced > 0 else 0.0


def strike_rate_vs_bowler_type(
    balls_df: pd.DataFrame,
    player: str,
    bowler_types: dict[str, list[str]],
    cutoff_date: pd.Timestamp,
) -> dict[str, float]:
    """SR vs pace / spin / swing bowling. bowler_types maps type → list of bowler names."""
    result = {}
    df = _filter_batter(balls_df, player, cutoff_date)
    for btype, bowlers in bowler_types.items():
        sub = df[df["bowler"].isin(bowlers)]
        balls = len(sub)
        runs = sub["runs_batter"].sum()
        result[btype] = float(runs / balls * 100) if balls > 0 else 0.0
    return result


def powerplay_contribution(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> dict[str, float]:
    """Runs and balls faced in overs 1-6 during the season."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = df[df["season"] == season]
    pp = df[df["over"] < 6]
    return {
        "runs": float(pp["runs_batter"].sum()),
        "balls_faced": float(len(pp)),
    }


def death_overs_sr(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Strike rate in overs 17-20."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = df[(df["season"] == season) & (df["over"] >= 16)]
    balls = len(df)
    return float(df["runs_batter"].sum() / balls * 100) if balls > 0 else 0.0


def boundary_pct(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Percentage of runs scored via fours and sixes."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = df[df["season"] == season]
    total_runs = df["runs_batter"].sum()
    boundary_runs = df[df["runs_batter"].isin([4, 6])]["runs_batter"].sum()
    return float(boundary_runs / total_runs) if total_runs > 0 else 0.0


def chase_avg(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Batting average when team is chasing (innings 2)."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = df[df["innings"] == 2]
    return _batting_average(df, player)


def pressure_index(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    rrr_threshold: float = 10.0,
) -> float:
    """
    Performance when team RRR exceeds threshold.
    Approximated by overs 16+ in a chase with fewer than 5 wickets left.
    """
    df = _filter_batter(balls_df, player, cutoff_date)
    pressure = df[(df["innings"] == 2) & (df["over"] >= 15)]
    return _batting_average(pressure, player)


def h2h_avg_vs_team(
    balls_df: pd.DataFrame,
    player: str,
    opponent_team: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Average runs scored against a specific opponent."""
    df = _filter_batter(balls_df, player, cutoff_date)
    df = df[
        ((df["team1"] == opponent_team) | (df["team2"] == opponent_team))
    ]
    return _batting_average(df, player)


def form_momentum(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    n_innings: int = 5,
    decay: float = 0.8,
) -> float:
    """Exponentially weighted recency score over last N innings."""
    df = _filter_batter(balls_df, player, cutoff_date)
    match_runs = (
        df.groupby(["match_id", "date"])["runs_batter"]
        .sum()
        .reset_index()
        .sort_values("date", ascending=False)
        .head(n_innings)
    )
    if match_runs.empty:
        return 0.0
    weights = np.array([decay ** i for i in range(len(match_runs))])
    return float(np.dot(match_runs["runs_batter"].values, weights) / weights.sum())


# ── helpers ──────────────────────────────────────────────────────────────────

def _filter_batter(df: pd.DataFrame, player: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return df[(df["batsman"] == player) & (df["date"] < cutoff_date)]


def _apply_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    n_map = {"last_5": 5, "last_10": 10}
    if window in n_map:
        recent_matches = (
            df.sort_values("date", ascending=False)["match_id"]
            .drop_duplicates()
            .head(n_map[window])
        )
        return df[df["match_id"].isin(recent_matches)]
    return df  # "season" or "career" → return all


def _batting_average(df: pd.DataFrame, player: str) -> float:
    innings = df.groupby("match_id")["runs_batter"].sum()
    dismissals = df[df["player_out"] == player].groupby("match_id").size()
    total_runs = innings.sum()
    total_dismissals = dismissals.sum()
    return float(total_runs / total_dismissals) if total_dismissals > 0 else float(total_runs)
