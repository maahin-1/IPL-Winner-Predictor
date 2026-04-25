"""
Venue & context feature computations — prd.feature_store.venue_and_context.
MITIGATION: R4 — cutoff_date enforced on every function.
"""
from __future__ import annotations

import pandas as pd


def venue_avg_first_innings(
    balls_df: pd.DataFrame,
    venue: str,
    cutoff_date: pd.Timestamp,
    last_n_seasons: int = 3,
) -> float:
    """Average first innings score at venue over last N seasons."""
    df = _filter_venue(balls_df, venue, cutoff_date, last_n_seasons)
    df = df[df["innings"] == 1]
    if df.empty:
        return 0.0
    totals = df.groupby("match_id")["runs_total"].sum()
    return float(totals.mean())


def venue_chase_win_pct(
    match_results_df: pd.DataFrame,
    venue: str,
    cutoff_date: pd.Timestamp,
    last_n_seasons: int = 3,
) -> float:
    """Win percentage when chasing at the venue over last N seasons."""
    df = match_results_df[
        (match_results_df["venue"] == venue)
        & (match_results_df["date"] < cutoff_date)
    ]
    if last_n_seasons:
        seasons = sorted(df["season"].unique(), reverse=True)[:last_n_seasons]
        df = df[df["season"].isin(seasons)]
    chasing = df[df["batting_second"] == True]
    total = len(chasing)
    wins = (chasing["result"] == "win").sum()
    return float(wins / total) if total > 0 else 0.5


def pitch_type(
    venue_meta_df: pd.DataFrame,
    venue: str,
    season: int,
) -> str:
    """Categorical pitch type: seam_friendly / spin_friendly / batting_paradise."""
    row = venue_meta_df[
        (venue_meta_df["venue"] == venue) & (venue_meta_df["season"] == season)
    ]
    if row.empty:
        return "batting_paradise"
    return str(row.iloc[0].get("pitch_type", "batting_paradise"))


def dew_factor_flag(
    venue_meta_df: pd.DataFrame,
    venue: str,
    cutoff_date: pd.Timestamp,
    last_n_seasons: int = 3,
) -> int:
    """Binary flag: 1 if historical dew impact in evening games at this venue."""
    df = venue_meta_df[
        (venue_meta_df["venue"] == venue)
        & (venue_meta_df["date"] < cutoff_date)
    ]
    if last_n_seasons:
        seasons = sorted(df["season"].unique(), reverse=True)[:last_n_seasons]
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return 0
    return int(df["dew_reported"].mean() > 0.3)


def toss_win_advantage(
    match_results_df: pd.DataFrame,
    venue: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Win percentage for toss winner at this specific venue (career)."""
    df = match_results_df[
        (match_results_df["venue"] == venue)
        & (match_results_df["date"] < cutoff_date)
    ]
    total = len(df)
    toss_and_match_wins = (df["toss_winner"] == df["winner"]).sum()
    return float(toss_and_match_wins / total) if total > 0 else 0.5


def h2h_venue_record(
    match_results_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    venue: str,
    cutoff_date: pd.Timestamp,
) -> dict[str, int]:
    """Head-to-head record between two teams at a specific venue (career)."""
    df = match_results_df[
        (match_results_df["venue"] == venue)
        & (match_results_df["date"] < cutoff_date)
        & (
            ((match_results_df["team1"] == team_a) & (match_results_df["team2"] == team_b))
            | ((match_results_df["team1"] == team_b) & (match_results_df["team2"] == team_a))
        )
    ]
    return {
        team_a: int((df["winner"] == team_a).sum()),
        team_b: int((df["winner"] == team_b).sum()),
        "no_result": int((~df["winner"].isin([team_a, team_b])).sum()),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _filter_venue(
    df: pd.DataFrame,
    venue: str,
    cutoff_date: pd.Timestamp,
    last_n_seasons: int,
) -> pd.DataFrame:
    df = df[(df["venue"] == venue) & (df["date"] < cutoff_date)]
    if last_n_seasons:
        seasons = sorted(df["season"].unique(), reverse=True)[:last_n_seasons]
        df = df[df["season"].isin(seasons)]
    return df
