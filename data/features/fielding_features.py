"""
Fielding & DRS feature computations — prd.feature_store.fielding_and_drs.
MITIGATION: R4 — cutoff_date enforced on every function.
"""
from __future__ import annotations

import pandas as pd


def catch_success_rate(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    window: str = "season",
    season: int | None = None,
) -> float:
    """Catches taken divided by chances created (season or career)."""
    df = _filter_fielder(balls_df, player, cutoff_date)
    if window == "season" and season is not None:
        df = df[df["season"] == season]
    chances = df[df["wicket"] == True]
    catches = df[(df["wicket"] == True) & (df["wicket_kind"] == "caught")]
    total_chances = len(chances)
    taken = len(catches[catches["fielder"] == player])
    return float(taken / total_chances) if total_chances > 0 else 0.0


def drs_success_rate_team(
    match_events_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """
    Team DRS review conversion rate.
    match_events_df must contain columns: [team, date, season, drs_taken, drs_successful].
    """
    df = match_events_df[
        (match_events_df["team"] == team)
        & (match_events_df["date"] < cutoff_date)
        & (match_events_df["season"] == season)
    ]
    total = df["drs_taken"].sum()
    successful = df["drs_successful"].sum()
    return float(successful / total) if total > 0 else 0.0


def drop_catch_impact(
    balls_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Average runs scored after a dropped catch event within the same match."""
    df = balls_df[
        (balls_df["date"] < cutoff_date)
        & (balls_df["season"] == season)
    ]
    drop_matches = df[
        (df["wicket_kind"] == "dropped_catch") &
        ((df["team1"] == team) | (df["team2"] == team))
    ]["match_id"].unique()
    if len(drop_matches) == 0:
        return 0.0

    post_drop_runs = []
    for mid in drop_matches:
        match_df = df[df["match_id"] == mid].sort_values(["innings", "over", "ball"])
        drop_idx = match_df[match_df["wicket_kind"] == "dropped_catch"].index
        for idx in drop_idx:
            pos = match_df.index.get_loc(idx)
            after = match_df.iloc[pos + 1:]
            post_drop_runs.append(after["runs_batter"].sum())

    return float(pd.Series(post_drop_runs).mean()) if post_drop_runs else 0.0


def run_out_contribution(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Direct and indirect run-outs per match."""
    df = _filter_fielder(balls_df, player, cutoff_date)
    df = df[(df["season"] == season) & (df["wicket_kind"].isin(["run out", "run_out"]))]
    n_matches = df["match_id"].nunique()
    if n_matches == 0:
        return 0.0
    return float(len(df) / n_matches)


def boundary_save_rate(
    balls_df: pd.DataFrame,
    player: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """
    Boundary prevention index from outfield.
    Approximated by: 1 - (boundaries conceded while fielding / total fielding balls).
    Requires a `boundary_saved` boolean column if available.
    """
    df = _filter_fielder(balls_df, player, cutoff_date)
    df = df[df["season"] == season]
    if "boundary_saved" in df.columns:
        total = len(df)
        saved = df["boundary_saved"].sum()
        return float(saved / total) if total > 0 else 0.0
    # fallback: return 0 — data not available without ball-tracking enrichment
    return 0.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _filter_fielder(df: pd.DataFrame, player: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return df[(df["fielder"] == player) & (df["date"] < cutoff_date)]
