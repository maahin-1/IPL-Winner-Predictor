"""
Coach & management feature computations — prd.feature_store.coach_and_management.
MITIGATION: R4 — cutoff_date enforced on every function.
"""
from __future__ import annotations

import pandas as pd


def coach_win_rate(
    match_results_df: pd.DataFrame,
    coach: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Overall IPL win percentage under current coach (career)."""
    df = _filter_coach(match_results_df, coach, cutoff_date)
    total = len(df)
    wins = (df["result"] == "win").sum()
    return float(wins / total) if total > 0 else 0.0


def coach_chase_win_rate(
    match_results_df: pd.DataFrame,
    coach: str,
    cutoff_date: pd.Timestamp,
) -> float:
    """Win percentage when chasing under current coach (career)."""
    df = _filter_coach(match_results_df, coach, cutoff_date)
    chasing = df[df["batting_second"] == True]
    total = len(chasing)
    wins = (chasing["result"] == "win").sum()
    return float(wins / total) if total > 0 else 0.0


def coach_death_bowling_economy(
    balls_df: pd.DataFrame,
    team: str,
    coach_appointment_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """Team average death economy under current coach (season)."""
    df = balls_df[
        (balls_df["date"] >= coach_appointment_date)
        & (balls_df["date"] < cutoff_date)
        & (balls_df["season"] == season)
        & (balls_df["over"] >= 16)
    ]
    bowling_df = df[(df["team1"] == team) | (df["team2"] == team)]
    balls = len(bowling_df)
    if balls == 0:
        return 0.0
    overs = balls / 6.0
    return float(bowling_df["runs_total"].sum() / overs)


def team_strategy_fingerprint(
    balls_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """
    Aggression index: ratio of early powerplay usage (overs 1-3) to
    late powerplay usage (overs 4-6). Higher = more top-heavy aggression.
    """
    df = balls_df[
        (balls_df["date"] < cutoff_date)
        & (balls_df["season"] == season)
        & ((balls_df["team1"] == team) | (balls_df["team2"] == team))
        & (balls_df["over"] < 6)
    ]
    early = df[df["over"] < 3]["runs_batter"].sum()
    late = df[df["over"] >= 3]["runs_batter"].sum()
    return float(early / late) if late > 0 else 1.0


def lineup_stability_score(
    match_squads_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    season: int,
) -> float:
    """
    Consistency of playing XI selection across matches (season).
    match_squads_df must have columns: [team, season, match_id, players (list/set)].
    Score = average Jaccard similarity between consecutive match squads.
    """
    df = match_squads_df[
        (match_squads_df["team"] == team)
        & (match_squads_df["season"] == season)
        & (match_squads_df["date"] < cutoff_date)
    ].sort_values("date")
    if len(df) < 2:
        return 1.0
    scores = []
    squads = [set(row) for row in df["players"]]
    for a, b in zip(squads, squads[1:]):
        union = len(a | b)
        intersection = len(a & b)
        scores.append(intersection / union if union > 0 else 1.0)
    return float(pd.Series(scores).mean())


def coaching_change_impact(
    match_results_df: pd.DataFrame,
    team: str,
    change_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    lookback_days: int = 365,
) -> float:
    """Win rate delta since coaching change appointment vs prior window."""
    before_start = change_date - pd.Timedelta(days=lookback_days)
    before = match_results_df[
        (match_results_df["team"] == team)
        & (match_results_df["date"] >= before_start)
        & (match_results_df["date"] < change_date)
    ]
    after = match_results_df[
        (match_results_df["team"] == team)
        & (match_results_df["date"] >= change_date)
        & (match_results_df["date"] < cutoff_date)
    ]
    before_wr = (before["result"] == "win").mean() if len(before) > 0 else 0.0
    after_wr = (after["result"] == "win").mean() if len(after) > 0 else 0.0
    return float(after_wr - before_wr)


# ── helpers ──────────────────────────────────────────────────────────────────

def _filter_coach(df: pd.DataFrame, coach: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return df[(df["coach"] == coach) & (df["date"] < cutoff_date)]
