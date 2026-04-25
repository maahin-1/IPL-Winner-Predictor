"""
Feature registry — maps PRD feature names → computation functions.
All 28 features from prd.feature_store are registered here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FeatureSpec:
    name: str
    group: str
    dtype: str
    windows: list[str]
    compute_fn: Callable[..., Any]
    description: str


_REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> FeatureSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Feature '{name}' not found in registry. Check prd.feature_store.")
    return _REGISTRY[name]


def all_features() -> list[FeatureSpec]:
    return list(_REGISTRY.values())


def features_by_group(group: str) -> list[FeatureSpec]:
    return [f for f in _REGISTRY.values() if f.group == group]


# ── Registration ─────────────────────────────────────────────────────────────

from data.features.batsman_features import (
    batting_avg_recent,
    strike_rate_venue,
    strike_rate_vs_bowler_type,
    powerplay_contribution,
    death_overs_sr,
    boundary_pct,
    chase_avg,
    pressure_index,
    h2h_avg_vs_team,
    form_momentum,
)
from data.features.bowler_features import (
    economy_rate_recent,
    wicket_prob_by_phase,
    dot_ball_pct,
    bowling_at_venue,
    vs_batsman_type,
    death_specialist_score,
    pressure_wicket_rate,
    h2h_wickets_vs_team,
)
from data.features.fielding_features import (
    catch_success_rate,
    drs_success_rate_team,
    drop_catch_impact,
    run_out_contribution,
    boundary_save_rate,
)
from data.features.coach_features import (
    coach_win_rate,
    coach_chase_win_rate,
    coach_death_bowling_economy,
    team_strategy_fingerprint,
    lineup_stability_score,
    coaching_change_impact,
)
from data.features.venue_features import (
    venue_avg_first_innings,
    venue_chase_win_pct,
    pitch_type,
    dew_factor_flag,
    toss_win_advantage,
    h2h_venue_record,
)

_BATSMAN_FEATURES = [
    ("batting_avg_recent", ["last_5", "last_10", "season"], "float", batting_avg_recent,
     "Batting average over last N matches"),
    ("strike_rate_venue", ["career"], "float", strike_rate_venue,
     "Strike rate at the specific match venue"),
    ("strike_rate_vs_bowler_type", ["career", "last_2_seasons"], "dict",
     strike_rate_vs_bowler_type, "SR vs pace / spin / swing bowling"),
    ("powerplay_contribution", ["season"], "float", powerplay_contribution,
     "Runs and balls faced in overs 1-6"),
    ("death_overs_sr", ["season"], "float", death_overs_sr,
     "Strike rate in overs 17-20"),
    ("boundary_pct", ["season"], "float", boundary_pct,
     "Percentage of runs scored via fours and sixes"),
    ("chase_avg", ["career"], "float", chase_avg,
     "Batting average when team is chasing"),
    ("pressure_index", ["career"], "float", pressure_index,
     "Performance when team RRR exceeds 10"),
    ("h2h_avg_vs_team", ["career"], "float", h2h_avg_vs_team,
     "Average runs against the specific opponent"),
    ("form_momentum", ["last_5_innings"], "float", form_momentum,
     "Exponentially weighted recency score"),
]

_BOWLER_FEATURES = [
    ("economy_rate_recent", ["last_5", "last_10", "season"], "float", economy_rate_recent,
     "Economy rate over last N matches"),
    ("wicket_prob_by_phase", ["season"], "dict", wicket_prob_by_phase,
     "Wicket probability in powerplay / middle / death overs"),
    ("dot_ball_pct", ["season"], "float", dot_ball_pct,
     "Percentage of deliveries resulting in dot balls"),
    ("bowling_at_venue", ["career"], "dict", bowling_at_venue,
     "Economy and wicket rate at specific venue"),
    ("vs_batsman_type", ["career"], "dict", vs_batsman_type,
     "Economy vs left-hand and right-hand batsmen"),
    ("death_specialist_score", ["season"], "float", death_specialist_score,
     "Composite: economy + wickets in overs 17-20"),
    ("pressure_wicket_rate", ["career"], "float", pressure_wicket_rate,
     "Wickets taken when team urgently needs breakthrough"),
    ("h2h_wickets_vs_team", ["career"], "float", h2h_wickets_vs_team,
     "Total wickets taken against specific opponent"),
]

_FIELDING_FEATURES = [
    ("catch_success_rate", ["season", "career"], "float", catch_success_rate,
     "Catches taken divided by chances created"),
    ("drs_success_rate_team", ["season"], "float", drs_success_rate_team,
     "Team DRS review conversion rate"),
    ("drop_catch_impact", ["season"], "float", drop_catch_impact,
     "Runs scored after a dropped catch event"),
    ("run_out_contribution", ["season"], "float", run_out_contribution,
     "Direct and indirect run-outs per match"),
    ("boundary_save_rate", ["season"], "float", boundary_save_rate,
     "Boundary prevention index from outfield"),
]

_COACH_FEATURES = [
    ("coach_win_rate", ["career"], "float", coach_win_rate,
     "Overall IPL win percentage under current coach"),
    ("coach_chase_win_rate", ["career"], "float", coach_chase_win_rate,
     "Win percentage when chasing under current coach"),
    ("coach_death_bowling_economy", ["season"], "float", coach_death_bowling_economy,
     "Team average death economy under current coach"),
    ("team_strategy_fingerprint", ["season"], "float", team_strategy_fingerprint,
     "Aggression index: early vs late powerplay usage ratio"),
    ("lineup_stability_score", ["season"], "float", lineup_stability_score,
     "Consistency of playing XI selection across matches"),
    ("coaching_change_impact", ["since_appointment"], "float", coaching_change_impact,
     "Win rate delta since coaching change appointment"),
]

_VENUE_FEATURES = [
    ("venue_avg_first_innings", ["last_3_seasons"], "float", venue_avg_first_innings,
     "Average first innings score at venue"),
    ("venue_chase_win_pct", ["last_3_seasons"], "float", venue_chase_win_pct,
     "Win percentage when chasing at venue"),
    ("pitch_type", ["current_season"], "categorical", pitch_type,
     "Categorical: seam_friendly / spin_friendly / batting_paradise"),
    ("dew_factor_flag", ["last_3_seasons"], "binary", dew_factor_flag,
     "Binary flag for historical dew impact in evening games"),
    ("toss_win_advantage", ["career"], "float", toss_win_advantage,
     "Win percentage for toss winner at this specific venue"),
    ("h2h_venue_record", ["career"], "dict", h2h_venue_record,
     "Head-to-head record between two teams at this venue"),
]

for name, windows, dtype, fn, desc in _BATSMAN_FEATURES:
    register(FeatureSpec(name=name, group="batsman", dtype=dtype,
                         windows=windows, compute_fn=fn, description=desc))

for name, windows, dtype, fn, desc in _BOWLER_FEATURES:
    register(FeatureSpec(name=name, group="bowler", dtype=dtype,
                         windows=windows, compute_fn=fn, description=desc))

for name, windows, dtype, fn, desc in _FIELDING_FEATURES:
    register(FeatureSpec(name=name, group="fielding_and_drs", dtype=dtype,
                         windows=windows, compute_fn=fn, description=desc))

for name, windows, dtype, fn, desc in _COACH_FEATURES:
    register(FeatureSpec(name=name, group="coach_and_management", dtype=dtype,
                         windows=windows, compute_fn=fn, description=desc))

for name, windows, dtype, fn, desc in _VENUE_FEATURES:
    register(FeatureSpec(name=name, group="venue_and_context", dtype=dtype,
                         windows=windows, compute_fn=fn, description=desc))
