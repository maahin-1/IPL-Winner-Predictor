"""
Per-over inference orchestrator — feature → ML → LLM → output.
Latency budget: end-to-end < 30s per prd.system_architecture.latency_sla.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from data.ingest.cricbuzz_client import MatchState
from llm.cost_tracker import CostTracker
from llm.llm_caller import LLMCaller
from llm.prompt_builder import MatchSnapshot
from llm.response_parser import LLMOutput, FALLBACK_OUTPUT
from models.meta_learner import MetaLearner
from models.model_a_prematch import ModelA_PreMatchXGB
from models.model_b_inmatch import ModelB_InMatchLGB
from models.model_c_player_impact import ModelC_PlayerImpactXGB
from models.model_d_season import ModelD_SeasonTrajLGB
from pipeline.momentum_detector import MomentumDetector

logger = logging.getLogger(__name__)

LATENCY_BUDGET_MS = 30_000


@dataclass
class PredictionPayload:
    """Unified output object — maps to all 11 signals S1-S11."""
    match_id: str
    over: int
    # S1
    win_probability: dict[str, float]
    # S2
    season_winner_probability: dict[str, float]
    # S3
    playoff_qualification_prob: dict[str, float]
    # S4
    team_leaderboard_rank: dict[str, int]
    # S5
    player_impact_score: dict[str, float]
    # S6
    fantasy_point_projection: dict[str, float]
    # S7
    momentum_alert: Optional[dict]
    # S8
    llm_narrative: str
    # S9
    confidence_tag: str
    # S10
    key_risk_flag: str
    # S11
    fantasy_captain_pick: str
    latency_ms: float


class PredictionOrchestrator:
    def __init__(
        self,
        model_a: ModelA_PreMatchXGB,
        model_b: ModelB_InMatchLGB,
        model_c: ModelC_PlayerImpactXGB,
        model_d: ModelD_SeasonTrajLGB,
        meta_learner: MetaLearner,
        llm_caller: LLMCaller,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self._model_a = model_a
        self._model_b = model_b
        self._model_c = model_c
        self._model_d = model_d
        self._meta = meta_learner
        self._llm = llm_caller
        self._momentum = MomentumDetector()
        self._cost = cost_tracker or CostTracker()
        self._prev_win_prob: dict[str, float] = {}

    def run(
        self,
        match_state: MatchState,
        feature_vectors: dict,
    ) -> PredictionPayload:
        """
        Main per-over trigger. Composes full PredictionPayload.
        feature_vectors: pre-computed features dict keyed by model input names.
        """
        t0 = time.monotonic()

        # ── L1: Base model inference (budget: 80ms) ──────────────────────────
        pre_features = pd.DataFrame([feature_vectors.get("prematch", {})])
        live_features = pd.DataFrame([feature_vectors.get("live", {})])
        player_features = pd.DataFrame([feature_vectors.get("player", {})])
        season_features = pd.DataFrame([feature_vectors.get("season", {})])

        try:
            a_prob = self._model_a.predict_win_probability(pre_features)
        except Exception:
            a_prob = 0.5

        try:
            b_prob = self._model_b.predict_win_probability(live_features)
        except Exception:
            b_prob = 0.5

        try:
            c_score = self._model_c.predict_impact_score(player_features)
        except Exception:
            c_score = 50.0

        try:
            d_out = self._model_d.predict(season_features)
        except Exception:
            d_out = {"season_winner_probability": 0.1, "playoff_qualification_probability": 0.4}

        # ── L2: Meta-learner (budget: 20ms) ──────────────────────────────────
        meta_features = pd.DataFrame([{
            "model_a_output": a_prob,
            "model_b_output": b_prob,
            "model_c_output": c_score / 100.0,
            "model_d_output": d_out["season_winner_probability"],
            "match_phase": _over_to_phase_int(match_state.over),
            "venue_dew_flag": feature_vectors.get("venue_dew_flag", 0),
            "team_strategy_fingerprint": feature_vectors.get("team_strategy_fingerprint", 0.5),
        }])
        try:
            calibrated_prob = self._meta.predict_calibrated_win_probability(meta_features)
        except Exception:
            calibrated_prob = b_prob

        team1 = match_state.team1
        team2 = match_state.team2
        win_prob = {team1: round(calibrated_prob, 4), team2: round(1 - calibrated_prob, 4)}

        # ── Momentum detection ────────────────────────────────────────────────
        prev_p = self._prev_win_prob.get(team1, calibrated_prob)
        delta = calibrated_prob - prev_p
        self._prev_win_prob[team1] = calibrated_prob
        momentum_alert = self._momentum.check(match_state, delta)

        # ── L3: LLM (budget: 8000ms) ─────────────────────────────────────────
        snapshot = MatchSnapshot(
            match_id=match_state.match_id,
            team1=team1,
            team2=team2,
            innings=match_state.innings,
            over=match_state.over,
            score=match_state.score,
            wickets=match_state.wickets,
            target=match_state.target,
            rrr=match_state.rrr,
            run_rate=match_state.run_rate,
            batting_team=match_state.batting_team,
            bowling_team=match_state.bowling_team,
            win_probability_team1=calibrated_prob,
            key_batsmen=feature_vectors.get("key_batsmen", []),
            key_bowler=feature_vectors.get("key_bowler", ""),
            venue=feature_vectors.get("venue", ""),
            pitch_type=feature_vectors.get("pitch_type", "batting_paradise"),
            dew_flag=bool(feature_vectors.get("venue_dew_flag", 0)),
            team1_form=feature_vectors.get("team1_form", ""),
            team2_form=feature_vectors.get("team2_form", ""),
            player_impact_scores=feature_vectors.get("player_impact_scores", {}),
            momentum_delta=delta if delta != 0 else None,
        )
        llm_out: LLMOutput = self._llm.call_for_over(snapshot)

        latency_ms = (time.monotonic() - t0) * 1000
        if latency_ms > LATENCY_BUDGET_MS:
            logger.warning(
                "End-to-end latency %.0fms exceeded 30s SLA for match %s over %d",
                latency_ms, match_state.match_id, match_state.over,
            )

        return PredictionPayload(
            match_id=match_state.match_id,
            over=match_state.over,
            win_probability=win_prob,
            season_winner_probability={
                team1: d_out["season_winner_probability"],
                team2: 1 - d_out["season_winner_probability"],
            },
            playoff_qualification_prob={
                team1: d_out["playoff_qualification_probability"],
                team2: d_out.get("playoff_qualification_probability_t2", 0.5),
            },
            team_leaderboard_rank=feature_vectors.get("leaderboard_rank", {team1: 5, team2: 5}),
            player_impact_score=feature_vectors.get("player_impact_scores", {}),
            fantasy_point_projection=feature_vectors.get("fantasy_projections", {}),
            momentum_alert=momentum_alert,
            llm_narrative=llm_out.llm_narrative,
            confidence_tag=llm_out.confidence_tag,
            key_risk_flag=llm_out.key_risk_flag,
            fantasy_captain_pick=llm_out.fantasy_captain_pick,
            latency_ms=round(latency_ms, 1),
        )


def _over_to_phase_int(over: int) -> int:
    if over < 6:
        return 0  # powerplay
    if over < 16:
        return 1  # middle
    return 2      # death
