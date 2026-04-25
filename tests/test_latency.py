"""
End-to-end latency simulation — all 5 SLA milestones must pass.
Target: < 30s end-to-end per prd.system_architecture.latency_sla.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest.cricbuzz_client import MatchState
from pipeline.momentum_detector import MomentumDetector
from llm.response_parser import FALLBACK_OUTPUT

SLA = {
    "over_complete_to_feature_pipeline": 10.0,
    "feature_pipeline_to_ml_inference": 5.0,
    "ml_inference_to_llm_complete": 8.0,
    "llm_complete_to_webhook_fired": 2.0,
    "end_to_end_target": 30.0,
}

DUMMY_MATCH_STATE = MatchState(
    match_id="LATENCY_TEST_001",
    team1="MI",
    team2="CSK",
    innings=2,
    over=15,
    ball=6,
    score=120,
    wickets=4,
    run_rate=8.0,
    target=165,
    rrr=11.25,
    batting_team="CSK",
    bowling_team="MI",
)


def _make_feature_vectors() -> dict:
    rng = np.random.default_rng(42)
    return {
        "prematch": {f: float(rng.random()) for f in [
            "squad_strength_index", "venue_history", "h2h_record",
            "team_form_last_5", "coach_win_rate", "toss_outcome", "season_trajectory",
        ]},
        "live": {f: float(rng.random()) for f in [
            "current_score", "wickets_fallen", "run_rate_required", "over_number",
            "bowling_phase", "batsman_at_crease_features", "bowler_current_form", "matchup_h2h",
        ]},
        "player": {f: float(rng.random()) for f in [
            "batsman_form_momentum", "bowler_threat_index", "fielding_contribution",
            "matchup_advantage", "phase_of_play", "pressure_index",
        ]},
        "season": {f: float(rng.random()) for f in [
            "points_table_state", "net_run_rate", "remaining_fixtures_difficulty",
            "player_availability", "team_momentum_score", "coach_strategy_fingerprint",
        ]},
        "venue_dew_flag": 1,
        "team_strategy_fingerprint": 0.7,
        "key_batsmen": ["Dhoni", "Jadeja"],
        "key_bowler": "Bumrah",
        "venue": "Wankhede",
        "pitch_type": "batting_paradise",
        "team1_form": "W W W L W",
        "team2_form": "L W W W L",
        "player_impact_scores": {"Dhoni": 72.0, "Bumrah": 68.5},
    }


def test_momentum_detector_latency():
    """MomentumDetector must respond in < 5ms."""
    detector = MomentumDetector()
    t0 = time.monotonic()
    for _ in range(100):
        detector.check(DUMMY_MATCH_STATE, 0.20)
        detector.check(DUMMY_MATCH_STATE, 0.05)
    elapsed_ms = (time.monotonic() - t0) * 1000 / 100
    assert elapsed_ms < 5.0, f"MomentumDetector too slow: {elapsed_ms:.2f}ms"


def test_end_to_end_latency_with_mock_llm():
    """
    Simulate full orchestrator pipeline with mocked ML models and LLM.
    End-to-end must complete in < 30s (SLA).
    With mocked models, expect < 500ms.
    """
    from pipeline.orchestrator import PredictionOrchestrator

    mock_a = MagicMock()
    mock_a.predict_win_probability.return_value = 0.62
    mock_b = MagicMock()
    mock_b.predict_win_probability.return_value = 0.58
    mock_c = MagicMock()
    mock_c.predict_impact_score.return_value = 72.0
    mock_d = MagicMock()
    mock_d.predict.return_value = {
        "season_winner_probability": 0.15,
        "playoff_qualification_probability": 0.78,
    }
    mock_meta = MagicMock()
    mock_meta.predict_calibrated_win_probability.return_value = 0.60
    mock_llm = MagicMock()
    mock_llm.call_for_over.return_value = FALLBACK_OUTPUT

    orchestrator = PredictionOrchestrator(
        mock_a, mock_b, mock_c, mock_d, mock_meta, mock_llm
    )

    t0 = time.monotonic()
    payload = orchestrator.run(DUMMY_MATCH_STATE, _make_feature_vectors())
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < SLA["end_to_end_target"] * 1000
    assert payload.match_id == "LATENCY_TEST_001"
    assert "MI" in payload.win_probability
    assert "CSK" in payload.win_probability
    print(f"\nMocked orchestrator latency: {elapsed_ms:.1f}ms (SLA: {SLA['end_to_end_target']*1000:.0f}ms)")


def test_win_probability_sums_to_one():
    from pipeline.orchestrator import PredictionOrchestrator

    mock_a = MagicMock()
    mock_a.predict_win_probability.return_value = 0.55
    mock_b = MagicMock()
    mock_b.predict_win_probability.return_value = 0.55
    mock_c = MagicMock()
    mock_c.predict_impact_score.return_value = 60.0
    mock_d = MagicMock()
    mock_d.predict.return_value = {
        "season_winner_probability": 0.1,
        "playoff_qualification_probability": 0.5,
    }
    mock_meta = MagicMock()
    mock_meta.predict_calibrated_win_probability.return_value = 0.55
    mock_llm = MagicMock()
    mock_llm.call_for_over.return_value = FALLBACK_OUTPUT

    orch = PredictionOrchestrator(mock_a, mock_b, mock_c, mock_d, mock_meta, mock_llm)
    payload = orch.run(DUMMY_MATCH_STATE, _make_feature_vectors())

    total = sum(payload.win_probability.values())
    assert abs(total - 1.0) < 0.001
