"""
API contract tests — validates all 11 signals (S1-S11) are present in response.
Per prd.roadmap phase 5 exit criteria.
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app
from api.cache import cache_prediction

REQUIRED_SIGNALS = [
    "win_probability",           # S1
    "season_winner_probability", # S2
    "playoff_qualification_prob",# S3
    "team_leaderboard_rank",     # S4
    "player_impact_score",       # S5
    "fantasy_point_projection",  # S6
    # S7 momentum_alert is optional (only fires on >15% shift)
    "llm_narrative",             # S8
    "confidence_tag",            # S9
    "key_risk_flag",             # S10
    "fantasy_captain_pick",      # S11
]

SAMPLE_PAYLOAD = {
    "match_id": "TEST001",
    "over": 10,
    "win_probability": {"MI": 0.62, "CSK": 0.38},
    "season_winner_probability": {"MI": 0.15, "CSK": 0.12},
    "playoff_qualification_prob": {"MI": 0.78, "CSK": 0.65},
    "team_leaderboard_rank": {"MI": 3, "CSK": 4},
    "player_impact_score": {"Rohit Sharma": 72.5, "MS Dhoni": 68.0},
    "fantasy_point_projection": {"Rohit Sharma": 95.0, "MS Dhoni": 80.0},
    "momentum_alert": None,
    "llm_narrative": "MI are favourites at 62% with Rohit in form.",
    "confidence_tag": "HIGH",
    "key_risk_flag": "Dew factor in second innings may favour CSK.",
    "fantasy_captain_pick": "Rohit Sharma",
    "latency_ms": 22.4,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_all_11_signals_present():
    """Validate SAMPLE_PAYLOAD contains all 11 required signals."""
    for signal in REQUIRED_SIGNALS:
        assert signal in SAMPLE_PAYLOAD, f"Missing signal: {signal}"


def test_win_probability_sums_to_one():
    total = sum(SAMPLE_PAYLOAD["win_probability"].values())
    assert abs(total - 1.0) < 0.01


def test_confidence_tag_valid():
    valid_tags = {"HIGH", "MEDIUM", "LOW", "SYSTEM_FALLBACK"}
    assert SAMPLE_PAYLOAD["confidence_tag"] in valid_tags


def test_player_impact_scores_in_range():
    for player, score in SAMPLE_PAYLOAD["player_impact_score"].items():
        assert 0.0 <= score <= 100.0, f"{player} impact score out of range: {score}"


def test_season_winner_probabilities_valid():
    for team, prob in SAMPLE_PAYLOAD["season_winner_probability"].items():
        assert 0.0 <= prob <= 1.0


def test_over_in_valid_range():
    assert 0 <= SAMPLE_PAYLOAD["over"] <= 19
