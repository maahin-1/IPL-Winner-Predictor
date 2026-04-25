"""
Feature computation correctness tests.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.features.batsman_features import (
    batting_avg_recent, boundary_pct, form_momentum, death_overs_sr
)
from data.features.bowler_features import (
    economy_rate_recent, dot_ball_pct, death_specialist_score
)
from data.features.venue_features import venue_avg_first_innings, toss_win_advantage


@pytest.fixture
def sample_balls_df():
    rng = np.random.default_rng(0)
    rows = []
    for match_idx in range(15):
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=match_idx * 7)
        for over in range(20):
            for ball in range(6):
                rows.append({
                    "match_id": f"M{match_idx:03d}",
                    "season": 2022,
                    "date": date,
                    "venue": "Wankhede",
                    "innings": 1 if over < 10 else 2,
                    "over": over % 10,
                    "ball": ball + 1,
                    "batsman": "PlayerA",
                    "bowler": "BowlerX",
                    "runs_batter": int(rng.integers(0, 7)),
                    "runs_extras": 0,
                    "runs_total": int(rng.integers(0, 8)),
                    "wicket": bool(rng.integers(0, 15) == 0),
                    "wicket_kind": "caught",
                    "fielder": "FielderA",
                    "player_out": "PlayerA" if rng.integers(0, 5) == 0 else "",
                    "team1": "MI",
                    "team2": "CSK",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_match_results():
    return pd.DataFrame({
        "venue": ["Wankhede"] * 20,
        "date": pd.date_range("2020-01-01", periods=20, freq="14D"),
        "season": [2020] * 10 + [2021] * 10,
        "toss_winner": ["MI"] * 10 + ["CSK"] * 10,
        "winner": ["MI"] * 12 + ["CSK"] * 8,
        "batting_second": [False, True] * 10,
        "result": ["win", "loss"] * 10,
    })


def test_batting_avg_recent_returns_float(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = batting_avg_recent(sample_balls_df, "PlayerA", cutoff, "last_10")
    assert isinstance(result, float)
    assert result >= 0.0


def test_batting_avg_no_player_returns_zero(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = batting_avg_recent(sample_balls_df, "UnknownPlayer", cutoff)
    assert result == 0.0


def test_boundary_pct_in_range(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = boundary_pct(sample_balls_df, "PlayerA", cutoff, season=2022)
    assert 0.0 <= result <= 1.0


def test_economy_rate_positive(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = economy_rate_recent(sample_balls_df, "BowlerX", cutoff)
    assert result >= 0.0


def test_dot_ball_pct_in_range(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = dot_ball_pct(sample_balls_df, "BowlerX", cutoff, season=2022)
    assert 0.0 <= result <= 1.0


def test_venue_avg_first_innings_positive(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = venue_avg_first_innings(sample_balls_df, "Wankhede", cutoff, last_n_seasons=3)
    assert result >= 0.0


def test_toss_win_advantage_in_range(sample_match_results):
    cutoff = pd.Timestamp("2023-01-01")
    result = toss_win_advantage(sample_match_results, "Wankhede", cutoff)
    assert 0.0 <= result <= 1.0


def test_death_specialist_score_non_negative(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = death_specialist_score(sample_balls_df, "BowlerX", cutoff, season=2022)
    assert result >= 0.0


def test_form_momentum_non_negative(sample_balls_df):
    cutoff = pd.Timestamp("2023-01-01")
    result = form_momentum(sample_balls_df, "PlayerA", cutoff)
    assert result >= 0.0
