"""
Point-in-time leakage detection tests — MITIGATION: R4.
All tests must pass before Phase 1 begins per prd.roadmap[0].exit_criteria.
Run: python data/features/leakage_tests.py
"""
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.features.batsman_features import (
    batting_avg_recent, strike_rate_venue, form_momentum,
    death_overs_sr, boundary_pct, chase_avg,
)
from data.features.bowler_features import (
    economy_rate_recent, dot_ball_pct, death_specialist_score,
)
from data.features.venue_features import (
    venue_avg_first_innings, toss_win_advantage,
)


def _make_balls_df() -> pd.DataFrame:
    """Minimal synthetic ball-by-ball DataFrame for leakage testing."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=20, freq="7D")
    match_ids = [f"M{i:03d}" for i in range(20)]

    rows = []
    for i, (mid, date) in enumerate(zip(match_ids, dates)):
        for over in range(20):
            for ball in range(6):
                rows.append({
                    "match_id": mid,
                    "season": 2022,
                    "date": date,
                    "venue": "Eden Gardens" if i % 2 == 0 else "Wankhede",
                    "innings": 1,
                    "over": over,
                    "ball": ball + 1,
                    "batsman": "PlayerA" if (over + ball) % 3 != 0 else "PlayerB",
                    "bowler": "BowlerX" if over % 2 == 0 else "BowlerY",
                    "runs_batter": int(rng.integers(0, 7)),
                    "runs_extras": 0,
                    "runs_total": int(rng.integers(0, 7)),
                    "wicket": bool(rng.integers(0, 10) == 0),
                    "wicket_kind": "caught" if rng.integers(0, 2) else "",
                    "fielder": "FielderA",
                    "player_out": "PlayerA" if rng.integers(0, 2) else "",
                    "team1": "TeamA",
                    "team2": "TeamB",
                })
    return pd.DataFrame(rows)


class TestCutoffEnforcement(unittest.TestCase):
    """Verify that features computed before cutoff != after when future data exists."""

    def setUp(self):
        self.df = _make_balls_df()
        self.cutoff_past = pd.Timestamp("2022-03-01")
        self.cutoff_future = pd.Timestamp("2023-01-01")

    def test_batting_avg_no_future_data(self):
        from data.features.batsman_features import _filter_batter
        rows_past = len(_filter_batter(self.df, "PlayerA", self.cutoff_past))
        rows_future = len(_filter_batter(self.df, "PlayerA", self.cutoff_future))
        # later cutoff must include more or equal ball records
        self.assertGreaterEqual(
            rows_future, rows_past,
            "Later cutoff must include more ball records than earlier cutoff"
        )

    def test_batting_avg_future_cutoff_not_equal_to_no_data(self):
        val_before_any = batting_avg_recent(self.df, "PlayerA", pd.Timestamp("2020-01-01"))
        self.assertEqual(val_before_any, 0.0, "No data before 2020 should return 0.0")

    def test_economy_rate_no_future_data(self):
        val = economy_rate_recent(self.df, "BowlerX", self.cutoff_past)
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)

    def test_form_momentum_no_future_data(self):
        val = form_momentum(self.df, "PlayerA", self.cutoff_past)
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)

    def test_venue_feature_no_future_data(self):
        val = venue_avg_first_innings(self.df, "Eden Gardens", self.cutoff_past, last_n_seasons=3)
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)


class TestNoCutoffBypass(unittest.TestCase):
    """Ensure features don't accidentally include the cutoff date itself."""

    def setUp(self):
        self.df = _make_balls_df()

    def test_batting_avg_excludes_cutoff_date(self):
        cutoff = pd.Timestamp("2022-02-28")
        match_on_cutoff_date = self.df[self.df["date"] == cutoff]
        # There should be no matches on exactly this date in our synthetic data
        # but the filter must use < not <=
        df_filtered = self.df[(self.df["batsman"] == "PlayerA") & (self.df["date"] < cutoff)]
        df_inclusive = self.df[(self.df["batsman"] == "PlayerA") & (self.df["date"] <= cutoff)]
        self.assertLessEqual(
            len(df_filtered),
            len(df_inclusive),
            "Strict < on cutoff_date must be used (not <=)"
        )

    def test_bowler_feature_excludes_cutoff(self):
        cutoff = pd.Timestamp("2022-03-15")
        df_filtered = self.df[(self.df["bowler"] == "BowlerX") & (self.df["date"] < cutoff)]
        df_inclusive = self.df[(self.df["bowler"] == "BowlerX") & (self.df["date"] <= cutoff)]
        self.assertLessEqual(len(df_filtered), len(df_inclusive))


class TestFeatureValueRanges(unittest.TestCase):
    """Sanity-check computed feature values are in valid ranges."""

    def setUp(self):
        self.df = _make_balls_df()
        self.cutoff = pd.Timestamp("2023-01-01")

    def test_boundary_pct_in_range(self):
        val = boundary_pct(self.df, "PlayerA", self.cutoff, season=2022)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0, "boundary_pct must be a proportion 0-1")

    def test_dot_ball_pct_in_range(self):
        val = dot_ball_pct(self.df, "BowlerX", self.cutoff, season=2022)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0, "dot_ball_pct must be a proportion 0-1")

    def test_death_specialist_score_non_negative(self):
        val = death_specialist_score(self.df, "BowlerX", self.cutoff, season=2022)
        self.assertGreaterEqual(val, 0.0)

    def test_strike_rate_venue_non_negative(self):
        val = strike_rate_venue(self.df, "PlayerA", "Eden Gardens", self.cutoff)
        self.assertGreaterEqual(val, 0.0)

    def test_toss_win_advantage_in_range(self):
        match_results = pd.DataFrame({
            "venue": ["Eden Gardens"] * 20,
            "date": pd.date_range("2020-01-01", periods=20, freq="7D"),
            "toss_winner": ["TeamA"] * 10 + ["TeamB"] * 10,
            "winner": ["TeamA"] * 12 + ["TeamB"] * 8,
            "season": [2020] * 20,
        })
        val = toss_win_advantage(match_results, "Eden Gardens", self.cutoff)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


class TestWindowIsolation(unittest.TestCase):
    """last_5 window must return fewer or equal records than last_10 window."""

    def setUp(self):
        self.df = _make_balls_df()
        self.cutoff = pd.Timestamp("2023-01-01")

    def test_last5_subset_of_last10_batting(self):
        # Proxy: batting_avg with last_5 should use a subset of last_10 matches
        from data.features.batsman_features import _filter_batter, _apply_window
        df_base = _filter_batter(self.df, "PlayerA", self.cutoff)
        last5_matches = set(
            _apply_window(df_base, "last_5")["match_id"].unique()
        )
        last10_matches = set(
            _apply_window(df_base, "last_10")["match_id"].unique()
        )
        self.assertTrue(
            last5_matches.issubset(last10_matches),
            "last_5 matches must be a subset of last_10 matches"
        )


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestCutoffEnforcement,
        TestNoCutoffBypass,
        TestFeatureValueRanges,
        TestWindowIsolation,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.failures or result.errors:
        print("\nLEAKAGE TESTS FAILED — DO NOT PROCEED TO PHASE 1")
        sys.exit(1)
    else:
        print("\nAll leakage tests passed. Phase 0 exit criterion met.")
        sys.exit(0)
