"""
MODEL-D: SeasonTrajLGB — season winner and playoff qualifier probability.
Algorithm: LightGBM | Output: [season_winner_probability, playoff_qualification_probability]
Training data: IPL 2012-2024 season-level progression data
Timing: rolling_post_match
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

HYPERPARAMS = {
    "n_estimators": 500,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "verbose": -1,
}

INPUT_FEATURES = [
    "points_table_state",
    "net_run_rate",
    "remaining_fixtures_difficulty",
    "player_availability",
    "team_momentum_score",
    "coach_strategy_fingerprint",
]


class ModelD_SeasonTrajLGB(BaseEstimator):
    """Dual-output season trajectory model: winner + playoff probabilities."""

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = hyperparams or HYPERPARAMS
        self._model_winner: Optional[lgb.LGBMClassifier] = None
        self._model_playoff: Optional[lgb.LGBMClassifier] = None

    def fit(
        self,
        X: pd.DataFrame,
        y_winner: pd.Series,
        y_playoff: pd.Series,
    ) -> "ModelD_SeasonTrajLGB":
        self._validate_features(X)
        self._model_winner = lgb.LGBMClassifier(**self.hyperparams)
        self._model_winner.fit(
            X[INPUT_FEATURES], y_winner,
            callbacks=[lgb.log_evaluation(False)],
        )
        self._model_playoff = lgb.LGBMClassifier(**self.hyperparams)
        self._model_playoff.fit(
            X[INPUT_FEATURES], y_playoff,
            callbacks=[lgb.log_evaluation(False)],
        )
        logger.info("MODEL-D trained on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, float]:
        """Returns season_winner_probability and playoff_qualification_probability."""
        if self._model_winner is None or self._model_playoff is None:
            raise RuntimeError("MODEL-D not trained yet")
        self._validate_features(X)
        winner_prob = float(self._model_winner.predict_proba(X[INPUT_FEATURES])[0, 1])
        playoff_prob = float(self._model_playoff.predict_proba(X[INPUT_FEATURES])[0, 1])
        return {
            "season_winner_probability": winner_prob,
            "playoff_qualification_probability": playoff_prob,
        }

    def save(self, winner_path: Path, playoff_path: Path) -> None:
        winner_path.parent.mkdir(parents=True, exist_ok=True)
        if self._model_winner:
            self._model_winner.booster_.save_model(str(winner_path))
        if self._model_playoff:
            self._model_playoff.booster_.save_model(str(playoff_path))

    @classmethod
    def load(cls, winner_path: Path, playoff_path: Path) -> "ModelD_SeasonTrajLGB":
        instance = cls()
        instance._model_winner = lgb.LGBMClassifier()
        instance._model_winner._Booster = lgb.Booster(model_file=str(winner_path))
        instance._model_playoff = lgb.LGBMClassifier()
        instance._model_playoff._Booster = lgb.Booster(model_file=str(playoff_path))
        return instance

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(INPUT_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"MODEL-D missing features: {missing}")
