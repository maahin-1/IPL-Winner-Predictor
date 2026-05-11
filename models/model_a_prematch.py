"""
MODEL-A: PreMatchXGB — pre-match win probability.
Algorithm: XGBoost | Output: win_probability_pre_match
Training data: CricSheet IPL 2008-2021
Hyperparams: per prd.model_architecture.layers[0].models[0].hyperparams
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.3,
    "gamma": 0.5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

INPUT_FEATURES = [
    "squad_strength_index",
    "venue_history",
    "h2h_record",
    "team_form_last_5",
    "team_form_last_10",
    "opponent_form_last_10",
    "toss_win_pct",
    "venue_away_record",
    "win_streak",
    "current_season_win_rate",
    "focal_won_toss",
    "focal_is_chasing",
]


class ModelA_PreMatchXGB(BaseEstimator, ClassifierMixin):
    """Pre-match win probability using XGBoost ensemble."""

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = hyperparams or HYPERPARAMS
        self._model: Optional[xgb.XGBClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "ModelA_PreMatchXGB":
        self._validate_features(X)
        self._model = xgb.XGBClassifier(**self.hyperparams)
        self._model.fit(
            X[INPUT_FEATURES],
            y,
            sample_weight=sample_weight,
            eval_set=[(X[INPUT_FEATURES], y)],
            verbose=False,
        )
        logger.info("MODEL-A trained on %d samples", len(X))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MODEL-A not trained yet")
        self._validate_features(X)
        return self._model.predict_proba(X[INPUT_FEATURES])

    def predict_win_probability(self, X: pd.DataFrame) -> float:
        """Return win probability for team1 (single sample)."""
        proba = self.predict_proba(X)
        return float(proba[0, 1])

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("MODEL-A saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ModelA_PreMatchXGB":
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(path))
        logger.info("MODEL-A loaded from %s", path)
        return instance

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(INPUT_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"MODEL-A missing features: {missing}")
