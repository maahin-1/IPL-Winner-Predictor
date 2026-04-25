"""
MODEL-B: InMatchLGB — live over-by-over win probability.
Algorithm: LightGBM | Output: win_probability_live
Training data: CricSheet ball-by-ball IPL 2008-2021
Hyperparams: per prd.model_architecture.layers[0].models[1].hyperparams
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

HYPERPARAMS = {
    "n_estimators": 1000,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

INPUT_FEATURES = [
    "current_score",
    "wickets_fallen",
    "run_rate_required",
    "over_number",
    "bowling_phase",
    "batsman_at_crease_features",
    "bowler_current_form",
    "matchup_h2h",
]


class ModelB_InMatchLGB(BaseEstimator, ClassifierMixin):
    """Live over-by-over win probability using LightGBM."""

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = hyperparams or HYPERPARAMS
        self._model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelB_InMatchLGB":
        self._validate_features(X)
        self._model = lgb.LGBMClassifier(**self.hyperparams)
        self._model.fit(
            X[INPUT_FEATURES],
            y,
            eval_set=[(X[INPUT_FEATURES], y)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
        )
        logger.info("MODEL-B trained on %d samples", len(X))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MODEL-B not trained yet")
        self._validate_features(X)
        return self._model.predict_proba(X[INPUT_FEATURES])

    def predict_win_probability(self, X: pd.DataFrame) -> float:
        """Return live win probability for chasing/batting team (single sample)."""
        proba = self.predict_proba(X)
        return float(proba[0, 1])

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path))
        logger.info("MODEL-B saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ModelB_InMatchLGB":
        instance = cls()
        booster = lgb.Booster(model_file=str(path))
        instance._model = lgb.LGBMClassifier()
        instance._model._Booster = booster
        logger.info("MODEL-B loaded from %s", path)
        return instance

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(INPUT_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"MODEL-B missing features: {missing}")
