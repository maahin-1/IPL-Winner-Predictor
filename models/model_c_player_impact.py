"""
MODEL-C: PlayerImpactXGB — per-player in-match impact score (0-100).
Algorithm: XGBoost | Output: player_impact_score
Training data: CricSheet + ESPNcricinfo player match aggregates 2015-2024
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

INPUT_FEATURES = [
    "batsman_form_momentum",
    "bowler_threat_index",
    "fielding_contribution",
    "matchup_advantage",
    "phase_of_play",
    "pressure_index",
]


class ModelC_PlayerImpactXGB(BaseEstimator, RegressorMixin):
    """Per-player in-match impact score regressor (output range 0-100)."""

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = hyperparams or {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        self._model: Optional[xgb.XGBRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelC_PlayerImpactXGB":
        self._validate_features(X)
        y_clipped = y.clip(0, 100)
        self._model = xgb.XGBRegressor(**self.hyperparams)
        self._model.fit(X[INPUT_FEATURES], y_clipped, verbose=False)
        logger.info("MODEL-C trained on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MODEL-C not trained yet")
        self._validate_features(X)
        raw = self._model.predict(X[INPUT_FEATURES])
        return np.clip(raw, 0, 100)

    def predict_impact_score(self, X: pd.DataFrame) -> float:
        """Impact score for a single player observation."""
        return float(self.predict(X)[0])

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "ModelC_PlayerImpactXGB":
        instance = cls()
        instance._model = xgb.XGBRegressor()
        instance._model.load_model(str(path))
        return instance

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(INPUT_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"MODEL-C missing features: {missing}")
