"""
L2: Meta-Learner — stacking generalisation across all base models.
Algorithm: Logistic Regression (Platt-calibrated by construction)
Input: outputs of MODEL-A, MODEL-B, MODEL-C, MODEL-D + contextual signals
Output: calibrated_win_probability
Online learning: fine-tune weights after every 10 live matches.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

META_FEATURES = [
    "model_a_output",   # MODEL-A.output: win_probability_pre_match
    "model_b_output",   # MODEL-B.output: win_probability_live
    "model_c_output",   # MODEL-C.output: player_impact_score (team aggregate)
    "model_d_output",   # MODEL-D.output: season_winner_probability
    "match_phase",
    "venue_dew_flag",
    "team_strategy_fingerprint",
]

META_HYPERPARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42,
}


class MetaLearner:
    """
    L2 stacking meta-learner (Logistic Regression).
    Naturally calibrated — avoids XGB overconfidence on small stacking datasets.
    Held-out test set: IPL 2024.
    """

    def __init__(self, n_folds: int = 5, hyperparams: Optional[dict] = None):
        self.n_folds = n_folds  # kept for API compatibility
        self.hyperparams = hyperparams or META_HYPERPARAMS
        self._model: Optional[LogisticRegression] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaLearner":
        self._validate_features(X)
        Xm = X[META_FEATURES].values
        ym = y.values
        self._model = LogisticRegression(**self.hyperparams)
        self._model.fit(Xm, ym)
        logger.info("Meta-learner (LR) trained on %d samples", len(ym))
        return self

    def predict_calibrated_win_probability(self, X: pd.DataFrame) -> float:
        """Returns calibrated win probability from L2 meta-learner."""
        if self._model is None:
            raise RuntimeError("MetaLearner not trained")
        self._validate_features(X)
        proba = self._model.predict_proba(X[META_FEATURES].values)
        return float(proba[0, 1])

    def online_fine_tune(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fine-tune meta-learner weights after every 10 live matches.
        Per prd.model_architecture.layers[1].online_learning.
        """
        if self._model is None:
            raise RuntimeError("MetaLearner must be trained before fine-tuning")
        self._validate_features(X)
        self._model.fit(X[META_FEATURES].values, y.values)
        logger.info("Meta-learner online fine-tuned on %d samples", len(X))

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))

    @classmethod
    def load(cls, path: Path) -> "MetaLearner":
        instance = cls()
        instance._model = joblib.load(str(path))
        return instance

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(META_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"MetaLearner missing features: {missing}")
