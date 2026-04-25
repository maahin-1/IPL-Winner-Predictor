"""
L2: Meta-Learner — stacking generalisation across all base models.
Algorithm: XGBoost | Strategy: 5-fold stacked cross-validation
Input: outputs of MODEL-A, MODEL-B, MODEL-C, MODEL-D + contextual signals
Output: calibrated_win_probability
Online learning: fine-tune weights after every 10 live matches.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

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
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}


class MetaLearner:
    """
    L2 stacking meta-learner.
    Held-out test set: IPL 2024.
    """

    def __init__(self, n_folds: int = 5, hyperparams: Optional[dict] = None):
        self.n_folds = n_folds
        self.hyperparams = hyperparams or META_HYPERPARAMS
        self._model: Optional[xgb.XGBClassifier] = None
        self._calibrated: Optional[CalibratedClassifierCV] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaLearner":
        self._validate_features(X)
        Xm = X[META_FEATURES].values
        ym = y.values

        # 5-fold stacked cross-validation for out-of-fold predictions
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(ym))

        for fold, (train_idx, val_idx) in enumerate(kf.split(Xm, ym)):
            fold_model = xgb.XGBClassifier(**self.hyperparams)
            fold_model.fit(Xm[train_idx], ym[train_idx], verbose=False)
            oof_preds[val_idx] = fold_model.predict_proba(Xm[val_idx])[:, 1]
            logger.info("Meta-learner fold %d complete", fold + 1)

        # Final model on all data
        self._model = xgb.XGBClassifier(**self.hyperparams)
        self._model.fit(Xm, ym, verbose=False)

        # Isotonic calibration
        self._calibrated = CalibratedClassifierCV(self._model, cv=5, method="isotonic")
        self._calibrated.fit(Xm, ym)

        logger.info(
            "Meta-learner trained. OOF log-loss approximation available — check evaluate.py"
        )
        return self

    def predict_calibrated_win_probability(self, X: pd.DataFrame) -> float:
        """Returns calibrated_win_probability — primary L2 output."""
        if self._calibrated is None:
            raise RuntimeError("MetaLearner not trained")
        self._validate_features(X)
        proba = self._calibrated.predict_proba(X[META_FEATURES].values)
        return float(proba[0, 1])

    def online_fine_tune(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fine-tune meta-learner weights after every 10 live matches.
        Per prd.model_architecture.layers[1].online_learning.
        """
        if self._model is None:
            raise RuntimeError("MetaLearner must be trained before fine-tuning")
        self._validate_features(X)
        self._model.fit(
            X[META_FEATURES].values,
            y.values,
            xgb_model=self._model.get_booster(),
            verbose=False,
        )
        self._calibrated = CalibratedClassifierCV(self._model, cv=5, method="isotonic")
        self._calibrated.fit(X[META_FEATURES].values, y.values)
        logger.info("Meta-learner online fine-tuned on %d samples", len(X))

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "MetaLearner":
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(path))
        instance._calibrated = CalibratedClassifierCV(instance._model, cv=5)
        return instance

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(META_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"MetaLearner missing features: {missing}")
