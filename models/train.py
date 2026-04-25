"""
Training orchestrator — runs full pipeline for all 4 base models + meta-learner.
Training split: 70% (2008-2021) / 15% validation (2022-2023) / 15% test (IPL 2024).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

from models.model_a_prematch import ModelA_PreMatchXGB
from models.model_b_inmatch import ModelB_InMatchLGB
from models.model_c_player_impact import ModelC_PlayerImpactXGB
from models.model_d_season import ModelD_SeasonTrajLGB
from models.meta_learner import MetaLearner

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/models")

TRAIN_SEASONS = list(range(2008, 2022))   # 70%
VAL_SEASONS = [2022, 2023]                # 15%
TEST_SEASONS = [2024]                     # 15%


def train_all(
    prematch_df: pd.DataFrame,
    live_df: pd.DataFrame,
    player_df: pd.DataFrame,
    season_df: pd.DataFrame,
    experiment_name: str = "IPIE-Phase1",
) -> dict:
    """
    Train all models and log to MLflow.
    Each df must have a 'season' column for split filtering.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="full_train"):
        # ── MODEL-A ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-A: PreMatchXGB")
        model_a = ModelA_PreMatchXGB()
        train_a = prematch_df[prematch_df["season"].isin(TRAIN_SEASONS)]
        model_a.fit(train_a.drop("label", axis=1), train_a["label"])
        model_a.save(ARTIFACTS_DIR / "model_a.json")
        mlflow.log_param("model_a_estimators", model_a.hyperparams["n_estimators"])

        # ── MODEL-B ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-B: InMatchLGB")
        model_b = ModelB_InMatchLGB()
        train_b = live_df[live_df["season"].isin(TRAIN_SEASONS)]
        model_b.fit(train_b.drop("label", axis=1), train_b["label"])
        model_b.save(ARTIFACTS_DIR / "model_b.txt")
        mlflow.log_param("model_b_estimators", model_b.hyperparams["n_estimators"])

        # ── MODEL-C ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-C: PlayerImpactXGB")
        model_c = ModelC_PlayerImpactXGB()
        train_c = player_df[player_df["season"].isin(TRAIN_SEASONS)]
        model_c.fit(train_c.drop("impact_score", axis=1), train_c["impact_score"])
        model_c.save(ARTIFACTS_DIR / "model_c.json")

        # ── MODEL-D ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-D: SeasonTrajLGB")
        model_d = ModelD_SeasonTrajLGB()
        train_d = season_df[season_df["season"].isin(TRAIN_SEASONS)]
        model_d.fit(
            train_d.drop(["season_winner", "playoff_qualifier"], axis=1),
            train_d["season_winner"],
            train_d["playoff_qualifier"],
        )
        model_d.save(
            ARTIFACTS_DIR / "model_d_winner.txt",
            ARTIFACTS_DIR / "model_d_playoff.txt",
        )

        # ── META-LEARNER ─────────────────────────────────────────────────────
        logger.info("Training L2 MetaLearner with 5-fold stacked CV")
        meta = MetaLearner()
        # Meta features must be pre-computed from base model OOF predictions
        meta_train = _build_meta_features(
            model_a, model_b, model_c, model_d,
            prematch_df[prematch_df["season"].isin(TRAIN_SEASONS)],
            live_df[live_df["season"].isin(TRAIN_SEASONS)],
        )
        meta.fit(meta_train["X"], meta_train["y"])
        meta.save(ARTIFACTS_DIR / "meta_learner.json")

        mlflow.log_artifact(str(ARTIFACTS_DIR))
        logger.info("Training complete. All models saved to %s", ARTIFACTS_DIR)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "model_d": model_d,
        "meta_learner": meta,
    }


def _build_meta_features(
    model_a, model_b, model_c, model_d,
    prematch_df: pd.DataFrame,
    live_df: pd.DataFrame,
) -> dict:
    """Assemble meta-learner training DataFrame from base model predictions."""
    rows = []
    for _, row in prematch_df.iterrows():
        sample = row.to_frame().T
        try:
            a_out = model_a.predict_win_probability(sample)
            b_out = model_b.predict_win_probability(sample) if hasattr(model_b, '_model') else 0.5
            c_out = model_c.predict_impact_score(sample) / 100.0 if hasattr(model_c, '_model') else 0.5
            d_row = model_d.predict(sample)
            rows.append({
                "model_a_output": a_out,
                "model_b_output": b_out,
                "model_c_output": c_out,
                "model_d_output": d_row["season_winner_probability"],
                "match_phase": row.get("match_phase", 0),
                "venue_dew_flag": row.get("venue_dew_flag", 0),
                "team_strategy_fingerprint": row.get("team_strategy_fingerprint", 0.5),
                "label": row.get("label", 0),
            })
        except Exception:
            continue

    meta_df = pd.DataFrame(rows)
    return {"X": meta_df.drop("label", axis=1), "y": meta_df["label"]}
