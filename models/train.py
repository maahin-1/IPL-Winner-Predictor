"""
Training orchestrator — runs full pipeline for all 4 base models + meta-learner.
Training split: 70% (2008-2021) / 15% validation (2022-2023) / 15% test (IPL 2024).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from models.model_a_prematch import ModelA_PreMatchXGB
from models.model_b_inmatch import ModelB_InMatchLGB
from models.model_c_player_impact import ModelC_PlayerImpactXGB
from models.model_d_season import ModelD_SeasonTrajLGB
from models.meta_learner import MetaLearner

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/models")

TRAIN_SEASONS = list(range(2019, 2024))   # 2019-2023 post-mega-auction era, most relevant to 2024
VAL_SEASONS = [2022, 2023]                # still used for MODEL-B early stopping
TEST_SEASONS = [2024]                     # held-out test


def train_all(
    prematch_df: pd.DataFrame,
    live_df: pd.DataFrame,
    player_df: pd.DataFrame,
    season_df: pd.DataFrame,
    experiment_name: str = "IPIE-Phase1",
    val_seasons: list[int] | None = None,
) -> dict:
    """
    Train all models and log to MLflow.
    Each df must have a 'season' column for split filtering.
    val_seasons: validation seasons for MODEL-B early stopping (default: VAL_SEASONS).
    """
    _val = val_seasons or VAL_SEASONS
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="full_train"):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # ── MODEL-A ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-A: PreMatchXGB")
        model_a = ModelA_PreMatchXGB()
        train_a = prematch_df[prematch_df["season"].isin(TRAIN_SEASONS)]
        feat_cols_a = [c for c in train_a.columns if c not in ("match_id", "date", "label", "season")]
        weights_a = np.exp(0.4 * (train_a["season"] - train_a["season"].min())).values
        model_a.fit(train_a[feat_cols_a], train_a["label"], sample_weight=weights_a)
        model_a.save(ARTIFACTS_DIR / "model_a.json")
        mlflow.log_param("model_a_estimators", model_a.hyperparams["n_estimators"])

        # ── MODEL-B ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-B: InMatchLGB")
        model_b = ModelB_InMatchLGB()
        train_b = live_df[live_df["season"].isin(TRAIN_SEASONS)]
        val_b = live_df[live_df["season"].isin(_val)]
        feat_cols_b = [c for c in train_b.columns if c not in ("match_id", "label", "season")]
        # Later overs are far more predictive — exponential phase weighting
        weights_b = np.exp(0.08 * train_b["over_number"].values) if "over_number" in train_b.columns else None
        model_b.fit(
            train_b[feat_cols_b], train_b["label"],
            sample_weight=weights_b,
            X_val=val_b[feat_cols_b] if not val_b.empty else None,
            y_val=val_b["label"] if not val_b.empty else None,
        )
        model_b.save(ARTIFACTS_DIR / "model_b.txt")
        mlflow.log_param("model_b_estimators", model_b.hyperparams["n_estimators"])

        # ── MODEL-C ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-C: PlayerImpactXGB")
        model_c = ModelC_PlayerImpactXGB()
        train_c = player_df[player_df["season"].isin(TRAIN_SEASONS)]
        feat_cols_c = [c for c in train_c.columns
                       if c not in ("match_id", "season", "player", "impact_score")]
        model_c.fit(train_c[feat_cols_c], train_c["impact_score"])
        model_c.save(ARTIFACTS_DIR / "model_c.json")

        # ── MODEL-D ──────────────────────────────────────────────────────────
        logger.info("Training MODEL-D: SeasonTrajLGB")
        model_d = ModelD_SeasonTrajLGB()
        train_d = season_df[season_df["season"].isin(TRAIN_SEASONS)]
        feat_cols_d = [c for c in train_d.columns
                       if c not in ("season", "team", "season_winner", "playoff_qualifier")]
        weights_d = np.exp(0.2 * (train_d["season"] - train_d["season"].min())).values
        model_d.fit(
            train_d[feat_cols_d],
            train_d["season_winner"],
            train_d["playoff_qualifier"],
            sample_weight=weights_d,
        )
        model_d.save(
            ARTIFACTS_DIR / "model_d_winner.txt",
            ARTIFACTS_DIR / "model_d_playoff.txt",
        )

        # ── META-LEARNER ─────────────────────────────────────────────────────
        logger.info("Training L2 MetaLearner with 5-fold stacked CV")
        meta = MetaLearner()
        meta_train = _build_meta_features(
            model_a, model_b, model_c, model_d,
            prematch_df[prematch_df["season"].isin(TRAIN_SEASONS)],
            live_df[live_df["season"].isin(TRAIN_SEASONS)],
            player_df[player_df["season"].isin(TRAIN_SEASONS)],
            season_df[season_df["season"].isin(TRAIN_SEASONS)],
        )
        meta.fit(meta_train["X"], meta_train["y"])
        meta.save(ARTIFACTS_DIR / "meta_learner.pkl")

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
    player_df: pd.DataFrame,
    season_df: pd.DataFrame,
) -> dict:
    """
    Assemble meta-learner training DataFrame from base model OOF predictions.
    Each model uses its own feature set — prematch_df for A, live_df for B, etc.
    """
    rows = []
    for match_id in prematch_df["match_id"].unique():
        pm_row = prematch_df[prematch_df["match_id"] == match_id]
        live_rows = live_df[live_df["match_id"] == match_id]

        try:
            a_out = model_a.predict_win_probability(pm_row)
        except Exception:
            a_out = 0.5

        # MODEL-B: last available ball (maximises signal; consistent with evaluate.py)
        if not live_rows.empty and hasattr(model_b, "_model") and model_b._model is not None:
            snap = live_rows.sort_values("over_number").iloc[[-1]]
            try:
                b_out = model_b.predict_win_probability(snap)
            except Exception:
                b_out = a_out
        else:
            b_out = a_out

        # MODEL-C: mean impact score normalised to [0,1] (impact range 0-100)
        from models.model_c_player_impact import INPUT_FEATURES as FEAT_C
        match_players = player_df[player_df["match_id"] == match_id]
        if (not match_players.empty and hasattr(model_c, "_model") and model_c._model is not None
                and all(f in match_players.columns for f in FEAT_C)):
            try:
                c_scores = model_c.predict(match_players[FEAT_C])
                c_out = float(np.mean(c_scores)) / 100.0
            except Exception:
                c_out = 0.5
        else:
            c_out = 0.5

        # MODEL-D: season winner probability for team1
        from models.model_d_season import INPUT_FEATURES as FEAT_D
        season = int(pm_row["season"].iloc[0])
        team1 = pm_row.get("team1", pd.Series([None])).iloc[0] if "team1" in pm_row.columns else None
        s_rows = season_df[(season_df["season"] == season) & (season_df["team"] == team1)]
        if (not s_rows.empty and hasattr(model_d, "_model_winner")
                and model_d._model_winner is not None
                and model_d._model_winner._Booster is not None):
            try:
                d_out = float(model_d._model_winner._Booster.predict(
                    s_rows[FEAT_D].values.astype(float))[0])
            except Exception:
                d_out = 0.5
        else:
            d_out = 0.5

        label = int(pm_row["label"].iloc[0])
        rows.append({
            "model_a_output": a_out,
            "model_b_output": b_out,
            "model_c_output": c_out,
            "model_d_output": d_out,
            "match_phase": 0.5,
            "venue_dew_flag": 0.0,
            "team_strategy_fingerprint": 0.5,
            "label": label,
        })

    meta_df = pd.DataFrame(rows)
    return {"X": meta_df.drop("label", axis=1), "y": meta_df["label"]}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from pathlib import Path
    processed = Path("data/processed")

    frames = {}
    for name in ("prematch_df", "live_df", "player_df", "season_df"):
        path = processed / f"{name}.parquet"
        if not path.exists():
            print(f"ERROR: {path} not found — run: python data/features/feature_matrix_builder.py")
            sys.exit(1)
        frames[name] = pd.read_parquet(path)
        print(f"  Loaded {name}: {len(frames[name]):,} rows")

    models = train_all(**frames)
    print("Training complete:", list(models.keys()))
