"""
MLflow experiment tracking configuration.
"""
import os
import mlflow

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAMES = {
    "phase1": "IPIE-Phase1-BaseModels",
    "phase2": "IPIE-Phase2-MetaLearner",
    "phase3": "IPIE-Phase3-LLMLayer",
    "live": "IPIE-LiveSeason",
}


def setup_mlflow(phase: str = "phase1") -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    name = EXPERIMENT_NAMES.get(phase, f"IPIE-{phase}")
    mlflow.set_experiment(name)
    return name


def log_model_metrics(metrics: dict, params: dict | None = None) -> None:
    mlflow.log_metrics(metrics)
    if params:
        mlflow.log_params(params)
