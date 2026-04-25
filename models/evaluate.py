"""
Offline evaluation — computes all metrics from prd.evaluation.offline_metrics.
Held-out test set: IPL 2024.
Targets:
  - match_winner_accuracy > 72%
  - brier_score < 0.18
  - log_loss < 0.45
  - season_winner_top3_accuracy > 85%
  - playoff_qualifier_accuracy > 80%
  - fantasy_point_projection_mae < 12 pts
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
)

logger = logging.getLogger(__name__)

TARGETS = {
    "match_winner_accuracy": (">", 0.72),
    "brier_score": ("<", 0.18),
    "log_loss": ("<", 0.45),
    "season_winner_top3_accuracy": (">", 0.85),
    "playoff_qualifier_accuracy": (">", 0.80),
    "fantasy_point_projection_mae": ("<", 12.0),
}


@dataclass
class EvalResult:
    metric: str
    value: float
    target_op: str
    target_val: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.metric}: {self.value:.4f} (target {self.target_op} {self.target_val})"


def evaluate_match_winner(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> tuple[EvalResult, EvalResult, EvalResult]:
    """Returns accuracy, brier_score, log_loss EvalResults."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    return (
        _result("match_winner_accuracy", acc),
        _result("brier_score", brier),
        _result("log_loss", ll),
    )


def evaluate_season_winner(
    y_true_top3: np.ndarray,
    y_pred_top3: np.ndarray,
) -> EvalResult:
    """Season winner top-3 accuracy."""
    acc = accuracy_score(y_true_top3, y_pred_top3)
    return _result("season_winner_top3_accuracy", acc)


def evaluate_playoff_qualifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvalResult:
    """Playoff qualifier accuracy (measured post match-30 per PRD)."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    return _result("playoff_qualifier_accuracy", acc)


def evaluate_fantasy_projection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvalResult:
    """Fantasy point projection MAE."""
    mae = mean_absolute_error(y_true, y_pred)
    return _result("fantasy_point_projection_mae", mae)


def run_full_evaluation(
    match_y_true: np.ndarray,
    match_y_prob: np.ndarray,
    season_top3_true: np.ndarray,
    season_top3_pred: np.ndarray,
    playoff_y_true: np.ndarray,
    playoff_y_prob: np.ndarray,
    fantasy_y_true: np.ndarray,
    fantasy_y_pred: np.ndarray,
    output_path: Optional[Path] = None,
) -> list[EvalResult]:
    results = list(evaluate_match_winner(match_y_true, match_y_prob))
    results.append(evaluate_season_winner(season_top3_true, season_top3_pred))
    results.append(evaluate_playoff_qualifier(playoff_y_true, playoff_y_prob))
    results.append(evaluate_fantasy_projection(fantasy_y_true, fantasy_y_pred))

    print("\n=== IPIE Offline Evaluation Results ===")
    all_passed = True
    for r in results:
        print(r)
        if not r.passed:
            all_passed = False

    if not all_passed:
        print("\nWARNING: One or more metrics did not meet PRD targets.")
    else:
        print("\nAll metrics passed PRD targets.")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([
                {"metric": r.metric, "value": r.value, "passed": r.passed}
                for r in results
            ], indent=2)
        )

    return results


def _result(metric: str, value: float) -> EvalResult:
    op, target = TARGETS[metric]
    passed = value > target if op == ">" else value < target
    return EvalResult(metric=metric, value=value, target_op=op, target_val=target, passed=passed)
