"""
Rolling accuracy tracker — alerts if match accuracy drops below 65% in any 10-match window.
Per prd.evaluation.online_monitoring[3] and prd.model_architecture.retraining_triggers[1].
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ACCURACY_ALERT_THRESHOLD = 0.65
ROLLING_WINDOW = 10


@dataclass
class MatchResult:
    match_id: str
    predicted_winner: str
    actual_winner: str

    @property
    def correct(self) -> bool:
        return self.predicted_winner == self.actual_winner


class AccuracyTracker:
    def __init__(
        self,
        window: int = ROLLING_WINDOW,
        alert_threshold: float = ACCURACY_ALERT_THRESHOLD,
    ):
        self._window = window
        self._threshold = alert_threshold
        self._results: deque[MatchResult] = deque(maxlen=window)

    def record(self, result: MatchResult) -> None:
        self._results.append(result)
        if len(self._results) == self._window:
            accuracy = self._rolling_accuracy()
            if accuracy < self._threshold:
                logger.warning(
                    "ACCURACY ALERT: rolling %d-match accuracy %.1f%% < %.0f%% threshold. "
                    "Triggering meta-learner fine-tune.",
                    self._window, accuracy * 100, self._threshold * 100,
                )
                self._trigger_fine_tune_hook()
            else:
                logger.info(
                    "Rolling %d-match accuracy: %.1f%% — above threshold",
                    self._window, accuracy * 100,
                )

    def _rolling_accuracy(self) -> float:
        if not self._results:
            return 1.0
        return sum(r.correct for r in self._results) / len(self._results)

    def current_accuracy(self) -> float:
        return self._rolling_accuracy()

    def _trigger_fine_tune_hook(self) -> None:
        """Hook point — in production, enqueues a fine-tune Airflow DAG run."""
        logger.info("Fine-tune hook triggered — enqueue meta-learner online_fine_tune job")
