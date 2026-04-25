"""
Evidently AI data drift detection — triggered post-match on all feature distributions.
Per prd.evaluation.online_monitoring[0].
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, reference_df: pd.DataFrame):
        self._reference = reference_df

    def run(
        self,
        current_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> dict:
        report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
        report.run(reference_data=self._reference, current_data=current_df)

        result = report.as_dict()
        drift_detected = result["metrics"][1]["result"]["dataset_drift"]

        if drift_detected:
            logger.warning(
                "DATA DRIFT DETECTED post-match — check feature distributions. "
                "Consider triggering meta-learner fine-tune."
            )
        else:
            logger.info("No significant data drift detected post-match.")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, default=str))

        return {"drift_detected": drift_detected, "full_report": result}
