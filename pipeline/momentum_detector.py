"""
Momentum detector — fires alert when win probability shifts > 15% in one over.
Threshold: prd.api_output_spec.momentum_alert_threshold_pct = 15
Signal: S7 momentum_alert.
"""
from __future__ import annotations

import logging
from typing import Optional

from data.ingest.cricbuzz_client import MatchState

logger = logging.getLogger(__name__)

ALERT_THRESHOLD = 0.15   # 15 percentage points per PRD


class MomentumDetector:
    def check(
        self,
        match_state: MatchState,
        delta: float,
    ) -> Optional[dict]:
        """
        Returns momentum_alert dict if |delta| > ALERT_THRESHOLD, else None.
        Fields per prd.api_output_spec.signals[S7]: alert_type, trigger, delta, over, description.
        """
        if abs(delta) <= ALERT_THRESHOLD:
            return None

        direction = "swing_toward" if delta > 0 else "swing_against"
        team = match_state.batting_team if delta > 0 else match_state.bowling_team

        alert = {
            "alert_type": "momentum_shift",
            "trigger": direction,
            "delta": round(delta, 4),
            "over": match_state.over,
            "description": (
                f"Win probability shifted {abs(delta):.1%} in over {match_state.over + 1} "
                f"— momentum swinging toward {team}."
            ),
        }
        logger.info(
            "MOMENTUM ALERT: match=%s over=%d delta=%.2f%%",
            match_state.match_id, match_state.over, delta * 100,
        )
        return alert
