"""
Pydantic models for all 11 output signals S1-S11.
Per prd.api_output_spec.signals.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceTag(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SYSTEM_FALLBACK = "SYSTEM_FALLBACK"


class MomentumAlert(BaseModel):
    """S7: momentum_alert — fired when win probability shifts > 15% in one over."""
    alert_type: str
    trigger: str
    delta: float = Field(ge=-1.0, le=1.0)
    over: int = Field(ge=0, le=19)
    description: str


class PredictionPayloadResponse(BaseModel):
    """Full API response object — all 11 signals per PRD."""
    match_id: str
    over: int = Field(ge=0, le=19)

    # S1: win_probability
    win_probability: dict[str, float] = Field(
        description="Real-time match win probability per team (0.0-1.0)"
    )

    # S2: season_winner_probability
    season_winner_probability: dict[str, float] = Field(
        description="Probability of winning IPL trophy (0.0-1.0)"
    )

    # S3: playoff_qualification_prob
    playoff_qualification_prob: dict[str, float] = Field(
        description="Probability of reaching playoffs (0.0-1.0)"
    )

    # S4: team_leaderboard_rank
    team_leaderboard_rank: dict[str, int] = Field(
        description="Current season predicted finish rank (1-10)"
    )

    # S5: player_impact_score
    player_impact_score: dict[str, float] = Field(
        description="Composite in-match contribution index per player (0-100)"
    )

    # S6: fantasy_point_projection
    fantasy_point_projection: dict[str, float] = Field(
        description="Projected fantasy points for remainder of match (0-200)"
    )

    # S7: momentum_alert
    momentum_alert: Optional[MomentumAlert] = Field(
        default=None,
        description="Fired when win probability shifts >15% in one over"
    )

    # S8: llm_narrative
    llm_narrative: str = Field(
        description="3-sentence plain English prediction explanation from LLM"
    )

    # S9: confidence_tag
    confidence_tag: ConfidenceTag = Field(
        description="Model confidence level"
    )

    # S10: key_risk_flag
    key_risk_flag: str = Field(
        description="Primary identified threat to current prediction"
    )

    # S11: fantasy_captain_pick
    fantasy_captain_pick: str = Field(
        description="Recommended captain based on projection delta"
    )

    latency_ms: float = Field(description="End-to-end inference latency in ms")


class SeasonPredictionsResponse(BaseModel):
    """GET /v1/season/predictions response."""
    season: int
    teams: list[dict]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0"
