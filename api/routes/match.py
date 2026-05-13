"""
GET /v1/match/{id}/prediction — REST poll endpoint.
Rate limit: 60 RPM per prd.api_output_spec.rate_limits.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.cache import get_cached_prediction
from api.schemas import PredictionPayloadResponse

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_models_loaded = False


def _load_models():
    """Stub — model loading happens in orchestrator. Kept for FastAPI startup hook."""
    global _models_loaded
    _models_loaded = True
    logger.info("Models ready")


@router.get("/match/{match_id}/prediction", response_model=PredictionPayloadResponse)
@limiter.limit("60/minute")
async def get_match_prediction(match_id: str, request: Request):
    """
    Returns the latest PredictionPayload for the given match.
    Data is served from Redis cache (last-known state) for latency.
    """
    cached = await get_cached_prediction(match_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction available for match {match_id}. Match may not have started.",
        )
    return cached
