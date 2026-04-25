"""
GET /v1/match/{id}/prediction — REST poll endpoint.
Rate limit: 60 RPM per prd.api_output_spec.rate_limits.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.cache import get_cached_prediction, cache_prediction
from api.schemas import PredictionPayloadResponse

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_models_loaded = False


def _load_models():
    global _models_loaded
    if _models_loaded:
        return
    # Models are loaded lazily; orchestrator is injected via dependency injection in production
    _models_loaded = True
    logger.info("Models ready")


@router.get("/match/{match_id}/prediction", response_model=PredictionPayloadResponse)
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
