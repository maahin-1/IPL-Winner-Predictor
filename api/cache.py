"""
Redis cache — stores last-known prediction state for low-latency fallback.
Per prd.system_architecture.pipeline_steps[7].
MITIGATION: R5 — Redis buffer for webhook overload during finals.

In simulation mode (no Redis running) falls back to an in-memory dict automatically.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
PREDICTION_TTL_S = 7200

# In-memory fallback used when Redis is unavailable
_memory_store: dict[str, str] = {}
_redis = None
_redis_available: Optional[bool] = None


async def _get_redis():
    global _redis, _redis_available
    if _redis_available is False:
        return None
    try:
        import redis.asyncio as aioredis
        if _redis is None:
            _redis = aioredis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        await _redis.ping()
        _redis_available = True
        return _redis
    except Exception:
        _redis_available = False
        logger.info("Redis unavailable — using in-memory cache (simulation mode)")
        return None


def _prediction_key(match_id: str) -> str:
    return f"ipie:prediction:{match_id}"


async def cache_prediction(match_id: str, payload: dict) -> None:
    key = _prediction_key(match_id)
    serialized = json.dumps(payload)
    r = await _get_redis()
    if r is not None:
        await r.setex(key, PREDICTION_TTL_S, serialized)
    else:
        _memory_store[key] = serialized
    logger.debug("Cached prediction for match %s", match_id)


async def get_cached_prediction(match_id: str) -> Optional[dict]:
    key = _prediction_key(match_id)
    r = await _get_redis()
    if r is not None:
        raw = await r.get(key)
    else:
        raw = _memory_store.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def invalidate_match(match_id: str) -> None:
    key = _prediction_key(match_id)
    r = await _get_redis()
    if r is not None:
        await r.delete(key)
    else:
        _memory_store.pop(key, None)
