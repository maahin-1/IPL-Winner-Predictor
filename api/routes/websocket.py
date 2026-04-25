"""
WebSocket endpoint — ws://ipie/live/{match_id}
Pushes PredictionPayload to connected clients on each over.
Concurrent match limit: 10 per prd.api_output_spec.rate_limits.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_CONCURRENT_MATCHES = 10
_active_connections: dict[str, list[WebSocket]] = {}


@router.websocket("/live/{match_id}")
async def websocket_live(websocket: WebSocket, match_id: str):
    if len(_active_connections) >= MAX_CONCURRENT_MATCHES and match_id not in _active_connections:
        await websocket.close(code=1013, reason="Max concurrent match streams reached")
        return

    await websocket.accept()
    _active_connections.setdefault(match_id, []).append(websocket)
    logger.info("WebSocket client connected for match %s (%d clients)", match_id,
                len(_active_connections[match_id]))

    try:
        while True:
            # Keep connection alive; actual pushes come from broadcast_over_update
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        _active_connections[match_id].remove(websocket)
        if not _active_connections[match_id]:
            del _active_connections[match_id]
        logger.info("WebSocket client disconnected from match %s", match_id)


async def broadcast_over_update(match_id: str, payload: dict) -> None:
    """Called by orchestrator after each over to push updates to all connected clients."""
    clients = _active_connections.get(match_id, [])
    if not clients:
        return
    message = json.dumps(payload)
    disconnected = []
    for ws in clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)
