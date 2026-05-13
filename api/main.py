"""
FastAPI application entry point — IPIE REST + WebSocket API.
Exposes endpoints per prd.api_output_spec.integration_modes.
Rate limits: 60 RPM REST, 10 concurrent WebSocket matches.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes import match, season, websocket
from api.routes.match import limiter
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models into memory
    from api.routes.match import _load_models
    _load_models()
    yield
    # Shutdown: flush cost tracker, close Redis
    pass


app = FastAPI(
    title="IPL Predictive Intelligence Engine (IPIE)",
    version="1.0",
    description="Real-time IPL win prediction API with LLM-generated narratives.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(match.router, prefix="/v1", tags=["match"])
app.include_router(season.router, prefix="/v1", tags=["season"])
app.include_router(websocket.router, tags=["websocket"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")
