"""
GET /v1/season/predictions — season-level predictions endpoint.
Refresh: daily at 06:00 IST per prd.api_output_spec.rate_limits.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import SeasonPredictionsResponse

router = APIRouter()

_season_cache: dict[int, dict] = {}


@router.get("/season/predictions", response_model=SeasonPredictionsResponse)
async def get_season_predictions(season: int = 2026):
    """Returns season winner and playoff probability for all 10 teams."""
    if season not in _season_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No season predictions for {season}. Run the seasonal update job.",
        )
    data = _season_cache[season]
    return SeasonPredictionsResponse(
        season=season,
        teams=data["teams"],
        generated_at=data["generated_at"],
    )


def update_season_cache(season: int, teams: list[dict], generated_at: str) -> None:
    """Called by the daily Airflow task to refresh season predictions."""
    _season_cache[season] = {"teams": teams, "generated_at": generated_at}
