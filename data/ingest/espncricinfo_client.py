"""
T3: ESPNcricinfo player profile and career stats fetcher.
Used for player biographical features and career aggregates (supplement CricSheet).
"""
import logging
import os
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

BASE_URL = "https://hs-consumer-api.espncricinfo.com/v1"
DEFAULT_TIMEOUT_S = 15


class ESPNcricinfoClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "IPIE-DataPipeline/1.0",
            "Accept": "application/json",
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    )
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    def get_player_profile(self, player_id: int) -> dict:
        return self._get(f"/player/{player_id}/bio")

    def get_player_batting_stats(self, player_id: int, format_: str = "T20") -> dict:
        return self._get(f"/player/{player_id}/batting-stats", params={"format": format_})

    def get_player_bowling_stats(self, player_id: int, format_: str = "T20") -> dict:
        return self._get(f"/player/{player_id}/bowling-stats", params={"format": format_})

    def get_team_squad(self, team_id: int, season: int) -> list[dict]:
        data = self._get(f"/team/{team_id}/squad", params={"season": season})
        return data.get("players", [])

    def search_player(self, name: str) -> list[dict]:
        data = self._get("/search/players", params={"query": name})
        return data.get("results", [])

    def bulk_fetch_players(
        self,
        player_ids: list[int],
        delay_s: float = 1.0,
    ) -> dict[int, dict]:
        """Fetch batting + bowling stats for a list of players with polite delay."""
        results = {}
        for pid in player_ids:
            try:
                batting = self.get_player_batting_stats(pid)
                bowling = self.get_player_bowling_stats(pid)
                results[pid] = {"batting": batting, "bowling": bowling}
                time.sleep(delay_s)
            except Exception as e:
                logger.warning("Failed to fetch player %d: %s", pid, e)
        return results
