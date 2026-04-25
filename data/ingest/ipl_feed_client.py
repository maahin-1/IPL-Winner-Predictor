"""
T4: Official IPL feed parser — squad announcements, playing XI, toss, injuries.
Most reliable source for playing XI and toss per PRD data_sources[T4].
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

IPL_FEED_BASE = os.environ.get("IPL_FEED_BASE_URL", "https://ipl-feed.example.com/api/v1")
DEFAULT_TIMEOUT_S = 10


@dataclass
class PlayingXI:
    match_id: str
    team: str
    players: list[str]
    captain: str
    keeper: str


@dataclass
class TossResult:
    match_id: str
    winner: str
    decision: str  # "bat" or "field"


@dataclass
class InjuryUpdate:
    player: str
    team: str
    status: str  # "available", "doubtful", "ruled_out"
    note: str


class IPLFeedClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("IPL_FEED_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    )
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self.session.get(f"{IPL_FEED_BASE}{path}", params=params, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    def get_playing_xi(self, match_id: str) -> tuple[PlayingXI, PlayingXI]:
        data = self._get(f"/matches/{match_id}/playing-xi")
        teams = data.get("teams", [])
        result = []
        for t in teams[:2]:
            result.append(PlayingXI(
                match_id=match_id,
                team=t.get("name", ""),
                players=[p["name"] for p in t.get("players", [])],
                captain=t.get("captain", {}).get("name", ""),
                keeper=t.get("wicketKeeper", {}).get("name", ""),
            ))
        return tuple(result)

    def get_toss_result(self, match_id: str) -> TossResult:
        data = self._get(f"/matches/{match_id}/toss")
        return TossResult(
            match_id=match_id,
            winner=data.get("winner", ""),
            decision=data.get("decision", ""),
        )

    def get_injury_updates(self, season: int) -> list[InjuryUpdate]:
        data = self._get("/injuries", params={"season": season})
        return [
            InjuryUpdate(
                player=u.get("player", ""),
                team=u.get("team", ""),
                status=u.get("status", "available"),
                note=u.get("note", ""),
            )
            for u in data.get("updates", [])
        ]

    def get_match_schedule(self, season: int) -> list[dict]:
        data = self._get("/schedule", params={"season": season})
        return data.get("matches", [])
