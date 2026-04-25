"""
T2: Cricbuzz live API client — ball-by-ball event stream.
Implements exponential backoff per PRD data_sources[T2].notes.
MITIGATION: R1 — multi-source fallback triggers if this client fails.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.cricbuzz.com/api/v1"
DEFAULT_TIMEOUT_S = 10
MAX_RETRIES = 5


@dataclass
class MatchState:
    match_id: str
    team1: str
    team2: str
    innings: int
    over: int
    ball: int
    score: int
    wickets: int
    run_rate: float
    target: Optional[int]
    rrr: Optional[float]
    batting_team: str
    bowling_team: str
    raw: dict = field(default_factory=dict)


class CricbuzzClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CRICBUZZ_API_KEY", "")
        if not self.api_key:
            logger.warning("CRICBUZZ_API_KEY not set — live feed will fail")
        self.session = requests.Session()
        self.session.headers.update({
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com",
        })

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    def get_live_score(self, match_id: str) -> MatchState:
        data = self._get(f"/matches/{match_id}/live-score")
        return self._parse_live_score(match_id, data)

    def get_scorecard(self, match_id: str) -> dict:
        return self._get(f"/matches/{match_id}/scorecard")

    def get_commentary(self, match_id: str, innings: int = 1) -> list[dict]:
        data = self._get(f"/matches/{match_id}/commentary/{innings}")
        return data.get("commentary", [])

    def get_live_matches(self) -> list[dict]:
        data = self._get("/matches/live")
        return data.get("matches", [])

    def _parse_live_score(self, match_id: str, data: dict) -> MatchState:
        score_data = data.get("score", {})
        innings_data = data.get("innings", {})
        return MatchState(
            match_id=match_id,
            team1=data.get("team1", {}).get("name", ""),
            team2=data.get("team2", {}).get("name", ""),
            innings=int(innings_data.get("number", 1)),
            over=int(score_data.get("over", 0)),
            ball=int(score_data.get("ball", 0)),
            score=int(score_data.get("runs", 0)),
            wickets=int(score_data.get("wickets", 0)),
            run_rate=float(score_data.get("runRate", 0.0)),
            target=score_data.get("target"),
            rrr=score_data.get("requiredRunRate"),
            batting_team=data.get("battingTeam", {}).get("name", ""),
            bowling_team=data.get("bowlingTeam", {}).get("name", ""),
            raw=data,
        )

    def stream_over_events(
        self,
        match_id: str,
        on_over_complete: Callable[[MatchState], None],
        poll_interval_s: float = 15.0,
    ) -> None:
        """
        Poll live score every poll_interval_s seconds; call on_over_complete
        each time the over number increments.
        """
        last_over = -1
        logger.info("Starting live stream for match %s", match_id)
        while True:
            try:
                state = self.get_live_score(match_id)
                if state.over > last_over:
                    last_over = state.over
                    logger.info("Over %d complete — triggering inference", state.over)
                    on_over_complete(state)
            except Exception as e:
                logger.error("Stream error for match %s: %s", match_id, e)
            time.sleep(poll_interval_s)
