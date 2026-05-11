"""
L3: OpenRouter API caller with 30s timeout and graceful fallback.
MITIGATION: R2 — never blocks ML output; falls back to SYSTEM_FALLBACK on timeout/error.
Model: google/gemma-3-27b-it:free (primary) → meta-llama/llama-3.1-8b-instruct:free (fast fallback).

When OPENROUTER_API_KEY is not set, returns template-based narratives so the
full pipeline runs in simulation mode without any API calls.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from llm.prompt_builder import MatchSnapshot, build_prompt, SYSTEM_PROMPT
from llm.response_parser import LLMOutput, parse_llm_response, FALLBACK_OUTPUT

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL = "openai/gpt-oss-120b:free"
FALLBACK_MODEL = "openai/gpt-oss-20b:free"
TIMEOUT_S = 30
MAX_TOKENS = 400
MAX_CALLS_PER_OVER = 2


def _template_narrative(snapshot: MatchSnapshot) -> LLMOutput:
    """Generate a plausible narrative without an API call (simulation mode)."""
    p = snapshot.win_probability_team1
    leader = snapshot.team1 if p >= 0.5 else snapshot.team2
    lead_pct = int(max(p, 1 - p) * 100)

    if snapshot.innings == 2:
        needed = (snapshot.target or 0) - snapshot.score
        balls_left = max((20 - snapshot.over) * 6, 1)
        context = (
            f"{snapshot.batting_team} need {needed} off {balls_left} balls "
            f"(RRR {snapshot.rrr:.1f})." if snapshot.rrr else
            f"{snapshot.batting_team} are batting in the second innings."
        )
    else:
        context = (
            f"{snapshot.batting_team} are {snapshot.score}/{snapshot.wickets} "
            f"after {snapshot.over} overs."
        )

    if snapshot.momentum_delta and abs(snapshot.momentum_delta) >= 0.15:
        swing = "swung decisively" if snapshot.momentum_delta > 0 else "shifted dramatically"
        momentum_note = f" Momentum has {swing} — a pivotal over."
    else:
        momentum_note = ""

    if lead_pct >= 75:
        confidence_tag = "HIGH"
        tone = "firmly in control"
    elif lead_pct >= 60:
        confidence_tag = "MEDIUM"
        tone = "holding a slight edge"
    else:
        confidence_tag = "LOW"
        tone = "evenly matched in a tight contest"

    narrative = (
        f"{context} {leader} are {tone} at {lead_pct}% win probability.{momentum_note}"
    )

    captain = snapshot.fantasy_captain_pick if hasattr(snapshot, "fantasy_captain_pick") else (
        snapshot.key_batsmen[0] if snapshot.key_batsmen else "TBD"
    )

    return LLMOutput(
        llm_narrative=narrative,
        confidence_tag=confidence_tag,
        key_risk_flag=f"{'Dew' if snapshot.dew_flag else 'Pitch conditions'} may be a factor.",
        fantasy_captain_pick=captain,
        is_fallback=False,
    )


class LLMCaller:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model or PRIMARY_MODEL
        self._fallback_model = fallback_model or FALLBACK_MODEL
        self._client = None
        self._calls_this_over: int = 0
        self._current_over: int = -1

        if self._api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=self._api_key,
                    timeout=TIMEOUT_S,
                )
            except ImportError:
                logger.warning("openai package not installed — using template narratives")
        else:
            logger.info("OPENROUTER_API_KEY not set — using template narratives (simulation mode)")

    def call_for_over(self, snapshot: MatchSnapshot) -> LLMOutput:
        if self._is_low_stakes(snapshot):
            logger.debug("Low-stakes over %d — skipping LLM call", snapshot.over)
            return FALLBACK_OUTPUT

        if not self._check_call_budget(snapshot.over):
            logger.warning("LLM call budget exhausted for over %d", snapshot.over)
            return FALLBACK_OUTPUT

        if self._client is None:
            return _template_narrative(snapshot)

        messages = build_prompt(snapshot)
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        for model_id in (self._model, self._fallback_model):
            try:
                response = self._client.chat.completions.create(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    messages=chat_messages,
                )
                raw_text = response.choices[0].message.content or ""
                output = parse_llm_response(raw_text)
                logger.info(
                    "LLM call succeeded (%s) for match %s over %d",
                    model_id, snapshot.match_id, snapshot.over,
                )
                return output

            except Exception as e:
                logger.warning(
                    "LLM call failed on %s for match %s over %d: %s",
                    model_id, snapshot.match_id, snapshot.over, e,
                )
                continue

        logger.error("All OpenRouter models failed for match %s — R2 fallback", snapshot.match_id)
        return FALLBACK_OUTPUT

    def _check_call_budget(self, over: int) -> bool:
        if over != self._current_over:
            self._current_over = over
            self._calls_this_over = 0
        if self._calls_this_over >= MAX_CALLS_PER_OVER:
            return False
        self._calls_this_over += 1
        return True

    @staticmethod
    def _is_low_stakes(snapshot: MatchSnapshot) -> bool:
        if snapshot.momentum_delta is None:
            return False
        return abs(snapshot.momentum_delta) < 0.05
