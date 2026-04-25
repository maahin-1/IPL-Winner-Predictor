"""
L3: LLM JSON response parser — extracts all 4 LLM output fields.
Outputs: llm_narrative, confidence_tag, key_risk_flag, fantasy_captain_pick.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

ConfidenceTag = Literal["HIGH", "MEDIUM", "LOW", "SYSTEM_FALLBACK"]

VALID_CONFIDENCE_TAGS = {"HIGH", "MEDIUM", "LOW"}


@dataclass
class LLMOutput:
    llm_narrative: str
    confidence_tag: ConfidenceTag
    key_risk_flag: str
    fantasy_captain_pick: str
    is_fallback: bool = False


FALLBACK_OUTPUT = LLMOutput(
    llm_narrative="",
    confidence_tag="SYSTEM_FALLBACK",
    key_risk_flag="LLM unavailable",
    fantasy_captain_pick="",
    is_fallback=True,
)


def parse_llm_response(raw_text: str) -> LLMOutput:
    """
    Parse structured JSON from LLM response text.
    Falls back to FALLBACK_OUTPUT on any parse error.
    """
    try:
        data = _extract_json(raw_text)
        confidence = str(data.get("confidence_tag", "LOW")).upper()
        if confidence not in VALID_CONFIDENCE_TAGS:
            confidence = "LOW"

        return LLMOutput(
            llm_narrative=str(data.get("llm_narrative", "")).strip(),
            confidence_tag=confidence,
            key_risk_flag=str(data.get("key_risk_flag", "")).strip(),
            fantasy_captain_pick=str(data.get("fantasy_captain_pick", "")).strip(),
            is_fallback=False,
        )
    except Exception as e:
        logger.warning("Failed to parse LLM response: %s. Raw: %.200s", e, raw_text)
        return FALLBACK_OUTPUT


def _extract_json(text: str) -> dict:
    """Extract JSON object from raw text, tolerating surrounding prose."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in LLM response")
