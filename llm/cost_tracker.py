"""
LLM API cost tracker — per-match and season simulation.
Addresses OQ3: at scale (74 matches × 40 overs × 2 LLM calls/over),
OpenRouter free-tier model pricing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# google/gemma-3-27b-it:free via OpenRouter — $0/M tokens (free tier).
# Set non-zero if switching to paid model.
INPUT_COST_PER_M = 0.0
OUTPUT_COST_PER_M = 0.0

AVG_INPUT_TOKENS_PER_CALL = 600
AVG_OUTPUT_TOKENS_PER_CALL = 150

IPL_MATCHES_PER_SEASON = 74
OVERS_PER_MATCH = 40
MAX_CALLS_PER_OVER = 2


@dataclass
class CostEntry:
    match_id: str
    over: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    entries: list[CostEntry] = field(default_factory=list)
    _total_cost: float = 0.0

    def record(self, match_id: str, over: int, input_tokens: int, output_tokens: int) -> float:
        cost = _compute_cost(input_tokens, output_tokens)
        self.entries.append(CostEntry(match_id, over, input_tokens, output_tokens, cost))
        self._total_cost += cost
        return cost

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    def summary(self) -> dict:
        return {
            "total_calls": len(self.entries),
            "total_cost_usd": round(self._total_cost, 4),
            "avg_cost_per_call_usd": round(self._total_cost / len(self.entries), 6) if self.entries else 0,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "entries": [
                {"match_id": e.match_id, "over": e.over, "cost_usd": e.cost_usd}
                for e in self.entries
            ],
        }
        path.write_text(json.dumps(data, indent=2))


def simulate_season_cost(
    matches: int = IPL_MATCHES_PER_SEASON,
    overs_per_match: int = OVERS_PER_MATCH,
    calls_per_over: int = MAX_CALLS_PER_OVER,
    low_stakes_skip_rate: float = 0.3,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Simulate total LLM cost for a full IPL season.
    low_stakes_skip_rate: fraction of overs skipped due to <5% probability delta.
    """
    effective_calls_per_over = calls_per_over * (1.0 - low_stakes_skip_rate)
    total_calls = int(matches * overs_per_match * effective_calls_per_over)
    total_input_tokens = total_calls * AVG_INPUT_TOKENS_PER_CALL
    total_output_tokens = total_calls * AVG_OUTPUT_TOKENS_PER_CALL
    total_cost = _compute_cost(total_input_tokens, total_output_tokens)

    result = {
        "matches": matches,
        "overs_per_match": overs_per_match,
        "max_calls_per_over": calls_per_over,
        "low_stakes_skip_rate": low_stakes_skip_rate,
        "effective_calls_total": total_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 2),
    }

    print("\n=== IPIE Season LLM Cost Simulation (OQ3) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))

    return result


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * INPUT_COST_PER_M
        + (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    )


if __name__ == "__main__":
    simulate_season_cost(output_path=Path("artifacts/llm_cost_estimate.json"))
