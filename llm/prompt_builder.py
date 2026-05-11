"""
L3: Prompt builder — constructs structured JSON prompt from match state.
Output format: structured_json per prd.model_architecture.layers[2].input_format.
Strategy: stateless_per_over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MatchSnapshot:
    match_id: str
    team1: str
    team2: str
    innings: int
    over: int
    score: int
    wickets: int
    target: Optional[int]
    rrr: Optional[float]
    run_rate: float
    batting_team: str
    bowling_team: str
    win_probability_team1: float  # From L2 calibrated output (S1)
    key_batsmen: list[str]
    key_bowler: str
    venue: str
    pitch_type: str
    dew_flag: bool
    team1_form: str   # e.g. "W W L W W"
    team2_form: str
    player_impact_scores: dict[str, float]  # top 3 players
    momentum_delta: Optional[float]  # last over delta


SYSTEM_PROMPT = """You are IPIE — IPL Predictive Intelligence Engine. You output structured cricket match analysis.

STRICT GROUNDING RULES (violations = invalid response):
1. Use ONLY the data provided in match_snapshot. Never invent players, scores, overs, or stats.
2. Player names MUST come from `key_batsmen`, `key_bowler`, or `top_impact_players` keys. Do not name any other player.
3. Numbers in the narrative (overs, score, run rate, win %, target, RRR) MUST match the snapshot exactly. The current over is `over` (0-indexed); display as "over {over+1}".
4. Identify batting/bowling teams ONLY from `batting_team` and `bowling_team` fields. Do not infer from `team1`/`team2`.
5. The favoured team is whichever has higher `win_probability`. State it explicitly.
6. Venue is `venue` field — do not substitute any other ground.

OUTPUT SCHEMA — respond with ONLY this JSON, no markdown, no extra keys:
{
  "llm_narrative": "<exactly 3 sentences. Each sentence MUST add a distinct insight: (1) current state + who leads, (2) key tactical factor specific to this snapshot, (3) primary swing factor or what must happen to change the outcome>",
  "confidence_tag": "<HIGH | MEDIUM | LOW — HIGH if win_probability gap ≥ 0.30, MEDIUM if 0.10-0.30, LOW if < 0.10>",
  "key_risk_flag": "<one specific threat using only named players/conditions from the snapshot>",
  "fantasy_captain_pick": "<exact player name from key_batsmen or top_impact_players>"
}

If you cannot ground a statement in the snapshot, omit it. Brevity over filler."""


def build_prompt(snapshot: MatchSnapshot) -> list[dict]:
    """
    Build the messages array for the Anthropic API call.
    Returns [system, user] structure for claude-sonnet-4-20250514.
    """
    over_label = f"Over {snapshot.over + 1}"
    situation = _describe_situation(snapshot)

    user_content = json.dumps({
        "match_snapshot": {
            "match": f"{snapshot.team1} vs {snapshot.team2}",
            "innings": snapshot.innings,
            "over": over_label,
            "score": f"{snapshot.score}/{snapshot.wickets}",
            "run_rate": round(snapshot.run_rate, 2) if snapshot.run_rate is not None else None,
            "target": snapshot.target,
            "required_run_rate": round(snapshot.rrr, 2) if snapshot.rrr else None,
            "batting_team": snapshot.batting_team,
            "bowling_team": snapshot.bowling_team,
            "win_probability": {
                snapshot.team1: round(snapshot.win_probability_team1, 3),
                snapshot.team2: round(1.0 - snapshot.win_probability_team1, 3),
            },
            "venue": snapshot.venue,
            "pitch_type": snapshot.pitch_type,
            "dew_expected": snapshot.dew_flag,
            "key_batsmen": snapshot.key_batsmen,
            "key_bowler": snapshot.key_bowler,
            "team_form": {
                snapshot.team1: snapshot.team1_form,
                snapshot.team2: snapshot.team2_form,
            },
            "top_impact_players": snapshot.player_impact_scores,
            "momentum_shift_last_over": snapshot.momentum_delta,
            "situation_summary": situation,
        }
    }, indent=2)

    return [
        {"role": "user", "content": user_content},
    ]


def _describe_situation(s: MatchSnapshot) -> str:
    if s.innings == 1:
        phase = _over_to_phase(s.over)
        return f"{s.batting_team} batting first, {phase}, {s.score}/{s.wickets} after {s.over + 1} overs."
    else:
        balls_left = max(0, (20 - s.over) * 6)
        needed = (s.target or 0) - s.score
        return (
            f"{s.batting_team} chasing {s.target}, need {needed} from {balls_left} balls, "
            f"RRR {round(s.rrr or 0, 2)}."
        )


def _over_to_phase(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 16:
        return "middle overs"
    return "death overs"
