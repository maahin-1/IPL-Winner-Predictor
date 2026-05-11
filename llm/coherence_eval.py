"""
Phase 3 exit criterion: coherence score > 4.0/5.0 on 20 LLM narratives.
Evaluation strategy: LLM-as-judge via OpenRouter free-tier models.
Generator : google/gemma-3-27b-it:free
Judge     : meta-llama/llama-3.1-8b-instruct:free  (smaller for cheap scoring)
Requires OPENROUTER_API_KEY.

Usage:
    python llm/coherence_eval.py                         # score 20 narratives, print report
    python llm/coherence_eval.py --output artifacts/coherence_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Force UTF-8 stdout (Windows cp1252 default crashes on en-dashes etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import re
import time

from dotenv import load_dotenv

load_dotenv()


def _call_with_retry(client, model, messages, max_tokens, max_attempts=4, temperature=0.2):
    """Call OpenRouter with exponential backoff on 429, respecting Retry-After."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(
                model=model, max_tokens=max_tokens, messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" not in msg:
                raise
            m = re.search(r"retry_after_seconds['\"]?\s*[:=]\s*(\d+)", msg)
            wait = int(m.group(1)) + 2 if m else 30
            print(f"    [429 backoff] sleeping {wait}s (attempt {attempt+1}/{max_attempts})")
            time.sleep(wait)
    raise last_err

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "openai/gpt-oss-120b:free"
JUDGE_MODEL = "openai/gpt-oss-20b:free"

_JUDGE_SYSTEM = """You evaluate cricket match prediction narratives. You receive the full ground-truth match snapshot AND the narrative. Score coherence ONLY against the provided snapshot — do not invoke outside knowledge.

Scoring rubric (1-5):
5 = All numbers/players/teams match the snapshot exactly; 3 distinct insights; cricket-specific.
4 = Fully grounded in snapshot data, minor redundancy or vagueness, no factual errors.
3 = Generic but not wrong; could apply to many matches; lacks specific insight.
2 = Contains 1 factual error vs the snapshot (wrong number, wrong team, wrong player, wrong venue).
1 = Multiple factual errors or invented players/stats not in the snapshot.

Verify before scoring:
- Numbers (over, score, run rate, win %, target, RRR) match snapshot exactly
- Player names appear in key_batsmen / key_bowler / top_impact_players
- Venue and batting/bowling teams match snapshot
- Confidence in the narrative aligns with win_probability gap

Respond ONLY with valid JSON: {"score": <1-5>, "reason": "<one sentence citing specific evidence>"}"""


TEST_SCENARIOS = [
    dict(
        match_id="eval_01", team1="MI", team2="CSK", innings=1, over=5,
        score=52, wickets=2, target=None, rrr=None, run_rate=8.67,
        batting_team="MI", bowling_team="CSK", win_probability_team1=0.58,
        key_batsmen=["Rohit Sharma", "Suryakumar Yadav"], key_bowler="Jadeja",
        venue="Wankhede", pitch_type="flat", dew_flag=False,
        team1_form="W W L W W", team2_form="L W W L W",
        player_impact_scores={"Rohit Sharma": 72, "SKY": 68, "Jadeja": 41},
        momentum_delta=None,
    ),
    dict(
        match_id="eval_02", team1="RCB", team2="DC", innings=2, over=15,
        score=140, wickets=5, target=165, rrr=12.5, run_rate=9.33,
        batting_team="DC", bowling_team="RCB", win_probability_team1=0.21,
        key_batsmen=["Warner", "Pant"], key_bowler="Siraj",
        venue="Chinnaswamy", pitch_type="batting", dew_flag=False,
        team1_form="W W W L W", team2_form="W L L W L",
        player_impact_scores={"Warner": 80, "Pant": 55, "Siraj": 62},
        momentum_delta=-0.18,
    ),
    dict(
        match_id="eval_03", team1="KKR", team2="SRH", innings=1, over=19,
        score=198, wickets=4, target=None, rrr=None, run_rate=10.42,
        batting_team="KKR", bowling_team="SRH", win_probability_team1=0.71,
        key_batsmen=["Venkatesh Iyer", "Russell"], key_bowler="Cummins",
        venue="Eden Gardens", pitch_type="flat", dew_flag=True,
        team1_form="W L W W W", team2_form="W W W L W",
        player_impact_scores={"Russell": 88, "Iyer": 75, "Cummins": 70},
        momentum_delta=0.21,
    ),
    dict(
        match_id="eval_04", team1="GT", team2="RR", innings=2, over=18,
        score=170, wickets=8, target=175, rrr=5.0, run_rate=9.44,
        batting_team="RR", bowling_team="GT", win_probability_team1=0.12,
        key_batsmen=["Sandeep Sharma"], key_bowler="Rashid Khan",
        venue="Narendra Modi", pitch_type="slow", dew_flag=False,
        team1_form="L L W L L", team2_form="W W L W W",
        player_impact_scores={"Rashid": 90, "Sandeep": 45, "Siraj": 60},
        momentum_delta=-0.25,
    ),
    dict(
        match_id="eval_05", team1="MI", team2="RCB", innings=2, over=12,
        score=102, wickets=2, target=148, rrr=6.0, run_rate=8.5,
        batting_team="MI", bowling_team="RCB", win_probability_team1=0.73,
        key_batsmen=["Rohit Sharma", "Tilak Varma"], key_bowler="Chahal",
        venue="Wankhede", pitch_type="flat", dew_flag=True,
        team1_form="W W W W L", team2_form="L L W L W",
        player_impact_scores={"Rohit": 85, "Tilak": 79, "Chahal": 55},
        momentum_delta=0.15,
    ),
    dict(
        match_id="eval_06", team1="CSK", team2="KKR", innings=1, over=2,
        score=18, wickets=0, target=None, rrr=None, run_rate=9.0,
        batting_team="CSK", bowling_team="KKR", win_probability_team1=0.55,
        key_batsmen=["Conway", "Gaikwad"], key_bowler="Harshit Rana",
        venue="Chepauk", pitch_type="slow", dew_flag=False,
        team1_form="L W L W L", team2_form="W W L W W",
        player_impact_scores={"Conway": 60, "Gaikwad": 58, "Harshit": 52},
        momentum_delta=None,
    ),
    dict(
        match_id="eval_07", team1="DC", team2="GT", innings=2, over=19,
        score=195, wickets=7, target=198, rrr=18.0, run_rate=10.26,
        batting_team="DC", bowling_team="GT", win_probability_team1=0.08,
        key_batsmen=["Axar Patel"], key_bowler="Noor Ahmad",
        venue="Kotla", pitch_type="slow", dew_flag=False,
        team1_form="L L L W L", team2_form="W W W W W",
        player_impact_scores={"Axar": 65, "Noor": 88, "Rashid": 77},
        momentum_delta=-0.32,
    ),
    dict(
        match_id="eval_08", team1="SRH", team2="RR", innings=1, over=14,
        score=148, wickets=3, target=None, rrr=None, run_rate=10.57,
        batting_team="SRH", bowling_team="RR", win_probability_team1=0.63,
        key_batsmen=["Abhishek Sharma", "Klaasen"], key_bowler="Chahal",
        venue="Rajiv Gandhi", pitch_type="batting", dew_flag=False,
        team1_form="W W W L W", team2_form="W L W L W",
        player_impact_scores={"Abhishek": 82, "Klaasen": 77, "Chahal": 58},
        momentum_delta=0.12,
    ),
    dict(
        match_id="eval_09", team1="KKR", team2="GT", innings=1, over=17,
        score=175, wickets=2, target=None, rrr=None, run_rate=10.29,
        batting_team="KKR", bowling_team="GT", win_probability_team1=0.77,
        key_batsmen=["Venkatesh Iyer", "Rinku Singh"], key_bowler="Rashid Khan",
        venue="Eden Gardens", pitch_type="flat", dew_flag=True,
        team1_form="W W W W W", team2_form="L L L W L",
        player_impact_scores={"Iyer": 90, "Rinku": 83, "Rashid": 61},
        momentum_delta=0.16,
    ),
    dict(
        match_id="eval_10", team1="RR", team2="DC", innings=2, over=10,
        score=88, wickets=6, target=142, rrr=9.1, run_rate=8.8,
        batting_team="DC", bowling_team="RR", win_probability_team1=0.32,
        key_batsmen=["Axar Patel"], key_bowler="Trent Boult",
        venue="Kotla", pitch_type="batting", dew_flag=False,
        team1_form="L W W L W", team2_form="W L L W W",
        player_impact_scores={"Axar": 66, "Boult": 80, "Sandeep": 48},
        momentum_delta=-0.14,
    ),
    dict(
        match_id="eval_11", team1="LSG", team2="PBKS", innings=2, over=7,
        score=65, wickets=3, target=130, rrr=8.75, run_rate=9.29,
        batting_team="LSG", bowling_team="PBKS", win_probability_team1=0.61,
        key_batsmen=["Pooran", "Stoinis"], key_bowler="Arshdeep",
        venue="Ekana", pitch_type="flat", dew_flag=False,
        team1_form="W L W W L", team2_form="L L W L L",
        player_impact_scores={"Pooran": 71, "Stoinis": 65, "Arshdeep": 55},
        momentum_delta=0.08,
    ),
    dict(
        match_id="eval_12", team1="CSK", team2="SRH", innings=2, over=16,
        score=155, wickets=4, target=162, rrr=4.5, run_rate=9.69,
        batting_team="CSK", bowling_team="SRH", win_probability_team1=0.44,
        key_batsmen=["Dhoni", "Jadeja"], key_bowler="Pat Cummins",
        venue="Chepauk", pitch_type="slow", dew_flag=False,
        team1_form="W L L W W", team2_form="W W L W L",
        player_impact_scores={"Dhoni": 78, "Jadeja": 72, "Cummins": 66},
        momentum_delta=-0.08,
    ),
    dict(
        match_id="eval_13", team1="GT", team2="LSG", innings=2, over=18,
        score=168, wickets=9, target=172, rrr=8.0, run_rate=9.33,
        batting_team="GT", bowling_team="LSG", win_probability_team1=0.14,
        key_batsmen=["Noor Ahmad"], key_bowler="Mohit Sharma",
        venue="Narendra Modi", pitch_type="slow", dew_flag=False,
        team1_form="L W L L L", team2_form="W W W L W",
        player_impact_scores={"Noor": 85, "Mohit": 60, "Rashid": 79},
        momentum_delta=-0.28,
    ),
    dict(
        match_id="eval_14", team1="MI", team2="DC", innings=1, over=11,
        score=110, wickets=3, target=None, rrr=None, run_rate=10.0,
        batting_team="MI", bowling_team="DC", win_probability_team1=0.62,
        key_batsmen=["Rohit Sharma", "Bumrah"], key_bowler="Axar Patel",
        venue="Wankhede", pitch_type="flat", dew_flag=True,
        team1_form="W W L W W", team2_form="L L W L L",
        player_impact_scores={"Rohit": 80, "Bumrah": 75, "Axar": 58},
        momentum_delta=0.10,
    ),
    dict(
        match_id="eval_15", team1="SRH", team2="CSK", innings=2, over=14,
        score=135, wickets=4, target=155, rrr=6.0, run_rate=9.64,
        batting_team="SRH", bowling_team="CSK", win_probability_team1=0.69,
        key_batsmen=["Abhishek Sharma", "Cummins"], key_bowler="Pathirana",
        venue="Rajiv Gandhi", pitch_type="batting", dew_flag=False,
        team1_form="W W W W L", team2_form="L W L W L",
        player_impact_scores={"Abhishek": 76, "Cummins": 70, "Pathirana": 82},
        momentum_delta=0.18,
    ),
    dict(
        match_id="eval_16", team1="KKR", team2="RCB", innings=1, over=19,
        score=201, wickets=5, target=None, rrr=None, run_rate=10.58,
        batting_team="KKR", bowling_team="RCB", win_probability_team1=0.72,
        key_batsmen=["Venkatesh Iyer", "Russell"], key_bowler="Chahal",
        venue="Eden Gardens", pitch_type="flat", dew_flag=False,
        team1_form="W W L W W", team2_form="L W L L W",
        player_impact_scores={"Iyer": 88, "Russell": 95, "Chahal": 62},
        momentum_delta=0.22,
    ),
    dict(
        match_id="eval_17", team1="RR", team2="GT", innings=2, over=17,
        score=158, wickets=7, target=170, rrr=7.2, run_rate=9.29,
        batting_team="RR", bowling_team="GT", win_probability_team1=0.22,
        key_batsmen=["Sandeep Sharma"], key_bowler="Rashid Khan",
        venue="Sawai Mansingh", pitch_type="slow", dew_flag=False,
        team1_form="W L L L W", team2_form="W W W L W",
        player_impact_scores={"Rashid": 91, "Sandeep": 50, "Noor": 68},
        momentum_delta=-0.19,
    ),
    dict(
        match_id="eval_18", team1="PBKS", team2="MI", innings=1, over=5,
        score=48, wickets=1, target=None, rrr=None, run_rate=9.6,
        batting_team="PBKS", bowling_team="MI", win_probability_team1=0.54,
        key_batsmen=["Prabhsimran Singh", "Shashank Singh"], key_bowler="Bumrah",
        venue="PCA", pitch_type="pitch_seamer", dew_flag=False,
        team1_form="W W L W W", team2_form="W L W W L",
        player_impact_scores={"Prabhsimran": 62, "Shashank": 59, "Bumrah": 75},
        momentum_delta=0.06,
    ),
    dict(
        match_id="eval_19", team1="RCB", team2="MI", innings=1, over=0,
        score=8, wickets=0, target=None, rrr=None, run_rate=None,
        batting_team="RCB", bowling_team="MI", win_probability_team1=0.50,
        key_batsmen=["Kohli", "Du Plessis"], key_bowler="Bumrah",
        venue="Chinnaswamy", pitch_type="batting", dew_flag=False,
        team1_form="L W W L W", team2_form="W W W W L",
        player_impact_scores={"Kohli": 55, "Du Plessis": 52, "Bumrah": 70},
        momentum_delta=None,
    ),
    dict(
        match_id="eval_20", team1="DC", team2="PBKS", innings=2, over=13,
        score=118, wickets=5, target=155, rrr=7.5, run_rate=9.08,
        batting_team="DC", bowling_team="PBKS", win_probability_team1=0.48,
        key_batsmen=["Axar Patel", "Tristan Stubbs"], key_bowler="Arshdeep",
        venue="Kotla", pitch_type="batting", dew_flag=False,
        team1_form="L L W W L", team2_form="W L W L W",
        player_impact_scores={"Axar": 70, "Stubbs": 65, "Arshdeep": 73},
        momentum_delta=-0.06,
    ),
]


def run_coherence_eval(output_path: Path | None = None) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set — cannot run coherence eval")
        print("Set the key in .env and re-run: python llm/coherence_eval.py")
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed — pip install openai")
        return {}

    from llm.prompt_builder import MatchSnapshot, build_prompt, SYSTEM_PROMPT
    from llm.response_parser import parse_llm_response

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=30)

    scores = []
    results = []

    print("\n=== Phase 3 Coherence Evaluation (LLM-as-judge) ===\n")

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        snap = MatchSnapshot(**scenario)
        messages = build_prompt(snap)

        # Generate narrative
        try:
            gen_resp = _call_with_retry(
                client, GENERATOR_MODEL,
                [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                max_tokens=400,
                temperature=0.0,
            )
            raw = gen_resp.choices[0].message.content or ""
            llm_out = parse_llm_response(raw)
        except Exception as e:
            logger.warning("Generation failed for %s: %s", snap.match_id, e)
            continue

        if llm_out.is_fallback or not llm_out.llm_narrative:
            continue

        # Judge coherence — give judge full snapshot for grounded verification
        snapshot_for_judge = {
            "match": f"{snap.team1} vs {snap.team2}",
            "innings": snap.innings,
            "over_zero_indexed": snap.over,
            "over_display": snap.over + 1,
            "score": f"{snap.score}/{snap.wickets}",
            "run_rate": snap.run_rate,
            "target": snap.target,
            "rrr": snap.rrr,
            "batting_team": snap.batting_team,
            "bowling_team": snap.bowling_team,
            "win_probability": {
                snap.team1: round(snap.win_probability_team1, 3),
                snap.team2: round(1 - snap.win_probability_team1, 3),
            },
            "venue": snap.venue,
            "pitch_type": snap.pitch_type,
            "dew_flag": snap.dew_flag,
            "key_batsmen": snap.key_batsmen,
            "key_bowler": snap.key_bowler,
            "top_impact_players": snap.player_impact_scores,
        }
        judge_prompt = (
            f"GROUND-TRUTH SNAPSHOT:\n{json.dumps(snapshot_for_judge, indent=2)}\n\n"
            f"NARRATIVE TO SCORE:\n\"{llm_out.llm_narrative}\"\n\n"
            f"Score 1-5 per rubric. Cite a specific snapshot field in your reason."
        )
        try:
            judge_resp = _call_with_retry(
                client, JUDGE_MODEL,
                [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt},
                ],
                max_tokens=100,
            )
            judge_raw = (judge_resp.choices[0].message.content or "").strip()
            # Strip code fences if present
            if judge_raw.startswith("```"):
                judge_raw = judge_raw.split("```")[1]
                if judge_raw.startswith("json"):
                    judge_raw = judge_raw[4:].strip()
            judge_data = json.loads(judge_raw)
            score = int(judge_data.get("score", 3))
            reason = judge_data.get("reason", "")
        except Exception as e:
            logger.warning("Judge failed for %s: %s", snap.match_id, e)
            score, reason = 3, "judge parse error"

        scores.append(score)
        results.append({
            "match_id": snap.match_id,
            "matchup": f"{snap.team1} vs {snap.team2}",
            "narrative": llm_out.llm_narrative,
            "confidence_tag": llm_out.confidence_tag,
            "key_risk_flag": llm_out.key_risk_flag,
            "fantasy_captain_pick": llm_out.fantasy_captain_pick,
            "coherence_score": score,
            "judge_reason": reason,
        })

        status = "PASS" if score >= 4 else "FAIL"
        print(f"[{i:02d}] {snap.team1} vs {snap.team2} — score {score}/5 [{status}]")
        print(f"     {llm_out.llm_narrative}")
        print(f"     Judge: {reason}\n")

    if not scores:
        print("No scores collected.")
        return {}

    avg = sum(scores) / len(scores)
    passed = avg > 4.0
    print(f"\n{'='*50}")
    print(f"Avg coherence score : {avg:.2f} / 5.0")
    print(f"Phase 3 exit check  : {'PASS' if passed else 'FAIL'} (target > 4.0)")
    print(f"Narratives scored   : {len(scores)} / {len(TEST_SCENARIOS)}")

    report = {
        "avg_coherence_score": round(avg, 2),
        "phase3_exit_passed": passed,
        "narratives_scored": len(scores),
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Report saved to {output_path}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_coherence_eval(args.output)
