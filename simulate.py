"""
IPIE Match Simulation
=====================
Runs a full T20 IPL match end-to-end through the prediction pipeline
without requiring real data, trained models, Redis, Kafka, or an API key.

Usage:
    python simulate.py                          # MI vs CSK, default seed
    python simulate.py "MI" "RCB"               # custom teams
    python simulate.py "MI" "CSK" --seed 7      # reproducible run
    python simulate.py "MI" "CSK" --innings 1   # first innings only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.ingest.cricbuzz_client import MatchState
from llm.llm_caller import LLMCaller, _template_narrative
from llm.prompt_builder import MatchSnapshot
from pipeline.kafka_consumer import SimulatedMatchStream
from pipeline.momentum_detector import MomentumDetector

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

IPL_TEAMS = ["MI", "CSK", "RCB", "DC", "KKR", "SRH", "PBKS", "RR", "GT", "LSG"]

SQUAD: dict[str, list[str]] = {
    "MI":   ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Hardik Pandya", "Bumrah"],
    "CSK":  ["Ruturaj Gaikwad", "Devon Conway", "Shivam Dube", "MS Dhoni", "Deepak Chahar"],
    "RCB":  ["Faf du Plessis", "Virat Kohli", "Glenn Maxwell", "Dinesh Karthik", "Siraj"],
    "DC":   ["Prithvi Shaw", "David Warner", "Rishabh Pant", "Axar Patel", "Kuldeep Yadav"],
    "KKR":  ["Shreyas Iyer", "Venkatesh Iyer", "Andre Russell", "Sunil Narine", "Varun Chakravarthy"],
    "SRH":  ["Mayank Agarwal", "Abhishek Sharma", "Aiden Markram", "Heinrich Klaasen", "Bhuvneshwar"],
    "PBKS": ["Shikhar Dhawan", "Jonny Bairstow", "Liam Livingstone", "Sam Curran", "Arshdeep Singh"],
    "RR":   ["Jos Buttler", "Yashasvi Jaiswal", "Sanju Samson", "Shimron Hetmyer", "Trent Boult"],
    "GT":   ["Shubman Gill", "Wriddhiman Saha", "Hardik Pandya", "David Miller", "Mohammed Shami"],
    "LSG":  ["KL Rahul", "Quinton de Kock", "Marcus Stoinis", "Deepak Hooda", "Mohsin Khan"],
}


def _bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _colour_prob(p: float) -> str:
    if p >= 0.70:
        return f"{GREEN}{p:.0%}{RESET}"
    if p >= 0.50:
        return f"{CYAN}{p:.0%}{RESET}"
    if p >= 0.35:
        return f"{YELLOW}{p:.0%}{RESET}"
    return f"{RED}{p:.0%}{RESET}"


class SimulatedOrchestrator:
    """
    Runs the full IPIE pipeline per over using synthetic probabilities.
    No trained models or API keys required.
    """

    def __init__(self, team1: str, team2: str, seed: int):
        self._team1 = team1
        self._team2 = team2
        self._rng = np.random.default_rng(seed)
        self._llm = LLMCaller()
        self._momentum = MomentumDetector()
        self._prev_prob: float = 0.5
        # Pre-generate a realistic probability curve for the match
        self._prob_curve = self._generate_prob_curve()
        self._over_index = 0

    def _generate_prob_curve(self) -> list[float]:
        """
        Brownian-motion probability walk bounded to [0.05, 0.95].
        Slight mean reversion keeps it from getting stuck at extremes.
        """
        p = 0.5
        curve = []
        for _ in range(40):  # 20 overs × 2 innings
            step = self._rng.normal(0, 0.06)
            p = p + step - 0.02 * (p - 0.5)  # mean reversion
            p = float(np.clip(p, 0.05, 0.95))
            curve.append(round(p, 4))
        return curve

    def predict(self, event: dict) -> dict:
        """Given an over-complete event, return a full prediction payload."""
        p = self._prob_curve[min(self._over_index, len(self._prob_curve) - 1)]
        self._over_index += 1

        team1, team2 = self._team1, self._team2
        over = event["over"]
        innings = event["innings"]

        delta = p - self._prev_prob
        self._prev_prob = p

        state = MatchState(
            match_id=event["match_id"],
            team1=team1,
            team2=team2,
            innings=innings,
            over=over,
            ball=6,
            score=event["score"],
            wickets=event["wickets"],
            run_rate=event["run_rate"],
            target=event.get("target"),
            rrr=event.get("rrr"),
            batting_team=team1 if innings == 1 else team2,
            bowling_team=team2 if innings == 1 else team1,
        )

        squad1 = SQUAD.get(team1, [f"{team1} Player {i}" for i in range(1, 6)])
        squad2 = SQUAD.get(team2, [f"{team2} Player {i}" for i in range(1, 6)])

        snapshot = MatchSnapshot(
            match_id=event["match_id"],
            team1=team1,
            team2=team2,
            innings=innings,
            over=over,
            score=event["score"],
            wickets=event["wickets"],
            target=event.get("target"),
            rrr=event.get("rrr"),
            run_rate=event["run_rate"],
            batting_team=state.batting_team,
            bowling_team=state.bowling_team,
            win_probability_team1=p,
            key_batsmen=squad1[:2],
            key_bowler=squad2[-1],
            venue="Wankhede Stadium",
            pitch_type="batting_paradise",
            dew_flag=over > 14,
            team1_form="W W L W W",
            team2_form="L W W W L",
            player_impact_scores={squad1[0]: round(float(self._rng.uniform(55, 90)), 1)},
            momentum_delta=delta if over > 1 else None,
        )

        # In simulation, always generate a narrative (bypasses budget/skip logic)
        llm_out = _template_narrative(snapshot)
        momentum_alert = self._momentum.check(state, delta)

        # Player impact scores for all 10 squad players
        all_players = squad1 + squad2
        impact_scores = {
            p: round(float(self._rng.uniform(30, 95)), 1) for p in all_players
        }
        fantasy_proj = {
            p: round(float(self._rng.uniform(40, 120)), 1) for p in all_players
        }

        return {
            "match_id": event["match_id"],
            "over": over,
            "innings": innings,
            "score": event["score"],
            "wickets": event["wickets"],
            "target": event.get("target"),
            "win_probability": {team1: p, team2: round(1 - p, 4)},
            "season_winner_probability": {
                team1: round(float(self._rng.uniform(0.05, 0.20)), 3),
                team2: round(float(self._rng.uniform(0.05, 0.20)), 3),
            },
            "playoff_qualification_prob": {
                team1: round(float(self._rng.uniform(0.50, 0.90)), 3),
                team2: round(float(self._rng.uniform(0.50, 0.90)), 3),
            },
            "team_leaderboard_rank": {team1: 3, team2: 5},
            "player_impact_score": impact_scores,
            "fantasy_point_projection": fantasy_proj,
            "momentum_alert": momentum_alert,
            "llm_narrative": llm_out.llm_narrative,
            "confidence_tag": llm_out.confidence_tag,
            "key_risk_flag": llm_out.key_risk_flag,
            "fantasy_captain_pick": llm_out.fantasy_captain_pick,
        }


def _print_header(team1: str, team2: str, seed: int) -> None:
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  IPL Predictive Intelligence Engine (IPIE) - Match Simulation{RESET}")
    print(f"  {team1}  vs  {team2}   |  seed={seed}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()


def _print_over(payload: dict, team1: str, team2: str) -> None:
    inn = payload["innings"]
    over = payload["over"]
    score = payload["score"]
    wkts = payload["wickets"]
    p1 = payload["win_probability"][team1]
    p2 = payload["win_probability"][team2]
    conf = payload["confidence_tag"]
    target = payload.get("target")

    inn_label = f"Inn {inn}"
    score_str = f"{score}/{wkts}"
    if target and inn == 2:
        needed = max(target - score, 0)
        score_str += f"  (need {needed})"

    alert = payload.get("momentum_alert")
    alert_str = ""
    if alert:
        d = alert.get("delta", 0)
        alert_str = f"  {YELLOW}!! MOMENTUM ALERT  d={d:+.0%}{RESET}"

    conf_colour = GREEN if conf == "HIGH" else (CYAN if conf == "MEDIUM" else YELLOW)

    print(
        f"  {inn_label} Over {over:>2}  {score_str:<18}"
        f"  {team1} {_colour_prob(p1)} {_bar(p1, 16)} "
        f"{_bar(p2, 16)} {_colour_prob(p2)} {team2}"
        f"  [{conf_colour}{conf}{RESET}]"
        f"{alert_str}"
    )
    print(f"           {CYAN}>{RESET} {payload['llm_narrative']}")
    print(f"           {CYAN}Captain:{RESET} {payload['fantasy_captain_pick']}"
          f"   {CYAN}Risk:{RESET} {payload['key_risk_flag']}")
    print()


def _print_result(history: list[dict], team1: str, team2: str) -> None:
    final = history[-1]
    p1 = final["win_probability"][team1]
    winner = team1 if p1 >= 0.5 else team2
    alerts = [h for h in history if h.get("momentum_alert")]

    print(f"{BOLD}{'-' * 70}{RESET}")
    print(f"{BOLD}  FINAL PREDICTION{RESET}")
    print(f"  Predicted winner : {BOLD}{GREEN}{winner}{RESET}")
    print(f"  Win probability  : {_colour_prob(max(p1, 1 - p1))}")
    print(f"  Momentum alerts  : {len(alerts)} during the match")
    print(f"  Fantasy captain  : {final['fantasy_captain_pick']}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()


def run_simulation(
    team1: str = "MI",
    team2: str = "CSK",
    seed: int = 42,
    innings_filter: Optional[int] = None,
    delay: float = 0.0,
) -> list[dict]:
    """
    Run a full match simulation and return the list of per-over payloads.
    delay: seconds to sleep between overs (0 = instant, useful for demo).
    """
    _print_header(team1, team2, seed)

    orchestrator = SimulatedOrchestrator(team1, team2, seed)
    stream = SimulatedMatchStream(f"SIM-{team1}-{team2}", team1, team2, seed)
    history: list[dict] = []

    def on_over_complete(event: dict) -> None:
        if innings_filter and event["innings"] != innings_filter:
            return
        payload = orchestrator.predict(event)
        history.append(payload)
        _print_over(payload, team1, team2)
        if delay:
            time.sleep(delay)

    stream.stream(on_over_complete)
    _print_result(history, team1, team2)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="IPIE Match Simulation")
    parser.add_argument("team1", nargs="?", default="MI",
                        help=f"Batting team (default: MI). Options: {', '.join(IPL_TEAMS)}")
    parser.add_argument("team2", nargs="?", default="CSK",
                        help="Bowling team (default: CSK)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--innings", type=int, choices=[1, 2], default=None,
                        help="Show only innings 1 or 2 (default: both)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between overs for live-feel output (default: 0)")
    args = parser.parse_args()

    if args.team1 == args.team2:
        print("Error: team1 and team2 must be different.")
        sys.exit(1)

    run_simulation(args.team1, args.team2, args.seed, args.innings, args.delay)


if __name__ == "__main__":
    main()
