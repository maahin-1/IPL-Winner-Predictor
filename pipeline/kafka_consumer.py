"""
Kafka consumer — live ball event stream ingestion.
Connects to Cricbuzz live stream via Kafka/Pub-Sub.
Per prd.system_architecture.pipeline_steps[0].
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    _KAFKA_AVAILABLE = True
except ImportError:
    KafkaConsumer = None  # type: ignore[assignment,misc]
    KafkaError = Exception  # type: ignore[assignment,misc]
    _KAFKA_AVAILABLE = False

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
BALL_EVENTS_TOPIC = "ipl.ball.events"
OVER_COMPLETE_TOPIC = "ipl.over.complete"
GROUP_ID = "ipie-orchestrator"


class BallEventConsumer:
    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP,
        group_id: str = GROUP_ID,
    ):
        self._bootstrap = bootstrap_servers
        self._group_id = group_id
        self._consumer: Optional[KafkaConsumer] = None

    def start(
        self,
        on_ball_event: Callable[[dict], None],
        on_over_complete: Callable[[dict], None],
    ) -> None:
        if not _KAFKA_AVAILABLE:
            raise RuntimeError(
                "kafka-python is not installed. "
                "Run: pip install kafka-python  or use SimulatedMatchStream for local testing."
            )
        self._consumer = KafkaConsumer(
            BALL_EVENTS_TOPIC,
            OVER_COMPLETE_TOPIC,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        logger.info("Kafka consumer started on topics: %s, %s", BALL_EVENTS_TOPIC, OVER_COMPLETE_TOPIC)
        try:
            for message in self._consumer:
                event = message.value
                if message.topic == OVER_COMPLETE_TOPIC:
                    on_over_complete(event)
                else:
                    on_ball_event(event)
        except KafkaError as e:
            logger.error("Kafka consumer error: %s", e)
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        if self._consumer:
            self._consumer.close()
            logger.info("Kafka consumer stopped")


class SimulatedMatchStream:
    """
    Generates synthetic ball-by-ball events without a live Kafka broker.
    Used in simulation mode and integration tests.
    """

    def __init__(self, match_id: str, team1: str, team2: str, seed: int = 42):
        self._match_id = match_id
        self._team1 = team1
        self._team2 = team2
        self._rng = __import__("numpy").random.default_rng(seed)

    def _simulate_innings(self, target: int | None = None) -> list[dict]:
        """Returns a list of over-complete events for one innings."""
        events = []
        score, wickets = 0, 0
        for over in range(20):
            if wickets >= 10:
                break
            over_runs = int(self._rng.integers(3, 18))
            over_wickets = int(self._rng.integers(0, 3) == 0)
            score += over_runs
            wickets = min(wickets + over_wickets, 10)
            run_rate = score / (over + 1) if over >= 0 else 0.0
            rrr = None
            if target:
                balls_remaining = max((20 - over - 1) * 6, 1)
                needed = max(target - score, 0)
                rrr = round(needed / balls_remaining * 6, 2)
            events.append({
                "match_id": self._match_id,
                "team1": self._team1,
                "team2": self._team2,
                "innings": 1 if target is None else 2,
                "over": over + 1,
                "score": score,
                "wickets": wickets,
                "run_rate": round(run_rate, 2),
                "target": target,
                "rrr": rrr,
            })
        return events

    def stream(self, on_over_complete: Callable[[dict], None]) -> None:
        """
        Emit over-complete events synchronously (used by simulate.py).
        Simulates both innings sequentially.
        """
        innings1 = self._simulate_innings()
        first_innings_total = innings1[-1]["score"] if innings1 else 150
        target = first_innings_total + 1

        for event in innings1:
            on_over_complete(event)

        innings2 = self._simulate_innings(target=target)
        for event in innings2:
            on_over_complete(event)
