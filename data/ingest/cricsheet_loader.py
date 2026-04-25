"""
T1: CricSheet IPL YAML archive → Parquet
Schema: [match_id, over, ball, batsman, bowler, runs, wicket, fielder, timestamp]

Download source: https://cricsheet.org/downloads/
Coverage: IPL 2008-2025
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema([
    pa.field("match_id", pa.string()),
    pa.field("season", pa.int32()),
    pa.field("date", pa.date32()),
    pa.field("venue", pa.string()),
    pa.field("team1", pa.string()),
    pa.field("team2", pa.string()),
    pa.field("toss_winner", pa.string()),
    pa.field("toss_decision", pa.string()),
    pa.field("winner", pa.string()),
    pa.field("innings", pa.int8()),
    pa.field("over", pa.int8()),
    pa.field("ball", pa.int8()),
    pa.field("batsman", pa.string()),
    pa.field("non_striker", pa.string()),
    pa.field("bowler", pa.string()),
    pa.field("runs_batter", pa.int8()),
    pa.field("runs_extras", pa.int8()),
    pa.field("runs_total", pa.int8()),
    pa.field("wicket", pa.bool_()),
    pa.field("wicket_kind", pa.string()),
    pa.field("fielder", pa.string()),
    pa.field("player_out", pa.string()),
    pa.field("timestamp", pa.timestamp("s")),
])


def _extract_match_meta(info: dict, match_id: str) -> dict:
    meta = {
        "match_id": match_id,
        "season": int(info.get("season", 0)),
        "venue": info.get("venue", ""),
        "toss_winner": info.get("toss", {}).get("winner", ""),
        "toss_decision": info.get("toss", {}).get("decision", ""),
        "winner": "",
        "team1": "",
        "team2": "",
        "date": None,
    }
    teams = info.get("teams", [])
    if len(teams) >= 2:
        meta["team1"] = teams[0]
        meta["team2"] = teams[1]
    outcomes = info.get("outcome", {})
    meta["winner"] = outcomes.get("winner", outcomes.get("result", "no_result"))
    dates = info.get("dates", [])
    if dates:
        meta["date"] = pd.to_datetime(dates[0]).date()
    return meta


def _parse_delivery(delivery: dict, over: int, ball_num: int) -> dict:
    batsman = delivery.get("batter", delivery.get("batsman", ""))
    bowler = delivery.get("bowler", "")
    non_striker = delivery.get("non_striker", "")
    runs = delivery.get("runs", {})
    wickets = delivery.get("wickets", [])
    wicket = len(wickets) > 0
    wicket_kind = wickets[0].get("kind", "") if wicket else ""
    player_out = wickets[0].get("player_out", "") if wicket else ""
    fielders = wickets[0].get("fielders", []) if wicket else []
    fielder = fielders[0].get("name", "") if fielders else ""

    return {
        "over": over,
        "ball": ball_num,
        "batsman": batsman,
        "non_striker": non_striker,
        "bowler": bowler,
        "runs_batter": int(runs.get("batter", 0)),
        "runs_extras": int(runs.get("extras", 0)),
        "runs_total": int(runs.get("total", 0)),
        "wicket": wicket,
        "wicket_kind": wicket_kind,
        "fielder": fielder,
        "player_out": player_out,
    }


def parse_yaml_to_records(yaml_path: Path) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    match_id = yaml_path.stem
    meta = _extract_match_meta(data.get("info", {}), match_id)
    records = []

    for innings_idx, innings in enumerate(data.get("innings", []), start=1):
        for over_data in innings.get("overs", []):
            over_num = int(over_data.get("over", 0))
            for ball_idx, delivery in enumerate(over_data.get("deliveries", []), start=1):
                row = {**meta, "innings": innings_idx}
                row.update(_parse_delivery(delivery, over_num, ball_idx))
                row["timestamp"] = pd.Timestamp(meta["date"]) if meta["date"] else pd.NaT
                records.append(row)

    return records


def load_cricsheet_archive(
    yaml_dir: Path,
    output_path: Path,
    seasons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Parse all IPL YAML files in yaml_dir and write to Parquet.
    seasons: if provided, only include those season years.
    """
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {yaml_dir}")

    logger.info("Found %d YAML files in %s", len(yaml_files), yaml_dir)
    all_records: list[dict] = []

    for path in yaml_files:
        try:
            records = parse_yaml_to_records(path)
            if seasons:
                records = [r for r in records if r.get("season") in seasons]
            all_records.extend(records)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", path.name, e)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No records parsed — check yaml_dir path and file contents")

    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Written %d ball records to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    yaml_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/cricsheet/ipl")
    out = Path("data/processed/ipl_balls.parquet")
    df = load_cricsheet_archive(yaml_dir, out)
    print(f"Loaded {len(df):,} deliveries, {df['match_id'].nunique()} matches")
    print(df.dtypes)
