# IPL Predictive Intelligence Engine (IPIE)

A real-time AI system that predicts IPL match winners, season champions, and playoff qualifiers — designed as the backend intelligence layer for a fantasy sports platform.

> **This repository is a full simulation.** All components are wired up and runnable locally without API keys, a database, or a live data feed.

---

## Quick Start

```bash
# Install dependencies (Python 3.12+ recommended)
pip install -r requirements.txt

# Run a simulated MI vs CSK match
python simulate.py

# Custom teams and seed
python simulate.py RCB KKR --seed 7

# Slow it down for a live feel
python simulate.py MI RR --delay 0.5
```

**Available teams:** MI, CSK, RCB, DC, KKR, SRH, PBKS, RR, GT, LSG

---

## Architecture

```
Live Ball Feed (Kafka / Simulated)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                 L1 Base Ensemble                    │
│  MODEL-A (PreMatchXGB) ── MODEL-B (InMatchLGB)     │
│  MODEL-C (PlayerImpactXGB) ── MODEL-D (SeasonTrajLGB)│
└─────────────────┬───────────────────────────────────┘
                  │  4 probability outputs
                  ▼
┌─────────────────────────────────────────────────────┐
│           L2 Meta-Learner (XGBoost)                 │
│     5-fold stacked CV + isotonic calibration        │
└─────────────────┬───────────────────────────────────┘
                  │  calibrated win probability
                  ▼
┌─────────────────────────────────────────────────────┐
│           L3 LLM (Claude Sonnet)                    │
│  Narrative · Confidence tag · Risk flag · Captain   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
        11 API Signals (S1–S11)
   REST · WebSocket · Webhook (FastAPI)
```

**Latency SLA:** end-to-end < 30 s · L1 ≤ 80 ms · L2 ≤ 20 ms · L3 ≤ 8 s

---

## Output Signals (S1–S11)

| Signal | Description |
|--------|-------------|
| S1 | Win probability per team |
| S2 | Season winner probability |
| S3 | Playoff qualification probability |
| S4 | Points table leaderboard rank |
| S5 | Player impact scores (0–100) |
| S6 | Fantasy point projections |
| S7 | Momentum alert (fires when Δ > 15%) |
| S8 | LLM narrative |
| S9 | Confidence tag (HIGH / MEDIUM / LOW) |
| S10 | Key risk flag |
| S11 | Fantasy captain recommendation |

---

## Project Structure

```
├── simulate.py              ← Run a simulated match end-to-end
├── ipl_prd.json             ← Product Requirements Document (source of truth)
├── prd_patch.py             ← Safe PRD mutation utility
│
├── data/
│   ├── ingest/              ← 5 data source clients (T1–T5)
│   ├── features/            ← 28 point-in-time features across 5 groups
│   └── feast/               ← Feast feature store config
│
├── models/
│   ├── model_a_prematch.py  ← MODEL-A: PreMatchXGB
│   ├── model_b_inmatch.py   ← MODEL-B: InMatchLGB
│   ├── model_c_player_impact.py ← MODEL-C: PlayerImpactXGB
│   ├── model_d_season.py    ← MODEL-D: SeasonTrajLGB
│   ├── meta_learner.py      ← L2 stacking meta-learner
│   ├── train.py             ← Full training pipeline
│   └── evaluate.py          ← Offline metrics (accuracy, Brier, log-loss)
│
├── llm/
│   ├── llm_caller.py        ← Claude API caller (template fallback if no key)
│   ├── prompt_builder.py    ← Structured JSON prompt from match state
│   ├── response_parser.py   ← LLM JSON → signal fields
│   └── cost_tracker.py      ← Season-scale cost simulation
│
├── pipeline/
│   ├── orchestrator.py      ← Per-over: features → L1 → L2 → L3 → payload
│   ├── kafka_consumer.py    ← Live stream consumer + SimulatedMatchStream
│   └── momentum_detector.py ← Fires S7 alert when Δ > 15%
│
├── api/
│   ├── main.py              ← FastAPI app
│   ├── routes/              ← /v1/match, /v1/season, /live websocket
│   ├── schemas.py           ← Pydantic models for all 11 signals
│   ├── cache.py             ← Redis cache (in-memory fallback in sim mode)
│   └── webhook_dispatcher.py← POST webhooks with 3× retry
│
├── monitoring/
│   ├── drift_detector.py    ← Evidently AI post-match drift checks
│   ├── accuracy_tracker.py  ← Rolling 10-match accuracy alert
│   └── grafana_dashboard.json
│
└── tests/
    ├── test_features.py     ← Feature value ranges
    ├── test_leakage.py      ← Point-in-time leakage detection
    ├── test_api.py          ← API contract (all 11 signals)
    ├── test_models.py       ← Model output schema
    └── test_latency.py      ← End-to-end latency simulation
```

---

## Running the API Server

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

Endpoints:
- `GET /v1/match/{match_id}/prediction`
- `GET /v1/season/predictions`
- `WS  /live/{match_id}`
- `GET /health`

---

## Running Tests

```bash
# Leakage tests (Phase 0 exit criterion)
python data/features/leakage_tests.py

# Full test suite
pytest tests/ -v
```

---

## Training Real Models

1. Download IPL YAMLs from [cricsheet.org/downloads](https://cricsheet.org/downloads/) into `data/raw/cricsheet/ipl/`
2. Parse to Parquet:
   ```bash
   python data/ingest/cricsheet_loader.py data/raw/cricsheet/ipl
   ```
3. Train all models:
   ```bash
   python models/train.py
   ```
   Target: > 68% match accuracy on 2024 held-out set (Phase 1 exit criterion).

4. (Optional) Add your Anthropic API key to `.env` for live LLM narratives:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

---

## Production Dependencies

Core simulation runs on `requirements.txt`. For full production deployment (Kafka, Redis, MLflow, Feast, Airflow):

```bash
pip install -r requirements-prod.txt
```

See `DECISIONS.md` for unresolved cloud/infra decisions (OQ1, OQ2, OQ4, OQ5).

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Data Foundation — feature store + leakage tests | ✅ Complete |
| 1 | Base Models — train MODEL-A through MODEL-D | ⏳ Needs CricSheet data |
| 2 | Meta-Learner — stacked CV, Brier < 0.18 | ⏳ After Phase 1 |
| 3 | LLM Layer — narratives, cost tracking | ✅ Scaffolded |
| 4 | Live Pipeline — Kafka, per-over inference | ✅ Scaffolded |
| 5 | Fantasy API — all 11 signals via REST/WS | ✅ Scaffolded |
| 6 | Live Season — IPL 2026 deployment | ⏳ Phase 5 → 6 |
