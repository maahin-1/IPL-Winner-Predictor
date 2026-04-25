# CLAUDE.md — IPL Predictive Intelligence Engine (IPIE)
# Instructions for Claude Code

---

## 1. What This Repository Is

You are building the **IPL Predictive Intelligence Engine (IPIE)** — a real-time AI system that
predicts IPL match winners, season champions, and playoff qualifiers. It serves as the backend
intelligence layer for a fantasy sports platform.

The full Product Requirements Document lives at:

```
./ipl_prd.json
```

**Your first action on every session: read and parse this file.**

```bash
cat ipl_prd.json | python3 -c "import sys,json; prd=json.load(sys.stdin); print(prd['meta'])"
```

The JSON is the single source of truth for this project. All feature names, model IDs, signal
names, latency targets, and evaluation thresholds in your code must match the PRD exactly.

---

## 2. How to Read the PRD

The `ipl_prd.json` file has the following top-level keys. Read them in this order:

```
meta                 → Project identity, version, author
prediction_targets   → What we are predicting (PT-001, PT-002, PT-003)
goals                → Strategic goals (G1-G5) and non-goals
data_sources         → 5 tiers of input data (T1-T5) with URLs and priorities
feature_store        → All engineered features grouped by player role
model_architecture   → 3-layer stack (L1 base ensemble → L2 meta → L3 LLM)
system_architecture  → 9-step pipeline + latency SLA + cloud options
api_output_spec      → 11 output signals, integration modes, rate limits
evaluation           → Offline metrics targets + online monitoring approach
roadmap              → 7 phases (Phase 0 → Phase 6) with exit criteria
risks                → 6 identified risks (R1-R6) with mitigations
open_questions       → 5 deferred decisions (OQ1-OQ5)
```

To extract any section in your terminal:

```bash
# See all feature names for batsmen
cat ipl_prd.json | python3 -c "
import sys, json
prd = json.load(sys.stdin)
for f in prd['feature_store']['feature_groups']['batsman']:
    print(f['name'], '-', f['description'])
"

# See all model IDs and algorithms
cat ipl_prd.json | python3 -c "
import sys, json
prd = json.load(sys.stdin)
for m in prd['model_architecture']['layers'][0]['models']:
    print(m['id'], m['algorithm'], '-', m['purpose'])
"

# See latency SLA
cat ipl_prd.json | python3 -c "
import sys, json
prd = json.load(sys.stdin)
for k,v in prd['system_architecture']['latency_sla'].items():
    print(k, ':', v)
"
```

---

## 3. Guiding Principles for Every File You Write

Before writing any code, resolve these three questions from the PRD:

1. **Which phase does this belong to?** Check `prd.roadmap[].phase` and its `exit_criteria`.
   Do not build Phase 2 code before Phase 1's exit criterion is met.

2. **Which model/signal does this touch?** Use the exact `id` values from the PRD
   (`MODEL-A`, `MODEL-B`, `S1`, `S5`, etc.) as variable names, class names, or config keys.

3. **Does this respect the latency SLA?** Every component has a budget.
   `prd.system_architecture.latency_sla` is the contract.

---

## 4. Repository Structure to Build

```
ipie/
│
├── ipl_prd.json                  ← PRD — never edit manually; update via prd_patch.py
├── CLAUDE.md                     ← This file
│
├── data/
│   ├── ingest/
│   │   ├── cricsheet_loader.py   ← T1: Parse CricSheet CSVs/YAMLs to parquet
│   │   ├── cricbuzz_client.py    ← T2: Live Cricbuzz API client with retry logic
│   │   ├── espncricinfo_client.py← T3: ESPNcricinfo player profile fetcher
│   │   ├── ipl_feed_client.py    ← T4: Official IPL feed parser
│   │   └── scraper/              ← T5: Scrapy spiders for coach/venue data
│   │
│   ├── features/
│   │   ├── batsman_features.py   ← All features in prd.feature_store.batsman
│   │   ├── bowler_features.py    ← All features in prd.feature_store.bowler
│   │   ├── fielding_features.py  ← All features in prd.feature_store.fielding_and_drs
│   │   ├── coach_features.py     ← All features in prd.feature_store.coach_and_management
│   │   ├── venue_features.py     ← All features in prd.feature_store.venue_and_context
│   │   ├── feature_registry.py   ← Maps PRD feature names → computation functions
│   │   └── leakage_tests.py      ← Unit tests enforcing point-in-time correctness (R4)
│   │
│   └── feast/
│       └── feature_store.yaml    ← Feast config (online: Redis, offline: BigQuery/S3)
│
├── models/
│   ├── model_a_prematch.py       ← MODEL-A: PreMatchXGB
│   ├── model_b_inmatch.py        ← MODEL-B: InMatchLGB
│   ├── model_c_player_impact.py  ← MODEL-C: PlayerImpactXGB
│   ├── model_d_season.py         ← MODEL-D: SeasonTrajLGB
│   ├── meta_learner.py           ← L2: Stacking meta-learner (XGBoost)
│   ├── train.py                  ← Orchestrates full training pipeline
│   ├── evaluate.py               ← Computes all metrics in prd.evaluation.offline_metrics
│   └── registry/
│       └── mlflow_config.py      ← MLflow experiment tracking
│
├── llm/
│   ├── llm_caller.py             ← L3: Anthropic API caller with timeout + fallback
│   ├── prompt_builder.py         ← Builds structured JSON prompt from match state
│   ├── response_parser.py        ← Parses LLM JSON → narrative, confidence, risk, captain
│   └── cost_tracker.py           ← Tracks per-match LLM API cost (see OQ3)
│
├── pipeline/
│   ├── orchestrator.py           ← Per-over trigger: feature → ML → LLM → output
│   ├── kafka_consumer.py         ← Consumes live ball event stream
│   └── momentum_detector.py     ← Fires alerts when delta > 15% (prd.api_output_spec.momentum_alert_threshold_pct)
│
├── api/
│   ├── main.py                   ← FastAPI app entry point
│   ├── routes/
│   │   ├── match.py              ← GET /v1/match/{id}/prediction
│   │   ├── season.py             ← GET /v1/season/predictions
│   │   └── websocket.py          ← ws://ipie/live/{match_id}
│   ├── schemas.py                ← Pydantic models for all 11 signals (S1-S11)
│   ├── webhook_dispatcher.py     ← Sends POST webhooks with 3x retry
│   └── cache.py                  ← Redis integration for last-known state
│
├── monitoring/
│   ├── drift_detector.py         ← Evidently AI data drift checks (post-match)
│   ├── accuracy_tracker.py       ← Rolling 10-match accuracy alert trigger
│   └── grafana_dashboard.json    ← Grafana dashboard config
│
└── tests/
    ├── test_features.py          ← Feature computation correctness
    ├── test_leakage.py           ← Point-in-time leakage detection
    ├── test_models.py            ← Model output range and schema validation
    ├── test_api.py               ← API contract tests against prd.api_output_spec
    └── test_latency.py           ← End-to-end latency simulation (target: <30s)
```

---

## 5. Build Order — Follow the Roadmap

The `prd.roadmap` array defines the build sequence. **Do not skip phases.**

### Phase 0 — Data Foundation (Weeks 1-3)
Exit criterion: `Feature store populated for 2008-2024`

Tasks in order:
1. Download CricSheet IPL YAML archive from `https://cricsheet.org/downloads/`
2. Build `data/ingest/cricsheet_loader.py` — parse YAMLs to parquet with schema:
   `[match_id, over, ball, batsman, bowler, runs, wicket, fielder, timestamp]`
3. Build all 5 feature group files under `data/features/`
4. Register features in `data/features/feature_registry.py`
5. Write `data/features/leakage_tests.py` — all tests must pass before Phase 1
6. Configure Feast in `data/feast/feature_store.yaml`
7. Validate: run `python data/features/leakage_tests.py` — zero failures required

### Phase 1 — Base Models (Weeks 4-7)
Exit criterion: `Match accuracy > 68% offline`

Tasks in order:
1. Train MODEL-A using hyperparams from `prd.model_architecture.layers[0].models[0].hyperparams`
2. Train MODEL-B (LightGBM on ball-by-ball data)
3. Train MODEL-C (player impact score)
4. Train MODEL-D (season trajectory)
5. Run `models/evaluate.py` — verify accuracy > 68% on 2024 held-out test set
6. Log all experiments to MLflow

### Phase 2 — Meta-Learner (Weeks 8-9)
Exit criterion: `Brier score < 0.18 on validation set`

Tasks in order:
1. Build `models/meta_learner.py` with 5-fold stacked CV
2. Input: outputs of MODEL-A, MODEL-B, MODEL-C, MODEL-D + contextual signals
3. Validate Brier score < 0.18 — check `prd.evaluation.offline_metrics[1].target`

### Phase 3 — LLM Layer (Weeks 10-11)
Exit criterion: `Coherence score > 4.0/5.0 human eval`

Tasks in order:
1. Build `llm/prompt_builder.py` — outputs structured JSON from match state
2. Build `llm/llm_caller.py` — uses Anthropic SDK, respects 30s timeout, falls back gracefully
3. Build `llm/response_parser.py` — extracts all 4 LLM output fields
4. Validate cost: run `llm/cost_tracker.py` against 74-match simulation (see OQ3)
5. Human eval: score 20 narratives for coherence on a 1-5 scale. Target: > 4.0.

### Phase 4 — Live Pipeline (Weeks 12-14)
Exit criterion: `End-to-end < 30s on test match simulation`

Tasks in order:
1. Build `pipeline/kafka_consumer.py` — connect to Cricbuzz live stream
2. Build `pipeline/orchestrator.py` — triggers inference on each over completion
3. Build `pipeline/momentum_detector.py` — fires alert if delta > 15%
4. Run latency simulation: `tests/test_latency.py`
5. All 5 SLA milestones in `prd.system_architecture.latency_sla` must pass

### Phase 5 — Fantasy API (Weeks 15-16)
Exit criterion: `All 11 signals delivered per over`

Tasks in order:
1. Build `api/schemas.py` — Pydantic models for all signals S1-S11
2. Build all API routes: match, season, websocket
3. Build `api/webhook_dispatcher.py` with 3x retry queue
4. Build `api/cache.py` Redis integration
5. Run `tests/test_api.py` — validate all 11 signals present in response

### Phase 6 — Live Season (IPL 2026)
Exit criterion: `Season accuracy > 72% maintained`

Tasks:
1. Deploy to cloud (GCP Cloud Run or AWS Lambda — see `prd.system_architecture.cloud_options`)
2. Enable `monitoring/drift_detector.py` (post-match Evidently checks)
3. Enable `monitoring/accuracy_tracker.py` (alert if < 65% in rolling 10-match window)
4. Set up Grafana dashboard from `monitoring/grafana_dashboard.json`

---

## 6. Naming Conventions

These are enforced rules — never deviate:

| PRD Object | Code Convention | Example |
|---|---|---|
| Model IDs (MODEL-A etc.) | Class names | `class ModelA_PreMatchXGB` |
| Feature names | Snake_case, exact match | `batting_avg_recent`, `death_specialist_score` |
| Signal IDs (S1-S11) | Response dict keys | `{"win_probability": 0.72, ...}` |
| Risk IDs (R1-R6) | Comment tags in mitigations | `# MITIGATION: R4 - leakage guard` |
| Phase numbers | Git branch prefix | `phase-0/data-foundation` |

---

## 7. Critical Constraints — Never Violate

These come directly from PRD risks and architecture decisions:

```
R4 — DATA LEAKAGE:
  - Every feature must use only data available BEFORE the prediction moment
  - Feast point-in-time joins are mandatory — never use a raw table join
  - `leakage_tests.py` must pass on every commit

R2 — LLM FALLBACK:
  - LLM call timeout: 30 seconds hard limit
  - If timeout or API error: set confidence_tag = "SYSTEM_FALLBACK", skip narrative
  - Never block ML output waiting for LLM

L1 LATENCY (80ms):
  - All 4 base models must complete inference within 80ms combined
  - Use ONNX export or BentoML runner for production serving

MOMENTUM ALERT THRESHOLD:
  - Fire alert ONLY when delta > 0.15 (15 percentage points)
  - Source: prd.api_output_spec.momentum_alert_threshold_pct = 15
```

---

## 8. Environment Setup

```bash
# Python 3.11+
pip install -r requirements.txt

# Key packages
# xgboost lightgbm scikit-learn pandas pyarrow
# feast redis anthropic fastapi uvicorn
# evidently mlflow bentoml
# kafka-python scrapy playwright

# Environment variables required
ANTHROPIC_API_KEY=...          # For L3 LLM layer
CRICBUZZ_API_KEY=...           # For T2 live data
REDIS_URL=redis://localhost:6379
FEAST_REGISTRY_PATH=./data/feast/registry.db
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 9. Validation Checklist Before Any PR

Run these in order. All must pass:

```bash
# 1. PRD schema integrity
python3 -c "import json; json.load(open('ipl_prd.json')); print('PRD JSON valid')"

# 2. Feature leakage tests
python data/features/leakage_tests.py

# 3. Model output schema validation
pytest tests/test_models.py -v

# 4. API contract tests (all 11 signals)
pytest tests/test_api.py -v

# 5. Latency simulation
pytest tests/test_latency.py -v --benchmark
```

---

## 10. Open Questions to Resolve Before Specific Phases

Before Phase 0: resolve OQ5 (cloud provider selection)
Before Phase 1: resolve OQ2 (injury flag as hard vs soft feature)
Before Phase 3: resolve OQ3 (LLM cost at 74-match scale — run the cost sim first)
Before Phase 5: resolve OQ4 (fantasy platform data sharing agreement)
Before Phase 6: resolve OQ1 (pitch condition real-time data sourcing)

Track these in a `DECISIONS.md` file. When resolved, update `ipl_prd.json` under `open_questions`
by setting a `resolution` key on the relevant item.

---

## 11. How to Update the PRD

Never hand-edit `ipl_prd.json`. Use the patch script:

```bash
python prd_patch.py --key "open_questions[0].resolution" --value "Resolved: using historical venue profiles only for v1"
python prd_patch.py --key "meta.version" --value "1.1"
```

If `prd_patch.py` does not exist yet, create it as your first task.

---

*This file is the contract between the PRD and the codebase. When in doubt, the PRD wins.*
