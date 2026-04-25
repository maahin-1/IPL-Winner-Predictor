# DECISIONS.md — Open Questions Resolution Tracker
# Per CLAUDE.md §10 — resolve before the specified phase.

| ID   | Question                                                                 | Required Before | Status      | Resolution |
|------|--------------------------------------------------------------------------|-----------------|-------------|------------|
| OQ1  | Real-time pitch condition data from broadcast APIs vs historical profiles | Phase 6         | OPEN        |            |
| OQ2  | Injury/availability updates: hard feature vs soft signal                 | Phase 1         | OPEN        |            |
| OQ3  | LLM cost at 74-match scale — run cost_tracker.py simulation              | Phase 3         | RESOLVED    | ~$16.78/season (4,144 effective calls, 30% low-stakes skip rate). Acceptable. |
| OQ4  | Fantasy platform data sharing agreement for personalised context         | Phase 5         | OPEN        |            |
| OQ5  | Cloud provider selection: GCP vs AWS vs hybrid                           | Phase 0         | OPEN        |            |

## How to resolve

```bash
# After deciding, update the PRD:
python prd_patch.py --key "open_questions[0].resolution" --value "Resolved: using historical venue profiles only for v1"
```
