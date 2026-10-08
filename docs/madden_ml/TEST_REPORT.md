# Madden ML coach — test report (Sprint 3)

Branch: `madden-ml/coach` (PR #20). Environment: Ubuntu, Python 3.12, `pip install -e '.[dev]'`, `PYTHONPATH=.`.

## Full suite (run twice)

```bash
PYTHONPATH=. python3 -m pytest tests -p no:cacheprovider -q
```

| Run | Result |
| --- | --- |
| 1 | **373 passed** in ~46s |
| 2 | **373 passed** (repeat) |

No nondeterministic failures observed across the two full runs.

`tests/madden_ml`: **42 passed** (schema + pipeline + Sprint 3 live/eligibility/identity).

## Sprint 3 coverage added

- Shared `evaluate_live_shadow` for HTML / terminal / one-shot
- `LiveDecisionTracker` unique game/snap ids (never opponent_id as game_id)
- HTML execution verification (used recommended / different / unknown)
- Outcome linkage via `ml_snap_id` + `ml_outcomes`
- Training eligibility + supervised-only play-action training
- Hardened promotion gate (train-only baseline; mins 80/30/4; +0.02 log-loss)
- Feature schema `madden-ml.features.2` (family one-hots; no numeric hashes)
- Sealed decision pipeline behind `CFB_COACH_SEALED_PIPELINE` (default off)
- Local DB find/backup/inspect/export/sanitized + richer `ml report`

## Integration demonstration type

**Controller-method + temporary-database E2E** (`tests/madden_ml/test_sprint3_live.py`) plus fixture-based software demonstration.

Data used in fixtures:

- `tests/fixtures/madden_ml/labeled_offense_snaps.jsonl` — labeled offense rows (execution **unverified** → not supervised-eligible)
- `tests/fixtures/cpu_19-0_offense_snaps.csv` — historical offense snaps (recommendations only)

Verified executions in fixtures: **0**. Supervised training from fixtures alone yields gate **insufficient** (correct). Hybrid promotion remains blocked.

Shadow demo: heuristic pick unchanged; model failures do not interrupt live calls.

## Real vs fixture counts (this environment)

| Source | Count |
| --- | --- |
| Personal Franchise DB on agent VM | **0** (not present; do not invent) |
| Fixture labeled JSONL | 20 rows / 4 games |
| Fixture CPU CSV | 35 rows |
| Verified executions (fixtures) | 0 |
| E2E temp-DB snaps (acceptance test) | generated per run |

Use `python -m cfb_coach ml find-db` / `ml inspect` on the Franchise laptop to count real games.
