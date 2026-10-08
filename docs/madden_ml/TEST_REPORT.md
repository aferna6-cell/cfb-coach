# Madden ML coach — test report (sprint vertical slice)

Branch: `madden-ml/coach` (PR #20). Environment: Ubuntu, Python 3.12, `pip install -e '.[dev]'`, `PYTHONPATH=.`.

## Full suite (run twice)

```bash
PYTHONPATH=. python -m pytest tests -p no:cacheprovider -q
```

| Run | Result |
| --- | --- |
| 1 | **358 passed** in ~39s |
| 2 | **358 passed** in ~38s |

No nondeterministic failures observed across the two full runs.

`tests/madden_ml`: **27 passed** (schema + pipeline) on consecutive runs.

## Baseline vs this branch

From `docs/madden_ml/QA_BASELINE.md` (`madden-ml/qa` at `2def811`): collection aborted on missing numpy (320 collected / 1 error). This branch already includes numpy in the `dev` extra and the demo-cwd fix from earlier PR #20 commits. Current suite collects and passes **358** tests (schema + pipeline + prior suite + VOD flag tests from this branch tip).

## Integration demonstration type

**Fixture-based software demonstration** (not real-world play-calling effectiveness).

Data used:

- `tests/fixtures/madden_ml/labeled_offense_snaps.jsonl` — 20 labeled offense rows across 4 games
- `tests/fixtures/cpu_19-0_offense_snaps.csv` — 35 historical offense snaps (recommendations only; not verified executions)

Usable labeled rows: **55**. Verified executions: **0** (recommendations are never treated as verified actions).

Training produced `madden-ml.logit.offense.7`. Held-out gate: **insufficient** (holdout group size 5 &lt; 10). Model may be used for shadow analysis; hybrid promotion remains blocked.

Shadow demo: heuristic pick unchanged; model latency ~0.3 ms (budget 150 ms).
