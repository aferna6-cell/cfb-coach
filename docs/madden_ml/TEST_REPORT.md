# Madden ML coach — test report (Sprint 3.1 + HTML idempotency)

Branch: `madden-ml/coach` (PR #20). Environment: Ubuntu, Python 3.12, `pip install -e '.[dev]'`, `PYTHONPATH=.`.

## Full suite (run twice)

```bash
PYTHONPATH=. python3 -m pytest tests -p no:cacheprovider -q
```

| Run | Result |
| --- | --- |
| 1 | see latest push notes |
| 2 | see latest push notes |

CI: `.github/workflows/pytest.yml` runs the full suite twice (includes Playwright Chromium for browser idempotency).

HTML browser pad now mints/reuses `idempotency_key` via `claimIdempotencyKey` / `apiAction` / `withUiLock` (sessionStorage-backed). Covered by `tests/madden_ml/test_html_idempotency.py` and `tests/madden_ml/test_browser_idempotency_playwright.py`.

## Sprint 3.1 integrity coverage

- Training features attribute results to **executed** action (Gun Trips / Inside Zone), not recommendation (Mesh)
- Unknown execution excluded from supervised `_row_vector`
- Trusted VOD / outside-book executed plays
- Canonical dedupe across snaps / play_records / ml_* (one active row; latest outcome)
- HTTP idempotency on `/api/result_call` (duplicate POST does not double-advance)
- Undo voids `ml_outcomes` into `ml_outcome_audit` and allows relog
- SQLite `backup()` API + read-only `mode=ro` inspect (no migration)
- Eval: zero baseline rate, mixed recommendation-only skipped, corrupted artifact fails, hybrid_activation=false
- Two-game HTML readiness acceptance (backup → inspect → supervised export → train/eval)

## Laptop readiness

See final agent report YES/NO and `docs/madden_ml/WORKFLOW.md` smoke-test steps.
