# Madden ML coach — Ubuntu workflow (Sprint 3.1)

Opt-in ML commands. Default coaching mode remains **heuristic**. Hybrid stays inactive. Sealed pipeline flag stays **off**.

Provenance: PR #20 (`madden-ml/coach`). Do not merge without approval.

## Setup (Franchise laptop)

```bash
cd ~/cfb-coach   # or your clone path
git fetch origin madden-ml/coach
git checkout madden-ml/coach
git pull origin madden-ml/coach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional: point at your Madden DB explicitly:

```bash
export CFB_COACH_MADDEN_DB="$HOME/.cfb-coach/madden27.db"
```

## Safe local database operations

**Inspect and export never migrate or write the gameplay DB.** They open SQLite `mode=ro`.

```bash
# 1. Find the active Madden database
python -m cfb_coach ml find-db

# 2. Consistent backup (sqlite3 backup API; WAL-safe; source untouched)
python -m cfb_coach ml backup-db
# optional: python -m cfb_coach ml backup-db --out ~/madden-backups

# 3. Read-only inspect / data-quality report (no schema changes)
python -m cfb_coach ml inspect

# 4. Export (source read-only)
python -m cfb_coach ml export --out /tmp/madden_ml_rows.jsonl
python -m cfb_coach ml export --out /tmp/madden_ml_supervised.jsonl --supervised-only
python -m cfb_coach ml export --out /tmp/madden_ml_sanitized.jsonl --sanitized

# 5. Train / evaluate write only to the registry dir — not madden27.db
python -m cfb_coach ml train --seed 7 --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach ml evaluate --seed 7 --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach ml status
python -m cfb_coach ml report
```

### Explicit migration (only when needed)

If a historical DB requires schema migration, **back up first**, then migrate with an explicit flag:

```bash
python -m cfb_coach ml backup-db --out ~/madden-backups
python -m cfb_coach ml migrate-db --backup ~/madden-backups/madden27.backup.<stamp>.db
```

Never open the live gameplay DB with a migrating `CoachDB` merely to inspect it.

## Live HTML coaching + verified executions

```bash
python -m cfb_coach ml shadow --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach play --game madden27 -o <opponent>
```

After each play, optionally confirm: **unknown** (default) | **used recommended** | **used different**.

Recommendations are never auto-marked as executed. Supervised training attributes results to the **verified executed** play (or trusted VOD observation), not the recommendation when they differ.

The default HTML live pad sends `idempotency_key` on `/api/call`, `/api/result_call`, `/api/undo`, `/api/end_game`, and `/api/book_apply`. Keys are stored in `sessionStorage` so retries after network failure reuse the same key; a new intentional action mints a new key. Double-clicks are ignored while a request is in flight.

## Data-quality smoke test (laptop)

```bash
python -m cfb_coach ml find-db
python -m cfb_coach ml backup-db
python -m cfb_coach ml inspect
# Expect: unique_games, verified_executions, supervised_training_rows, eligibility breakdown
# Expect: read_only: True
python -m cfb_coach ml export --out /tmp/smoke_supervised.jsonl --supervised-only
python -m cfb_coach ml train --seed 7 --registry /tmp/madden_ml_registry_smoke
python -m cfb_coach ml evaluate --seed 7 --registry /tmp/madden_ml_registry_smoke
# Gate should remain insufficient/failed until enough authentic verified games exist.
# hybrid_activation remains false.
```

## Training eligibility

| Class | Supervised play-action training | Action features |
| --- | --- | --- |
| `verified_execution` | Yes | Executed formation/play |
| `trusted_vod` | Yes | Observed VOD action |
| `outcome_known_execution_uncertain` | No | None (recommendation kept separately) |
| `recommendation_only` | No | None |
| `unlabeled` / `excluded` | No | None |

## Sealed pipeline

Disabled. See `docs/madden_ml/SEALED_PIPELINE.md`. Do not set `CFB_COACH_SEALED_PIPELINE=1` during real games.

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests -p no:cacheprovider -q
PYTHONPATH=. python3 -m pytest tests/madden_ml -p no:cacheprovider -q
```
