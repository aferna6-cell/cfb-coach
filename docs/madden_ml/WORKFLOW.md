# Madden ML coach — Ubuntu workflow (Sprint 3)

Opt-in ML commands. Default coaching mode remains **heuristic**. Hybrid stays inactive.

Provenance: PR #20 (`madden-ml/coach`). Do not merge without approval. Do not activate hybrid calling.

## Setup

```bash
cd /workspace   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Local Franchise database (your laptop)

Agents do not have your personal Madden DB. Use these commands on the machine that plays Franchise games. The source database is never modified by inspect/export/train.

```bash
# 1. Find the active Madden database (honors $CFB_COACH_MADDEN_DB, else ~/.cfb-coach/madden27.db)
python -m cfb_coach ml find-db

# 2. Back it up safely (timestamped copy; source untouched)
python -m cfb_coach ml backup-db
# optional: python -m cfb_coach ml backup-db --out ~/madden-backups

# 3. Inspect without modifying the source
python -m cfb_coach ml inspect

# 4. Validate historical game records / data-quality report
python -m cfb_coach ml inspect
# report fields: total_snaps, unique_games, verified_executions,
# supervised_training_rows, outcome_only_examples, cpu_vs_human, eligibility, duplicates

# 5. Export training-ready rows (full history; does not invent data)
python -m cfb_coach ml export --out /tmp/madden_ml_rows.jsonl

# Supervised play-specific rows only (verified execution or trusted VOD)
python -m cfb_coach ml export --out /tmp/madden_ml_supervised.jsonl --supervised-only

# Optional sanitized export for debugging shares (ids hashed; paths dropped)
python -m cfb_coach ml export --out /tmp/madden_ml_sanitized.jsonl --sanitized

# 6. Train and evaluate from the local dataset
python -m cfb_coach ml train --seed 7 --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach ml evaluate --seed 7 --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach ml list --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach ml status
```

**Do not commit** `madden27.db`, backups, or personal JSONL exports to GitHub.

When no local database exists, commands report empty counts. They do not fabricate Franchise snaps.

## Live HTML coaching + execution verification

```bash
# Optional shadow scoring (heuristic/VOD call still displayed)
python -m cfb_coach ml shadow --registry ~/.cfb-coach/madden_ml_registry

# Default HTML live coach (execution verify enabled for Madden)
python -m cfb_coach play --game madden27 -o <opponent>

# After each play, optionally confirm:
#   unknown (default) | used recommended | used different
# Recommendations are never auto-marked as executed.

python -m cfb_coach ml report          # per-game shadow + data-quality
python -m cfb_coach ml heuristic       # or: ml shadow --off
```

Terminal / one-shot paths share the same `evaluate_live_shadow` hook and session/snap identity as HTML.

## Training eligibility

| Class | Supervised play-action training |
| --- | --- |
| `verified_execution` | Yes |
| `trusted_vod` | Yes |
| `outcome_known_execution_uncertain` | No (kept for tendencies / analysis) |
| `recommendation_only` | No |
| `unlabeled` | No |
| `excluded` | No |

## Promotion gate (conservative)

Pre-chosen thresholds (not tuned after seeing favorable metrics):

- Min supervised labeled rows: **80**
- Min supervised holdout rows: **30**
- Min holdout games/groups: **4**
- Min absolute log-loss improvement vs **train-only** baseline: **0.02**
- Baseline parameters come from training data only (no holdout leakage)
- Recommendation-only rows cannot pass the gate
- A passing offline gate does **not** authorize hybrid; shadow validation + explicit approval are still required

## Sealed decision pipeline

Disabled by default. Enable only for parity experiments:

```bash
export CFB_COACH_SEALED_PIPELINE=1
# or meta: ml_sealed_pipeline=1
```

Heuristic compatibility mode keeps displaying the legacy call and emits a difference report.

## Fixture / CI inspect (no personal DB)

```bash
python -m cfb_coach ml inspect --path tests/fixtures/madden_ml/labeled_offense_snaps.jsonl
python -m cfb_coach ml inspect --path tests/fixtures/cpu_19-0_offense_snaps.csv
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests -p no:cacheprovider -q
PYTHONPATH=. python -m pytest tests/madden_ml -p no:cacheprovider -q
```
