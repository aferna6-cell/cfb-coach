# Madden ML coach — Ubuntu workflow

Opt-in ML commands. Default coaching mode remains **heuristic**. Hybrid stays inactive this sprint.

Provenance: continues PR #20 (`madden-ml/coach`). QA baseline from `madden-ml/qa` is filed at `docs/madden_ml/QA_BASELINE.md`.

## Setup

```bash
cd /workspace   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Inspect / export training data

```bash
# Coverage report (Madden DB + optional extra files)
python -m cfb_coach ml inspect --path tests/fixtures/madden_ml/labeled_offense_snaps.jsonl
python -m cfb_coach ml inspect --path tests/fixtures/cpu_19-0_offense_snaps.csv

# Export JSONL
python -m cfb_coach ml export --out /tmp/madden_ml_rows.jsonl \
  --path tests/fixtures/madden_ml/labeled_offense_snaps.jsonl \
  --path tests/fixtures/cpu_19-0_offense_snaps.csv
```

Recommendations are never treated as verified executions. Missing labels stay unknown.

## Train / evaluate / registry

```bash
python -m cfb_coach ml train --seed 7 \
  --path tests/fixtures/madden_ml/labeled_offense_snaps.jsonl \
  --path tests/fixtures/cpu_19-0_offense_snaps.csv \
  --registry ~/.cfb-coach/madden_ml_registry

python -m cfb_coach ml evaluate --seed 7 \
  --path tests/fixtures/madden_ml/labeled_offense_snaps.jsonl \
  --path tests/fixtures/cpu_19-0_offense_snaps.csv \
  --registry ~/.cfb-coach/madden_ml_registry

python -m cfb_coach ml list --registry ~/.cfb-coach/madden_ml_registry
python -m cfb_coach ml status
```

A gate of `insufficient` or `failed` means the model may be used for **shadow** analysis only. Hybrid promotion remains off.

## Shadow mode

```bash
python -m cfb_coach ml shadow --registry ~/.cfb-coach/madden_ml_registry
# live call still prints the heuristic / VOD recommendation
python -m cfb_coach call --game madden27 -o <opp> -s "1&10 my 25"

python -m cfb_coach ml report
python -m cfb_coach ml heuristic    # or: ml shadow --off
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests -p no:cacheprovider -q
PYTHONPATH=. python -m pytest tests/madden_ml -p no:cacheprovider -q
```
