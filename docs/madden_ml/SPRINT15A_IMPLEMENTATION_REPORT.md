# Sprint 15A Implementation Report

Base: `cursor/sprint14-1-film-review-5f33` @ `79ebaf6`  
Branch: `cursor/sprint15a-expert-vod-9556`

## Delivered

1. **Architecture audit** — Documented how `vod_success` / `vod_model` interact with the Madden experimental coordinator (`VOD_PRIOR_INTERACTION`, `SPRINT15A_EXPERT_LEARNING.md`). Joint path does not call `vod_model`.
2. **Three evidence categories** — `learning_sources.py` retains expert, personal, and general evidence independently with fingerprint dedupe.
3. **Expert film ingestion** — `expert_film.py` extends the Sprint 14.1 film pipeline with provenance manifests, permission statuses, and human snap annotations (invisible fields stay unknown). No download/scrape.
4. **Expert policy + outcome models** — `expert_policy.py` trains confidence-aware contextual preferences (concept/formation/play/family/adjustment + drive transitions) and a separate verified-outcome model.
5. **Personalization** — `personalization.py` blends expert and personal evidence by context/concept using effective sample size and uncertainty (no fixed game/% schedule).
6. **Model-primary coordinator** — `expert_signal.py` + joint wiring: cap `0.018`, shadow default, explicit promote/rollback, playbook guards.
7. **Explainable CLI** — `expert-learning`, `personal-learning`, `learning-compare`, plus import/annotate/export/train/eval commands.
8. **Evaluation** — Match/expert holdout + chronological personal eval; no counterfactual success claims.
9. **Pilot fixtures** — `tests/fixtures/expert_film/` synthetic reviewed annotations (not pro-game results).

## Performance / contracts retained

- Formation portfolio max 15 (unchanged designer)
- Live decision budget 150 ms (unchanged schema)
- Model-primary play-success scoring remains the base joint score
- Expert live delta ≤ `0.018` and only after gated promotion

## Test results

```text
PYTHONPATH=. python3 -m pytest tests/madden_ml/test_sprint15a_expert_learning.py -p no:cacheprovider -q
13 passed

PYTHONPATH=. python3 -m pytest tests -p no:cacheprovider -q
635 passed, 1 skipped in 62.72s
```

## Exact Ubuntu commands (pilot)

```bash
cd /path/to/cfb-coach
python3 -m pip install -r requirements.txt
export PYTHONPATH=.

FILM="$HOME/.cfb-coach/film/expert"
LEARN="$HOME/.cfb-coach/learning"

python3 -m cfb_coach ml expert-film-import \
  tests/fixtures/expert_film/pilot_match_a.json \
  --expert-id fixture-elite-1 --match-id expert-pilot-a \
  --store "$FILM" --permission-status fixture_synthetic --skip-decode

# Copy reviewed fixture annotations into the store (or use film-review / expert-annotate)
mkdir -p "$FILM/annotations"
cp tests/fixtures/expert_film/pilot_match_a.json "$FILM/annotations/expert-pilot-a.json"
cp tests/fixtures/expert_film/pilot_match_b.json "$FILM/annotations/expert-pilot-b.json"
python3 -m cfb_coach ml expert-film-import \
  tests/fixtures/expert_film/pilot_match_b.json \
  --expert-id fixture-elite-2 --match-id expert-pilot-b \
  --store "$FILM" --permission-status fixture_synthetic --skip-decode

python3 -m cfb_coach ml expert-export-evidence --store "$FILM" --learning-store "$LEARN"
python3 -m cfb_coach ml prepare-expert-dataset --learning-store "$LEARN" --out /tmp/expert_dataset.json
python3 -m cfb_coach ml train-expert-policy --learning-store "$LEARN" --no-register
python3 -m cfb_coach ml expert-learning --summary --learning-store "$LEARN" --store "$FILM"
python3 -m cfb_coach ml personal-learning -o cpu --learning-store "$LEARN" --ingest-general
python3 -m cfb_coach ml learning-compare --concept mesh --learning-store "$LEARN"
python3 -m cfb_coach ml learning-eval --learning-store "$LEARN"
python3 -m cfb_coach ml learning-eval --learning-store "$LEARN" --rollback
```
