# Sprint 15A — Expert VOD Foundation and Personalized Offensive Learning

## Architecture audit (VOD priors vs joint coordinator)

| Path | Role on this branch |
| --- | --- |
| `cfb_coach/vod_success/` | Offline trainer for v0.7 beater cells |
| `cfb_coach/vod_model/` | Heuristic prep/live **late** call prior (`apply_madden_call`) |
| `cfb_coach/madden/model/offense_joint_decision.py` | Model-primary experimental coordinator |
| Sprint 15A `expert_signal` | Bounded shadow/live prior **inside** joint scoring |

The experimental coordinator does **not** call `vod_model`. Expert/personal learning enters only through `expert_learning_adjustment` with cap `EXPERT_SIGNAL_CAP = 0.018`. This prevents double-counting VOD beaters with the new expert policy and prevents an older VOD scoring path from overriding joint decisions.

Documented machine-readable contract: `learning_sources.VOD_PRIOR_INTERACTION`.

## Evidence categories

Retained independently under `~/.cfb-coach/learning/{expert,personal,general}/`:

1. **Expert evidence** — human-reviewed elite-player demonstrations (action ≠ outcome)
2. **Personal evidence** — user’s verified executions only (never unexecuted recommendations)
3. **General football evidence** — compiled knowledge layers with provenance/uncertainty

## Pilot workflow (Ubuntu)

Fixtures ship under `tests/fixtures/expert_film/`. They are synthetic annotations, not professional-game results.

```bash
# 0) Stores
FILM="$HOME/.cfb-coach/film/expert"
LEARN="$HOME/.cfb-coach/learning"
mkdir -p "$FILM" "$LEARN"

# 1) Import an authorized local recording (or fixture with --skip-decode)
python3 -m cfb_coach ml expert-film-import \
  tests/fixtures/expert_film/pilot_match_a.json \
  --expert-id fixture-elite-1 --match-id expert-pilot-a \
  --store "$FILM" --permission-status fixture_synthetic --skip-decode

# Real footage (user-authorized local file; never auto-downloaded):
# python3 -m cfb_coach ml expert-film-import /path/to/game.mp4 \
#   --expert-id playerX --match-id match-2026-01-01 \
#   --store "$FILM" --permission-status user_authorized_local \
#   --competitive-mode ultimate_team --opponent-type human --patch title-update-1

# 2) Review via existing film-review UI, or apply label JSON
python3 -m cfb_coach ml film-review --game-id expert-pilot-a --store "$FILM" --html /tmp/review.html
python3 -m cfb_coach ml expert-annotate --match-id expert-pilot-a --candidate-id manual-0 \
  --store "$FILM" --labels /path/to/labels.json

# 3) Export reviewed snaps into independent expert evidence
python3 -m cfb_coach ml expert-export-evidence --store "$FILM" --learning-store "$LEARN"

# 4) Prepare dataset + train (shadow by default)
python3 -m cfb_coach ml prepare-expert-dataset --learning-store "$LEARN" --out /tmp/expert_dataset.json
python3 -m cfb_coach ml train-expert-policy --learning-store "$LEARN"

# 5) Ingest personal verified snaps + reports
python3 -m cfb_coach ml personal-learning -o cpu --learning-store "$LEARN" --ingest-general
python3 -m cfb_coach ml expert-learning --summary --learning-store "$LEARN" --store "$FILM"
python3 -m cfb_coach ml learning-compare --concept mesh --learning-store "$LEARN"

# 6) Shadow evaluation (no live influence)
python3 -m cfb_coach ml learning-eval --learning-store "$LEARN"

# 7) Explicit promotion only after gates pass; rollback anytime
python3 -m cfb_coach ml learning-eval --learning-store "$LEARN" --promote
python3 -m cfb_coach ml learning-eval --learning-store "$LEARN" --rollback
```

## Live influence gates

- Default mode: `shadow` (reports `shadow_delta`, applies `delta = 0`)
- Promote requires match holdout, expert holdout, and non-degradation checks
- Cap remains `0.018`; installed playbook is never replaced; unverified formations/adjustments are withheld
- 15-formation portfolio, 150 ms budget, model-primary scoring, opponent learning, and football intelligence are unchanged

## Modules

| Module | Responsibility |
| --- | --- |
| `learning_sources.py` | Source taxonomy + retention |
| `expert_film.py` | Manifest, annotations, export |
| `expert_policy.py` | Policy + outcome trainers |
| `personalization.py` | Context/concept blend weights |
| `expert_signal.py` | Shadow/promote/rollback + joint delta |
| `learning_reports.py` | CLI report bodies |
| `learning_eval.py` | Controlled comparisons |
