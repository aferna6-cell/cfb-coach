# Sprint 12 — Football intelligence on the offensive coordinator

The experimental Madden offense is one coordinator, not a second playcaller.
It still scores **every situationally eligible play** in the confirmed installed
book with the trained success model (`model_primary_contextual_variety.v2`).
Sprint 11 adds a joint choice of `(formation, play, adjustment plan)` on top
of that baseline. Heuristic selection remains the timeout and error fallback.

This does not merge to `main`, rewrite historical games, install a playbook,
arm a macro, or turn on human-opponent control.

## What the model may do

| Stage | Model | User |
| --- | --- | --- |
| Pregame portfolio | Scores every catalogued formation across normal, short yardage, third-and-long, red zone, goal line, backed-up, two-minute and clock situations. Chooses a formation portfolio. Installing a formation keeps **every** source-book play. | Physically edits Madden, then attests with `ml offense-design --stage` and `--confirm-installed`. |
| Live call | Picks a legal formation and play, then a legal adjustment plan or **NO ADJUSTMENT**. | Runs the play. The compact window shows `FORMATION — PLAY` and, only when an adjustment won, the controller steps under it. |
| Macro ideas | Composes sourced hot routes and protections, rejects conflicts, and stores a **draft** blueprint. | Verifies the editor rows and arms a free slot. Until then the draft is not callable. |
| Learning | Trains a shadow comparison only from verified executions plus an explicit "I applied the adjustment" or "I ran it unchanged" confirmation. Promotion is a separate command and shifts a score by at most 0.04. | Confirms what actually happened. A shown call is not a training label. |

## What the model must not assume

- A proposed playbook is not the applied playbook.
- A draft or verified-but-unarmed macro is not executable.
- A suggested adjustment was not applied unless the user checks the box.
- A previous coverage observation is not this snap's coverage.
- A research prior is not a measured causal gain.
- An on-the-fly button sequence is not a saved Custom Adjustment slot.
- No expected-points or win-probability model is claimed. Situation effects are named proxies in `football_situation.py`.
- CPU experimental mode stays opt-in. Human opponents stay on the existing explicit opt-in. Defense remains shadow-only.

## Live path

1. Situational pool from the full confirmed book (third-and-long prefers passes when any exist).
2. Model probabilities for every play in that pool.
3. Model-primary sampling, including anti-repeat. This is not a fixed rotation.
4. Joint search over the complete legal action space: every unmodified play, every legal single adjustment, every compatible multi-adjustment, and every verified armed macro. Each play keeps its best legal plan. A different unmodified play wins when its complete score is higher. Anti-repeat stays inside the model-primary selection score. Scores inside a 0.012 band are explored with a snap hash, not a fixed rotation. If recent recommendations are concentrated, that band can widen by at most 0.06 so a close alternative formation, concept, package, or adjustment can be sampled. A play whose learned probability is more than 0.10 ahead is not displaced. There is no play quota and no fixed rotation. A research prior cannot move a plan by more than 0.06. Football-knowledge priors are capped at 0.025 and drive-strategy priors at 0.015, and neither is added inside the model-primary selection score. A low-confidence prior does not overturn a clearly stronger learned play. The live path uses the compiled knowledge store and does not call a language model.
5. If the full path exceeds 150 ms, or the pick is illegal, the call rolls back to the heuristic.

Detailed probabilities stay in the decision record and the expandable "Why this adjustment" section.

## Pregame through postgame

```bash
cd /path/to/cfb-coach
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

# Read-only plan. Does not stage or install.
python -m cfb_coach ml offense-coordinator -o cpu --max-formations 15 --summary

# Same plan in the browser. Still not installed.
python -m cfb_coach ml offense-design -o cpu --max-formations 15 --no-open

# After you build the formations in Madden:
python -m cfb_coach ml offense-design -o cpu --stage
python -m cfb_coach ml offense-design -o cpu --confirm-installed PROPOSAL_ID \
  --attest "I installed every listed formation and all of its plays in the Madden editor."

# Draft macros only. Nothing is armed.
python -m cfb_coach ml offense-macro-lab
python -m cfb_coach ml offense-macro-lab --stage

# CPU experimental offense. Heuristic returns on timeout.
python -m cfb_coach ml experimental --retrain
python -m cfb_coach play --game madden27

# After verified snaps exist:
python -m cfb_coach ml offense-action-learn --train
python -m cfb_coach ml offense-action-learn --promote   # refuses sparse or one-game data
python -m cfb_coach ml offense-action-learn --rollback

# Synthetic comparison against the previous model-primary sampler.
python -m cfb_coach ml offense-coordinator --compare
```

Replace `PROPOSAL_ID` with the id printed by `--stage`. Do not point these commands at a copy of the franchise database you have not backed up. `ml experimental` writes the coach's own SQLite meta (mode and a shadow action model). It does not edit Madden's franchise file.

## CPU pilot on Ubuntu

`offense-coordinator`, `offense-design`, and `offense-design --stage` share `--max-formations` (default 15, maximum 15). Fifteen is the maximum, not a quota: the portfolio stops when another formation adds no situation coverage and no concept family. The same opponent, catalog, and limit produce the same `proposal_id` and `inventory_id`. Confirm checks that fingerprint. Do this only after the formations are actually built in Madden. An explicit smaller limit, such as `--max-formations 8`, is still valid.

The compact live window stays `FORMATION — PLAY`, with adjustment steps underneath only when an adjustment won. Known, inferred, and missing situation inputs are on the decision record and in the existing experimental line.

```bash
cd /path/to/cfb-coach
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

# 1. Back up the existing coach database. The source file is not modified.
python3 -m cfb_coach ml find-db
python3 -m cfb_coach ml backup-db

# 2. Select and stage the offense. 15 is the maximum, not a required count.
#    Preview and stage must print the same proposal_id and inventory_id.
python3 -m cfb_coach ml offense-coordinator -o cpu --max-formations 15
python3 -m cfb_coach ml offense-design -o cpu --max-formations 15 --text
python3 -m cfb_coach ml offense-design -o cpu --max-formations 15 --stage --text

# 3. Build that book in Madden, then confirm. Do not confirm before it is installed.
python3 -m cfb_coach ml offense-design --confirm-installed PROPOSAL_ID \
  --attest "I installed every listed formation and all of its plays in the Madden editor."

# 4. Retrain from verified executions already in the coach database.
#    This does not rewrite snap history. Action learning stays in shadow until promote.
python3 -m cfb_coach ml experimental --retrain
python3 -m cfb_coach ml offense-action-learn --train

# 5. Run the experimental offensive coordinator.
python3 -m cfb_coach play --game madden27

# 6. Record the play you actually ran, whether you applied the adjustment, and the outcome.
#    Check the adjustment box only if you applied it.
#    Check "ran unchanged" only if you ran the play with no adjustment.
#    Leaving both unchecked does not count as either.
#    Optional sit line: 3rd and 7 my 21 clock 1:24 timeouts 2 q4 score 21-14 showing cover 1
#    A previous snap's clock is not reused. "cover 1" without "showing" is the previous look.

# 7. Read-only postgame report. This does not write the database.
python3 -m cfb_coach ml offense-report
python3 -m cfb_coach ml offense-report --game-id SESSION_ID

# Inspect compiled football knowledge and the current strategic plan.
# Neither command installs a formation or rewrites snap history.
python3 -m cfb_coach ml football-knowledge --concept mesh
python3 -m cfb_coach ml offense-strategy -o cpu

# 8. Return to the previous heuristic coach.
python3 -m cfb_coach ml heuristic
```

Replace `PROPOSAL_ID` with the id printed by `--stage`. Replace `SESSION_ID` with the game id from the report.

## Evidence that is not in this environment

This workspace has `~/.cfb-coach/coach.db` and no `madden27.db`. The coach database has opponents and zero snaps, zero sessions, and zero ML decisions. It is not the Madden game log. `ml offense-report` says that explicitly when the Madden database is absent.

The user reported two earlier games. They are context from an older ML build, not Sprint 12 measurements, and they are not stored in this workspace:

- `9f2ebdeb9d8f4e2d`: reported win 28–7, 63 recommendations, 52 verified executions, no recommended adjustments. Three formations, three labeled concept families, and 46 of 63 recommendations from Gun Doubles Clamp Stack. That concentration is a diversity baseline from an earlier coordinator, not a Sprint 12 result and not a measure of decision quality. A more varied synthetic call sheet is not evidence the offense would have scored more than 28.
- `d2e3214fbb944af9`: 4 recommendations and 2 verified executions. No stored final score. Not a completed win.

Regression checks use fixtures marked synthetic. A synthetic call that looks better is not a win-rate claim.

## Football knowledge

`cfb_coach/madden/model/football_knowledge.py` is a versioned store. General principles, Madden research, verified in-game details, empirical results, and hypotheses stay in separate layers. A play name is a hypothesis about a concept family, not a route diagram. Missing assignments and controller inputs stay unknown. Defensive diagnosis separates observed, inferred, and unknown looks. A two-high shell is not Cover 2 or Quarters, and a previous snap is not the current coverage. Sparse opponent shells use a uniform Bayesian prior and are published only after the sample actually moves that prior.

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests -p no:cacheprovider -q
PYTHONPATH=. python3 -m pytest tests/madden_ml/test_sprint12_football_intelligence.py tests/madden_ml/test_sprint11_offensive_coordinator.py tests/madden_ml/test_sprint11_1_hardening.py -q
```
