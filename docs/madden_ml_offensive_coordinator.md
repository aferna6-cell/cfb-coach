# Sprint 11.1 — Offensive coordinator hardening

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
4. Joint search over the complete legal action space: every unmodified play, every legal single adjustment, every compatible multi-adjustment, and every verified armed macro. Each play keeps its best legal plan. A different unmodified play wins when its complete score is higher. Anti-repeat stays inside the model-primary selection score. Scores inside a 0.012 band are explored with a snap hash, not a fixed rotation. A research prior cannot move a plan by more than 0.06, so it does not overturn a clearly stronger learned play.
5. If the full path exceeds 150 ms, or the pick is illegal, the call rolls back to the heuristic.

Detailed probabilities stay in the decision record and the expandable "Why this adjustment" section.

## Pregame through postgame

```bash
cd /path/to/cfb-coach
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

# Read-only plan. Does not stage or install.
python -m cfb_coach ml offense-coordinator -o cpu --max-formations 8 --summary

# Same plan in the browser. Still not installed.
python -m cfb_coach ml offense-design -o cpu --max-formations 8 --no-open

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

`offense-coordinator`, `offense-design`, and `offense-design --stage` share `--max-formations` (default 8). The same opponent, catalog, and limit produce the same `proposal_id` and `inventory_id`. Confirm checks that fingerprint. Do this only after the formations are actually built in Madden.

The compact live window stays `FORMATION — PLAY`, with adjustment steps underneath only when an adjustment won. Known, inferred, and missing situation inputs are on the decision record and in the existing experimental line.

```bash
cd /path/to/cfb-coach
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

# 1. Back up the existing coach database. The source file is not modified.
python3 -m cfb_coach ml find-db
python3 -m cfb_coach ml backup-db

# 2. Select and stage the eight-formation offense.
#    Preview and stage must print the same proposal_id and inventory_id.
python3 -m cfb_coach ml offense-coordinator -o cpu --max-formations 8
python3 -m cfb_coach ml offense-design -o cpu --max-formations 8 --text
python3 -m cfb_coach ml offense-design -o cpu --max-formations 8 --stage --text

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

# 8. Return to the previous heuristic coach.
python3 -m cfb_coach ml heuristic
```

Replace `PROPOSAL_ID` with the id printed by `--stage`. Replace `SESSION_ID` with the game id from the report.

## Evidence that is not in this environment

This workspace has `~/.cfb-coach/coach.db` and no `madden27.db`. The coach database has opponents and zero snaps, zero sessions, and zero ML decisions. It is not the Madden game log. The two reported CPU blowout wins are not available here. `ml offense-report` says that explicitly when the Madden database is absent. Regression checks use fixtures marked synthetic. Do not treat those scenarios as the user's games.

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests -p no:cacheprovider -q
PYTHONPATH=. python3 -m pytest tests/madden_ml/test_sprint11_offensive_coordinator.py tests/madden_ml/test_sprint11_1_hardening.py -q
```
