# Sprint 11 — Adaptive offensive coordinator

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
4. Joint search: every eligible play is scored. `NO ADJUSTMENT` on the sampled play is the default. A sourced hot route, protection, compatible pair, or armed macro can replace it only when its research margin clears the no-action bar after an execution-cost penalty.
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

## Evidence that is not in this environment

The two reported CPU blowouts were not found as logs or as a franchise SQLite file in this workspace. Regression checks use fixtures marked `synthetic`. Do not treat those scenarios as the user's games.

## Tests

```bash
PYTHONPATH=. python -m pytest tests -p no:cacheprovider -q
PYTHONPATH=. python -m pytest tests/madden_ml/test_sprint11_offensive_coordinator.py -q
```
