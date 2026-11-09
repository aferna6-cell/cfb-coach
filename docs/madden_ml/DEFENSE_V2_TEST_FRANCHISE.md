# Defensive Coordinator V2 — testing franchise foundation

**Status: shadow-only; NOT live-ready.** This branch implements a defensive risk learner and read-only pilot CLI. The main franchise, active model, installed books, and game database are not changed. This branch is stacked on the latest CPU2 development branch (PR #138), which also has outstanding validation requirements.

## Objectives

The secondary Madden franchise lets the user play every offensive and defensive snap. We will collect actual executed defensive calls, opponent offensive concepts and observed results, train a defensive policy, and test it without putting the main franchise at risk.

## What the foundation includes

- A verified-execution reader for **CPU and human** defensive snaps. The older defensive logistic reader excludes CPU opponents; this pilot does not.
- Exact franchise isolation through game_sessions.dynasty; an absent or empty identity fails closed.
- An empirical-Bayes outcome model with shrinkage at overall, context, coverage-family, concept-matchup and specific-play levels. The parameters are learned from real execution labels, not a hard-coded defensive counter table.
- In-game opponent concept beliefs derived from the current session's **strictly earlier, closed, sequence-numbered** snaps. Opponent observations can inform tendencies even when our own defensive execution has not been verified.
- A shadow-only ranking API that requires each candidate to be explicitly marked both installed and legal.
- A chronological, whole-game evaluation against the context-only outcome baseline. Games with no earlier training history are omitted; missing sequence numbers are excluded from time-ordered evaluation.
- A prediction-only holdout gate. This does **not** establish causal defensive policy improvement.

## Read-only developer commands

Use an entirely separate testing-franchise DB configuration and confirm the existing CFB_COACH_DB and CFB_COACH_MADDEN_DB values will not point to the primary franchise.

~~~bash
# Run from repository root. Replace file paths and EXACT recorded identity.
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py inventory --db /PATH/TO/TEST/madden27.db

PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py audit \
  --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE"

PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py evaluate \
  --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE"

PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py train \
  --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" \
  --out /tmp/cfb-defense-v2-shadow.json
~~~

The CLI opens the source SQLite database read-only. Only train writes a new shadow artifact to an explicitly supplied, non-live directory. Training requires at least one fully verified defensive execution. No executed call is inferred from a recommendation. A row missing snap_seq can be used for verified outcome fitting, but never for next-snap beliefs or chronological replay.

Developer-only preview requires a saved shadow artifact, exact test franchise/session/opponent/next sequence and a JSON list of candidate calls with explicit installed=true and legal=true. It rejects artifacts trained on the same game or on a game starting later. **It does not itself verify that caller-supplied installed flags are true; it must not operate live without the confirmed inventory lock.**

## Remaining first-game gates

1. Establish separate game database and clear franchise identity; confirm opponent and session IDs plus sequence numbers from actual post-play submissions.
2. Make sure the existing UI saves **actual executed** defensive formation/play (including CPU games), rather than the displayed recommendation; show explicit unknown when not confirmed.
3. Run targeted/full tests and the 150 ms loaded benchmark on the owner's laptop. The sandbox does not have the user's private DB or installed artifacts.
4. Connect V2 to the existing defensive coordinator only behind a testing-franchise opt-in, with full installed inventory/legality checks and fallback attribution.
5. Build model-selected pregame defensive formations (up to 15) and verify installation, plus legal individually confirmed defensive adjustments. Retain NO ADJUSTMENT.
6. Introduce safe postgame challenger training/evaluation and rollback; do not promote on an unproven outcome.
7. Evaluate real held-out games. Better outcome-prediction log loss alone is not evidence that following a different defensive policy will improve wins.

**The defensive V2 model is NOT yet ready for live playcalling. The testing franchise can still begin collecting high-value verified snaps using the existing caller.**

## Future work: shared offense/defense evidence

Track separately (a) defenses *observed from opponents* against our offense, (b) defenses *we actually executed* against CPU/human offenses, and (c) opponent offensive tendencies. Attach game, side, opponent, CPU/human type, outcome, exact play, confirmation/evidence tier and training cutoff. These can eventually share scheme-vs-concept features with strong shrinkage and a controlled ablation. An opponent successfully using a defense does **not** prove that our team, play or adjustments would have the same effect. Do not copy observed success directly into our defensive policy.
