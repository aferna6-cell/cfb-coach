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


## V2 implementation update (Nov 9)

Current development branch includes guarded live V2 selector, model-scored pregame stock-formation proposal, observed multi-outcome risk, confidence-gated learned defensive structural features, earlier-snap opponent beliefs and offline champion/challenger postgame evaluation. All experiments are opt-in; no main franchise changes or automatic model promotion.

### Test franchise commands (read-only except writing temporary shadow artifacts)

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py inventory --db /PATH/TO/TEST/madden27.db
    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py audit --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE"
    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py train --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" --out /tmp/test-defense-v2-shadow.json
    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py evaluate --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE"
    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py design --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" --model /tmp/test-defense-v2-shadow.json --out /tmp/test-defense-v2-playbook-proposal.json

For the explicitly isolated test franchise, after confirming a physical installed book matches the stored book and the shadow artifact was trained before the current game:

    export CFB_COACH_DEFENSE_V2_TEST_FRANCHISE="EXACT_DYNASTY_VALUE"
    export CFB_COACH_DEFENSE_V2_ARTIFACT=/tmp/test-defense-v2-shadow.json
    python3 -m cfb_coach play --game madden27 --defense-model v2 --cpu-defense ...

The final command is a developer-facing example; fill in the actual required opponent arguments using play --help. In terminal, choose defense side (d). Omit --defense-model v2 for instant rollback next session. V2 cannot run against the default primary Madden DB and rejects an unknown or mismatched game/franchise.

Postgame challenge (never promotes weights):

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py postgame --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" --model /tmp/test-defense-v2-shadow.json --out /tmp/test-defense-v2-next-candidate.json

### What is and is not ready

Implemented: actual optional V2 model selection path, verified-execution-only CPU and human rows, stock installed book restriction, a maximum of 15 formations, learned opponent concept probabilities, structured coverage semantics, outcome utility estimates and independent challenger training. All individual defender responsibilities that lack diagram verification are UNKNOWN. Model-scored pregame proposals do not automatically install or arm themselves; they still require the existing physical editor verification workflow.

Not implemented or confirmed: individual defensive adjustment effect learning and safe adjusted-play scoring, physically verified controller actions, full eleven-defender role assignments for every play, and loaded local full-book 150 ms performance. Real-game execution capture and snap chronology need to be checked on the owner's machine. A good predictive metric does not prove the policy improves win rate. Main franchise is unchanged.

**Current overall verdict: NO-GO for declaring the complete same-as-offense coordinator ready.** An experimental stock-play-only V2 test is possible only after the local setup and performance gates pass. With no verified defensive executions in the new test franchise, use the established caller and collect confirmations first.
