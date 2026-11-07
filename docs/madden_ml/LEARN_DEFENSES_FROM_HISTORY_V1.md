# Madden NFL 27 — teaching defense from real logged games (baseline v1)

**Draft research / offline-only.** This is not a new live rules-based playcaller and does not promote any existing model. It extends PR #107's Madden coverage intelligence and PR #110's offensive route-art ML features.

## What we mean by "understand football"

The model receives two different kinds of information:

1. **Schematic football knowledge**: `coverage_attack_knowledge.json` describes 20 separate named/schematic coverage cases, their defenders' *possible* conditional reactions, 16 named coverage/man checks, and conditional offensive stresses. It distinguishes Cover 1 Hole vs Robber, Cover 2 Man vs zone vs Tampa 2, Cover 3 Match vs Sky, Quarters vs Palms vs Drop, and Cover 6 vs Cover 9. It's a **prior and explanation layer**, not a collection of Madden-confirmed assignments nor empirical play-beats-coverage rules.
2. **Observed outcomes**: `historical_defense_baseline.py` reads a snapshot of the user's Madden DB with SQLite `mode=ro`, isolates **our offense snaps** (opponents were on defense), normalizes recorded `coverage_seen` without inventing variants, and measures the user's verified executed calls and outcomes against each recorded coverage. Historical results are **associations**, not causes or certain future answers.

EA's Madden 27 coverage-check gameplay documentation: <https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay>.

## Source and time boundaries

- `snaps.side='offense'` is the correct side for studying what defensive calls CPU or human opponents used **against our offense**. `side='defense'` is instead our own defensive playcalling/outcome history, reserved for the next defensive-coordinator project.
- `coverage_seen` from the existing DB is a historical label; the schema does not prove which values were read directly from Madden's post-play display. **Do not claim 758 exact post-play verified calls.** New captures must store their own post-play evidence and snap IDs.
- If `coverage_seen='pressure'`, this only shows an unspecified pressure label, **not Cover 0**.
- `executed_verification='verified'` verifies **our executed offensive call**, not the opponent coverage.
- All pre-snap prediction features use earlier complete games only. The actual defensive label and play result are exclusively **post-play training targets**, never inputs to their own prediction.
- An `opponent_id='cpu'` is currently pooled across CPU teams; we **cannot** claim opponent-team-specific adaptation without preserving a unique CPU team identifier.

## Reproduce against a COPY of your Madden backup

```sh
python scripts/madden_ml/eval_historical_defense_baseline.py \
  --db /path/to/READ_ONLY_COPY_OF_madden27.db \
  --output-dir /tmp/cfb-defense-history
```

Outputs are `historical_defense_baseline_report.json`, `HISTORICAL_DEFENSE_BASELINE.md`, and `historical_defense_matchups.csv`. These contain aggregates only, no complete raw game history. Never write to the original database. No external packages needed for the evaluator.

## Results from the November 6 backup (not a live benchmark)

- 1,097 of our offensive snaps; 872 historical defensive labels, consisting of 758 named coverage labels and 114 pressure-only labels.
- 655 verified executions of *our* offensive plays; 440 have a named historical coverage and a usable outcome.
- 27 session IDs; 630 historical named coverage labels tested across 22 chronological held-out sessions (first five used for training).
- Historical overall-frequency baseline: **Brier 0.87761, top-choice 23.49%**; opponent/down-distance contextual model: **Brier 0.87975, top-choice 23.17%**. **The contextual model did not beat the simple baseline**, so there is no basis to promote it.
- Stricter held-out slice (440 verified own executions): overall-frequency Brier **0.89862**, contextual Brier **0.89701**; their top-choice accuracy was **19.32%** and **18.86%** respectively. No consistent win.
- 5 specific play-versus-coverage cells and 34 formation-versus-coverage cells have at least 5 samples. Some examples: Mtn PA Flood vs logged Cover 2 Man **7/7** and Mtn Stick Wheel vs logged Cover 2 Man **6/6** using a 40/60/100 down-distance success metric. These are tiny, observational, selected matchups with a different-opponent confounding problem. Do not treat them as proven automatic counters.

## Optional research integration with the offense structural model

`predict_coverage(history, pre_snap_situation)` returns a full probability distribution over named coverage cases; `offense_model_mix(prediction)` converts this to the existing `offense_play_features.scheme_distribution` contract. The latter now **preserves exact** Palms, Quarters, Cover 3 Match, Tampa 2 and split-field variants when named, instead of collapsing them into generic Cover 4 or Cover 2 buckets. Generic labels remain **generic**.

Use this optional probability distribution for **offline shadow evaluation only**. The learned offensive residual remains responsible for ranking eligible plays using the 91 verified-name stock route-art feature records. The knowledge dictionary must never force "Cover 2 -> deep post" without checking the actual called routes, predicted coverage probability, and post-snap defenders.

Nothing in this PR connects these experimental probabilities into the live on-screen selector. That is intentional until the 150ms latency, whole-game accuracy and execution-reconciliation gates pass.

## Follow-up work

1. Add a post-play vision/UI capture for Madden's revealed opponent call. The event needs: session+snap identifier; pre-snap prediction; revealed call string; screenshot/frame locator; human/automatic verification status; and any Madden adjustment information. Preserve the original historical labels separately.
2. Fit and evaluate a coverage predictor with richer **pre-snap** visual observations (front/box, CB leverage, safety rotations) using sequential games. Don't leak Madden's post-play display.
3. Evaluate the structural offense model vs PR #110 baseline on the verified executed calls; use whole-game splits and actual opponent coverage where known. Separate CPU and human results; don't pool all CPU franchises as one opponent.
4. For read progression, keep advice conditional on identified receivers. PR #110's route-art geometry does not identify all five receiver routes; it cannot truthfully say an exact WR is running a post unless player-level route evidence is present.
5. Keep custom-macro and individual-adjustment decisions **specific or `NO ADJUSTMENT`**, while verifying legality. No automatic merges, promotions or live coach modifications.

## Test scope

`tests/madden_ml/test_historical_defense_baseline.py` contains local baseline/SQL/provenance/leakage tests. `tests/madden_ml/test_coverage_exact_scheme_bridge.py` checks the full-repo offense feature interface; run this and the full suite in a normal developer checkout. The results here are a **baseline**, not proof of a better ML coordinator.