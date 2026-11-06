# Priority 2 plan: heuristic coach vs ML on held-out games

Status: plan only. Nothing here trains, activates, or promotes a model.
Facts below come from a read-only copy of the owner's 2026-11-06 database backup
(`madden27.backup.20261106T130346Z.db`).

## 1. Where Priority 1 left the data

| | Offense | Defense |
|---|---|---|
| Logged snaps (deduplicated) | 1,097 | 260 |
| Verified executed play + known result (play-specific training) | 655 (was 596) | 0 |
| Known result, execution uncertain (context only) | 442 | 260 |
| Verified games | 16 | 0 |
| Verified vs CPU / human | 572 / 83 | 0 / 0 |
| Verified per snap / by standing attestation | 147 / 508 | — |

* The +59 are verified touchdowns stored as `td` with no yards. No other row changed
  label or eligibility, no snap was added or lost, and there are 0 duplicate snap ids.
* All 83 human-opponent offensive rows are verified only by the standing attestation
  ("I always run the displayed call"), not by a per-snap confirmation.
* All 1,129 logged decisions were made in `experimental` mode (1,115 model picks,
  14 heuristic fallbacks after timeouts). 953 carry `propensity_method=deterministic`.
* Model latency: p50 20 ms, p95 96 ms, max 379 ms; 14 timeouts over the 150 ms budget.
* Only 632/1,357 snaps carry a `drive_id` (120 drives).

## 2. What can and cannot be measured from history

**Historical replay cannot prove an unchosen play would have worked.** The log
records the result of the play that was run. If the ML coach would have called Four
Verticals on a snap where Mesh was run for +3, nothing in the log says what Four
Verticals would have gained: the defense's call, the user's read, and the stick
skill all happened only once. Three further problems make naive replay misleading:

1. *Selection bias.* Each logged play was chosen because a coach thought it fit the
   situation. Plays that "look good" in replay were seldom tried in bad spots.
2. *No exploration record.* With deterministic picks there is no logged probability
   of choosing any alternative, so inverse-propensity (off-policy) estimates are
   undefined for most snaps. Estimators also need the alternative to have actually
   been called sometimes in similar situations ("overlap").
3. *Feedback.* Opponents (especially humans) react to what we call. A different
   call sequence changes what the defense does next.

So history supports three honest things: (a) **predictive** checks of an outcome
model on plays that were actually run, (b) **operational** checks (legal calls,
latency, fallbacks), and (c) descriptive results with uncertainty. Only a
**prospective controlled test** can show one coach is better.

## 3. Offline held-out evaluation (build first)

New module `cfb_coach/madden/model/policy_heldout_eval.py` plus CLI
`ml heldout-eval --db PATH --out REPORT_DIR`. Read-only (`mode=ro`), reuses
`dataset.build_rows`, `evaluate.py` grouping rules, `eval_scoring.decision_legal`,
`eval_scoring.wilson_interval`, and the `eval_harness` live path.

### Splits (leakage controls)

* Unit of split = **game** (`session_id`). Never split snaps of one game.
* **Rolling-origin, chronological**: for held-out game *g*, fit every learned
  component only on games that *started before* *g*. No future game informs a
  past decision. Report the first few folds as "cold start".
* **Strata reported separately**: CPU vs human; offense vs defense; per-snap vs
  standing-attestation verification (sensitivity analysis).
* **Pre-snap features only** at decision time. `coverage_seen` / `concept_seen`
  are post-snap (`schema.LEAKAGE_FIELDS`) and may only appear in training tables.
* **In-game memory** for snap *k* uses only snaps with `snap_seq < k` from the
  same game.
* **Never score an existing artifact on games it was trained on.** The installed
  `offense.json` (Nov 3) and `defense_v1/v2.json` saw most games. Retrain per fold
  from scratch, or restrict to games after the artifact's training cutoff.
* Hyperparameters (shrinkage, lambda) chosen **inside** the training fold only
  (pattern: `defense_logistic_train.choose_lambda`).
* Label rule frozen before looking at results: the 40/60/100% rule in
  `dataset._success_from_yards`; terminal kinds first.
* Duplicate snap identities across folds fail the run
  (`duplicate_snap_identity_across_groups`, already in `evaluate.py`).

### Metrics

For each held-out game and each policy (heuristic, ML candidate), with pre-snap state
reconstructed from the log:

| Metric | Definition | Valid as |
|---|---|---|
| Legal call rate | pick is in the book installed for that session (`ml_run_config.playbook_version`) and any macro is armed (`decision_legal`) | hard gate, must be 100% |
| Latency | p50/p95/max of the full decision path through `eval_harness` with a real clock; timeout rate over 150 ms | hard gate |
| Fallback rate | share of snaps where ML fell back to heuristic | operational |
| Outcome-model calibration | log loss, Brier, calibration bins for P(success), P(conversion on 3rd/4th), P(explosive), P(turnover) on **executed** verified plays | predictive (valid) |
| Agreement | share of snaps where each policy's pick equals the executed play | descriptive only |
| Matched-subset results | success, conversion, explosive (run ≥10, pass ≥20), turnover rate on snaps where the policy's pick = executed play, with Wilson 95% intervals and n | descriptive only, biased |
| Defense, game level | stop rate (offense failed by the same rule), TDs allowed per opponent drive, takeaways | descriptive; no play-level credit until defense is verified |

Report every number with its n and interval. With 83 human verified snaps across 2
opponents, human-stratum results are anecdotes; say so in the report.

### Acceptance for "ready for a live test" (not "better")

ML candidate: 100% legal; p95 ≤ 120 ms and ≤ 1% timeouts; outcome-model log loss
better than the base-rate model on held-out games in both strata; no fold where it
is materially worse (`gate()` thresholds in `defense_logistic_train` as template).

## 4. Prospective comparison (the only proof)

* Randomize the coach **per drive** (or per game) with a logged seed: heuristic vs
  ML. Record `policy_source`, the random assignment, and `behavior_propensity`
  (0.5) in `ml_decisions`. Keep the per-snap confirm tap for both arms.
* Compare per-drive outcomes: points per drive, success rate, 3rd/4th conversion,
  explosive and turnover rate; on defense, stops and TDs allowed per drive.
* Size: success rate has SD ≈ 0.5 per snap. Detecting a 5-point difference needs
  roughly 1,500 snaps per arm (~25 offensive games each). Plan for CPU first,
  report human games separately.
* Adding small, logged exploration (e.g. softmax over near-ties with recorded
  probability) makes later off-policy estimates possible.

## 5. Small NFL-data prototype (nflverse / nflreadpy), offline only

Goal: a real-football prior for down/distance/field-position play-type success
that can later be blended (capped) into Madden priors. Not connected to live
playcalling.

* **Location**: `cfb_coach/nfl_prior/` with optional dependency group
  `nfl = ["nflreadpy", "polars"]` in `pyproject.toml`; nothing in `cfb_coach.madden`
  imports it. Cache under `~/.cfb-coach/nfl_cache/` (parquet), never in the repo.
* **Data**: `nflreadpy.load_pbp([2019..2024])` for play type, down, `ydstogo`,
  `yardline_100`, `shotgun`, `no_huddle`, `pass_length`, `pass_location`, `run_gap`,
  `score_differential`, `half_seconds_remaining`, `epa`, `success`, `first_down`,
  `touchdown`, `interception`, `fumble_lost`. Optional:
  `load_participation` (formation, personnel, defenders in box, coverage where
  available) and `load_ftn_charting` (2022+). Respect nflverse licenses and
  attribution (FTN charting is CC-BY-SA).
* **Model**: a shrinkage table / small logistic model of P(success) and P(explosive)
  by situation bucket × play family (run inside/outside, short/deep pass, screen),
  using the **same 40/60/100% success rule and the same buckets** as the Madden
  dataset so the two are comparable. Explicitly *not* a play-by-play neural net.
* **Validation**: split by **season** (train 2019–2023, test 2024) and by game;
  report calibration and the size of situational effects (e.g. 3rd-and-long
  pass vs run).
* **Transfer check** (offline): compare NFL bucket rates with Madden verified
  offense buckets; report agreement and where Madden diverges (Madden mechanics
  differ). Output a report JSON; do not write any Madden artifact.
* **Exit criteria** before any future live use: a separate PR that adds a capped
  prior (like `MAX_RESEARCH_INFLUENCE`) behind a flag, evaluated with section 3.

## 6. Order of work

1. Merge PR #93 after owner review (labels, strict defense, candidate-only training).
2. Implement section 3 offline evaluator + report; run on the backup copy.
3. Add per-drive randomization + propensity logging (section 4) behind a flag.
4. NFL prototype (section 5) in parallel; offline report only.
