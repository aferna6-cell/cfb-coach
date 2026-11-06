# Integrating research models into live play-calling — design (not implemented)

Status: design only. Nothing in this sprint connects a research model to the live caller.
The experimental-primary live mode and its heuristic fallback stay exactly as they are.

## What exists today

- **Live path.** `experimental_live` scores candidates with the experimental shrinkage model,
  `select_from_database` applies repeat penalties, `choose_joint_action` picks the play and
  adjustment, and `joint_override_guard.review_override` vetoes thin overrides. On timeout the
  heuristic pick is used.
- **Measured live reliability** (1,129 logged decisions): 1.2% heuristic fallback (all timeouts),
  model latency p50 20 ms, p95 96 ms, p99 153 ms against the fixed 150 ms budget.
- **Logging hooks.** Every `CoachingDecision` already carries `shadow_scores` / `shadow_pick`,
  which are logged but never used to choose. This is the natural first integration point.
- **Offline evidence** (`OFFLINE_EVAL_BASELINE.md`, `cfb_coach/nfl_prior`):
  - The experimental model's success forecasts beat the base rate only slightly, and only in
    CPU games.
  - Down/distance tables help conversion forecasts.
  - Opponent run/pass is not yet predictable from our data.
  - On 239k NFL plays, XGBoost wins every NFL task, ahead of the MLP and the table.

## Principles

1. A research model only reaches live play-calling through these stages, in order: offline →
   shadow → randomised comparison → gated promotion. Each stage has written acceptance criteria.
   Promotion is never automatic.
2. Live inference keeps the 150 ms decision budget. A new component gets a fixed sub-budget and is
   dropped for that snap (never waited on) when it overruns.
3. Training data rules are unchanged:
   - Only verified executions are credited to a play.
   - Standing-attestation rows stay identifiable.
   - External NFL data carries its own provenance (`external_nfl_prior`) and never enters Madden
     training rows.
4. Every live artifact records its model version, data fingerprint, config and seed. It can be
   switched off by one config flag, and the default is off.

## Stage 1 — shadow scoring

| Component | What runs live | Sub-budget | Output |
|---|---|---|---|
| NFL situational prior | Precomputed lookup table (JSON) exported from the winning offline model on a grid of down × distance × field × clock × score × run/pass | < 1 ms | `shadow_scores` entry `nfl_prior` |
| Opponent run/pass prior | `opponent_tendencies.estimate_run_pass` over this game's observed opponent plays | < 2 ms | defensive decision trace field `opponent_pass_p` with `confidence` |
| Future outcome model | Same table-export approach; no xgboost/torch import in the live process | < 2 ms | `shadow_scores` entry |

Why tables and not the models themselves:
- A single-row XGBoost call measured 4–15 ms p95 through pandas, and the MLP 1–2 ms.
- The live model already uses up to 96 ms at p95.
- Importing torch or xgboost into the live process adds seconds of start-up time and new failure
  modes.
- A table lookup is deterministic, versioned and trivially fast.

Shadow exit criteria:
- At least 4 games with zero shadow errors.
- Shadow p99 below its sub-budget.
- Total decision p99 not worse than before (paired by game).
- Shadow outputs reproducible from logged inputs.

## Stage 2 — randomised comparison

The evaluation found that only 6% of live picks matched the heuristic, and that no decision logged
a usable randomisation probability. So "does the candidate call better plays?" can only be answered
prospectively.

- **Unit:** randomise per drive (not per snap) between the current policy and the candidate policy.
  Log `behavior_propensity` = 0.5 and the arm on every decision.
- **Size:** about 1,570 offensive snaps per arm to detect a 5-point success-rate difference
  (`PRIORITY2_HELDOUT_EVAL_PLAN.md`). At about 45 snaps per game, that is roughly 35 games per arm,
  and needs a user decision.
- **Analysis:** game-clustered difference in success rate and EPA-like outcomes, with
  per-situation slices. Defensive arms use stop rate, explosives allowed and TDs allowed, because
  our defensive execution is not verified.
- **Stopping:** a fixed sample size, with no peeking-based early promotion. Stop early only for
  harm, when the lower confidence bound is worse by more than 5 points after 15 games per arm.

## Stage 3 — gated promotion

- Register the candidate through `registry.py` with its gate report.
- The candidate must pass `heldout_eval` on the latest games:
  - log loss no worse than the current model, with the interval excluding harm;
  - calibration (ECE) no worse by more than 0.01.
- It must also pass the randomised comparison above.
- The user explicitly installs it, and the previous artifact stays as a one-flag rollback.

## Component notes

### NFL prior → Madden

- **Domain mapping.**
  - Scale clock features by the configured Madden quarter length.
  - Map college-specific rules (clock stoppage on first downs, overtime) explicitly rather than
    silently.
  - Field position uses the same `yardline_100` convention.
- **Use.** Feed it as the top level of the experimental model's shrinkage hierarchy, as a prior
  with a small pseudo-count, so Madden evidence overrides it quickly. Do not use it as a direct
  score.
- **Prerequisite.** A `heldout_eval` variant `experimental+nfl_prior` must beat the plain
  experimental model on held-out Madden games before stage 1 starts.

### Opponent run/pass prior

Fix two existing issues first, behind tests, before any live use:
- **The verification gate.** `build_defense_profile` only counts an opponent's observed concept
  when our own defensive execution was verified. As a result, the 192 observed opponent plays in
  `snaps` are never used. Observed opponent behaviour should not depend on our execution status.
- **Double counting.** The profile's `run_share` sums the `tendencies` table and snap counts. Once
  the gate is fixed, the same play would be counted twice unless the two sources are deduplicated.

The current evidence does not support live use. Revisit when there are about 3 games per human
opponent, and when the pooled model includes clock and score, which the NFL slices show matter most
(two-minute and score-state slices).

## Open decisions for the user

1. Whether standing-attestation rows (508 of 655 verified plays) remain training-eligible, or are
   down-weighted.
2. Whether to run the per-drive randomised comparison, and for how many games.
3. Whether the NFL prior should be tried as the top shrinkage level in an offline variant next.
