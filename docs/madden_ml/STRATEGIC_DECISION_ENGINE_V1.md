# Strategic Decision Engine v1 (defense first) — research results

Offline and shadow work only. Nothing here changes the live caller: the
experimental model stays primary, the heuristic fallback policy is unchanged,
and opponent models and NFL priors are not live overrides.

## What was built

| Piece | File | Role |
|---|---|---|
| Shared pre-snap context | `cfb_coach/madden/model/strategic_context.py` | One typed, side-aware picture of the snap. Every value is known / derived / inferred / missing with a source. Inconsistent inputs are reported and dropped, never clamped. Post-snap fields are listed in `POST_SNAP_FIELDS` and `assert_pre_snap` rejects them. Our own calls and the opponent's observed plays are kept in separate history fields. |
| Context audit | `scripts/madden_ml/context_feature_audit.py` | Which context fields the database actually has. |
| Opponent prediction comparison | `scripts/madden_ml/opponent_prediction_eval.py` | League baseline vs the existing partial-pooling estimator vs logistic regression, boosted trees and a small neural network; chronological and leave-one-opponent-out. |
| Defensive candidate assessment | `cfb_coach/madden/model/defense_strategy_eval.py` | Ranks legal defensive calls with a structured explanation. Reuses `light_box_allowed`, `call_family`, `COUNTERS`, `defense_success_delta` and `offense_success_probability`. Not read by the live caller. |
| Held-out check | `scripts/madden_ml/defense_strategy_heldout.py` | Threat calibration, legality and latency on logged human games. |
| Offense opportunities | `scripts/madden_ml/offense_opportunity_audit.py` | Descriptive rates with intervals. |

## Context actually available (human defensive snaps, n = 260)

- Always: down, distance, field position (yards to goal, side-aware), our call, result.
- Often: quarter (62%), score before the snap (46%; verified pre-snap — the score
  changes on the row after every touchdown, 75 of 75), the previous snap's observed
  play as typed in the situation text (52%, correctly marked "previous snap").
- Rarely: game clock (1%).
- Never: timeouts, offensive or defensive personnel, a pre-snap formation read of
  the opponent, a live pre-snap concept look on defense.
- Our decision records contain no pre-snap context block.

## Opponent data

Seven human opponents, **one game each** (30–45 defensive snaps per game, 192 with a
run/pass label, 190 with a concept family). One name given for this sprint does not
match a stored opponent spelling exactly and one stored opponent is not on the given
list; this was reported, not reconciled by inventing aliases, and no rows were changed.

Consequence: "individual" learning can only use earlier snaps of the same game, and
leave-one-opponent-out is the same as leave-one-game-out.

## Opponent prediction results (log loss, lower is better; 95% game-clustered intervals)

Run vs pass, chronological (deployable; 5 games, 134 plays):

| Model | Log loss | vs league situational table |
|---|---|---|
| coin flip (0.5) | 0.693 | −0.038 [−0.076, −0.001] |
| existing pooling + style | 0.719 | −0.012 [−0.036, +0.011] |
| existing pooling | 0.724 | −0.006 [−0.019, +0.009] |
| league situational table | 0.731 | — |
| boosted trees | 0.798 | +0.067 [−0.031, +0.137] |
| logistic regression | 2.122 | +1.39 (overconfident) |
| small neural network | 2.326 | +1.60 (overconfident) |

Leave-one-opponent-out (all 7 games): boosted trees 0.677 vs coin flip 0.693 vs
league table 0.705; the trees' gain over the table, −0.028 [−0.065, +0.003], is not
significant.

**Plain reading:** with seven one-game opponents, nothing predicts run vs pass better
than a coin flip in a deployable way. League situational rates are slightly *worse*
than 50/50 — these opponents differ from one another more than situations predict.
Gradient-boosted trees and the neural network overfit; they are not supported by
this much data.

Concept family (7 classes), the part that matters for coverage choice:

| Model | Chronological | Leave-one-out |
|---|---|---|
| league by down + this game's earlier plays (n/(n+12)) | **1.313** (−0.160 [−0.384, +0.032]) | **1.302** (−0.100 [−0.255, +0.027]) |
| league by down | 1.473 | 1.401 |
| boosted trees | 2.040 | 1.715 |
| small neural network | 5.595 | 2.677 |

The in-game blend is the best model in both schemes; the interval narrowly crosses zero.

## Defensive evaluator, held out on logged games

- Threat (opponent concept family) log loss: evaluator 1.325 vs situation prior 1.431,
  difference −0.106 [−0.192, −0.009]. Using this game's earlier plays is what helps:
  −0.138 [−0.255, −0.048] versus the same evaluator without game history.
- Top-1 family accuracy 48%.
- Installed book: 6 formations, 123 calls (121 scored by the defense model).
  On average 81% of calls are legal per snap (light boxes removed in the red zone,
  on short yardage and against run-leaning opponents).
- Latency for all 123 calls: p50 2.9 ms, p95 3.5 ms, max 7.3 ms (budget 150 ms).
- **Not evaluated:** whether a higher-ranked call leads to better outcomes. None of the
  260 human defensive snaps has a confirmed or attested execution, so which call was
  actually run is unknown. No defensive call can be called measured-better today.

Each candidate explanation lists components with their evidence class:
`measured` (none available yet), `model_unverified` (defense model trained on logged,
unconfirmed calls), `observed_opponent`, `prior` (coded counters) and `hypothesis`
(explosive exposure by shell). Priors weigh 0.35 and hypotheses 0.15 against 1.0 for
measured outcomes. Calls whose coverage is not recognized are ranked last and left
to the established caller. Many calls tie because the evaluator reasons at the level
of coverage family and front, not individual play design.

## Synthetic scenarios (behaviour checks only)

`tests/test_defense_strategy_eval.py` covers 1st & 10 vs run-heavy, 3rd & long vs
pass-heavy, goal line vs a frequent runner, red zone vs quick passing, two-minute
defense while leading, a tendency change within a game and no history. They check
direction and consistency, not performance.

## Offense opportunities (descriptive; executed, eligible plays only)

Human games have only 83 eligible executed offensive plays; most areas are too thin.
Against the CPU (572 plays):

- Against a pressure look, passes succeeded 78% (n 27) vs runs 30% (n 23).
- 3rd & 7+: passes converted 46% (n 39); runs rarely called.
- Red zone: pass touchdowns 37% (n 59) vs run 24% (n 51).
- Repeating the exact play within 3 snaps: 44% (n 9) vs 58% fresh — too few repeats to
  conclude anything; the current variety is already high (top play 5% of calls).
- Turnovers: passes 5.3% vs runs 0.6%; explosives: passes 21% vs runs 17%.

These are hypotheses for the next offensive experiment, not changes.

## Custom adjustment intelligence — plan (no live feature)

1. Keep three separate fields per snap: *recommended* adjustment, *executed* adjustment,
   and an explicit *no adjustment* confirmation. A missing value is "unknown", never
   "none".
2. The context object carries the recommended adjustment as a pre-snap input; the
   executed adjustment is post-snap (it belongs in `POST_SNAP_FIELDS` and is already listed).
3. Only adjustments whose execution is confirmed may become training data; recommended
   ones are logged for evaluation only.
4. Unverified adjustments are never activated by the engine; they may be shown as
   suggestions with their evidence class.
5. Current data: about 29 executed adjustments and 480 snaps with an explicit "no
   adjustment" confirmation (373 explicitly not confirmed) — enough to start measuring
   compliance, not effect.

## Missing information that limits the engine

1. Confirmed defensive executions (the single biggest gap): without them no defensive
   call can be measured.
2. More than one game per human opponent.
3. Pre-snap offensive formation / personnel of the opponent on defensive snaps.
4. Game clock and timeouts on most snaps.
5. A pre-snap context block written into each decision record.
