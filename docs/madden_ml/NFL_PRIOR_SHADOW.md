# NFL prior: opt-in shadow logging

Shadow only. The live recommended play, its ranking, the logged decision
and the active playcalling mode are unchanged. Nothing is turned on by
default.

## Why a shadow and not a switch

On held-out Madden games (`NFL_PRIOR_MADDEN_HELDOUT.md`), adding the
calibrated NFL situational prior to the experimental model improved CPU-game
success predictions (log loss −0.016 [−0.032, −0.0002]). It showed no gain on
human games (83 plays), and calibration got slightly worse. That is enough
to justify watching it live, but not enough to let it pick plays.

## Pieces

| Piece | Where | Notes |
|---|---|---|
| Table export | `scripts/madden_ml/export_nfl_prior_table.py` | Research venv. Bakes the XGBoost scores into a down × distance × yards-to-goal JSON table with Platt calibration and the base rate fit on verified Madden plays. Writes outside the live data directory and refuses to write inside it. |
| Live lookup | `cfb_coach/madden/model/nfl_prior_shadow.py` | Pure Python with no numpy or xgboost. It re-scores the model's top 8 ranked candidates with the prior blended in. |
| Hook | `experimental_live.apply_experimental_offense` | Runs after the call is final, inside its own `try`, and stores only a short status in `ml_experimental["nfl_prior_shadow"]`. |

## Turning it on and off

- On: `CFB_COACH_NFL_SHADOW=/path/to/nfl-prior-shadow.<run>.<stamp>.json`
- Log path (optional): `CFB_COACH_NFL_SHADOW_LOG=/path/to/log.jsonl`. It defaults to a file next to the table.
- Off: unset the variable. Normal playcalling never reads the table.

## What each log line holds

The log records:

- `model_version` and prior strength
- the situation: down, distance, yardline and yards to goal
- opponent type, and `in_validated_scope` (CPU only)
- the live pick, plus its model, NFL and blended probabilities
- the shadow's top candidate, and whether it would differ from the live pick
- `confidence`, the blended distance from 0.5
- every scored candidate
- `latency_ms`

Preview calls without a snap id can log more than once per snap, so
analysis should deduplicate on session and snap sequence.

## Safety checks (tests in `tests/madden_ml/test_nfl_prior_shadow.py`)

- Off by default: no table read and no file written.
- The live pick is identical with the shadow off, on, or raising an exception, and none of those cases falls back to the heuristic.
- A broken table yields an `error` status, not an exception.
- When less than 5 ms of the decision budget remains, the shadow skips itself.

Measured on the exported table: warm lookup 0.18 ms median and 0.29 ms
p95; the one-time cold load (reading JSON and importing) is about 13 ms. The
table differs from the full model by 0.039 mean (0.11 p95) on verified plays,
because its bins are coarse and quarter is fixed at 2.

## What would justify more than a shadow

1. At least two more CPU games logged with the shadow on, compared on verified plays only.
2. Calibration checked per confidence bin.
3. Human-game evidence. None exists yet, so the prior stays out of human games.

There is no live randomized comparison and no automatic promotion.
