# NFL XGBoost prior vs the Madden experimental model — held-out Madden games

Research only. Nothing in this document is connected to live play calling, and
no saved model, playbook or live database was changed.

Reproduce (research venv with `.[nfl-research]`, Madden DB opened read-only):

```
python scripts/madden_ml/nfl_prior_heldout_eval.py --db AUDIT_COPY.db \
    --nfl-run ~/.cfb-research/nfl/runs/run-20261106T144158Z --out ~/.cfb-research/madden-nfl-eval
```

## Setup

- **Data**: 655 verified offensive plays across 16 Madden games (572 CPU, 83 human),
  plus 192 human defensive snaps with the opponent's observed run/pass.
- **NFL models**: the XGBoost success (EPA > 0) and run/pass models from the NFL
  prototype (trained 2016–2022, early-stopped on 2023, never shown Madden data).
- **Folds**: whole games, rolling origin (train only on earlier games, at least
  3 earlier test games). 13 test games for offense; `exclude_opponent` folds only
  cover the 2 human games because every CPU game shares one opponent id.
- **Feature gap**: Madden snaps carry down, distance, yardline and usually
  quarter. Score, clock and timeouts are imputed at neutral values.
- **Label gap**: Madden `success` is the logged result; NFL success is EPA > 0.
  `nfl_xgb_calibrated` re-fits intercept and slope on each fold's training games.
- Intervals are 95% game-clustered bootstrap (1,000 resamples). Negative Δ = better.

## Offense success (chronological folds)

| subset | plays | experimental | NFL calibrated | experimental + NFL (logit average) | base rate |
|---|---|---|---|---|---|
| CPU | 516 | 0.6722 (AUC 0.568) | 0.6578 (AUC 0.610) | 0.6602 (AUC 0.604) | 0.6822 |
| human | 83 | 0.7033 (AUC 0.508) | 0.7089 (AUC 0.548) | 0.7010 (AUC 0.528) | 0.7026 |

Log loss vs the experimental model (CPU games):

- experimental + NFL: **−0.0120 [−0.0209, −0.0040]**, which is better and excludes zero.
- NFL calibrated alone: −0.0143 [−0.0317, +0.0006], which is better on average but not resolved.
- NFL raw (no recalibration): +0.0193 [−0.0083, +0.0494]. It ranks plays well (AUC 0.618)
  but its EPA-based base rate is wrong for Madden's label.
- NFL-rooted shrinkage table: worse than the base rate. Dropped.

Human games: no model beats the base rate; all intervals include zero (83 plays, 2 games).

### Play-category mapping

The NFL models know only run versus pass. `NFL_PLAY_INPUT` maps the Madden
family classifier onto that: run → run, pass → pass, screen → pass, and
unknown → a mixture weighted by the NFL run/pass probability for that
situation. Of 1,097 offensive rows, 185 map to run, 422 to pass and 490 to the
mixture. Concept, formation, play action and RPO reads do not transfer; any
play-specific knowledge has to come from the Madden model.

### Tuned prior strength (validation games only)

`experimental+nfl_tuned` scores
`logit p_experimental + w × (logit p_nfl_calibrated − logit training base rate)`,
with `w` in {0, 0.1, 0.25, 0.5, 1.0}, chosen on the latest quarter of each
fold's training games. The test game is never used. The chosen strength was
1.0 in 11 of 13 chronological folds and 0.5 in 2.

| subset | plays | experimental | exp + NFL (fixed average) | exp + NFL (tuned) | Δ tuned vs experimental |
|---|---|---|---|---|---|
| all | 599 | 0.6765 | 0.6658 | **0.6630** | −0.0136 [−0.0282, −0.0009] |
| CPU | 516 | 0.6722 | 0.6602 | **0.6564** | −0.0158 [−0.0321, −0.0002] |
| human | 83 | 0.7033 | 0.7010 | 0.7038 | +0.0004 [−0.0130, +0.0093] |

Slices (log loss, all games; experimental → tuned):

- Per-snap or legacy verification (91): 0.6735 → 0.6685. Standing-toggle attestation (508): 0.6770 → 0.6619. The gain holds in both, and is smaller where verification is strongest.
- Short yardage, distance ≤ 2 (65): 0.6305 → 0.6241, though the fixed average does best (0.6183). In CPU games alone the tuned blend is slightly worse than experimental (0.6324 vs 0.6296).
- Red zone (138): 0.6891 → 0.6828. Outside the red zone (461): 0.6727 → 0.6570.

Calibration: expected calibration error rises from 0.011 (experimental) to
0.041 (tuned) while log loss and AUC (0.564 → 0.605) improve. The blend
sharpens rankings but over-commits slightly, so any shadow run should record
calibration per confidence bin.

Third/fourth-down conversion (117 plays): NFL raw 0.6445 and calibrated 0.6425 vs
situation table 0.6514 and base rate 0.6980. The NFL scores beat the base rate, but
they are not distinguishable from the existing Madden situation table.

## Opponent run/pass (human defensive snaps)

Nothing reliably beats a coin flip (0.693). The raw NFL run/pass model ranks better
(AUC 0.62–0.64), but its log loss is 0.699–0.714, and after calibration its log loss is 0.706–0.710, no better
than the pooled Madden priors and no better than a coin flip.

## Decision

| question | decision |
|---|---|
| Promote any NFL model to live play calling | **No-go** (out of scope; offline evidence only) |
| Shadow-log `experimental + NFL` for CPU games next | **Go**, in shadow only, recording both scores, never the shown call; opt-in, off by default |
| Use the NFL prior for human games | **No-go**: no measurable gain on 83 plays / 2 games |
| NFL prior for opponent run/pass | **No-go**: no gain over a coin flip |
| NFL-rooted shrinkage table | **No-go**: worse than the base rate |

Before any shadow run:

1. Export the calibrated NFL score as a static down × distance × field × run/pass
   table, so the live process has no xgboost dependency and stays inside the
   150 ms budget.
2. Re-check on the next CPU games: the CPU gain comes from one opponent id, so
   opponent-level generalisation is untested.
3. Log score, clock and timeouts. These are imputed today.
