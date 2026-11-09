# unified_v1 validation report

**Scope.** Synthetic ground truth + this machine's in-process latency. No real Madden game log, owner database or champion artifact was available, so NOTHING below is evidence of real-game strength.

Python 3.13.18 on Linux-6.18.44-fc-v310-x86_64-with-glibc2.39

## 1. World model vs cold-start prior (planted matchup effects, whole-game expanding-window folds)

| games | rows | RMSE vs TRUE mean: prior | RMSE vs TRUE mean: learned | MAE yards prior | MAE learned | MAE diff (learned-prior) 95% game-bootstrap CI | Brier first-down prior | learned |
|---|---|---|---|---|---|---|---|---|
| 14 | 840 | 1.095 | 0.864 | 4.077 | 4.064 | -0.013 [-0.077, +0.054] | 0.2325 | 0.2203 |
| 30 | 1800 | 1.083 | 0.623 | 4.050 | 3.975 | -0.076 [-0.123, -0.025] | 0.2219 | 0.2102 |
| 60 | 3600 | 1.095 | 0.544 | 4.134 | 4.081 | -0.053 [-0.088, -0.020] | 0.2245 | 0.2127 |

Reading: with little data the learned model is no better than the prior (it correctly stays close to it); it overtakes the prior as games accumulate. MAE barely moves because outcome noise (sd 5 yd) dominates; the RMSE-vs-truth column is the sensitive one.

## 2. Bayesian opponent beliefs vs pooled-frequency baseline (planted sequential habit)

| habit | scored snaps | belief log-loss | frequency log-loss | belief better |
|---|---|---|---|---|
| 0.0 | 550 | 1.481 | 1.486 | True |
| 0.4 | 550 | 1.331 | 1.499 | True |
| 0.7 | 550 | 1.027 | 1.488 | True |

## 3. Live latency through the REAL continuous loop (`cmd_game`, 100 snaps per side, persisted store, cold reload audit)

Books = the PROPOSED 15-offense / 13-defense catalog books (324 / 297 plays); model = labelled cold-start prior. Latency = Enter on the situation -> card ready (parse + decision + persistence + render).

| run | snaps | p50 ms | p95 ms | max ms | timeouts >150ms | search-deadline fallbacks | audit |
|---|---|---|---|---|---|---|---|
| cold-start, proposed books | 200 | 24.2 | 40.8 | 55.8 | 0 | 0 | PASS |

Worst case (exact-level evidence on EVERY offensive play, pure-noise effects so nothing can be pruned) measured separately: p50 ~60 ms, p95 ~109 ms, max ~133 ms, occasionally hitting the 85% search deadline (flagged as a fallback). Real histories are far sparser. This is NOT the owner's laptop: re-run `local-finish` there.

## 4. Comparisons NOT run (and why)

- **Regularised boosted-tree challenger:** not run. scikit-learn/xgboost are not installed in this environment and, more importantly, there is no real labelled game data on which a tree-vs-hierarchical comparison would mean anything. Spec: compare 'when the data supports evaluation'.
- **Existing offense/defense models (experimental_model, defense_v1/v2):** not run head-to-head; their training/eval artifacts need the owner's database.
- **Calibrated expected points:** not fitted. `fit_state_value` refuses below 200 real possessions; the provisional objective is used and labelled uncalibrated.
