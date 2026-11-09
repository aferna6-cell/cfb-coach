#!/usr/bin/env python3
"""Reproducible validation report for unified_v1 (SYNTHETIC ground truth + loaded-latency on this machine).

    PYTHONPATH=. python3 scripts/madden_ml/unified_validation_report.py --out docs/madden_ml/UNIFIED_V1_VALIDATION_REPORT.md

Everything measured here validates the LEARNING MACHINERY against a planted truth. It makes no claim about real
Madden football strength: no real game log was available to this build.
"""
from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
from pathlib import Path

from cfb_coach.madden.unified import postgame as PG, preflight as PF, sim
from cfb_coach.madden.unified.types import DEF, OFF
from cfb_coach.madden.unified.world_model import WorldModel
from tests.unified.helpers import coordinator


def truth_rmse(rows, make):
    se = n = 0
    order = sorted({r["game_order"] for r in rows})
    for g in order[2:]:
        m = make([r for r in rows if r["game_order"] < g])
        for r in [r for r in rows if r["game_order"] == g]:
            st, p, d, o = PG._row_objs(r)
            t = sim.true_mean(p.concept, d.family, 4.0 if p.play_type == "run" else 6.0)
            se += (m.predict(st, p, d).e_yards - t) ** 2
            n += 1
    return math.sqrt(se / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = ["# unified_v1 validation report", "",
         "**Scope.** Synthetic ground truth + this machine's in-process latency. No real Madden game log, owner database or "
         "champion artifact was available, so NOTHING below is evidence of real-game strength.", "",
         f"Python {platform.python_version()} on {platform.platform()}", "",
         "## 1. World model vs cold-start prior (planted matchup effects, whole-game expanding-window folds)", "",
         "| games | rows | RMSE vs TRUE mean: prior | RMSE vs TRUE mean: learned | MAE yards prior | MAE learned | MAE diff (learned-prior) 95% game-bootstrap CI | Brier first-down prior | learned |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ng, seed in ((14, 1), (30, 2), (60, 3)):
        rows, _ = PG.validate_labels(sim.gen_games(ng, 60, seed=seed))
        cold = PG.eval_world_model(rows, lambda tr: WorldModel())
        ch = PG.eval_world_model(rows, PG.fit_on)
        ci = PG.bootstrap_diff(ch["_per_game_abs"], cold["_per_game_abs"])
        L.append(f"| {ng} | {len(rows)} | {truth_rmse(rows, lambda tr: WorldModel()):.3f} | {truth_rmse(rows, PG.fit_on):.3f} | {cold['mae_yards']:.3f} | "
                 f"{ch['mae_yards']:.3f} | {ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | {cold['brier_first_down']:.4f} | {ch['brier_first_down']:.4f} |")
    L += ["", "Reading: with little data the learned model is no better than the prior (it correctly stays close to it); "
          "it overtakes the prior as games accumulate. MAE barely moves because outcome noise (sd 5 yd) dominates; the RMSE-vs-truth column is the sensitive one.", "",
          "## 2. Bayesian opponent beliefs vs pooled-frequency baseline (planted sequential habit)", "",
          "| habit | scored snaps | belief log-loss | frequency log-loss | belief better |", "|---|---|---|---|---|"]
    for habit in (0.0, 0.4, 0.7):
        r = PG.eval_opponent(sim.gen_games(12, 50, seed=5, habit=habit), DEF)
        L.append(f"| {habit} | {r['n']} | {r['belief_logloss']:.3f} | {r['frequency_logloss']:.3f} | {r['belief_better']} |")
    L += ["", "## 3. Live latency (100 full cycles per side, in-process, repo snapshot books)", "",
          "| side | formations | candidates | p50 ms | p95 ms | max ms | timeouts >150ms | fallbacks | card/action mismatches |", "|---|---|---|---|---|---|---|---|---|"]
    for side in (OFF, DEF):
        co = coordinator(side)
        r = PF.run_cycle(co, game_id="val", n=100)
        L.append(f"| {side} | {len(co.inventory)} | {len(co.legal_candidates())} | {r['latency_ms']['p50']:.1f} | {r['latency_ms']['p95']:.1f} | "
                 f"{r['latency_ms']['max']:.1f} | {r['timeouts']} | {r['fallbacks']} | {r['card_action_mismatches']} |")
    L += ["", "This is NOT the owner's loaded UI/DB latency; re-run `preflight` on that machine.", "",
          "## 4. Comparisons NOT run (and why)", "",
          "- **Regularised boosted-tree challenger:** not run. scikit-learn/xgboost are not installed in this environment and, more importantly, "
          "there is no real labelled game data on which a tree-vs-hierarchical comparison would mean anything. Spec: compare 'when the data supports evaluation'.",
          "- **Existing offense/defense models (experimental_model, defense_v1/v2):** not run head-to-head; their training/eval artifacts need the owner's database.",
          "- **Calibrated expected points:** not fitted. `fit_state_value` refuses below 200 real possessions; the provisional objective is used and labelled uncalibrated."]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
