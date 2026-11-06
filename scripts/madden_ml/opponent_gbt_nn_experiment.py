"""Compare gradient-boosted trees and a small MLP with the pooled opponent prior.

Reads the ``opponent_run_pass`` feature rows written by ``ml heldout-eval``.
Only raw per-snap features are used: the pooled-prior probabilities are fold
outputs fit on other games (including, under leave-one-opponent-out, the test
game), so feeding them to a learner would leak labels across folds.
``chronological`` trains only on earlier games; ``leave_one_opponent_out``
trains on every other game (one game per opponent). Research only; needs
scikit-learn.

    python scripts/madden_ml/opponent_gbt_nn_experiment.py REPORT.json [--scheme leave_one_opponent_out]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FEATURES = ("down", "distance", "n_history", "p_individual_only")


def log_loss(ys, ps):
    ps = [min(1 - 1e-6, max(1e-6, p)) for p in ps]
    return -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for y, p in zip(ys, ps)) / len(ys)


def main(argv=None) -> int:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--scheme", default="leave_one_opponent_out")
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args(argv)
    rows = json.loads(Path(args.report).read_text())["opponent_run_pass"][args.scheme]["feature_rows"]
    rows = [r for r in rows if r.get("down") is not None and r.get("distance") is not None]
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([int(r["y"]) for r in rows])
    games = np.array([r["game"] for r in rows])
    order = list(dict.fromkeys(r["game"] for r in rows))
    out = {"gbt": np.full(len(y), np.nan), "mlp": np.full(len(y), np.nan)}
    for i, g in enumerate(order):
        test = games == g
        train = ~test if args.scheme != "chronological" else np.isin(games, order[:i])
        if train.sum() < 40 or len(set(y[train])) < 2:
            continue
        gbt = HistGradientBoostingClassifier(max_depth=2, max_iter=50, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=args.seed)
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=2000,
                                                            random_state=args.seed))
        out["gbt"][test] = gbt.fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        out["mlp"][test] = mlp.fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    scored = ~np.isnan(out["gbt"])
    ys = y[scored]
    pick = [r for r, s in zip(rows, scored) if s]
    result = {
        "scheme": args.scheme,
        "features": list(FEATURES),
        "n": int(scored.sum()),
        "games": len(set(games[scored])),
        "log_loss": {
            "coin_flip": log_loss(ys, [0.5] * len(ys)),
            "league_situation": log_loss(ys, [r["p_league_situation"] for r in pick]),
            "league+individual": log_loss(ys, [r["p_league+individual"] for r in pick]),
            "league+style+individual": log_loss(ys, [r["p_league+style+individual"] for r in pick]),
            "gbt": log_loss(ys, out["gbt"][scored]),
            "mlp": log_loss(ys, out["mlp"][scored]),
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
