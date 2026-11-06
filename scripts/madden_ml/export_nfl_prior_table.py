"""Export the calibrated NFL prior as a static table for the opt-in live shadow.

Research venv only (xgboost, pandas). Reads the Madden database read-only,
writes outside the live data directory, and activates nothing: the live
process uses the table only if ``CFB_COACH_NFL_SHADOW`` points at it.

Cells are down x distance bin x yards-to-goal bin; each holds the raw NFL
success probability for a run and a pass and the NFL pass rate. Calibration
(Platt intercept/slope) and the base rate are fit on every verified Madden
offensive play. The prior strength defaults to the value validation chose
most often in the held-out comparison.

    python scripts/madden_ml/export_nfl_prior_table.py --db AUDIT_COPY.db \
        --nfl-run ~/.cfb-research/nfl/runs/RUN --out ~/.cfb-research/nfl-shadow
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.madden.model import nfl_prior_shadow as shadow  # noqa: E402
from cfb_coach.nfl_prior.data import OUTCOME_FEATURES, SITUATION_FEATURES  # noqa: E402

DISTANCE_EDGES = [1, 2, 3, 6, 10, 15]
DISTANCE_REPS = [1, 2, 3, 5, 8, 12, 20]
YARDLINE_EDGES = [5, 10, 20, 40, 60, 80]
YARDLINE_REPS = [3, 8, 15, 30, 50, 70, 88]


def _eval_module():
    spec = importlib.util.spec_from_file_location("nfl_prior_heldout_eval", ROOT / "scripts" / "madden_ml" / "nfl_prior_heldout_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_table(run_dir: Path, db: str, prior_strength: float) -> dict:
    import numpy as np

    ev = _eval_module()
    succ = ev.load_xgb(run_dir / "models" / "success" / "xgboost.json")
    rp = ev.load_xgb(run_dir / "models" / "run_pass" / "xgboost.json")
    keys, rows = [], []
    for down in (1, 2, 3, 4):
        for di, dist in enumerate(DISTANCE_REPS):
            for yi, ytg in enumerate(YARDLINE_REPS):
                keys.append(f"{down}|{di}|{yi}")
                rows.append({"side": "offense", "down": down, "distance": min(dist, ytg), "yardline": 100 - ytg})
    p_pass = ev._predict(rp, rows, SITUATION_FEATURES, None)
    p_run = ev._predict(succ, rows, OUTCOME_FEATURES, np.zeros(len(rows)))
    p_throw = ev._predict(succ, rows, OUTCOME_FEATURES, np.ones(len(rows)))
    cells = {k: {"run": round(float(r), 5), "pass": round(float(t), 5), "p_pass": round(float(p), 5)}
             for k, r, t, p in zip(keys, p_run, p_throw, p_pass)}

    data = he.load_eval_data(db)
    madden = [dict(r) for r in data.rows]
    ev.attach_nfl_scores(madden, run_dir)
    train = [(r, y) for r in madden if he.is_verified_offense(r) for y in [he.offense_label(r, "success")] if y is not None]
    a, b = ev.fit_platt([ev._logit(r["_nfl_success"]) for r, _ in train], [y for _, y in train])
    base = (sum(y for _, y in train) + 1) / (len(train) + 2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    table = {
        "schema": shadow.TABLE_SCHEMA,
        "model_version": f"nfl-prior-shadow.{run_dir.name}.{stamp}",
        "nfl_run": str(run_dir),
        "created": stamp,
        "distance_edges": DISTANCE_EDGES,
        "yardline_edges": YARDLINE_EDGES,
        "cells": cells,
        "platt": [a, b],
        "base_rate": base,
        "prior_strength": prior_strength,
        "calibration_rows": len(train),
        "scope_note": "Validated offline on CPU games only; human games showed no gain.",
    }
    shadow.validate_table(table)
    diffs = []
    for r, _ in train:
        hit = shadow.nfl_probability(table, down=r.get("down"), distance=r.get("distance"),
                                     yardline=r.get("yardline"), play=r.get("action_play"))
        if hit is not None:
            direct = ev._sigmoid(a + b * ev._logit(r["_nfl_success"]))
            diffs.append(abs(hit[0] - direct))
    diffs.sort()
    table["fidelity_vs_direct_model"] = {
        "rows": len(diffs), "covered_share": round(len(diffs) / max(1, len(train)), 4),
        "mean_abs_diff": round(sum(diffs) / max(1, len(diffs)), 4),
        "p95_abs_diff": round(diffs[int(0.95 * (len(diffs) - 1))], 4) if diffs else None,
    }
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--nfl-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior-strength", type=float, default=1.0)
    args = ap.parse_args(argv)

    from cfb_coach.games import data_dir

    out_root = Path(args.out).expanduser().resolve()
    live = Path(data_dir()).expanduser().resolve()
    if out_root == live or live in out_root.parents:
        raise SystemExit(f"refusing to write inside the live data directory {live}")
    table = build_table(Path(args.nfl_run).expanduser().resolve(), str(Path(args.db).expanduser().resolve()),
                        args.prior_strength)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{table['model_version']}.json"
    path.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(path)
    print(json.dumps({k: table[k] for k in ("platt", "base_rate", "prior_strength", "calibration_rows",
                                            "fidelity_vs_direct_model")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
