#!/usr/bin/env python3
"""One command: audit your data, retrain the route-aware offense, evaluate on held-out games, replay shadow
decisions, run scenarios A-F through the real live loop, and print a GO / CONDITIONAL GO / NO-GO.

  python scripts/madden_ml/offense_route_aware_readiness.py --db /path/to/COPY_of_madden27.db --out ~/route_aware_readiness
  python scripts/madden_ml/offense_route_aware_readiness.py --selfcheck --out /tmp/selfcheck   # no DB: pipeline self-check ONLY

The database is opened read-only (work on a COPY anyway). Nothing is written inside ~/.cfb-coach and nothing is
enabled; `--install` only copies the artifact to the location the live coach looks in. Enabling is a separate,
explicit command:  python -m cfb_coach ml offense-ranker route_aware_preview
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.madden.model import offense_readiness as rd  # noqa: E402
from cfb_coach.madden.model import offense_route_ranker as rr  # noqa: E402
from cfb_coach.madden.model import offense_structural_eval as ev  # noqa: E402
from cfb_coach.madden.model import offense_structural_model as sm  # noqa: E402
from cfb_coach.madden.model import offense_synthetic as syn  # noqa: E402
from cfb_coach.madden.model import offense_time_correct as tc  # noqa: E402

TESTS = ["tests/madden_ml/test_route_aware_live_integration.py", "tests/madden_ml/test_offense_intel_model.py",
         "tests/madden_ml/test_offense_intel_assignments.py", "tests/madden_ml/test_offense_live_coverage.py",
         "tests/madden_ml/test_cpu_live_coverage_pilot_bridge.py", "tests/madden_ml/test_offense_route_art.py"]


def run_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], cwd=ROOT,
                       capture_output=True, text=True)
    m = re.search(r"(\d+) passed", r.stdout)
    f = re.search(r"(\d+) failed", r.stdout)
    return {"passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0, "failed_new": int(f.group(1)) if f else 0,
            "preexisting_failures": 0, "files": TESTS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="COPY of your Madden database (opened read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selfcheck", action="store_true", help="no DB: run the pipeline on planted synthetic data (NOT evidence)")
    ap.add_argument("--install", action="store_true", help="copy the artifact where the live coach looks (does not enable it)")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--min-train-games", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    live = (Path.home() / ".cfb-coach").resolve()
    if out == live or live in out.parents:
        print("refusing to write inside the live data directory", file=sys.stderr)
        return 2
    if not a.db and not a.selfcheck:
        print("no --db given. Pass a COPY of your game database, or --selfcheck for a pipeline self-check (not evidence).", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    real = bool(a.db)
    if real:
        data = rd.load_rows(a.db)
        source = f"real database copy: {a.db}"
    else:
        rows = syn.make_rows(truth="contradict", n_games=30, snaps=60, seed=21)
        data = he.EvalData(rows=rows, game_order=syn.game_order(rows), source="synthetic planted-signal self-check")
        source = "SYNTHETIC SELF-CHECK (planted signal): proves plumbing and learnability only"
    report: dict = {"version": rd.READINESS_VERSION, "source": source, "real_data": real}
    report["data_audit"] = rd.audit_data(data, a.db)
    print("training data audit done:", report["data_audit"]["usable_for_supervised_training"], "usable verified offense snaps")
    report["evaluation"] = rd.evaluate_models(data, n_boot=a.bootstrap, min_train_games=a.min_train_games)
    art = sm.train(data.rows)
    art_path = sm.save_artifact(art, out / "offense_route_aware.json")
    report["artifact"] = {"path": str(art_path), "digest": sm.artifact_digest(art), "trained_games": art.trained_games,
                          "verified_rows": art.n_train, "residual_active": art.residual_active, "notes": art.notes,
                          "feature_version": art.feature_version, "n_features": len(art.names), "valid": sm.validate_artifact(art) == []}
    report["shadow_replay"] = rd.shadow_replay(data, out_path=out / "shadow_replay.jsonl", min_train_games=a.min_train_games) \
        if report["evaluation"].get("folds") else None
    report["scenarios"] = rd.scenario_suite(art, out / "scenario_tmp")
    shutil.rmtree(out / "scenario_tmp", ignore_errors=True)
    report["tests"] = run_tests() if a.run_tests else None
    report["readiness"] = rd.gates(scenarios=report["scenarios"], shadow=report["shadow_replay"],
                                   evaluation=report["evaluation"], tests=report["tests"], real_data=real)
    if a.install and real and report["artifact"]["valid"]:
        dest = rr.default_artifact_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(art_path, dest)
        report["installed_to"] = str(dest)
    (out / "readiness_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    r = report["readiness"]
    print(f"\n=== {source}\nVERDICT: {r['verdict']}\n{r['reason']}")
    for k, v in r["gates"].items():
        print(f"  [{v['status']:>13}] {k}: {v['detail']}")
    print(f"\nreport: {out / 'readiness_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
