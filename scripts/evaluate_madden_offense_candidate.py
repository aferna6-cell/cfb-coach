#!/usr/bin/env python3
"""Leakage-aware offline CPU offense candidate training and calibration check.

Read-only SQLite DB. Never installs, activates or overwrites the live model.
Training excludes the latest session for evaluation. Report is on the plays
actually executed, NOT predicted wins from counterfactual alternative calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import dataset, experimental_model

LATEST = "3d9b477c10df4827"


def metrics(artifact, examples):
    brier = 0.0
    loss = 0.0
    calibrated = 0.0
    for r in examples:
        pred = experimental_model.predict_success(
            artifact, formation=r["executed_formation"], play=r["executed_play"],
            down=r.get("down"), distance=r.get("distance"),
            yardline=r.get("yardline"), opponent_id=r.get("opponent_id"),
            opponent_type=r.get("opponent_type"), coverage_hint=None,
            coverage_source="none",
        )
        p = max(1e-5, min(1-1e-5, float(pred["probability"])))
        y = 1 if r.get("success") == "true" else 0
        brier += (p-y)**2
        loss -= y*math.log(p) + (1-y)*math.log(1-p)
        calibrated += p
    n = max(1, len(examples))
    return {
        "n": len(examples),
        "brier": round(brier/n, 5),
        "log_loss": round(loss/n, 5),
        "mean_predicted_success": round(calibrated/n, 5),
        "actual_success_rate": round(
            sum(r.get("success") == "true" for r in examples)/n, 5
        ),
    }


def _timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def main(db_path: Path, out_dir: Path, session: str = LATEST) -> None:
    db = CoachDB.open_read_only(db_path)
    try:
        sessions = {str(r["session_id"]): r["started_ts"]
                    for r in db.conn.execute(
                        "SELECT session_id, started_ts FROM game_sessions"
                    )}
        hold_start = _timestamp(sessions.get(session))
        if not hold_start:
            raise SystemExit("Requested holdout session or timestamp is missing")
        rows = dataset.build_rows(db=db)
        eligible = [r for r in dataset.supervised_rows(rows)
                    if str(r.get("side") or "").startswith("o")
                    and r.get("success") in ("true", "false")
                    and r.get("executed_formation") and r.get("executed_play")
                    and r.get("session_id") in sessions]
        training = [r for r in eligible if
                    _timestamp(sessions.get(r["session_id"])) is not None
                    and _timestamp(sessions.get(r["session_id"])) < hold_start]
        test = [r for r in eligible if r.get("session_id") == session]
        if len(training) < 30 or len(test) < 15:
            raise SystemExit(
                f"Insufficient actual verified rows: train={len(training)}, test={len(test)}"
            )

        before = experimental_model.train_experimental(training, side="offense", seed=7)
        baseline_success = sum(r["success"] == "true" for r in training)/len(training)
        baseline_p = max(1e-5, min(1-1e-5, baseline_success))
        baseline_brier = sum((baseline_p - (r["success"]=="true"))**2
                             for r in test)/len(test)
        report = {
            "evaluation": "chronological_session_holdout_verified_execution_only",
            "holdout_session": session,
            "training_rows": len(training),
            "test_rows": len(test),
            "no_current_clock_or_showing": True,
            "no_future_labels_in_fitting": True,
            "retrained_on_history_only": {
                "model_version": before.model_version,
                **metrics(before, test),
            },
            "constant_training_rate_baseline_brier": round(baseline_brier, 5),
            "installed_model": None,
            "promotion": {
                "automatic": False,
                "recommendation": "not established by one small observational CPU holdout",
            },
            "caveat": (
                "Only scored observed executed plays; selection-policy changes "
                "cannot be assigned hypothetical yardage, touchdowns or win rate."
            ),
        }

        configured = db.get_meta("ml_experimental_artifact")
        path = Path(configured).expanduser() if configured else None
        if path and path.exists():
            incumbent = experimental_model.load_artifact(path)
            trained = _timestamp(getattr(incumbent, "trained_at", None))
            clean = trained is not None and trained < hold_start
            report["installed_model"] = {
                "path": str(path), "version": incumbent.model_version,
                "pre_holdout_training_date": clean,
                **metrics(incumbent, test),
            }
            if not clean:
                report["installed_model"]["warning"] = (
                    "Artifact training date may include holdout observations. "
                    "Do not use this comparison for a promotion decision."
                )
        else:
            report["installed_model"] = {
                "available": False,
                "reason": "Installed artifact is on user's computer, not in DB backup",
            }

        # Build a future candidate with all eligible historical data.
        # This file is *not* installed and does not change ml_mode.
        next_candidate = experimental_model.train_experimental(eligible, side="offense", seed=7)
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = out_dir / "offense_candidate_NOT_INSTALLED.json"
        if candidate_path.exists():
            raise SystemExit(f"Refusing to overwrite existing candidate: {candidate_path}")
        experimental_model.save_artifact(next_candidate, candidate_path)
        report["candidate_for_future"] = {
            "path": str(candidate_path),
            "version": next_candidate.model_version,
            "n_supervised": next_candidate.n_supervised,
            "installed": False,
            "auto_promoted": False,
        }
        report_path = out_dir / "madden_offense_holdout_report.json"
        if report_path.exists():
            raise SystemExit(f"Refusing to overwrite existing report: {report_path}")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--session", default=LATEST)
    a = p.parse_args()
    main(a.db.expanduser(), a.out_dir.expanduser(), a.session)
