#!/usr/bin/env python3
"""Train a NEW OFFLINE candidate from a verified Madden database, never activate it.

Creates a consistent isolated SQLite backup, runs existing verified-data model
training and a separate game-held-out evaluation, and records the outcome. This
is a postgame CANDIDATE workflow, not automatic promotion, and the held-out
evaluation measures an experimental refit—not the saved candidate's causal lift.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True, help="Original Madden DB (opened read-only)")
    ap.add_argument("--out", type=Path, required=True, help="NEW review folder, outside the live data directory")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--min-train-games", type=int, default=3)
    args = ap.parse_args()

    src = args.db.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not src.is_file():
        ap.error(f"database not found: {src}")
    live_dir = src.parent
    if out == live_dir or live_dir in out.parents:
        ap.error("Output must not be inside the live database directory")
    if out.exists():
        ap.error(f"Output already exists (refuse overwrite): {out}")
    if args.bootstrap < 20:
        ap.error("--bootstrap must be >= 20")
    out.mkdir(parents=True)
    copy = out / "training_snapshot.db"
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as base:
        with sqlite3.connect(copy) as target:
            base.backup(target)
    sha = _digest(copy)
    candidate = out / "candidate_experimental_offense.json"
    heldout = out / "heldout"
    commands = [
        [
            sys.executable, "-m", "cfb_coach", "ml", "train-experimental",
            "--side", "offense", "--db", str(copy), "--out", str(candidate),
        ],
        [
            sys.executable, "-m", "cfb_coach", "ml", "heldout-eval",
            "--db", str(copy), "--out", str(heldout),
            "--min-train-games", str(args.min_train_games),
            "--bootstrap", str(args.bootstrap),
        ],
    ]
    log = []
    failure = None
    # heldout-eval deliberately refuses output inside data_dir(). Put the
    # isolated application data dir outside the output folder while pointing
    # Madden DB explicitly at the consistent, never-live SQLite snapshot.
    with tempfile.TemporaryDirectory(prefix="cfb_cpu2_train_") as scratch:
        env = dict(os.environ, PYTHONPATH=str(ROOT), CFB_COACH_MADDEN_DB=str(copy),
                   CFB_COACH_DB=str(Path(scratch) / "cfb.db"))
        for cmd in commands:
            result = subprocess.run(
                cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                check=False,
            )
            entry = {
                "command": cmd, "exit_code": result.returncode,
                "stdout": result.stdout[-24000:],
                "stderr": result.stderr[-12000:],
            }
            log.append(entry)
            if result.returncode:
                failure = cmd
                break
    report = {
        "schema": "cfb.cpu2.postgame_candidate.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_db": str(src),
        "read_only_source": True,
        "snapshot_sha256": sha,
        "candidate": str(candidate) if candidate.is_file() else None,
        "heldout_report_dir": str(heldout) if heldout.is_dir() else None,
        "candidate_promoted": False,
        "active_model_modified": False,
        "requires_owner_review_and_explicit_install": True,
        "notes": [
            "Training data must be audited for verified execution and outcome provenance.",
            "This held-out refit does not automatically prove candidate superiority to champion.",
            "An incomplete candidate is not safe for live activation.",
            "Never learn effects of unconfirmed recommended hot routes.",
            "Results should be compared to the saved live champion, calibration, and runtime budget.",
        ],
        "steps": log,
        "failure": failure,
    }
    (out / "CPU2_CANDIDATE_TRAINING_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))
    return 1 if failure or not candidate.is_file() else 0


if __name__ == "__main__":
    raise SystemExit(main())
