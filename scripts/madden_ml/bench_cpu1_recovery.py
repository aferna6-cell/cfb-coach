#!/usr/bin/env python3
"""Read-only-source loaded CPU pilot benchmark. NEVER writes the owner's DB.

Usage:
  python3 scripts/madden_ml/bench_cpu1_recovery.py --db ~/.cfb-coach/madden27.db --samples 100
  python3 scripts/madden_ml/bench_cpu1_recovery.py --db ~/.cfb-coach/madden27.db --samples 100 --with-macros

A temporary SQLite backup isolates all test writes; the active ML configuration
and artifacts are read from that copy. Prints source attribution and latency.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sqlite3
import statistics
import tempfile
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--with-macros", action="store_true")
    args = p.parse_args()
    if args.samples < 5:
        p.error("--samples must be at least five")
    source = args.db.expanduser().resolve()
    if not source.is_file():
        p.error(f"database not found: {source}")

    # Consistent SQLite backup; never open original for writes.
    with tempfile.TemporaryDirectory(prefix="cfb_cpu1_bench_") as directory:
        root = Path(directory)
        copied = root / "game_copy.db"
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
            with sqlite3.connect(copied) as dst:
                src.backup(dst)
        # Keep all generated shadow logs outside the owner's ~/.cfb-coach.
        os.environ["HOME"] = str(root)
        os.environ["CFB_COACH_MADDEN_DB"] = str(copied)

        from cfb_coach.db import CoachDB
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.situation import Situation
        from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
        from types import SimpleNamespace

        cases = [
            ("1st and 10", 1, 10, 35),
            ("2nd and 7", 2, 7, 42),
            ("3rd and 12", 3, 12, 47),
            ("3rd and 2", 3, 2, 17),
            ("2nd and 18", 2, 18, 60),
        ]
        rows = []
        db = CoachDB(copied)
        try:
            for i in range(args.samples + 2):
                raw, down, distance, yardline = cases[i % len(cases)]
                sit = Situation(
                    raw=raw, side="offense",
                    down=down, distance=distance, yardline=yardline,
                )
                t0 = time.perf_counter()
                call = make_call(
                    sit, "cpu", db=db,
                    rng=random.Random(1300 + i),
                    live_macros=args.with_macros,
                )
                wall = (time.perf_counter() - t0) * 1000.0
                validate_live_call(
                    db, SimpleNamespace(formation=call.formation, play=call.play),
                    opponent_id="cpu",
                )
                info = getattr(call, "ml_experimental", None) or {}
                if i >= 2:
                    rows.append({
                        "wall_ms": round(wall, 3),
                        "model_ms": info.get("latency_ms"),
                        "source": info.get("fallback_source") or (
                            "full_model" if info.get("fell_back") is False else "unattributed"
                        ),
                        "reason": info.get("fallback_reason"),
                        "play": [call.formation, call.play],
                        "macro": call.macro,
                    })
        finally:
            db.close()

    numeric = sorted(
        float(x["model_ms"]) for x in rows if x["model_ms"] is not None
    )
    p95 = numeric[min(len(numeric) - 1, max(0, int(0.95 * len(numeric))))] if numeric else None
    summary = {
        "source_database": str(source),
        "samples": len(rows),
        "with_macros": args.with_macros,
        "model_p50_ms": statistics.median(numeric) if numeric else None,
        "model_p95_ms": p95,
        "model_max_ms": max(numeric) if numeric else None,
        "budget_ms": 150,
        "full_model": sum(x["source"] == "full_model" for x in rows),
        "previous_experimental_fallback": sum(
            x["source"] == "previous_experimental_model" for x in rows
        ),
        "heuristic_or_unattributed": sum(
            x["source"] not in ("full_model", "previous_experimental_model") for x in rows
        ),
        "timeout_count": sum(x["reason"] == "timeout" for x in rows),
        "macro_recommendations": sum(bool(x["macro"]) for x in rows),
        "all_calls_installed_legal": True,
        "note": "Benchmark simulated pre-snap calls on a temporary database; not a Madden game.",
    }
    print(json.dumps(summary, indent=2))
    # A single legal model fallback does not invalidate the experiment.
    # Recurrent timeout or any heuristic downgrade is a hard NO-GO.
    return 0 if (
        summary["full_model"] >= 0.95 * len(rows)
        and summary["heuristic_or_unattributed"] == 0
        and summary["timeout_count"] <= max(1, len(rows) // 100)
        and p95 is not None and p95 < 150
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
