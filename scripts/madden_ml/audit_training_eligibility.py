"""Read-only training-data audit for the Madden coach.

Usage:
    python scripts/madden_ml/audit_training_eligibility.py --db ~/.cfb-coach/madden27.db

This tool does not retrain, change game logs, or activate any model.
Counts are deduplicated logical snaps from the same dataset builder used by training.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from cfb_coach.madden.model import dataset


def summarize(db_path: Path) -> dict:
    """Audit using the real training eligibility and normalization rules."""
    source = db_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Madden database not found: {source}")
    # URI mode=ro forbids writes to the uploaded / live source database.
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = dataset.build_rows(db=SimpleNamespace(conn=conn, path=source))
    finally:
        conn.close()

    by_side = {}
    for side in ("offense", "defense"):
        subset = [r for r in rows if r.get("side") == side]
        labels = Counter(str(r.get("success") or "unknown") for r in subset)
        verified = [r for r in subset if r.get("eligibility") == dataset.ELIGIBILITY_VERIFIED_EXECUTION]
        uncertain = [r for r in subset if r.get("eligibility") == dataset.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC]
        by_side[side] = {
            "recorded_logical_snaps": len(subset),
            "verified_executed_and_labeled": len(verified),
            "known_result_execution_uncertain": len(uncertain),
            "recommendation_only_or_unlabeled": len(subset) - len(verified) - len(uncertain),
            "success_labels": dict(labels),
            "verified_touchdowns": sum(
                1 for r in verified if str(r.get("result") or "").lower().strip() == "td"
            ),
        }
    return {
        "database": str(source),
        "logical_snap_count": len(rows),
        "trainable_verified_offense": by_side["offense"]["verified_executed_and_labeled"],
        "trainable_verified_defense": by_side["defense"]["verified_executed_and_labeled"],
        "explanation": ("Verified executed play + known outcome = play-specific training. "
                        "A displayed call with uncertain execution does not qualify."),
        "by_side": by_side,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.db), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
