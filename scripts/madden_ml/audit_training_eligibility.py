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
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

# An editable install may point at a different checkout; audit with this one.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
            "verified_games": len({r.get("game_id") for r in verified if r.get("game_id")}),
            "verified_by_opponent_type": dict(
                Counter(str(r.get("opponent_type") or "unknown") for r in verified)
            ),
            "verified_by_basis": dict(Counter(
                r.get("execution_attestation_source") or "per_snap_or_legacy" for r in verified
            )),
        }
    snap_ids = Counter(r.get("snap_id") for r in rows if r.get("snap_id"))
    return {
        "database": str(source),
        "dataset_module": dataset.__file__,
        "logical_snap_count": len(rows),
        "duplicate_snap_ids": sum(n - 1 for n in snap_ids.values() if n > 1),
        "trainable_verified_offense": by_side["offense"]["verified_executed_and_labeled"],
        "trainable_verified_defense": by_side["defense"]["verified_executed_and_labeled"],
        "explanation": ("Verified executed play + known outcome = play-specific training. "
                        "A displayed call with uncertain execution does not qualify. "
                        "'standing_user_attestation' means the owner pre-committed to running "
                        "every displayed call; it was not confirmed snap by snap."),
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
