"""Validate the daily AI research files before they're committed.

    python scripts/validate_ai_research.py            # research/cfb27.json + research/madden27.json
    python scripts/validate_ai_research.py path.json  # one file

Exits 1 on any schema problem, a future timestamp, or an opponent id the coach doesn't know.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfb_coach.ai_research import GAMES, age_hours, repo_path, validate  # noqa: E402


def _known_opponents() -> set[str]:
    """League opponent ids (CFB and Madden share the league's people)."""
    try:
        seed = json.loads((ROOT / "cfb_coach" / "data" / "seed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set((seed.get("opponents") or {}).keys())


def check(path: Path) -> list[str]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"cannot read JSON: {exc}"]
    game = doc.get("game") if isinstance(doc, dict) else None
    errs = validate(doc, game if game in GAMES else None)
    if errs:
        return errs
    age = age_hours(doc, datetime.now(timezone.utc) + timedelta(minutes=10))
    if age is not None and age < 0:
        errs.append("researched_at is in the future")
    known = _known_opponents()
    unknown = sorted(set(doc.get("opponents") or {}) - known)
    if known and unknown:
        errs.append(f"unknown opponent ids {unknown} (known: {sorted(known)})")
    return errs


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [repo_path(g) for g in GAMES]
    bad = 0
    for p in paths:
        errs = check(p)
        if errs:
            bad += 1
            print(f"FAIL {p}")
            for e in errs:
                print(f"  - {e}")
        else:
            doc = json.loads(p.read_text(encoding="utf-8"))
            print(f"OK   {p}: {len(doc['findings'])} findings, {len(doc['sources'])} sources, "
                  f"{len(doc.get('opponents') or {})} opponents, researched_at {doc['researched_at']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
