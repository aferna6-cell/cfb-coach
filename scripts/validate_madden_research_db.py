#!/usr/bin/env python3
"""Validate the Madden 27 research DB before it is pushed (the daily research routine runs this).

    python3 scripts/validate_madden_research_db.py [path]   # default: cfb_coach/data/madden27/research_db.json

Exit 0 = usable (prep will accept it), 1 = problems (printed). Also prints a short summary:
macros, researched fields per macro, sources, and controls that are not yet confirmed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfb_coach.madden import research_db as rdb  # noqa: E402


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / rdb.DB_REL_PATH
    try:
        db = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"FAIL: cannot read {path}: {exc}")
        return 1
    errs = rdb.validate(db)
    for e in errs:
        print(f"FAIL: {e}")
    rdb._STATE.update(db=db, origin=str(path))  # summarise exactly this file
    print(f"{path}: updated {db.get('updated')}, {len(db.get('sources') or [])} sources, "
          f"{len(db.get('defense_macros') or [])} defense macros, "
          f"{len(db.get('offense_adjustments') or [])} offense / {len(db.get('defense_adjustments') or [])} defense adjustments")
    for m in rdb.defense_macros():
        rows = rdb.full_settings(m["id"])
        n = sum(1 for r in rows if r.get("source") != "default")
        new = [r["setting"] for r in rows if r.get("new_field")]
        print(f"  {m['id']:<16} #{m.get('meta_rank', '?'):<3} {n:>2}/{len(rows)} fields researched"
              + (f"  (fields not in editor list: {', '.join(new)})" if new else ""))
    for side in ("offense", "defense"):
        for key, c in ((db.get("controls_xbox") or {}).get(side) or {}).items():
            if c.get("confidence") != "confirmed":
                print(f"  controls {side}.{key}: {c.get('confidence')} — {c.get('buttons')}")
    rdb.reset()
    print("OK" if not errs else f"{len(errs)} problem(s)")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
