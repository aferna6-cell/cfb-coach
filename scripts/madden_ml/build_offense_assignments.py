#!/usr/bin/env python3
"""Build cfb_coach/data/madden27/offense_assignments.json from the immutable snapshot + sourced annotations.

Deterministic. Reads only repo data files; never touches ~/.cfb-coach, the live DB or models.
  python scripts/madden_ml/build_offense_assignments.py            # writes the data file
  python scripts/madden_ml/build_offense_assignments.py --check    # fails if the file is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import offense_assignments as oa  # noqa: E402
from cfb_coach.madden.model import offense_taxonomy as tx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output", default=str(oa.DB_PATH))
    a = ap.parse_args()
    snap = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    ann = json.loads(oa.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    text = json.dumps(oa.build_database(snap, ann), indent=1, sort_keys=True) + "\n"
    tax_text = json.dumps(tx.as_dict(), indent=1, sort_keys=True) + "\n"
    out = Path(a.output)
    tax_out = out.with_name("offense_taxonomy.json")
    if a.check:
        ok = (out.exists() and out.read_text(encoding="utf-8") == text
              and tax_out.exists() and tax_out.read_text(encoding="utf-8") == tax_text)
        print("offense_assignments.json + offense_taxonomy.json are", "current" if ok else "STALE")
        return 0 if ok else 1
    out.write_text(text, encoding="utf-8")
    tax_out.write_text(tax_text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} bytes) and {tax_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
