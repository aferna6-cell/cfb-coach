#!/usr/bin/env python3
"""One report: repository football universe (committed, lawful sources) + the private ingestion store, side by side, units never merged.

  python scripts/madden_ml/audit_football_universe.py [--store ~/cfb_acemadden] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden import catalog as repo_catalog  # noqa: E402
from cfb_coach.madden.model import ace_catalog as ac  # noqa: E402
from cfb_coach.madden.model import football_universe as fu  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=str(Path.home() / "cfb_acemadden"))
    ap.add_argument("--out")
    a = ap.parse_args()
    reference = {s: [(f, p) for b in repo_catalog.book_names(s) for f, ps in repo_catalog.book_formations(s, b).items() for p in ps] for s in ac.SIDES}
    store = ac.Store(a.store)
    recs = store.read()
    perm = next((r["permitted_use"] for r in recs.values()), {"status": "no data imported yet"})
    rep = {"repository_universe": fu.coverage_report(), "ingestion_store": ac.audit(recs, reference=reference, permission=perm),
           "units_note": "listings, memberships, unique pairs and art-bearing plays are different units and are never summed across sources"}
    if not recs:
        rep["ingestion_store"]["NOTE"] = "the ingestion store is empty: full extraction has not occurred"
    text = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
