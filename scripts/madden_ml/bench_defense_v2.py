#!/usr/bin/env python3
"""Synthetic, in-repository 15-formation defensive V2 timing diagnostic.

NOT a loaded owner-database/live-overlay benchmark. Model and session inputs
here are deterministic fixtures, so the output cannot establish game readiness.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

from cfb_coach.madden import catalog_full
from cfb_coach.madden.model import defense_semantics
from cfb_coach.madden.model import defense_v2_pilot as v2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--max-formations", type=int, default=15)
    args = p.parse_args()
    if args.runs < 10 or not 1 <= args.max_formations <= 15:
        p.error("runs >= 10 and max-formations 1..15 required")
    records = catalog_full.formation_records("defense")
    book = {}
    for rec in sorted(records, key=lambda r: (-len(r["plays"]), r["formation"], r["source_book"])):
        form = str(rec["formation"])
        if form not in book:
            book[form] = list(dict.fromkeys(rec["plays"]))
        if len(book) == args.max_formations:
            break
    if not book:
        p.error("source catalog contains no formations")
    calls = [{"formation": f, "play": play, "installed": True, "legal": True}
             for f, plays in book.items() for play in plays]
    from cfb_coach.madden.defense_select import call_family
    # Only for repeatable timing. These are fake training observations.
    train = []
    for i, c in enumerate(calls[:80]):
        family = call_family(c["play"])
        if family:
            train.append(v2.Snap("synthetic-game", "cpu", "2026-01-01", i + 1,
                                 c["formation"], c["play"], family, "long",
                                 "cross", bool(i % 2),
                                 yards=float(i % 21), touchdown=bool(i % 20 == 0),
                                 turnover=bool(i % 25 == 0)))
    model = v2.fit(train)
    belief = {c: 1.0 / len(v2.CONCEPTS) for c in v2.CONCEPTS}
    latencies = []
    for i in range(args.runs + 5):
        t = time.perf_counter()
        ranking = v2.rank(model, context="long", calls=calls, belief=belief)
        if ranking["ranked"]:
            top = ranking["ranked"][0]
            defense_semantics.explain(top["play"], top["formation"])
        duration = (time.perf_counter() - t) * 1000
        if i >= 5:
            latencies.append(duration)
    vals = sorted(latencies)
    p95 = vals[min(len(vals)-1, math.ceil(len(vals)*0.95)-1)]
    report = {
        "schema": "madden.defense.v2.synthetic_benchmark.v1",
        "no_real_game_data": True, "not_end_to_end_live": True,
        "formations": len(book), "candidate_calls": len(calls),
        "calls_scored": len(ranking["ranked"]),
        "synthetic_train_rows": len(train), "runs": args.runs,
        "p50_ms": round(statistics.median(vals), 3),
        "p95_ms": round(p95, 3), "max_ms": round(max(vals), 3),
        "150ms_rank_only_gate": "PASS" if p95 < 150 else "NO_GO",
    }
    print(json.dumps(report, indent=2))
    return 0 if p95 < 150 else 2


if __name__ == "__main__":
    raise SystemExit(main())
