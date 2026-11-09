#!/usr/bin/env python3
"""Read-only defense V2 scoring smoke benchmark, NOT a full live-path release test.

Use this against the physically confirmed testing-franchise installed book and
the *same model artifact* intended for gameplay. Measures model rank + joint
adjustments + USER reads, including cold-first-call latency. It does not time
session creation, DB queries, UI rendering or decision logging; therefore a
passing core benchmark does NOT establish 150ms end-to-end readiness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import defense_v2_adjustments as actions
from cfb_coach.madden.model import defense_v2_knowledge as knowledge
from cfb_coach.madden.model import defense_v2_live as live
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.playbook import eligible, active_books

SCENARIOS = (("early", "run"), ("short", "rpo"), ("medium", "cross"),
             ("long", "vert"), ("long", "flood"), ("early", "stack"))


def benchmark(db: CoachDB, model: dict, *, n: int = 100) -> dict:
    if n < 10:
        raise ValueError("benchmark needs at least ten varied calls")
    books = eligible(active_books(db, ("defense",)))
    calls = live._confirmed_book(db, books["defense"])
    samples = []
    picks = []
    def once(i: int) -> None:
        context, concept = SCENARIOS[i % len(SCENARIOS)]
        confidence = 0.78 if concept != "stack" else 0.25
        if concept == "stack":
            belief = {c: 1.0 / len(v2.CONCEPTS) for c in v2.CONCEPTS}
        else:
            other = (1 - confidence) / (len(v2.CONCEPTS) - 1)
            belief = {c: confidence if c == concept else other for c in v2.CONCEPTS}
        sit = SimpleNamespace(
            distance={"short": 2, "early": 10, "medium": 5, "long": 10}[context],
            down=3 if context in ("medium", "long") else 1,
            goal_line=False, red_zone=False, extras={}
        )
        start = time.perf_counter()
        ranked = v2.rank(model, context=context, calls=calls, belief=belief)["ranked"]
        choice = actions.choose(model=model, sit=sit, ranked=ranked,
                                belief=belief, context=context)
        advice = knowledge.user_guidance(
            play=choice["play"], formation=choice["formation"],
            family=choice["family"], belief=belief, sit=sit)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if not ranked or not advice["first_read"]:
            raise AssertionError("V2 missing call or user coaching")
        samples.append(elapsed_ms)
        picks.append({"formation": choice["formation"], "play": choice["play"],
                      "adjustment": choice["action_id"], "user_read": advice["first_read"],
                      "duration_ms": round(elapsed_ms, 3)})
    # This cold call includes module/candidate cache warmup, as live first snap might.
    for index in range(n):
        once(index)
    ordered = sorted(samples)
    p95 = ordered[max(0, int(0.95 * (n - 1)))]
    return {
        "schema": "madden.defense.v2.core_bench.v1",
        "runs": n, "installed_forms": len(books["defense"]),
        "installed_calls": len(calls),
        "cold_first_ms": round(samples[0], 3),
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(p95, 3), "max_ms": round(max(samples), 3),
        "core_only_at_or_below_150ms": bool(p95 <= 150.0),
        "complete_live_p95_verified": False,
        "user_database_untouched": True,
        "samples": picks[:6],
        "release_status": "CORE_MEASUREMENT_ONLY_NOT_GAME_READY",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    if model.get("schema") != v2.SCHEMA:
        p.error("wrong V2 artifact")
    db = CoachDB(args.db, read_only=True)
    try:
        report = benchmark(db, model, n=args.runs)
    finally:
        db.close()
    if args.out:
        if args.out.resolve() == args.db.resolve() or args.out.resolve() == args.model.resolve():
            p.error("benchmark output cannot replace database or model")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
