"""Read-only replay: current books vs pregame-architect books on logged game situations.

For each session the architect designs both books from evidence logged BEFORE
that session started, then every logged situation is replayed through the live
pickers twice: once with the currently applied book, once with the proposed
book. Reported per side: formation usage and concentration, legality (every
call inside the book it was called from), latency, and model fallback rates.

This compares selection behavior only. It never claims a replayed call would
have changed a result, and it writes nothing: the database is opened
read-only and no proposal files are saved. Runs without clock or a pre-snap
"showing" field.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import defense_v1, experimental_model, game_evidence, playbook_architect
from cfb_coach.madden.model.experimental_live import resolve_artifact_path
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.opponents import is_cpu_opponent

DEFAULT_SESSIONS = ("698831a80afb434b", "3d9b477c10df4827")


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _established(rows: Sequence[Mapping[str, Any]], side: str, opponent: str,
                 book: Mapping[str, Sequence[str]], seed: str) -> dict[str, Any]:
    rng = random.Random(seed)
    usage: Counter[str] = Counter()
    illegal = errors = 0
    lat: list[float] = []
    for row in rows:
        raw = str(row["situation_raw"] or "")
        sit = parse_madden_situation(raw if side == "offense" else f"d {raw}", default_side=side)
        sit.two_minute = False
        t0 = time.perf_counter()
        try:
            call = make_call(sit, opponent, None, rng=rng, playbook={side: dict(book)},
                             cpu_defense=side == "defense")
        except Exception:  # noqa: BLE001 - counted, replay continues
            errors += 1
            continue
        lat.append(1000 * (time.perf_counter() - t0))
        usage[str(call.formation)] += 1
        if call.play not in (book.get(call.formation) or []):
            illegal += 1
    n = sum(usage.values())
    return {"calls": n, "errors": errors, "illegal": illegal, "formations_used": len(usage),
            "top": usage.most_common(3), "top_share": round(usage.most_common(1)[0][1] / n, 3) if n else None,
            "p50_ms": round(statistics.median(lat), 2) if lat else None, "p95_ms": round(_pct(lat, 0.95), 2)}


def _model_defense(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any] | None,
                   book: Mapping[str, Sequence[str]], scope: bool, session: str) -> dict[str, Any] | None:
    if artifact is None:
        return None
    usage: Counter[str] = Counter()
    fallbacks: Counter[str] = Counter()
    illegal = 0
    lat: list[float] = []
    recent: list[tuple[str, ...]] = []
    for i, row in enumerate(rows, 1):
        sit = parse_madden_situation(f"d {row['situation_raw'] or ''}", default_side="defense")
        sit.two_minute = False
        try:
            out = defense_v1.pick_call(artifact, sit, book, recent=recent, seed=f"{session}:{i}:replay",
                                       opponent_profile=None, installed_scope=scope)
        except defense_v1.DefenseV1Error as exc:
            fallbacks[exc.reason] += 1
            continue
        pick = out["pick"]
        lat.append(out["elapsed_ms"])
        usage[str(pick["formation"])] += 1
        if pick["play"] not in (book.get(pick["formation"]) or []):
            illegal += 1
        recent.append((pick["formation"], pick["play"], pick.get("family") or "", pick.get("package") or ""))
    n = len(rows)
    return {"model_calls": sum(usage.values()), "fallback_rate": round(sum(fallbacks.values()) / n, 3) if n else None,
            "fallbacks": dict(fallbacks), "illegal": illegal, "formations_used": len(usage),
            "top": usage.most_common(3), "max_ms": round(max(lat), 2) if lat else None}


def replay(db_path: Path, sessions: Sequence[str], defense_artifact: Path | None) -> None:
    db = CoachDB.open_read_only(db_path)
    try:
        applied = playbook.load_books(db)
        art_path = resolve_artifact_path(db)
        art = experimental_model.load_artifact(art_path) if art_path else None
        quality = str(getattr(art, "evidence_quality", "prior_driven") or "prior_driven")
        try:
            d_art = defense_v1.load_artifact(defense_artifact) if defense_artifact else None
        except defense_v1.DefenseV1Error:
            d_art = None
        for sid in sessions:
            game = db.conn.execute("SELECT opponent_id, started_ts, score, result_wl FROM game_sessions "
                                   "WHERE session_id=?", (sid,)).fetchone()
            if not game:
                print(f"\n{sid}: session not found, skipped")
                continue
            opp, start = str(game["opponent_id"]), str(game["started_ts"])
            fixed = game_evidence.corrected_result({"session_id": sid, "score": game["score"],
                                                    "result_wl": game["result_wl"]})
            kind = "cpu" if is_cpu_opponent(opp) else "human"
            print(f"\n=== {sid} vs {opp} ({kind}) {fixed['result_wl']} {fixed['score']}"
                  + (f" (stored {fixed['stored_score']}, corrected)" if fixed["corrected"] else ""))
            print(f"Evidence cut-off: rows before {start} only.")
            designs = {
                "offense": playbook_architect.design_side(
                    "offense", opponent_id=opp, db=db, applied=applied.get("offense"),
                    evidence=game_evidence.offense_evidence(db, opp, before_ts=start), art=art,
                    evidence_quality=quality, opponent_type=kind),
                "defense": playbook_architect.design_side(
                    "defense", opponent_id=opp, db=db, applied=applied.get("defense"),
                    evidence=game_evidence.defense_evidence(db, opp, before_ts=start), opponent_type=kind),
            }
            for side in ("offense", "defense"):
                rows = list(db.conn.execute(
                    "SELECT situation_raw FROM snaps WHERE session_id=? AND side=? "
                    "ORDER BY COALESCE(snap_seq, id), id", (sid, side)))
                d = designs[side]
                print(f"\n[{side}] {len(rows)} logged situations | proposal {d['fingerprint']} | "
                      f"keep {len(d['changes']['keep'])}, add {d['changes']['add']}, "
                      f"remove {d['changes']['remove']}, switch source book {d['changes']['new_source']}")
                if not rows:
                    print("  no logged situations for this side (CPU games are offense-only unless --cpu-defense)")
                    continue
                books = {"current": dict((applied.get(side) or {}).get("formations") or {}),
                         "proposed": dict(d["book"]["formations"])}
                for label, book in books.items():
                    if not book:
                        print(f"  {label:<8} no applied book")
                        continue
                    est = _established(rows, side, opp, book, f"{sid}:{side}")
                    print(f"  {label:<8} established caller: {est}")
                    if side == "defense":
                        model = _model_defense(rows, d_art, book, scope=(label == "proposed"), session=sid)
                        print(f"  {label:<8} defense model: {model if model else 'no local defense artifact'}")
        print("\nSelection behavior only. Replayed calls did not happen in these games and say nothing about results.")
        print("Read-only: no database writes, no proposal files, no installs.")
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--session", action="append", help="session id (repeatable); default Ryan loss + latest CPU win")
    p.add_argument("--defense-artifact", type=Path, default=defense_v1.default_artifact_path())
    args = p.parse_args()
    replay(args.db, args.session or list(DEFAULT_SESSIONS), args.defense_artifact)
