#!/usr/bin/env python3
"""Read-only defensive V2 shadow CLI. Never writes a source DB or live artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from cfb_coach.madden.model import defense_v2_pilot as v2


def _open(db: Path) -> sqlite3.Connection:
    if not db.is_file():
        raise FileNotFoundError(f"database not found: {db}")
    conn = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _print(data: dict | list) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inventory", "audit", "train", "evaluate", "preview", "design", "postgame", "cold-start"))
    p.add_argument("--db", type=Path, required=True, help="explicit path to an existing Madden DB")
    p.add_argument("--franchise", help="exact recorded game_sessions.dynasty, not a guess")
    p.add_argument("--out", type=Path, help="output SHADOW artifact, outside live game data")
    p.add_argument("--model", type=Path, help="existing V2 shadow artifact for preview")
    p.add_argument("--calls-json", type=Path, help="explicit installed/legal candidate list (JSON)")
    p.add_argument("--session", help="current/new game session ID for preview")
    p.add_argument("--opponent", help="opponent ID for preview")
    p.add_argument("--next-seq", type=int, help="next pre-snap sequence number")
    p.add_argument("--down", type=int)
    p.add_argument("--distance", type=int)
    args = p.parse_args()
    if args.mode != "inventory" and not args.franchise:
        p.error("--franchise is required to avoid mixing franchises")

    with _open(args.db) as conn:
        if args.mode == "inventory":
            _print(v2.available_franchises(conn))
            return 0
        rows, stats = v2.verified_rows(conn, franchise=args.franchise)
        if args.mode == "cold-start":
            if not args.out:
                p.error("cold-start requires --out outside the live game directory")
            if rows:
                p.error("verified defensive executions exist: use train, not cold-start")
            out = args.out.resolve()
            if out == args.db.resolve() or Path.home().joinpath(".cfb-coach") in out.parents:
                p.error("refusing to place a cold-start artifact in the live data directory")
            model = v2.cold_start(args.franchise)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
            _print({"status": "EXPERIMENTAL_PRIOR_ONLY_NOT_TRAINED",
                    "artifact": str(out), "franchise": args.franchise,
                    "verified_defensive_executions": 0,
                    "gameplay_effectiveness_proven": False})
            return 0
        if args.mode == "audit":
            _print({"franchise": args.franchise, **stats})
            return 0
        if args.mode == "evaluate":
            _print({"franchise": args.franchise, "training_audit": stats,
                    "validation": v2.evaluate(rows)})
            return 0
        if args.mode == "postgame":
            if not args.model or not args.out:
                p.error("postgame requires --model (saved champion) and --out (new candidate)")
            champion = json.loads(args.model.read_text(encoding="utf-8"))
            if champion.get("trained_franchise") != args.franchise:
                p.error("saved champion franchise mismatch")
            report = v2.compare_champion_challenger(rows, champion)
            if not rows:
                p.error("no verified defensive execution rows for postgame update")
            challenger = v2.fit(rows)
            challenger["trained_franchise"] = args.franchise
            challenger["trained_sessions"] = sorted({r.session for r in rows})
            challenger["trained_through_started_ts"] = max(r.started_ts for r in rows)
            out = args.out.resolve()
            if out == args.db.resolve() or out == args.model.resolve() or Path.home().joinpath(".cfb-coach") in out.parents:
                p.error("candidate must not replace the DB, saved champion or any live artifact")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(challenger, indent=2, sort_keys=True), encoding="utf-8")
            _print({"status": "CANDIDATE_ONLY_NOT_PROMOTED", "candidate": str(out),
                    "verified": len(rows), "comparison": report})
            return 0
        if args.mode == "design":
            if not args.model or not args.out:
                p.error("design requires --model and --out")
            from cfb_coach.madden.model import defense_v2_playbook
            from cfb_coach.madden import catalog
            art = json.loads(args.model.read_text(encoding="utf-8"))
            if art.get("trained_franchise") != args.franchise:
                p.error("model franchise must match requested test franchise")
            sources = {name: catalog.book_formations("defense", name)
                       for name in catalog.book_names("defense")}
            proposed = defense_v2_playbook.propose(catalogue=sources, model=art)
            out = args.out.resolve()
            if out == args.db.resolve() or Path.home().joinpath(".cfb-coach") in out.parents:
                p.error("refusing to write a proposal into the live game directory")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(proposed, indent=2, sort_keys=True), encoding="utf-8")
            _print({"status": "PROPOSAL_ONLY_NOT_INSTALLED", "path": str(out),
                    "formations": proposed["n_formations"], "plays": proposed["n_plays"],
                    "fingerprint": proposed["proposal_fingerprint"]})
            return 0
        if args.mode == "train":
            if args.out is None:
                p.error("train requires --out")
            if not rows:
                p.error("no verified defensive executions; audit the new franchise first")
            out = args.out.resolve()
            if out == args.db.resolve() or Path.home().joinpath(".cfb-coach") in out.parents:
                p.error("refusing to place a shadow artifact in the live data directory")
            model = v2.fit(rows)
            model["trained_franchise"] = args.franchise
            model["trained_sessions"] = sorted({r.session for r in rows})
            model["trained_through_started_ts"] = max(r.started_ts for r in rows)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
            _print({"status": "SHADOW_ONLY", "artifact": str(out),
                    "franchise": args.franchise, "verified": len(rows)})
            return 0

        if not all((args.model, args.calls_json, args.session, args.opponent, args.next_seq)):
            p.error("preview requires --model --calls-json --session --opponent --next-seq")
        model = json.loads(args.model.read_text(encoding="utf-8"))
        if model.get("schema") != v2.SCHEMA or model.get("policy") != "shadow_only":
            p.error("invalid or non-shadow V2 artifact")
        current = conn.execute(
            "SELECT started_ts FROM game_sessions WHERE session_id=? AND opponent_id=? AND dynasty=?",
            (args.session, args.opponent, args.franchise)).fetchone()
        if current is None:
            p.error("game session/opponent/franchise mismatch")
        if (args.session in model.get("trained_sessions", []) or
                not model.get("trained_through_started_ts") or
                not current[0] or current[0] <= model["trained_through_started_ts"]):
            p.error("shadow artifact includes this game or a future game: chronological leak")
        history = v2.earlier_opponent_observations(
            conn, franchise=args.franchise, session=args.session,
            opponent=args.opponent, next_seq=args.next_seq)
        belief = v2.concept_belief(
            model, history, session=args.session, opponent=args.opponent, next_seq=args.next_seq)
        candidates = json.loads(args.calls_json.read_text(encoding="utf-8"))
        if not isinstance(candidates, list):
            p.error("--calls-json must be a JSON list of explicit candidate records")
        ranked = v2.rank(
            model, context=v2.context_key(args.down, args.distance),
            calls=candidates, belief=belief)
        ranked["ranked"] = ranked["ranked"][:10]
        ranked["opponent_concept_belief"] = belief
        ranked["earlier_named_concepts"] = len(history)
        _print(ranked)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
