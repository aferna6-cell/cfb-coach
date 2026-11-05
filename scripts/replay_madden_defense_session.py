"""Read-only defense-v2 policy replay of one completed Madden human game.

Requires local defense_v1.json (not stored in Git) and installed playbook.
Compares selection concentration/guards/latency only, NEVER counterfactual wins.
Runs without clock or presnap "showing" data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import defense_v1
from cfb_coach.madden.playbook import active_books, eligible
from cfb_coach.madden.situation import parse_madden_situation

DEFAULT_RYAN_SESSION = "698831a80afb434b"


def _peak_streak(items):
    best = cur = 0
    prev = None
    for item in items:
        if item == prev:
            cur += 1
        else:
            cur = 1
        prev = item
        best = max(best, cur)
    return best


def replay(db_path: Path, artifact_path: Path, session_id: str) -> None:
    artifact = defense_v1.load_artifact(artifact_path)
    if artifact["schema"] != defense_v1.SCHEMA_V2:
        raise SystemExit("STOP: requires a situation-aware defense v2 artifact")
    db = CoachDB.open_read_only(db_path)
    try:
        book = eligible(active_books(db, ("defense",)))["defense"]
        rows = list(db.conn.execute(
            "SELECT snap_seq, situation_raw, formation, play, concept_seen "
            "FROM snaps WHERE session_id=? AND side='defense' "
            "ORDER BY COALESCE(snap_seq,id),id", (session_id,),
        ))
        if not rows:
            raise SystemExit("STOP: no defensive snaps in requested session")
        game = db.conn.execute(
            "SELECT opponent_id, score, result_wl FROM game_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not game:
            raise SystemExit("STOP: session missing")
        old_calls = [(row["formation"], row["play"]) for row in rows]
        new_calls = []
        recent = []
        prior_concepts = []
        latencies = []
        fallbacks = Counter()
        run_guard_errors = 0
        for index, row in enumerate(rows, 1):
            raw = str(row["situation_raw"] or "")
            sit = parse_madden_situation(raw, default_side="defense")
            # Deliberately do not add any clock/pre-snap look or score inference.
            setattr(sit, "two_minute", False)
            had_run_lean = defense_v1.recent_run_heavy(prior_concepts)
            try:
                picked = defense_v1.pick_call(
                    artifact, sit, book, recent=recent,
                    recent_concepts=prior_concepts,
                    seed=f"{session_id}:{row['snap_seq'] or index}:defense-v2",
                    # No full-game opponent profile here: it would leak future snaps.
                    opponent_profile=None,
                )
                choice = picked["pick"]
                key = (choice["formation"], choice["play"])
                if key[1] not in book.get(key[0], []):
                    raise AssertionError("replay produced out-of-book call")
                latencies.append(picked["elapsed_ms"])
                if had_run_lean and defense_v1._is_light_box(choice):
                    run_guard_errors += 1
                recent.append((key[0], key[1],
                               choice.get("family") or "",
                               choice.get("package") or ""))
            except defense_v1.DefenseV1Error as exc:
                fallbacks[exc.reason] += 1
                # Replay a real fallback as the established historical call.
                key = (row["formation"], row["play"])
                recent.append((key[0], key[1]))
            new_calls.append(key)
            if row["concept_seen"]:
                prior_concepts.append(str(row["concept_seen"]))
        print(f"Session: {session_id} vs {game['opponent_id']}; {game['result_wl']} {game['score']}")
        print(f"Model schema: {artifact['schema']} | defensive snaps: {len(rows)}")
        print("Original formations:", dict(Counter(form for form, _ in old_calls)))
        print("Replay formations:  ", dict(Counter(form for form, _ in new_calls)))
        print("Original top calls:", Counter(old_calls).most_common(5))
        print("Replay top calls:  ", Counter(new_calls).most_common(5))
        print("Replay changed:", sum(a != b for a, b in zip(old_calls, new_calls)),
              "of", len(rows), "(selection only, not proof of improvement)")
        print("Original max exact-play streak:", _peak_streak(old_calls))
        print("Replay max exact-play streak:  ", _peak_streak(new_calls))
        print("Recorded fallback reasons:", dict(fallbacks))
        print("Run-heavy light-box guard failures:", run_guard_errors)
        if latencies:
            print("Model inference latency ms, max:",
                  round(max(latencies), 2),
                  "p95:", round(sorted(latencies)[int(.95*(len(latencies)-1))], 2))
        print("NO outcome counterfactual; replay decisions did not occur in the game.")
        print("NO future-data opponent priors; only prior in-game concepts used.")
        print("NO DB writes, retrain, install, or model activation.")
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--artifact", type=Path, default=defense_v1.default_artifact_path())
    p.add_argument("--session", default=DEFAULT_RYAN_SESSION)
    args = p.parse_args()
    replay(args.db, args.artifact, args.session)
