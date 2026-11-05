#!/usr/bin/env python3
"""Guarded repair of the Nov 4 CPU winner-first final-score typo.

Preview by default. When --apply is passed, make a consistent SQLite backup,
then repair only final-score and associated reconciliation metadata. Never
alter individual plays or invent any missing touchdowns.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

SESSION = "3d9b477c10df4827"
CORRECT = "35-7"
ERRONEOUS = "42-7"


def repair(path: Path, *, apply: bool = False) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode={'rw' if apply else 'ro'}", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM game_sessions WHERE session_id=?", (SESSION,)).fetchone()
        if row is None:
            raise ValueError("Target session not found; refusing to edit unrelated game")
        if (row["dynasty"], row["opponent_id"], row["result_wl"], row["snaplog_score"]) != (
            "primary", "cpu", "win", CORRECT
        ):
            raise ValueError("Target identity or snap-log score disagrees with user correction")
        tally = conn.execute(
            "SELECT COUNT(*) AS snaps, "
            "SUM(CASE WHEN result='td' THEN 1 ELSE 0 END) AS tds, "
            "SUM(CASE WHEN result='int' THEN 1 ELSE 0 END) AS ints "
            "FROM snaps WHERE session_id=?", (SESSION,)
        ).fetchone()
        if (tally["snaps"], tally["tds"], tally["ints"]) != (41, 5, 3):
            raise ValueError("Snap/TD/INT counts changed; refusing automated repair")
        if row["score"] not in (ERRONEOUS, CORRECT):
            raise ValueError("Stored final score does not match known values")
        already = (row["score"] == CORRECT
                   and row["score_gap_for"] == 0
                   and row["score_gap_against"] == 0
                   and row["score_reconcile"] == "match")
        print(f"Session {SESSION}: recorded {row['score']}, actual W {CORRECT}; 41 snaps, 5 TD, 3 INT")
        if already:
            print("Already corrected.")
            return
        if not apply:
            print("Preview only. Add --apply to back up and fix the metadata.")
            return
        outdir = path.parent / "backups"
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = outdir / f"{path.stem}.pre-scorefix.{stamp}.db"
        with sqlite3.connect(str(backup)) as dst:
            conn.backup(dst)
        print(f"Backup: {backup}")
        meta = conn.execute("SELECT value FROM meta WHERE key='last_game:cpu'").fetchone()
        try:
            last = json.loads(meta["value"]) if meta else None
        except (json.JSONDecodeError, TypeError):
            last = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            changed = conn.execute(
                "UPDATE game_sessions SET score=?, score_gap_for=0, "
                "score_gap_against=0, score_reconcile='match' "
                "WHERE session_id=? AND score=? AND snaplog_score=?",
                (CORRECT, SESSION, ERRONEOUS, CORRECT),
            )
            if changed.rowcount != 1:
                raise ValueError("Concurrent change or already repaired; refused")
            if isinstance(last, dict) and last.get("session_id") == SESSION:
                last["score"] = CORRECT
                conn.execute(
                    "UPDATE meta SET value=? WHERE key='last_game:cpu'",
                    (json.dumps(last, sort_keys=True),),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        print("Corrected final score + reconciliation; snaps and ML decisions unchanged.")
    finally:
        conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    repair(args.db, apply=args.apply)
