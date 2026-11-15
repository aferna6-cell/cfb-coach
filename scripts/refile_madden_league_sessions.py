#!/usr/bin/env python3
"""Re-file two Madden sessions that were logged under the wrong opponent or league.

1. Session 601248169c3f4332 (Chiefs vs Colts) was saved under opponent_id cpu
   because the alias "nd" substring-matched "Indianapolis". Retag it as a
   Chiefs *online* league game vs the Indianapolis Colts. Never move it to the
   Titans franchise.

2. A Titans vs Colts game started on or after 2026-11-15 16:20 America/New_York
   may have been launched with ``--franchise online --opponent-team
   "Indianapolis Colts"``, so it was saved as the Chiefs user (and possibly
   opponent_id cpu). Move that session into the Titans franchise vs the Colts.

Preview by default. Pass --apply to write. --db is required; this script never
guesses a live database path.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

CHIEFS_COLTS_SESSION = "601248169c3f4332"
TITANS_CUTOVER = datetime(2026, 11, 15, 16, 20, tzinfo=ZoneInfo("America/New_York"))
COLTS = "Indianapolis Colts"
CHIEFS = "Kansas City Chiefs"
TITANS = "Tennessee Titans"

CHIEFS_TARGET = {
    "opponent_id": "online_indianapolis_colts",
    "dynasty": "online",
    "league": "online",
    "user_team": CHIEFS,
    "opponent_team": COLTS,
    "opponent_type": "human",
}
TITANS_TARGET = {
    "opponent_id": "titans_indianapolis_colts",
    "dynasty": "titans",
    "league": "titans",
    "user_team": TITANS,
    "opponent_team": COLTS,
    "opponent_type": "human",
}

_SESSION_COLS = ("opponent_id", "dynasty", "league", "user_team", "opponent_team", "opponent_type")
_SNAP_COLS = ("opponent_id", "league", "user_team", "opponent_team")
_DECISION_COLS = ("opponent_id", "league", "opponent_team")


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import timezone

        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_get(row: sqlite3.Row, key: str) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def matches_target(row: sqlite3.Row, target: dict[str, str]) -> bool:
    return all(_row_get(row, k) == v for k, v in target.items())


def describe(row: sqlite3.Row) -> str:
    ts = _row_get(row, "started_ts")
    return (
        f"{row['session_id']} started={ts} dynasty={_row_get(row, 'dynasty')!r} "
        f"opponent_id={_row_get(row, 'opponent_id')!r} league={_row_get(row, 'league')!r} "
        f"user_team={_row_get(row, 'user_team')!r} opponent_team={_row_get(row, 'opponent_team')!r}"
    )


def open_db(path: Path, *, apply: bool) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError(f"Database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode={'rw' if apply else 'ro'}", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _looks_like_misfiled_colts(row: sqlite3.Row) -> bool:
    oid = str(_row_get(row, "opponent_id") or "")
    opp_team = _row_get(row, "opponent_team")
    if opp_team and opp_team != COLTS:
        return False
    if opp_team == COLTS:
        return True
    return oid in {"cpu", "online_indianapolis_colts"}


def find_chiefs_session(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM game_sessions WHERE session_id = ?", (CHIEFS_COLTS_SESSION,)
    ).fetchone()


def find_titans_candidates(conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    """Return (already_correct, misfiled_candidates) after the Titans cutover."""
    already: list[sqlite3.Row] = []
    cands: list[sqlite3.Row] = []
    for row in conn.execute("SELECT * FROM game_sessions ORDER BY started_ts").fetchall():
        if row["session_id"] == CHIEFS_COLTS_SESSION:
            continue
        ts = parse_ts(_row_get(row, "started_ts"))
        if ts is None or ts < TITANS_CUTOVER:
            continue
        if matches_target(row, TITANS_TARGET):
            already.append(row)
            continue
        if _looks_like_misfiled_colts(row):
            cands.append(row)
    return already, cands


def pick_titans_session(
    already: list[sqlite3.Row], cands: list[sqlite3.Row]
) -> tuple[sqlite3.Row | None, str | None]:
    """Pick one session to retag, or explain why not."""
    if already and not cands:
        return already[0], None
    if already and cands:
        return None, (
            "Found an already-tagged Titans vs Colts session and also a misfiled "
            "candidate after the cutover; refusing to guess.\n"
            + "\n".join("  already " + describe(r) for r in already)
            + "\n"
            + "\n".join("  candidate " + describe(r) for r in cands)
        )
    if not cands:
        return None, (
            "No Titans vs Colts session found with started_ts on or after "
            f"{TITANS_CUTOVER.isoformat()} (America/New_York). "
            "If the game has not been logged yet, re-run this after it exists."
        )
    # Prefer a row already tagged as Colts / online_indianapolis_colts over a bare cpu row.
    def _rank(row: sqlite3.Row) -> tuple[int, str]:
        oid = str(_row_get(row, "opponent_id") or "")
        opp_team = _row_get(row, "opponent_team")
        if opp_team == COLTS or oid == "online_indianapolis_colts":
            return (0, row["session_id"])
        return (1, row["session_id"])

    ranked = sorted(cands, key=_rank)
    clear = [r for r in ranked if _rank(r)[0] == 0]
    bare = [r for r in ranked if _rank(r)[0] == 1]
    if len(clear) > 1:
        return None, (
            "Multiple misfiled Colts sessions after the cutover; refusing to guess.\n"
            + "\n".join("  " + describe(r) for r in clear)
        )
    if clear:
        return clear[0], None
    if len(bare) > 1:
        return None, (
            "Multiple untagged cpu sessions after the cutover; refusing to guess.\n"
            + "\n".join("  " + describe(r) for r in bare)
        )
    return bare[0], None


def counts(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    snaps = conn.execute("SELECT COUNT(*) FROM snaps WHERE session_id = ?", (session_id,)).fetchone()[0]
    decisions = 0
    if table_cols(conn, "ravens_defense_decisions"):
        decisions = conn.execute(
            "SELECT COUNT(*) FROM ravens_defense_decisions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    return {"snaps": int(snaps), "decisions": int(decisions)}


def ensure_opponent(conn: sqlite3.Connection, oid: str, display_name: str) -> None:
    cols = table_cols(conn, "opponents")
    if not cols or "id" not in cols:
        return
    existing = conn.execute("SELECT id FROM opponents WHERE id = ?", (oid,)).fetchone()
    if existing:
        return
    profile = {
        "id": oid,
        "display_name": display_name,
        "skill": "human",
        "confidence": "cold",
        "cold_start": True,
    }
    fields = ["id"]
    values: list[Any] = [oid]
    if "display_name" in cols:
        fields.append("display_name")
        values.append(display_name)
    if "team_now" in cols:
        fields.append("team_now")
        values.append(display_name)
    if "skill" in cols:
        fields.append("skill")
        values.append("human")
    if "confidence" in cols:
        fields.append("confidence")
        values.append("cold")
    if "profile_json" in cols:
        fields.append("profile_json")
        values.append(json.dumps(profile))
    placeholders = ", ".join("?" for _ in fields)
    conn.execute(f"INSERT INTO opponents ({', '.join(fields)}) VALUES ({placeholders})", values)


def apply_retag(conn: sqlite3.Connection, session_id: str, target: dict[str, str]) -> None:
    sess_cols = table_cols(conn, "game_sessions")
    set_cols = [c for c in _SESSION_COLS if c in sess_cols]
    assignments = ", ".join(f"{c} = ?" for c in set_cols)
    conn.execute(
        f"UPDATE game_sessions SET {assignments} WHERE session_id = ?",
        [target[c] for c in set_cols] + [session_id],
    )
    snap_cols = table_cols(conn, "snaps")
    snap_set = [c for c in _SNAP_COLS if c in snap_cols]
    if snap_set:
        assignments = ", ".join(f"{c} = ?" for c in snap_set)
        conn.execute(
            f"UPDATE snaps SET {assignments} WHERE session_id = ?",
            [target[c] for c in snap_set] + [session_id],
        )
    dec_cols = table_cols(conn, "ravens_defense_decisions")
    dec_set = [c for c in _DECISION_COLS if c in dec_cols]
    if dec_set:
        assignments = ", ".join(f"{c} = ?" for c in dec_set)
        conn.execute(
            f"UPDATE ravens_defense_decisions SET {assignments} WHERE session_id = ?",
            [target[c] for c in dec_set] + [session_id],
        )
    ensure_opponent(conn, target["opponent_id"], target["opponent_team"])


def plan_job(name: str, row: sqlite3.Row | None, target: dict[str, str], missing: str | None) -> dict[str, Any]:
    if missing and row is None:
        return {"name": name, "status": "missing", "message": missing, "session_id": None}
    if row is None:
        return {"name": name, "status": "missing", "message": missing or "session not found", "session_id": None}
    if matches_target(row, target):
        return {
            "name": name,
            "status": "already",
            "session_id": row["session_id"],
            "message": f"Already tagged correctly: {describe(row)}",
            "row": row,
            "target": target,
        }
    return {
        "name": name,
        "status": "needs",
        "session_id": row["session_id"],
        "message": f"Will retag {describe(row)}\n    -> " + " ".join(f"{k}={v!r}" for k, v in target.items()),
        "row": row,
        "target": target,
    }


def build_plan(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    chiefs = find_chiefs_session(conn)
    chiefs_missing = (
        None
        if chiefs is not None
        else f"Session {CHIEFS_COLTS_SESSION} not found; leaving the Chiefs refile as a no-op."
    )
    already, cands = find_titans_candidates(conn)
    titans_row, titans_err = pick_titans_session(already, cands)
    # pick_titans_session returns already-correct row with err=None; treat as already.
    jobs = [
        plan_job(
            "chiefs_colts_601248169c3f4332",
            chiefs,
            CHIEFS_TARGET,
            chiefs_missing,
        ),
        plan_job("titans_colts_after_cutover", titans_row, TITANS_TARGET, titans_err),
    ]
    return jobs


def run(path: Path, *, apply: bool = False) -> int:
    path = path.expanduser().resolve()
    conn = open_db(path, apply=apply)
    try:
        jobs = build_plan(conn)
        print(f"Database: {path}")
        rc = 0
        for job in jobs:
            n = counts(conn, job["session_id"]) if job["session_id"] else {"snaps": 0, "decisions": 0}
            extra = f" ({n['snaps']} snaps, {n['decisions']} defense decisions)" if job["session_id"] else ""
            print(f"[{job['name']}] {job['status']}{extra}")
            print(f"  {job['message']}")
            if job["status"] == "missing" and job["name"] == "titans_colts_after_cutover":
                # Not fatal: the game may not have been logged yet.
                continue
            if job["status"] == "missing":
                # Chiefs session missing is also non-fatal on a DB that never had it.
                continue
        needed = [j for j in jobs if j["status"] == "needs"]
        if not needed:
            print("Nothing to apply.")
            return 0
        if not apply:
            print("Preview only. Add --apply to write these changes.")
            return 0
        conn.execute("BEGIN IMMEDIATE")
        try:
            for job in needed:
                apply_retag(conn, job["session_id"], job["target"])
                # Confirm the row now matches. If not, roll back.
                row = conn.execute(
                    "SELECT * FROM game_sessions WHERE session_id = ?", (job["session_id"],)
                ).fetchone()
                if row is None or not matches_target(row, job["target"]):
                    raise ValueError(f"Retag of {job['session_id']} did not stick; rolling back.")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        print("Applied.")
        for job in needed:
            row = conn.execute(
                "SELECT * FROM game_sessions WHERE session_id = ?", (job["session_id"],)
            ).fetchone()
            print(f"  now {describe(row)}")
        return rc
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        required=True,
        help="Path to madden27.db (required; no default, so a live DB is never implied)",
    )
    p.add_argument("--apply", action="store_true", help="Write the retags (default is preview only)")
    args = p.parse_args(argv)
    try:
        return run(Path(args.db), apply=bool(args.apply))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
