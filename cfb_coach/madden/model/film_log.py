"""Read coach snap identities for film review. Never writes the gameplay database.

A snap identifier is exported only when `ml_snap_id` or a decision `snap_id`
is already stored. A recommended play is not copied into `executed_play`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping


def export_coach_log(
    db_path: str | Path | None,
    game_id: str,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    """Return stored snaps for one game. Missing identifiers are omitted."""
    path = None if db_path is None else Path(db_path)
    report: dict[str, Any] = {
        "ok": False,
        "read_only": True,
        "history_modified": False,
        "game_id": game_id,
        "requested_opponent_id": opponent_id,
        "opponent_id": None,
        "snaps": [],
        "omitted_missing_snap_id": 0,
        "omitted_decisions_without_snap_id": 0,
        "invented_snap_identifiers": 0,
        "invented_executed_plays": 0,
    }
    if path is None or not path.is_file():
        report["error"] = "missing_database"
        report["path"] = None if path is None else str(path)
        return report
    report["path"] = str(path)
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        report["error"] = str(exc)
        return report
    conn.row_factory = sqlite3.Row
    try:
        return _export(conn, report, game_id, opponent_id)
    except sqlite3.Error as exc:
        report["error"] = str(exc)
        report["snaps"] = []
        return report
    finally:
        conn.close()


def _export(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    game_id: str,
    opponent_id: str | None,
) -> dict[str, Any]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    session_opponent, session_found = _session_opponent(conn, tables, game_id)
    report["session_found"] = session_found
    report["session_opponent_id"] = session_opponent
    if (
        opponent_id
        and session_opponent
        and str(session_opponent) != str(opponent_id)
    ):
        report["ok"] = False
        report["reason"] = "opponent_mismatch"
        report["snaps"] = []
        return report
    snap_cols = _columns(conn, tables, "snaps")
    if "snaps" not in tables:
        report["ok"] = True
        report["reason"] = "snaps_table_missing"
        return report
    if "session_id" not in snap_cols:
        report["ok"] = True
        report["reason"] = "snaps_not_linked_to_game"
        report["snaps"] = []
        return report
    snap_rows = list(conn.execute(
        "SELECT * FROM snaps WHERE session_id = ?", (game_id,),
    ))
    decisions = _decisions_for_game(conn, tables, game_id)
    by_snap: dict[str, sqlite3.Row] = {}
    for row in decisions:
        snap_id = _text(row, "snap_id")
        if not snap_id:
            report["omitted_decisions_without_snap_id"] += 1
            continue
        by_snap.setdefault(snap_id, row)
    outcome_ids = []
    exported: dict[str, dict[str, Any]] = {}
    for row in snap_rows:
        snap_id = _text(row, "ml_snap_id")
        if not snap_id:
            report["omitted_missing_snap_id"] += 1
            continue
        outcome_ids.append(snap_id)
        exported[snap_id] = _row_from_parts(
            snap_id=snap_id,
            game_id=game_id,
            session_opponent=session_opponent,
            snap=row,
            decision=by_snap.get(snap_id),
        )
    for snap_id, decision in by_snap.items():
        if snap_id in exported:
            continue
        outcome_ids.append(snap_id)
        exported[snap_id] = _row_from_parts(
            snap_id=snap_id,
            game_id=game_id,
            session_opponent=session_opponent,
            snap=None,
            decision=decision,
        )
    outcomes = _outcomes_for(conn, tables, outcome_ids, game_id)
    for snap_id, payload in exported.items():
        _apply_outcome(payload, outcomes.get(snap_id))
    report["snaps"] = sorted(
        exported.values(),
        key=lambda row: (row.get("snap_seq") is None, row.get("snap_seq") or 0, str(row.get("snap_id"))),
    )
    opponents = {row.get("opponent_id") for row in report["snaps"] if row.get("opponent_id")}
    report["opponent_id"] = session_opponent if session_opponent else (next(iter(opponents)) if len(opponents) == 1 else None)
    report["ok"] = True
    report["reason"] = None if report["snaps"] or report["omitted_missing_snap_id"] == 0 else "no_linked_snap_identifiers"
    if not report["snaps"] and report["omitted_missing_snap_id"]:
        report["reason"] = "no_linked_snap_identifiers"
    return report


def _session_opponent(
    conn: sqlite3.Connection, tables: set[str], game_id: str,
) -> tuple[str | None, bool]:
    columns = _columns(conn, tables, "game_sessions")
    if "opponent_id" not in columns or "session_id" not in columns:
        return None, False
    row = conn.execute(
        "SELECT opponent_id FROM game_sessions WHERE session_id = ?",
        (game_id,),
    ).fetchone()
    if row is None:
        return None, False
    return _text(row, "opponent_id"), True


def _decisions_for_game(
    conn: sqlite3.Connection, tables: set[str], game_id: str,
) -> list[sqlite3.Row]:
    columns = _columns(conn, tables, "ml_decisions")
    if "snap_id" not in columns:
        return []
    clauses = []
    params: list[str] = []
    if "game_id" in columns:
        clauses.append("game_id = ?")
        params.append(game_id)
    if "session_id" in columns:
        clauses.append("session_id = ?")
        params.append(game_id)
    if not clauses:
        return []
    sql = f"SELECT * FROM ml_decisions WHERE {' OR '.join(clauses)}"
    return list(conn.execute(sql, params))


def _outcomes_for(
    conn: sqlite3.Connection,
    tables: set[str],
    snap_ids: list[str],
    game_id: str,
) -> dict[str, sqlite3.Row]:
    columns = _columns(conn, tables, "ml_outcomes")
    if "snap_id" not in columns or not snap_ids:
        return {}
    unique = list(dict.fromkeys(snap_ids))
    placeholders = ",".join("?" * len(unique))
    rows = list(conn.execute(
        f"SELECT * FROM ml_outcomes WHERE snap_id IN ({placeholders})",
        unique,
    ))
    found: dict[str, sqlite3.Row] = {}
    for row in rows:
        if "game_id" in columns:
            stored_game = _text(row, "game_id")
            if stored_game and stored_game != str(game_id):
                continue
        snap_id = _text(row, "snap_id")
        if snap_id:
            found[snap_id] = row
    return found


def _row_from_parts(
    *,
    snap_id: str,
    game_id: str,
    session_opponent: str | None,
    snap: sqlite3.Row | None,
    decision: sqlite3.Row | None,
) -> dict[str, Any]:
    snap_opponent = _text(snap, "opponent_id") if snap is not None else None
    if session_opponent and snap_opponent and session_opponent != snap_opponent:
        opponent = None
        opponent_source = None
        opponent_conflict = True
    elif session_opponent:
        opponent = session_opponent
        opponent_source = "game_session"
        opponent_conflict = False
    elif snap_opponent:
        opponent = snap_opponent
        opponent_source = "snap"
        opponent_conflict = False
    else:
        opponent = None
        opponent_source = None
        opponent_conflict = False
    executed_play = _text(snap, "executed_play") if snap is not None else None
    executed_formation = _text(snap, "executed_formation") if snap is not None else None
    executed_status = _text(snap, "executed_status") if snap is not None else None
    executed_verification = _stored_verification(snap)
    logged_play, logged_play_source = _logged(
        executed_play,
        _text(decision, "final_play") if decision is not None else None,
        _text(snap, "play") if snap is not None else None,
    )
    logged_formation, logged_formation_source = _logged(
        executed_formation,
        _text(decision, "final_formation") if decision is not None else None,
        _text(snap, "formation") if snap is not None else None,
    )
    snap_seq = _number(snap, "snap_seq") if snap is not None else None
    if snap_seq is None and decision is not None:
        snap_seq = _number(decision, "snap_seq")
    return {
        "snap_id": snap_id,
        "game_id": game_id,
        "session_id": game_id,
        "opponent_id": opponent,
        "opponent_source": opponent_source,
        "opponent_conflict": opponent_conflict,
        "snap_seq": snap_seq,
        "down": _number(snap, "down") if snap is not None else None,
        "distance": _number(snap, "distance") if snap is not None else None,
        "quarter": _number(snap, "quarter") if snap is not None else None,
        "clock_seconds": _number(snap, "clock_seconds") if snap is not None else None,
        "logged_formation": logged_formation,
        "logged_formation_source": logged_formation_source,
        "logged_play": logged_play,
        "logged_play_source": logged_play_source,
        "executed_formation": executed_formation,
        "executed_play": executed_play,
        "executed_status": executed_status,
        "executed_verification": executed_verification,
    }


def _apply_outcome(payload: dict[str, Any], outcome: sqlite3.Row | None) -> None:
    if outcome is None:
        return
    for key in ("executed_play", "executed_formation", "executed_status"):
        if payload.get(key) is None:
            payload[key] = _text(outcome, key)
    if payload.get("executed_verification") is None:
        payload["executed_verification"] = _stored_verification(outcome)
    if payload.get("logged_play") is None and payload.get("executed_play"):
        payload["logged_play"] = payload["executed_play"]
        payload["logged_play_source"] = "executed"
    if payload.get("logged_formation") is None and payload.get("executed_formation"):
        payload["logged_formation"] = payload["executed_formation"]
        payload["logged_formation_source"] = "executed"


def _logged(
    executed: str | None, decision: str | None, snap_value: str | None,
) -> tuple[str | None, str | None]:
    if executed:
        return executed, "executed"
    if decision:
        return decision, "decision"
    if snap_value:
        return snap_value, "snap"
    return None, None


def _columns(conn: sqlite3.Connection, tables: set[str], table: str) -> set[str]:
    if table not in tables:
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _text(row: sqlite3.Row | None, key: str) -> str | None:
    if row is None or key not in row.keys():
        return None
    value = row[key]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(row: sqlite3.Row | None, key: str) -> int | float | None:
    if row is None or key not in row.keys() or row[key] is None:
        return None
    value = row[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text) if text.isdigit() else float(text)
    except (TypeError, ValueError):
        return None


def _stored_verification(row: sqlite3.Row | None) -> str | None:
    """A missing verification column stays null. It is not treated as verified."""
    return _text(row, "executed_verification")


def log_snaps_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    """Accept the export array or the report object written by an older file."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        snaps = payload.get("snaps")
        if isinstance(snaps, list):
            return [row for row in snaps if isinstance(row, dict)]
    return []
