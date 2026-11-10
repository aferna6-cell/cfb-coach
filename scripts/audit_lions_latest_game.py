#!/usr/bin/env python3
"""Read-only audit of the latest Lions legacy-coordinator game.

No replay, training, migration, or model/book mutations. The most recent game
with logged snaps is selected unless --game-id is given. Prints honest counts
even when the game was not marked complete or an outcome was not recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.madden.model.offense_postgame import evaluate_saved_games
from cfb_coach.outcome import parse_outcome

DEFAULT_DB = Path.home() / "lions-retrained-legacy" / "madden27-lions.db"


def _get(row: sqlite3.Row | dict | None, name: str, default=None):
    if row is None:
        return default
    try:
        return row[name] if name in row.keys() else default
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def _tables(db: CoachDB) -> set[str]:
    return {r[0] for r in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}


def _recent_sessions(db: CoachDB, limit: int = 8) -> list[dict[str, Any]]:
    if not {"game_sessions", "snaps"}.issubset(_tables(db)):
        return []
    rows = db.conn.execute(
        "SELECT gs.*, (SELECT COUNT(*) FROM snaps WHERE session_id = gs.session_id) "
        "AS logged_snaps FROM game_sessions gs "
        "ORDER BY gs.started_ts DESC, gs.rowid DESC LIMIT ?", (limit * 3,)
    ).fetchall()
    return [
        {"game_id": _get(r, "session_id"),
         "opponent": _get(r, "opponent_id"),
         "started": _get(r, "started_ts"),
         "ended": _get(r, "ended_ts"),
         "score": _get(r, "score"),
         "result": _get(r, "result_wl"),
         "snaps": _get(r, "logged_snaps")}
        for r in rows if (_get(r, "logged_snaps") or 0) > 0
    ][:limit]


def _conversion(snap: sqlite3.Row) -> bool | None:
    result = str(_get(snap, "result") or "")
    if not result:
        return None
    parsed = parse_outcome(result)
    if parsed.kind in {"td", "convert"}:
        return True
    if parsed.kind in {"incomplete", "int", "fumble", "sack", "stop"}:
        return False
    yards = _get(snap, "yards")
    dist = _get(snap, "distance")
    if yards is None or dist is None:
        return None
    try:
        return int(yards) >= int(dist)
    except (ValueError, TypeError):
        return None


def _summarize_snaps(rows: list[sqlite3.Row], side: str) -> dict[str, Any]:
    data = [r for r in rows
            if str(_get(r, "side") or "").lower().startswith(side[0])]
    completed = [r for r in data if str(_get(r, "result") or "").strip()]
    unknown = len(data) - len(completed)
    plays = [
        (str(_get(r, "formation") or ""), str(_get(r, "play") or ""))
        for r in data if _get(r, "formation") and _get(r, "play")
    ]
    known = [int(_get(r, "yards")) for r in completed if _get(r, "yards") is not None]
    outcome_counts = Counter(parse_outcome(str(_get(r, "result"))).kind for r in completed)
    by_formation = Counter(f for f, p in plays)
    by_play = Counter(f"{f} — {p}" for f, p in plays)
    verified = sum(
        1 for r in completed
        if str(_get(r, "executed_verification") or "") == "verified"
        and str(_get(r, "executed_status") or "") == "identified"
    )
    third = [r for r in completed if _get(r, "down") == 3]
    third_known = [_conversion(r) for r in third]
    third_known = [value for value in third_known if value is not None]
    drives = {str(_get(r, "drive_id")) for r in data if _get(r, "drive_id")}
    adjacent_repeats = sum(1 for old, new in zip(plays, plays[1:]) if old == new)
    short_rotation = sum(
        1 for i in range(2, len(plays)) if plays[i] == plays[i - 2]
    )
    long_rotation = sum(
        1 for i in range(3, len(plays)) if plays[i] == plays[i - 3]
    )
    return {
        "side": side,
        "logged_snaps": len(data),
        "result_recorded": len(completed),
        "results_missing": unknown,
        "verified_executions": verified,
        "distinct_formation_play_pairs": len(set(plays)),
        "distinct_formations": len(by_formation),
        "formations": dict(by_formation.most_common()),
        "most_called_plays": dict(by_play.most_common(12)),
        "same_exact_play_consecutive": adjacent_repeats,
        "same_exact_play_two_calls_apart": short_rotation,
        "same_exact_play_three_calls_apart": long_rotation,
        "outcomes": dict(outcome_counts),
        "yards_known_plays": len(known),
        "yards_known_total": sum(known),
        "yards_per_recorded_yardage_play": round(sum(known)/len(known), 2) if known else None,
        "gains_or_yards_allowed_15_plus": sum(n >= 15 for n in known),
        "negative_yardage": sum(n < 0 for n in known),
        "third_down_results": len(third),
        "third_down_conversion_known": len(third_known),
        "third_down_conversions": sum(third_known),
        "drives_with_id": len(drives),
        "note": (
            "For defense, yards and third-down conversions are CPU offensive "
            "gains/conversions (bad for our defense). For offense they are ours. "
            "Touchdown and turnover plays can lack yardage, so yards/play only "
            "covers rows with recorded numeric yards. Repetitions count "
            "recommendations, not necessarily verified executions."
        ),
    }


def _model_audit(db: CoachDB, game_id: str) -> dict[str, Any]:
    if "ml_decisions" not in _tables(db):
        return {"status": "missing ml_decisions table"}
    rows = db.conn.execute(
        "SELECT * FROM ml_decisions WHERE game_id=? OR session_id=? "
        "ORDER BY id ASC", (game_id, game_id)
    ).fetchall()
    modes = Counter(str(_get(r, "mode") or "unknown") for r in rows)
    policy = Counter(str(_get(r, "policy_source") or "unknown") for r in rows)
    status = Counter(str(_get(r, "shadow_status") or "unknown") for r in rows)
    versions = Counter(str(_get(r, "model_version") or "none") for r in rows)
    experiment = [r for r in rows if _get(r, "mode") == "experimental"]
    non_ok = [r for r in experiment
              if str(_get(r, "shadow_status") or "").lower() != "ok"]
    values = [float(_get(r, "latency_ms_model")) for r in experiment
              if _get(r, "latency_ms_model") is not None]
    return {
        "decision_rows": len(rows),
        "modes": dict(modes),
        "policy_sources": dict(policy),
        "model_versions_recorded": dict(versions),
        "statuses": dict(status),
        "experimental_decisions": len(experiment),
        "experimental_non_ok": len(non_ok),
        "non_ok_details": [
            {"status": _get(r, "shadow_status"),
             "final": f"{_get(r, 'final_formation')} — {_get(r, 'final_play')}",
             "heuristic": f"{_get(r, 'heuristic_formation')} — {_get(r, 'heuristic_play')}"}
            for r in non_ok[:12]
        ],
        "latency_ms_mean": round(sum(values)/len(values), 2) if values else None,
        "latency_ms_max": round(max(values), 2) if values else None,
        "important": (
            "Experimental mode alone does not prove the ML selected the play. "
            "Check model versions and non-OK statuses for fallback."
        ),
    }


def audit(db_path: Path, game_id: str | None = None) -> dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(f"Cannot read Lions DB: {db_path}")
    db = CoachDB.open_read_only(db_path)
    try:
        check = db.conn.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise ValueError(f"DB integrity check failed: {check}")
        recent = _recent_sessions(db)
        if game_id is None:
            if not recent:
                return {"status": "no sessions with logged snaps", "database": str(db_path)}
            selected = recent[0]["game_id"]
        else:
            selected = game_id
        stored = next((s for s in recent if s["game_id"] == selected), None)
        if stored is None and "game_sessions" in _tables(db):
            row = db.conn.execute(
                "SELECT * FROM game_sessions WHERE session_id=?", (selected,)
            ).fetchone()
            if row:
                stored = {
                    "game_id": selected, "opponent": _get(row, "opponent_id"),
                    "started": _get(row, "started_ts"),
                    "ended": _get(row, "ended_ts"),
                    "score": _get(row, "score"),
                    "result": _get(row, "result_wl")}
        snaps = db.conn.execute(
            "SELECT * FROM snaps WHERE session_id=? ORDER BY id ASC",
            (selected,)
        ).fetchall()
        if not snaps:
            return {"status": "no snaps for game", "game_id": selected,
                    "recent_games": recent}
        try:
            existing_offense_report = evaluate_saved_games(
                db, game_id=selected, limit=1
            )["games"][0]
        except (IndexError, KeyError, TypeError, sqlite3.Error) as error:
            existing_offense_report = {"unavailable": str(error)}
        return {
            "audit": "Lions legacy; read-only postgame",
            "database": str(db_path),
            "database_integrity": "ok",
            "game_id": selected,
            "session": stored,
            "finished": bool(stored and stored.get("ended")),
            "recent_games": recent[:4],
            "offense": _summarize_snaps(snaps, "offense"),
            "defense": _summarize_snaps(snaps, "defense"),
            "offensive_model_decisions": _model_audit(db, selected),
            "existing_offense_evaluation": existing_offense_report,
            "history_modified": False,
        }
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help="The Lions retrained-legacy database; read-only")
    p.add_argument("--game-id", default=None,
                   help="Select an older specific game ID instead of most recent")
    p.add_argument("--compact", action="store_true",
                   help="Keep only score, snap totals, play diversity, repeats, and model health")
    args = p.parse_args()
    try:
        report = audit(args.db.expanduser().resolve(), args.game_id)
    except (FileNotFoundError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"AUDIT ERROR: {exc}")
        return 2
    if args.compact and "offense" in report:
        report["existing_offense_evaluation"] = {
            k: v for k, v in report["existing_offense_evaluation"].items()
            if k in ("third_down", "red_zone", "prediction_vs_outcome",
                     "fallback_count", "inventory", "missing_data",
                     "verified_executed_offensive_plays")
        }
        report["recent_games"] = report["recent_games"][:2]
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
