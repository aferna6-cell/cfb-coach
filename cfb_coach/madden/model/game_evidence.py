"""Read-only game evidence for the pregame playbook architect.

Three kinds of evidence, kept apart on purpose:
  - Game results, with owner-reported score corrections applied on read.
  - Caller attribution per session (which system called offense and defense).
  - Formation outcomes from identified, user-verified executions only.

Defensive snaps without a verified execution still tell us what the opponent
did in a situation (yards on 3rd down, red-zone results). That is situation
evidence; it is never attached to a defensive call as a training label.
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any, Mapping, Sequence

from cfb_coach.madden.data import _data_path
from cfb_coach.madden.model.experimental_model import _converted_offense

# Beta prior strength for formation success rates. Eight snaps of evidence
# move a formation halfway from the overall rate.
SHRINK_SNAPS = 8.0
REPEATED_FAILURE_MIN = 3


@lru_cache(maxsize=1)
def _corrections() -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(_data_path("result_corrections.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(raw.get("corrections") or {})


def _rows(db: Any, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
    if db is None or not hasattr(db, "conn"):
        return []
    try:
        return [dict(r) for r in db.conn.execute(sql, tuple(args)).fetchall()]
    except Exception:  # noqa: BLE001 - older databases lack some tables
        return []


def corrected_result(row: Mapping[str, Any]) -> dict[str, Any]:
    stored = str(row.get("score") or "") or None
    fix = _corrections().get(str(row.get("session_id") or ""))
    if fix and stored == fix.get("stored_score"):
        return {"score": fix["score"], "result_wl": fix.get("result_wl") or row.get("result_wl"),
                "stored_score": stored, "corrected": True, "correction_reason": fix.get("reason")}
    return {"score": stored, "result_wl": row.get("result_wl"), "stored_score": stored,
            "corrected": False, "correction_reason": None}


def caller_attribution(db: Any, session_id: str) -> dict[str, str]:
    """Which system called each side. Defense decisions carry a defense policy version."""
    rows = _rows(db, "SELECT mode, policy_source, policy_version, COUNT(*) AS n FROM ml_decisions "
                     "WHERE session_id=? GROUP BY 1,2,3", (session_id,))
    out = {"offense": "", "defense": ""}
    for side in out:
        mine = [r for r in rows if ("defense" in str(r.get("policy_version") or "").lower()) == (side == "defense")]
        model = [r for r in mine if str(r.get("policy_source") or "") == "model"]
        if model:
            top = max(model, key=lambda r: int(r["n"]))
            out[side] = f"ML {top['mode']} ({top['policy_version']}, {sum(int(r['n']) for r in model)} decisions)"
        else:
            out[side] = "heuristic coach (no ML decision rows)"
    return out


def recent_games(db: Any, *, opponent_id: str | None = None, limit: int = 8,
                 before_ts: str | None = None) -> list[dict[str, Any]]:
    sql = ("SELECT session_id, opponent_id, started_ts, ended_ts, result_wl, score, snaplog_score, "
           "score_reconcile, play_count FROM game_sessions WHERE score IS NOT NULL")
    args: list[Any] = []
    if opponent_id:
        sql += " AND opponent_id=?"
        args.append(opponent_id)
    if before_ts:
        sql += " AND started_ts < ?"
        args.append(before_ts)
    sql += " ORDER BY started_ts DESC LIMIT ?"
    args.append(int(limit))
    out = []
    for row in _rows(db, sql, args):
        sid = str(row["session_id"])
        sides = {r["side"]: r for r in _rows(
            db, "SELECT side, COUNT(*) AS n, SUM(CASE WHEN executed_status='identified' AND "
                "executed_verification='verified' THEN 1 ELSE 0 END) AS verified FROM snaps "
                "WHERE session_id=? GROUP BY side", (sid,))}
        usage = Counter(r["formation"] for r in _rows(
            db, "SELECT formation FROM snaps WHERE session_id=? AND side='defense' AND formation IS NOT NULL",
            (sid,)))
        out.append({
            "session_id": sid,
            "opponent_id": row.get("opponent_id"),
            "started_ts": row.get("started_ts"),
            **corrected_result(row),
            "snaplog_score": row.get("snaplog_score"),
            "callers": caller_attribution(db, sid),
            "offense_snaps": int((sides.get("offense") or {}).get("n") or 0),
            "offense_verified": int((sides.get("offense") or {}).get("verified") or 0),
            "defense_snaps": int((sides.get("defense") or {}).get("n") or 0),
            "defense_verified": int((sides.get("defense") or {}).get("verified") or 0),
            "defense_formation_calls": dict(usage.most_common()),
        })
    return out


def _scope_ids(opponent_id: str) -> list[str]:
    from cfb_coach.opponents import is_cpu_opponent

    if is_cpu_opponent(opponent_id):
        return ["cpu", "cpu_dynasty", "notre_dame"]
    return [opponent_id]


def verified_offense_rows(db: Any, opponent_id: str | None, *, before_ts: str | None = None) -> list[dict[str, Any]]:
    """Identified, user-verified offensive executions with a result. ``None`` means every opponent."""
    sql = ("SELECT s.id, s.session_id, s.opponent_id, s.executed_formation, s.executed_play, s.result, "
           "s.yards, s.down, s.distance, s.yardline, s.ts FROM snaps s WHERE s.side='offense' "
           "AND s.executed_status='identified' AND s.executed_verification='verified' "
           "AND s.executed_formation IS NOT NULL AND s.executed_play IS NOT NULL AND s.result IS NOT NULL")
    args: list[Any] = []
    if opponent_id:
        ids = _scope_ids(opponent_id)
        sql += f" AND s.opponent_id IN ({','.join('?' * len(ids))})"
        args += ids
    if before_ts:
        sql += " AND s.ts < ?"
        args.append(before_ts)
    return _rows(db, sql + " ORDER BY s.id", args)


def _situation_of(row: Mapping[str, Any]) -> str:
    try:
        down = int(row.get("down") or 0)
        dist = int(row.get("distance") or 10)
        yl = int(row.get("yardline") or 0)
    except (TypeError, ValueError):
        return "normal"
    if yl >= 95:
        return "goal_line"
    if yl >= 80:
        return "red_zone"
    if 0 < yl <= 10:
        return "backed_up"
    if down >= 3 and dist <= 2:
        return "short_yardage"
    if down >= 3 and dist >= 7:
        return "third_and_long"
    return "normal"


def offense_evidence(db: Any, opponent_id: str, *, before_ts: str | None = None) -> dict[str, Any]:
    """Shrunk per-formation success, situation need, and repeated play failures."""
    scoped = verified_offense_rows(db, opponent_id, before_ts=before_ts)
    every = verified_offense_rows(db, None, before_ts=before_ts)
    labeled = [(r, _converted_offense(r)) for r in scoped]
    labeled = [(r, ok) for r, ok in labeled if ok is not None]
    base_all = [ok for ok in (_converted_offense(r) for r in every) if ok is not None]
    prior = (sum(base_all) / len(base_all)) if base_all else 0.4
    forms: dict[str, list[bool]] = {}
    plays: dict[tuple[str, str], list[bool]] = {}
    sits: dict[str, list[bool]] = {}
    for row, ok in labeled:
        forms.setdefault(str(row["executed_formation"]), []).append(ok)
        plays.setdefault((str(row["executed_formation"]), str(row["executed_play"])), []).append(ok)
        sits.setdefault(_situation_of(row), []).append(ok)

    def shrunk(vals: list[bool]) -> float:
        return (sum(vals) + prior * SHRINK_SNAPS) / (len(vals) + SHRINK_SNAPS)

    formation = {
        name: {"n": len(v), "successes": int(sum(v)), "rate": round(shrunk(v), 4),
               "delta": round(shrunk(v) - prior, 4)}
        for name, v in forms.items()
    }
    failing = []
    for (form, play), v in plays.items():
        fails = len(v) - sum(v)
        if fails >= REPEATED_FAILURE_MIN and shrunk(v) < prior - 0.08:
            failing.append({"formation": form, "play": play, "n": len(v), "fails": fails,
                            "rate": round(shrunk(v), 3)})
    situation = {
        name: {"n": len(v), "rate": round(sum(v) / len(v), 3), "shrunk": round(shrunk(v), 3)}
        for name, v in sits.items()
    }
    return {
        "scope": _scope_ids(opponent_id) if opponent_id else ["all"],
        "verified_rows": len(scoped),
        "labeled_rows": len(labeled),
        "prior_success": round(prior, 4),
        "formation": formation,
        "repeated_failures": sorted(failing, key=lambda r: (r["rate"], -r["n"])),
        "situation": situation,
        "label": "first-down conversion (or TD) on identified, user-verified executions",
    }


def _defense_situation(row: Mapping[str, Any]) -> str:
    try:
        down = int(row.get("down") or 0)
        dist = int(row.get("distance") or 10)
        yl = int(row.get("yardline") or 0)
    except (TypeError, ValueError):
        return "early"
    # Stored yardline is the opponent's field position from our logging convention.
    if yl >= 95:
        return "goal_line"
    if yl >= 80:
        return "red_zone"
    if down in (2, 3, 4) and dist <= 3:
        return "short"
    if down in (3, 4) and dist >= 7:
        return "long"
    if down in (3, 4):
        return "medium"
    return "early"


def defense_evidence(db: Any, opponent_id: str, *, before_ts: str | None = None) -> dict[str, Any]:
    """Opponent tendencies, situation results against us, and our call concentration."""
    from cfb_coach.madden.model.opponent_tendencies import build_defense_profile

    sql = ("SELECT session_id, formation, play, down, distance, yardline, result, yards, concept_seen, "
           "executed_status, executed_verification FROM snaps WHERE side='defense' AND opponent_id=?")
    args: list[Any] = [opponent_id]
    if before_ts:
        sql += " AND ts < ?"
        args.append(before_ts)
    rows = _rows(db, sql, args)
    verified = [r for r in rows if str(r.get("executed_status") or "") == "identified"
                and str(r.get("executed_verification") or "") == "verified"]
    sits: dict[str, list[bool]] = {}
    for row in rows:
        ok = _converted_offense(row)
        if ok is not None:
            sits.setdefault(_defense_situation(row), []).append(ok)
    allowed = [ok for v in sits.values() for ok in v]
    base = (sum(allowed) / len(allowed)) if allowed else None
    by_session: dict[str, Counter[str]] = {}
    for row in rows:
        if row.get("formation"):
            by_session.setdefault(str(row["session_id"]), Counter())[str(row["formation"])] += 1
    concentration = []
    for sid, counts in by_session.items():
        total = sum(counts.values())
        form, n = counts.most_common(1)[0]
        concentration.append({"session_id": sid, "calls": total, "top_formation": form, "top_count": n,
                              "top_share": round(n / total, 3), "formations": dict(counts.most_common())})
    profile = build_defense_profile(db, opponent_id) if db is not None else {}
    stops: dict[str, list[bool]] = {}
    for row in verified:
        ok = _converted_offense(row)
        if ok is not None and row.get("formation"):
            stops.setdefault(str(row["formation"]), []).append(not ok)
    stop_prior = 1.0 - base if base is not None else 0.6
    formation = {}
    for name, v in stops.items():
        rate = (sum(v) + stop_prior * SHRINK_SNAPS) / (len(v) + SHRINK_SNAPS)
        formation[name] = {"n": len(v), "rate": round(rate, 4), "delta": round(rate - stop_prior, 4)}
    return {
        "snaps": len(rows),
        "verified_executions": len(verified),
        "formation": formation,
        "opponent_success_allowed": round(base, 3) if base is not None else None,
        "situation_allowed": {k: {"n": len(v), "rate": round(sum(v) / len(v), 3)} for k, v in sits.items()},
        "call_concentration": sorted(concentration, key=lambda r: -r["top_share"]),
        "opponent_profile": {k: profile.get(k) for k in ("n", "shrink", "run_share", "concepts", "summary")},
        "labels": {
            "situation_allowed": "opponent conversion by situation from logged results; the defensive call "
                                 "executed is unverified, so these never label a call",
            "call_concentration": "logged defensive calls (recommendations; execution unverified)",
        },
    }


__all__ = [
    "caller_attribution", "corrected_result", "defense_evidence", "offense_evidence",
    "recent_games", "verified_offense_rows",
]
