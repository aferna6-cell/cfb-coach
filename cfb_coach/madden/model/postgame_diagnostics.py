"""Postgame third-down and interception diagnostics from logged fields only.

Read-only. Every value comes from ``snaps`` and the linked ``ml_decisions``
row. A field that was not logged is reported as ``None``; nothing is inferred
from the result text beyond the outcome kind (``int``, ``td``, yards).
"""
from __future__ import annotations

import json
from typing import Any

from cfb_coach.outcome import parse_outcome

DISTANCE_BUCKETS = (("short", 1, 3), ("medium", 4, 6), ("long", 7, 10), ("very_long", 11, 99))


def distance_bucket(distance: Any) -> str | None:
    try:
        d = int(distance)
    except (TypeError, ValueError):
        return None
    for name, lo, hi in DISTANCE_BUCKETS:
        if lo <= d <= hi:
            return name
    return None


def _columns(conn: Any, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def _converted(result: Any, distance: Any) -> bool | None:
    text = str(result or "").strip()
    if not text:
        return None
    oc = parse_outcome(text)
    if oc.kind == "td" or oc.kind == "convert":
        return True
    if oc.kind in ("int", "fumble", "incomplete", "sack", "stop"):
        return False
    if oc.yards is None:
        return None
    try:
        return int(oc.yards) >= int(distance)
    except (TypeError, ValueError):
        return None


def _decision(conn: Any, decision_id: Any) -> dict[str, Any] | None:
    if decision_id is None or "id" not in _columns(conn, "ml_decisions"):
        return None
    row = conn.execute("SELECT * FROM ml_decisions WHERE id = ?", (decision_id,)).fetchone()
    if row is None:
        return None
    keys = [d[0] for d in conn.execute("SELECT * FROM ml_decisions LIMIT 0").description]
    rec = dict(zip(keys, row))
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(rec.get("decision_json") or "{}") or {}
    except (TypeError, ValueError):
        payload = {}
    exp = payload.get("experimental_offense") or {}
    return {
        "mode": rec.get("mode"),
        "policy_source": rec.get("policy_source"),
        "model_pick": _pick(rec.get("final_formation"), rec.get("final_play")),
        "heuristic_pick": _pick(rec.get("heuristic_formation"), rec.get("heuristic_play")),
        "agree": None if rec.get("agree") is None else bool(rec.get("agree")),
        "confidence": exp.get("probability") if isinstance(exp, dict) else None,
        "uncertainty": exp.get("uncertainty") if isinstance(exp, dict) else None,
        "fell_back": payload.get("fell_back"),
    }


def _pick(formation: Any, play: Any) -> str | None:
    if not formation and not play:
        return None
    return f"{formation or '?'} — {play or '?'}"


def session_diagnostics(conn: Any, session_id: str) -> dict[str, Any]:
    """Third-down conversion by distance bucket, every INT, and agreement."""
    cols = _columns(conn, "snaps")
    if "session_id" not in cols:
        return {"session_id": session_id, "error": "snaps has no session_id"}
    want = [c for c in (
        "id", "side", "down", "distance", "yardline", "quarter", "result", "formation",
        "play", "coverage_seen", "ml_decision_id", "drive_id", "game_clock",
    ) if c in cols]
    rows = [
        dict(zip(want, r)) for r in conn.execute(
            f"SELECT {', '.join(want)} FROM snaps WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    ]
    offense = [r for r in rows if str(r.get("side") or "offense").startswith("o")]
    third: dict[str, dict[str, Any]] = {}
    for r in offense:
        if r.get("down") != 3:
            continue
        bucket = distance_bucket(r.get("distance")) or "unknown"
        slot = third.setdefault(bucket, {"attempts": 0, "converted": 0, "unknown": 0, "calls": []})
        slot["attempts"] += 1
        conv = _converted(r.get("result"), r.get("distance"))
        if conv is None:
            slot["unknown"] += 1
        elif conv:
            slot["converted"] += 1
        dec = _decision(conn, r.get("ml_decision_id"))
        slot["calls"].append({
            "distance": r.get("distance"),
            "formation": r.get("formation"),
            "play": r.get("play"),
            "result": r.get("result"),
            "converted": conv,
            "model_pick": (dec or {}).get("model_pick"),
            "heuristic_pick": (dec or {}).get("heuristic_pick"),
            "confidence": (dec or {}).get("confidence"),
        })
    ints = []
    for r in offense:
        if parse_outcome(str(r.get("result") or "")).kind != "int":
            continue
        dec = _decision(conn, r.get("ml_decision_id")) or {}
        ints.append({
            "snap_row": r.get("id"),
            "quarter": r.get("quarter"),
            "down": r.get("down"),
            "distance": r.get("distance"),
            "yardline": r.get("yardline"),
            "formation": r.get("formation"),
            "play": r.get("play"),
            "coverage": r.get("coverage_seen"),
            "model_pick": dec.get("model_pick"),
            "heuristic_pick": dec.get("heuristic_pick"),
            "agree": dec.get("agree"),
            "confidence": dec.get("confidence"),
            "mode": dec.get("mode"),
        })
    agree = {"decisions": 0, "agree": 0, "disagree": 0, "unknown": 0}
    if "game_id" in _columns(conn, "ml_decisions") or "session_id" in _columns(conn, "ml_decisions"):
        key = "session_id" if "session_id" in _columns(conn, "ml_decisions") else "game_id"
        for (value,) in conn.execute(
            f"SELECT agree FROM ml_decisions WHERE {key} = ?", (session_id,)
        ).fetchall():
            agree["decisions"] += 1
            if value is None:
                agree["unknown"] += 1
            elif int(value):
                agree["agree"] += 1
            else:
                agree["disagree"] += 1
    return {
        "session_id": session_id,
        "offense_snaps": len(offense),
        "third_down": third,
        "interceptions": ints,
        "model_vs_heuristic": agree,
    }


def format_diagnostics(diag: dict[str, Any]) -> str:
    lines = [f"## Diagnostics — session {diag.get('session_id')}"]
    third = diag.get("third_down") or {}
    lines.append("Third down by distance (converted/attempts, unknown result):")
    if not third:
        lines.append("  (no third-down snaps logged)")
    for name, _lo, _hi in DISTANCE_BUCKETS + (("unknown", 0, 0),):
        slot = third.get(name)
        if not slot:
            continue
        lines.append(f"  {name}: {slot['converted']}/{slot['attempts']} (unknown {slot['unknown']})")
        for c in slot["calls"]:
            conf = c.get("confidence")
            conf_s = "n/a" if conf is None else f"{float(conf):.3f}"
            lines.append(
                f"    3&{c.get('distance')}: {c.get('formation')} — {c.get('play')} -> {c.get('result')}"
                f" | model {c.get('model_pick') or 'n/a'} | heuristic {c.get('heuristic_pick') or 'n/a'}"
                f" | p {conf_s}"
            )
    ints = diag.get("interceptions") or []
    lines.append(f"Interceptions: {len(ints)}")
    for i in ints:
        conf = i.get("confidence")
        conf_s = "n/a" if conf is None else f"{float(conf):.3f}"
        lines.append(
            f"  Q{i.get('quarter')} {i.get('down')}&{i.get('distance')}: {i.get('formation')} — {i.get('play')}"
            f" vs {i.get('coverage') or 'coverage not logged'} | model {i.get('model_pick') or 'n/a'}"
            f" | heuristic {i.get('heuristic_pick') or 'n/a'} | p {conf_s}"
        )
    a = diag.get("model_vs_heuristic") or {}
    lines.append(
        f"Model vs heuristic: agree {a.get('agree', 0)}/{a.get('decisions', 0)}"
        f" (disagree {a.get('disagree', 0)}, unknown {a.get('unknown', 0)})"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sqlite3

    ap = argparse.ArgumentParser(description="Read-only postgame diagnostics")
    ap.add_argument("db")
    ap.add_argument("session_id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    diag = session_diagnostics(conn, args.session_id)
    print(json.dumps(diag, indent=2, default=str) if args.json else format_diagnostics(diag))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
