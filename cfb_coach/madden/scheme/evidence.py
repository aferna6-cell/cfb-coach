"""Variant-level logging and the rule for what counts as variant evidence.

Additive tables (created only on a writable live DB, never by training / evaluation):
  civil_scheme_decisions   one row per civil call shown: variant id, base play, exact adjustments, opponent,
                           game (session) id, pre-snap situation, pre-snap belief, model version, ranking.
  civil_scheme_executions  owner report of what was actually run when it was NOT the full variant:
                           applied / skipped / changed adjustments (``ml civil-scheme report-adjustments``).
The outcome itself (result, yards, coverage seen afterwards, whether the call was followed) stays in the
existing ``snaps`` row, linked by ``recommended_adjustment_id = 'scheme:<variant_id>'``.

Evidence rule:
  * variant evidence   = a closed snap whose executed formation/play is the variant's, executed_status
                         'identified', AND executed_adjustment_id == 'scheme:<id>' (the adjustment confirmed),
                         OR an owner execution report for that snap. A report's skipped / changed
                         adjustments are preserved: the design is rebuilt from the applied subset only.
  * base-play evidence = the same snap with no adjustment confirmed and no report: NOT variant evidence and
                         excluded from training (what was run is unknown).
  * a recommendation that was not executed is never evidence of anything.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden import catalog as madden_catalog
from cfb_coach.madden.scheme.catalog import SchemeCatalog

ADJ_PREFIX = "scheme:"

DDL = (
    """CREATE TABLE IF NOT EXISTS civil_scheme_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, session_id TEXT, opponent_id TEXT,
        mode TEXT NOT NULL, status TEXT NOT NULL, variant_id TEXT, formation TEXT, base_play TEXT,
        adjustments_json TEXT, down INTEGER, distance INTEGER, yardline INTEGER, situation_raw TEXT,
        belief_json TEXT, prediction_json TEXT, ranking_json TEXT, model_version TEXT, catalog_version TEXT,
        explanation TEXT, snap_floor_id INTEGER)""",
    """CREATE TABLE IF NOT EXISTS civil_scheme_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, snap_id INTEGER NOT NULL, variant_id TEXT NOT NULL,
        applied_json TEXT NOT NULL, skipped_json TEXT NOT NULL, changed_json TEXT NOT NULL, note TEXT,
        source TEXT NOT NULL DEFAULT 'owner_cli')""",
)


def ensure_tables(conn: Any) -> None:
    for d in DDL:
        conn.execute(d)
    conn.commit()


def _has_table(conn: Any, name: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def log_decision(conn: Any, rec: Mapping[str, Any]) -> int | None:
    ensure_tables(conn)
    floor = conn.execute("SELECT COALESCE(MAX(id), 0) FROM snaps").fetchone()[0]
    cur = conn.execute(
        "INSERT INTO civil_scheme_decisions (snap_floor_id, ts, session_id, opponent_id, mode, status, variant_id, formation, base_play,"
        " adjustments_json, down, distance, yardline, situation_raw, belief_json, prediction_json, ranking_json,"
        " model_version, catalog_version, explanation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (floor, time.strftime("%Y-%m-%dT%H:%M:%S%z"), rec.get("session_id"), rec.get("opponent_id"), rec["mode"],
         rec["status"], rec.get("variant_id"), rec.get("formation"), rec.get("base_play"),
         json.dumps(rec.get("adjustments") or []), rec.get("down"), rec.get("distance"), rec.get("yardline"),
         rec.get("situation_raw"), json.dumps(rec.get("belief") or {}), json.dumps(rec.get("prediction") or {}),
         json.dumps(rec.get("ranking") or []), rec.get("model_version"), rec.get("catalog_version"),
         rec.get("explanation")))
    conn.commit()
    return cur.lastrowid


def report_execution(conn: Any, cat: SchemeCatalog, *, snap_id: int, applied: Sequence[int] = (),
                     skipped: Sequence[int] = (), changed: Mapping[int, str] | None = None, note: str = "") -> dict[str, Any]:
    """Owner report: which of the variant's adjustments (1-based on the CLI, 0-based here) were really run."""
    row = conn.execute("SELECT recommended_adjustment_id, result FROM snaps WHERE id = ?", (snap_id,)).fetchone()
    if row is None:
        raise ValueError(f"snap {snap_id} not found")
    rec_id = str(row[0] or "")
    if not rec_id.startswith(ADJ_PREFIX):
        raise ValueError(f"snap {snap_id} was not a civil-scheme recommendation")
    vid = rec_id[len(ADJ_PREFIX):]
    v = cat.get(vid)
    if v is None:
        raise ValueError(f"{vid} is not a catalog variant")
    n = len(v.adjustments)
    changed = dict(changed or {})
    allidx = set(applied) | set(skipped) | set(changed)
    if any(i < 0 or i >= n for i in allidx):
        raise ValueError(f"adjustment numbers must be 1..{n}")
    if set(applied) & (set(skipped) | set(changed)) or set(skipped) & set(changed):
        raise ValueError("an adjustment can be applied, skipped or changed, not two of these")
    # Unlisted adjustments are UNKNOWN, not applied.
    unknown = sorted(set(range(n)) - allidx)
    ensure_tables(conn)
    conn.execute("INSERT INTO civil_scheme_executions (ts, snap_id, variant_id, applied_json, skipped_json, changed_json,"
                 " note) VALUES (?,?,?,?,?,?,?)",
                 (time.strftime("%Y-%m-%dT%H:%M:%S%z"), snap_id, vid, json.dumps(sorted(applied)),
                  json.dumps(sorted(skipped)), json.dumps({str(k): v for k, v in changed.items()}), note))
    conn.commit()
    return {"snap_id": snap_id, "variant_id": vid, "applied": sorted(applied), "skipped": sorted(skipped),
            "changed": changed, "unknown": unknown}


@dataclass
class VariantEvidence:
    snap_id: int
    snap_key: str
    session_id: str
    variant_id: str
    applied: tuple[int, ...]            # adjustment indices actually run
    complete: bool                      # every adjustment applied as written
    usable: bool                        # may enter training (no changed / unknown adjustments)
    source: str


def _n(s: Any) -> str:
    return madden_catalog.norm(str(s or ""))


def variant_evidence(conn: Any, cat: SchemeCatalog) -> list[VariantEvidence]:
    """Read-only SELECTs. Returns one entry per closed snap that is verified evidence about a variant."""
    reports: dict[int, tuple] = {}
    if _has_table(conn, "civil_scheme_executions"):
        for sid, vid, ap, sk, ch in conn.execute(
                "SELECT snap_id, variant_id, applied_json, skipped_json, changed_json FROM civil_scheme_executions ORDER BY id"):
            reports[int(sid)] = (vid, json.loads(ap), json.loads(sk), json.loads(ch))
    out: list[VariantEvidence] = []
    rows = conn.execute(
        "SELECT id, ml_snap_id, session_id, recommended_adjustment_id, executed_adjustment_id, executed_status,"
        " executed_formation, executed_play, result FROM snaps WHERE recommended_adjustment_id LIKE 'scheme:%'"
        " AND result IS NOT NULL AND trim(result) != '' ORDER BY id").fetchall()
    for sid, ml, sess, rec, exe, status, ef, ep, _res in rows:
        vid = str(rec)[len(ADJ_PREFIX):]
        v = cat.get(vid)
        if v is None or str(status or "") != "identified":
            continue
        if (_n(ef), _n(ep)) != (_n(v.formation), _n(v.base_play)):
            continue                           # a different play was run: not evidence about this variant
        key = str(ml) if ml not in (None, "") else str(sid)
        n = len(v.adjustments)
        rep = reports.get(int(sid))
        if rep is not None and rep[0] == vid:
            applied = tuple(sorted(int(i) for i in rep[1]))
            changed = rep[3]
            unknown = set(range(n)) - set(applied) - {int(i) for i in rep[2]} - {int(k) for k in changed}
            out.append(VariantEvidence(int(sid), key, str(sess or ""), vid, applied, len(applied) == n,
                                       usable=not changed and not unknown, source="owner_execution_report"))
        elif str(exe or "") == rec:
            out.append(VariantEvidence(int(sid), key, str(sess or ""), vid, tuple(range(n)), True, True,
                                       "adjustment_confirmed"))
    return out


def excluded_snap_keys(conn: Any) -> set[str]:
    """Civil recommendations whose adjustments were not confirmed or reported: what ran is unknown -> excluded."""
    keys: set[str] = set()
    for sid, ml, rec, exe in conn.execute(
            "SELECT id, ml_snap_id, recommended_adjustment_id, executed_adjustment_id FROM snaps"
            " WHERE recommended_adjustment_id LIKE 'scheme:%'"):
        keys.add(str(ml) if ml not in (None, "") else str(sid))
    return keys


def variant_log(conn: Any, cat: SchemeCatalog) -> list[dict[str, Any]]:
    """The item-7 per-variant log: decision + outcome + what was actually executed."""
    ev = {e.snap_id: e for e in variant_evidence(conn, cat)}
    decisions: list[tuple] = []
    if _has_table(conn, "civil_scheme_decisions"):
        decisions = conn.execute(
            "SELECT id, snap_floor_id, session_id, opponent_id, variant_id, base_play, adjustments_json, down, distance,"
            " yardline, belief_json, mode FROM civil_scheme_decisions WHERE status = 'CALL' ORDER BY id").fetchall()
    out = []
    from cfb_coach.outcome import parse_outcome
    for did, floor, sess, opp, vid, base, adjs, down, dist, yl, belief, mode in decisions:
        snap = conn.execute(
            "SELECT id, result, yards, coverage_seen, executed_status, executed_play, executed_adjustment_id FROM snaps"
            " WHERE session_id IS ? AND recommended_adjustment_id = ? AND id > ? ORDER BY id LIMIT 1",
            (sess, ADJ_PREFIX + str(vid), floor or 0)).fetchone()
        rec: dict[str, Any] = {"decision_id": did, "variant_id": vid, "base_play": base, "opponent_id": opp,
                               "game_id": sess, "mode": mode, "situation": {"down": down, "distance": dist, "yardline": yl},
                               "pre_snap_belief": json.loads(belief or "{}"),
                               "recommended_adjustments": json.loads(adjs or "[]")}
        if snap is None:
            rec.update({"snap_id": None, "followed": None, "evidence": "no outcome logged (not evidence)"})
        else:
            sid, res, yards, cov, status, eplay, eadj = snap
            kind = parse_outcome(str(res)).kind if res else None
            e = ev.get(int(sid))
            v = cat.get(str(vid))
            rec.update({
                "snap_id": sid, "result": res, "yards": yards, "actual_coverage": cov,
                "completion": None if kind is None else kind not in ("incomplete", "int", "sack"),
                "first_down_or_td": kind in ("td", "convert") if kind else None, "touchdown": kind == "td",
                "followed": bool(e is not None),
                "executed_play": eplay,
                "actual_adjustments": ([[a.target, a.action] for a in v.adjustments if a.index in e.applied]
                                       if (e and v) else None),
                "evidence": (e.source + ("" if e.usable else " (changed/unknown adjustments: kept, not trained)"))
                            if e else "not executed as recommended (not evidence)",
            })
        out.append(rec)
    return out
