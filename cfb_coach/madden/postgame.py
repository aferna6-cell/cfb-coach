"""Madden postgame — learn from logged snaps, macro validation, lab → primary promotions.

Storage reuses the CFB promotion helpers (same DB meta shape) inside the
separate Madden DB, so CFB Alabama promotions are never touched.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cfb_coach.dynasty import list_alabama_promotions, promote, record_promotions
from cfb_coach.gameplan import _result_success
from cfb_coach.madden.franchise import LAB, normalize_profile, profile_config
from cfb_coach.madden.macros import FAILED, PROVEN, macro_status, set_status

PROMOTE_MIN_SNAPS = 3
PROMOTE_MIN_RATE = 0.6
FAIL_MAX_RATE = 0.3
LEARN_KEY = "learned_snap_id"


def _base_macro(raw: str | None) -> str | None:
    name = (raw or "").split(" [", 1)[0].strip().upper()
    return None if name in ("", "NONE") else name


def learn(db: Any, opponent_id: str) -> dict[str, Any]:
    """Learn from this opponent's new snaps: CFB retrain (rebuild_all) + macro proven/failed status."""
    since = int(db.get_meta(f"{LEARN_KEY}:{opponent_id}") or 0)
    snaps = list(reversed(db.get_recent_snaps(opponent_id, since_id=since, limit=500)))
    plays: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    macros: dict[str, list[bool]] = defaultdict(list)
    for s in snaps:
        ok = _result_success(s["result"], s["side"])
        if ok is None:
            continue
        plays[(s["side"], s["formation"] or "?", s["play"] or "?")].append(ok)
        m = _base_macro(s["macro"])
        if m:
            macros[m].append(ok)
    # CFB parity (v1.12 learning): rebuild every learned weight — per-opponent play / zone
    # (open / red zone / goal line) / play-vs-look / macro weights — from ALL logged snaps +
    # game W/L, with red-zone and turnover-aware scoring. Same code path as CFB postgame,
    # written into this game's own DB (madden27.db, same schema as coach.db).
    from cfb_coach.gameplan import learn_from_snaps

    retrain = learn_from_snaps(db, opponent_id, since_id=since or None)
    status_changes: list[str] = []
    for m, res in macros.items():
        # Career record across all opponents in the Madden DB
        rows = db.conn.execute(
            "SELECT side, result FROM snaps WHERE UPPER(macro) LIKE ?", (f"{m}%",)
        ).fetchall()
        rec = [r for r in (_result_success(r["result"], r["side"]) for r in rows) if r is not None]
        if len(rec) >= PROMOTE_MIN_SNAPS:
            rate = sum(rec) / len(rec)
            new = PROVEN if rate >= PROMOTE_MIN_RATE else FAILED if rate <= FAIL_MAX_RATE else None
            if new and macro_status(m, db) != new:
                set_status(db, m, new)
                status_changes.append(f"{m} → {new} ({sum(rec)}/{len(rec)} success)")
    session_id = None
    try:
        r = db.conn.execute(
            "SELECT session_id FROM snaps WHERE opponent_id = ? AND session_id IS NOT NULL AND id > ? "
            "ORDER BY id DESC LIMIT 1", (opponent_id, since)).fetchone()
        session_id = r[0] if r else None
    except Exception:  # noqa: BLE001
        session_id = None
    if snaps:
        db.set_meta(f"{LEARN_KEY}:{opponent_id}", str(max(int(s["id"]) for s in snaps)))
    return {
        "snaps": len(snaps),
        "plays": plays,
        "macros": macros,
        "status_changes": status_changes,
        "grades": retrain["grades"],
        "vs_look_changes": retrain["vs_look_changes"],
        "changes": retrain["changes"],
        "rebuild": retrain["rebuild"],
        "session_id": session_id,
    }


def summary(db: Any, opponent_id: str, profile: str | None = None) -> str:
    pid = normalize_profile(profile)
    pcfg = profile_config(pid)
    out = learn(db, opponent_id)
    lines = [
        f"# POSTGAME — Madden 27 Franchise vs {opponent_id}",
        f"Profile: {pcfg['label']} ({pcfg['mode']}) — team: {pcfg['team_label']}",
        f"New snaps learned: {out['snaps']}",
    ]
    for side in ("offense", "defense"):
        rows = [(k, v) for k, v in out["plays"].items() if k[0] == side]
        if not rows:
            continue
        lines.append(f"## {side.title()}")
        for (_, form, play), res in sorted(rows, key=lambda kv: -len(kv[1])):
            lines.append(f"  {form} — {play}: {sum(res)}/{len(res)} success")
    if out["macros"]:
        lines.append("## Macros used")
        for m, res in out["macros"].items():
            lines.append(f"  {m} [{macro_status(m, db)}]: {sum(res)}/{len(res)} success")
    try:
        from cfb_coach.learning import RULES_VERSION, format_postgame_v2

        lines.append(f"## Retrain (all {db.count_snaps(opponent_id)} logged snaps re-scored under rules {RULES_VERSION} → "
                     "per-opponent weights; red zone + turnover aware)")
        lines.extend(format_postgame_v2(db, opponent_id, out["rebuild"], session_id=out.get("session_id")))
    except Exception as exc:  # noqa: BLE001 — never break postgame
        lines.append(f"  (retrain report unavailable: {type(exc).__name__}: {exc})")
    grades = out.get("grades") or []
    if grades:
        from cfb_coach.retrain import format_grades_summary

        lines.append("## Play vs coverage/look")
        lines.extend(format_grades_summary(grades))
    for ch in out["status_changes"]:
        lines.append(f"  STATUS: {ch}")

    if pid == LAB:
        promos = []
        for (side, form, play), res in out["plays"].items():
            if len(res) >= PROMOTE_MIN_SNAPS and sum(res) / len(res) >= PROMOTE_MIN_RATE:
                promos.append({
                    "kind": "gameplan_overlay", "target": f"{form} — {play}", "side": side,
                    "status": "pending", "note": f"lab {sum(res)}/{len(res)} vs {opponent_id}",
                })
        for m, res in out["macros"].items():
            if len(res) >= PROMOTE_MIN_SNAPS and sum(res) / len(res) >= PROMOTE_MIN_RATE:
                promos.append({
                    "kind": "macro_loadout", "target": m, "status": "pending",
                    "note": f"lab macro held {sum(res)}/{len(res)} — consider Active-8 on primary",
                })
        if promos:
            record_promotions(db, promos)
            lines.append(f"## Lab → primary: {len(promos)} promotion candidate(s) recorded")
    pending = format_promotions(db)
    if "No primary promotions" not in pending:
        lines.append(pending)
    return "\n".join(lines)


def format_promotions(db: Any) -> str:
    items = list_alabama_promotions(db)
    if not items:
        return "No primary promotions pending (Franchise lab successes promote here)."
    lines = ["## Primary promotions (from Franchise lab)"]
    for p in items:
        lines.append(f"  [{p.get('status') or 'pending'}] {p.get('kind')}: {p.get('target')} — {p.get('note', '')}")
    lines.append("  Use: promote --game madden27 --accept-all   (or --target NAME)")
    return "\n".join(lines)


__all__ = ["format_promotions", "learn", "promote", "summary"]
