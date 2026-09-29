"""Madden macros (v1.17) — 10 research-built DEFENSE macros per opponent, with buttons.

Defense macros and their settings come from the research DB (``madden/research_db.py``,
refreshed by the daily research routine and pulled by every prep): every Custom Adjustments
field gets the value a cited source names, else Default. Offense uses no macros — see
``madden/adjustments.py``.

Status starts meta_grounded and is overridden per Madden DB by postgame (proven / failed)
via meta key `macro_status_json`.
"""

from __future__ import annotations

import json
from typing import Any

from cfb_coach.macros import FAILED, META_GROUNDED, PROVEN, UNVALIDATED, normalize_status
from cfb_coach.madden import research_db as rdb
from cfb_coach.madden.macro_pool import display_name, families, macro_side, pool_macro

GAME = "madden27"
PER_SIDE = 10  # v1.17: 10 defense macros per opponent (offense = adjustments, no macros)
USER_ACTIVE_CAP = 8  # legacy (pre-v1.17) Active-8 cap: only used to read / migrate old selections
STATUS_META_KEY = "macro_status_json"


def status_overrides(db: Any) -> dict[str, str]:
    if db is None:
        return {}
    raw = db.get_meta(STATUS_META_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return {str(k).upper(): normalize_status(v) for k, v in dict(data).items()}


def set_status(db: Any, name: str, status: str) -> None:
    cur = status_overrides(db)
    cur[name.upper()] = normalize_status(status)
    db.set_meta(STATUS_META_KEY, json.dumps(cur))


def macro_status(name: str | None, db: Any = None) -> str:
    key = (name or "").upper()
    over = status_overrides(db)
    if key in over:
        return over[key]
    m = pool_macro(key)
    return normalize_status((m or {}).get("validated_status")) if m else UNVALIDATED


def tag_live(name: str | None, db: Any = None) -> str:
    if not name or name.lower() == "none":
        return "none"
    s = macro_status(name, db)
    return name if s == PROVEN else f"{name} [{s}]"


def split_loadout(active: list[str]) -> dict[str, list[str]]:
    return {
        "defense": [m for m in active if macro_side(m) == "defense"],
        "offense": [m for m in active if macro_side(m) == "offense"],
    }


# ---------------------------------------------------------------------------
# Research-built settings + buttons
# ---------------------------------------------------------------------------

EDITOR_PATH = ["Create & Share", "Custom Adjustments", "Defense", "Create / Edit"]


def activate_buttons(mid: str) -> str:
    """How to fire it at the line (from the DB's cited Xbox controls)."""
    return f"{rdb.buttons('defense', 'custom_adjustments').replace('pick the macro', display_name(mid))}"


def settings_rows(mid: str) -> list[dict[str, Any]]:
    """Every editor field: researched value (+ source) or Default."""
    return rdb.full_settings(mid) if pool_macro(mid) else []


def key_settings(mid: str, limit: int = 6) -> str:
    """One-line summary of the RESEARCHED fields (the full list is in the drill-down)."""
    bits = [f"{r['setting']} {r['value']}" for r in rdb.researched_rows(mid)] if pool_macro(mid) else []
    more = len(bits) - limit
    return " · ".join(bits[:limit]) + (f" · … (+{more} more — open the macro)" if more > 0 else "")


def _pairs_in_book(mid: str, book: dict[str, list[str]] | None, *, cap: int = 6) -> list[str]:
    m = pool_macro(mid) or {}
    base = m.get("base") or {}
    play = str(base.get("play") or "")
    if not book:
        return [m["shell_pair"]] if m.get("shell_pair") else []
    out = [f"{p} ({f})" for f, ps in book.items() for p in ps if play and p.lower() in play.lower()]
    return out[:cap]


def macro_detail(mid: str, book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    key = (mid or "").upper()
    meta = pool_macro(key) or {}
    rows = settings_rows(key)
    srcs = rdb.sources()
    used = [s for s in dict.fromkeys(r["source"] for r in rows if r.get("source") != "default")]
    researched = [r for r in rows if r.get("source") != "default"]
    for r in rows:
        s = srcs.get(r.get("source") or "")
        r["research"] = (f"{s['title']}" + (f" — {r['note']}" if r.get("note") else "")) if s else ""
    return {
        "id": key,
        "side": "defense",
        "xbox_name": display_name(key),
        "editor_path": list(EDITOR_PATH),
        "in_game": f"At the line: {activate_buttons(key)}",
        "buttons": activate_buttons(key),
        "settings": rows,
        "has_settings": bool(researched),
        "needs_settings": not researched,
        "n_researched": len(researched),
        "n_fields": len(rows),
        "gaps": [],
        "gap_rows": [],
        "settings_source": ("research-built from the research DB (daily routine); every field without a "
                            "cited value is Default — " + rdb.status()["line"]),
        "fire_when": meta.get("when_to_arm") or "",
        "pairs_with": _pairs_in_book(key, book),
        "shell_pair": meta.get("shell_pair") or "",
        "key": key_settings(key),
        "purpose": meta.get("purpose") or "",
        "sources": [{"id": sid, "title": srcs[sid]["title"], "url": srcs[sid].get("url", ""),
                     "supports": "research-built settings"} for sid in used + [s for s in meta.get("sources") or []
                                                                             if s not in used] if sid in srcs],
        "hot_route_menu_source": "",
        "limits": "Values are what the cited sources say; fields no source names are Default.",
    }


def copy_block(detail: dict[str, Any]) -> str:
    """Tick-by-tick: EVERY editor field (researched value or Default), then how to fire it."""
    name = detail.get("xbox_name") or detail.get("id")
    lines = [f"MACRO: {name} (defense) — research-built, {detail.get('n_researched', 0)}/{detail.get('n_fields', 0)} "
             "fields from sources, the rest Default",
             "Path: " + " > ".join(detail.get("editor_path") or []), f"[ ] Name: {name}"]
    if detail.get("shell_pair"):
        lines.append(f"Base: {detail['shell_pair']}")
    cur = None
    for r in detail.get("settings") or []:
        if r["section"] != cur:
            cur = r["section"]
            lines.append(cur)
        tag = "" if r.get("source") == "default" else f"   <- {r.get('source')}"
        lines.append(f"  [ ] {r['setting']}: {r['value']}{tag}")
    lines.append("[ ] Save → set Active")
    lines.append(f"In game: {detail.get('buttons')}")
    if detail.get("fire_when"):
        lines.append(f"Fire when: {detail['fire_when']}")
    return "\n".join(lines)


def attach_detail(card: dict[str, Any], book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    det = macro_detail(str(card.get("id") or ""), book)
    card["ingame"] = det
    card["copy_block"] = copy_block(det)
    card["book_plays"] = det["pairs_with"]
    card["missing_settings"] = det["needs_settings"]
    return card


def missing_settings_report(ids: list[str]) -> list[str]:
    return [f"{mid}: no researched settings in the research DB (every field Default)"
            for mid in ids if pool_macro(mid) and not rdb.researched_rows(mid)]


def copy_checklist(selection: dict[str, list[str]]) -> str:
    """One checklist for the defense 10: name, researched fields, button to fire it."""
    ids = list(selection.get("defense") or [])
    if not ids:
        return "DEFENSE: none (CPU game = offense only; offense uses adjustments, not macros)"
    lines = [f"DEFENSE ({len(ids)}) — build in Create & Share > Custom Adjustments > Defense, then set Active"]
    for i, mid in enumerate(ids, 1):
        n = len(rdb.researched_rows(mid))
        flag = f"{n} researched field(s), rest Default" if n else "[!] no researched fields"
        lines.append(f"  [ ] {i:>2}. {display_name(mid)} — {flag} — fire: {activate_buttons(mid)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The defense 10 per opponent (what live `play` may fire) + migration of the old Active 8
# ---------------------------------------------------------------------------

ACTIVE_META_KEY = "active_macros:{opp}"
LEGACY_BACKUP_KEY = "active_macros_legacy8:{opp}"


def _clean(ids: list[Any], side: str) -> list[str]:
    out: list[str] = []
    for x in ids or []:
        k = str(x).upper()
        if pool_macro(k) and macro_side(k) == side and k not in out:
            out.append(k)
    return out[:PER_SIDE]


def store_selection(db: Any, opponent_id: str, selection: dict[str, list[str]]) -> None:
    from datetime import datetime, timezone

    if db is None:
        return
    db.set_meta(ACTIVE_META_KEY.format(opp=opponent_id), json.dumps({
        "schema": 2,
        "offense": [],
        "defense": _clean(selection.get("defense") or [], "defense"),
        "ts": datetime.now(timezone.utc).isoformat()}))


def migrate_selection(db: Any, opponent_id: str) -> bool:
    """Old ``{"active": [8 ids]}`` → schema 2. The old record is kept verbatim as a backup
    (``active_macros_legacy8:<opp>``); ids the research DB still has carry into the defense list,
    and the next prep gives every old pick a carry-over bonus. Idempotent."""
    if db is None:
        return False
    key = ACTIVE_META_KEY.format(opp=opponent_id)
    raw = db.get_meta(key)
    if not raw:
        return False
    try:
        rec = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(rec, dict) or rec.get("schema") == 2 or "active" not in rec:
        return False
    old = [str(x).upper() for x in rec.get("active") or []]
    db.set_meta(LEGACY_BACKUP_KEY.format(opp=opponent_id), raw)
    db.set_meta(key, json.dumps({"schema": 2, "offense": [], "defense": _clean(old, "defense"),
                                 "legacy": old, "ts": rec.get("ts"), "migrated_from": "active_8"}))
    return True


def migrate_all_selections(db: Any) -> list[str]:
    """Migrate every opponent's stored Active 8 in this Madden DB (e.g. jaxon)."""
    if db is None:
        return []
    rows = db.conn.execute("SELECT key FROM meta WHERE key LIKE 'active_macros:%'").fetchall()
    return [r[0].split(":", 1)[1] for r in rows if migrate_selection(db, r[0].split(":", 1)[1])]


def legacy_picks(db: Any, opponent_id: str) -> list[str]:
    """Every macro name from the migrated Active 8 (kept even when the DB no longer has it)."""
    if db is None:
        return []
    raw = db.get_meta(LEGACY_BACKUP_KEY.format(opp=opponent_id))
    try:
        return [str(x).upper() for x in (json.loads(raw).get("active") or [])] if raw else []
    except ValueError:
        return []


def load_selection(db: Any, opponent_id: str) -> dict[str, list[str]] | None:
    if db is None:
        return None
    migrate_selection(db, opponent_id)
    raw = db.get_meta(ACTIVE_META_KEY.format(opp=opponent_id))
    if not raw:
        return None
    try:
        rec = json.loads(raw)
    except ValueError:
        return None
    sel = {"offense": [], "defense": _clean(rec.get("defense") or [], "defense")}
    return sel if sel["defense"] else None


def load_active(db: Any, opponent_id: str) -> list[str] | None:
    sel = load_selection(db, opponent_id)
    return sel["defense"] if sel else None


def as_selection(active: list[str] | dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Normalize a flat list or {offense, defense} into per-side lists (order kept)."""
    if isinstance(active, dict):
        return {"offense": [], "defense": [str(x).upper() for x in active.get("defense") or []]}
    return {"offense": [], "defense": [str(x).upper() for x in active or [] if pool_macro(str(x))]}


# ---------------------------------------------------------------------------
# Live: at most one macro per D snap, from the defense 10, only when it helps
# ---------------------------------------------------------------------------

LEARNED_SUPPRESS = -0.15


def macro_info(mid: str, why: str, *, weight: float | None = None) -> dict[str, Any]:
    meta = pool_macro(mid) or {}
    det = macro_detail(mid)
    return {
        "id": mid,
        "name": det["xbox_name"],
        "side": "defense",
        "why": why,
        "key": det["key"] or "no researched fields",
        "buttons": det["buttons"],
        "settings": [r for r in det["settings"] if r.get("source") != "default"],
        "missing_settings": det["needs_settings"],
        "fire_when": meta.get("when_to_arm") or "",
        "learned_weight": weight,
    }


def _suppressed(mid: str, weights: dict[str, float] | None) -> bool:
    w = (weights or {}).get(mid)
    return w is not None and w <= LEARNED_SUPPRESS


def best_for_family(family: str | None, ranked: list[str], weights: dict[str, float] | None = None) -> str | None:
    """Highest-ranked macro in ``ranked`` (the defense 10, prep order) that answers ``family``."""
    if not family:
        return None
    ok = [m for m in ranked if not _suppressed(m, weights)]
    primary = next((m for m in ok if families(m)[:1] == [family]), None)  # its main answer first
    return primary or next((m for m in ok if family in families(m)), None)


__all__ = [
    "FAILED", "META_GROUNDED", "PER_SIDE", "PROVEN", "USER_ACTIVE_CAP", "activate_buttons", "as_selection",
    "attach_detail", "best_for_family", "copy_block", "copy_checklist", "key_settings", "legacy_picks",
    "load_active", "load_selection", "macro_detail", "macro_info", "macro_side", "macro_status",
    "migrate_all_selections", "migrate_selection", "missing_settings_report", "set_status", "settings_rows",
    "split_loadout", "status_overrides", "store_selection", "tag_live",
]
