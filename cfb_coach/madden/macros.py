"""Madden macros (v1.17) — shared settings, 10 offense + 10 defense per opponent, live picks.

Settings come from the ONE store CFB uses (``cfb_coach.macro_settings``): a Madden macro with
the same name as a CFB macro gets that macro's settings word for word; one with no match
"needs settings" (flagged, never filled). Everything Aidan didn't set is Default.

Status starts meta_grounded and is overridden per Madden DB by postgame (proven / failed)
via meta key `macro_status_json`.
"""

from __future__ import annotations

import json
from typing import Any

from cfb_coach import macro_settings as ms
from cfb_coach.macros import FAILED, META_GROUNDED, PROVEN, UNVALIDATED, normalize_status
from cfb_coach.madden.macro_pool import display_name, families, macro_side, pool_ids, pool_macro

GAME = "madden27"
PER_SIDE = 10  # v1.17: 10 offense + 10 defense per opponent (replaces the Active 8)
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
# Aidan's settings (shared with CFB)
# ---------------------------------------------------------------------------

EDITOR_PATH = {
    "offense": ["Create & Share", "Custom Adjustments", "Offense", "Create / Edit"],
    "defense": ["Create & Share", "Custom Adjustments", "Defense", "Create / Edit"],
}
IN_GAME = "At the line: LB → pick the custom adjustment (10 offense + 10 defense per opponent)"


def settings_path():
    return ms.settings_path()


def save_user_settings(mid: str, settings: list[str], *, replace: bool = False,
                       xbox_name: str | None = None, clear: bool = False,
                       this_game_only: bool = False) -> dict[str, Any]:
    """Store Aidan's exact settings, verbatim, in the store CFB reads too (shared by default;
    ``this_game_only`` = a Madden-only override)."""
    key = mid.upper()
    if not pool_macro(key):
        raise ValueError(f"unknown Madden macro {mid!r} (known: {', '.join(pool_ids())})")
    return ms.save_settings(key, macro_side(key), settings, game=GAME if this_game_only else None,
                            replace=replace, xbox_name=xbox_name, clear=clear)


def aidan_settings(mid: str) -> list[dict[str, Any]]:
    """His settings for this macro, verbatim from the shared store. Nothing else."""
    key = (mid or "").upper()
    if not pool_macro(key):
        return []
    return ms.settings_for(key, macro_side(key), GAME)


def research_guesses(mid: str) -> list[dict[str, str]]:
    """The Madden research catalog's approximate fields (details page, reference only)."""
    out = []
    for k, ent in ((pool_macro(mid) or {}).get("full_settings") or {}).items():
        if not isinstance(ent, dict) or (ent.get("status") or "").lower() == "confirmed":
            continue
        sec, _, name = k.partition(" / ")
        out.append({"section": sec, "setting": name or sec, "research_value": str(ent.get("value") or "")})
    return out


def key_settings(mid: str, limit: int = 6) -> str:
    """One-line summary for the live call / card header (the full list is in the drill-down)."""
    rows = aidan_settings(mid)
    bits = [f"{r['setting']} {r['value']}" if r["setting"] != r["section"] else f"{r['section']}: {r['value']}"
            for r in rows]
    more = len(bits) - limit
    return " · ".join(bits[:limit]) + (f" · … (+{more} more — open the macro)" if more > 0 else "")


def _pairs_in_book(mid: str, book: dict[str, list[str]] | None, *, cap: int = 6) -> list[str]:
    import re

    m = pool_macro(mid) or {}
    shell = str(m.get("shell_pair") or "")
    out: list[str] = []
    if not book:
        return [shell] if shell else []
    for f, ps in book.items():
        for p in ps:
            if shell and ((f in shell and p in shell) or (p in shell and len(p) > 5)):
                out.append(f"{p} ({f})")
    if len(out) < cap:
        fam = (m.get("concept_family") or (m.get("families") or [""])[0])
        rx = {"vert": r"cover 4|quarters", "flood": r"cover 3 match|cover 3", "cross": r"tampa|cover 2",
              "stack": r"cover 3 match|cover 1", "scram": r"cover 4|spy", "run": r"cover 3 sky|cover 1|cover 3",
              "rpo": r"cover 3|match", "pressure": r"slant|stick|mesh|stutter|quick|screen|sim|fire|blitz",
              "man": r"mesh|slant|drive|cross|wheel|switch", "two_high": r"inside zone|zone|stretch|dig|drive",
              "single_high": r"flood|sail|cross|smash|corner", "cover2": r"smash|corner|dig|post",
              "red_zone": r"slant|fade|stick|spot|mesh"}.get(fam)
        if (m.get("side") == "defense") and fam == "pressure":
            rx = r"sim|fire|blitz|mug"
        if mid == "O-RPO":
            rx = r"rpo|alert|inside zone"
        if rx:
            for f, ps in book.items():
                for p in ps:
                    lab = f"{p} ({f})"
                    if lab not in out and re.search(rx, p, re.I):
                        out.append(lab)
                    if len(out) >= cap:
                        break
    return out[:cap]


def macro_detail(mid: str, book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Drill-down for one macro (CFB `offense_macro_detail` shape, both sides)."""
    key = (mid or "").upper()
    meta = pool_macro(key) or {}
    side = meta.get("side") or "defense"
    rows = aidan_settings(key)
    user = ms.user_entry(key, side)
    rep = ms.match_report(key, side, GAME)
    guesses = [] if rows else research_guesses(key)
    if rep["cfb_match"]:
        src = f"shared with CFB 27 macro {rep['cfb_match']} (same name) — {ms.settings_path()}"
    else:
        src = f"no CFB 27 macro named {key} — yours only if entered ({ms.settings_path()})"
    return {
        "id": key,
        "side": side,
        "xbox_name": user.get("xbox_name") or display_name(key),
        "editor_path": list(EDITOR_PATH.get(side) or []),
        "in_game": IN_GAME,
        "settings": rows,
        "has_settings": bool(rows),
        "needs_settings": not rows,
        "cfb_match": rep["cfb_match"],
        "gaps": [f"{g['section']} / {g['setting']} — research guess only: {g['research_value']}" for g in guesses],
        "gap_rows": guesses,
        "settings_source": src,
        "fire_when": meta.get("when_to_arm") or "",
        "pairs_with": _pairs_in_book(key, book),
        "shell_pair": meta.get("shell_pair") or "",
        "key": key_settings(key),
        "purpose": meta.get("purpose") or "",
        "sources": ([{"id": "research", "title": meta.get("source"), "url": "",
                      "supports": "purpose / when to arm only — not your settings"}]
                    if meta.get("source") else []),
        "hot_route_menu_source": "",
        "limits": ("Settings are Aidan's, shared by name with CFB 27 (same editor); a field he didn't "
                   "set is Default. Research labels are approximate and never used as settings."),
        "set_cmd": f'PYTHONPATH=. python3 -m cfb_coach macro-settings --game madden27 {key} --set "Section: Setting = value"',
    }


def copy_block(detail: dict[str, Any]) -> str:
    """Tick-by-tick: Aidan's settings verbatim, everything else Default; a loud flag when none."""
    name = detail.get("xbox_name") or detail.get("id")
    lines = [f"MACRO: {name} ({detail.get('side')})", "Path: " + " > ".join(detail.get("editor_path") or []),
             f"[ ] Name: {name}"]
    cur = None
    for r in detail.get("settings") or []:
        if r["setting"] == r["section"]:
            lines.append(f"[ ] {r['section']}: {r['value']}")
            cur = None
            continue
        if r["section"] != cur:
            cur = r["section"]
            lines.append(cur)
        lines.append(f"  [ ] {r['setting']}: {r['value']}")
    if not detail.get("settings"):
        lines.append("[!] NEEDS SETTINGS — no exact settings from Aidan on file for this macro "
                     f"(no CFB macro with this name). Enter them: {detail.get('set_cmd')}")
    lines.append("[ ] Everything else: Default")
    lines.append("[ ] Save")
    lines.append(f"In game: LB -> {name}")
    if detail.get("fire_when"):
        lines.append(f"Fire when: {detail['fire_when']}")
    if detail.get("pairs_with"):
        lines.append("Pairs with: " + ", ".join(detail["pairs_with"]))
    return "\n".join(lines)


def attach_detail(card: dict[str, Any], book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    det = macro_detail(str(card.get("id") or ""), book)
    card["ingame"] = det
    card["copy_block"] = copy_block(det)
    card["book_plays"] = det["pairs_with"]
    card["missing_settings"] = det["needs_settings"]
    return card


def missing_settings_report(ids: list[str]) -> list[str]:
    return [f"{mid} ({macro_side(mid)}): no CFB macro with this name and none entered"
            for mid in ids if not aidan_settings(mid)]


def copy_checklist(selection: dict[str, list[str]]) -> str:
    """One checklist for the whole loadout (offense 10 then defense 10), needs-settings flagged."""
    lines: list[str] = []
    for side in ("offense", "defense"):
        ids = list(selection.get(side) or [])
        if not ids:
            continue
        lines.append(f"{side.upper()} ({len(ids)})")
        for i, mid in enumerate(ids, 1):
            rows = aidan_settings(mid)
            flag = f"{len(rows)} setting(s) — shared" if rows else "[!] NEEDS SETTINGS"
            lines.append(f"  [ ] {i:>2}. {display_name(mid)} ({mid}) — {flag}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10 + 10 per opponent (what live `play` may fire) + migration of the old Active 8
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
        "offense": _clean(selection.get("offense") or [], "offense"),
        "defense": _clean(selection.get("defense") or [], "defense"),
        "ts": datetime.now(timezone.utc).isoformat()}))


def migrate_selection(db: Any, opponent_id: str) -> bool:
    """Old ``{"active": [8 ids]}`` → ``{"schema": 2, "offense": [...], "defense": [...]}``.
    Every stored macro is kept (split by side); the old record is backed up. Idempotent."""
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
    db.set_meta(key, json.dumps({
        "schema": 2,
        "offense": _clean([m for m in old if macro_side(m) == "offense"], "offense"),
        "defense": _clean([m for m in old if macro_side(m) == "defense"], "defense"),
        "ts": rec.get("ts"), "migrated_from": "active_8"}))
    return True


def migrate_all_selections(db: Any) -> list[str]:
    """Migrate every opponent's stored Active 8 in this Madden DB (e.g. jaxon)."""
    if db is None:
        return []
    rows = db.conn.execute("SELECT key FROM meta WHERE key LIKE 'active_macros:%'").fetchall()
    return [r[0].split(":", 1)[1] for r in rows if migrate_selection(db, r[0].split(":", 1)[1])]


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
    sel = {s: _clean(rec.get(s) or [], s) for s in ("offense", "defense")}
    return sel if sel["offense"] or sel["defense"] else None


def load_active(db: Any, opponent_id: str) -> list[str] | None:
    """Flat list (offense then defense) of the stored selection — legacy callers."""
    sel = load_selection(db, opponent_id)
    return (sel["offense"] + sel["defense"]) if sel else None


def as_selection(active: list[str] | dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Normalize a flat list or {offense, defense} into the per-side lists (order kept)."""
    if isinstance(active, dict):
        return {s: [str(x).upper() for x in active.get(s) or []] for s in ("offense", "defense")}
    flat = [str(x).upper() for x in active or []]
    return {s: [m for m in flat if pool_macro(m) and macro_side(m) == s] for s in ("offense", "defense")}


# ---------------------------------------------------------------------------
# Live picks: at most one macro per snap, from THAT side's 10, only when it helps
# ---------------------------------------------------------------------------

LEARNED_SUPPRESS = -0.15


def macro_info(mid: str, why: str, *, weight: float | None = None) -> dict[str, Any]:
    meta = pool_macro(mid) or {}
    det = macro_detail(mid)
    return {
        "id": mid,
        "name": det["xbox_name"],
        "side": meta.get("side") or "defense",
        "why": why,
        "key": det["key"] or "needs settings — no exact settings on file",
        "settings": det["settings"],
        "missing_settings": det["needs_settings"],
        "fire_when": meta.get("when_to_arm") or "",
        "learned_weight": weight,
    }


def _suppressed(mid: str, weights: dict[str, float] | None) -> bool:
    w = (weights or {}).get(mid)
    return w is not None and w <= LEARNED_SUPPRESS


def best_for_family(family: str | None, ranked: list[str], weights: dict[str, float] | None = None) -> str | None:
    """Highest-ranked macro in ``ranked`` (one side's 10, prep order) that answers ``family``."""
    if not family:
        return None
    return next((m for m in ranked if family in families(m) and not _suppressed(m, weights)), None)


def suggest_offense_macro(
    *,
    play: str,
    coverage_class: str | None,
    coverage_source: str,
    repeated: bool,
    active: list[str] | dict[str, list[str]],
    archetype: str = "",
    passing_down: bool = False,
    weights: dict[str, float] | None = None,
    zone: str = "open",
    down: int | None = None,
) -> dict[str, Any] | None:
    """At most one macro from the OFFENSE 10 for this snap, or None (most snaps).

    A coverage answer needs the look REPEATED this game (Madden arm rule, same doctrine as the
    D macros); an RPO play takes the RPO macro only vs a live / repeated zone look (CFB rule);
    red-zone passes on 3rd/4th down take RZ; a pressure persona on a passing down takes
    protection. Otherwise no macro."""
    import re

    from cfb_coach.madden.catalog import is_run

    ranked = as_selection(active)["offense"]
    if not ranked or not play:
        return None
    rpo = bool(re.search(r"rpo|alert", play, re.I))
    run = is_run(play) and not rpo
    is_pass = not run and not rpo
    look_ok = bool(coverage_class) and repeated and coverage_source in ("live", "last")
    w = weights or {}

    def pick(fam: str, why: str) -> dict[str, Any] | None:
        mid = best_for_family(fam, ranked, w)
        return macro_info(mid, why, weight=w.get(mid)) if mid else None

    tries: list[tuple[str, str]] = []
    if look_ok and is_pass:
        look = {"pressure": "pressure", "man": "man", "cover2": "cover2",
                "single_high": "single_high", "two_high": "two_high"}.get(coverage_class or "")
        if look:
            tries.append((look, f"repeated {coverage_class.replace('_', ' ')} look on {play}"))
    if look_ok and run and coverage_class == "two_high":
        tries.append(("two_high_run", f"repeated two-high look — run {play}"))
    zone_look = coverage_class in ("two_high", "cover2", "single_high") and (coverage_source == "live" or repeated)
    if rpo and zone_look:  # CFB O-RPO fire rule: a zone look (light box / soft edge), not every RPO
        tries.append(("rpo", f"RPO {play} vs {coverage_source} {coverage_class.replace('_', ' ')} look — give/keep read"))
    if is_pass and zone in ("rz", "gl") and down in (3, 4):
        tries.append(("red_zone", f"{'goal-to-go' if zone == 'gl' else 'red zone'} money down — {play}"))
    if is_pass and passing_down and archetype == "pressure_heavy":
        tries.append(("pressure", "pressure persona on a passing down — protection first"))
    for fam, why in tries:
        got = pick(fam, why)
        if got:
            return got
    return None


__all__ = [
    "FAILED",
    "META_GROUNDED",
    "PER_SIDE",
    "PROVEN",
    "USER_ACTIVE_CAP",
    "aidan_settings",
    "as_selection",
    "attach_detail",
    "best_for_family",
    "copy_block",
    "copy_checklist",
    "load_active",
    "load_selection",
    "macro_detail",
    "macro_side",
    "macro_status",
    "migrate_all_selections",
    "migrate_selection",
    "missing_settings_report",
    "save_user_settings",
    "set_status",
    "split_loadout",
    "status_overrides",
    "store_selection",
    "suggest_offense_macro",
    "tag_live",
]
