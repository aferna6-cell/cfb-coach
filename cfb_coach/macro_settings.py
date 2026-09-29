"""Aidan's Custom Adjustment settings — ONE source shared by CFB 27 and Madden 27 (v1.17).

Both games have the same macro editor and Aidan's settings are the same in both, so a
macro's settings are keyed by its NAME (+ side), not by game:

  1. base   = CFB's source of truth: the confirmed rows in ``data/macro_catalog.json``
              (his exact sheets), word for word — never reworded.
  2. shared = rows he enters with ``macro-settings`` from EITHER game
              (``~/.cfb-coach/macro_settings.json``). Replaces the same (section, setting)
              of the base, otherwise appended.
  3. override (optional) = rows for one game only (``macro-settings --this-game-only``),
              for a setting that really only applies in that game.

A name matches only when it is the same name on the same side (catalog id or Xbox name,
case-insensitive): ``MAN`` is not ``O-MAN``. A macro with no rows from any layer "needs
settings" — nothing is ever guessed or filled from research. Everything Aidan didn't set
is Default.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHARED_FILENAME = "macro_settings.json"
LEGACY_MADDEN_FILENAME = "madden27_macros.json"  # pre-v1.17 Madden-only copy (migrated)
GAMES = ("cfb27", "madden27")


def settings_path() -> Path:
    from cfb_coach.games import data_dir

    return data_dir() / SHARED_FILENAME


# ---------------------------------------------------------------------------
# CFB catalog (the base layer) — exact rows only
# ---------------------------------------------------------------------------

def _cfb_macros() -> dict[str, Any]:
    from cfb_coach.macros import load_macro_catalog

    return load_macro_catalog().get("macros") or {}


def cfb_match(name: str | None, side: str | None = None) -> str | None:
    """CFB catalog id with the SAME name (id or Xbox name) on the same side, else None."""
    key = (name or "").strip().upper()
    if not key:
        return None
    macros = _cfb_macros()
    hits = []
    for mid, meta in macros.items():
        if side and (meta.get("side") or "defense") != side:
            continue
        names = {mid.upper(), str(meta.get("xbox_name") or "").upper()}
        if key in names:
            hits.append(mid)
    if key in hits:  # exact id beats an Xbox-name hit
        return key
    return hits[0] if len(hits) == 1 else None


def cfb_catalog_rows(cfb_id: str) -> list[dict[str, Any]]:
    """Aidan's exact rows for a CFB macro, verbatim from ``macro_catalog.json``.

    Offense: the confirmed route assignments / protection / blocking fields (the CFB v1.15
    drill-down rows). Defense: the confirmed rows of his exact sheets (entries that carry an
    in-editor ``section``; the "Notes" rows are commentary, not settings)."""
    meta = _cfb_macros().get(cfb_id) or {}
    if (meta.get("side") or "defense") == "offense":
        from cfb_coach.macros import catalog_offense_rows

        return [dict(r) for r in catalog_offense_rows(cfb_id)]
    rows: list[dict[str, Any]] = []
    for key, ent in (meta.get("full_settings") or {}).items():
        if not isinstance(ent, dict) or (ent.get("status") or "").lower() != "confirmed":
            continue
        sec = str(ent.get("section") or "").strip()
        if not sec or sec.lower() == "notes":
            continue
        _, _, name = key.partition(" / ")
        rows.append({"section": sec, "setting": name.strip() or sec, "value": str(ent.get("value") or "")})
    return rows


# ---------------------------------------------------------------------------
# Shared user store
# ---------------------------------------------------------------------------

def load_store() -> dict[str, Any]:
    migrate_legacy_madden_file()
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        data.setdefault("macros", {})
        return data
    except (OSError, ValueError):
        return {"version": 1, "macros": {}}


def _write_store(data: dict[str, Any]) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["version"] = 1
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def store_key(name: str, side: str) -> str:
    """One entry per macro across both games: the CFB id when the name matches, else the name."""
    return cfb_match(name, side) or (name or "").strip().upper()


def split_setting(raw: str) -> dict[str, str]:
    """'Coverage: Shading = Over the top' | 'Shading = Over the top' | 'Coverage / Shading: x'."""
    txt = raw.strip()
    sec = ""
    if "=" in txt:
        left, val = txt.split("=", 1)
        if ":" in left:
            sec, name = left.split(":", 1)
        elif "/" in left:
            sec, name = left.split("/", 1)
        else:
            name = left
    elif ":" in txt:
        left, val = txt.split(":", 1)
        if "/" in left:
            sec, name = left.split("/", 1)
        else:
            sec, name = "", left
    else:
        raise ValueError(f"setting needs a value: use \"Section: Setting = value\" (got {raw!r})")
    name, val, sec = name.strip(), val.strip(), sec.strip()
    if not name or not val:
        raise ValueError(f"setting needs a name and a value (got {raw!r})")
    return {"section": sec or name, "setting": name, "value": val}


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (a["section"].lower() == b["section"].lower()
            and a["setting"].lower() == b["setting"].lower())


def _merge(base: list[dict[str, Any]], top: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = [dict(r) for r in base]
    for r in top:
        row = {"section": r["section"], "setting": r["setting"], "value": r["value"], "source": source}
        idx = next((i for i, o in enumerate(out) if _same(o, row)), None)
        if idx is None:
            out.append(row)
        else:
            out[idx] = row
    return out


def save_settings(name: str, side: str, settings: list[str], *, game: str | None = None,
                  replace: bool = False, xbox_name: str | None = None, clear: bool = False) -> dict[str, Any]:
    """Store Aidan's exact settings, verbatim. ``game`` set → a per-game override for that game
    only; None → shared (both games see it). Returns the stored entry."""
    if game is not None and game not in GAMES:
        raise ValueError(f"unknown game {game!r}")
    key = store_key(name, side)
    data = load_store()
    macros = data.setdefault("macros", {})
    ent = macros.setdefault(key, {"side": side, "settings": []})
    ent["side"] = side
    target = ent.setdefault("overrides", {}).setdefault(game, {"settings": []}) if game else ent
    if clear:
        target["settings"] = []
        if game:
            ent["overrides"].pop(game, None)
    else:
        rows = [] if replace else list(target.get("settings") or [])
        for raw in settings:
            row = split_setting(raw)
            rows = [r for r in rows if not _same(r, row)] + [row]
        target["settings"] = rows
    if xbox_name:
        ent["xbox_name"] = xbox_name
    target["updated"] = datetime.now(timezone.utc).isoformat()
    if not ent.get("settings") and not ent.get("overrides") and not ent.get("xbox_name"):
        macros.pop(key, None)
    _write_store(data)
    return macros.get(key) or {}


def user_entry(name: str, side: str) -> dict[str, Any]:
    return (load_store().get("macros") or {}).get(store_key(name, side)) or {}


def settings_for(name: str, side: str, game: str) -> list[dict[str, Any]]:
    """Aidan's settings for this macro in this game: CFB base → shared edits → this game's
    override. Each row carries ``source``. Empty list = needs settings (nothing invented)."""
    cfb_id = cfb_match(name, side)
    rows = [dict(r, source="CFB 27 macro (shared)") for r in (cfb_catalog_rows(cfb_id) if cfb_id else [])]
    ent = user_entry(name, side)
    rows = _merge(rows, ent.get("settings") or [], "yours (shared)")
    over = ((ent.get("overrides") or {}).get(game) or {}).get("settings") or []
    return _merge(rows, over, f"yours ({game} only)")


def has_user_rows(name: str, side: str, game: str) -> bool:
    ent = user_entry(name, side)
    return bool(ent.get("settings") or ((ent.get("overrides") or {}).get(game) or {}).get("settings"))


def match_report(name: str, side: str, game: str = "madden27") -> dict[str, Any]:
    """How this macro gets its settings (for listings + the report)."""
    cfb_id = cfb_match(name, side)
    rows = settings_for(name, side, game)
    return {"name": (name or "").upper(), "side": side, "cfb_match": cfb_id,
            "n_settings": len(rows), "needs_settings": not rows}


# ---------------------------------------------------------------------------
# Migration: the pre-v1.17 Madden-only file → the shared store
# ---------------------------------------------------------------------------

def migrate_legacy_madden_file() -> list[str]:
    """Fold ``madden27_macros.json`` (Madden's own copy) into the shared store, then rename it
    ``.migrated``. Nothing is lost: a macro CFB already has settings for keeps CFB unchanged
    and gets Aidan's Madden rows as a madden27-only override; any other macro's rows become
    shared. Idempotent."""
    from cfb_coach.games import data_dir

    legacy = data_dir() / LEGACY_MADDEN_FILENAME
    if not legacy.exists():
        return []
    try:
        old = json.loads(legacy.read_text(encoding="utf-8")).get("macros") or {}
    except (OSError, ValueError):
        return []
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"version": 1, "macros": {}}
    macros = data.setdefault("macros", {})
    moved: list[str] = []
    for mid, rec in old.items():
        rows = [r for r in (rec or {}).get("settings") or [] if r.get("setting") and r.get("value")]
        if not rows and not (rec or {}).get("xbox_name"):
            continue
        from cfb_coach.madden.data import get_macro as madden_macro

        side = ((madden_macro(mid) or {}).get("side")) or ("offense" if mid.upper().startswith("O-") else "defense")
        key = store_key(mid, side)
        ent = macros.setdefault(key, {"side": side, "settings": []})
        cfb_id = cfb_match(mid, side)
        target = (ent.setdefault("overrides", {}).setdefault("madden27", {"settings": []})
                  if cfb_id and cfb_catalog_rows(cfb_id) else ent)
        cur = list(target.get("settings") or [])
        for r in rows:
            row = {"section": r.get("section") or r["setting"], "setting": r["setting"], "value": str(r["value"])}
            cur = [c for c in cur if not _same(c, row)] + [row]
        target["settings"] = cur
        target["updated"] = rec.get("updated") or datetime.now(timezone.utc).isoformat()
        if rec.get("xbox_name"):
            ent["xbox_name"] = rec["xbox_name"]
        moved.append(key)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["version"] = 1
    data.setdefault("migrated", []).append({"from": LEGACY_MADDEN_FILENAME, "macros": moved,
                                            "ts": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    legacy.rename(legacy.with_name(LEGACY_MADDEN_FILENAME + ".migrated"))
    return moved


# ---------------------------------------------------------------------------
# CFB `macro-settings` (edits the same shared store Madden reads)
# ---------------------------------------------------------------------------

def cfb_cli(args: Any) -> int:
    """``macro-settings --game cfb27 [NAME --set ...]`` — same store as Madden."""
    game = "cfb27"
    macros = _cfb_macros()
    if args.macro:
        name = args.macro.upper()
        sides = sorted({(m.get("side") or "defense") for mid, m in macros.items()
                        if name in (mid.upper(), str(m.get("xbox_name") or "").upper())})
        side = getattr(args, "side", None) or (sides[0] if len(sides) == 1 else None)
        if side is None:
            raise SystemExit(f"{name}: unknown CFB macro (or both sides use it — pass --side offense|defense)")
        if args.settings or args.clear or args.xbox_name:
            try:
                save_settings(name, side, args.settings, game=game if args.this_game_only else None,
                              replace=bool(args.replace), xbox_name=args.xbox_name, clear=bool(args.clear))
            except ValueError as exc:
                raise SystemExit(str(exc)) from None
            print(f"Saved → {settings_path()} ({'CFB 27 only' if args.this_game_only else 'shared: CFB 27 + Madden 27'})")
        print(f"MACRO: {name} ({side})")
        for r in settings_for(name, side, game):
            lab = r["value"] if r["setting"] == r["section"] else f"{r['setting']}: {r['value']}"
            print(f"  [ ] {r['section']} / {lab}" if r["setting"] != r["section"] else f"  [ ] {r['section']}: {r['value']}")
        print("  [ ] Everything else: Default")
        return 0
    print(f"CFB 27 macro settings (shared with Madden 27) → {settings_path()}")
    for mid, m in macros.items():
        side = m.get("side") or "defense"
        n = len(settings_for(mid, side, game))
        st = f"{n} setting(s) on file" if n else "NEEDS SETTINGS"
        print(f"  {mid:<14} ({side}) {st}")
    return 0


__all__ = [
    "cfb_catalog_rows",
    "cfb_cli",
    "cfb_match",
    "has_user_rows",
    "load_store",
    "match_report",
    "migrate_legacy_madden_file",
    "save_settings",
    "settings_for",
    "settings_path",
    "split_setting",
    "store_key",
]
