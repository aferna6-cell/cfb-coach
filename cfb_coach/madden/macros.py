"""Madden macro loadout — 8-cap Active (O+D), validation tags, swap plans.

Status starts from the catalog (all meta_grounded) and is overridden per
Madden DB by postgame (proven / failed) via meta key `macro_status_json`.
"""

from __future__ import annotations

import json
from typing import Any

from cfb_coach.macros import FAILED, META_GROUNDED, PROVEN, UNVALIDATED, normalize_status
from cfb_coach.madden.data import get_macro, load_macro_catalog

USER_ACTIVE_CAP = 8
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
    m = get_macro(key)
    return normalize_status((m or {}).get("validated_status")) if m else UNVALIDATED


def tag_live(name: str | None, db: Any = None) -> str:
    if not name or name.lower() == "none":
        return "none"
    s = macro_status(name, db)
    return name if s == PROVEN else f"{name} [{s}]"


def macro_side(name: str) -> str:
    return (get_macro(name) or {}).get("side") or "defense"


def split_loadout(active: list[str]) -> dict[str, list[str]]:
    return {
        "defense": [m for m in active if macro_side(m) == "defense"],
        "offense": [m for m in active if macro_side(m) == "offense"],
    }


def swap_plan(add: str, active: list[str], archetype: str | None) -> dict[str, Any]:
    """ADD at 8/8 → pick which Active macro to deactivate first."""
    hints = load_macro_catalog().get("swap_priority") or {}
    pref = hints.get((archetype or "").lower()) or hints.get("default") or {}
    same_side = [m for m in active if macro_side(m) == macro_side(add)]
    bench = next((m for m in pref.get("prefer_bench") or [] if m in active), None)
    bench = bench or (same_side[-1] if same_side else active[-1])
    return {
        "add": add,
        "bench": bench,
        "bench_side": macro_side(bench),
        "why": pref.get("why") or "keep coverage answers for this persona",
        "xbox_steps": [
            f"Deactivate {bench} (keeps Active at 8/8 O+D)",
            f"Build {add} from its copy block (pre-snap recipe; menu names approx)",
            f"Set {add} Active — confirm Active count is 8, not 9",
        ],
    }


def loadout_cards(
    active: list[str],
    db: Any = None,
    *,
    offense_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cards for ONLY the Active loadout (≤8). CPU → offense macros only."""
    names = [m for m in active if not offense_only or macro_side(m) == "offense"]
    cards: list[dict[str, Any]] = []
    for name in names:
        m = get_macro(name) or {"name": name, "side": "?"}
        m["id"] = name
        m["slot"] = "active"
        m["validated_status"] = macro_status(name, db)
        cards.append(m)
    split = split_loadout(active)
    total = len(names)
    cap = USER_ACTIVE_CAP
    meter = (
        f"Active {total}/{cap} (offense only — CPU)"
        if offense_only
        else f"Active {total}/{cap} (D {len(split['defense'])} + O {len(split['offense'])})"
    )
    loadout = {
        "meter": meter,
        "total": total,
        "defense": [] if offense_only else split["defense"],
        "offense": split["offense"],
    }
    return cards, loadout


__all__ = [
    "FAILED",
    "META_GROUNDED",
    "PROVEN",
    "USER_ACTIVE_CAP",
    "loadout_cards",
    "macro_side",
    "macro_status",
    "set_status",
    "split_loadout",
    "status_overrides",
    "swap_plan",
    "tag_live",
]
