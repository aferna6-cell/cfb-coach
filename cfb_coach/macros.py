"""Macro catalog — 8-cap Active budget, validation badges, swap plans, copy blocks."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

# Aidan's USER online-dynasty hard cap (O+D Custom Adjustments combined).
# EA UI may advertise 10 — honor 8 for user/online dynasty.
USER_ACTIVE_CAP = 8

VALIDATED = "validated"
NEEDS_LAB = "needs_lab"
UNVALIDATED = "unvalidated"

XBOX_PATH = [
    "Create & Share",
    "Custom Adjustments",
    "Offense or Defense",
    "Create / Edit macro → set ticks → Save",
    "Set Active (max 8 O+D combined for USER dynasty)",
    "In-game: LB to open / select Active macros",
]


def _catalog_path() -> Path:
    try:
        ref = resources.files("cfb_coach").joinpath("data/macro_catalog.json")
        with resources.as_file(ref) as p:
            return Path(p)
    except Exception:
        return Path(__file__).resolve().parent / "data" / "macro_catalog.json"


@lru_cache(maxsize=1)
def load_macro_catalog() -> dict[str, Any]:
    with _catalog_path().open(encoding="utf-8") as f:
        return json.load(f)


def get_macro(name: str) -> dict[str, Any] | None:
    cat = load_macro_catalog()
    macros = cat.get("macros") or {}
    key = (name or "").upper().strip()
    if key in macros:
        return copy.deepcopy(macros[key])
    aliases = {
        "HEAT_O": "O-HEAT",
        "OHEAT": "O-HEAT",
        "RUN_O": "O-RUN",
        "ORUN": "O-RUN",
        "RPO_O": "O-RPO",
        "ORPO": "O-RPO",
    }
    alt = aliases.get(key)
    if alt and alt in macros:
        return copy.deepcopy(macros[alt])
    for mid, meta in macros.items():
        xbox = (meta.get("xbox_name") or meta.get("name") or "").upper()
        if xbox == key and meta.get("side") == "offense":
            return copy.deepcopy(meta)
    return None


def validation_status(name: str) -> str:
    m = get_macro(name)
    if not m:
        return UNVALIDATED
    return m.get("validated_status") or UNVALIDATED


def is_validated(name: str) -> bool:
    return validation_status(name) == VALIDATED


def active_cap(*, cpu: bool = False) -> int:
    cat = load_macro_catalog()
    cap_info = cat.get("active_cap") or {}
    if isinstance(cap_info, dict):
        return int(cap_info.get("user_online_dynasty", USER_ACTIVE_CAP))
    return USER_ACTIVE_CAP


def count_active(inventory: dict[str, Any] | None) -> dict[str, Any]:
    """Count Active Custom Adjustments across O+D."""
    inv = inventory or {}
    d = list(inv.get("macros_active") or [])
    o = list(
        inv.get("offensive_macros")
        or inv.get("offensive_macros_active")
        or []
    )
    total = len(d) + len(o)
    cap = active_cap()
    return {
        "defense": d,
        "offense": o,
        "defense_count": len(d),
        "offense_count": len(o),
        "total": total,
        "cap": cap,
        "at_cap": total >= cap,
        "slots_free": max(0, cap - total),
        "meter": f"Active {total}/{cap}",
    }


def _swap_hints(opponent_id: str) -> dict[str, Any]:
    cat = load_macro_catalog()
    hints = cat.get("swap_priority_hints") or {}
    oid = (opponent_id or "").lower()
    if oid == "gavin":
        return hints.get("vs_gavin") or hints.get("default") or {}
    if oid == "quen":
        return hints.get("vs_quen") or hints.get("default") or {}
    if oid == "tiano":
        return hints.get("vs_tiano") or hints.get("default") or {}
    return hints.get("default") or {}


def build_swap_plan(
    add_macro: str,
    inventory: dict[str, Any],
    *,
    opponent_id: str = "",
) -> dict[str, Any] | None:
    """If ADD would exceed Active 8 O+D, return exact swap plan + Xbox steps."""
    budget = count_active(inventory)
    if budget["slots_free"] > 0:
        return None

    add = (add_macro or "").upper().strip()
    add_meta = get_macro(add) or {}
    add_side = add_meta.get("side") or "defense"
    active_d = list(budget["defense"])
    active_o = list(budget["offense"])

    hints = _swap_hints(opponent_id)
    prefer = list(hints.get("prefer_bench") or [])
    why = hints.get("why") or "free a slot under Aidan's 8-cap (O+D combined)"

    bench_from = None
    bench_side = None
    for cand in prefer:
        if cand in active_d:
            bench_from, bench_side = cand, "defense"
            break
        if cand in active_o:
            bench_from, bench_side = cand, "offense"
            break

    if bench_from is None:
        for cand in ("HEAT", "RUN-OUT", "RUN-IN", "SCRAM"):
            if cand in active_d and cand != add:
                bench_from, bench_side = cand, "defense"
                break
    if bench_from is None and active_d:
        bench_from, bench_side = active_d[-1], "defense"
    elif bench_from is None and active_o:
        bench_from, bench_side = active_o[-1], "offense"

    if bench_from is None:
        return {
            "needed": True,
            "add": add,
            "add_side": add_side,
            "error": "At 8/8 Active with no identifiable bench candidate",
            "xbox_steps": list(XBOX_PATH),
        }

    bench_meta = get_macro(bench_from) or {}
    bench_label = "Offense" if bench_side == "offense" else "Defense"
    add_label = "Offense" if add_side == "offense" else "Defense"
    steps = [
        f"At {budget['meter']} — cannot ADD {add} without a swap "
        f"(Aidan USER cap = 8 O+D combined; EA UI may show 10 — ignore).",
        "Xbox path:",
        f"  1. Create & Share → Custom Adjustments → {bench_label}",
        f"  2. Open Active macro {bench_from} → clear Active / deactivate",
        f"  3. Create & Share → Custom Adjustments → {add_label}",
        f"  4. Create/Edit {add} using its full settings copy block → Save",
        f"  5. Set {add} Active (confirm meter ≤ 8/8)",
        f"  6. In-game: LB → select {add} when armed",
        f"Why bench {bench_from}: {why}",
    ]
    return {
        "needed": True,
        "add": add,
        "add_side": add_side,
        "add_validation": add_meta.get("validated_status") or NEEDS_LAB,
        "bench": bench_from,
        "bench_side": bench_side,
        "bench_validation": bench_meta.get("validated_status") or UNVALIDATED,
        "why": why,
        "budget_before": budget["meter"],
        "budget_after": f"Active {budget['total']}/{budget['cap']} (swap {bench_from}→{add})",
        "xbox_steps": steps,
        "prefer_bench_list": prefer,
    }


def enrich_macro_delta(
    delta: dict[str, Any],
    inventory: dict[str, Any],
    *,
    opponent_id: str = "",
) -> dict[str, Any]:
    """Attach validation + copy_block + optional swap_plan to a macro delta."""
    out: dict[str, Any] = dict(delta)
    target = (delta.get("target") or "").upper()
    meta = get_macro(target) or {}
    status = meta.get("validated_status") or UNVALIDATED
    out["validated_status"] = status
    out["validation_badge"] = status
    if meta.get("copy_block"):
        out["copy_block"] = meta["copy_block"]
    if meta.get("full_settings"):
        out["full_settings"] = meta["full_settings"]
    if meta.get("side"):
        out["side"] = meta["side"]
    if meta.get("purpose"):
        out["purpose"] = meta["purpose"]
    if meta.get("when_to_arm"):
        out["when_to_arm"] = meta["when_to_arm"]
    if meta.get("xbox_steps"):
        out["xbox_steps"] = meta["xbox_steps"]

    if (delta.get("action") or "").upper() == "ADD":
        plan = build_swap_plan(target, inventory, opponent_id=opponent_id)
        if plan:
            out["swap_plan"] = plan
            detail = (delta.get("detail") or "").strip()
            swap_bit = (
                f"SWAP REQUIRED: deactivate {plan.get('bench')} "
                f"({plan.get('bench_side')}) before Activating {target} "
                f"(was {plan.get('budget_before')})"
            )
            out["detail"] = f"{detail} | {swap_bit}" if detail else swap_bit
    return out


def tag_live_macro(name: str | None) -> str:
    """Prefix needs_lab / unvalidated macros for live caller display."""
    if not name or name.lower() in ("none", ""):
        return name or "none"
    status = validation_status(name)
    if status == VALIDATED:
        return name
    if status == NEEDS_LAB:
        return f"{name} [needs_lab]"
    return f"{name} [unvalidated]"


def prefer_validated(candidates: list[str]) -> list[str]:
    def key(n: str) -> tuple[int, str]:
        s = validation_status(n)
        rank = 0 if s == VALIDATED else (1 if s == NEEDS_LAB else 2)
        return (rank, n)

    return sorted(candidates, key=key)


def catalog_inventory_cards(
    inventory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Cards for prep browser: Active + benched + create candidates."""
    cat = load_macro_catalog()
    macros = cat.get("macros") or {}
    inv = inventory or {}
    active_d = set(inv.get("macros_active") or [])
    benched = set(inv.get("macros_benched") or [])
    active_o = set(inv.get("offensive_macros") or [])

    # Default Temple set when inventory empty
    if not inventory:
        default = cat.get("default_active_user") or {}
        active_d = set(default.get("defense") or [])
        active_o = set(default.get("offense") or [])

    cards: list[dict[str, Any]] = []
    for mid, meta in macros.items():
        slot = "reference"
        name = meta.get("name") or mid
        if mid in active_d or name in active_o or mid in active_o:
            slot = "active"
        elif mid in benched or meta.get("inventory") == "benched":
            slot = "benched"
        elif meta.get("inventory") == "create_candidate":
            slot = "create"
        elif meta.get("inventory") == "reference":
            slot = "reference"
        elif meta.get("inventory") == "active" and not inventory:
            slot = "active"

        cards.append(
            {
                "id": mid,
                "name": name,
                "side": meta.get("side"),
                "purpose": meta.get("purpose"),
                "when_to_arm": meta.get("when_to_arm"),
                "validated_status": meta.get("validated_status"),
                "slot": slot,
                "copy_block": meta.get("copy_block") or "",
                "full_settings": meta.get("full_settings") or {},
                "xbox_steps": meta.get("xbox_steps") or list(XBOX_PATH),
            }
        )
    return cards
