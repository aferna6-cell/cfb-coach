"""Macro catalog — 8-cap Active loadout, validation badges, swap plans, copy blocks.

Prep UI shows ONLY the active loadout (≤8) — never the benched catalog (FLOOD/SCREEN).
"""

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

# Validation ladder (prep may suggest only ≥ meta_grounded):
#   proven        — survived his games / Temple baseline
#   meta_grounded — CREATE candidate grounded in CFB27 meta + film; OK to bring into game
#   unvalidated   — not grounded enough for prep deltas
#   failed        — got cooked; demote / cooking
PROVEN = "proven"
META_GROUNDED = "meta_grounded"
UNVALIDATED = "unvalidated"
FAILED = "failed"

# Legacy aliases (pre-1.4.1)
VALIDATED = PROVEN
NEEDS_LAB = META_GROUNDED

_STATUS_ALIASES = {
    "validated": PROVEN,
    "needs_lab": META_GROUNDED,
    "needs-lab": META_GROUNDED,
    "proven": PROVEN,
    "meta_grounded": META_GROUNDED,
    "meta-grounded": META_GROUNDED,
    "unvalidated": UNVALIDATED,
    "failed": FAILED,
    "cooking": FAILED,
}

PREP_ELIGIBLE = frozenset({PROVEN, META_GROUNDED})

XBOX_PATH = [
    "Create & Share",
    "Custom Adjustments",
    "Offense or Defense",
    "Create / Edit macro → set ticks → Save",
    "Set Active (max 8 O+D combined for USER dynasty)",
    "In-game: LB to open / select Active macros",
]


def normalize_status(status: str | None) -> str:
    s = (status or UNVALIDATED).lower().replace("-", "_").strip()
    return _STATUS_ALIASES.get(s, UNVALIDATED)


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
    return normalize_status(m.get("validated_status"))


def is_proven(name: str) -> bool:
    return validation_status(name) == PROVEN


def is_validated(name: str) -> bool:
    """Legacy alias for is_proven."""
    return is_proven(name)


def is_prep_eligible(status_or_name: str, *, as_name: bool = False) -> bool:
    """True if status is proven or meta_grounded (OK for prep delta list)."""
    if as_name:
        return validation_status(status_or_name) in PREP_ELIGIBLE
    return normalize_status(status_or_name) in PREP_ELIGIBLE


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
        "add_validation": normalize_status(
            add_meta.get("validated_status") or META_GROUNDED
        ),
        "bench": bench_from,
        "bench_side": bench_side,
        "bench_validation": normalize_status(
            bench_meta.get("validated_status") or UNVALIDATED
        ),
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
    status = normalize_status(meta.get("validated_status") or UNVALIDATED)
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
    """Tag non-proven macros for live caller display."""
    if not name or name.lower() in ("none", ""):
        return name or "none"
    status = validation_status(name)
    if status == PROVEN:
        return name
    if status == META_GROUNDED:
        return f"{name} [meta_grounded]"
    if status == FAILED:
        return f"{name} [failed]"
    return f"{name} [unvalidated]"


def prefer_proven(candidates: list[str]) -> list[str]:
    rank = {PROVEN: 0, META_GROUNDED: 1, UNVALIDATED: 2, FAILED: 3}

    def key(n: str) -> tuple[int, str]:
        s = validation_status(n)
        return (rank.get(s, 9), n)

    return sorted(candidates, key=key)


def prefer_validated(candidates: list[str]) -> list[str]:
    """Legacy alias for prefer_proven."""
    return prefer_proven(candidates)




def resolve_loadout_after_swaps(
    inventory: dict[str, Any] | None = None,
    shown_deltas: list[dict[str, Any]] | None = None,
    *,
    offense_only: bool = False,
) -> dict[str, Any]:
    """Active O+D names after applying proposed ADD/BENCH/UNBENCH swaps.

    Never expands the bench catalog into the loadout — only mutates the Active set.
    """
    inv = inventory or {}
    active_d = list(inv.get("macros_active") or [])
    active_o = list(
        inv.get("offensive_macros")
        or inv.get("offensive_macros_active")
        or []
    )
    # Default Temple Active-8 when inventory empty
    if not active_d and not active_o and not inventory:
        cat = load_macro_catalog()
        default = cat.get("default_active_user") or {}
        active_d = list(default.get("defense") or [])
        active_o = list(default.get("offense") or [])

    replacing: list[dict[str, str]] = []

    for d in shown_deltas or []:
        if d.get("kind") != "macro":
            continue
        action = (d.get("action") or "").upper()
        target = (d.get("target") or "").upper().strip()
        if not target:
            continue
        meta = get_macro(target) or {}
        target_side = meta.get("side") or "defense"
        swap = d.get("swap_plan") or {}

        if action == "ADD":
            bench = (swap.get("bench") or "").upper().strip()
            bench_side = swap.get("bench_side") or "defense"
            add_side = swap.get("add_side") or target_side
            if bench:
                if bench_side == "defense" and bench in active_d:
                    active_d = [m for m in active_d if m != bench]
                elif bench_side == "offense" and bench in active_o:
                    active_o = [m for m in active_o if m != bench]
                replacing.append(
                    {
                        "bench": bench,
                        "add": target,
                        "bench_side": bench_side,
                        "add_side": add_side,
                    }
                )
            if add_side == "offense":
                if target not in active_o:
                    active_o.append(target)
            else:
                if target not in active_d:
                    active_d.append(target)
        elif action == "BENCH":
            if target in active_d:
                active_d = [m for m in active_d if m != target]
            if target in active_o:
                active_o = [m for m in active_o if m != target]
        elif action == "UNBENCH":
            if target_side == "offense":
                if target not in active_o:
                    active_o.append(target)
            else:
                if target not in active_d:
                    active_d.append(target)

    # Hard cap 8 O+D combined
    total = active_d + active_o
    if len(total) > USER_ACTIVE_CAP:
        # Prefer keeping defense order, then offense — truncate offense first
        keep_d = active_d[:USER_ACTIVE_CAP]
        remain = USER_ACTIVE_CAP - len(keep_d)
        active_d = keep_d
        active_o = active_o[: max(0, remain)]

    if offense_only:
        return {
            "defense": [],
            "offense": list(active_o),
            "replacing": [
                r
                for r in replacing
                if r.get("add_side") == "offense" or r.get("bench_side") == "offense"
            ],
            "offense_only": True,
            "note": "N/A — offense only",
            "total": len(active_o),
            "cap": USER_ACTIVE_CAP,
            "meter": f"Active {len(active_o)}/{USER_ACTIVE_CAP} (O-only)",
        }

    total_n = len(active_d) + len(active_o)
    return {
        "defense": list(active_d),
        "offense": list(active_o),
        "replacing": replacing,
        "offense_only": False,
        "note": "",
        "total": total_n,
        "cap": USER_ACTIVE_CAP,
        "meter": f"Active {total_n}/{USER_ACTIVE_CAP}",
    }


def active_loadout_cards(
    inventory: dict[str, Any] | None = None,
    shown_deltas: list[dict[str, Any]] | None = None,
    *,
    offense_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prep browser cards: ONLY the active loadout (≤8). Never benched/catalog dump.

    If a swap is proposed, cards reflect the 8 *after* the swap; caller shows
    one line 'replacing X with Y'.
    """
    loadout = resolve_loadout_after_swaps(
        inventory, shown_deltas, offense_only=offense_only
    )
    names: list[str] = []
    if not offense_only:
        names.extend(loadout["defense"])
    names.extend(loadout["offense"])
    names = names[:USER_ACTIVE_CAP]

    cards: list[dict[str, Any]] = []
    for mid in names:
        meta = get_macro(mid) or {}
        name = meta.get("name") or mid
        cards.append(
            {
                "id": mid,
                "name": name,
                "side": meta.get("side") or (
                    "offense" if mid in loadout["offense"] else "defense"
                ),
                "purpose": meta.get("purpose") or "",
                "when_to_arm": meta.get("when_to_arm") or "",
                "validated_status": normalize_status(
                    meta.get("validated_status") or PROVEN
                ),
                "slot": "active",
                "copy_block": meta.get("copy_block") or "",
                "full_settings": meta.get("full_settings") or {},
                "xbox_steps": meta.get("xbox_steps") or list(XBOX_PATH),
            }
        )
    return cards, loadout


def catalog_inventory_cards(
    inventory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Full catalog cards (Active + benched + create). Prefer active_loadout_cards for prep UI."""
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
                "validated_status": normalize_status(meta.get("validated_status")),
                "slot": slot,
                "copy_block": meta.get("copy_block") or "",
                "full_settings": meta.get("full_settings") or {},
                "xbox_steps": meta.get("xbox_steps") or list(XBOX_PATH),
            }
        )
    return cards


# ---------------------------------------------------------------------------
# v1.14: CPU (offense-only) loadout chosen from the custom playbook + research
# ---------------------------------------------------------------------------

OFFENSE_MACRO_PRIORITY = ["RZ", "O-RUN", "MATCH", "C2", "MAN", "O-RPO", "ZERO", "PROT", "O-HEAT", "C3", "SHOT"]
# research concept -> offense macros it supports
OFFENSE_MACRO_CONCEPTS = {
    "run_first": ["O-RUN", "MATCH"],
    "inside_zone": ["O-RUN", "MATCH"],
    "duo_power": ["O-RUN", "RZ"],
    "mesh": ["MAN", "C2"],
    "whip": ["MAN"],
    "spot_flat": ["C2", "RZ"],
    "rpo": ["O-RPO"],
    "cpu_gl_wall": ["RZ", "ZERO", "PROT"],
    "dive": ["RZ"],
}


def _macro_plays(meta: dict[str, Any]) -> list[str]:
    import re

    raw = str(((meta.get("full_settings") or {}).get("individual_assignments") or {}).get("value") or "")
    raw = raw.split("—")[0]
    try:
        from cfb_coach.cfb_catalog import formations_with_play
    except Exception:  # noqa: BLE001
        formations_with_play = None  # type: ignore[assignment]
    out = [p.strip() for p in re.split(r"[,/;]", raw) if p.strip()]
    return [p for p in out if p[:1].isupper() and (formations_with_play is None or formations_with_play(p))]


def offense_book_loadout(
    book_plays: dict[str, list[str]] | None,
    *,
    concept_signals: dict[str, int] | None = None,
    rz_signals: dict[str, int] | None = None,
    cap: int = USER_ACTIVE_CAP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Up to ``cap`` offensive custom-adjustment macros for a CPU (offense-only) game.

    Ranked by a fixed doctrine priority, the share of each macro's plays that are in
    the custom playbook, and this prep's research concept signals. Returns cards in
    the same shape as :func:`active_loadout_cards`."""
    import re

    def nk(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    in_book = {nk(p) for ps in (book_plays or {}).values() for p in ps}
    sig: dict[str, float] = {}
    for c, n in (concept_signals or {}).items():
        sig[c] = sig.get(c, 0.0) + float(n or 0)
    for c, n in (rz_signals or {}).items():
        sig[c] = sig.get(c, 0.0) + float(n or 0)
    scored = []
    for i, mid in enumerate(OFFENSE_MACRO_PRIORITY):
        meta = get_macro(mid) or {}
        if not meta or (meta.get("side") or "") != "offense":
            continue
        plays = _macro_plays(meta)
        hits = [p for p in plays if nk(p) in in_book]
        share = len(hits) / len(plays) if plays else 0.0
        research = sum(min(3.0, v) / 3.0 for c, v in sig.items() if mid in OFFENSE_MACRO_CONCEPTS.get(c, []))
        score = (1.0 - 0.05 * i) + 0.3 * share + 0.1 * min(2.0, research)
        why = []
        if plays:
            why.append(f"{len(hits)}/{len(plays)} of its plays in your book")
        if research:
            why.append("backed by this prep's research")
        scored.append((round(score, 3), i, mid, meta, "; ".join(why), hits))
    scored.sort(key=lambda t: (-t[0], t[1]))
    cards = []
    for score, _i, mid, meta, why, hits in scored[:cap]:
        cards.append(attach_offense_detail({
            "id": mid,
            "name": meta.get("name") or mid,
            "side": "offense",
            "purpose": meta.get("purpose") or "",
            "when_to_arm": meta.get("when_to_arm") or "",
            "validated_status": normalize_status(meta.get("validated_status") or PROVEN),
            "slot": "active",
            "copy_block": meta.get("copy_block") or "",
            "full_settings": meta.get("full_settings") or {},
            "xbox_steps": meta.get("xbox_steps") or list(XBOX_PATH),
            "why": why,
            "book_plays": hits,
            "score": score,
        }, book_plays))
    names = [c["id"] for c in cards]
    loadout = {"defense": [], "offense": names, "replacing": [], "offense_only": True,
               "note": "N/A — offense only", "total": len(names), "cap": USER_ACTIVE_CAP,
               "meter": f"Active {len(names)}/{USER_ACTIVE_CAP} (O-only)", "source": "custom playbook + research"}
    return cards, loadout


# ---------------------------------------------------------------------------
# v1.15: exact CFB 27 offense Custom Adjustment settings (drill-down) + live
# macro suggestion (only from the Active 8, only when it adds value)
# ---------------------------------------------------------------------------

_OSET_CACHE: dict[str, Any] | None = None


def load_offense_settings() -> dict[str, Any]:
    """``data/cfb27_offense_macros.json``: ordered in-game rows per offense macro,
    their status (confirmed / assumed / default), citations and live fire rules."""
    global _OSET_CACHE
    if _OSET_CACHE is None:
        import json

        p = Path(__file__).resolve().parent / "data" / "cfb27_offense_macros.json"
        try:
            _OSET_CACHE = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _OSET_CACHE = {"macros": {}, "sections": [], "sources": []}
    return _OSET_CACHE


def _short_value(v: str) -> str:
    import re

    v = re.sub(r"\s*\((?:LS|RS|D-pad|LB|LT|RT|RB)[^)]*\)", "", str(v or ""))
    return v.split(" — ")[0].strip()


def macro_key_settings(mid: str) -> str:
    """One-line summary of the non-default rows (what the live window shows)."""
    m = (load_offense_settings().get("macros") or {}).get(mid) or {}
    bits = []
    for r in m.get("settings") or []:
        if r.get("status") == "default" or str(r.get("value", "")).startswith("Default"):
            continue
        label = {"ID the Mike": "Mike ID", "OL Technique": "OL", "Chip Block — TE": "TE chip", "Chip Block — HB": "HB chip"}.get(r["setting"], r["setting"])
        bits.append(f"{label} {_short_value(r['value'])}")
    if not bits and m.get("settings"):
        bits.append("stock routes; blocking cleanup only")
    return " · ".join(bits)


def _book_pairs(mid: str, book_plays: dict[str, list[str]] | None, *, cap: int = 6) -> list[str]:
    """'Play (Formation)' for plays in his book the macro is built for."""
    import re

    m = (load_offense_settings().get("macros") or {}).get(mid) or {}
    names = list(m.get("pairs_with") or [])
    if not book_plays:
        return names[:cap]

    def nk(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    out: list[str] = []
    seen: set[str] = set()
    for want in names:  # named pairs first, in doctrine order
        for f, ps in book_plays.items():
            for p in ps:
                if nk(p) == nk(want) and nk(p) not in seen:
                    out.append(f"{p} ({f})")
                    seen.add(nk(p))
    rx = m.get("play_re")
    if rx and len(out) < cap:
        for f, ps in book_plays.items():
            for p in ps:
                if nk(p) in seen or not re.search(rx, p, re.I) or not _play_kind_ok(m.get("fire") or {}, p):
                    continue
                out.append(f"{p} ({f})")
                seen.add(nk(p))
    return out[:cap]


def offense_macro_detail(mid: str, book_plays: dict[str, list[str]] | None = None) -> dict[str, Any] | None:
    """Everything the prep drill-down shows for one offense macro, or None if not an
    offense macro with in-game settings on file."""
    data = load_offense_settings()
    m = (data.get("macros") or {}).get(mid)
    if not m:
        return None
    rows = list(m.get("settings") or [])
    cited = {c for r in rows for c in (r.get("cite") or [])}
    for s in data.get("sections") or []:
        cited.update(s.get("cite") or [])
    meta = get_macro(mid) or {}
    return {
        "id": mid,
        "xbox_name": meta.get("xbox_name") or meta.get("name") or mid,
        "editor_path": list(data.get("editor_path") or []),
        "in_game": data.get("in_game") or "",
        "sections": list(data.get("sections") or []),
        "settings": rows,
        "fire_when": m.get("fire_when") or "",
        "pairs_with": _book_pairs(mid, book_plays),
        "key": macro_key_settings(mid),
        "assumed": [r for r in rows if r.get("status") == "assumed"],
        "sources": [s for s in data.get("sources") or [] if s.get("id") in cited],
        "limits": data.get("limits") or "",
    }


def offense_copy_block(detail: dict[str, Any]) -> str:
    """Tick-by-tick checklist in the in-game order (every row, defaults included)."""
    name = detail.get("xbox_name") or detail.get("id")
    lines = [f"MACRO: {name} (offense)", "Path: " + " > ".join(detail.get("editor_path") or []), f"[ ] Name: {name}"]
    cur = None
    for r in detail.get("settings") or []:
        if r["section"] != cur:
            cur = r["section"]
            lines.append(cur + (" (per depth-chart slot)" if cur == "Hot Routes" else ""))
        lines.append(f"  [ ] {r['setting']}: {r['value']}")
    lines.append("[ ] Save -> set Active (Aidan cap 8)")
    lines.append(f"In game: LB -> {name}")
    if detail.get("fire_when"):
        lines.append(f"Fire when: {detail['fire_when']}")
    if detail.get("pairs_with"):
        lines.append("Pairs with: " + ", ".join(detail["pairs_with"]))
    return "\n".join(lines)


def attach_offense_detail(card: dict[str, Any], book_plays: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Add ``ingame`` (drill-down) to an offense macro card and regenerate its copy block."""
    if (card.get("side") or "") != "offense":
        return card
    det = offense_macro_detail(str(card.get("id") or ""), book_plays)
    if det:
        card["ingame"] = det
        card["copy_block"] = offense_copy_block(det)
        card["book_plays"] = det["pairs_with"] if book_plays else card.get("book_plays") or []
    return card


# --- live suggestion ----------------------------------------------------------

def classify_coverage(cov: str | None) -> set[str]:
    c = (cov or "").lower()
    out: set[str] = set()
    if not c:
        return out
    if "cover 0" in c or "zero" in c:
        out.add("c0")
    if "pressure" in c or "blitz" in c:
        out.add("pressure")
    if "cover 1" in c or "man" in c:
        out.add("man")
    if "cover 2" in c or "invert" in c or "tampa" in c:
        out.update({"c2", "two_high"})
    if "cover 3" in c or "sky" in c or "buzz" in c:
        out.add("c3")
    if any(x in c for x in ("cover 4", "quarters", "palms", "cover 6", "cover 9", "match")):
        out.update({"match", "two_high"})
    if "two-high" in c or "two high" in c or "split" in c:
        out.add("two_high")
    return out


def _play_kind_ok(fire: dict[str, Any], play: str) -> bool:
    import re

    from cfb_coach.cfb_catalog import is_run

    rpo = bool(re.search(r"rpo", play or "", re.I))
    if fire.get("rpo"):
        return rpo
    if fire.get("run"):
        return is_run(play) and not rpo
    if fire.get("pass"):
        return not is_run(play) and not rpo
    return True


# Priority when several macros fit the same snap (most specific answer first)
LIVE_MACRO_PRIORITY = ["ZERO", "O-HEAT", "PROT", "MAN", "C2", "C3", "MATCH", "O-RPO", "O-RUN", "RZ", "SHOT"]
LEARNED_SUPPRESS = -0.15  # capped macro weight at/below this for this opponent → stop suggesting it


def suggest_offense_macro(
    *,
    zone: str,
    play: str,
    formation: str = "",
    coverage: str | None = None,
    coverage_source: str = "none",
    active: list[str] | None = None,
    down: int | None = None,
    repeated: bool = False,
    weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Pick at most one offense macro for this snap, or None.

    Only macros in ``active`` (the current Active 8) are ever returned. Coverage macros
    need a live pre-snap look (or a look repeated in this situation) — a last-snap
    coverage alone never fires one. RZ is situational (inside the 20 / goal-to-go on a
    paired pass). Plays must fit the macro (pass macros on passes, RUN on runs, RPO on
    RPOs, and the play must be one the macro is built for)."""
    import re

    act = [a for a in (active or []) if a]
    if not act or not play:
        return None
    data = load_offense_settings().get("macros") or {}
    cls = classify_coverage(coverage)
    cov_ok = bool(cls) and (coverage_source == "live" or repeated)
    have_heat = any(a in act for a in ("O-HEAT", "PROT"))
    for mid in LIVE_MACRO_PRIORITY:
        if mid not in act or mid not in data:
            continue
        m = data[mid]
        fire = m.get("fire") or {}
        if zone not in (fire.get("zones") or ["open", "rz", "gl"]):
            continue
        if not _play_kind_ok(fire, play):
            continue
        named = any(re.sub(r"[^a-z0-9]", "", p.lower()) == re.sub(r"[^a-z0-9]", "", play.lower()) for p in m.get("pairs_with") or [])
        if not named and not (m.get("play_re") and re.search(m["play_re"], play, re.I)):
            continue
        if fire.get("downs") and down not in fire["downs"]:
            continue
        want = set(fire.get("coverages") or [])
        if want:
            hit = cov_ok and bool(cls & want)
            if not hit and fire.get("fallback_pressure") and not have_heat:
                hit = cov_ok and "pressure" in cls
            if not hit:
                continue
        if any(a.get("coverage") in cls and re.search(a.get("play_re") or "$^", play, re.I) for a in fire.get("avoid") or []):
            continue
        w = (weights or {}).get(mid)
        if w is not None and w <= LEARNED_SUPPRESS:
            continue
        meta = get_macro(mid) or {}
        trig = f"{coverage_source} {coverage} look" if want else f"{'goal-to-go' if zone == 'gl' else 'red zone'} pass"
        return {
            "id": mid,
            "name": meta.get("xbox_name") or meta.get("name") or mid,
            "why": f"{trig} on {play} ({(meta.get('purpose') or m.get('fire_when', '').split('.')[0]).replace(' — ', ': ')})",
            "key": macro_key_settings(mid),
            "settings": [r for r in m.get("settings") or [] if r.get("status") != "default"
                         and not str(r.get("value", "")).startswith("Default")] or list(m.get("settings") or []),
            "fire_when": m.get("fire_when") or "",
            "learned_weight": w,
        }
    return None


def active_offense_macros(db: Any, opponent_id: str) -> list[str]:
    """The current offense Active 8 for live calls: what the last prep showed (stored
    in meta), else the CPU loadout from the playbook of record, else the dynasty's
    offensive Active list. Never anything outside that list."""
    import json

    if db is None:
        return []
    try:
        raw = db.get_meta(f"active_macros_o:{opponent_id}")
        if raw:
            ids = [_offense_id(str(x)) for x in (json.loads(raw).get("offense") or [])]
            return [i for i in ids if i][:USER_ACTIVE_CAP]
    except Exception:  # noqa: BLE001
        pass
    try:
        from cfb_coach.opponents import is_cpu_opponent

        if is_cpu_opponent(opponent_id):
            from cfb_coach.cfb_playbook import callable_book

            book = callable_book(db) or {}
            if book.get("formations"):
                _cards, lo = offense_book_loadout(book["formations"])
                return list(lo.get("offense") or [])[:USER_ACTIVE_CAP]
    except Exception:  # noqa: BLE001
        pass
    try:
        lo = resolve_loadout_after_swaps(None)
        return [i for i in (_offense_id(m) for m in (lo.get("offense") or [])) if i][:USER_ACTIVE_CAP]
    except Exception:  # noqa: BLE001
        return []


def _offense_id(x: str) -> str | None:
    """Catalog id of an offense macro given its id or Xbox name ('RUN' -> 'O-RUN')."""
    x = (x or "").strip().upper()
    for cand in (x, f"O-{x}"):
        meta = (load_macro_catalog().get("macros") or {}).get(cand)
        if meta and (meta.get("side") or "") == "offense":
            return cand
    return None


def store_active_offense_macros(db: Any, opponent_id: str, ids: list[str]) -> None:
    import json
    from datetime import datetime, timezone

    if db is None:
        return
    db.set_meta(f"active_macros_o:{opponent_id}", json.dumps(
        {"offense": list(ids)[:USER_ACTIVE_CAP], "ts": datetime.now(timezone.utc).isoformat()}))
