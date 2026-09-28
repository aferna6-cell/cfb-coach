"""Madden 27 macro pool (v1.17): the Madden research macros + Aidan's CFB 27 macros.

Madden 27 and CFB 27 share the Custom Adjustments editor, so every CFB macro is also a
Madden macro. A Madden catalog macro with the SAME name on the same side as a CFB macro
(HEAT, O-RPO) is one macro — one pool entry, CFB settings. Names are never changed:
``O-MAN`` and ``MAN`` are two different macros.

Pool entries carry only descriptive metadata (side, purpose, when to arm, families).
Settings always come from ``cfb_coach.macro_settings`` — never from here.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from cfb_coach.madden.data import get_macro as madden_catalog_macro
from cfb_coach.madden.data import load_macro_catalog

# What each CFB macro answers, from its catalog purpose line. Defense = the Madden concept
# families (`madden.situation.concept_family`); offense = the live look it answers.
CFB_FAMILIES: dict[str, list[str]] = {
    "CROSS": ["cross"],
    "VERT": ["vert"],
    "BUNCH": ["stack", "cross"],
    "RPO": ["rpo"],
    "SCRAM": ["scram"],
    "RUN-IN": ["run"],
    "RUN-OUT": ["run"],
    "HEAT": ["pressure"],
    "FLOOD": ["flood"],
    "SCREEN": ["screen", "rpo"],
    "GLASS": ["cross", "stack"],
    "CONTAIN-SCRAM": ["scram"],
    "SPOT-LOCK": ["cross"],
    "PROT": ["pressure"],
    "O-HEAT": ["pressure"],
    "ZERO": ["pressure", "cover0"],
    "MAN": ["man"],
    "C3": ["single_high"],
    "C2": ["cover2"],
    "MATCH": ["two_high"],
    "SHOT": ["single_high", "man"],
    "O-RUN": ["two_high_run"],
    "O-RPO": ["rpo"],
    "RZ": ["red_zone"],
}
MADDEN_O_FAMILIES = {"O-PROT": ["pressure"], "O-MAN": ["man"], "O-RPO": ["rpo"]}


def _cfb_catalog() -> dict[str, Any]:
    from cfb_coach.macros import load_macro_catalog as cfb_catalog

    return cfb_catalog().get("macros") or {}


@lru_cache(maxsize=1)
def _pool() -> dict[str, dict[str, Any]]:
    from cfb_coach.macro_settings import cfb_match

    out: dict[str, dict[str, Any]] = {}
    matched: set[str] = set()
    for mid, m in (load_macro_catalog().get("macros") or {}).items():
        side = m.get("side") or "defense"
        cfb_id = cfb_match(mid, side)
        if cfb_id:
            matched.add(cfb_id)
        fam = MADDEN_O_FAMILIES.get(mid) if side == "offense" else (
            [m["concept_family"]] if m.get("concept_family") else [])
        out[mid] = dict(copy.deepcopy(m), id=mid, side=side, origin="madden", cfb_match=cfb_id,
                        families=fam or CFB_FAMILIES.get(cfb_id or "", []))
    for cid, m in _cfb_catalog().items():
        if cid in matched or cid in out:
            continue
        side = m.get("side") or "defense"
        out[cid] = {
            "id": cid,
            "name": m.get("name") or cid,
            "xbox_name": m.get("xbox_name") or m.get("name") or cid,
            "side": side,
            "purpose": m.get("purpose") or "",
            "when_to_arm": m.get("when_to_arm") or "",
            # validation is per game: CFB "proven" was earned in CFB — Madden starts meta_grounded
            "validated_status": "meta_grounded",
            "source": "CFB 27 macro (same Custom Adjustments editor) — " + str(m.get("source") or ""),
            "origin": "cfb",
            "cfb_match": cid,
            "families": list(CFB_FAMILIES.get(cid, [])),
        }
    return out


def pool_ids(side: str | None = None) -> list[str]:
    """Every Madden macro id (Madden catalog order, then CFB catalog order)."""
    return [k for k, v in _pool().items() if side is None or v["side"] == side]


def pool_macro(name: str | None) -> dict[str, Any] | None:
    key = (name or "").strip().upper()
    m = _pool().get(key)
    return copy.deepcopy(m) if m else None


def macro_side(name: str | None) -> str:
    return (pool_macro(name) or {}).get("side") or "defense"


def families(name: str | None) -> list[str]:
    return list((pool_macro(name) or {}).get("families") or [])


def display_name(name: str | None) -> str:
    m = pool_macro(name) or {}
    return str(m.get("xbox_name") or m.get("id") or name or "")


def is_madden_catalog(name: str | None) -> bool:
    return madden_catalog_macro(name) is not None


__all__ = ["CFB_FAMILIES", "display_name", "families", "is_madden_catalog", "macro_side", "pool_ids", "pool_macro"]
