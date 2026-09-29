"""Madden 27 macro pool (v1.17): the DEFENSE macros in the research DB.

Offense uses no macros — offense calls carry pre-snap adjustments (audibles, hot routes,
protection) instead (``madden/adjustments.py``). Defense macros come from the daily research
routine (``madden/research_db.py``), each with research-built settings for every editor field.
"""

from __future__ import annotations

from typing import Any

from cfb_coach.madden import research_db as rdb


def _entry(m: dict[str, Any]) -> dict[str, Any]:
    base = m.get("base") or {}
    return {
        "id": m["id"].upper(),
        "name": m.get("name") or m["id"],
        "xbox_name": m.get("name") or m["id"],
        "side": "defense",
        "purpose": m.get("purpose") or "",
        "when_to_arm": m.get("when") or "",
        "families": list(m.get("answers") or []),
        "meta_rank": int(m.get("meta_rank") or 99),
        "shell_pair": " — ".join(x for x in (base.get("formation"), base.get("play")) if x),
        "base": dict(base),
        "validated_status": "meta_grounded",
        "sources": list(m.get("sources") or []),
        "origin": "research",
    }


def pool_ids(side: str | None = None) -> list[str]:
    if side == "offense":
        return []
    return [m["id"].upper() for m in rdb.defense_macros()]


def pool_macro(name: str | None) -> dict[str, Any] | None:
    m = rdb.defense_macro(name)
    return _entry(m) if m else None


def macro_side(name: str | None) -> str:
    return "defense" if pool_macro(name) else "offense"


def families(name: str | None) -> list[str]:
    return list((pool_macro(name) or {}).get("families") or [])


def display_name(name: str | None) -> str:
    m = pool_macro(name) or {}
    return str(m.get("xbox_name") or name or "")


__all__ = ["display_name", "families", "macro_side", "pool_ids", "pool_macro"]
