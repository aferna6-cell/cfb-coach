"""Opt-in, data-only Civil match + Nickel 2-4 Wide user-role safety adapter.

No live-call switch, ML training or implicit guide setup. A guide USER role is
valid only for its exact formation/play/opponent alignment and assumed check.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "madden27.civil.match_nickel_combined.v1"
DATASET = Path(__file__).resolve().parents[2] / "datasets/madden27/civil_match_nickel_combined.json"


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    raw = json.loads(Path(path or DATASET).read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError("unexpected defense guide catalog schema")
    entries = raw.get("entries")
    if not isinstance(entries, list) or len({e.get("id") for e in entries}) != len(entries):
        raise ValueError("invalid or duplicate defense guide IDs")
    for e in entries:
        for key in ("id", "formation", "play", "offense_shape", "source", "user_evidence"):
            if not isinstance(e.get(key), str) or not e[key]:
                raise ValueError("missing defense guide provenance/identity: " + key)
        if e.get("user_evidence") == "explicit_user" and not e.get("user"):
            raise ValueError("explicit USER evidence is missing the USER role")
        if e.get("stock_first_eligible") and (e.get("user_evidence") != "explicit_user" or e.get("additional_setup")):
            raise ValueError("stock eligible recipe has no explicit user or requires hidden setup")
    return raw


def resolve_user(
    formation: str,
    play: str,
    offense_shape: str,
    installed_book: Mapping[str, Sequence[str]],
    *,
    owner_match_checks_configured: bool = False,
    active_coaching_profile: str | None = None,
    confirmed_macros: Sequence[str] = (),
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Provide exactly the user/macro or a blocking reason; NEVER invent roles.

    Only configured match-check stock cards are currently eligible. The Nickel
    screenshot packages require manual transformations beyond the named macro,
    so are intentionally blocked from stock-first mode until implemented.
    """
    cat = catalog if catalog is not None else load_catalog()
    if cat.get("schema") != SCHEMA:
        raise ValueError("unknown defense guide schema")
    if play not in installed_book.get(formation, ()):
        return {"ok": False, "reason": "NOT_INSTALLED"}
    matched = [e for e in cat["entries"] if e["formation"] == formation and e["play"] == play and e["offense_shape"] == offense_shape]
    if not matched:
        return {"ok": False, "reason": "NO_MATCHING_GUIDE_USER_FOR_THIS_ALIGNMENT"}
    eligible = []
    blockers = set()
    for e in matched:
        if e.get("required_coaching_profile") != active_coaching_profile:
            blockers.add("COACHING_ADJUSTMENT_PROFILE_UNCONFIRMED_OR_INCOMPATIBLE")
        elif e.get("user_evidence") != "explicit_user" or not e.get("user"):
            blockers.add("USER_NOT_EXPLICIT_IN_GUIDE")
        elif not e.get("stock_first_eligible") or e.get("additional_setup"):
            blockers.add("REQUIRES_UNDISPLAYED_MANUAL_SETUP")
        elif e.get("check") and e["kind"] == "match_guide" and not owner_match_checks_configured:
            blockers.add("MATCH_CHECKS_NOT_CONFIRMED")
        elif e.get("macro") and e["macro"] not in confirmed_macros:
            blockers.add("MACRO_NOT_CONFIRMED")
        else:
            eligible.append(e)
    if len(eligible) != 1:
        return {"ok": False, "reason": "AMBIGUOUS_GUIDE_USER" if eligible else ",".join(sorted(blockers))}
    e = eligible[0]
    return {"ok": True, "formation": formation, "play": play, "user": e["user"],
            "macro": e.get("macro"), "check": e.get("check"), "guide_id": e["id"],
            "user_provenance": "GUIDE_EXPLICIT_NOT_XBOX_VERIFIED"}


def compact_card(resolved: Mapping[str, Any]) -> str:
    """Only play, who to user, applicable macro. No implicit omitted setup."""
    if resolved.get("ok") is not True:
        raise ValueError("cannot display unverified USER assignment")
    text = f"PLAY: {resolved['play']} ({resolved['formation']})\nUSER: {resolved['user']}"
    if resolved.get("macro"):
        text += f"\nMACRO: {resolved['macro']}"
    return text


__all__ = ["load_catalog", "resolve_user", "compact_card"]
