"""No-input, non-executed look-conditional advice for a live Madden offensive call.

A conditional note is NOT a current coverage observation, a selected adjustment,
or evidence that the user executed one. The main formation/play remains unchanged.
No clock or live "showing" input is required. Only already-armed, confirmed,
play-compatible Custom Adjustments can appear.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

# Human-readable conditional label, synthetic *hypothetical* look for eligibility.
_LOOKS = (
    ("Cover 2", "Cover 2"),
    ("Cover 6", "Cover 6"),
    ("Cover 3", "Cover 3 Sky"),
    ("Cover 1 / man", "Cover 1"),
    ("Cover 4", "Cover 4 Quarters"),
    ("Pressure", "pressure"),
)


def _prior_look_counts(db: Any, opponent_id: str) -> Counter[str]:
    """Historical observations prioritize hints, never assert this snap's look."""
    counts: Counter[str] = Counter()
    if not db or not opponent_id:
        return counts
    try:
        rows = db.get_recent_snaps(opponent_id, side="offense", limit=40)
    except Exception:  # noqa: BLE001 - optional hint cannot block live play
        return counts
    for row in rows:
        try:
            seen = str(row["coverage_seen"] or "").lower()
        except (TypeError, KeyError, IndexError):
            continue
        if "cover 2" in seen or "tampa 2" in seen:
            counts["Cover 2"] += 1
        elif "cover 6" in seen or "cover 9" in seen or "quarters" in seen:
            counts["Cover 6"] += 1
        elif "cover 3" in seen:
            counts["Cover 3"] += 1
        elif "cover 1" in seen or "man" in seen:
            counts["Cover 1 / man"] += 1
        elif "cover 4" in seen:
            counts["Cover 4"] += 1
        elif "pressure" in seen or "blitz" in seen or "cover 0" in seen:
            counts["Pressure"] += 1
    return counts


def suggest_conditional_action(
    call: Any,
    sit: Any,
    *,
    book: Mapping[str, Sequence[str]],
    active: Sequence[str],
    db: Any = None,
    opponent_id: str = "",
) -> dict[str, Any] | None:
    """Return at most one optional IF-look macro, without selecting or executing it.

    Requires a confirmed, armed macro and an exact formation+play pairing.
    This function deliberately does NOT set call.macro/call.adjustment and
    deliberately does NOT interpret the most recent coverage as a live look.
    """
    if getattr(call, "side", None) != "offense":
        return None
    if getattr(call, "macro", None) or getattr(call, "adjustment", None):
        return None
    if getattr(sit, "coverage_source", None) == "live":
        return None
    form, play = str(getattr(call, "formation", "") or ""), str(getattr(call, "play", "") or "")
    if play not in book.get(form, ()):
        return None

    from cfb_coach.madden.offense_macros import (
        clean_ids, offense_detail, pairs_in_book, suggest_for_snap,
    )

    armed = clean_ids(list(active or []))
    if not armed:
        return None
    zone = ("gl" if getattr(sit, "goal_line", False) else
            "rz" if getattr(sit, "red_zone", False) else "open")
    full_cap = 1 + sum(len(plays) for plays in book.values())
    pairs = f"{play} ({form})"
    matches: list[dict[str, Any]] = []
    for condition, hypothetical in _LOOKS:
        for mid in armed:
            if pairs not in pairs_in_book(mid, dict(book), cap=full_cap):
                continue
            detail = offense_detail(mid, dict(book))
            if detail.get("needs_settings") or detail.get("gaps") or not detail.get("settings"):
                continue
            buttons = str(detail.get("buttons") or "")
            if not buttons or any(token in buttons.upper() for token in ("VERIFY", "UNKNOWN", "NOT CONFIRMED")):
                continue
            # "live" is used *inside this hypothetical evaluation only*; the
            # actual situation and its coverage_source are never modified.
            suggestion = suggest_for_snap(
                zone=zone, play=play, coverage=hypothetical,
                coverage_source="live", active=[mid],
                down=getattr(sit, "down", None), repeated=False,
                book=dict(book),
            )
            if not suggestion or suggestion.get("id") != mid or suggestion.get("kind") != "look":
                continue
            matches.append({
                "condition": condition, "name": detail.get("xbox_name") or mid,
                "buttons": buttons, "macro_id": mid,
                "conditional_only": True, "executed": False,
                "source": "verified_armed_macro_hypothetical_look",
            })
            break
    if not matches:
        return None
    counts = _prior_look_counts(db, opponent_id)
    # A commonly logged coverage makes for a more useful IF clause, without
    # pretending the logged look is the current defense.
    return min(matches, key=lambda x: (-counts[x["condition"]], next(
        i for i, (label, _) in enumerate(_LOOKS) if label == x["condition"]
    )))
