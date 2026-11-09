"""Conservative next-down/spot projection for unified live HTML.

The pending GameState is the source of truth. Its yardline always counts
from the *offense's own goal*, even when the owner is playing defense.
This module never infers a new possession after TD/turnover/downs.
"""
from __future__ import annotations

import re
from typing import Any

from cfb_coach.outcome import parse_outcome
from cfb_coach.madden.unified.types import DEF, OFF, GameState


def next_situation(st: GameState, result_text: str) -> dict[str, Any]:
    """Suggest the next snap on the SAME possession, or require manual entry.

    Does not write to a game, recommend a play, or mark an execution.
    """
    raw = (result_text or "").strip()
    if not raw:
        return {"automatic": False, "reason": "Enter the last play's result first."}
    p = parse_outcome(raw)
    if p.kind in {"td", "int", "fumble", "stop", "convert", "unknown"} or re.search(
        r"\b(td|touchdown|interception|turnover|safety|punt|first\s+down|convert(?:ed)?)\b", raw, re.I
    ):
        return {"automatic": False, "reason": "Confirm the next possession, down and ball spot manually after this result."}
    if p.kind == "sack" and p.yards is None:
        return {"automatic": False, "reason": "Enter the sack yardage (e.g. 'sack -6') or set the next spot manually."}
    if p.yards is None:
        return {"automatic": False, "reason": "The yardage is unknown; enter the next situation manually."}
    if st.yardline is None or st.distance is None or st.down is None:
        return {"automatic": False, "reason": "The current snap is missing down, distance or ball position."}
    gain = int(p.yards)
    pos = st.yardline + gain
    if pos <= 0 or pos >= 100:
        return {"automatic": False, "reason": "A goal-line score or safety may have occurred; confirm the new possession and spot."}
    if st.down == 4 and gain < st.distance:
        return {"automatic": False, "reason": "Fourth-down failure: verify the next possession and ball spot manually."}

    if gain >= st.distance:
        down = 1
        distance = min(10, 100 - pos)
    else:
        down = st.down + 1
        distance = min(max(1, st.distance - gain), 100 - pos)

    # The interface always labels 'my' and 'opp' from the owner's perspective.
    our_coordinate = pos if st.our_side == OFF else 100 - pos
    territory = "my" if our_coordinate <= 50 else "opp"
    spot = our_coordinate if our_coordinate <= 50 else 100 - our_coordinate
    result = {
        "automatic": True,
        "side": st.our_side,
        "down": down,
        "distance": distance,
        "yardline_side": territory,
        "yardline": spot,
        "offense_yardline": pos,
        "yards": gain,
        "reason": "Same possession: projected from the previous snap.",
    }
    return result


def next_situation_line(st: GameState, result_text: str) -> tuple[str | None, str]:
    """Convert to the exact current unified input parser's syntax."""
    projected = next_situation(st, result_text)
    if not projected["automatic"]:
        return None, projected["reason"]
    side = "o" if st.our_side == OFF else "d"
    return (f"{side} {projected['down']}&{projected['distance']} "
            f"{projected['yardline_side']} {projected['yardline']}"), ""
