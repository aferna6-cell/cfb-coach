"""Unified next-situation tracking using the ESTABLISHED HTML spot engine.

Historically the established HTML controller used cfb_coach.last_snap.advance_ball
to determine the new down/distance/yardline after an outcome. Reuse that code
rather than maintaining a separate, divergent game-rule implementation.

The result is a *suggestion* for the next call. A user-supplied situation is
always accepted for a possession change, touchdown, turnover, or correction.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from cfb_coach.last_snap import BallSpot, advance_ball
from cfb_coach.outcome import parse_outcome
from cfb_coach.madden.unified.types import OFF, GameState

# The old HTML pad asked for the next down after these results. They do not
# reliably identify who received a kickoff, which field position to use, or
# whether the game has a special scoring rule.
_CHANGE_OF_POSSESSION = re.compile(
    r"\b(td|touchdown|int|interception|intercepted|fumble|turnover|safety|punt|field\s+goal|fg|first\s+down|convert(?:ed)?)\b",
    re.I,
)


def next_situation(st: GameState, result_text: str) -> dict[str, Any]:
    raw = (result_text or "").strip()
    if not raw:
        return {"automatic": False, "reason": "Enter the previous result first."}
    p = parse_outcome(raw)
    if p.kind in {"td", "int", "fumble", "stop", "convert", "unknown"} or _CHANGE_OF_POSSESSION.search(raw):
        return {"automatic": False, "reason": "Possession or spot may have changed — enter the next situation, then Submit."}
    if p.kind == "sack" and p.yards is None:
        return {"automatic": False, "reason": "Enter sack yardage, or set the next down and spot manually."}
    if st.yardline is None or st.down is None or st.distance is None:
        return {"automatic": False, "reason": "Down, distance or field position is unavailable; enter the next situation."}

    # The old HTML tracker uses the owner's goal-line coordinate; the shared
    # world model stores a coordinate from whichever team is on offense.
    owner_yl = st.yardline if st.our_side == OFF else 100 - st.yardline
    sit = SimpleNamespace(down=st.down, distance=st.distance, yardline=owner_yl)
    prev = BallSpot(down=st.down, distance=st.distance, yardline=owner_yl)
    nxt, message = advance_ball(sit, raw, st.our_side, prev)
    if nxt.down is None or nxt.distance is None:
        return {"automatic": False, "reason": message or "Change of possession — enter the next down and spot."}
    yl = nxt.yardline
    # Don't clamp end-zone plays to the one-yard line in the unified model.
    if p.yards is not None and not 1 <= st.yardline + p.yards <= 99:
        return {"automatic": False, "reason": "Goal line crossed — confirm the new possession and spot."}
    if yl is None or not 1 <= yl <= 99:
        return {"automatic": False, "reason": "Ball position unavailable — enter the next situation."}
    return {
        "automatic": True,
        "side": st.our_side,
        "down": nxt.down,
        "distance": nxt.distance,
        "yardline_side": "my" if yl <= 50 else "opp",
        "yardline": yl if yl <= 50 else 100 - yl,
        "offense_yardline": yl if st.our_side == OFF else 100 - yl,
        "yards": p.yards,
        "reason": "Tracked using the established HTML ball-spot engine.",
    }


def next_situation_line(st: GameState, result_text: str) -> tuple[str | None, str]:
    predicted = next_situation(st, result_text)
    if not predicted["automatic"]:
        return None, predicted["reason"]
    side = "o" if st.our_side == OFF else "d"
    return (f"{side} {predicted['down']}&{predicted['distance']} "
            f"{predicted['yardline_side']} {predicted['yardline']}"), ""
