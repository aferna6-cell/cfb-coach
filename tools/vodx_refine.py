"""Snap refine entry used by vodx.py extract_one.

The box copy of this module runs the side, score, and possession passes and
then ``vodx_quality.apply_quality``. This file is that quality call, plus the
clock and spot helpers ``apply_quality`` imports when the box module is not
the one on the path.
"""

from __future__ import annotations

import re

from vodx_quality import apply_quality


def refine_rows(rows: list[dict], frames, game: str | None = None):
    del game
    return apply_quality(rows, frames)


def clock_s(clock) -> int | None:
    match = re.match(r"(\d{1,2}):(\d{2})", clock or "")
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def clocks_ok(prev: dict, nxt: dict) -> bool:
    """Missing clocks are compatible. A clock that runs backward a lot is not."""
    left, right = clock_s(prev.get("clock")), clock_s(nxt.get("clock"))
    if left is None or right is None:
        return True
    return -2 <= (left - right) <= 50


def spot_deltas(s0, s1) -> set[int]:
    """Yard change between two field spots, both directions when the side is unknown."""

    def numbers(spot) -> list[int]:
        if isinstance(spot, dict):
            spot = spot.get("yardline", spot.get("spot", spot.get("value")))
        if isinstance(spot, bool) or spot is None or spot == "":
            return []
        if isinstance(spot, (int, float)):
            return [int(spot)]
        text = str(spot).strip()
        side = ""
        if text.lower().startswith("own"):
            side, text = "own", text[3:]
        elif text.lower().startswith("opp"):
            side, text = "opp", text[3:]
        try:
            number = int(text.strip())
        except ValueError:
            return []
        if side == "opp" and 1 <= number <= 50:
            return [100 - number]
        return [number]

    out: set[int] = set()
    for left in numbers(s0):
        for right in numbers(s1):
            out.add(right - left)
    return out
