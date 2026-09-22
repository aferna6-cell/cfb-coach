"""Formation family from player centroids — classical geometry, pure Python."""

from __future__ import annotations

from typing import Iterable, Sequence

# (x, y) normalized: x left→right [0,1], y near→far [0,1]; offense near bottom.
PlayerXY = tuple[float, float]


def _ball_x(offense: list[PlayerXY]) -> float:
    """Estimate ball / center from OL-ish cluster near midfield x."""
    olish = [
        p
        for p in offense
        if 0.30 <= p[0] <= 0.70 and 0.25 <= p[1] <= 0.42
    ]
    if olish:
        return sum(p[0] for p in olish) / len(olish)
    near_hash = [p for p in offense if abs(p[0] - 0.5) < 0.12]
    if near_hash:
        return sum(p[0] for p in near_hash) / len(near_hash)
    xs = sorted(p[0] for p in offense)
    return xs[len(xs) // 2]


def _span(group: list[PlayerXY]) -> float:
    if len(group) < 2:
        return 0.0
    gx = [p[0] for p in group]
    return max(gx) - min(gx)


def classify_formation(
    players: Sequence[PlayerXY] | Iterable[PlayerXY],
    *,
    conf_threshold: float = 0.35,
) -> tuple[str, str, float]:
    """Classify offense formation family from WR/skill centroids.

    Returns (formation, side, confidence) where formation is one of
    bunch|trips|empty|2x2|other|unknown and side is L|R|"".
    """
    pts = [(float(x), float(y)) for x, y in players]
    if len(pts) < 3:
        return "unknown", "", 0.15

    offense = [p for p in pts if p[1] < 0.55]
    if len(offense) < 3:
        offense = list(pts)

    ball_x = _ball_x(offense)

    skill = [
        p
        for p in offense
        if abs(p[0] - ball_x) > 0.06 or p[1] < 0.28
    ]
    if len(skill) < 3:
        skill = [p for p in offense if abs(p[0] - ball_x) > 0.03]
    if len(skill) < 3:
        skill = list(offense)

    left = [p for p in skill if p[0] < ball_x - 0.03]
    right = [p for p in skill if p[0] > ball_x + 0.03]

    near_backfield = [
        p for p in offense if abs(p[0] - ball_x) < 0.08 and p[1] < 0.40
    ]
    wide = [p for p in skill if abs(p[0] - ball_x) > 0.16]

    # 1) Bunch: 3+ tight on one side
    for side_label, group in (("L", left), ("R", right)):
        if len(group) >= 3 and _span(group) <= 0.10:
            return "bunch", side_label, 0.75

    # 2) Empty: many wide, sparse backfield (before trips — empty looks 3-wide)
    if len(wide) >= 5 and len(near_backfield) <= 1:
        return "empty", "", 0.7
    if (
        len(wide) >= 4
        and len(near_backfield) <= 1
        and len(left) >= 2
        and len(right) >= 2
        and len(left) + len(right) >= 5
    ):
        return "empty", "", 0.7

    # 3) Trips: 3+ on one side, looser
    for side_label, group in (("L", left), ("R", right)):
        if len(group) >= 3 and _span(group) <= 0.35:
            return "trips", side_label, 0.65

    # 4) 2x2 balanced
    if len(left) >= 2 and len(right) >= 2 and abs(len(left) - len(right)) <= 1:
        return "2x2", "", 0.7

    if len(left) >= 3 or len(right) >= 3:
        side = "L" if len(left) >= len(right) else "R"
        return "other", side, 0.45

    if len(skill) >= 4:
        return "other", "", 0.4

    conf = 0.25
    if conf < conf_threshold:
        return "unknown", "", conf
    return "other", "", conf
