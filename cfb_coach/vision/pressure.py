"""Coarse pressure / blitz side from defender centroids — pure Python."""

from __future__ import annotations

from typing import Iterable, Sequence

PlayerXY = tuple[float, float]


def classify_pressure(
    defenders: Sequence[PlayerXY] | Iterable[PlayerXY],
    *,
    los_y: float = 0.5,
    conf_threshold: float = 0.35,
) -> tuple[str, float]:
    """Coarse pressure: none|show|left|right|middle|all_out|unknown.

    Heuristic: count defenders crowded near LOS (y near los_y) vs dropped.
    Side bias from x relative to 0.5.
    """
    pts = [(float(x), float(y)) for x, y in defenders]
    if len(pts) < 4:
        return "unknown", 0.2

    # Near LOS rushers
    near = [p for p in pts if abs(p[1] - los_y) < 0.08]
    deep = [p for p in pts if p[1] > los_y + 0.15]

    n_near = len(near)
    if n_near <= 4 and len(deep) >= 3:
        return "none", 0.55
    if n_near == 5:
        return "show", 0.5
    if n_near >= 7:
        return "all_out", 0.7
    if n_near >= 6:
        # side bias
        left_n = sum(1 for p in near if p[0] < 0.42)
        right_n = sum(1 for p in near if p[0] > 0.58)
        mid_n = n_near - left_n - right_n
        if left_n >= right_n + 2 and left_n >= mid_n:
            return "left", 0.6
        if right_n >= left_n + 2 and right_n >= mid_n:
            return "right", 0.6
        return "middle", 0.55

    conf = 0.3
    if conf < conf_threshold:
        return "unknown", conf
    return "show", conf


def classify_shell(
    defenders: Sequence[PlayerXY] | Iterable[PlayerXY],
    *,
    conf_threshold: float = 0.35,
) -> tuple[str, float]:
    """one_high vs two_high from deepest defenders (safeties)."""
    pts = [(float(x), float(y)) for x, y in defenders]
    if len(pts) < 2:
        return "unknown", 0.15

    # Deepest = highest y
    by_depth = sorted(pts, key=lambda p: p[1], reverse=True)
    deep = by_depth[:4]
    # Safeties roughly y > 0.7
    safeties = [p for p in deep if p[1] >= 0.65]
    if len(safeties) >= 2:
        xs = sorted(p[0] for p in safeties[:2])
        if abs(xs[0] - xs[1]) > 0.18:
            return "two_high", 0.65
        return "one_high", 0.55
    if len(safeties) == 1:
        return "one_high", 0.5

    conf = 0.25
    if conf < conf_threshold:
        return "unknown", conf
    return "unknown", conf
