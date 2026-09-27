"""Field-zone bucketing shared by the situation parser, retrain, and play caller.

Yardline convention (confirmed in ``situation.parse_situation`` and the live
HTML pad): ``yardline`` runs 0-100 from OUR goal line toward the opponent's.
"my 21" -> 21, "opp 14" -> 86, so ``100 - yardline`` = yards to the goal.

Zones:
  * ``gl``   goal line / goal-to-go: inside the opponent's 5, OR goal-to-go
             (distance >= yards to goal) inside the opponent's 10.
             "1st & goal at the 6" -> gl.
  * ``rz``   red zone: inside the opponent's 20 (yardline >= 80), not gl.
             "3rd & 2 at the 12" -> rz.
  * ``open`` everything else (including backed up in our own territory).
"""

from __future__ import annotations

OPEN = "open"
RED_ZONE = "rz"
GOAL_LINE = "gl"
ZONES = (OPEN, RED_ZONE, GOAL_LINE)
ZONE_LABELS = {OPEN: "open field", RED_ZONE: "red zone", GOAL_LINE: "goal line / goal-to-go"}

RED_ZONE_YARDS = 20  # inside the opponent's 20
GOAL_LINE_YARDS = 5  # inside the opponent's 5 is always goal line
GOAL_TO_GO_MAX_YARDS = 10  # goal-to-go counts as goal line when this close


def yards_to_goal(yardline: int | None) -> int | None:
    if yardline is None:
        return None
    try:
        yl = int(yardline)
    except (TypeError, ValueError):
        return None
    if yl < 0 or yl > 100:
        return None
    return 100 - yl


def is_goal_to_go(yardline: int | None, distance: int | None) -> bool:
    ytg = yards_to_goal(yardline)
    if ytg is None or distance is None:
        return False
    try:
        return int(distance) >= ytg
    except (TypeError, ValueError):
        return False


def field_zone(
    yardline: int | None,
    distance: int | None = None,
    *,
    red_zone_flag: bool = False,
    goal_line_flag: bool = False,
) -> str:
    """Bucket a snap into open / rz / gl.

    Explicit flags (typed "rz" / "gl" / "&goal") only matter when the yardline is
    unknown; a known yardline always wins so own-territory snaps stay ``open``.
    """
    ytg = yards_to_goal(yardline)
    if ytg is None:
        if goal_line_flag:
            return GOAL_LINE
        if red_zone_flag:
            return RED_ZONE
        return OPEN
    if ytg <= GOAL_LINE_YARDS:
        return GOAL_LINE
    if ytg <= GOAL_TO_GO_MAX_YARDS and is_goal_to_go(yardline, distance):
        return GOAL_LINE
    if ytg <= RED_ZONE_YARDS:
        return RED_ZONE
    return OPEN


def zone_of_situation(sit) -> str:
    return field_zone(
        getattr(sit, "yardline", None),
        getattr(sit, "distance", None),
        red_zone_flag=bool(getattr(sit, "red_zone", False)),
        goal_line_flag=bool(getattr(sit, "goal_line", False)),
    )


__all__ = [
    "GOAL_LINE",
    "OPEN",
    "RED_ZONE",
    "ZONES",
    "ZONE_LABELS",
    "field_zone",
    "is_goal_to_go",
    "yards_to_goal",
    "zone_of_situation",
]
