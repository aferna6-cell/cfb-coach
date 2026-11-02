"""Result-quality pass. A blank distance is missing, not a string to subtract.

Goal-to-go snaps keep ``goal_to_go`` and carry ``distance is None`` when the
yards-to-goal were not read. ``yy = dist - ndist`` only runs when both
distances are ints.
"""

from __future__ import annotations

try:
    from vodx_pr88 import as_int, yards_delta
except ImportError:  # box layout: this file sits next to vodx.py, pr88 is on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vodx_pr88 import as_int, yards_delta


def apply_quality(rows: list[dict], frames=None, game: str | None = None) -> dict:
    """Fill yards from the next snap when both distances are numbers.

    ``frames`` and ``game`` are accepted so vodx_refine can pass them through.
    A goal-to-go row with an empty distance is left alone.
    """
    del frames, game
    for row in rows:
        if "distance" in row:
            row["distance"] = as_int(row.get("distance"))
        if "down" in row:
            row["down"] = as_int(row.get("down"))
        if "yardline" in row:
            row["yardline"] = as_int(row.get("yardline"))
        if "yards" in row:
            row["yards"] = as_int(row.get("yards"))
    for row, nxt in zip(rows, rows[1:]):
        dist = as_int(row.get("distance"))
        ndist = as_int(nxt.get("distance"))
        down = as_int(row.get("down"))
        ndown = as_int(nxt.get("down"))
        if dist is None or ndist is None or down is None or ndown is None:
            continue
        if ndown == down + 1 and row.get("yards") is None:
            yy = yards_delta(dist, ndist)
            if yy is not None:
                row["yards"] = yy
    return {"rows": len(rows)}
