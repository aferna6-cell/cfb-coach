"""Result-quality pass. A blank distance is missing, not a string to subtract.

Goal-to-go snaps keep ``goal_to_go`` and carry ``distance is None`` when the
yards-to-goal were not read. Yards still resolve from that yard line, or from
the distance the HUD painted before the clear. ``\"\"`` is never subtracted.
A TD or turnover already on the row is left in place.
"""

from __future__ import annotations

try:
    from vodx_pr88 import as_int, fill_result_yards
except ImportError:  # box layout: this file sits next to vodx.py, pr88 is on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vodx_pr88 import as_int, fill_result_yards


def apply_quality(rows: list[dict], frames=None, game: str | None = None) -> dict:
    """Fill yards from the next snap.

    ``frames`` and ``game`` are accepted so vodx_refine can pass them through.
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
        if "baseline_distance" in row:
            row["baseline_distance"] = as_int(row.get("baseline_distance"))
    fill_result_yards(rows)
    return {"rows": len(rows)}


apply_quality._blank_safe = True
