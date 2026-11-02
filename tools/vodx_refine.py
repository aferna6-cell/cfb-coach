"""Snap refine entry used by vodx.py extract_one.

The quality pass is the step that used to do ``yy = dist - ndist`` on a
goal-to-go row whose distance was the empty string.
"""

from __future__ import annotations

from vodx_quality import apply_quality


def refine_rows(rows: list[dict], frames, game: str) -> dict:
    return apply_quality(rows, frames, game)
