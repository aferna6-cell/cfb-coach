"""Test helper: lock a fixture's own offense book as the applied playbook.

Live offense calls pass ``validate_live_call``, which always checks
``playbook.load_books`` (meta ``active_playbook_json`` → applied). Fixtures that
hand an in-memory book to ``make_call`` must persist that same book as applied,
exactly like ``prep --mark-applied`` / ``apply_pending`` do. No plays are added.
"""
from __future__ import annotations

import copy
from typing import Any

from cfb_coach.madden import playbook


def apply_offense_book(db: Any, formations: dict[str, list[str]], *,
                       defense: dict[str, list[str]] | None = None) -> None:
    state = playbook._load_state(db)
    state["applied"]["offense"] = {
        "side": "offense", "mode": "custom", "name": "Test fixture offense",
        "rev": 1, "formations": copy.deepcopy(dict(formations)), "audibles": {},
        "formation_sources": {},
    }
    if defense is not None:
        state["applied"]["defense"] = {
            "side": "defense", "mode": "custom", "name": "Test fixture defense",
            "rev": 1, "formations": copy.deepcopy(dict(defense)), "audibles": {},
            "formation_sources": {},
        }
    playbook._save_state(db, state)
