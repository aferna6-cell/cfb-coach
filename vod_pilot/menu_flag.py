"""Flag a frame that is a menu, lobby, splash, play-select, or replay.

The flag is only set when a marker is on the screen. No marker means the
flag stays off with confidence 0. That is not a claim that the frame is gameplay.
"""

from __future__ import annotations

import re

# Longer phrases first. A short REPLAY line counts; the word inside a sentence does not.
_MARKERS: tuple[tuple[str, float], ...] = (
    ("EA SPORTS", 0.95),
    ("ELECTRONIC ARTS", 0.95),
    ("INSTANT REPLAY", 0.93),
    ("REPLAY CHALLENGE", 0.9),
    ("PAUSE MENU", 0.95),
    ("PAUSED", 0.9),
    ("ULTIMATE TEAM", 0.92),
    ("MUT LOBBY", 0.92),
    ("LOBBY", 0.88),
    ("SELECT A PLAY", 0.8),
    ("SELECT PLAY", 0.8),
    ("PLAY CALL", 0.72),
    ("CHOOSE A FORMATION", 0.8),
)


def menu_flag(texts: list[str]) -> tuple[bool, float]:
    """Return ``(is_menu_or_lobby, confidence)``. Confidence is 0 when the flag is off."""
    best = 0.0
    for line in texts:
        text = " ".join(str(line).upper().split())
        if not text:
            continue
        compact = text.replace(" ", "")
        for marker, conf in _MARKERS:
            if marker in text or marker.replace(" ", "") in compact:
                best = max(best, conf)
        if text in {"REPLAY", "PAUSE"}:
            best = max(best, 0.9)
        elif re.fullmatch(r"REPLAY\s+\d{1,2}:\d{2}", text):
            best = max(best, 0.9)
    if best <= 0:
        return False, 0.0
    return True, best
