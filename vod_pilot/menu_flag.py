"""Flag a frame that is a menu, lobby, splash, play-select, or replay.

The flag is only set when a marker is on the screen. No marker means the
flag stays off with confidence 0. That is not a claim that the frame is gameplay.

A marker must be the whole phrase. ``SELECT PLAY`` does not match
``SELECT PLAYER``, and the CFB play-clock label is not a play-select screen.
At 360p ``PLAY CLOCK`` is often read as ``PLAY CALL`` next to the 1-40 clock
number. That pair, on a frame that already has a down, stays gameplay.
"""

from __future__ import annotations

import re

from vod_pilot.digits import normalize_digits, safe_int

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

_DOWN = re.compile(r"\b[1-4](?:ST|ND|RD|TH)\b", re.I)
_PLAY_CLOCK = re.compile(r"^PLAY\s*CL(?:OCK|K)$", re.I)
_PLAY_CALL = re.compile(r"^PLAY\s*CALL(?:\s*\d{1,2})?$", re.I)
_SMALL_NUMBER = re.compile(r"^\d{1,2}$")


def _phrase(text: str, phrase: str) -> bool:
    """Whole phrase only. ``SELECT PLAY`` is not a prefix of ``SELECT PLAYER``."""
    return re.search(rf"(?<![A-Z0-9]){re.escape(phrase)}(?![A-Z0-9])", text) is not None


def _has_down(texts: list[str]) -> bool:
    return any(_DOWN.search(text) for text in texts)


def _has_play_clock_number(texts: list[str]) -> bool:
    for text in texts:
        number = safe_int(normalize_digits(text))
        if number is not None and _SMALL_NUMBER.fullmatch(normalize_digits(text)) and 1 <= number <= 40:
            return True
    return False


def _is_play_clock_label(text: str, texts: list[str]) -> bool:
    """CFB paints PLAY CLOCK on the bug. 360p often reads it as PLAY CALL."""
    if _PLAY_CLOCK.fullmatch(text):
        return True
    if _PLAY_CALL.fullmatch(text) and _has_down(texts) and (
        bool(re.search(r"\d", text)) or _has_play_clock_number(texts)
    ):
        return True
    return False


def menu_flag(texts: list[str]) -> tuple[bool, float]:
    """Return ``(is_menu_or_lobby, confidence)``. Confidence is 0 when the flag is off."""
    lines = [" ".join(str(line).upper().split()) for line in texts]
    lines = [line for line in lines if line]
    best = 0.0
    for text in lines:
        if _is_play_clock_label(text, lines):
            continue
        for marker, conf in _MARKERS:
            if _phrase(text, marker):
                best = max(best, conf)
        if text in {"REPLAY", "PAUSE"}:
            best = max(best, 0.9)
        elif re.fullmatch(r"REPLAY\s+\d{1,2}:\d{2}", text):
            best = max(best, 0.9)
    if best <= 0:
        return False, 0.0
    return True, best
