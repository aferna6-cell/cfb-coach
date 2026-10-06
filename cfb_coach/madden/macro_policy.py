"""When a live Madden call may show a Custom Adjustment.

The play is chosen first. This only decides whether that call also shows a macro.
Silence is the default. A macro is shown only when the trigger is unambiguous,
and the same one never comes back on the next snap. Zero in a game is a normal
result. If the look is mixed or only a previous snap, do not suggest.

A coverage or concept macro needs that look on the field right now (``showing`` /
``live`` / ``pre-snap``) and already confirmed in this game's log. Red zone,
goal line, two-minute, and protecting a lead skip the look count. Both still
wait out a cooldown, once per drive, and only a couple of times per side.
"""

from __future__ import annotations

import re
from typing import Any

# Prior logged tells of this coverage class or concept family, not counting
# the snap being called. One typed name, even "showing", is not enough.
REPEAT_BEFORE_FIRE = 3
# Those tells must be a clear majority of the looks logged on that side.
# A mixed tendency is not unambiguous, so it stays quiet.
CONFIDENCE = 0.60
# The same macro stays off for this many later snaps on that side.
# 6 is longer than one snap, so it is never suggested back to back.
COOLDOWN_SNAPS = 6
# One shown macro per possession. A side change starts the next one.
PER_DRIVE = 1
# Zero to a few per side for the whole session. Zero is a normal game.
PER_GAME = 2

_BLANK = frozenset({"", "none", "null"})
_SUGGEST_SPLIT = re.compile(r"\s+[—–]\s+|\s+-\s+")


def macro_id_from_suggest(text: str | None) -> str | None:
    """Macro id at the front of a SUGGEST line, or None when it is not a macro."""
    if not text:
        return None
    head = _SUGGEST_SPLIT.split(str(text), maxsplit=1)[0]
    head = head.split("[", 1)[0].strip()
    if not head or head.lower() in _BLANK:
        return None
    if _known(head):
        return head
    return None


def resolve_live_macros(explicit: bool | None = None) -> bool:
    """Whether live calls may show macros.

    ``False`` (the ``--no-macros`` flag) forces them off. ``True`` forces them on.
    ``None`` reads ``live_macros`` from the Madden config, which defaults to on.
    The rarity rules still apply when this is on.
    """
    if explicit is not None:
        return bool(explicit)
    try:
        from cfb_coach.madden.franchise import load_config

        return bool(load_config().get("live_macros", True))
    except Exception:  # noqa: BLE001 — a missing config stays on; the rarity rules still apply
        return True


def allow_macro(
    db: Any,
    *,
    opponent_id: str,
    side: str,
    macro_id: str,
    kind: str,
    session_id: str | None = None,
    family: str | None = None,
    live: bool = False,
    enabled: bool = True,
) -> bool:
    """True when this snap may show ``macro_id``.

    ``kind`` is ``look`` (coverage / concept) or ``situation`` (field / clock).
    A look also needs ``live`` (the look is on the field this snap). A previous
    snap is not enough. No database: a look stays quiet; a situation trigger
    may show once. ``enabled`` False is the off switch: nothing is shown.
    """
    if not enabled:
        return False
    if not macro_id or str(macro_id).strip().lower() in _BLANK:
        return False
    if kind != "situation" and not live:
        return False
    if db is None:
        return kind == "situation"
    try:
        snaps = _game_snaps(db, opponent_id, session_id)
    except Exception:  # noqa: BLE001 — a bad log must not invent a macro
        return False
    if not _within_caps(snaps, side, macro_id):
        return False
    if kind == "situation":
        return True
    matches, looks = _look_stats(snaps, side, macro_id, family)
    if matches < REPEAT_BEFORE_FIRE or looks <= 0:
        return False
    return (matches / looks) >= CONFIDENCE


def _known(name: str) -> bool:
    from cfb_coach.madden.macro_pool import pool_macro
    from cfb_coach.madden.offense_macros import known

    return bool(pool_macro(name)) or known(name)


def _shown_id(raw: Any) -> str | None:
    text = str(raw or "").split("[", 1)[0].strip()
    if text.lower() in _BLANK:
        return None
    return text


def _row_session(row: Any) -> str | None:
    try:
        return row["session_id"]
    except (KeyError, IndexError, TypeError):
        return None


def _game_snaps(db: Any, opponent_id: str, session_id: str | None) -> list[Any]:
    """This session, oldest first. No session id: the unscoped terminal log."""
    if session_id:
        rows = db.get_session_snaps(session_id)
        return [r for r in rows if r["opponent_id"] == opponent_id]
    rows = db.get_recent_snaps(opponent_id, limit=500)
    bare = [r for r in rows if not _row_session(r)]
    return list(reversed(bare))


def _within_caps(snaps: list[Any], side: str, macro_id: str) -> bool:
    shown = 0
    for row in snaps:
        if row["side"] == side and _shown_id(row["macro"]):
            shown += 1
    if shown >= PER_GAME:
        return False
    drive: list[Any] = []
    for row in reversed(snaps):
        if row["side"] != side:
            break
        drive.append(row)
    if sum(1 for row in drive if _shown_id(row["macro"])) >= PER_DRIVE:
        return False
    side_rows = [row for row in snaps if row["side"] == side]
    recent = side_rows[-COOLDOWN_SNAPS:]
    return not any(
        _shown_id(row["macro"]) and _shown_id(row["macro"]).upper() == macro_id.upper()
        for row in recent
    )


def _look_stats(
    snaps: list[Any], side: str, macro_id: str, family: str | None,
) -> tuple[int, int]:
    side_rows = [row for row in snaps if row["side"] == side]
    if side == "defense":
        from cfb_coach.madden.situation import concept_family

        looks = [row for row in side_rows if (row["concept_seen"] or "").strip()]
        matches = [row for row in looks if family and concept_family(row["concept_seen"]) == family]
        return len(matches), len(looks)
    from cfb_coach.macros import classify_coverage
    from cfb_coach.madden.offense_macros import load_offense_settings

    fire = ((load_offense_settings().get("macros") or {}).get(macro_id) or {}).get("fire") or {}
    want = set(fire.get("coverages") or [])
    if fire.get("fallback_pressure"):
        want.add("pressure")
    looks = [row for row in side_rows if (row["coverage_seen"] or "").strip()]
    if not want:
        return 0, len(looks)
    matches = [row for row in looks if classify_coverage(row["coverage_seen"]) & want]
    return len(matches), len(looks)


__all__ = [
    "CONFIDENCE", "COOLDOWN_SNAPS", "PER_DRIVE", "PER_GAME", "REPEAT_BEFORE_FIRE",
    "allow_macro", "macro_id_from_suggest", "resolve_live_macros",
]
