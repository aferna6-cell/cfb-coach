"""Whether a live Madden call may show a Custom Adjustment, and which id was shown.

The play is chosen first. With live macros on (the default), a matching trigger
shows the macro the way it did before the rarity caps: one live look, a repeated
previous-snap coverage, a repeated concept, or a situation such as the red zone.
``--no-macros`` and ``config --no-macros`` turn the label off. The snap log stores
the id on screen, including a defense SUGGEST line, or ``none``.
"""

from __future__ import annotations

import re

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
    """
    if explicit is not None:
        return bool(explicit)
    try:
        from cfb_coach.madden.franchise import load_config

        return bool(load_config().get("live_macros", True))
    except Exception:  # noqa: BLE001 — a missing config stays on
        return True


def _known(name: str) -> bool:
    from cfb_coach.madden.macro_pool import pool_macro
    from cfb_coach.madden.offense_macros import known

    return bool(pool_macro(name)) or known(name)


__all__ = ["macro_id_from_suggest", "resolve_live_macros"]
