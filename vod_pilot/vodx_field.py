"""Ball-spot token on the scorebug.

The arrow is a glyph, not part of the yard line. ``￥`` and ``y`` are own,
``a`` is opponent, and a leading ``1`` on a three-digit read is the arrow
the way a leading ``4`` already was. Side stays unknown when the prefix is
not one of those glyphs. Confidence below 0.5 is not a spot.
"""

from __future__ import annotations

from vod_pilot.digits import normalize_digits, safe_int
from vod_pilot.tokens import Token, as_tok

OWN_GLYPH = set("vV^¥￥yY▲")
OPP_GLYPH = set("aA▼")


def spot_token(text: str, conf: float | None = None) -> tuple[int, str | None] | None:
    """Return ``(yard, side)``. ``side`` is ``own``, ``opp``, or None.

    None for the whole token means it is not a spot. A yard with side None
    is a number whose arrow was not read.
    """
    if conf is not None and conf < 0.5:
        return None
    raw = normalize_digits(text).strip().replace(" ", "")
    if not raw:
        return None
    if raw[0].isdigit():
        if len(raw) == 3 and raw.isdigit() and raw[0] in "147":
            yard = _yard(raw[1:])
            if yard is None:
                return None
            if raw[0] == "7":
                return yard, "own"
            return yard, None
        return None
    number = raw[1:]
    if not number.isdigit() or not 1 <= len(number) <= 2:
        return None
    yard = _yard(number)
    if yard is None:
        return None
    glyph = raw[0]
    if glyph in OWN_GLYPH:
        return yard, "own"
    if glyph in OPP_GLYPH:
        return yard, "opp"
    return yard, None


def _yard(number: str) -> int | None:
    yard = safe_int(normalize_digits(number))
    if yard is None or not 1 <= yard <= 50:
        return None
    return yard


def sided_spot(tokens: list[Token]) -> tuple[int, str, float] | None:
    """One agreed own/opp spot. Two different spots is not a read."""
    found: list[tuple[int, str, float]] = []
    for item in tokens:
        tok = as_tok(item)
        got = spot_token(tok.text, tok.conf)
        if got is None or got[1] is None:
            continue
        found.append((got[0], got[1], tok.conf if tok.conf is not None else 0.65))
    unique = list(dict.fromkeys((yard, side) for yard, side, _ in found))
    if len(unique) != 1:
        return None
    yard, side = unique[0]
    conf = min(item_conf for item_yard, item_side, item_conf in found if (item_yard, item_side) == (yard, side))
    return yard, side, conf
