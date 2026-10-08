"""Turn RapidOCR tokens from a Madden / CFB scorebug into situation fields.

The scorebug ampersand is a stylized glyph. RapidOCR reads ``1st & 10`` as
``1ST810`` (the '&' becomes '8'). That correction is applied only when a
digit follows the 8, so a real ``1st & 8`` (``1ST8``) is left alone.
"""

from __future__ import annotations

import re

_ORDINAL = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}
_CLOCK = re.compile(
    r"([1-4])(?:ST|ND|RD|TH)\s*(\d{1,2}):(\d{2})",
    re.I,
)
_EXPLICIT_DD = re.compile(
    r"\b([1-4])(?:ST|ND|RD|TH)\s*(?:&|AND)\s*(\d{1,2}|G|GOAL)\b",
    re.I,
)
_YARD_OWN = re.compile(r"\b(?:OWN|MY|OUR)\s*(\d{1,2})\b", re.I)
_YARD_OPP = re.compile(r"\b(?:OPP|THEIR)\s*(\d{1,2})\b", re.I)
_YARD_BALL = re.compile(r"\bBALL\s*ON\s*(\d{1,2})\b", re.I)

_FLAG_WORDS = (
    ("int", re.compile(r"\bINTERCEPT", re.I)),
    ("fumble", re.compile(r"\bFUMBLE", re.I)),
    ("td", re.compile(r"\bTOUCHDOWN\b|\bTD\b", re.I)),
    ("incomplete", re.compile(r"\bINCOMPLETE\b", re.I)),
    # "Sack the QB 0/3" is a challenge tracker, not a result bug.
    ("sack", re.compile(r"SACK(?!\s*THE\s*QB)", re.I)),
)


def _clean_token(token: str) -> str:
    return re.sub(r"[^A-Z0-9&]", "", token.upper())


def parse_down_distance_token(token: str) -> tuple[int, int] | None:
    """Parse one OCR token. Clock tokens (they contain ':') are not downs."""
    if ":" in token:
        return None
    t = _clean_token(token)
    if not t:
        return None
    for name, down in _ORDINAL.items():
        idx = t.find(name)
        if idx < 0:
            continue
        rest = t[idx + len(name) :]
        parsed = _parse_rest(down, rest)
        if parsed is not None:
            return parsed
    return None


def _parse_rest(down: int, rest: str) -> tuple[int, int] | None:
    if not rest:
        return None
    if rest.startswith("&"):
        rest = rest[1:]
    elif rest.startswith("8") and len(rest) > 1 and rest[1:].isdigit():
        dist = int(rest[1:])
        if 1 <= dist <= 40:
            return down, dist
        return None
    if rest in {"G", "GOAL"}:
        return down, 1
    if rest.isdigit():
        dist = int(rest)
        if 1 <= dist <= 40:
            return down, dist
    return None


def parse_down_distance(texts: list[str]) -> tuple[int, int] | None:
    for token in texts:
        got = parse_down_distance_token(token)
        if got is not None:
            return got
    blob = " ".join(texts)
    match = _EXPLICIT_DD.search(blob)
    if not match:
        return None
    down = int(match.group(1))
    raw = match.group(2).upper()
    if raw in {"G", "GOAL"}:
        return down, 1
    dist = int(raw)
    if 1 <= dist <= 40:
        return down, dist
    return None


def parse_clock(texts: list[str]) -> tuple[int, int] | None:
    """Return (quarter, seconds remaining) from a ``1st 3:56`` style token."""
    for token in texts:
        match = _CLOCK.search(token)
        if not match:
            continue
        quarter = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        if minutes > 15 or seconds > 59:
            continue
        return quarter, minutes * 60 + seconds
    return None


def parse_yardline_phrase(texts: list[str], dd: tuple[int, int] | None = None) -> tuple[int | None, str]:
    """Return (coach yardline 0-100, phrase) only for explicit own/opp/ball-on.

    A lone ``35`` painted on the field is not converted. Own versus opponent
    is not knowable from that digit. ``own N`` is stored as N, which matches
    the coach's 0-at-our-goal scale when the row's side is the offense on screen.
    """
    blob = " ".join(texts)
    mashed = re.sub(r"[^A-Z0-9]", "", blob.upper())
    own_mashed = _yard_near_down(texts, dd, "OWN")
    if own_mashed is not None:
        return own_mashed, f"own {own_mashed}"
    opp_mashed = _yard_near_down(texts, dd, "OPP")
    if opp_mashed is not None:
        return max(1, min(99, 100 - opp_mashed)), f"opp {opp_mashed}"
    # Spaced forms: "OWN 22", "OPP 40".
    own = _YARD_OWN.search(blob)
    if own and not re.search(r"ONOWN\d", mashed):
        yl = int(own.group(1))
        if 1 <= yl <= 50:
            return yl, f"own {yl}"
    opp = _YARD_OPP.search(blob)
    if opp and not re.search(r"ONOPP\d", mashed):
        yl = int(opp.group(1))
        if 1 <= yl <= 50:
            return max(1, min(99, 100 - yl)), f"opp {yl}"
    ball = _YARD_BALL.search(blob)
    if ball:
        yl = int(ball.group(1))
        if 1 <= yl <= 50:
            return None, f"ball on {yl}"
    return None, ""


def _yard_near_down(texts: list[str], dd: tuple[int, int] | None, side: str) -> int | None:
    """Read ``1st&15onown22`` style tokens. Ignore a second spot tied to another down."""
    found: list[int] = []
    for token in texts:
        compact = re.sub(r"[^A-Z0-9&]", "", token.upper())
        match = re.search(rf"ON{side}(\d{{1,2}})", compact)
        if not match:
            continue
        yl = int(match.group(1))
        if not 1 <= yl <= 50:
            continue
        lead = re.search(r"(1ST|2ND|3RD|4TH)&(\d{1,2})", compact)
        if dd and lead:
            lead_dd = (_ORDINAL[lead.group(1)], int(lead.group(2)))
            if lead_dd != dd:
                continue
        found.append(yl)
    unique = list(dict.fromkeys(found))
    if len(unique) == 1:
        return unique[0]
    return None


def parse_flags(texts: list[str]) -> set[str]:
    blob = " ".join(texts)
    found: set[str] = set()
    for name, rx in _FLAG_WORDS:
        if rx.search(blob):
            found.add(name)
    return found


def clocks_compatible(
    prev_q: int | None,
    prev_clock: int | None,
    next_q: int | None,
    next_clock: int | None,
) -> bool:
    """True when two HUD reads can be the same drive, one play apart.

    Missing clocks are treated as compatible so a down change can still be
    scored, but the caller should lower confidence in that case.
    """
    if prev_q and next_q and prev_q != next_q:
        return False
    if prev_clock is None or next_clock is None:
        return True
    delta = prev_clock - next_clock  # time elapsed
    return -2 <= delta <= 50
