"""Turn RapidOCR tokens from a Madden / CFB scorebug into situation fields.

The scorebug ampersand is a stylized glyph. RapidOCR reads ``1st & 10`` as
``1ST810`` (the '&' becomes '8'). That correction is applied only when a
digit follows the 8, so a real ``1st & 8`` (``1ST8``) is left alone.
"""

from __future__ import annotations

import re

from vod_pilot.digits import normalize_digits, safe_int

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
    return re.sub(r"[^A-Z0-9&]", "", normalize_digits(token).upper())


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
    elif rest.startswith("8") and len(rest) > 1:
        dist = safe_int(rest[1:])
        if dist is not None and 1 <= dist <= 40:
            return down, dist
        return None
    if rest in {"G", "GOAL"}:
        return down, 1
    dist = safe_int(rest)
    if dist is not None and 1 <= dist <= 40:
        return down, dist
    return None


_DD_PREFIXES: list[tuple[str, int]] = [
    ("1ST", 1), ("IST", 1), ("LST", 1), ("15T", 1), ("7ST", 1), ("75T", 1),
    ("3RD", 3), ("3R0", 3), ("3RO", 3), ("SRD", 3), ("BRD", 3), ("3RI", 3), ("3R1", 3), ("8RD", 3),
    ("2ND", 2), ("2NO", 2), ("2N0", 2), ("ZND", 2), ("2NB", 2),
    ("4TH", 4), ("4TN", 4),
    ("1T", 1), ("ST", 1), ("5T", 1), ("4M", 4),
    ("ND", 2), ("RD", 3), ("TH", 4),
    ("20", 2), ("30", 3),
    ("S", 1), ("T", 1),
]
_DD_PREFIXES.sort(key=lambda item: len(item[0]), reverse=True)
_NTH = {1: "1ST", 2: "2ND", 3: "3RD", 4: "4TH"}
_GOAL_REST = {"GOAL", "G", "SGOAL", "&GOAL", "6GOAL", "8GOAL"}


def norm_dd(text: str, bare: bool = False) -> str | None:
    """Normalize one down-and-distance token to ``NTH&D``, ``NTH&GOAL``, or ``NTH&INCHES``.

    ``bare`` is only for a token sitting in the down-and-distance slot. It is
    what turns ``32`` into 3rd & 2. Clocks (a colon) are never a down.
    """
    if ":" in text:
        return None
    compact = re.sub(r"[^A-Z0-9&]", "", normalize_digits(text).upper())
    if not compact:
        return None
    ordinal = _ordinal_dd(compact)
    loose = _bare_dd(compact) if bare else None
    if ordinal and loose and ordinal != loose:
        return None
    return ordinal or loose


def _ordinal_dd(compact: str) -> str | None:
    hits = [item for item in _DD_PREFIXES if compact.startswith(item[0])]
    if not hits:
        return None
    longest = max(len(item[0]) for item in hits)
    top = [item for item in hits if len(item[0]) == longest]
    if len({item[1] for item in top}) != 1:
        return None
    rest = _dd_rest(compact[longest:])
    if rest is None:
        return None
    return f"{_NTH[top[0][1]]}&{rest}"


def _dd_rest(rest: str) -> str | None:
    if not rest:
        return None
    if rest in _GOAL_REST or (rest.endswith("GOAL") and rest.replace("&", "") in _GOAL_REST | {"GOAL"}):
        return "GOAL"
    if rest == "INCHES":
        return "INCHES"
    if rest[0] in "&86E" and 1 <= len(rest[1:]) <= 2:
        dist = safe_int(rest[1:])
        if dist is not None and 1 <= dist <= 40:
            return str(dist)
    value = safe_int(rest)
    if value is not None and 1 <= value <= 40 and rest.isdigit():
        return str(value)
    if rest[:1] in "68":
        stripped = safe_int(rest[1:])
        if stripped is not None and value is not None and value > 40 and 1 <= stripped <= 40:
            return str(stripped)
    return None


def _bare_dd(compact: str) -> str | None:
    # '1st & 10' with the ampersand dropped is '110', which is also a spot.
    if compact == "110" or not re.fullmatch(r"[1-4][0-9&]{1,4}", compact):
        return None
    down = int(compact[0])
    rest = compact[1:]
    splits = [rest]
    if rest.startswith("0"):
        splits.append(rest[1:])
    options: list[tuple[int, int]] = []
    for mid in splits:
        if not mid:
            continue
        if 1 <= len(mid) <= 2:
            dist = safe_int(mid)
            if dist is not None and mid.isdigit() and 1 <= dist <= 40:
                options.append((down, dist))
        if len(mid) > 1 and mid[0] in "68&" and 1 <= len(mid[1:]) <= 2:
            dist = safe_int(mid[1:])
            if dist is not None and 1 <= dist <= 40:
                options.append((down, dist))
    unique = list(dict.fromkeys(options))
    if len(unique) != 1:
        return None
    return f"{_NTH[unique[0][0]]}&{unique[0][1]}"


def parse_dd_normed(tokens: list, layout: str = "madden") -> tuple[int, int] | None:
    """One agreed down and distance. Disagreeing tokens are not a read."""
    from vod_pilot.tokens import as_tok

    readings: list[tuple[int, int]] = []
    for item in tokens:
        tok = as_tok(item)
        if ":" in tok.text:
            continue
        bare = _in_dd_slot(tok.x, tok.y, layout) and layout != "cfb"
        got = norm_dd(tok.text, bare=bare)
        parsed = _norm_pair(got)
        if parsed is not None:
            readings.append(parsed)
    unique = list(dict.fromkeys(readings))
    if len(unique) != 1:
        return None
    return unique[0]


def _in_dd_slot(x: float | None, y: float | None, layout: str) -> bool:
    if x is None or y is None:
        return False
    if layout == "cfb":
        return 0.45 <= x <= 0.55 and y >= 0.88
    return 0.83 <= x <= 0.91 and y >= 0.90


def _norm_pair(got: str | None) -> tuple[int, int] | None:
    if not got or "&" not in got:
        return None
    down_s, rest = got.split("&", 1)
    down = safe_int(down_s[:1])
    if down is None:
        return None
    if rest == "GOAL" or rest == "INCHES":
        return down, 1
    dist = safe_int(rest)
    if dist is None:
        return None
    return down, dist


def goal_down(token: str) -> int | None:
    """Down when the token says goal-to-go. Does not turn that into distance 1."""
    if ":" in token:
        return None
    compact = re.sub(r"[^A-Z0-9&]", "", token.upper())
    for name, down in (("1ST", 1), ("2ND", 2), ("3RD", 3), ("4TH", 4), ("IST", 1)):
        if not compact.startswith(name):
            continue
        rest = compact[len(name) :]
        if rest in {"GOAL", "G", "SGOAL", "&GOAL", "6GOAL", "8GOAL"} or rest.endswith("GOAL"):
            return down
    match = re.search(r"\b([1-4])(?:ST|ND|RD|TH)\s*(?:&|AND)\s*GOAL\b", normalize_digits(token), re.I)
    if match:
        return safe_int(match.group(1))
    return None


def parse_down_distance(texts: list[str]) -> tuple[int, int] | None:
    for token in texts:
        got = parse_down_distance_token(token)
        if got is not None:
            return got
    blob = " ".join(normalize_digits(token) for token in texts)
    match = _EXPLICIT_DD.search(blob)
    if not match:
        return None
    down = safe_int(match.group(1))
    raw = match.group(2).upper()
    if down is None:
        return None
    if raw in {"G", "GOAL"}:
        return down, 1
    dist = safe_int(raw)
    if dist is not None and 1 <= dist <= 40:
        return down, dist
    return None


def parse_clock(texts: list[str]) -> tuple[int, int] | None:
    """Return (quarter, seconds remaining) from a ``1st 3:56`` style token."""
    for token in texts:
        match = _CLOCK.search(normalize_digits(token))
        if not match:
            continue
        quarter = safe_int(match.group(1))
        minutes = safe_int(match.group(2))
        seconds = safe_int(match.group(3))
        if quarter is None or minutes is None or seconds is None:
            continue
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
    texts = [normalize_digits(token) for token in texts]
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
        yl = safe_int(own.group(1))
        if yl is not None and 1 <= yl <= 50:
            return yl, f"own {yl}"
    opp = _YARD_OPP.search(blob)
    if opp and not re.search(r"ONOPP\d", mashed):
        yl = safe_int(opp.group(1))
        if yl is not None and 1 <= yl <= 50:
            return max(1, min(99, 100 - yl)), f"opp {yl}"
    ball = _YARD_BALL.search(blob)
    if ball:
        yl = safe_int(ball.group(1))
        if yl is not None and 1 <= yl <= 50:
            return None, f"ball on {yl}"
    return None, ""


def _yard_near_down(texts: list[str], dd: tuple[int, int] | None, side: str) -> int | None:
    """Read ``1st&15onown22`` style tokens. Ignore a second spot tied to another down."""
    found: list[int] = []
    for token in texts:
        compact = re.sub(r"[^A-Z0-9&]", "", normalize_digits(token).upper())
        match = re.search(rf"ON{side}(\d{{1,2}})", compact)
        if not match:
            continue
        yl = safe_int(match.group(1))
        if yl is None or not 1 <= yl <= 50:
            continue
        lead = re.search(r"(1ST|2ND|3RD|4TH)&(\d{1,2})", compact)
        if dd and lead:
            lead_dist = safe_int(lead.group(2))
            if lead_dist is None:
                continue
            lead_dd = (_ORDINAL[lead.group(1)], lead_dist)
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
