"""OCR digits that are not ASCII.

Circled and enclosed numbers (``③``, ``⑩``, ``⓵``, ``❶``, ``➀``) pass
``str.isdigit`` and ``\\d``, then ``int()`` raises ValueError. Normalize with
``unicodedata.digit`` and NFKC before any parse. A value that is still not a
number is missing, not an exception.
"""

from __future__ import annotations

import unicodedata

_ONES = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
}
_TEENS = {
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
}
_TENS = {
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}


def _from_name(ch: str) -> str | None:
    """Dingbat tens that neither digit() nor NFKC decompose (``❿``, ``➉``, ``➓``)."""
    name = unicodedata.name(ch, "")
    for marker in ("DIGIT ", "NUMBER "):
        if marker not in name:
            continue
        words = name.split(marker, 1)[1].split()
        if len(words) == 1:
            if words[0] in _ONES:
                return str(_ONES[words[0]])
            if words[0] in _TEENS:
                return str(_TEENS[words[0]])
        if len(words) == 2 and words[0] in _TENS and words[1] in _ONES:
            return str(_TENS[words[0]] + _ONES[words[1]])
    return None


def _one_char(ch: str) -> str:
    if "0" <= ch <= "9":
        return ch
    try:
        return str(unicodedata.digit(ch))
    except (TypeError, ValueError):
        pass
    folded = unicodedata.normalize("NFKC", ch)
    if folded != ch and folded.isascii() and folded.isdigit():
        return folded
    if len(folded) == 1 and folded != ch:
        try:
            return str(unicodedata.digit(folded))
        except (TypeError, ValueError):
            pass
    named = _from_name(ch)
    if named is not None:
        return named
    return ch


def normalize_digits(text: str) -> str:
    """Replace enclosed digits with ASCII. Other characters stay put."""
    if not text:
        return text
    return "".join(_one_char(ch) for ch in str(text))


def safe_int(value: object) -> int | None:
    """Parse a number. Circled digits count. Anything else is None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            return None
        return int(value)
    text = normalize_digits(str(value)).strip()
    if not text:
        return None
    body = text[1:] if text[0] == "-" else text
    if not body.isascii() or not body.isdigit():
        return None
    try:
        return int(text)
    except ValueError:
        return None
