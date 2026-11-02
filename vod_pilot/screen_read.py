"""Read a formation name that is actually on the screen.

A formation tab lists every row. OCR cannot see the highlight, so several
different names is not a read. A long tooltip that names exactly one
formation is the selected row, same rule the pilot already uses for plays.
A custom (MUT) line is kept only when it is one front and not a prefix of
two packaged formations. Nothing here substitutes a different book's play.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_NON_ALNUM = re.compile(r"[^A-Z0-9]")
# Personnel or a package plus a set word. Bare "NICKEL" or bare "3-4" is not
# enough: those strings are shared by several fronts.
_CUSTOM_FRONT = re.compile(
    r"^(?:GOAL LINE|BIG NICKEL|NICKEL|DIME|DOLLAR|QUARTER|46)"
    r"(?:\s+[A-Z0-9][A-Z0-9]{1,14}){1,4}$"
    r"|^(?:\d\s*-\s*\d(?:\s*-\s*\d)?)\s+[A-Z][A-Z0-9]{1,14}"
    r"(?:\s+[A-Z0-9][A-Z0-9]{0,14}){0,3}$",
    re.I,
)
_CUSTOM_OFFENSE = re.compile(
    r"^(?:GUN|SHOTGUN|SINGLEBACK|I FORM|I-FORM|PISTOL|STRONG|WEAK|EMPTY|"
    r"WILDCAT|FULL HOUSE|HAIL MARY|GOAL LINE|NEAR|FAR|BUNCH)"
    r"(?:\s+[A-Z0-9][A-Z0-9]{1,16}){1,5}$",
    re.I,
)


def compact(text: str) -> str:
    return _NON_ALNUM.sub("", text.upper())


def clean_ocr_line(text: str) -> str:
    """Fix 0/1 swapped for O/I inside a word, including a leading swap (``0VER``)."""
    raw = text.strip().replace("—", "-").replace("–", "-").replace("‐", "-").upper()
    words = []
    for word in raw.split():
        chars: list[str] = []
        for i, ch in enumerate(word):
            prev_alpha = i > 0 and word[i - 1].isalpha()
            next_alpha = i + 1 < len(word) and word[i + 1].isalpha()
            if ch == "0" and (prev_alpha or next_alpha):
                chars.append("O")
            elif ch == "1" and (prev_alpha or next_alpha):
                chars.append("I")
            else:
                chars.append(ch)
        words.append("".join(chars))
    return " ".join(words)


def _repo_playbook() -> Path:
    return Path(__file__).resolve().parents[1] / "cfb_coach" / "data" / "madden27" / "playbooks.json"


@lru_cache(maxsize=1)
def _books() -> dict:
    path = _repo_playbook()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("books") or {}


def formation_names(side: str) -> list[str]:
    """Every packaged formation on this side, from every book."""
    found: list[str] = []
    seen: set[str] = set()
    for body in (_books().get(side) or {}).values():
        if not isinstance(body, dict):
            continue
        for name in (body.get("formations") or {}):
            if isinstance(name, str) and name not in seen:
                seen.add(name)
                found.append(name)
    found.sort(key=lambda name: len(compact(name)), reverse=True)
    return found


def _line_hits(line: str, names: list[str], *, long: bool) -> list[str]:
    cleaned = clean_ocr_line(line)
    key = compact(cleaned)
    if len(key) < 4:
        return []
    hits: list[str] = []
    for name in names:
        nkey = compact(name)
        if len(nkey) < 4:
            continue
        if key == nkey or (long and nkey in key):
            hits.append(name)
    # A shorter name that is only a piece of a longer name also on this line
    # is not a second formation.
    kept: list[str] = []
    for name in hits:
        nkey = compact(name)
        if any(name != other and nkey in compact(other) and compact(other) in key for other in hits):
            continue
        kept.append(name)
    return kept


def _custom_line(line: str, names: list[str], pattern: re.Pattern[str]) -> str:
    cleaned = clean_ocr_line(line)
    if len(cleaned) < 5 or len(cleaned) >= 40:
        return ""
    if not pattern.match(cleaned):
        return ""
    key = compact(cleaned)
    if any(compact(name) == key for name in names):
        return ""
    # "NICKEL" / "DIME 3-2" style prefixes belong to several real fronts.
    extended = [name for name in names if compact(name).startswith(key) and compact(name) != key]
    if len(extended) >= 2:
        return ""
    return re.sub(r"\s+", " ", cleaned).strip()


def unique_screen_name(
    texts: list[str],
    names: list[str],
    *,
    custom: re.Pattern[str] | None = None,
) -> tuple[str, float]:
    """Return ``(name, confidence)``. Empty name means the screen did not say.

    Confidence is 0 when the name is empty. A value above 0 is the evidence
    for that one string, not a guess among several.
    """
    short: list[str] = []
    long: list[str] = []
    for line in texts:
        text = str(line).strip()
        if not text:
            continue
        hits = _line_hits(text, names, long=len(text) >= 40)
        if len(hits) != 1:
            continue
        if len(text) >= 40:
            long.append(hits[0])
        else:
            short.append(hits[0])
    short_u = list(dict.fromkeys(short))
    long_u = list(dict.fromkeys(long))
    if len(long_u) == 1:
        # The description pane names the highlighted row.
        conf = 0.8 if len(short_u) > 1 else 0.86
        return long_u[0], conf
    if len(long_u) > 1 or len(short_u) > 1:
        return "", 0.0
    if len(short_u) == 1:
        return short_u[0], 0.9
    if custom is not None:
        customs = []
        for line in texts:
            if len(str(line)) >= 40:
                continue
            got = _custom_line(str(line), names, custom)
            if got:
                customs.append(got)
        customs_u = list(dict.fromkeys(customs))
        if len(customs_u) == 1:
            return customs_u[0], 0.64
    return "", 0.0


def read_defensive_formation(texts: list[str], names: list[str] | None = None) -> tuple[str, float]:
    return unique_screen_name(texts, names if names is not None else formation_names("defense"), custom=_CUSTOM_FRONT)


def read_offensive_formation(texts: list[str], names: list[str] | None = None) -> tuple[str, float]:
    return unique_screen_name(texts, names if names is not None else formation_names("offense"), custom=_CUSTOM_OFFENSE)
