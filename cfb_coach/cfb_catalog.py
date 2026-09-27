"""CFB 27 formation -> play catalog (the playbook vocabulary).

Source: ``data/cfb27_formations.json`` built from CFB.FAN formation pages
(``scripts/build_cfb27_catalog.py``). Each formation lists the union of plays it
has across every CFB 27 playbook — the custom-playbook editor can pull a
formation from any source book, so this is what the autonomous book may choose
from. Also owns name canonicalisation so the book, the live caller and the
logged snaps always use one spelling.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA = Path(__file__).resolve().parent / "data" / "cfb27_formations.json"


def norm(s: str | None) -> str:
    """Case/punctuation-insensitive key: 'Z Spot GoalLine' == 'z spot goal-line'."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def pair_key(formation: str, play: str) -> str:
    return f"{formation}::{play}"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    try:
        return json.loads(_DATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"formations": {}}


def formations() -> dict[str, list[str]]:
    return {f: list(r.get("plays") or []) for f, r in (load_catalog().get("formations") or {}).items()}


def formation_info(formation: str) -> dict[str, Any]:
    return dict((load_catalog().get("formations") or {}).get(canonical_formation(formation) or "", {}) or {})


@lru_cache(maxsize=1)
def _form_index() -> dict[str, str]:
    return {norm(f): f for f in formations()}


@lru_cache(maxsize=1)
def _play_index() -> dict[str, dict[str, str]]:
    return {f: {norm(p): p for p in plays} for f, plays in formations().items()}


def canonical_formation(name: str | None) -> str | None:
    return _form_index().get(norm(name))


def canonical_play(formation: str | None, play: str | None) -> str | None:
    """Catalog spelling of ``play`` inside ``formation`` (None if not in that formation)."""
    f = canonical_formation(formation)
    if not f:
        return None
    return _play_index().get(f, {}).get(norm(play))


def canonical_pair(formation: str, play: str) -> tuple[str, str, bool]:
    """(formation, play, verified). Unverified names are returned as logged."""
    f = canonical_formation(formation)
    if f:
        p = canonical_play(f, play)
        if p:
            return f, p, True
        return f, play, False
    return formation, play, False


def formations_with_play(play: str) -> list[str]:
    k = norm(play)
    return [f for f, idx in _play_index().items() if k in idx]


# --- Zone fit (what the live caller may use in a field zone) -------------------
_DEEP = re.compile(r"vert|all go|flood|dagger|sluggo|shot|deep|bomb|hail|double post|dbl post|corner post|clear out|seam", re.I)
_GL_ONLY = re.compile(r"goal\s*line|goalline", re.I)
_RUN = re.compile(r"zone|dive|duo|power|counter|trap|draw|base|stretch|toss|sweep|wham|slam|iso|blast|sneak|option|lead", re.I)


def is_run(play: str) -> bool:
    return bool(_RUN.search(play or "")) and not re.search(r"\bpa\b|rpo|alert", play or "", re.I)


def zone_fit(play: str, zone: str) -> bool:
    """Coarse legality of a book play in a field zone (ranking still decides)."""
    p = play or ""
    if zone == "open":
        return not _GL_ONLY.search(p)
    # red zone / goal line: no deep shots with no field to work
    if _DEEP.search(p) and not re.search(r"rz|red zone|goal", p, re.I):
        return False
    return True


__all__ = [
    "canonical_formation",
    "canonical_pair",
    "canonical_play",
    "formation_info",
    "formations",
    "formations_with_play",
    "is_run",
    "load_catalog",
    "norm",
    "pair_key",
    "zone_fit",
]
