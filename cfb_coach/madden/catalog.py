"""Madden 27 stock-book catalog: book -> formation -> plays (per book, not a union).

Source: ``data/madden27/playbooks.json`` built from the Huddle.gg Madden 27 playbook
database by ``scripts/build_madden27_playbooks.py``. Unlike CFB 27 (where the custom
editor pulls a formation with its union of plays), a Madden formation's play list
depends on the BOOK: e.g. Gun Doubles Clamp Stack has Texas Y-Stutter Wheel and Same
Side Zone in the Buccaneers book but not in Shotgun Classic or the Lions book.
Huddle groups Gun formations under "Gun"; in game that family is "Shotgun".
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from cfb_coach.madden.data import _data_path

SIDES = ("offense", "defense")


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@lru_cache(maxsize=1)
def load_playbooks() -> dict[str, Any]:
    try:
        return json.loads(_data_path("playbooks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"books": {"offense": {}, "defense": {}}}


def books(side: str) -> dict[str, dict[str, Any]]:
    return dict((load_playbooks().get("books") or {}).get(side) or {})


def book_names(side: str) -> list[str]:
    return list(books(side))


def book_team(side: str, book: str) -> str | None:
    return (books(side).get(book) or {}).get("team")


def book_formations(side: str, book: str) -> dict[str, list[str]]:
    """Every formation in ``book`` with its plays in that book (empty if not catalogued)."""
    forms = (books(side).get(book) or {}).get("formations") or {}
    return {f: list(r.get("plays") or []) for f, r in forms.items()}


def formation_family(side: str, book: str, formation: str) -> str:
    rec = ((books(side).get(book) or {}).get("formations") or {}).get(formation) or {}
    return str(rec.get("family") or "")


def canonical_play(side: str, book: str, formation: str, play: str) -> str | None:
    plays = book_formations(side, book).get(formation) or []
    k = norm(play)
    return next((p for p in plays if norm(p) == k), None)


def books_with_formation(side: str, formation: str) -> list[str]:
    return [b for b, rec in books(side).items() if formation in (rec.get("formations") or {})]


def all_formations(side: str) -> dict[str, list[str]]:
    """Union vocabulary (formation -> every play it has in any catalogued book): research only."""
    out: dict[str, list[str]] = {}
    for rec in books(side).values():
        for f, r in (rec.get("formations") or {}).items():
            cur = out.setdefault(f, [])
            for p in r.get("plays") or []:
                if p not in cur:
                    cur.append(p)
    return out


_RUN = re.compile(r"zone|dive|power|counter|trap|draw|stretch|toss|sweep|iso\b|blast|sneak|read option|"
                  r"off tackle|pin pull|slam|\blead\b|wham|duo|jet|qb keep|speed option|triple option|hb mid", re.I)
_NOT_RUN = re.compile(r"\bpa\b|rpo|alert|screen|option wheel|y option|choice|pass", re.I)
_GL_ONLY = re.compile(r"goal\s*line|goalline|qb sneak|\bsneak\b", re.I)
_DEEP = re.compile(r"vert|all go|flood|dagger|sluggo|shot|deep|bomb|hail|double post|dbl post|corner post|"
                   r"clear|seam|fades?\b|streak", re.I)


def is_run(play: str) -> bool:
    return bool(_RUN.search(play or "")) and not _NOT_RUN.search(play or "")


def is_deep(play: str) -> bool:
    return bool(_DEEP.search(play or ""))


def zone_fit(play: str, zone: str) -> bool:
    if zone == "open":
        return not _GL_ONLY.search(play or "") and "hail mary" not in (play or "").lower()
    if "hail mary" in (play or "").lower():
        return False
    return not (is_deep(play) and not re.search(r"rz|red zone|goal", play or "", re.I))


__all__ = [
    "SIDES", "all_formations", "book_formations", "book_names", "book_team", "books", "books_with_formation",
    "canonical_play", "formation_family", "is_deep", "is_run", "load_playbooks", "norm", "zone_fit",
]
