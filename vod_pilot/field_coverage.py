"""Per-field fill rates for one extract.

This replaces a book-match rate. A play that is not in a packaged book still
counts as a play when the screen read it. ``book_match`` is optional and is
never copied over the read name.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

_COMPACT = re.compile(r"[^A-Z0-9]")
_DEF_PREFIXES = (
    "3-3",
    "3-4",
    "4-2",
    "4-3",
    "4-4",
    "5-2",
    "46",
    "DIME",
    "DOLLAR",
    "NICKEL",
    "GOAL LINE",
    "BIG NICKEL",
    "QUARTER",
)


def _compact(text: str) -> str:
    return _COMPACT.sub("", text.upper())


def _is_defensive_front(name: str) -> bool:
    upper = name.upper()
    return any(upper.startswith(prefix) for prefix in _DEF_PREFIXES)


def _offensive_formation(snap: Any) -> bool:
    name = str(_value(snap, "formation") or "")
    return bool(name) and not _is_defensive_front(name)


def _value(snap: Any, attr: str) -> Any:
    if isinstance(snap, dict):
        return snap.get(attr)
    return getattr(snap, attr, None)


def _defensive_formation(snap: Any) -> bool:
    front = str(_value(snap, "def_formation") or "")
    formation = str(_value(snap, "formation") or "")
    if front:
        return True
    return bool(formation) and _is_defensive_front(formation)


def _filled_result(snap: Any) -> bool:
    kind = str(_value(snap, "result_kind") or "")
    text = str(_value(snap, "result") or "")
    if kind in {"", "unknown"}:
        return False
    return bool(text)


_CHECKS: tuple[tuple[str, Callable[[Any], bool]], ...] = (
    ("offensive_formation", _offensive_formation),
    ("play", lambda snap: bool(_value(snap, "play")) and _value(snap, "play") != "Kickoff"),
    ("defensive_formation", _defensive_formation),
    ("defensive_call", lambda snap: bool(_value(snap, "coverage_seen"))),
    ("adjustments", lambda snap: bool(_value(snap, "adjustments"))),
    ("score", lambda snap: _value(snap, "score_left") is not None and _value(snap, "score_right") is not None),
    ("yardline", lambda snap: _value(snap, "yardline") not in (None, "")),
    ("result", _filled_result),
)


def _rates(snaps: list[Any]) -> dict[str, Any]:
    count = len(snaps)
    out: dict[str, Any] = {"rows": count}
    for name, check in _CHECKS:
        if not count:
            out[name] = None
        else:
            out[name] = round(sum(1 for snap in snaps if check(snap)) / count, 3)
    return out


def coverage_report(snaps: list[Any]) -> dict[str, Any]:
    """Fill rates for the label fields. No book-match rate."""
    report = _rates(snaps)
    by_side: dict[str, Any] = {}
    for side in ("offense", "defense"):
        rows = [snap for snap in snaps if str(_value(snap, "side") or "") == side]
        by_side[side] = _rates(rows)
    report["by_side"] = by_side
    return report


@lru_cache(maxsize=1)
def _catalog_plays() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[1] / "cfb_coach" / "data" / "madden27" / "playbooks.json"
    if not path.is_file():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    found: list[str] = []
    seen: set[str] = set()
    for side in ("offense", "defense"):
        for body in ((data.get("books") or {}).get(side) or {}).values():
            if not isinstance(body, dict):
                continue
            for info in (body.get("formations") or {}).values():
                if not isinstance(info, dict):
                    continue
                for play in info.get("plays") or []:
                    if isinstance(play, str) and play not in seen:
                        seen.add(play)
                        found.append(play)
    return tuple(found)


def exact_book_match(play: str, names: list[str] | None = None) -> str:
    """Exact compact match only. An OCR miss stays unmatched.

    The caller keeps ``play`` as read. This return value is a separate field.
    """
    if not play:
        return ""
    catalog = list(names) if names is not None else list(_catalog_plays())
    key = _compact(play)
    hits = [name for name in catalog if _compact(name) == key]
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return unique[0]
    return ""
