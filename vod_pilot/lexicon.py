"""Formation, play, and coverage names taken from the packaged playbooks.

Matching is on letters and digits only, so ``BUNCHSTRNASTY`` can hit
``Gun Bunch Str Nasty``. A menu that lists several formations returns no
single name: the called play is the highlighted row, and OCR does not know
which row that is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_NORM_RX = re.compile(r"[^A-Z0-9]")
_COVERAGE_FALLBACK = (
    "Cover 0",
    "Cover 1",
    "Cover 2",
    "Cover 3",
    "Cover 4",
    "Cover 6",
    "Cover 9",
    "Tampa 2",
    "Quarters",
    "Palms",
    "Man",
    "Cover 2 Invert",
    "Cover 4 Quarters",
    "Cover 3 Match",
    "Cover 6",
)


def _norm(text: str) -> str:
    return _NORM_RX.sub("", text.upper())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_madden_playbook() -> Path:
    return _repo_root() / "cfb_coach" / "data" / "madden27" / "playbooks.json"


def default_cfb_formations() -> Path:
    return _repo_root() / "cfb_coach" / "data" / "cfb27_formations.json"


@dataclass
class Lexicon:
    formations: list[str] = field(default_factory=list)
    plays: list[str] = field(default_factory=list)
    coverages: list[str] = field(default_factory=list)
    _form_keys: list[tuple[str, str]] = field(default_factory=list)
    _play_keys: list[tuple[str, str]] = field(default_factory=list)
    _cov_keys: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._form_keys = _formation_index(self.formations)
        self._tail_keys = _tail_index(self.formations)
        self._play_keys = _index(self.plays, min_len=6)
        self._cov_keys = _index(self.coverages, min_len=4)

    def match_formation(self, texts: list[str]) -> str:
        hits = self.formation_hits(texts)
        if len(hits) == 1:
            return hits[0]
        return ""

    def formation_hits(self, texts: list[str]) -> list[str]:
        """Formation names on this frame.

        Gun and Shotgun share the tail ``BUNCHSTRNASTY``, so that tail is not
        a unique alias. A frame that also contains ``CLUSTER`` is a menu.
        One matched name plus any other formation row means abstain.
        """
        blob = _norm(" ".join(texts))
        if not blob:
            return []
        present = [(key, owners) for key, owners in self._tail_keys if key in blob]
        kept: list[tuple[str, list[str]]] = []
        for key, owners in present:
            if any(key != other and key in other for other, _ in present):
                continue
            kept.append((key, owners))
        if len(kept) > 1:
            return []
        if len(kept) == 1:
            owners = list(dict.fromkeys(kept[0][1]))
            if len(owners) == 1:
                return owners
            return []
        return _hits(self._form_keys, texts)

    def match_play(self, texts: list[str]) -> str:
        """A play name counts only when exactly one lexicon play is present.

        The play-call screen prints the whole formation tab. Several hits means
        we can see the menu and cannot see which row is selected.
        """
        hits = _hits(self._play_keys, texts)
        if len(hits) == 1:
            return hits[0]
        # Tooltip case: one play name buried in a long sentence, plus short
        # menu crumbs that are substrings of that same name, already dropped.
        long_lines = [t for t in texts if len(t) >= 35]
        if long_lines:
            long_hits = _hits(self._play_keys, long_lines)
            if len(long_hits) == 1:
                return long_hits[0]
        return ""

    def match_coverage(self, texts: list[str]) -> str:
        return _unique(self._cov_keys, texts)


def _tail_index(names: list[str]) -> list[tuple[str, list[str]]]:
    """Map a normalized formation string to every book name that uses it.

    Includes the full name and the name with the family word removed, so
    ``BUNCHSTRNASTY`` is visible even when it belongs to both Gun and Shotgun.
    """
    tails: dict[str, list[str]] = {}
    for name in names:
        parts = name.split()
        variants = [name]
        if len(parts) >= 2:
            variants.append(" ".join(parts[1:]))
        for variant in variants:
            key = _norm(variant)
            if len(key) < 7:
                continue
            bucket = tails.setdefault(key, [])
            if name not in bucket:
                bucket.append(name)
    indexed = sorted(tails.items(), key=lambda item: len(item[0]), reverse=True)
    return indexed


def _formation_index(names: list[str]) -> list[tuple[str, str]]:
    """Index the full formation and, when unique, the name without its family.

    OCR often drops the leading ``Gun`` and returns ``BUNCHSTRNASTY``.
    """
    indexed = _index(names, min_len=8)
    tails: dict[str, list[str]] = {}
    for name in names:
        parts = name.split()
        if len(parts) < 2:
            continue
        tail = _norm(" ".join(parts[1:]))
        if len(tail) < 7:
            continue
        tails.setdefault(tail, []).append(name)
    have = {key for key, _ in indexed}
    for tail, owners in tails.items():
        unique = list(dict.fromkeys(owners))
        if len(unique) == 1 and tail not in have:
            indexed.append((tail, unique[0]))
    indexed.sort(key=lambda item: len(item[0]), reverse=True)
    return indexed


def _index(names: list[str], *, min_len: int) -> list[tuple[str, str]]:
    indexed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        key = _norm(name)
        if len(key) < min_len or key in seen:
            continue
        seen.add(key)
        indexed.append((key, name))
    indexed.sort(key=lambda item: len(item[0]), reverse=True)
    return indexed


def _hits(index: list[tuple[str, str]], texts: list[str]) -> list[str]:
    blob = _norm(" ".join(texts))
    if not blob:
        return []
    found: list[tuple[str, str]] = []
    for key, name in index:
        if key in blob:
            found.append((key, name))
    # Drop a name that is only a piece of a longer name we also matched.
    kept: list[str] = []
    for key, name in found:
        if any(key != other and key in other for other, _ in found):
            continue
        kept.append(name)
    return kept


def _unique(index: list[tuple[str, str]], texts: list[str]) -> str:
    hits = _hits(index, texts)
    if len(hits) == 1:
        return hits[0]
    return ""


def _add_plays(body: dict, plays: set[str], formations: set[str]) -> None:
    formations_obj = body.get("formations") or {}
    if not isinstance(formations_obj, dict):
        return
    for fname, info in formations_obj.items():
        if isinstance(fname, str) and fname.strip():
            formations.add(fname.strip())
        if not isinstance(info, dict):
            continue
        family = str(info.get("family") or "").strip()
        sett = str(info.get("set") or "").strip()
        if family and sett:
            formations.add(f"{family} {sett}")
        for play in info.get("plays") or []:
            if isinstance(play, str) and play.strip():
                plays.add(play.strip())


def load_lexicon(
    madden_playbook: Path | None = None,
    cfb_formations: Path | None = None,
) -> Lexicon:
    formations: set[str] = set()
    plays: set[str] = set()
    coverages: set[str] = set(_COVERAGE_FALLBACK)

    mpath = madden_playbook or default_madden_playbook()
    if mpath.is_file():
        data = json.loads(mpath.read_text(encoding="utf-8"))
        books = data.get("books") or {}
        for side in ("offense", "defense"):
            for body in (books.get(side) or {}).values():
                if isinstance(body, dict):
                    _add_plays(body, plays, formations)

    cpath = cfb_formations or default_cfb_formations()
    if cpath.is_file():
        data = json.loads(cpath.read_text(encoding="utf-8"))
        for fname, info in (data.get("formations") or {}).items():
            if isinstance(fname, str):
                formations.add(fname)
            if isinstance(info, dict):
                for play in info.get("plays") or []:
                    if isinstance(play, str):
                        plays.add(play)

    for play in list(plays):
        if play.lower().startswith("cover") or "quarters" in play.lower() or "tampa" in play.lower():
            coverages.add(play)

    return Lexicon(
        formations=sorted(formations),
        plays=sorted(plays),
        coverages=sorted(coverages),
    )
