"""Rank concepts and plays for one situation. This does not call a play.

A later decision pipeline can call ``rank_concepts`` instead of the
post-selection override in ``vod_model/live.py``. The function does not read
the database, the environment, or the pair model, and it does not edit a book.

Lookup, most specific first:

1. Coverage family, when the look names one that this stratum has seen.
2. ``down_dist`` (``down=<d>|dist=<band>``).
3. ``down``.
4. ``all``.

Distance bands match the concept trainer: short <=2, medium 3-6, long 7-10,
xlong 11+. The first level that has a concept at med or high after the
streamer rule is the decision level. Med needs at least 2 streamers. High
needs at least 3. A level where every concept is under that bar is not a
decision. Plays under a concept are already shrunk toward that concept and
are ranked by that shrunk rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cfb_coach.vod_model.concept_loader import ConceptModel

_NORM = re.compile(r"[^a-z0-9]")
_ACTIONABLE = {"med", "high"}
# Longer families before the tokens they contain.
_COVERAGE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cover_2_man", ("cover2man",)),
    ("cover_0", ("cover0", "coverzero")),
    ("tampa_2", ("tampa2", "tampa")),
    ("cover_1", ("cover1",)),
    ("cover_2", ("cover2",)),
    ("cover_3", ("cover3",)),
    ("cover_4", ("cover4", "quarters")),
    ("cover_6", ("cover6",)),
    ("cover_9", ("cover9",)),
    ("prevent", ("prevent",)),
    ("man_other", ("manother",)),
    ("zone_other", ("zoneother",)),
)


@dataclass(frozen=True)
class ConceptSituation:
    game: str
    opponent_type: str
    down: int | None = None
    distance: int | None = None
    coverage: str | None = None


@dataclass(frozen=True)
class RankedPlay:
    key: str
    name: str
    n: int
    n_vods: int
    n_streamers: int
    successes: int
    shrunk_success: float
    lower_bound: float
    baseline: float
    tier: str
    tier_with_streamer_rule: str
    in_book: bool
    formation: str | None


@dataclass(frozen=True)
class RankedConcept:
    key: str
    family: str
    n: int
    n_vods: int
    n_streamers: int
    successes: int
    raw_success: float
    shrunk_success: float
    lower_bound: float
    baseline: float
    tier: str
    tier_with_streamer_rule: str
    top_streamer: str
    top_streamer_share: float
    actionable: bool
    level: str
    bucket: str
    plays: tuple[RankedPlay, ...]

    @property
    def best_play(self) -> RankedPlay | None:
        return self.plays[0] if self.plays else None

    @property
    def best_in_book(self) -> RankedPlay | None:
        return next((play for play in self.plays if play.in_book), None)


@dataclass(frozen=True)
class ConceptRanking:
    version: str
    stratum: str
    coverage_family: str | None
    examined: tuple[str, ...]
    level: str | None
    bucket: str | None
    concepts: tuple[RankedConcept, ...]
    chosen: RankedConcept | None
    chosen_play: RankedPlay | None


def norm_name(value: str | None) -> str:
    return _NORM.sub("", (value or "").lower())


def distance_band(distance: int | None) -> str | None:
    if distance is None:
        return None
    if distance <= 2:
        return "short"
    if distance <= 6:
        return "medium"
    if distance <= 10:
        return "long"
    return "xlong"


def coverage_family(hint: str | None) -> str | None:
    """Map a look string onto a concept coverage family, or None."""
    token = norm_name(hint)
    if not token:
        return None
    if token in {family.replace("_", "") for family, _keys in _COVERAGE}:
        for family, _keys in _COVERAGE:
            if token == family.replace("_", ""):
                return family
    for family, keys in _COVERAGE:
        if any(key in token for key in keys):
            return family
    return None


def streamer_tier(cell: Mapping[str, Any]) -> str:
    """PR #18 tier, then the file's streamer rule (med >= 2 streamers, high >= 3)."""
    stored = cell.get("tier_with_streamer_rule")
    if stored:
        return str(stored)
    label = str(cell.get("tier") or "none")
    streamers = int(cell.get("n_streamers") or 0)
    if label == "high" and streamers < 3:
        label = "med"
    if label == "med" and streamers < 2:
        label = "low"
    return label


def _num(cell: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(cell.get(key) if cell.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _book_index(book: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for formation, plays in (book or {}).items():
        for play in plays:
            key = norm_name(play)
            if key and key not in found:
                found[key] = (str(formation), str(play))
    return found


def _rank_plays(cells: Sequence[Mapping[str, Any]], book_at: Mapping[str, tuple[str, str]]) -> tuple[RankedPlay, ...]:
    ranked: list[RankedPlay] = []
    for cell in cells:
        key = str(cell.get("key") or "")
        if not key:
            continue
        placed = book_at.get(key) or book_at.get(norm_name(str(cell.get("name") or "")))
        rule = streamer_tier(cell)
        ranked.append(RankedPlay(
            key=key,
            name=str(cell.get("name") or key),
            n=int(cell.get("n") or 0),
            n_vods=int(cell.get("n_vods") or 0),
            n_streamers=int(cell.get("n_streamers") or 0),
            successes=int(cell.get("successes") or 0),
            shrunk_success=_num(cell, "shrunk_success"),
            lower_bound=_num(cell, "lower_bound_90"),
            baseline=_num(cell, "baseline"),
            tier=str(cell.get("tier") or "none"),
            tier_with_streamer_rule=rule,
            in_book=placed is not None,
            formation=placed[0] if placed else None,
        ))
    ranked.sort(key=lambda play: (-play.shrunk_success, -play.n, play.key))
    return tuple(ranked)


def _concept_from(
    cell: Mapping[str, Any],
    *,
    family: str,
    level: str,
    bucket: str,
    book_at: Mapping[str, tuple[str, str]],
) -> RankedConcept:
    rule = streamer_tier(cell)
    return RankedConcept(
        key=str(cell.get("key") or ""),
        family=str(cell.get("family") or family),
        n=int(cell.get("n") or 0),
        n_vods=int(cell.get("n_vods") or 0),
        n_streamers=int(cell.get("n_streamers") or 0),
        successes=int(cell.get("successes") or 0),
        raw_success=_num(cell, "raw_success"),
        shrunk_success=_num(cell, "shrunk_success"),
        lower_bound=_num(cell, "lower_bound_90"),
        baseline=_num(cell, "baseline"),
        tier=str(cell.get("tier") or "none"),
        tier_with_streamer_rule=rule,
        top_streamer=str(cell.get("top_streamer") or ""),
        top_streamer_share=_num(cell, "top_streamer_share"),
        actionable=rule in _ACTIONABLE,
        level=level,
        bucket=bucket,
        plays=_rank_plays(list(cell.get("plays") or []), book_at),
    )


def _concepts_at(view: Mapping[str, Any], level: str, bucket: str) -> list[tuple[str, Mapping[str, Any]]]:
    if level == "coverage":
        body = (view.get("by_coverage") or {}).get(bucket) or {}
        return [(str(cell.get("family") or ""), cell) for cell in (body.get("concepts") or []) if isinstance(cell, dict)]
    table = (view.get("by_situation") or {}).get(level) or {}
    body = table.get(bucket) or {}
    found: list[tuple[str, Mapping[str, Any]]] = []
    for family in body.get("families") or []:
        if not isinstance(family, dict):
            continue
        family_key = str(family.get("key") or "")
        for cell in family.get("concepts") or []:
            if isinstance(cell, dict):
                found.append((family_key, cell))
    return found


def _levels(view: Mapping[str, Any], situation: ConceptSituation, family: str | None) -> list[tuple[str, str]]:
    levels: list[tuple[str, str]] = []
    if family and family in (view.get("by_coverage") or {}):
        levels.append(("coverage", family))
    band = distance_band(situation.distance)
    if situation.down in (1, 2, 3, 4) and band:
        bucket = f"down={situation.down}|dist={band}"
        if bucket in ((view.get("by_situation") or {}).get("down_dist") or {}):
            levels.append(("down_dist", bucket))
    if situation.down in (1, 2, 3, 4):
        bucket = f"down={situation.down}"
        if bucket in ((view.get("by_situation") or {}).get("down") or {}):
            levels.append(("down", bucket))
    if "all" in ((view.get("by_situation") or {}).get("all") or {}):
        levels.append(("all", "all"))
    return levels


def rank_concepts(
    model: ConceptModel,
    situation: ConceptSituation,
    book: Mapping[str, Sequence[str]] | None = None,
) -> ConceptRanking:
    """Rank concepts and their plays for ``situation``.

    ``book`` is an optional formation → plays map. It only marks which ranked
    plays are already in the book. It is not required, and it is not edited.
    """
    family = coverage_family(situation.coverage)
    view = model.view(situation.game, situation.opponent_type)
    empty = ConceptRanking(
        version=model.version,
        stratum=f"{situation.game}/{situation.opponent_type}",
        coverage_family=family,
        examined=(),
        level=None,
        bucket=None,
        concepts=(),
        chosen=None,
        chosen_play=None,
    )
    if not view:
        return empty
    book_at = _book_index(book)
    examined: list[str] = []
    fallback: tuple[RankedConcept, ...] = ()
    fallback_level: tuple[str, str] | None = None
    for level, bucket in _levels(view, situation, family):
        concepts = tuple(
            _concept_from(cell, family=family_key, level=level, bucket=bucket, book_at=book_at)
            for family_key, cell in _concepts_at(view, level, bucket)
        )
        concepts = tuple(sorted(concepts, key=lambda item: (-int(item.actionable), -item.shrunk_success, -item.lower_bound, -item.n, item.key)))
        examined.append(f"{level}:{bucket}")
        if fallback_level is None and concepts:
            fallback = concepts
            fallback_level = (level, bucket)
        actionable = [item for item in concepts if item.actionable]
        if not actionable:
            continue
        chosen = actionable[0]
        return ConceptRanking(
            version=model.version,
            stratum=f"{situation.game}/{situation.opponent_type}",
            coverage_family=family,
            examined=tuple(examined),
            level=level,
            bucket=bucket,
            concepts=concepts,
            chosen=chosen,
            chosen_play=chosen.best_in_book,
        )
    level, bucket = fallback_level or (None, None)
    return ConceptRanking(
        version=model.version,
        stratum=f"{situation.game}/{situation.opponent_type}",
        coverage_family=family,
        examined=tuple(examined),
        level=level,
        bucket=bucket,
        concepts=fallback,
        chosen=None,
        chosen_play=None,
    )
