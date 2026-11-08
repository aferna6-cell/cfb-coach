"""Read-only, local, offline-compiled diagram read structures for the live QB card.

The pregame compiler writes ONLY inferred route family/side/depth, never images,
URLs, verified player assignments or button mappings. No full catalog module
is imported in the live decision loop. Unknown/ambiguous routes stay UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.offense_play_features import PlayStructure
from cfb_coach.madden.model.offense_taxonomy import ROUTE_FAMILIES, RouteSpec

SCHEMA = "madden27.offense_live_reads.provisional.v1"
DOCUMENTED = frozenset(("diagram_verified", "in_game_verified", "source_documented"))
PROVISIONAL = frozenset(("partial", "high_confidence_extracted"))
SIDES = frozenset(("left", "middle", "right"))
BANDS = frozenset(("shallow", "short", "intermediate", "deep", "unspecified"))


def pairs_digest(pairs: Sequence[tuple[str, str]]) -> str:
    """Stable fingerprint of the entire confirmed formation/play inventory."""
    payload = sorted({(str(f), str(p)) for f, p in pairs})
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def default_path() -> Path:
    override = os.environ.get("CFB_COACH_OFFENSE_READ_CACHE")
    if override:
        return Path(override).expanduser()
    from cfb_coach.games import data_dir

    return Path(data_dir()) / "madden_ml_experimental" / "offense_live_reads.json"


@lru_cache(maxsize=6)
def _parse(path: str, mtime_ns: int, size: int) -> dict[str, Any] | None:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return None
    if not isinstance(record.get("pairs_digest"), str):
        return None
    rows = record.get("plays")
    if not isinstance(rows, list):
        return None
    index: dict[tuple[str, str], tuple[tuple[str, RouteSpec, str], ...]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        formation, play = row.get("formation"), row.get("play")
        if not isinstance(formation, str) or not isinstance(play, str) or not formation or not play:
            return None
        key = (formation, play)
        if key in index:
            return None
        routes = row.get("routes")
        if not isinstance(routes, list) or len(routes) > 12:
            return None
        parsed = []
        for index_number, route in enumerate(routes):
            if not isinstance(route, Mapping):
                return None
            family = route.get("family")
            side = route.get("side")
            band = route.get("band")
            tier = route.get("tier")
            if (family not in ROUTE_FAMILIES or side not in SIDES
                    or band not in BANDS or tier not in PROVISIONAL):
                return None
            # The provisional geometry has NO verified player identity.
            parsed.append((
                f"geom_{index_number}", RouteSpec(family, side, band), tier
            ))
        if parsed:
            index[key] = tuple(parsed)
    return {"pairs_digest": record["pairs_digest"], "index": index}


def structure_for_live_read(
    formation: str,
    play: str,
    stock: PlayStructure,
    *,
    installed_pairs_digest: str | None,
    path: str | Path | None = None,
) -> PlayStructure:
    """Preserve documented reads, else apply only an exact inventory-matched geometry cache.

    Anonymous geometry must never become eligible Madden hot-route targets:
    its skill_slots and read_order are always empty. This is a DISPLAY
    structure for the chosen stock play, not a new trained model feature.
    """
    if any(t in DOCUMENTED for _, _, t in stock.routes):
        return stock
    if not installed_pairs_digest:
        return stock
    candidate = Path(path).expanduser() if path is not None else default_path()
    try:
        stat = candidate.stat()
    except OSError:
        return stock
    parsed = _parse(str(candidate), stat.st_mtime_ns, stat.st_size)
    if not parsed or parsed["pairs_digest"] != installed_pairs_digest:
        return stock
    routes = parsed["index"].get((formation, play))
    if not routes:
        return stock
    return replace(
        stock, routes=routes, skill_slots=(), read_order=(),
        sourced_concepts=(), adjustments=()
    )
