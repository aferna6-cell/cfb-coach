"""Conservative, anonymous route geometry for provisional matchup scoring only."""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from statistics import median

from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model.offense_play_features import PlayStructure
from cfb_coach.madden.model.offense_taxonomy import RouteSpec

_DOCUMENTED = frozenset({"in_game_verified", "diagram_verified", "source_documented"})


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, OverflowError):
        return None
    return point if all(isfinite(v) for v in point) else None


def _family(candidate, points, start, center, side, band):
    if candidate == "vertical":
        return "seam" if side == "middle" else "streak"
    if candidate == "break_in":
        return {"short": "slant", "intermediate": "in", "deep": "post"}[band]
    if candidate == "break_out":
        return "corner" if band == "deep" else "out"
    if candidate == "curl_or_comeback":
        return {"short": "hitch", "intermediate": "curl", "deep": "comeback"}[band]
    if candidate == "flat_or_horizontal":
        xs = [start[0], *(p[0] for p in points)]
        return "drag" if min(xs) < center < max(xs) else "flat"
    if candidate == "backward_or_swing":
        return "swing"
    if candidate in {"stem_then_diagonal", "diagonal_upfield"}:
        # Last laterally moving segment, not identity, breaks, or claimed yards.
        end = points[-1]
        prev = next((p for p in reversed(points[:-1]) if p[0] != end[0]), None)
        if prev is None or side == "middle":
            return None
        inside = (end[0] - prev[0]) * (center - start[0]) > 0
        if band == "deep":
            return "post" if inside else "corner"
        if band == "short" and inside:
            return "slant"
    return None


def decode_provisional_geometry(
    geometry: Mapping,
    *,
    formation: str,
    play: str,
    feature_tier: str,
) -> PlayStructure:
    """Grok geometry graph -> PlayStructure. Never claims player identity or QB reads.

    Only diagram_px_unreviewed is supported: smaller y is upfield, so travel
    is start_y - end_y. Bands use marker x-spread, not yards: <=25% short,
    <=60% intermediate, otherwise deep. Alignment within 5% of that spread
    of the median marker x is middle. Without marker spread use route-start
    spread; without either scale leave geometry undecoded.
    """
    fallback = sm.structure_any(formation, play)
    if not isinstance(geometry, Mapping) or geometry.get("coordinate_frame") != "diagram_px_unreviewed":
        return fallback
    markers = [m for m in (geometry.get("markers") or ()) if isinstance(m, Mapping)]
    starts = [p for m in markers if (p := _point(m.get("start_px"))) is not None]
    candidates = []
    for marker in markers:
        for element in marker.get("elements") or ():
            if isinstance(element, Mapping) and element.get("kind") == "route_proposal":
                candidates.append((_point(marker.get("start_px")), element))
    for element in geometry.get("unowned_elements") or ():
        if isinstance(element, Mapping) and element.get("kind") == "route":
            candidates.append((_point(element.get("start_px")), element))
    routes = []
    for start, element in candidates:
        polyline = element.get("polyline") or ()
        points = [_point(p) for p in polyline]
        if len(points) < 2 or any(p is None for p in points):
            continue
        routes.append((start or points[0], points, element.get("route_family_candidate")))
    alignment = starts or [r[0] for r in routes]
    if not alignment:
        return fallback
    center = median(p[0] for p in alignment)
    width = max(p[0] for p in alignment) - min(p[0] for p in alignment)
    if width <= 0:
        alignment = [r[0] for r in routes]
        width = max((p[0] for p in alignment), default=0) - min((p[0] for p in alignment), default=0)
    if width <= 0:
        return fallback
    tier = "high_confidence_extracted" if feature_tier == "high_confidence_extracted" else "partial"
    decoded = []
    for start, points, candidate in sorted(routes, key=lambda r: r[0][0]):
        offset = start[0] - center
        side = "middle" if abs(offset) <= width * .05 else "left" if offset < 0 else "right"
        travel = (start[1] - points[-1][1]) / width
        band = "short" if travel <= .25 else "intermediate" if travel <= .60 else "deep"
        family = _family(candidate, points, start, center, side, band)
        if family is not None:
            decoded.append((f"geom_{len(decoded)}", RouteSpec(family, side, band), tier))
    if not decoded:
        return fallback
    return PlayStructure(formation=formation, play=play, routes=tuple(decoded),
                         skill_slots=(), personnel_tier="unknown", hint=fallback.hint)


def resolve_scoring_structure(
    formation: str,
    play: str,
    *,
    geometry: Mapping | None,
    feature_tier: str | None,
    model_use_permitted: bool,
) -> PlayStructure:
    """First-party documented routes win; else decoded provisional routes;
    else the existing hint-only structure_any fallback.
    """
    first_party = sm.structure_any(formation, play)
    if any(tier in _DOCUMENTED for _, _, tier in first_party.routes):
        return first_party
    if geometry is not None and model_use_permitted:
        decoded = decode_provisional_geometry(geometry, formation=formation, play=play,
                                              feature_tier=feature_tier or "partial")
        if decoded.n_known > 0:
            return decoded
    return first_party
