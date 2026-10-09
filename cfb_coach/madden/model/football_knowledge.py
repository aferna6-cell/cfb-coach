"""Compiled football knowledge for the offensive coordinator.

General coaching principles, Madden research, verified in-game details,
empirical results, and hypotheses stay in separate layers. A play name is
not proof of its route design. This module is the live representation:
it does not call a language model.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

KNOWLEDGE_VERSION = "football_knowledge.v1"
LIVE_PATH_CALLS_LLM = False

LAYERS = (
    "general_principle",
    "madden_gameplay",
    "verified_ingame",
    "empirical",
    "hypothesis",
)

# A name-only mapping is a hypothesis about the concept family, not a diagram.
_NAME_CONFIDENCE = 0.35

SOURCES: dict[str, dict[str, Any]] = {
    "usa_football_match_front": {
        "title": "Setting the front in match coverage (man-free with zone principles)",
        "url": "https://blogs.usafootball.com/blog/7313/setting-the-front-in-match-coverage-man-free-with-zone-principles",
        "layer": "general_principle",
        "retrieved": "2026-10-09",
        "game_versions": ["general_football"],
        "confidence": 0.55,
        "note": (
            "General coaching discussion of man-match fronts. It does not "
            "describe how Madden 27 draws a play."
        ),
    },
    "openplay_passing_concepts": {
        "title": "Passing concepts",
        "url": "https://openplayfootball.com/learn/passing-concepts",
        "layer": "general_principle",
        "retrieved": "2026-10-09",
        "game_versions": ["general_football"],
        "confidence": 0.55,
        "note": (
            "General concept families. A matching Madden play name is not "
            "evidence that the game uses this route drawing."
        ),
    },
    "madden_catalog": {
        "title": "Installed Madden offensive catalog",
        "url": None,
        "layer": "madden_gameplay",
        "retrieved": "2026-10-09",
        "game_versions": ["madden27"],
        "confidence": 0.9,
        "note": "Authoritative list of selectable formation/play names. Not a route diagram.",
    },
}


def _concept(
    concept_id: str,
    display_name: str,
    family: str,
    *,
    attempts: str,
    stresses: Sequence[str],
    route_depth: str,
    time_to_develop: str,
    horizontal: bool,
    vertical: bool,
    may_exploit: Sequence[str],
    reduced_by: Sequence[str],
    complements: Sequence[str],
    plausible_when: Sequence[str],
    sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "display_name": display_name,
        "family": family,
        "attempts": attempts,
        "stresses": list(stresses),
        "route_depth": route_depth,
        "time_to_develop": time_to_develop,
        "horizontal_conflict": horizontal,
        "vertical_conflict": vertical,
        "may_exploit": list(may_exploit),
        "reduced_by": list(reduced_by),
        "qb_read": None,
        "personnel": {
            "requirement": "not verified for Madden 27",
            "attributes_known": False,
        },
        "protection": "unknown for this Madden play; principle only",
        "complements": list(complements),
        "adjustment_opportunities": ["sourced_hot_route_or_protection_only_when_live_and_legal"],
        "plausible_when": list(plausible_when),
        "route_diagram": None,
        "player_assignments": None,
        "controller_inputs": None,
        "sources": list(sources),
        "layer": "general_principle",
        "confidence": 0.55,
        "game_versions": ["general_football"],
        "madden_27_applicability": "unverified",
        "name_match_is_not_route_proof": True,
    }


_PASS_SOURCES = ("openplay_passing_concepts", "usa_football_match_front")
_RUN_SOURCES = ("usa_football_match_front",)

CONCEPTS: dict[str, dict[str, Any]] = {
    row["concept_id"]: row
    for row in (
        _concept("mesh", "Mesh / shallow cross", "pass",
                 attempts="Two shallow crossers create a horizontal conflict underneath",
                 stresses=("underneath zone defender", "man-trail technique"),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("man", "off_coverage"),
                 reduced_by=("pressure", "collision", "robber"),
                 complements=("play_action", "four_verts", "screen"),
                 plausible_when=("short_or_medium_conversion", "observed_man"),
                 sources=_PASS_SOURCES),
        _concept("flood_sail", "Flood / Sail", "pass",
                 attempts="A three-level stretch to one side",
                 stresses=("flat defender", "curl or hook", "deep third or quarter"),
                 route_depth="multi_level", time_to_develop="intermediate", horizontal=False, vertical=True,
                 may_exploit=("cover_2", "cover_3", "two_high_unspecified"),
                 reduced_by=("pressure", "compressed_field"),
                 complements=("inside_zone", "play_action", "screen"),
                 plausible_when=("open_field", "observed_zone_shell"),
                 sources=_PASS_SOURCES),
        _concept("dagger", "Dagger", "pass",
                 attempts="A vertical clear-out with an intermediate in-breaking route",
                 stresses=("middle of the field between safeties",),
                 route_depth="intermediate", time_to_develop="intermediate", horizontal=False, vertical=True,
                 may_exploit=("cover_2", "two_high_unspecified"),
                 reduced_by=("pressure", "single_high_robber"),
                 complements=("play_action", "mesh"),
                 plausible_when=("open_field",),
                 sources=_PASS_SOURCES),
        _concept("drive", "Drive", "pass",
                 attempts="A shallow cross combined with an intermediate in-cut",
                 stresses=("linebacker depth", "underneath hook"),
                 route_depth="intermediate", time_to_develop="intermediate", horizontal=True, vertical=True,
                 may_exploit=("cover_3", "man"),
                 reduced_by=("pressure", "collision"),
                 complements=("inside_zone", "slant_flat"),
                 plausible_when=("medium_distance",),
                 sources=_PASS_SOURCES),
        _concept("levels", "Levels", "pass",
                 attempts="Two in-breaking routes at different depths",
                 stresses=("hook defender high-low",),
                 route_depth="intermediate", time_to_develop="intermediate", horizontal=False, vertical=True,
                 may_exploit=("cover_3", "cover_2"),
                 reduced_by=("pressure",),
                 complements=("play_action", "mesh"),
                 plausible_when=("medium_distance",),
                 sources=_PASS_SOURCES),
        _concept("smash", "Smash", "pass",
                 attempts="A hitch and a corner put the flat defender in conflict",
                 stresses=("flat defender",),
                 route_depth="intermediate", time_to_develop="intermediate", horizontal=False, vertical=True,
                 may_exploit=("cover_2", "cover_3"),
                 reduced_by=("press", "pressure"),
                 complements=("stick", "slant_flat"),
                 plausible_when=("sideline", "red_zone_if_not_compressed_past_the_corner"),
                 sources=_PASS_SOURCES),
        _concept("stick", "Stick", "pass",
                 attempts="A quick stick route with a flat option",
                 stresses=("flat and hook defenders",),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("cover_3", "off_coverage"),
                 reduced_by=("press", "collision"),
                 complements=("smash", "inside_zone"),
                 plausible_when=("short_or_medium_conversion",),
                 sources=_PASS_SOURCES),
        _concept("spacing", "Spacing", "pass",
                 attempts="Receivers settle in the voids of a zone",
                 stresses=("zone spacing",),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("zone", "cover_3", "cover_2"),
                 reduced_by=("man", "press"),
                 complements=("inside_zone", "screen"),
                 plausible_when=("observed_zone"),
                 sources=_PASS_SOURCES),
        _concept("curl_flat", "Curl-flat", "pass",
                 attempts="A curl and a flat stretch one side of the zone",
                 stresses=("curl/flat defender",),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("cover_3", "cover_2"),
                 reduced_by=("man", "press"),
                 complements=("stick", "flood_sail"),
                 plausible_when=("short_or_medium_conversion",),
                 sources=_PASS_SOURCES),
        _concept("four_verts", "Four verticals / seams", "pass",
                 attempts="Vertical routes stress the deep defenders",
                 stresses=("single-high safety", "deep quarters"),
                 route_depth="deep", time_to_develop="slow", horizontal=False, vertical=True,
                 may_exploit=("cover_1", "cover_3"),
                 reduced_by=("pressure", "two_high_unspecified", "compressed_field"),
                 complements=("play_action", "inside_zone", "screen"),
                 plausible_when=("need_explosive", "open_field", "single_high_observed"),
                 sources=_PASS_SOURCES),
        _concept("slant_flat", "Slant-flat", "pass",
                 attempts="A quick slant with a flat creates an immediate conflict",
                 stresses=("underneath man or flat defender",),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("man", "off_coverage", "cover_3"),
                 reduced_by=("press", "collision", "jam"),
                 complements=("mesh", "screen", "inside_zone"),
                 plausible_when=("pressure_live", "short_conversion"),
                 sources=_PASS_SOURCES),
        _concept("wheel", "Wheel", "pass",
                 attempts="A back or receiver wheels vertically after a flat stem",
                 stresses=("flat defender turning his back", "man trail"),
                 route_depth="deep", time_to_develop="slow", horizontal=False, vertical=True,
                 may_exploit=("man", "cover_3"),
                 reduced_by=("pressure", "two_high_unspecified"),
                 complements=("play_action", "screen"),
                 plausible_when=("open_field",),
                 sources=_PASS_SOURCES),
        _concept("texas_angle", "Texas / angle", "pass",
                 attempts="A back releases on an angle or Texas route through the linebackers",
                 stresses=("linebacker fit",),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("man", "cover_3"),
                 reduced_by=("pressure", "collision"),
                 complements=("inside_zone", "play_action"),
                 plausible_when=("short_conversion",),
                 sources=_PASS_SOURCES),
        _concept("screen", "Screen", "pass",
                 attempts="Throw behind the rush and let blockers work in space",
                 stresses=("unblocked edge or pursuing linebacker",),
                 route_depth="behind_los", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("pressure",),
                 reduced_by=("patient_flat_defender", "long_yardage"),
                 complements=("four_verts", "play_action", "inside_zone"),
                 plausible_when=("observed_pressure",),
                 sources=_PASS_SOURCES),
        _concept("play_action", "Play-action", "pass",
                 attempts="Sell a run look and throw behind the linebackers",
                 stresses=("linebackers who fit the run",),
                 route_depth="intermediate", time_to_develop="intermediate", horizontal=False, vertical=True,
                 may_exploit=("run_fit", "single_high"),
                 reduced_by=("pressure", "two_high_unspecified"),
                 complements=("inside_zone", "outside_zone", "four_verts"),
                 plausible_when=("after_verified_runs_in_the_same_look",),
                 sources=_PASS_SOURCES),
        _concept("rpo", "RPO", "rpo",
                 attempts="A run and a quick throw conflict the same defender",
                 stresses=("edge or flat defender",),
                 route_depth="short", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("light_box", "conflict_defender"),
                 reduced_by=("pressure",),
                 complements=("inside_zone", "slant_flat"),
                 plausible_when=("early_down",),
                 sources=_PASS_SOURCES),
        _concept("inside_zone", "Inside zone", "run",
                 attempts="Downhill run through interior gaps",
                 stresses=("interior run fits",),
                 route_depth="run", time_to_develop="quick", horizontal=False, vertical=False,
                 may_exploit=("light_box",),
                 reduced_by=("heavy_box", "two_minute_trailing"),
                 complements=("play_action", "boot", "screen"),
                 plausible_when=("short_yardage", "early_down", "protect_lead"),
                 sources=_RUN_SOURCES),
        _concept("outside_zone", "Outside zone", "run",
                 attempts="Stretch the defense horizontally and cut up",
                 stresses=("edge contain",),
                 route_depth="run", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("overpursuit",),
                 reduced_by=("set_edge", "heavy_box"),
                 complements=("play_action", "boot"),
                 plausible_when=("early_down",),
                 sources=_RUN_SOURCES),
        _concept("stretch", "Stretch", "run",
                 attempts="Wide zone stretch of the front",
                 stresses=("edge contain",),
                 route_depth="run", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("overpursuit",),
                 reduced_by=("set_edge",),
                 complements=("play_action", "boot"),
                 plausible_when=("early_down",),
                 sources=_RUN_SOURCES),
        _concept("power", "Power", "run",
                 attempts="Gap scheme with a kick-out and a down block",
                 stresses=("playside edge and linebacker",),
                 route_depth="run", time_to_develop="quick", horizontal=False, vertical=False,
                 may_exploit=("light_box",),
                 reduced_by=("heavy_box",),
                 complements=("play_action", "counter"),
                 plausible_when=("short_yardage", "goal_line"),
                 sources=_RUN_SOURCES),
        _concept("counter", "Counter", "run",
                 attempts="Misdirection against the flow of the blocks",
                 stresses=("backside pursuit",),
                 route_depth="run", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("fast_flow",),
                 reduced_by=("spill_player",),
                 complements=("power", "inside_zone"),
                 plausible_when=("early_down",),
                 sources=_RUN_SOURCES),
        _concept("trap", "Trap", "run",
                 attempts="Let a defender upfield and trap him",
                 stresses=("penetrating interior defender",),
                 route_depth="run", time_to_develop="quick", horizontal=False, vertical=False,
                 may_exploit=("pressure",),
                 reduced_by=("two-gapping_nose",),
                 complements=("inside_zone", "draw"),
                 plausible_when=("observed_pressure",),
                 sources=_RUN_SOURCES),
        _concept("draw", "Draw", "run",
                 attempts="Show pass and run after the rush comes upfield",
                 stresses=("pass rush lanes",),
                 route_depth="run", time_to_develop="intermediate", horizontal=False, vertical=False,
                 may_exploit=("pressure",),
                 reduced_by=("patient_linebacker",),
                 complements=("four_verts", "screen"),
                 plausible_when=("observed_pressure",),
                 sources=_RUN_SOURCES),
        _concept("dive", "Dive", "run",
                 attempts="Quick interior handoff in a short field",
                 stresses=("interior gap",),
                 route_depth="run", time_to_develop="quick", horizontal=False, vertical=False,
                 may_exploit=("goal_line_heavy",),
                 reduced_by=("open_field_light_box_is_not_required",),
                 complements=("qb_run", "play_action"),
                 plausible_when=("goal_line", "short_yardage"),
                 sources=_RUN_SOURCES),
        _concept("option", "Option", "run",
                 attempts="Read an edge defender and keep or pitch",
                 stresses=("edge defender",),
                 route_depth="run", time_to_develop="quick", horizontal=True, vertical=False,
                 may_exploit=("unblocked_edge",),
                 reduced_by=("two_minute_trailing",),
                 complements=("inside_zone", "rpo"),
                 plausible_when=("early_down",),
                 sources=_RUN_SOURCES),
        _concept("qb_run", "QB run", "run",
                 attempts="Quarterback keep, sneak, or designed run",
                 stresses=("interior or edge gap left by the read",),
                 route_depth="run", time_to_develop="quick", horizontal=False, vertical=False,
                 may_exploit=("short_yardage", "goal_line"),
                 reduced_by=("two_minute_trailing",),
                 complements=("dive", "option"),
                 plausible_when=("short_yardage", "goal_line"),
                 sources=_RUN_SOURCES),
    )
}

# Specific patterns before generic ones. A match is only a name hint.
_NAME_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmesh\b", re.I), "mesh"),
    (re.compile(r"shallow\s+cross", re.I), "mesh"),
    (re.compile(r"\bsail\b|\bflood\b", re.I), "flood_sail"),
    (re.compile(r"\bdagger\b", re.I), "dagger"),
    (re.compile(r"\bdrive\b", re.I), "drive"),
    (re.compile(r"\blevels\b", re.I), "levels"),
    (re.compile(r"\bsmash\b", re.I), "smash"),
    (re.compile(r"\bstick\b", re.I), "stick"),
    (re.compile(r"\bspacing\b", re.I), "spacing"),
    (re.compile(r"curl[\s-]*flat", re.I), "curl_flat"),
    (re.compile(r"four\s*vert|4\s*verts?|\bverticals\b|\bseams?\b", re.I), "four_verts"),
    (re.compile(r"slant[\s-]*flat|\bslants?\b", re.I), "slant_flat"),
    (re.compile(r"\bwheel\b", re.I), "wheel"),
    (re.compile(r"\btexas\b|angle\s+route", re.I), "texas_angle"),
    (re.compile(r"\bscreen\b", re.I), "screen"),
    (re.compile(r"play[\s-]*action|\bpa\b|\bboot\b", re.I), "play_action"),
    (re.compile(r"\brpo\b|\bbubble\b", re.I), "rpo"),
    (re.compile(r"inside\s+zone|\biz\b|\bduo\b", re.I), "inside_zone"),
    (re.compile(r"outside\s+zone|\boz\b", re.I), "outside_zone"),
    (re.compile(r"\bstretch\b", re.I), "stretch"),
    (re.compile(r"\bpower\b", re.I), "power"),
    (re.compile(r"\bcounter\b", re.I), "counter"),
    (re.compile(r"\btrap\b", re.I), "trap"),
    (re.compile(r"\bdraw\b", re.I), "draw"),
    (re.compile(r"\bdive\b", re.I), "dive"),
    (re.compile(r"\boption\b", re.I), "option"),
    (re.compile(r"\bsneak\b|qb\s+run|qb\s+power|qb\s+draw", re.I), "qb_run"),
    (re.compile(r"\bzone\b", re.I), "inside_zone"),
)


def validate_knowledge(store: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
    """Refuse invented routes, button sequences, or sourceless concepts."""
    problems: list[str] = []
    rows = store if store is not None else CONCEPTS
    for concept_id, row in rows.items():
        if row.get("route_diagram") is not None and row.get("layer") != "verified_ingame":
            problems.append(f"{concept_id}: route diagram without verified_ingame layer")
        if row.get("controller_inputs"):
            problems.append(f"{concept_id}: controller input is not knowledge")
        if row.get("player_assignments"):
            problems.append(f"{concept_id}: player assignment was not verified")
        if not row.get("sources"):
            problems.append(f"{concept_id}: missing sources")
        for source_id in row.get("sources") or []:
            if source_id not in SOURCES:
                problems.append(f"{concept_id}: unknown source {source_id}")
        if row.get("madden_27_applicability") == "verified" and row.get("layer") != "verified_ingame":
            problems.append(f"{concept_id}: Madden applicability marked verified without that layer")
    return problems


def compiled_knowledge() -> dict[str, Any]:
    """Efficient live snapshot. No network and no language-model call."""
    problems = validate_knowledge()
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "version": KNOWLEDGE_VERSION,
        "live_path_calls_llm": LIVE_PATH_CALLS_LLM,
        "concepts": CONCEPTS,
        "sources": SOURCES,
        "layers": list(LAYERS),
    }


def profile_for_play(play: str | None) -> dict[str, Any]:
    """Map a catalog play name to a concept. Unknown stays unknown."""
    text = play or ""
    for pattern, concept_id in _NAME_HINTS:
        if pattern.search(text):
            principle = dict(CONCEPTS[concept_id])
            return {
                "play": play,
                "matched": True,
                "match_basis": "play_name_hint",
                "confidence": _NAME_CONFIDENCE,
                "layer": "hypothesis",
                "route_information": "unknown",
                "route_diagram": None,
                "player_assignments": None,
                "controller_inputs": None,
                "name_is_not_route_proof": True,
                "principle": principle,
                "concept_id": concept_id,
                "sources": list(principle["sources"]) + ["madden_catalog"],
                "note": (
                    "The play name suggests this concept family. It is not a "
                    "verified Madden route diagram."
                ),
            }
    return {
        "play": play,
        "matched": False,
        "match_basis": None,
        "confidence": 0.0,
        "layer": "unknown",
        "route_information": "unknown",
        "route_diagram": None,
        "player_assignments": None,
        "controller_inputs": None,
        "name_is_not_route_proof": True,
        "principle": None,
        "concept_id": None,
        "sources": ["madden_catalog"],
        "note": "No concept is assigned. Missing route details stay unknown.",
    }


def concepts_in_plays(plays: Sequence[str]) -> set[str]:
    found: set[str] = set()
    for play in plays:
        concept_id = profile_for_play(play).get("concept_id")
        if concept_id:
            found.add(str(concept_id))
    return found


def knowledge_report(concept: str | None = None) -> dict[str, Any]:
    """Read-only view of one concept, or the catalog of concept ids."""
    problems = validate_knowledge()
    if concept:
        key = concept.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "flood": "flood_sail", "sail": "flood_sail", "verticals": "four_verts",
            "verts": "four_verts", "slant": "slant_flat", "texas": "texas_angle",
            "curl": "curl_flat", "pa": "play_action", "iz": "inside_zone",
            "oz": "outside_zone", "qb": "qb_run",
        }
        key = aliases.get(key, key)
        row = CONCEPTS.get(key)
        if row is None:
            return {
                "version": KNOWLEDGE_VERSION,
                "found": False,
                "concept": concept,
                "note": "Unknown concept id. Missing details were not invented.",
                "known_concepts": sorted(CONCEPTS),
            }
        return {
            "version": KNOWLEDGE_VERSION,
            "found": True,
            "live_path_calls_llm": False,
            "concept": row,
            "sources": [SOURCES[source_id] for source_id in row["sources"]],
            "validation_problems": problems,
            "note": "General principle plus a name hint. Not a Madden route diagram.",
        }
    return {
        "version": KNOWLEDGE_VERSION,
        "found": True,
        "live_path_calls_llm": False,
        "concepts": [
            {
                "concept_id": row["concept_id"],
                "display_name": row["display_name"],
                "family": row["family"],
                "layer": row["layer"],
                "confidence": row["confidence"],
                "madden_27_applicability": row["madden_27_applicability"],
                "route_diagram": None,
            }
            for row in CONCEPTS.values()
        ],
        "sources": list(SOURCES.values()),
        "validation_problems": problems,
    }
