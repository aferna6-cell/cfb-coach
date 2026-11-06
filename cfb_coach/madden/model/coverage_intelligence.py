"""Madden 27 coverage *scheme* intelligence: distinct from verified player play-art.

Each scheme is a knowledge model supported by EA's Madden 27 coverage-checks
and independently paraphrased public football education. The lookup may use
play-name words to ASSIGN A CANDIDATE SCHEME only. It cannot establish exact
who-rushes/who-covers facts. No learned outcome is claimed.

This module does not change the live coach and does not access proprietary
third-party play diagrams. Material marked inference cannot train individual
player assignments.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

EA = "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay"
MUT = "https://www.mut.gg/news/ask-huddle-42-cover-1-explained-in-madden/"


@dataclass(frozen=True)
class CoverageScheme:
    scheme: str
    family: str  # man|spot_zone|pattern_match|mixed|pressure|unknown
    deep_structure: str  # one_high|two_high|three_deep|four_deep|split_field|unknown
    philosophy: str
    bonus_defender: str | None = None
    quarter_side: str | None = None
    half_side: str | None = None
    variants: tuple[str, ...] = ()
    conditional_rules: tuple[Mapping[str, Any], ...] = ()
    scope: str = "schematic_not_play_art"
    sources: tuple[str, ...] = (EA,)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rule(trigger: str, behavior: str, level: str = "football_principle") -> dict[str, str]:
    return {"trigger": trigger, "expected_coverage_response": behavior,
            "evidence": level, "requires_confirmation_on_play_art": "true"}


SCHEMES: dict[str, CoverageScheme] = {
    "cover_1_hole": CoverageScheme(
        "cover_1_hole", "man", "one_high", "Man assignments with a low-middle hook/hole defender and one deep middle defender.",
        bonus_defender="middle_hook_hole_often_linebacker",
        conditional_rules=(_rule("receiver_enters_middle_hole", "hook player can assist man defenders in the middle"),),
        sources=(MUT,)),
    "cover_1_robber": CoverageScheme(
        "cover_1_robber", "man", "one_high", "Man coverage with a roaming low-middle robber, commonly a rotating safety.",
        bonus_defender="low_middle_robber_often_safety",
        conditional_rules=(_rule("shallow_cross_or_dig", "robber may undercut inside route while deep safety stays high"),),
        sources=(MUT,)),
    "cover_1_contain": CoverageScheme(
        "cover_1_contain", "man", "one_high", "Man coverage with edges encouraged to keep quarterback in the pocket and an interior QB spy.",
        bonus_defender="qb_spy_plus_edge_contain", sources=(MUT,)),
    "cover_1_pressure": CoverageScheme(
        "cover_1_pressure", "man", "one_high", "Man behind an extra rusher; additional middle help is reduced.",
        bonus_defender="added_pass_rusher", sources=(MUT,)),
    "cover_1_man_other": CoverageScheme(
        "cover_1_man_other", "man", "one_high", "One high man coverage. Bonus player or double-team assignment is play-dependent and unknown.",
        bonus_defender="unknown", sources=(MUT,)),
    "cover_2_man": CoverageScheme(
        "cover_2_man", "man", "two_high", "Two deep help players over man coverage underneath; not spot-drop Cover 2.",
        sources=(EA,)),
    "cover_2_spot": CoverageScheme(
        "cover_2_spot", "spot_zone", "two_high", "Two deep halves with underneath area defenders; unlike Cover 2 Man, no receiver is assigned to every underneath defender.",
        sources=(EA,)),
    "cover_2_invert": CoverageScheme(
        "cover_2_invert", "spot_zone", "two_high", "Post-snap rotation reverses customary coverage roles: outside defenders may sink to deep halves while safeties rotate underneath.",
        variants=("post_snap_rotation",),sources=(EA,)),
    "tampa_2": CoverageScheme(
        "tampa_2", "spot_zone", "two_high", "Two high halves and underneath zone players; middle-runner can carry a vertical threat through the deep middle.",
        conditional_rules=(_rule("vertical_threat_through_middle", "middle runner relates to middle seam rather than staying flat in low hook"),),
        sources=(EA,)),
    "cover_3_spot": CoverageScheme(
        "cover_3_spot", "spot_zone", "three_deep", "Three deep lanes and underneath zones. A one-high disguise does not alone prove Cover 3.",
        sources=(EA,)),
    "cover_3_match": CoverageScheme(
        "cover_3_match", "pattern_match", "three_deep", "Three-deep structure whose underneath/deep defenders read releases and can match receivers instead of only protecting a landmark.",
        conditional_rules=(_rule("receiver_vertical_release", "relevant pattern-match defender may carry receiver in assigned seam/deep rule"),
                           _rule("bunch_or_trips", "formation-check may supersede base rules; selected check required")),
        sources=(EA,)),
    "cover_4_drop": CoverageScheme(
        "cover_4_drop", "spot_zone", "four_deep", "Four deep quarters as spatial areas; not identical to four-deep pattern matching.",
        sources=(EA,)),
    "cover_4_quarters": CoverageScheme(
        "cover_4_quarters", "pattern_match", "four_deep", "Two-high quarters rules; defender responsibilities depend on vertical, inside and outside releases.",
        conditional_rules=(_rule("receiver_2_vertical", "quarters safety normally relates to number-2 vertical threat"),
                           _rule("receiver_2_under", "quarters safety can look for inside help rather than chase a nonvertical number-2"),
                           _rule("bunch_or_trips", "call-specific Box, Stubbie, Solo or other check can replace base behavior")),
        sources=(EA,)),
    "cover_4_palms": CoverageScheme(
        "cover_4_palms", "pattern_match", "four_deep", "Two-read Palms rules are not ordinary Quarters: outside corner and safety can exchange coverage when number-2 breaks quickly outside.",
        conditional_rules=(_rule("receiver_2_quick_out_or_flat", "palms corner may drive number-2 short out; safety can cover number-1 vertical"),
                           _rule("receiver_2_vertical", "keep safety connected to number-2 vertical under the selected check"),
                           _rule("bunch_or_stack", "selected match check may cause receiver exchanges")),
        sources=(EA,)),
    "cover_6": CoverageScheme(
        "cover_6", "mixed", "split_field", "Quarter-quarter-half: a quarters/match half of the formation and a Cover-2-style half; strength and check determine the two sides.",
        quarter_side="strength_by_default_unverified_on_play", half_side="weak_by_default_unverified_on_play",
        conditional_rules=(_rule("trips_to_quarters_side", "quarters-side defenders apply match check for number-2 and number-3"),
                           _rule("bunch_or_stack", "selected Palms/Quarters/Box/other check changes exchange rules")),
        sources=(EA,)),
    "cover_9": CoverageScheme(
        "cover_9", "mixed", "split_field", "Reverse-quarter-quarter-half orientation compared with Cover 6; half side and quarters side are reversed in the standard concept.",
        quarter_side="weak_by_default_unverified_on_play", half_side="strength_by_default_unverified_on_play",sources=(EA,)),
    "trap_2": CoverageScheme(
        "trap_2", "spot_zone", "two_high", "Cover 2 trap principle: underneath defender can jump a short/outside throw with deep support behind; exact Madden rules depend on play art.",
        sources=(EA,)),
    "unknown": CoverageScheme("unknown", "unknown", "unknown", "Scheme not resolved; no individual player roles inferred.", sources=()),
}

# EA's named checks: semantics are concept-level, NOT eleven Madden defender jobs.
MATCH_CHECKS: dict[str, str] = {
    "box": "Bunch distributed by directional coverage responsibilities.",
    "bingo": "Box-type check where an outside defender can lock number-1 if his release stays outside.",
    "triangle": "Coverage overload with a helper over multiple releasing receivers.",
    "point_triangle": "Triangle check specifically accounts for the bunch point receiver.",
    "skate": "Underneath width increase against a bunch/trips threat.",
    "skinny": "More aggressive carrying of certain vertical and outside releases.",
    "skinny_meg": "Skinny variant with backside corner locked on the isolated receiver.",
    "stress": "Disciplined deep spacing rather than chasing every vertical route.",
    "stubbie": "Trips: corner locks number-1 while inside defenders share numbers 2 and 3.",
    "stump": "Trips check with added help versus number-1 short release.",
    "solo": "Isolated receiver handled by backside corner; safety freed for other help.",
    "solo_cut": "Backside safety may take short inside-breaking release from isolated receiver.",
    "zone_it": "Turn off match and use fixed spatial zone rules for this check.",
}


@dataclass(frozen=True)
class ClassifiedPlay:
    scheme_id: str
    family: str
    evidence: str
    named_variations: tuple[str, ...]
    source: str
    allowed_as_verified_player_duties: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_defensive_play(name: str) -> ClassifiedPlay:
    """Infer the *scheme class* from name, not individual defender assignments.

    Pressure names and exotic packages are intentionally NOT interpreted by a
    number suffix: e.g. 'Overload 3 Seam' is not guaranteed Cover 3 Match.
    """
    n = re.sub(r"\s+", " ", (name or "").lower()).strip()
    variants = []
    if "press" in n: variants.append("press")
    if re.search(r"\broll\b", n): variants.append("roll")
    if "willie" in n: variants.append("willie_tag_meaning_unconfirmed")
    if "show 2" in n: variants.append("show_two_high")
    if "hard flat" in n: variants.append("hard_flat")
    if "cloud" in n: variants.append("cloud")
    if "buzz" in n: variants.append("buzz")
    if "wk" in n: variants.append("weak_tag_unconfirmed")
    if "field" in n: variants.append("field_tag_unconfirmed")
    if "double" in n or "dbl" in n: variants.append("double_team_tag")
    if "bracket" in n: variants.append("bracket_tag")

    scheme = "unknown"
    if re.search(r"\b(?:cover\s*)?1\s+(?:hole|robber|contain)\b", n):
        for tag in ("hole", "robber", "contain"):
            if re.search(r"\b"+tag+r"\b",n): scheme="cover_1_"+tag;break
    elif re.search(r"(?:\b1\s+robber\b)",n): scheme="cover_1_robber"
    elif re.search(r"(?:\b1\s+double\b|\bdouble\s+bracket\b|\bdouble\s+te\b)", n):scheme="cover_1_man_other"
    elif "cover 2 man" in n: scheme="cover_2_man"
    elif "tampa 2" in n: scheme="tampa_2"
    elif re.search(r"\b(?:cover\s*)?2\s+invert\b",n): scheme="cover_2_invert"
    elif "cover 2" in n and "roll" in n: scheme="cover_2_spot"
    elif re.search(r"\bcover 2\b",n): scheme="cover_2_spot"
    elif "nickel 2 trap" in n: scheme="trap_2"
    elif re.search(r"\b(?:cover|cov)\s*3\b",n):
        if "match" in n:scheme="cover_3_match"
        elif "lock" in n:scheme="unknown" # ambiguity: lock doesn't mean vanilla C3
        else: scheme="cover_3_spot"
    elif re.search(r"\bcover 4\b",n):
        if "palms" in n:scheme="cover_4_palms"
        elif "quarters" in n:scheme="cover_4_quarters"
        elif "drop" in n:scheme="cover_4_drop"
        else:scheme="unknown" # Cover 4 Press may not use standard quarters rules
    elif "cover 6" in n:scheme="cover_6"
    elif "cover 9" in n:scheme="cover_9"
    elif n in {"1 hole press", "1 robber press"}: scheme="cover_1_"+("hole" if "hole" in n else "robber")
    # For a name with a leading defensive number and a pressure tag, do NOT infer coverage
    elif re.search(r"\b(?:blitz|fire|pressure|sim|overload|storm)\b", n): variants.append("pressure_or_sim_pressure")
    elif n in ("cover 3",):scheme="cover_3_spot"
    else:
        if re.search(r"\b(?:blitz|fire|pressure|sim|overload|storm)\b",n):variants.append("pressure_or_sim_pressure")

    if scheme=="unknown" and re.search(r"\b(?:blitz|fire|pressure|sim|overload|storm)\b",n):
        if "pressure_or_sim_pressure" not in variants:variants.append("pressure_or_sim_pressure")
    a=SCHEMES[scheme]
    return ClassifiedPlay(scheme, a.family, "name_inferred_not_player_verified", tuple(variants),
                          "madden27.play_name+curated_scheme_principles")


def coverage_read(scheme_id: str, situation: Mapping[str, str]) -> dict[str, Any]:
    """Reason about route releases without fabricating exact defender jobs.

    The result is ADVISORY only; it is not an executable adjustment or evidence
    that a particular player will match a specific opponent on this Madden snap.
    """
    s=SCHEMES.get(scheme_id,SCHEMES["unknown"])
    checks=[dict(x) for x in s.conditional_rules if x["trigger"]==situation.get("release") or
            x["trigger"]==situation.get("offense_formation")]
    # An explicit in-game zone-it check overrides match only for the nominated snap.
    check=situation.get("verified_match_check", "")
    effective=s.family
    if check=="zone_it" and s.family in ("pattern_match", "mixed"):
        effective="spot_zone"
    return {"scheme":s.scheme,"base_family":s.family,"effective_family":effective,
            "triggered_rules":checks,"check":check or "unknown", "check_description":MATCH_CHECKS.get(check),
            "specific_defender_assignments":"unknown_without_play_art_and_selected_check"}


def public_catalog_record(side: str, book: str, formation: str, name: str) -> dict[str, Any]:
    """One knowledge-level catalog row; NEVER certifies a player responsibility."""
    c=classify_defensive_play(name) if side=="defense" else None
    return {"side":side,"book":book,"formation":formation,"play":name,
            "scheme":None if c is None else c.as_dict(),"player_duties_verified":0,
            "source_kind":"installed_user_catalog", "individual_duties_source":"unknown"}