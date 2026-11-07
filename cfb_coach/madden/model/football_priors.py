"""Structured, confidence-aware football priors: how offensive concepts match up against defensive profiles (and the inverse).

This EXTENDS ``football_knowledge`` (concept ids), ``coverage_intelligence`` / ``coverage_attack_knowledge.json`` (scheme conditional reads)
and ``offense_taxonomy`` (route-derived concepts); it does not duplicate them. It is a prior over a LEARNED model, not a play-calling table:

* every entry is a signed LOGIT LIFT in offensive success for one (defensive profile, offensive concept) pair, with a confidence, mechanism,
  counters, failure conditions and evidence ids; magnitudes are deliberately small hypotheses, never rules;
* entries are additive deltas on a profile hierarchy (e.g. Cover 3 spot -> Cover 3 match -> Cloud), so a variant refines its family;
* the model consumes a numeric ``edge`` (expected lift under the opponent's coverage MIXTURE), which is added to the learned score and then updated
  by gameplay (see ``adaptive_priors``). With no opponent evidence the mixture is the prior; with no concept evidence the edge is 0;
* concepts found by route geometry count fully; concepts guessed from a play NAME count ``NAME_TRUST`` (0.35) because a name is not a diagram;
* nothing here fires a play. The inverse (defense) reads the same table from the other side and is shadow-only by default.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

VERSION = "football_priors.v1"
NAME_TRUST = 0.35
EDGE_CAP = 0.60            # logit; summed lifts never exceed this, so stacked hypotheses cannot dominate learned evidence
CONCEPT_ALIASES = {"flood": "flood_sail", "four_verticals": "four_verts"}

EA = "ea_madden27_gameplay"          # https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay
OPEN = "openplay_passing_concepts"   # https://openplayfootball.com/learn/passing-concepts
USAF = "usa_football_match_front"    # https://blogs.usafootball.com/blog/7313/setting-the-front-in-match-coverage-man-free-with-zone-principles
MUTG = "mut_gg_cover_1_explained"    # https://www.mut.gg/news/ask-huddle-42-cover-1-explained-in-madden/ (one summary; unverified for Madden 27)
PRINCIPLE = "football_principle"

SOURCES = {EA: "EA Madden 27 gameplay deep dive (general mechanics: match checks, Smart Zones)", OPEN: "general passing-concept families",
           USAF: "general coaching discussion of match coverage", MUTG: "single third-party summary of Cover 1 variants (not first-party)",
           PRINCIPLE: "standard football reasoning; Madden's exact implementation unverified"}


@dataclass(frozen=True)
class Prior:
    id: str
    profile: str            # key into PROFILES
    concept: str            # football_knowledge concept id
    lift: float             # logit lift of offensive success (+ favours the offence)
    confidence: float       # 0..1, becomes the strength of the prior in adaptive updating
    mechanism: str
    counters: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = (PRINCIPLE,)
    needs_assignments: bool = False     # True: only meaningful with route-level structure, not a name


# ---------------------------------------------------------------------------------------------------- defensive profiles (predicates)
# A coverage descriptor is {"shell": "0".."4"|"split"|"unk", "type": "man"|"spot_zone"|"match"|"unk", "variants": set, "press": str}
PROFILES: dict[str, dict[str, Any]] = {
    "cover_0": {"shell": {"0"}},
    "man_one_high": {"shell": {"1"}, "type": {"man", "unk"}},
    "cover_1_robber": {"shell": {"1"}, "variants_any": {"robber"}},
    "cover_1_hole": {"shell": {"1"}, "variants_any": {"hole"}},
    "cover_2_zone": {"shell": {"2"}, "type": {"spot_zone", "unk", "match"}},
    "cover_2_man": {"shell": {"2"}, "type": {"man"}},
    "cover_2_trap": {"shell": {"2"}, "variants_any": {"trap"}},
    "cover_3_spot": {"shell": {"3"}, "type": {"spot_zone", "unk"}},
    "cover_3_match": {"shell": {"3"}, "type": {"match"}},
    "cover_3_cloud": {"shell": {"3"}, "variants_any": {"cloud"}},
    "cover_3_buzz": {"shell": {"3"}, "variants_any": {"buzz"}},
    "cover_4": {"shell": {"4"}},
    "quarters": {"shell": {"4"}, "variants_any": {"quarters"}},
    "palms": {"shell": {"4"}, "variants_any": {"palms"}},
    "cover_6": {"shell": {"split"}},
    "pressure": {"press_any": True},
    "pressure_zone": {"press": {"zone_blitz"}},
    "pressure_man": {"press": {"man_blitz"}},
}


def profile_matches(desc: Mapping[str, Any], name: str) -> bool:
    p = PROFILES[name]
    if "shell" in p and desc.get("shell") not in p["shell"]:
        return False
    if "type" in p and desc.get("type") not in p["type"]:
        return False
    if "variants_any" in p and not (set(desc.get("variants") or ()) & p["variants_any"]):
        return False
    if p.get("press_any") and desc.get("press") in (None, "", "standard"):
        return False
    if "press" in p and desc.get("press") not in p["press"]:
        return False
    return True


def _P(pid, profile, concept, lift, conf, mech, counters=(), fail=(), ev=(PRINCIPLE,), assign=False):
    return Prior(pid, profile, concept, lift, conf, mech, tuple(counters), tuple(fail), tuple(ev), assign)


PRIORS: tuple[Prior, ...] = (
    # ---- Cover 1 / man with one high safety
    _P("c1.mesh", "man_one_high", "mesh", 0.18, 0.55, "Two shallow crossers force trailing man defenders through traffic; the underneath rub delays them.",
       ("robber/hole defender sits on crossers", "switch/banjo assignments", "press disrupts the release"), ("pressure before the crossers clear",), (OPEN, MUTG, PRINCIPLE), True),
    _P("c1.slant", "man_one_high", "slant_flat", 0.10, 0.5, "Quick inside breaks beat a trailing defender whose leverage is outside.",
       ("inside leverage", "press technique", "free safety undercuts"), ("late throw",), (PRINCIPLE,), True),
    _P("c1.rub", "man_one_high", "rub", 0.14, 0.5, "Picks and rubs create separation from man assignments.", ("switching defenders", "bracketing"), ("offensive pass interference risk",), (PRINCIPLE,), True),
    _P("c1.deep_cross", "man_one_high", "deep_cross", 0.08, 0.4, "Deep crossers run away from outside-leveraged defenders; the single safety must choose.", ("robber drives inside",), (), (PRINCIPLE,), True),
    _P("c1.drive", "man_one_high", "drive", 0.06, 0.4, "Shallow crosser with a dig behind forces defenders to carry through traffic.", ("hole/robber defender",), (), (PRINCIPLE,), True),
    _P("c1.flood", "man_one_high", "flood_sail", -0.05, 0.4, "Zone-stretching flood overloads zones, but man defenders simply follow their receivers.", (), (), (PRINCIPLE,)),
    _P("c1.smash", "man_one_high", "smash", -0.05, 0.4, "Hitch/corner is a zone-spacing concept; against man the corner route is covered one-on-one.", (), (), (PRINCIPLE,)),
    _P("c1.four_verts", "man_one_high", "four_verts", 0.02, 0.3, "Verticals are a speed matchup against man; the lone deep safety can help one side.", ("single-high safety help",), ("pressure arriving first",), (PRINCIPLE,), True),
    _P("c1.screen", "man_one_high", "screen", 0.03, 0.3, "Man defenders trail their receivers, creating blockers' lanes only if the rush is controlled.", (), (), (PRINCIPLE,)),
    _P("c1.run_inside", "man_one_high", "inside_zone", -0.03, 0.3, "Extra defender can be added to the box when no one is needed deep.", (), (), (PRINCIPLE,)),
    _P("c1r.mesh", "cover_1_robber", "mesh", -0.12, 0.45, "A robber sitting in the middle gets under shallow crossers, removing most of the rub advantage.",
       ("crossers that pull the robber away",), (), (MUTG, PRINCIPLE), True),
    _P("c1r.deep_cross", "cover_1_robber", "deep_cross", -0.10, 0.4, "The robber's inside drive meets deep crossers.", (), (), (MUTG, PRINCIPLE), True),
    _P("c1r.flood", "cover_1_robber", "flood_sail", 0.10, 0.35, "Outside-stretching routes pull the robber's leverage away from the middle.", (), (), (EA, PRINCIPLE)),
    _P("c1h.drive", "cover_1_hole", "drive", -0.08, 0.35, "A hook-area defender sits underneath the shallow-dig combination.", (), (), (MUTG, PRINCIPLE), True),
    _P("c1h.mesh", "cover_1_hole", "mesh", -0.06, 0.35, "The hole defender takes away part of the crossing space.", (), (), (MUTG, PRINCIPLE), True),
    # ---- Cover 0 / all-out man
    _P("c0.slant", "cover_0", "slant_flat", 0.18, 0.55, "No deep help: a won quick inside break is a big gain, and quick throws beat the rush.", ("rush arrives first",), ("late protection calls",), (PRINCIPLE,), True),
    _P("c0.mesh", "cover_0", "mesh", 0.12, 0.5, "Crossers vs all man with no safety to undercut.", ("blitz pickup",), (), (PRINCIPLE,), True),
    _P("c0.screen", "cover_0", "screen", 0.12, 0.45, "Rushers vacate the underneath; quick screens attack the vacated space.", ("read by an unblocked rusher",), (), (PRINCIPLE,)),
    _P("c0.four_verts", "cover_0", "four_verts", 0.08, 0.4, "No deep help turns every vertical into a one-on-one, if the protection holds.", ("pressure beats the throw",), ("slow-developing protection",), (PRINCIPLE,), True),
    _P("c0.flood", "cover_0", "flood_sail", -0.05, 0.35, "Multi-level concepts take time against an all-out rush.", (), (), (PRINCIPLE,)),
    _P("c0.play_action", "cover_0", "play_action", -0.08, 0.4, "Play-action fakes cost time the rush does not give.", (), (), (PRINCIPLE,)),
    _P("c0.run_inside", "cover_0", "inside_zone", -0.08, 0.35, "A defender is added to the box.", (), (), (PRINCIPLE,)),
    # ---- Cover 2 zone
    _P("c2.smash", "cover_2_zone", "smash", 0.18, 0.6, "Hitch under the corner route puts the flat defender in conflict: sit on the hitch and the corner is open to the deep-half's edge.",
       ("trap or cloud coverage on the flat", "deep half sinking to the corner"), ("quarterback stares down the hitch",), (OPEN, PRINCIPLE), True),
    _P("c2.four_verts", "cover_2_zone", "four_verts", 0.05, 0.4, "Two safeties split the field, leaving the seam between them.", ("middle linebacker running with the seam (Tampa)",), (), (PRINCIPLE,), True),
    _P("c2.dagger", "cover_2_zone", "dagger", 0.06, 0.4, "A dig behind the hooks, between the safeties' depth.", ("hook defenders dropping deeper",), (), (PRINCIPLE,), True),
    _P("c2.levels", "cover_2_zone", "levels", 0.05, 0.4, "In-breakers at two depths stress the hook defenders.", (), (), (PRINCIPLE,), True),
    _P("c2.wheel", "cover_2_zone", "wheel", 0.05, 0.35, "A back wheeling up the sideline outruns the flat defender.", ("deep half rotating down",), (), (PRINCIPLE,), True),
    _P("c2.run_inside", "cover_2_zone", "inside_zone", 0.07, 0.4, "Two deep safeties mean a lighter box.", (), (), (PRINCIPLE,)),
    _P("c2.flood", "cover_2_zone", "flood_sail", 0.03, 0.3, "Three levels on one side stress the corner and deep half.", (), (), (PRINCIPLE,), True),
    _P("c2m.mesh", "cover_2_man", "mesh", 0.12, 0.45, "Man underneath with two deep: crossers get rubbed.", ("safety over the top reading crossers",), (), (PRINCIPLE,), True),
    _P("c2m.smash", "cover_2_man", "smash", -0.12, 0.5, "Cover 2 Man is NOT spot zone: the corner follows his man, so the hitch/corner conflict does not exist.", (), (), (PRINCIPLE,)),
    _P("c2m.slant", "cover_2_man", "slant_flat", 0.06, 0.4, "Quick inside breaks beat trailing man defenders.", ("press",), (), (PRINCIPLE,), True),
    _P("c2t.smash", "cover_2_trap", "smash", -0.10, 0.4, "Trap coverage baits the flat throw and sits on it.", (), (), (EA, PRINCIPLE)),
    _P("c2t.wheel", "cover_2_trap", "wheel", 0.04, 0.3, "Trapped flat defenders can be outrun by a wheel.", (), (), (PRINCIPLE,)),
    # ---- Cover 3 (spot / match / variants)
    _P("c3.four_verts", "cover_3_spot", "four_verts", 0.09, 0.5, "Three deep defenders must cover four vertical lanes: a seam is exposed unless the safety or curl defender carries it.", ("seam-carry rules", "safety help in the middle"), ("pressure",), (OPEN, PRINCIPLE), True),
    _P("c3.flood", "cover_3_spot", "flood_sail", 0.10, 0.55, "Three levels to one side overload the flat/curl and deep third.", ("corner or safety dropping to the flat",), (), (OPEN, PRINCIPLE), True),
    _P("c3.stick", "cover_3_spot", "stick", 0.05, 0.4, "Stick (hitch + flat/out) attacks the curl-flat defender.", (), (), (PRINCIPLE,), True),
    _P("c3.dagger", "cover_3_spot", "dagger", 0.06, 0.4, "A dig between the hooks and the deep thirds.", (), (), (PRINCIPLE,), True),
    _P("c3.curl_flat", "cover_3_spot", "curl_flat", -0.03, 0.35, "Curl-flat defenders are positioned to take curl/flat combinations.", (), (), (PRINCIPLE,), True),
    _P("c3.slant_flat", "cover_3_spot", "slant_flat", 0.04, 0.35, "Quick slants find soft spots between zone defenders.", (), (), (PRINCIPLE,), True),
    _P("c3.screen", "cover_3_spot", "screen", 0.05, 0.35, "Zone defenders drop away from the line.", (), (), (PRINCIPLE,)),
    _P("c3.wheel", "cover_3_spot", "wheel", -0.03, 0.3, "The sideline third is covered by a corner who sees the wheel develop.", (), (), (PRINCIPLE,)),
    _P("c3m.four_verts", "cover_3_match", "four_verts", -0.06, 0.45, "Match rules pass off and carry the seams, so verticals are met by matched defenders.", (), (), (EA, USAF), True),
    _P("c3m.flood", "cover_3_match", "flood_sail", -0.06, 0.45, "Match rules carry the flat and corner routes instead of leaving a spot-zone hole.", (), (), (EA, USAF), True),
    _P("c3m.mesh", "cover_3_match", "mesh", 0.03, 0.3, "Crossers may be passed off; match is not blind zone.", (), (), (EA, PRINCIPLE), True),
    _P("c3c.flood", "cover_3_cloud", "flood_sail", -0.03, 0.25, "Cloud typically puts a corner in the flat (hypothesis; source-only summary).", (), (), (PRINCIPLE,)),
    _P("c3b.stick", "cover_3_buzz", "stick", -0.03, 0.25, "Buzz typically puts a linebacker/safety in the flat or hook (hypothesis).", (), (), (PRINCIPLE,)),
    # ---- Cover 4 family (quarters, palms), split-field Cover 6
    _P("c4.four_verts", "cover_4", "four_verts", -0.07, 0.5, "Four deep defenders match the vertical routes.", (), (), (EA, PRINCIPLE), True),
    _P("c4.mesh", "cover_4", "mesh", 0.04, 0.35, "Underneath coverage is thinner when defenders are playing deep.", ("safeties reading the #2 receiver",), (), (PRINCIPLE,), True),
    _P("c4.stick", "cover_4", "stick", 0.04, 0.35, "Underneath hitch/flat windows are open when defenders play deep.", (), (), (PRINCIPLE,), True),
    _P("c4.levels", "cover_4", "levels", 0.05, 0.35, "Intermediate in-breakers sit between the deep and underneath defenders.", (), (), (PRINCIPLE,), True),
    _P("c4.run_inside", "cover_4", "inside_zone", 0.04, 0.3, "Safeties are deep, but quarter safeties are also quick run fitters: small edge only.", (), (), (PRINCIPLE,)),
    _P("c4q.dagger", "quarters", "dagger", 0.03, 0.3, "Safeties key the #2 vertical; a dig behind them can be open.", (), (), (EA, PRINCIPLE), True),
    _P("c4p.smash", "palms", "smash", -0.06, 0.35, "Palms uses a 2-read corner rule that sits on the short route (EA: Palms is a match call).", (), (), (EA, PRINCIPLE), True),
    _P("c6.smash", "cover_6", "smash", 0.06, 0.3, "The Cover 2 side of a split-field call keeps the flat/deep-half conflict (side-specific; strength side unknown here, so this is halved).", ("quarters side removes it",), (), (EA, PRINCIPLE), True),
    _P("c6.four_verts", "cover_6", "four_verts", -0.03, 0.3, "The quarters side matches verticals (side-specific; halved).", (), (), (EA, PRINCIPLE), True),
    # ---- Pressure (any blitz): the rush changes which concepts have time
    _P("pr.screen", "pressure", "screen", 0.12, 0.5, "Rushers leave the underneath space; a screen turns the rush into blockers' leverage.", ("unblocked rusher reads it", "no help behind the blitz"), (), (PRINCIPLE,)),
    _P("pr.slant", "pressure", "slant_flat", 0.09, 0.5, "Quick throws beat the rush and attack the vacated zone.", ("hot defender dropping into the lane",), (), (PRINCIPLE,), True),
    _P("pr.spot", "pressure", "spot", 0.05, 0.35, "Quick-game spot routes are a standard hot answer.", (), (), (PRINCIPLE,), True),
    _P("pr.draw", "pressure", "draw", 0.08, 0.4, "Rushers upfield create run lanes behind them.", ("spy or contain",), (), (PRINCIPLE,)),
    _P("pr.play_action", "pressure", "play_action", -0.07, 0.4, "The fake takes time the protection does not have.", (), (), (PRINCIPLE,)),
    _P("pr.four_verts", "pressure", "four_verts", -0.05, 0.4, "Deep routes need protection time.", ("zero coverage behind the rush makes it a one-on-one gamble",), (), (PRINCIPLE,), True),
    _P("pr.deep_cross", "pressure", "deep_cross", -0.05, 0.4, "Slow-developing: the rush arrives first.", (), (), (PRINCIPLE,), True),
    _P("pr.flood", "pressure", "flood_sail", -0.04, 0.35, "Three-level concepts take time.", (), (), (PRINCIPLE,)),
    _P("pz.screen", "pressure_zone", "screen", 0.04, 0.35, "A zone blitz drops some rushers into the underneath, so screens are less certain.", (), (), (EA, PRINCIPLE)),
    _P("pz.slant", "pressure_zone", "slant_flat", -0.03, 0.3, "Zone dogs drop into the quick-throw lanes (zone pressure style: EA).", (), (), (EA,), True),
    _P("pm.mesh", "pressure_man", "mesh", 0.05, 0.35, "Man behind the blitz: crossers get rubs and a free hot target.", (), (), (PRINCIPLE,), True),
    _P("pm.slant", "pressure_man", "slant_flat", 0.04, 0.35, "Man behind the blitz has no help for a won quick route.", (), (), (PRINCIPLE,), True),
)

# ---------------------------------------------------------------------------------------------------- route-level priors (for QB read order)
ROUTE_PRIORS: dict[str, dict[str, float]] = {
    "man": {"slant": .10, "drag": .12, "shallow_cross": .12, "cross": .08, "deep_cross": .08, "corner": .02, "post": .04, "streak": .02, "fade": .02,
            "wheel": .04, "swing": .04, "screen": .04, "out": -.02, "curl": -.02, "hitch": -.04, "flat": 0.0},
    "cover_0": {"slant": .12, "drag": .08, "shallow_cross": .08, "post": .10, "streak": .08, "fade": .06, "screen": .10, "swing": .06, "hitch": .02},
    "cover_2_zone": {"corner": .10, "streak": .06, "seam": .06, "hitch": .03, "curl": .03, "flat": .04, "wheel": .04, "post": -.04, "slant": 0.0, "drag": .02, "in": .04},
    "cover_3_spot": {"seam": .06, "streak": .06, "out": .03, "flat": .04, "post": .02, "drag": .04, "in": .04, "curl": -.02, "hitch": -.02, "corner": -.01, "wheel": -.02},
    "cover_3_match": {"seam": -.02, "streak": -.02, "out": 0.0, "flat": .0, "drag": .04, "in": .04, "post": .02, "slant": .03},
    "cover_4": {"streak": -.06, "seam": -.06, "curl": .03, "drag": .04, "flat": .04, "in": .04, "out": .02, "corner": -.02, "hitch": .03},
    "pressure": {"slant": .10, "hitch": .06, "flat": .08, "swing": .12, "screen": .12, "spot": .06, "streak": -.06, "post": -.06, "corner": -.06, "deep_cross": -.06},
}
ROUTE_PROFILE_ORDER = ("pressure", "cover_0", "cover_2_man", "man_one_high", "cover_2_zone", "cover_3_match", "cover_3_spot", "cover_4", "cover_6")


def route_group(desc: Mapping[str, Any]) -> str | None:
    if profile_matches(desc, "pressure"):
        return "pressure"
    if profile_matches(desc, "cover_0"):
        return "cover_0"
    if profile_matches(desc, "cover_2_man") or profile_matches(desc, "man_one_high"):
        return "man"
    if profile_matches(desc, "cover_2_zone"):
        return "cover_2_zone"
    if profile_matches(desc, "cover_3_match"):
        return "cover_3_match"
    if profile_matches(desc, "cover_3_spot"):
        return "cover_3_spot"
    if profile_matches(desc, "cover_4"):
        return "cover_4"
    return None


# ---------------------------------------------------------------------------------------------------- coverage descriptors
_LEGACY = {  # legacy mix keys (historical_defense_baseline buckets) -> descriptors
    "cover_0": ("0", "man", ()), "cover_1_unspecified": ("1", "man", ()), "cover_1_robber": ("1", "man", ("robber",)),
    "cover_1_hole": ("1", "man", ("hole",)), "cover_1_man_other": ("1", "man", ()), "cover_1_pressure": ("1", "man", ("pressure",)), "cover_1_contain": ("1", "man", ("contain_spy",)),
    "cover_2_man": ("2", "man", ()), "cover_2_spot": ("2", "spot_zone", ()), "cover_2_unspecified": ("2", "spot_zone", ()),
    "cover_2_invert": ("2", "spot_zone", ("invert",)), "tampa_2": ("2", "spot_zone", ("tampa",)), "trap_2": ("2", "spot_zone", ("trap",)),
    "cover_3_unspecified": ("3", "spot_zone", ()), "cover_3_match": ("3", "match", ()), "cover_3_sky": ("3", "spot_zone", ("sky",)),
    "cover_3_spot": ("3", "spot_zone", ()), "cover_4_unspecified": ("4", "unk", ()), "cover_4_quarters": ("4", "match", ("quarters",)),
    "cover_4_palms": ("4", "match", ("palms",)), "cover_4_drop": ("4", "spot_zone", ("drop",)), "cover_6": ("split", "match", ()),
}
_VAR_FROM_AXIS = {"dk_var_robber": "robber", "dk_var_hole": "hole", "dk_var_trap": "trap", "dk_var_cloud": "cloud", "dk_var_buzz": "buzz",
                  "dk_var_sky": "sky", "dk_var_quarters": "quarters", "dk_var_palms": "palms", "dk_var_invert": "invert", "dk_var_contain_spy": "contain_spy"}
_PRESS_AXES = ("zone_blitz", "man_blitz", "sim", "stunt", "unspecified")


def descriptor(key: str) -> dict[str, Any] | None:
    """Coverage descriptor for a mix key (legacy bucket or ``dk:`` key). ``None`` = no information (never a guess)."""
    if key in _LEGACY:
        sh, ty, va = _LEGACY[key]
        return {"shell": sh, "type": ty, "variants": set(va), "press": "man_blitz" if "pressure" in va else "standard"}   # cover_9 / unknown: no entry => abstain
    if key.startswith("dk:"):
        from cfb_coach.madden.model import defense_semantics as ds
        try:
            ax = ds.key_axes(key)
        except Exception:                                   # noqa: BLE001 - unknown key
            return None
        shell = next((n.removeprefix("dk_shell_") for n, v in ax.items() if n.startswith("dk_shell_") and v), "unk")
        typ = next((n.removeprefix("dk_type_") for n, v in ax.items() if n.startswith("dk_type_") and v), "unk")
        press = next((p for p in _PRESS_AXES if ax.get(f"dk_press_{p}")), "standard")
        return {"shell": shell, "type": typ, "variants": {v for n, v in _VAR_FROM_AXIS.items() if ax.get(n)}, "press": press}
    return None


# ---------------------------------------------------------------------------------------------------- concepts of a play
def play_concepts(structure: Any) -> dict[str, float]:
    """concept id -> trust. Route-derived concepts count 1.0; a name-only hint counts NAME_TRUST (a name is not a diagram)."""
    out: dict[str, float] = {}
    try:
        for c in structure.concepts():
            out[CONCEPT_ALIASES.get(c, c)] = 1.0
    except Exception:                                       # noqa: BLE001
        pass
    from cfb_coach.madden.model import football_knowledge as fk
    hint = fk.profile_for_play(getattr(structure, "play", None))
    cid = hint.get("concept_id")
    if cid and cid not in out:
        out[cid] = NAME_TRUST
    for c in (getattr(structure, "hint", None) or {}).get("concepts") or []:
        c = CONCEPT_ALIASES.get(c, c)
        if c in fk.CONCEPTS and c not in out:
            out[c] = NAME_TRUST
    return out


# ---------------------------------------------------------------------------------------------------- the numeric edge
_BY_PROFILE_CONCEPT: dict[str, dict[str, list[Prior]]] = {}
for _p in PRIORS:
    _BY_PROFILE_CONCEPT.setdefault(_p.profile, {}).setdefault(_p.concept, []).append(_p)
_DESC_CACHE: dict[str, dict[str, Any] | None] = {}


def _desc(key: str):
    if key not in _DESC_CACHE:
        _DESC_CACHE[key] = descriptor(key)
    return _DESC_CACHE[key]


def lift_for(desc: Mapping[str, Any], concept: str, *, side_factor: float = 1.0) -> tuple[float, float, list[str]]:
    """(lift, confidence-weighted strength, prior ids) of one concept vs one coverage descriptor: matching profile deltas add."""
    lift, conf_w, ids = 0.0, 0.0, []
    for prof, by_c in _BY_PROFILE_CONCEPT.items():
        if concept in by_c and profile_matches(desc, prof):
            for p in by_c[concept]:
                f = 0.5 if (p.profile == "cover_6") else 1.0         # split-field side unknown: halve
                lift += p.lift * f
                conf_w += p.confidence
                ids.append(p.id)
    return lift, conf_w, ids


def matchup_edge(structure: Any, mix: Mapping[str, float] | None, *, learned: Mapping[tuple[str, str], float] | None = None) -> dict[str, Any]:
    """Expected logit lift of this play against the opponent's coverage MIXTURE (opponent unseen => the mixture is the prior).

    ``learned`` optionally replaces a cell's prior lift with the posterior from ``adaptive_priors`` keyed (profile-group, concept).
    Returns edge (capped), se (uncertainty from prior confidence), and the cells that produced it (the auditable explanation)."""
    concepts = play_concepts(structure)
    total, var, cells = 0.0, 0.0, []
    if not mix or not concepts:
        return {"edge": 0.0, "se": 0.0, "cells": [], "concepts": concepts, "mass": 0.0}
    mass = 0.0
    for key, w in mix.items():
        d = _desc(key)
        if not d or w <= 0:
            continue
        mass += w
        for concept, trust in concepts.items():
            lift, conf_w, ids = lift_for(d, concept)
            if not ids:
                continue
            grp = route_group(d) or "other"
            if learned and (grp, concept) in learned:
                lift = learned[(grp, concept)]
            total += w * trust * lift
            var += (w * trust) ** 2 * max(0.0, 0.12 - 0.1 * min(1.0, conf_w / len(ids))) ** 2 * 4
            cells.append({"coverage": key, "concept": concept, "lift": round(lift, 4), "weight": round(w * trust, 4), "priors": ids})
    cap = EDGE_CAP
    edge = max(-cap, min(cap, total))
    cells.sort(key=lambda c: -abs(c["lift"] * c["weight"]))
    return {"edge": edge, "se": math.sqrt(var), "cells": cells[:6], "concepts": concepts, "mass": round(mass, 4)}


def route_read_scores(family: str, mix: Mapping[str, float] | None) -> float:
    """Coverage-conditioned prior preference of one ROUTE FAMILY (used to order QB reads from the transformed play)."""
    if not mix:
        return 0.0
    s = 0.0
    for key, w in mix.items():
        d = _desc(key)
        g = route_group(d) if d else None
        if g and w > 0:
            s += w * ROUTE_PRIORS.get(g, {}).get(family, 0.0)
    return s


def protection_edge(mix: Mapping[str, float] | None) -> tuple[float, float]:
    """(logit gain, se) of an extra pass-protection adjustment: helps vs pressure mass, costs a little when no pressure is expected."""
    if not mix:
        return 0.0, 0.0
    press = sum(w for k, w in mix.items() if (_desc(k) or {}).get("press") not in (None, "", "standard") or k == "dk:_generic_pressure")
    return 0.10 * press - 0.02 * (1.0 - press), 0.03


# ---------------------------------------------------------------------------------------------------- the defensive inverse (shadow)
def defense_call_edge(call_desc: Mapping[str, Any], offense_concepts: Mapping[str, float]) -> dict[str, Any]:
    """Expected logit lift ALLOWED to the offence by a defensive call versus the offence's concept distribution (negative = good for the defence).
    The same table, read from the other side; shadow-only until the defensive gates pass."""
    tot, cells = 0.0, []
    for concept, w in offense_concepts.items():
        lift, conf_w, ids = lift_for(call_desc, CONCEPT_ALIASES.get(concept, concept))
        if ids:
            tot += w * lift
            cells.append({"concept": concept, "lift_allowed": round(lift, 4), "priors": ids})
    return {"offense_edge_allowed": max(-EDGE_CAP, min(EDGE_CAP, tot)), "cells": cells}


def validate() -> list[str]:
    from cfb_coach.madden.model import football_knowledge as fk
    errs, seen = [], set()
    for p in PRIORS:
        if p.id in seen:
            errs.append(f"duplicate prior id {p.id}")
        seen.add(p.id)
        if p.concept not in fk.CONCEPTS:
            errs.append(f"{p.id}: unknown concept {p.concept}")
        if p.profile not in PROFILES:
            errs.append(f"{p.id}: unknown profile {p.profile}")
        if not (-0.35 <= p.lift <= 0.35):
            errs.append(f"{p.id}: lift {p.lift} outside the hypothesis range")
        if not (0.0 < p.confidence <= 1.0) or not p.mechanism or not p.evidence:
            errs.append(f"{p.id}: missing confidence/mechanism/evidence")
        for e in p.evidence:
            if e not in SOURCES:
                errs.append(f"{p.id}: unknown evidence {e}")
    return errs


def report() -> dict[str, Any]:
    return {"version": VERSION, "priors": len(PRIORS), "profiles": len(PROFILES), "concepts_covered": len({p.concept for p in PRIORS}),
            "needs_assignments": sum(p.needs_assignments for p in PRIORS), "sources": SOURCES, "edge_cap": EDGE_CAP, "name_trust": NAME_TRUST,
            "note": "hypotheses with confidence, not rules; magnitudes are logit lifts of a few percent"}
