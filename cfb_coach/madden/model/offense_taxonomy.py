"""Reusable Madden offensive taxonomy: routes, blocking schemes, concepts, evidence tiers.

Research-only. Vocabulary and structural definitions live here; *which* player
runs *what* on a given installed play lives in ``offense_assignments``. A play
name is never route proof. Depth bands are football-default priors for a route
family, not measured Madden yardage (``DEPTH_BASIS``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

TAXONOMY_VERSION = "offense_taxonomy.v1"

# Strongest first. ``unknown`` is an explicit value, never a default for "verified".
EVIDENCE_TIERS = ("in_game_verified", "diagram_verified", "source_documented",
                  "football_inference", "unknown")
TIER_RANK = {t: i for i, t in enumerate(EVIDENCE_TIERS)}
VERIFIED_TIERS = frozenset({"in_game_verified", "diagram_verified"})
DEPTH_BASIS = "family_default_prior_not_play_specific"

DEPTH_BANDS = ("shallow", "short", "intermediate", "deep", "unspecified")
# Approximate midpoint yards per band (football prior only).
BAND_YARDS = {"shallow": 3.0, "short": 6.0, "intermediate": 12.0, "deep": 20.0, "unspecified": 10.0}


@dataclass(frozen=True)
class RouteFamily:
    id: str
    category: str          # vertical | breaking | underneath | crossing | curl_sit | blocking | other
    default_band: str
    break_dir: str         # vertical | inside | outside | mixed | back | none
    hot_route_option: str | None  # exact Madden hot-route menu name, if one exists
    man_beater: float      # 0..1 football prior
    zone_beater: float
    quick: bool            # releases fast enough to answer pressure
    note: str = ""


def _rf(id_: str, cat: str, band: str, brk: str, hot: str | None, man: float, zone: float,
        quick: bool, note: str = "") -> tuple[str, RouteFamily]:
    return id_, RouteFamily(id_, cat, band, brk, hot, man, zone, quick, note)


ROUTE_FAMILIES: dict[str, RouteFamily] = dict([
    _rf("slant", "breaking", "short", "inside", "Slant", 0.8, 0.4, True),
    _rf("drag", "crossing", "shallow", "inside", "Drag", 0.8, 0.5, True, "shallow crosser"),
    _rf("shallow_cross", "crossing", "shallow", "inside", None, 0.8, 0.5, True),
    _rf("deep_cross", "crossing", "intermediate", "inside", None, 0.6, 0.6, False),
    _rf("cross", "crossing", "intermediate", "inside", None, 0.7, 0.5, False),
    _rf("in", "breaking", "short", "inside", "In", 0.5, 0.7, False,
        "band separates short in / speed in / deep in (dig)"),
    _rf("out", "breaking", "short", "outside", "Out", 0.4, 0.6, False),
    _rf("hitch", "curl_sit", "short", "back", None, 0.2, 0.7, True),
    _rf("curl", "curl_sit", "intermediate", "back", "Curl", 0.3, 0.8, False),
    _rf("comeback", "curl_sit", "deep", "back", "Comeback", 0.3, 0.6, False),
    _rf("streak", "vertical", "deep", "vertical", "Streak", 0.5, 0.4, False),
    _rf("fade", "vertical", "deep", "vertical", "Fade", 0.4, 0.3, False),
    _rf("seam", "vertical", "deep", "vertical", None, 0.5, 0.7, False),
    _rf("post", "breaking", "deep", "inside", "Post", 0.5, 0.7, False),
    _rf("corner", "breaking", "deep", "outside", "Corner", 0.5, 0.7, False),
    _rf("flat", "underneath", "shallow", "outside", "Flat", 0.4, 0.5, True),
    _rf("swing", "underneath", "shallow", "outside", None, 0.4, 0.5, True),
    _rf("wheel", "vertical", "deep", "mixed", None, 0.7, 0.5, False, "flat-to-vertical; stresses flat defender"),
    _rf("angle", "underneath", "short", "mixed", None, 0.6, 0.6, True, "RB angle / texas family"),
    _rf("texas", "underneath", "short", "mixed", None, 0.6, 0.6, True),
    _rf("pivot", "underneath", "short", "mixed", None, 0.5, 0.5, True),
    _rf("zig", "breaking", "intermediate", "mixed", None, 0.6, 0.5, False, "double move"),
    _rf("sluggo", "vertical", "deep", "mixed", None, 0.6, 0.4, False, "slant-and-go double move"),
    _rf("stick", "curl_sit", "short", "back", None, 0.3, 0.6, True),
    _rf("spot", "curl_sit", "short", "back", None, 0.3, 0.7, True),
    _rf("choice", "breaking", "short", "mixed", None, 0.6, 0.5, True, "option/choice route"),
    _rf("option", "breaking", "short", "mixed", None, 0.6, 0.5, True),
    _rf("screen", "other", "shallow", "none", None, 0.3, 0.3, True),
    _rf("smoke", "other", "shallow", "none", None, 0.3, 0.5, True),
    _rf("whip", "breaking", "short", "mixed", None, 0.6, 0.5, True),
    _rf("block", "blocking", "unspecified", "none", "Block", 0.0, 0.0, True,
        "receiver/back stays in to block (hot route 'Block')"),
    _rf("pass_block", "blocking", "unspecified", "none", None, 0.0, 0.0, True),
    _rf("block_and_release_hook", "blocking", "short", "back", None, 0.2, 0.5, True,
        "published 'Block and Release Hook'"),
    _rf("other", "other", "unspecified", "mixed", None, 0.4, 0.4, False,
        "published route name not mapped to a family; kept as published_route"),
])

# Published Madden route names -> (family, band). Only names seen in sources; extend via annotation.
PUBLISHED_ROUTE_MAP: dict[str, tuple[str, str]] = {
    "deep in": ("in", "deep"), "short in": ("in", "short"), "speed in": ("in", "short"),
    "speed out": ("out", "short"), "deep out": ("out", "deep"), "deep corner": ("corner", "deep"),
    "deep post": ("post", "deep"), "short post": ("post", "short"), "return": ("other", "unspecified"),
    "smash": ("other", "unspecified"), "block and release hook": ("block_and_release_hook", "short"),
}

# What Madden's individual hot-route menu offers (executable_actions.json, grade needs_user_test).
HOT_ROUTE_OPTIONS: tuple[str, ...] = tuple(sorted(
    f.hot_route_option for f in ROUTE_FAMILIES.values() if f.hot_route_option))
HOT_ROUTE_TO_FAMILY: dict[str, str] = {f.hot_route_option: f.id for f in ROUTE_FAMILIES.values()
                                       if f.hot_route_option}

# Stock-play routes that Madden's hot-route menu does NOT offer (so they can be lost but never re-added).
NOT_HOT_ROUTABLE = frozenset(f for f, v in ROUTE_FAMILIES.items() if not v.hot_route_option)


@dataclass(frozen=True)
class RouteSpec:
    """A normalized route: family + direction + depth. Equality ignores the source label."""
    family: str
    side: str = "unknown"          # left | right | middle | unknown (alignment side as published)
    band: str = "unspecified"
    published: str = ""

    @property
    def meta(self) -> RouteFamily:
        return ROUTE_FAMILIES[self.family]

    @property
    def yards(self) -> float:
        return BAND_YARDS.get(self.band, BAND_YARDS["unspecified"])

    @property
    def label(self) -> str:
        return f"{self.family}({self.band},{self.side})"


def route_from_job(task: str, params: Mapping[str, Any] | None) -> RouteSpec:
    """Normalize a sourced job (task + params) to a RouteSpec without guessing depth."""
    params = params or {}
    pub = str(params.get("published_route") or "")
    mapped = PUBLISHED_ROUTE_MAP.get(pub.lower()) if pub else None
    depth = params.get("depth")
    if task in ROUTE_FAMILIES and task != "other":
        fam = task
        if depth in DEPTH_BANDS:
            band = str(depth)
        elif mapped and mapped[0] == fam:
            band = mapped[1]
        else:
            band = ROUTE_FAMILIES[fam].default_band
    elif mapped:
        fam, band = mapped
    else:
        fam, band = "other", "unspecified"
    return RouteSpec(fam, str(params.get("side") or "unknown"), band, pub)


# ----------------------------------------------------------------- blocking

RUN_SCHEMES: dict[str, str] = {
    "inside_zone": "zone: OL step playside and combo to LBs; RB reads the first vertical crease",
    "mid_zone": "zone: tighter than outside zone, cutback allowed",
    "outside_zone": "zone: stretch the edge; RB presses outside then cuts",
    "split_zone": "zone with a backside blocker kicking/cutting across the formation",
    "power": "gap: backside guard (and often a lead blocker) pulls to lead through the hole",
    "counter": "gap: misdirection step then backside guard + another blocker pull",
    "duo": "gap/double-team: two double teams to the second level",
    "dive": "quick-hitting inside handoff",
    "draw": "pass-set OL, delayed handoff",
    "off_tackle": "gap run at the edge of the tackle box",
    "sweep": "pulling blockers lead a perimeter run",
    "qb_draw": "delayed QB run from pass-set OL",
    "read_option": "QB reads an unblocked defender: hand off or keep",
    "lead_read_option": "read option with a lead blocker",
    "toss": "pitch to the back running wide",
}
PASS_PROTECTIONS = ("half_slide", "full_slide", "max_protect", "man_protect", "play_action_protect",
                    "boot_protect", "screen_protect", "quick_set", "unknown")
OL_SLOTS = ("LT", "LG", "C", "RG", "RT")
FIXED_SLOTS = ("QB",) + OL_SLOTS
OL_TASKS = ("pass_set", "slide_left", "slide_right", "zone_step_left", "zone_step_right", "down_block",
            "reach_block", "pull_left", "pull_right", "double_team", "combo_to_second_level",
            "screen_release", "draw_set", "unknown")

# ----------------------------------------------------------------- concepts

CONCEPTS: dict[str, dict[str, Any]] = {
    "mesh": {"kind": "pass", "stress": ["man", "zone_hook"],
             "desc": "two shallow crossers from opposite sides at near-equal depth; rub/pick vs man"},
    "flood": {"kind": "pass", "stress": ["zone_flat", "cover_3", "cover_2"],
              "desc": "three levels to one side (flat, intermediate, deep outside)"},
    "levels": {"kind": "pass", "stress": ["zone_hook", "man"],
               "desc": "two in-breaking routes at different depths on the same side"},
    "drive": {"kind": "pass", "stress": ["zone_hook", "man"],
              "desc": "shallow crosser plus an in-breaker behind it"},
    "smash": {"kind": "pass", "stress": ["cover_2", "cover_4"],
              "desc": "hitch/curl underneath with a corner over the top on one side"},
    "dagger": {"kind": "pass", "stress": ["cover_3", "cover_1"],
               "desc": "vertical clears a safety; deep in/dig behind it"},
    "mills": {"kind": "pass", "stress": ["cover_3", "cover_1", "cover_2"],
              "desc": "deep post with a deep in/dig; post and dig cross at depth"},
    "stick": {"kind": "pass", "stress": ["zone_hook", "man_pick"],
              "desc": "short stick/hitch with flat and out; quick horizontal read"},
    "spacing": {"kind": "pass", "stress": ["zone"],
                "desc": "three-plus short routes at distinct horizontal landmarks"},
    "four_verticals": {"kind": "pass", "stress": ["cover_3", "cover_1", "cover_4"],
                       "desc": "four vertical-stem routes stretching the deep zone"},
    "post_dig": {"kind": "pass", "stress": ["cover_3", "cover_1"],
                 "desc": "post combined with a dig/deep in; different from post + underneath flat"},
    "hi_lo": {"kind": "pass", "stress": ["zone_hook", "zone_flat"],
              "desc": "short and deeper route on one defender's zone: high-low read"},
    "slant_flat": {"kind": "pass", "stress": ["man", "zone_flat"],
                   "desc": "slant with flat; quick answer to pressure"},
    "screen": {"kind": "screen", "stress": ["pressure", "man"], "desc": "blocked screen"},
    "play_action": {"kind": "play_action", "stress": ["run_fit_defenders"],
                    "desc": "run fake before dropback/boot"},
    "run_zone": {"kind": "run", "stress": ["base_front"], "desc": "zone blocking scheme"},
    "run_gap": {"kind": "run", "stress": ["light_box"], "desc": "gap/power/counter scheme"},
    "rpo": {"kind": "rpo", "stress": ["conflict_defender"], "desc": "run plus pass option off a defender read"},
    "option": {"kind": "option", "stress": ["unblocked_defender"], "desc": "QB read keep/give"},
}

# Madden-specific name tokens -> concept hint. Name-level ONLY (football_inference).
NAME_CONCEPT_TOKENS: tuple[tuple[str, str], ...] = (
    (r"\bmesh\b", "mesh"), (r"\bflood\b", "flood"), (r"\bsmash\b", "smash"), (r"\bdagger\b", "dagger"),
    (r"\bmills\b", "mills"), (r"\bspacing\b", "spacing"), (r"\bstick\b", "stick"),
    (r"\bverticals?\b|\bverts\b|\bgos\b", "four_verticals"), (r"\bhi lo\b", "hi_lo"),
    (r"\bslants?\b", "slant_flat"), (r"\bdrive\b", "drive"), (r"\bpost\b.*\bdig\b|\bdig\b.*\bpost\b", "post_dig"),
)

# Alignment slots by installed formation, as *documented by the sourced annotations only*.
SKILL_SLOTS_DOCUMENTED = {
    "Gun Trips X Nasty": ("HB1", "TE1", "WR1", "WR2", "SLWR"),
    "Gun Doubles Clamp Stack": ("HB1", "TE1", "WR1", "WR2", "SLWR"),
    "Pistol Trips": ("HB1", "TE1", "WR1", "WR2", "WR3"),
}


def as_dict() -> dict[str, Any]:
    """Serializable taxonomy for docs/exports."""
    return {
        "version": TAXONOMY_VERSION, "evidence_tiers": list(EVIDENCE_TIERS), "depth_basis": DEPTH_BASIS,
        "route_families": {k: {"category": v.category, "default_band": v.default_band, "break": v.break_dir,
                               "hot_route": v.hot_route_option, "man_beater": v.man_beater,
                               "zone_beater": v.zone_beater, "quick": v.quick, "note": v.note}
                           for k, v in ROUTE_FAMILIES.items()},
        "published_route_map": {k: list(v) for k, v in PUBLISHED_ROUTE_MAP.items()},
        "hot_route_options": list(HOT_ROUTE_OPTIONS),
        "run_schemes": RUN_SCHEMES, "pass_protections": list(PASS_PROTECTIONS),
        "ol_tasks": list(OL_TASKS), "concepts": CONCEPTS,
    }


# ----------------------------------------------------------------- structural concept detection


def _by(routes: Sequence[RouteSpec], *fams: str) -> list[RouteSpec]:
    return [r for r in routes if r.family in fams]


def detect_concepts(routes: Sequence[RouteSpec]) -> dict[str, dict[str, Any]]:
    """Concepts implied by the *known* routes' interaction (not by the play name).

    Each hit lists the routes involved. Fewer than the needed known routes yields
    nothing: absent routes are unknown, not absent concepts.
    """
    found: dict[str, dict[str, Any]] = {}
    r = [x for x in routes if x.meta.category != "blocking"]

    def hit(name: str, parts: Iterable[RouteSpec], rule: str) -> None:
        found[name] = {"routes": [p.label for p in parts], "rule": rule}

    cross = _by(r, "drag", "shallow_cross", "cross")
    sides = {c.side for c in cross if c.side in ("left", "right")}
    # Sides are alignment sides; opposite alignment sides with crossers => mesh-like.
    if len(cross) >= 2 and len(sides) == 2 and all(c.band in ("shallow", "short") for c in cross):
        hit("mesh", cross, "two shallow crossers aligned on opposite sides")
    inside = _by(r, "in", "drag", "shallow_cross", "deep_cross", "cross")
    for side in ("left", "right"):
        same = [x for x in inside if x.side == side]
        if len({x.band for x in same}) >= 2:
            hit("levels", same, f"in-breaking routes at two depths on the {side} side")
        shallow = [x for x in same if x.family in ("drag", "shallow_cross")]
        behind = [x for x in same if x.family in ("in", "deep_cross") and x.band in ("short", "intermediate", "deep")]
        if shallow and behind:
            hit("drive", shallow + behind, f"shallow crosser with in-breaker behind on {side}")
        under = [x for x in r if x.side == side and x.family in ("hitch", "curl", "stick", "spot", "out", "flat")]
        over = [x for x in r if x.side == side and x.family == "corner"]
        if under and over and any(u.family in ("hitch", "curl") for u in under):
            hit("smash", under + over, f"hitch/curl under corner on {side}")
        lv = [x for x in r if x.side == side]
        if len({x.band for x in lv}) >= 3 and any(x.family in ("flat", "swing", "wheel", "out") for x in lv) \
                and any(x.family in ("corner", "fade", "streak") for x in lv):
            hit("flood", lv, f"three depth levels on {side} with flat and deep outside")
        stick = [x for x in r if x.side == side and x.family in ("stick", "hitch")]
        flat = [x for x in r if x.side == side and x.family in ("flat", "swing", "out")]
        if stick and flat:
            hit("stick", stick + flat, f"stick/hitch with flat/out on {side}")
    posts = _by(r, "post")
    digs = [x for x in r if x.family == "in" and x.band in ("intermediate", "deep")]
    if posts and digs:
        hit("post_dig", posts + digs, "post with a dig/deep in")
        if any(p.band == "deep" for p in posts):
            hit("mills", posts + digs, "deep post with a dig/deep in")
    verts = _by(r, "streak", "fade", "seam", "wheel", "sluggo")
    if len(verts) >= 4:
        hit("four_verticals", verts, "four vertical-stem routes")
    elif verts and digs:
        hit("dagger", verts + digs, "vertical stem with a dig behind it")
    short = [x for x in r if x.band in ("shallow", "short") and x.family != "screen"]
    if len(short) >= 3 and len({x.side for x in short}) >= 2:
        hit("spacing", short, "three-plus short routes across the formation")
    slants = _by(r, "slant")
    if slants and _by(r, "flat", "swing"):
        hit("slant_flat", slants + _by(r, "flat", "swing"), "slant with flat")
    return found
