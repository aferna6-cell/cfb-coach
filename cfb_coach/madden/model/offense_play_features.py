"""Structural play representation, adjustment transforms and matchup features.

``PlayStructure`` is what the model sees: only slots with evidence contribute
routes; everything else stays unknown and is flagged as such. Name hints enter
through their own channel (``hint_*``) with a low-trust flag. Matchup features
are football PRIORS keyed by coverage-scheme shape; they are not measured
Madden results and are fed to the learner as ordinary regularized inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_art_features as af
from cfb_coach.madden.model.coverage_intelligence import SCHEMES

FEATURE_VERSION = "offense_play_features.v2_route_art"
FORMATIONS = ("Gun 5WR Tight", "Gun Doubles Clamp Stack", "Gun Trips X Nasty", "Pistol Deuce Close", "Pistol Trips")
PLAY_TYPES = ("pass", "run", "play_action", "screen", "rpo", "option")
STRUCT_CONCEPTS = ("mesh", "flood", "levels", "drive", "smash", "dagger", "mills", "stick", "spacing",
                   "four_verticals", "post_dig", "slant_flat")
HINT_CONCEPTS = tuple(c for c in tx.CONCEPTS if c not in ("levels",))
RUN_GROUPS = {"zone": ("inside_zone", "mid_zone", "outside_zone", "split_zone", "zone_unspecified"),
              "gap": ("power", "counter", "duo", "dive", "draw", "off_tackle", "sweep", "qb_draw", "toss"),
              "option": ("read_option", "lead_read_option")}


class StructureError(ValueError):
    pass


@dataclass(frozen=True)
class PlayStructure:
    formation: str
    play: str
    routes: tuple[tuple[str, tx.RouteSpec, str], ...]   # (slot, spec, tier)
    skill_slots: tuple[str, ...]
    personnel_tier: str
    hint: Mapping[str, Any]
    sourced_concepts: tuple[str, ...] = ()
    read_order: tuple[str, ...] = ()
    ol_known: int = 0
    adjustments: tuple[str, ...] = ()
    play_id: str = ""

    @property
    def n_known(self) -> int:
        return len(self.routes)

    def route_for(self, slot: str) -> tx.RouteSpec | None:
        return next((r for s, r, _ in self.routes if s == slot), None)

    @property
    def specs(self) -> list[tx.RouteSpec]:
        return [r for _, r, _ in self.routes]

    def concepts(self) -> dict[str, dict[str, Any]]:
        return tx.detect_concepts(self.specs)


def structure_from_record(p: Mapping[str, Any]) -> PlayStructure:
    routes = tuple((n, tx.RouteSpec(s["route"]["family"], s["route"]["side"], s["route"]["band"],
                                    s["route"].get("published", "")), s["tier"])
                   for n, s in p["slots"].items() if s["category"] == "route" and s["tier"] != "unknown")
    ol = sum(1 for n in tx.OL_SLOTS if p["slots"][n]["tier"] != "unknown")
    return PlayStructure(
        formation=p["formation"], play=p["play"], routes=routes,
        skill_slots=tuple(p["personnel"]["skill_slots"]) if p["personnel"]["tier"] != "unknown" else (),
        personnel_tier=p["personnel"]["tier"], hint=p["name_hint"],
        sourced_concepts=tuple(c["name"] for c in p["concepts"]),
        read_order=tuple(p["read_order"]["positions"]), ol_known=ol, play_id=p["play_id"])


# ----------------------------------------------------------------- adjustments (stock -> modified)


def eligible_hot_route_targets(s: PlayStructure) -> tuple[str, ...]:
    """Receivers whose original route is known (so the change is exactly representable)."""
    return tuple(slot for slot, spec, _ in s.routes if spec.family not in ("pass_block",) and slot in s.skill_slots)


def apply_hot_route(s: PlayStructure, slot: str, option: str) -> PlayStructure:
    """Exact transformation: only ``slot`` changes; all other routes stay as documented."""
    if option not in tx.HOT_ROUTE_OPTIONS:
        raise StructureError(f"{option!r} is not a Madden hot-route option")
    if slot not in s.skill_slots:
        raise StructureError(f"{slot!r} is not a documented eligible slot in {s.formation!r}")
    old = s.route_for(slot)
    if old is None:
        raise StructureError(f"{slot}: original assignment unknown; cannot represent a hot route")
    fam = tx.HOT_ROUTE_TO_FAMILY[option]
    if fam == old.family and old.band == tx.ROUTE_FAMILIES[fam].default_band:
        raise StructureError(f"{slot}: already runs {option}")
    new = tx.RouteSpec(fam, old.side, tx.ROUTE_FAMILIES[fam].default_band, f"hot route: {option}")
    routes = tuple((sl, (new if sl == slot else sp), ("football_inference" if sl == slot else t))
                   for sl, sp, t in s.routes)
    return replace(s, routes=routes, read_order=(), sourced_concepts=(),
                   adjustments=s.adjustments + (f"{slot}:{old.family}->{fam}",))


def apply_route_changes(s: PlayStructure, changes: Mapping[str, str]) -> PlayStructure:
    for slot, option in changes.items():
        s = apply_hot_route(s, slot, option)
    return s


def compare_structures(stock: PlayStructure, mod: PlayStructure) -> dict[str, Any]:
    a, b = set(stock.concepts()), set(mod.concepts())
    return {"lost_concepts": sorted(a - b), "gained_concepts": sorted(b - a), "kept_concepts": sorted(a & b),
            "changed_slots": list(mod.adjustments)}


# ----------------------------------------------------------------- feature vectors


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def feature_names() -> tuple[str, ...]:
    return tuple(_ALL_NAMES)


def structure_features(s: PlayStructure) -> dict[str, float]:
    f: dict[str, float] = {n: 0.0 for n in _STRUCT_NAMES}
    specs = [r for r in s.specs if r.meta.category != "blocking"]
    n = len(specs)
    f["known_route_frac"] = len(s.routes) / 5.0
    f["has_route_combo"] = 1.0 if n >= 2 else 0.0
    f["ol_known_frac"] = s.ol_known / 5.0
    for form in FORMATIONS:
        f["form_" + form.lower().replace(" ", "_")] = 1.0 if s.formation == form else 0.0
    f["form_pistol"] = 1.0 if s.formation.startswith("Pistol") else 0.0
    ptype = s.hint.get("play_type", "unknown")
    for t in PLAY_TYPES:
        f["hint_type_" + t] = 1.0 if ptype == t else 0.0
    f["hint_type_known"] = 1.0 if ptype != "unknown" else 0.0
    sch = s.hint.get("run_scheme")
    for g, members in RUN_GROUPS.items():
        f["hint_run_" + g] = 1.0 if sch in members else 0.0
    f["hint_motion"] = 1.0 if s.hint.get("motion") else 0.0
    for c in HINT_CONCEPTS:
        f["hint_c_" + c] = 1.0 if c in s.hint.get("concepts", ()) else 0.0
    if n:
        cats = [r.meta.category for r in specs]
        for c in ("vertical", "breaking", "underneath", "crossing", "curl_sit"):
            f["r_cat_" + c] = cats.count(c) / n
        ys = [r.yards for r in specs]
        f["r_mean_depth"] = _mean(ys) / 20.0
        f["r_depth_spread"] = (max(ys) - min(ys)) / 20.0
        f["r_levels"] = len({r.band for r in specs}) / 4.0
        sides = [r.side for r in specs]
        f["r_side_imbalance"] = abs(sides.count("left") - sides.count("right")) / n
        f["r_man_beater"] = _mean([r.meta.man_beater for r in specs])
        f["r_zone_beater"] = _mean([r.meta.zone_beater for r in specs])
        f["r_quick_share"] = sum(r.meta.quick for r in specs) / n
        f["r_deep_count"] = sum(r.band == "deep" for r in specs) / 3.0
        f["r_inside_break"] = sum(r.meta.break_dir == "inside" for r in specs) / n
        f["r_outside_break"] = sum(r.meta.break_dir == "outside" for r in specs) / n
        f["r_deep_middle"] = sum(r.family in ("seam", "post", "deep_cross") or (r.family == "in" and r.band == "deep")
                                 for r in specs) / 2.0
    for c in s.concepts():
        if c in STRUCT_CONCEPTS:
            f["s_c_" + c] = 1.0
    return f


# ----------------------------------------------------------------- coverage matchup (priors)

SHELL_OF = {
    "cover_1_hole": "one_high_man", "cover_1_robber": "one_high_man", "cover_1_contain": "one_high_man",
    "cover_1_man_other": "one_high_man", "cover_1_pressure": "pressure", "cover_0": "pressure",
    "cover_2_man": "two_high_man", "cover_2_spot": "two_high_zone", "tampa_2": "two_high_zone",
    "cover_2_invert": "two_high_zone", "trap_2": "two_high_zone", "cover_3_spot": "three_deep_spot",
    "cover_3_match": "three_deep_match", "cover_4_quarters": "quarters", "cover_4_palms": "palms",
    "cover_4_drop": "four_drop", "cover_6": "split_field", "cover_9": "split_field", "unknown": "unknown",
}
SHELLS = ("one_high_man", "pressure", "two_high_man", "two_high_zone", "three_deep_spot", "three_deep_match",
          "quarters", "palms", "four_drop", "split_field")
# Coverage bucket (coach logs / opponent profile) -> scheme ids, uniform within the bucket (a prior, not data).
BUCKET_TO_SCHEMES = {
    "cover0": ("cover_0",), "cover1": ("cover_1_hole", "cover_1_robber", "cover_1_contain", "cover_1_man_other"),
    "cover2": ("cover_2_spot", "cover_2_man", "tampa_2"), "cover2_man": ("cover_2_man",),
    "cover3": ("cover_3_spot", "cover_3_match"), "cover4": ("cover_4_quarters", "cover_4_palms", "cover_4_drop"),
    "cover6": ("cover_6",), "pressure": ("cover_0",), "other": ("unknown",), "unknown": ("unknown",),
}


def scheme_distribution(mix: Mapping[str, float] | None) -> tuple[dict[str, float], str]:
    """Opponent coverage-bucket shares -> scheme probabilities. Empty/None -> an explicit 'unknown' prior."""
    if not mix or sum(max(0.0, float(v)) for v in mix.values()) <= 0:
        return {"unknown": 1.0}, "no_pre_snap_information"
    total = sum(max(0.0, float(v)) for v in mix.values())
    out: dict[str, float] = {}
    for bucket, v in mix.items():
        schemes = BUCKET_TO_SCHEMES.get(bucket, ("unknown",))
        for sc in schemes:
            out[sc] = out.get(sc, 0.0) + max(0.0, float(v)) / total / len(schemes)
    return out, "opponent_coverage_mix_uniform_within_bucket"


def _shell_probs(dist: Mapping[str, float]) -> dict[str, float]:
    out = {s: 0.0 for s in SHELLS}
    for sc, p in dist.items():
        shell = SHELL_OF.get(sc, "unknown")
        if shell in out:
            out[shell] += p
    return out


def matchup_features(s: PlayStructure, dist: Mapping[str, float]) -> dict[str, float]:
    """P(shell) x structural stress of the KNOWN routes. Zero when routes are unknown."""
    f = {n: 0.0 for n in _MATCH_NAMES}
    probs = _shell_probs(dist)
    known = sum(probs.values())
    specs = [r for r in s.specs if r.meta.category != "blocking"]
    f["m_info"] = known
    f["m_entropy"] = -sum(p * math.log(p) for p in dist.values() if p > 0) / math.log(8)
    if not specs:
        return f
    n = len(specs)
    man = _mean([r.meta.man_beater for r in specs])
    zone = _mean([r.meta.zone_beater for r in specs])
    quick = sum(r.meta.quick for r in specs) / n
    deep = sum(r.band == "deep" for r in specs) / n
    deep_mid = sum(r.family in ("seam", "post", "deep_cross") or (r.family == "in" and r.band == "deep") for r in specs) / 2.0
    flat = sum(r.family in ("flat", "swing", "wheel", "out", "angle", "texas") for r in specs) / n
    vert = sum(r.meta.category == "vertical" for r in specs) / n
    cross = sum(r.meta.category == "crossing" for r in specs) / n
    sit = sum(r.meta.category == "curl_sit" for r in specs) / n
    quick_out = sum(r.family in ("out", "flat", "swing") and r.band in ("shallow", "short") for r in specs) / n
    f["m_one_high_man"] = probs["one_high_man"] * (man + 0.5 * cross)
    f["m_pressure_quick"] = probs["pressure"] * quick
    f["m_pressure_slow"] = probs["pressure"] * deep
    f["m_two_high_man"] = probs["two_high_man"] * (man + 0.5 * cross)
    f["m_two_high_zone"] = probs["two_high_zone"] * (flat + 0.5 * deep_mid)
    f["m_three_deep_spot"] = probs["three_deep_spot"] * (sit + 0.5 * flat)
    f["m_three_deep_match"] = probs["three_deep_match"] * (zone * 0.5 + man * 0.5 + 0.3 * cross)
    f["m_quarters"] = probs["quarters"] * (cross + 0.5 * sit - 0.3 * vert)
    f["m_palms"] = probs["palms"] * (quick_out + 0.5 * cross - 0.2 * vert)
    f["m_four_drop"] = probs["four_drop"] * (zone + 0.3 * sit)
    f["m_split_field"] = probs["split_field"] * 0.5 * (zone + man)
    return f


_STRUCT_NAMES: tuple[str, ...] = tuple(
    ["known_route_frac", "has_route_combo", "ol_known_frac"]
    + ["form_" + f.lower().replace(" ", "_") for f in FORMATIONS] + ["form_pistol"]
    + ["hint_type_" + t for t in PLAY_TYPES] + ["hint_type_known"]
    + ["hint_run_" + g for g in RUN_GROUPS] + ["hint_motion"]
    + ["hint_c_" + c for c in HINT_CONCEPTS]
    + ["r_cat_" + c for c in ("vertical", "breaking", "underneath", "crossing", "curl_sit")]
    + ["r_mean_depth", "r_depth_spread", "r_levels", "r_side_imbalance", "r_man_beater", "r_zone_beater",
       "r_quick_share", "r_deep_count", "r_inside_break", "r_outside_break", "r_deep_middle"]
    + ["s_c_" + c for c in STRUCT_CONCEPTS])
_MATCH_NAMES: tuple[str, ...] = (
    "m_info", "m_entropy", "m_one_high_man", "m_pressure_quick", "m_pressure_slow", "m_two_high_man",
    "m_two_high_zone", "m_three_deep_spot", "m_three_deep_match", "m_quarters", "m_palms", "m_four_drop",
    "m_split_field")
_ALL_NAMES: tuple[str, ...] = _STRUCT_NAMES + _MATCH_NAMES + af.FEATURE_NAMES
# Weak-prior groups get a larger L2 penalty in the model (name hints are low trust).
GROUPS = {"form": ("form_",), "hint": ("hint_",), "route": ("r_", "known_", "has_", "ol_"),
          "concept": ("s_c_",), "match": ("m_",), "art": ("art_",)}


def group_of(name: str) -> str:
    for g, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return g
    return "route"


def full_vector(s: PlayStructure, dist: Mapping[str, float]) -> list[float]:
    f = structure_features(s)
    f.update(matchup_features(s, dist))
    # These describe the ORIGINAL stock play-art layout, even for hypothetical
    # hot-routed candidates. Never claim the source image depicts that adjustment.
    f.update(af.art_features(s.formation, s.play))
    return [f.get(n, 0.0) for n in _ALL_NAMES]


_CACHE: dict[str, PlayStructure] = {}


def structure_lookup(formation: str, play: str, db: Mapping[str, Any] | None = None) -> PlayStructure | None:
    """Installed play -> structure; ``None`` for a play that is not in the 93-play inventory."""
    if db is None:
        key = f"{formation}|{play}"
        if not _CACHE:
            for p in oa.load_effective()["plays"]:
                _CACHE[f"{p['formation']}|{p['play']}"] = structure_from_record(p)
        return _CACHE.get(key)
    rec = oa.index(db).get((formation, play))
    return structure_from_record(rec) if rec else None
