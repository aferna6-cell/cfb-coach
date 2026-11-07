"""Short pre-snap quarterback card built from the called play's actual known assignments.

Never fabricates a progression: a documented read order is labelled DOCUMENTED,
a football-derived one INFERRED, and a play with no route knowledge says UNKNOWN.
"""
from __future__ import annotations

from typing import Any, Mapping

from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import football_priors as fpri
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_taxonomy as tx

# What the QB can actually see pre-snap for each expected shell (pre-snap indicators only).
INDICATOR = {
    "one_high_man": "one safety deep and corners tight on receivers: man or single-high",
    "pressure": "no deep safety help / extra defenders on the line: pressure or Cover 0",
    "two_high_man": "two safeties deep with corners tight: man under two deep",
    "two_high_zone": "two safeties deep, corners off: split-safety zone",
    "three_deep_spot": "one safety deep, corners bailing at the snap: three-deep zone",
    "three_deep_match": "one safety deep, corners/safeties reading releases: three-deep match",
    "quarters": "two safeties deep and creeping over #2: quarters",
    "palms": "two safeties deep with corners squatting on #1: palms/two-read",
    "four_drop": "two safeties deep, spot-dropping to deep fourths",
    "split_field": "different shells on each side of the field: split-field coverage",
    "unknown": "pre-snap coverage not observable: check safeties (one vs two deep) and corner depth",
}
_AFFINITY = {  # route family -> shells it attacks (football prior used ONLY for an inferred ordering)
    "drag": ("one_high_man", "two_high_man", "pressure"), "shallow_cross": ("one_high_man", "two_high_man"),
    "slant": ("one_high_man", "pressure", "two_high_man"), "in": ("three_deep_spot", "two_high_zone", "four_drop"),
    "post": ("three_deep_spot", "one_high_man", "split_field"), "corner": ("two_high_zone", "four_drop", "quarters"),
    "out": ("three_deep_spot", "palms"), "flat": ("two_high_zone", "pressure"), "swing": ("two_high_zone", "pressure"),
    "curl": ("three_deep_spot", "four_drop"), "hitch": ("three_deep_spot", "two_high_zone"),
    "wheel": ("two_high_zone", "one_high_man"), "streak": ("one_high_man", "three_deep_match"),
    "seam": ("two_high_zone", "one_high_man"), "deep_cross": ("three_deep_spot", "quarters"),
    "angle": ("one_high_man", "pressure"), "texas": ("one_high_man", "pressure"),
}
_CHECKDOWN = ("flat", "swing", "angle", "texas", "drag", "block_and_release_hook", "pivot")


def _describe(slot: str, r: tx.RouteSpec) -> str:
    pub = f" [{r.published}]" if r.published else ""
    return f"{slot} {r.family.replace('_', ' ')} ({r.band}, {r.side}){pub}"


def _shell_dist(dist: Mapping[str, float]) -> dict[str, float]:
    return pf._shell_probs(dist)


def build_card(decision: od.Decision, mix: Mapping[str, float] | None = None, *, priors: bool | None = None) -> dict[str, Any]:
    s = decision.final_structure
    if s is None:
        raise ValueError("decision has no play structure")
    dist, src = pf.scheme_distribution(mix)
    shells = _shell_dist(dist)
    top = max(shells, key=shells.get) if shells and max(shells.values()) > 0 else "unknown"
    top_p = shells.get(top, 0.0)
    use_priors = bool(getattr(decision, "football_priors", False)) if priors is None else priors
    primary = secondary = checkdown = "UNKNOWN: this play's receiver routes are not annotated yet"
    conditioned: list[str] = []
    basis = "unknown"
    why = []
    hint = s.hint
    play_type = hint.get("play_type", "unknown")
    if s.n_known:
        by_slot = {sl: sp for sl, sp, _ in s.routes}
        documented = [x for x in s.read_order if x in by_slot]
        if documented and not s.adjustments:
            order, basis = documented, "DOCUMENTED (third-party published read order; not Xbox-verified)"
        else:
            def aff(item: tuple[str, tx.RouteSpec]) -> float:
                if use_priors and mix:                  # coverage-conditioned prior preference of this route family under the opponent's mixture
                    return fpri.route_read_scores(item[1].family, mix) + 0.01 * item[1].meta.man_beater
                return sum(shells.get(sh, 0.0) for sh in _AFFINITY.get(item[1].family, ())) \
                    + 0.01 * item[1].meta.man_beater
            order = [sl for sl, _ in sorted(by_slot.items(), key=lambda kv: -aff(kv))
                     if by_slot[sl].meta.category != "blocking"]
            basis = ("INFERRED from football principles for the expected coverage" if not s.adjustments else
                     "INFERRED (hot route changed the play; documented order no longer applies)")
        order = [o for o in order if by_slot[o].meta.category != "blocking"]
        if use_priors and mix and basis.startswith("DOCUMENTED"):
            def _aff(sl: str) -> float:
                return fpri.route_read_scores(by_slot[sl].family, mix) + 0.01 * by_slot[sl].meta.man_beater
            conditioned = sorted(order, key=lambda sl: -_aff(sl))                  # shown beside the documented order, never replacing it
        primary = _describe(order[0], by_slot[order[0]]) if order else primary
        secondary = _describe(order[1], by_slot[order[1]]) if len(order) > 1 else "UNKNOWN"
        keep_in = any(a.action_id in ("off.pass_protection", "off.chip_block") for a in getattr(decision, "actions", ()))
        cd = [(sl, sp) for sl, sp in by_slot.items() if sp.family in _CHECKDOWN and sl not in order[:2]
              and not (keep_in and sl.startswith(("HB", "FB")))]               # a back kept in to block is not an outlet
        if keep_in:
            why.append("back kept in to protect: not offered as the checkdown")
        checkdown = _describe(*cd[0]) if cd else "UNKNOWN: no documented outlet route"
        concepts = sorted(s.concepts())
        if concepts:
            why.append("route combination: " + ", ".join(concepts))
        if s.adjustments:
            why.append("after " + "; ".join(s.adjustments))
    elif play_type in ("run", "option"):
        primary = f"RUN ({hint.get('run_scheme') or 'scheme unknown'}; name-level hint, lane and blocking UNKNOWN)"
        secondary = checkdown = "n/a (run)"
        basis = "name-level hint only"
    elif play_type == "rpo":
        primary = "RPO: run and pass components both UNKNOWN (name-level hint only)"
        basis = "name-level hint only"
    else:
        why.append("no sourced route data for this play")
    indicator = INDICATOR[top] + (f" (expected {top_p:.0%} from {src})" if top != "unknown" else "")
    lo, hi = decision.interval
    why.append(f"model {decision.probability:.0%} success (90% range {lo:.0%}-{hi:.0%})")
    if decision.adjustment != od.NO_ADJUSTMENT:
        why.append(f"gain {decision.probability - decision.stock_probability:+.1%} vs stock; {decision.basis}")
    return {
        "FORMATION": decision.formation, "PLAY": decision.play, "ADJUSTMENT": decision.adjustment,
        "PRIMARY READ": primary, "SECONDARY READ": secondary, "CHECKDOWN": checkdown,
        "DEFENSIVE INDICATOR": indicator, "WHY": "; ".join(why), "READ BASIS": basis,
        "COVERAGE-CONDITIONED ORDER": conditioned or None,
        "EXECUTABLE NOW": "yes" if decision.executable_now else "NO - adjustment not yet confirmed in-game; run the stock play",
    }


def format_card(card: Mapping[str, Any]) -> str:
    keys = ("FORMATION", "PLAY", "ADJUSTMENT", "PRIMARY READ", "SECONDARY READ", "CHECKDOWN",
            "DEFENSIVE INDICATOR", "WHY")
    lines = [f"{k}: {card[k]}" for k in keys]
    if card["READ BASIS"] != "unknown":
        lines.append(f"READ BASIS: {card['READ BASIS']}")
    if card["EXECUTABLE NOW"] != "yes":
        lines.append(f"EXECUTION: {card['EXECUTABLE NOW']}")
    return "\n".join(lines)
