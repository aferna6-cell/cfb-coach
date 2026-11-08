"""Short pre-snap quarterback card built from the called play's actual known assignments.

Never fabricates a progression: a documented read order is labelled DOCUMENTED,
a football-derived one INFERRED, and a play with no route knowledge says UNKNOWN.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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
    if slot.startswith("geom_"):
        # Grok extracted shape/location, not the receiving player's identity.
        # Never display anonymous geom_0 as if it were WR1/X/Y/B/RT.
        return f"{r.side} {r.family.replace('_', ' ')} ({r.band}; receiver unverified)"
    return f"{slot} {r.family.replace('_', ' ')} ({r.band}, {r.side}){pub}"


def _shell_dist(dist: Mapping[str, float]) -> dict[str, float]:
    return pf._shell_probs(dist)


def call_action_identity(call: Any) -> dict[str, Any]:
    """Snapshot the attached execution plan, not just its formation/play."""
    return deepcopy({"macro": getattr(call, "macro", None),
                     "adjustment": getattr(call, "adjustment", None),
                     "macro_settings": (getattr(call, "macro_info", None) or {}).get("settings")})


def use_selected_call(decision: od.Decision, call: Any) -> None:
    """Project the attached joint call into a card decision; never select another action.

    The live catalog resolver may have paired a macro after joint scoring. The
    attached call is authoritative. Unrepresentable effects suppress the stock
    progression rather than implying it still applies.
    """
    from cfb_coach.madden import research_db
    from cfb_coach.madden.adjustments import executable as ex

    stock = decision.final_structure
    if stock is None:
        raise ValueError("decision has no play structure")
    adjustment = getattr(call, "adjustment", None) or {}
    macro = getattr(call, "macro", None)
    decision.adjustment = ("MACRO " + str(macro) if macro else
                           str(adjustment.get("label") or adjustment.get("id") or od.NO_ADJUSTMENT))
    decision.basis = "selected_joint_call"
    if macro or adjustment:
        decision.matchup = None   # stock matchup is not the selected action's effect
    decision.notes = [str((getattr(call, "ml_offense_action", None) or {}).get("reason") or "")]
    actions = []
    unknown = bool(macro)
    rows = {r["id"]: r for r in research_db.offense_adjustments()}
    ids = adjustment.get("parts") or ([adjustment.get("id")] if adjustment else [])
    for action_id in ids:
        row = rows.get(action_id) or {}
        if not adjustment.get("parts"):
            row = {**row, **adjustment}
        kind = row.get("kind") or row.get("type")
        if kind == "hot_route" and row.get("target") and row.get("route"):
            actions.append(ex.PlannedAction("off.hot_route", {"target": row["target"], "route": row["route"]}))
        else:
            # Protection, custom macros and generic targets may change who can
            # release. No exact receiver effects are documented by this payload.
            unknown = True
    decision.actions = tuple(actions)
    if not unknown:
        try:
            decision.final_structure = od._apply_actions(stock, decision.actions, {})
        except pf.StructureError:
            unknown = True
    if unknown:
        decision.final_structure = replace(stock, routes=(), read_order=(), sourced_concepts=(),
                                           hint={**stock.hint, "play_type": "unknown"})
        decision.notes.append("selected adjustment's receiver effects are not representable; read order UNKNOWN")


def _contextual_read_order(
    routes: Mapping[str, tx.RouteSpec],
    documented: list[str],
    *,
    mix: Mapping[str, float] | None,
    context: Mapping[str, Any] | None,
    evidence_tiers: Mapping[str, str],
) -> tuple[list[str], str, list[str]]:
    """Rank ONLY routes shown by the selected play, using pre-snap beliefs.

    These are football-informed *hypotheses*, NOT learned per-receiver open rates.
    Published progressions are a small tie-breaker rather than a compulsory
    decision; uncertain/unknown coverages shrink all coverage-specific evidence.
    No previous-snap revealed call is passed as current-snap ground truth.
    """
    context = context or {}
    doc = [slot for slot in documented if slot in routes and routes[slot].meta.category != "blocking"]
    baseline = doc + [
        slot for slot in routes
        if slot not in doc and routes[slot].meta.category != "blocking"
    ]
    positive = {
        k: float(w) for k, w in (mix or {}).items()
        if isinstance(w, (float, int)) and not isinstance(w, bool) and 0 < w <= 1
    }
    total = sum(positive.values())
    known_mass = (sum(w for k, w in positive.items() if fpri.descriptor(k) is not None)
                  / total if total > 0 else 0.0)
    # A diffuse, data-poor history should not override a documented progression.
    informed = known_mass >= 0.35
    cover_scores = {
        slot: (fpri.route_read_scores(spec.family, positive) / max(total, 1e-9)
               if informed else 0.0)
        for slot, spec in routes.items()
    }

    distance = context.get("distance")
    down = context.get("down")
    urgent = (
        isinstance(down, int) and down >= 3 and
        isinstance(distance, (int, float)) and distance >= 8
    )
    short_to_gain = (
        isinstance(down, int) and down >= 3 and
        isinstance(distance, (int, float)) and distance <= 3
    )
    two_minute = bool(context.get("two_minute"))
    score_us, score_them = context.get("score_us"), context.get("score_them")
    trailing = (isinstance(score_us, (int, float)) and
                isinstance(score_them, (int, float)) and score_us < score_them)
    # Pressure is a *distribution* over possible next-snap defenses, not a
    # hard switch caused by the one coverage revealed on the last snap.
    pressure = sum(
        w for k, w in positive.items()
        if (fpri.descriptor(k) or {}).get("press") not in (None, "", "standard")
        or k == "cover_0"
    ) / max(total, 1e-9)

    scores = {}
    for slot in baseline:
        route = routes[slot]
        tier = evidence_tiers.get(slot)
        reliability = (0.55 if tier == "partial" else
                       0.8 if tier == "high_confidence_extracted" else 1.0)
        # Retain a very small generic route-quality prior for data-sparse
        # situations and compatibility with the existing no-adjustment reads.
        score = reliability * cover_scores[slot] * known_mass + 0.01 * route.meta.man_beater
        # Small documented-order tie break; the next-snap football evidence
        # may displace it only when the advantage is meaningful.
        if slot in doc:
            score += 0.012 if slot == doc[0] else 0.004
        if urgent:
            score += (0.048 if route.band in ("intermediate", "deep")
                      else -0.024 if route.band in ("short", "shallow") else 0.0)
        elif short_to_gain:
            score += (0.028 if route.band in ("short", "shallow")
                      else -0.022 if route.band == "deep" else 0.0)
        if trailing and two_minute:
            score += 0.018 if route.family in ("out", "corner", "flat") else 0.0
        if pressure > 0.0:
            score += 0.035 * pressure if route.meta.quick else -0.045 * pressure
        scores[slot] = score

    order = sorted(baseline, key=lambda slot: (-scores[slot], baseline.index(slot)))
    changed = bool(doc and order and order[0] != doc[0])
    notes = []
    if not informed:
        notes.append("defensive coverage uncertain: documented ordering preferred where available")
    if urgent:
        notes.append(f"3rd/4th-and-long: look for routes reaching the sticks; verify actual window")
    if short_to_gain:
        notes.append("short to gain: prefer routes that can uncover quickly")
    if trailing and two_minute:
        notes.append("trailing in two-minute: boundary route gets a modest preference")
    if pressure > 0.1:
        notes.append(f"possible pressure ({pressure:.0%} of recognized pre-snap mix): quicker developing routes preferred")
    if changed:
        notes.append("current recommendation differs from published progression")
    if doc:
        basis = ("INFERRED contextual read order (published progression retained as reference)"
                 if changed else
                 "DOCUMENTED baseline retained after contextual pre-snap comparison (not Xbox-verified)")
    elif baseline:
        basis = "INFERRED contextual read order from known route shapes; not a verified QB progression"
    else:
        basis = "unknown"
    return order, basis, notes


def build_card(decision: od.Decision, mix: Mapping[str, float] | None = None, *, priors: bool | None = None, situation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    s = decision.final_structure
    if s is None:
        raise ValueError("decision has no play structure")
    dist, src = pf.scheme_distribution(mix)
    shells = _shell_dist(dist)
    top = max(shells, key=shells.get) if shells and max(shells.values()) > 0 else "unknown"
    top_p = shells.get(top, 0.0)
    primary = secondary = checkdown = "UNKNOWN: this play's receiver routes are not annotated yet"
    conditioned: list[str] = []
    basis = "unknown"
    why = []
    published_baseline: list[str] = []
    hint = s.hint
    play_type = hint.get("play_type", "unknown")
    if s.n_known:
        by_slot = {sl: sp for sl, sp, _ in s.routes}
        documented = [x for x in s.read_order if x in by_slot]
        evidence_tiers = {slot: tier for slot, _, tier in s.routes}
        order, basis, read_notes = _contextual_read_order(
            by_slot, documented, mix=mix,
            context=situation, evidence_tiers=evidence_tiers,
        )
        anonymous = all(slot.startswith("geom_") and tier in (
            "partial", "high_confidence_extracted"
        ) for slot, _, tier in s.routes)
        if anonymous and not s.adjustments:
            # Preserve the established diagnostic/provenance token consumed by
            # the existing read coverage benchmark and regression tests.
            basis = ("INFERRED from unreviewed play-diagram geometry; "
                     "contextual coverage/situation preference, approximate "
                     "route shapes, receiver identities UNKNOWN")
        elif s.adjustments:
            basis += " [selected hot routes; re-ranked after the changes]"
        published_baseline = [_describe(sl, by_slot[sl]) for sl in documented]
        if published_baseline:
            why.append("published progression is a baseline, not a guaranteed best read")
        why.extend(read_notes)
        acts = list(getattr(decision, "actions", ()))
        blocking_slots = {str(a.params.get("target")) for a in acts if a.action_id == "off.chip_block" and a.params.get("target")}
        order = [o for o in order if o not in blocking_slots]
        conditioned = [_describe(sl, by_slot[sl]) for sl in order]
        primary = _describe(order[0], by_slot[order[0]]) if order else primary
        secondary = _describe(order[1], by_slot[order[1]]) if len(order) > 1 else "UNKNOWN"
        # A back leaves the checkdown list ONLY when a selected action demonstrably changes THAT player's assignment: a chip block names its target.
        # A pass-protection call (Half/Full Slide) is not known to keep the back in, so it removes nobody and the card says so.
        cd = [(sl, sp) for sl, sp in by_slot.items() if sp.family in _CHECKDOWN and sl not in order[:2] and sl not in blocking_slots]
        if blocking_slots & set(by_slot):
            why.append("chip block assigned to " + ", ".join(sorted(blocking_slots & set(by_slot))) + ": not offered as the checkdown")
        if any(a.action_id == "off.pass_protection" for a in acts):
            why.append("protection call: its effect on the backs' assignments is not documented, so no checkdown was removed")
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
    if decision.basis == "selected_joint_call":
        if decision.adjustment == od.NO_ADJUSTMENT:
            why.append(f"model {decision.probability:.0%} success (90% range {lo:.0%}-{hi:.0%})")
        else:
            why.append(f"stock model {decision.stock_probability:.0%} success; selected action effect not estimated")
        why.extend(n for n in decision.notes if n)
    else:
        why.append(f"model {decision.probability:.0%} success (90% range {lo:.0%}-{hi:.0%})")
    if decision.adjustment != od.NO_ADJUSTMENT and decision.basis != "selected_joint_call":
        why.append(f"gain {decision.probability - decision.stock_probability:+.1%} vs stock; {decision.basis}")
    return {
        "FORMATION": decision.formation, "PLAY": decision.play, "ADJUSTMENT": decision.adjustment,
        "PRIMARY READ": primary, "SECONDARY READ": secondary, "CHECKDOWN": checkdown,
        "DEFENSIVE INDICATOR": indicator, "WHY": "; ".join(why), "READ BASIS": basis,
        "COVERAGE-CONDITIONED ORDER": conditioned or None,
        "PUBLISHED BASELINE": published_baseline or None,
        "READ INPUTS": {
            "coverage_mix": ("historical pre-snap probability mixture, not last-snap certainty"
                             if mix else "none supplied"),
            "down": (situation or {}).get("down"),
            "distance": (situation or {}).get("distance"),
            "two_minute": (situation or {}).get("two_minute"),
            "score_us": (situation or {}).get("score_us"),
            "score_them": (situation or {}).get("score_them"),
        },
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
