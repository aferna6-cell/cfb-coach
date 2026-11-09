"""Cards' football content: QB reads from the FINAL (adjusted) play, and defensive USER assignments.

Rules: a receiver identity or route order is only stated where a documented/verified route exists for
that slot; everything else is reported at concept level with identity ``UNKNOWN``. A defensive
assignment is ``verified`` only when it comes from a verified ledger / owner role file; otherwise
the USER advice is a labelled football prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.unified.adjustments import opponent_tags
from cfb_coach.madden.unified.opponent import Belief
from cfb_coach.madden.unified.types import DEF, OFF, UNKNOWN, FootballAction


@dataclass(frozen=True)
class ReadPlan:
    primary: str
    secondary: str
    identity_known: bool
    basis: str
    changed_routes: Mapping[str, str] = field(default_factory=dict)
    explanation: str = ""


@dataclass(frozen=True)
class UserAssignment:
    position: str
    primary: str
    secondary: str
    verified: bool
    basis: str
    explanation: str = ""


def _route_score(route: str, belief: Belief) -> float | None:
    fam = tx.ROUTE_FAMILIES.get(route.lower())
    if fam is None:
        return None
    s = 0.0
    for sig, p in belief.dist.items():
        tags = opponent_tags(DEF, sig)
        man = 1.0 if "man" in tags else 0.5 if sig[0] == "match" else 0.0
        base = man * fam.man_beater + (1 - man) * fam.zone_beater
        if "pressure" in tags:
            base += 0.2 if fam.quick else -0.1
        s += p * base
    return s


def _anticipation(belief: Belief) -> str:
    man = sum(p for s, p in belief.dist.items() if s[0] == "man")
    blitz = sum(p for s, p in belief.dist.items() if s[0] == "pressure")
    two = sum(p for s, p in belief.dist.items() if s[1] in ("cover2", "cover4", "cover6", "cover9"))
    top = belief.top[0] if belief.top else None
    t = f"most likely {'/'.join(top[0])} ({top[1]:.0%})" if top else "no strong lean"
    return f"expect man {man:.0%}, pressure {blitz:.0%}, two-high {two:.0%}; {t}; source {belief.prediction_source}"


def offense_reads(action: FootballAction, belief: Belief) -> ReadPlan:
    routes: dict[str, str] = dict(action.modified_features.get("routes") or
                                  {a.slot: a.role for a in action.play.assignments if a.role != UNKNOWN})
    scored = [(slot, r, _route_score(r, belief)) for slot, r in routes.items()]
    scored = sorted([x for x in scored if x[2] is not None], key=lambda x: -x[2])
    changed = dict(action.modified_features.get("changed_routes") or {})
    anticipate = _anticipation(belief)
    if len(scored) >= 2:
        verified = any(a.verified for a in action.play.assignments)
        basis = "assignment_tier:" + ("verified" if verified else "source_documented") + (" +hot_route" if changed else "")
        return ReadPlan(f"{scored[0][0]} ({scored[0][1]})", f"{scored[1][0]} ({scored[1][1]})", True, basis, changed, anticipate)
    if len(scored) == 1:
        return ReadPlan(f"{scored[0][0]} ({scored[0][1]})", f"UNKNOWN (only {len(scored)} documented route)", False,
                        "partial_route_knowledge", changed, anticipate)
    from cfb_coach.madden.model.football_knowledge import CONCEPTS
    pr = CONCEPTS.get(str(action.play.concept), {})
    prim = f"concept-level: {pr.get('attempts')}" if pr.get("attempts") else f"concept-level: {action.play.concept}"
    sec = f"hold the stress point: {', '.join(pr.get('stresses', []))}" if pr.get("stresses") else "UNKNOWN"
    return ReadPlan(prim, sec, False, "concept_level_only: receiver identities and route order UNKNOWN for this play", changed, anticipate)


def _threats(belief: Belief) -> dict[str, float]:
    t = {"run": 0.0, "screen": 0.0, "vert": 0.0, "cross": 0.0, "pass_other": 0.0}
    for sig, p in belief.dist.items():
        tags = opponent_tags(OFF, sig)
        if sig[0] in ("run", "option"):
            t["run"] += p
        elif "screen" in tags:
            t["screen"] += p
        elif "vert" in tags:
            t["vert"] += p
        elif "cross" in tags or "flood" in tags:
            t["cross"] += p
        else:
            t["pass_other"] += p
    return t


_RESP = {"run": "fit the run: fill the nearest gap, spill nothing", "screen": "read screen: sink and trigger on the throw",
         "vert": "protect depth: stay over the top of vertical routes", "cross": "work hook/curl: jam and carry crossers",
         "pass_other": "play your zone/man job; break on the QB's eyes"}
_POS = {"run": "SS", "screen": "MIKE", "vert": "FS", "cross": "MIKE", "pass_other": "MIKE"}


def defense_user(action: FootballAction, belief: Belief, roles_file: str | None = None) -> UserAssignment:
    from cfb_coach.madden.model.defense_v2_roles import verified_roles

    th = _threats(belief)
    ranked = sorted(th, key=lambda k: -th[k])
    top, second = ranked[0], ranked[1]
    expl = "offense expected: " + ", ".join(f"{k} {th[k]:.0%}" for k in ranked[:3]) + f"; source {belief.prediction_source}"
    vr = verified_roles(roles_file, action.play.formation, action.play.play) if roles_file else {"slots": []}
    slots = list(vr.get("slots") or [])
    if slots:
        kw = {"run": ("run", "gap", "fit", "box"), "vert": ("deep", "post", "vert"), "cross": ("hook", "curl", "cross", "mid"),
              "screen": ("flat", "screen"), "pass_other": ("zone", "man", "cover")}

        def fit(s: Mapping[str, Any]) -> int:
            text = (str(s.get("role", "")) + " " + str(s.get("responsibility", ""))).lower()
            return sum(1 for w in kw[top] if w in text)

        best = max(slots, key=fit)
        return UserAssignment(f"{best['slot_id']} ({best['role']})", best["responsibility"], f"football prior: {_RESP[second]}",
                              True, "owner_xbox_verified_role (primary); secondary is an unverified football prior", expl)
    if action.play.assignments:  # catalog ledger with verified defenders
        v = [a for a in action.play.assignments if a.verified and a.role != UNKNOWN]
        if v:
            a = v[0]
            return UserAssignment(a.slot, a.role, f"football prior: {_RESP[second]}", True, "catalog_ledger_verified", expl)
    return UserAssignment(_POS[top], _RESP[top], _RESP[second], False,
                          "football_prior: this play's defender assignments are UNKNOWN; position is advisory only", expl)
