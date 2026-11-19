"""USER (the defender Aidan controls) for one complete candidate + offensive alignment.

LIVE_VALID only when Aidan verified that exact (call, Zone It or not, offensive alignment) in
Nickel 2-4 Wide / Dime 2-3 Odd in game, with the configured check. The 18 Civil Match-guide roles
were captured in a different formation (Dime 2-3) and are kept as internal hints, never displayed
as verified. Zone It never inherits a Match USER. Nickel stock and DB Fire 2 have no source USER.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from cfb_coach.madden.ravens_defense.catalog import Candidate, RavensCatalog
from cfb_coach.madden.ravens_defense.confirmations import Confirmations

UNVERIFIED = "UNVERIFIED"
_SHAPE_PATTERNS = (
    ("5wr_tight", r"\b(5\s*wr\s*tight|empty\s*tight|5wr)\b"),
    ("4x1", r"\b(4x1|quads|quad)\b"),
    ("bunch", r"\bbunch\b"),
    ("trips", r"\b(trips|trey|3x1)\b"),
    ("stacks", r"\bstacks?\b"),
    ("2x2", r"\b(2x2|doubles|double\s*slot|spread)\b"),
)


def parse_offense_shape(text: str | None) -> str | None:
    t = (text or "").lower()
    for shape, pat in _SHAPE_PATTERNS:
        if re.search(pat, t):
            return shape
    return None


def guide_rule(cat: RavensCatalog, call_id: str, shape: str | None) -> Mapping[str, Any] | None:
    hits = [r for r in cat.raw["user_assignment_rules"] if r["call_id"] == call_id and r["opponent_shape_key"] == shape]
    return hits[0] if len(hits) == 1 else None


def resolve(cat: RavensCatalog, cand: Candidate, shape: str | None, conf: Confirmations) -> dict[str, Any]:
    zone_it = cand.optional_macro == "ZONE_IT"
    rule = None if zone_it else guide_rule(cat, cand.call_id, shape)
    out: dict[str, Any] = {"status": UNVERIFIED, "user": None, "offense_shape": shape,
                           "guide_assignment_id": rule["assignment_id"] if rule else None,
                           "guide_hint_not_displayed": rule["guide_user"] if rule else None,
                           "guide_checks": list(rule["guide_checks"]) if rule else [], "missing": []}
    if shape is None:
        out["missing"].append("offensive alignment not entered for this snap (trips/bunch/2x2/stacks/5wr tight/4x1)")
        return out
    ver = conf.user_verification(cand.call_id, zone_it, shape)
    if ver and ver.get("formation") == cand.formation:
        if rule and rule["guide_checks"] and not str(ver.get("check") or "").strip():
            out["missing"].append("configured Match check not recorded with the in-game verification")
            return out
        if cand.call_id in ("match_cover9_show2", "nickel_cover9") and rule and not conf.cover9_show2_alias_confirmed:
            out["missing"].append("guide 'Cover 9' -> Ravens 'Cover 9 Show 2' mapping not confirmed")
            return out
        out.update(status="LIVE_VALID", user=str(ver["user"]), verified_on=ver.get("verified_on"),
                   check=ver.get("check"), provenance="owner_in_game")
        return out
    if zone_it:
        out["missing"].append("Zone It spot-drop USER not verified in game; the Match USER is NOT carried over")
    elif rule:
        out["missing"] += [
            f"guide role '{rule['guide_user']}' was captured outside {cand.formation} (Match guide Dime 2-3); "
            "transfer requires in-game confirmation",
            "configured Match check (" + ", ".join(rule["guide_checks"]) + ") not confirmed in game",
            "defender position for this alignment not confirmed in game",
        ]
        if rule.get("user_status") == "inferred_alignment_not_live_eligible":
            out["missing"].append("guide alignment for this rule is inferred, not stated")
    else:
        out["missing"].append(f"no sourced USER for {cand.candidate_id} vs {shape}; needs owner in-game verification")
    return out


def audit(cat: RavensCatalog, conf: Confirmations) -> list[dict[str, Any]]:
    """All 18 guide assignments: are they live-valid in Nickel 2-4 Wide?"""
    rows = []
    cands = {c.candidate_id: c for c in cat.candidates}
    for r in cat.raw["user_assignment_rules"]:
        res = resolve(cat, cands[r["call_id"]], r["opponent_shape_key"], conf)
        rows.append({"assignment_id": r["assignment_id"], "call_id": r["call_id"], "shape": r["opponent_shape_key"],
                     "guide_user": r["guide_user"], "guide_checks": r["guide_checks"], "status": res["status"],
                     "missing": res["missing"]})
    return rows
