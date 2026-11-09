"""Madden 27 defensive concept templates and opponent-conditioned USER advice.

A coverage-family responsibility template is NOT 11-player play art.  Preserve
that separation explicitly; only documented per-player diagrams may identify
the duties of particular defenders.  No hidden pre-snap information is used.

Sources:
EA Madden NFL 27 Gameplay Deep Dive (2026-06-04), coverage, match checks,
smart zones, leverage, pre-play defensive alignment.
Project defensive catalog: defense_semantics.explain().
"""
from __future__ import annotations

from typing import Any, Mapping
import re

from cfb_coach.madden.model import defense_semantics as ds

SOURCE = "ea_madden27_gameplay_deep_dive_2026"
# General responsibilities, not literal Madden 27 eleven-slot assignments.
SCHEMES: dict[str, dict[str, Any]] = {
    "cover0": {
        "family": "man_pressure", "shell": "zero-high",
        "template": ["Man coverage without guaranteed deep safety help",
                     "Pressure/rush distribution depends on exact call"],
        "caution": "A missed man assignment can expose a deep shot",
    },
    "cover1": {
        "family": "man", "shell": "single-high",
        "template": ["One deep middle safety as a general shell principle",
                     "Underneath man assignments; hole/robber help is variant-dependent"],
        "caution": "Do not abandon an assigned RB or TE in man coverage",
    },
    "cover2man": {
        "family": "man", "shell": "two-high",
        "template": ["Two deep halves as a general safety structure",
                     "Man coverage underneath"],
        "caution": "Low-hole help is not guaranteed unless the exact call provides it",
    },
    "cover2": {
        "family": "spot-zone", "shell": "two-high",
        "template": ["Two deep half-field zones as a general structure",
                     "Underneath flats and hook/curl areas (exact owner varies)"],
        "caution": "Middle/deep seam can be stressed by multiple vertical threats",
    },
    "tampa2": {
        "family": "spot-zone", "shell": "two-high",
        "template": ["Two deep halves",
                     "Tampa-style middle vertical carry added beneath the deep shell"],
        "caution": "Do not vacate the middle carry to chase a shallow route",
    },
    "cover3": {
        "family": "spot-or-match", "shell": "three-deep",
        "template": ["Three deep third responsibilities",
                     "Underneath curl/flat and hook areas; match rules vary by variant"],
        "caution": "Flood stresses one side; carry crossers only within assigned rules",
    },
    "cover4": {
        "family": "quarters-or-match", "shell": "four-deep",
        "template": ["Four quarters as a general two-high structure",
                     "Quarters/Palms can carry eligible receivers under matching rules"],
        "caution": "Run support, #2 verticals and Palms checks are call-dependent",
    },
    "cover6": {
        "family": "hybrid", "shell": "asymmetric split-field",
        "template": ["A quarters/half-field split as a general shell",
                     "The half/quarters side cannot be assigned without exact play art"],
        "caution": "Do not guess the quarters side from the name alone",
    },
    "cover9": {
        "family": "hybrid", "shell": "asymmetric split-field",
        "template": ["A mixed split-field shell; exact sides/rules are unverified"],
        "caution": "Do not claim the location of individual deep zones",
    },
    "pressure": {
        "family": "pressure", "shell": "unknown behind rush",
        "template": ["Pressure changes rushing/coverage numbers",
                     "Coverage behind the rush requires the actual defensive call"],
        "caution": "Do not user a rusher off his assigned rush lane",
    },
    "unknown": {
        "family": "unknown", "shell": "unknown",
        "template": ["No verified assignment template for this play"],
        "caution": "Play assigned responsibility; do not improvise a new shell",
    },
}
CONCEPT_LABELS = {
    "run": "run/inside zone", "rpo": "RPO", "cross": "crossers/mesh",
    "vert": "vertical seams", "flood": "flood/outside routes",
    "stack": "bunch/stack releases", "scram": "QB scramble",
}


def scheme_of(play: str, *, family: str | None = None) -> str:
    name = " ".join(str(play or "").lower().replace("-", " ").split())
    if "tampa 2" in name:
        return "tampa2"
    if re.search(r"\b(?:cover|cov)\s*2\s*man\b", name) or "2 man" in name:
        return "cover2man"
    if re.search(r"\b(?:cover|cov)\s*0\b", name):
        return "cover0"
    m = re.search(r"\b(?:cover|cov)\s*([1-4]|6|9)\b", name)
    if m:
        return "cover" + m.group(1)
    if family == "pressure":
        return "pressure"
    return "unknown"


def concept_knowledge(play: str, formation: str = "",
                      *, family: str | None = None) -> dict[str, Any]:
    """Preserve both true catalog axes and generic teaching templates."""
    semantic = ds.explain(play, formation)
    name = scheme_of(play, family=family)
    template = SCHEMES[name]
    variant = []
    lower = str(play).lower()
    if "robber" in lower or "hole" in lower:
        variant.append("low-hole/robber variant; exact user assignment needs play art")
    if "match" in lower or "palms" in lower or "quarters" in lower:
        variant.append("match rules depend on receiver releases; do not treat as static spot zones")
    if "buzz" in lower or "sky" in lower or "cloud" in lower:
        variant.append("rotation/flat responsibility may vary by named coverage")
    if "blitz" in lower or "fire" in lower or "sim" in lower:
        variant.append("rusher vs coverage identities require the exact diagram")
    return {
        "scheme": name, "concept_family": template["family"], "shell_template": template["shell"],
        "assignment_template": list(template["template"]),
        "variant_notes": variant, "caution": template["caution"],
        "named_defender_assignments_verified": False,
        "assignment_evidence": "scheme_template_not_play_art",
        "source": SOURCE, "play_catalog": semantic,
    }


def _context(sit: Any) -> str:
    if bool(getattr(sit, "goal_line", False)):
        return "goal_line"
    if bool(getattr(sit, "red_zone", False)):
        return "red_zone"
    try:
        if int(getattr(sit, "distance", 99)) <= 3:
            return "short"
    except (ValueError, TypeError):
        pass
    return "normal"


def user_guidance(*, play: str, formation: str, family: str,
                  belief: Mapping[str, float], sit: Any) -> dict[str, Any]:
    """Model-conditioned read sequence, not an instruction to abandon assignments."""
    structure = concept_knowledge(play, formation, family=family)
    scheme = structure["scheme"]
    prob = sorted(((c, float(p)) for c, p in belief.items()),
                  key=lambda x: (-x[1], x[0]))
    primary, confidence = prob[0] if prob else ("unknown", 0.0)
    # In a diffuse distribution, do not call a speculative specific matchup.
    trusted = confidence >= 0.40
    primary = primary if trusted else "unknown"
    early = _context(sit)

    if early in ("short", "goal_line"):
        first = "Fit the run/gap FIRST"
    elif scheme in ("cover1", "cover2man"):
        first = "Honor your man/underneath assignment FIRST"
    elif scheme in ("cover2", "tampa2"):
        first = "Hold your hook/middle seam assignment FIRST"
    elif scheme in ("cover4", "cover6", "cover9"):
        first = "Check run fit, then carry the assigned match FIRST"
    else:
        first = "Check run fit, then protect your underneath zone FIRST"

    reads = {
        "cross": "identify and carry the shallow crosser through YOUR assigned zone; avoid chasing it out of structure",
        "vert": "protect the seam/deep inside lane; do not bite on a drag",
        "flood": "recognize the flood-side high-low; take the underneath throw without abandoning deep help",
        "stack": "expect switches/rubs from bunch; communicate exchange, do not chase across other zones",
        "rpo": "read run action first, then close the quick slant/bubble within your responsibility",
        "run": "fit your assigned gap and attack the ballcarrier without giving up contain",
        "scram": "keep contain/spy responsibilities; plaster the nearest receiver only if your coverage allows",
        "unknown": "hold assignment integrity and react to the first real run/pass key",
    }
    priority = reads[primary]
    if scheme in ("cover1", "cover2man") and primary == "cross":
        priority = "stay attached to YOUR assigned receiver through the crosser; do not abandon man for another route"
    if "robber" in play.lower() or "hole" in play.lower():
        user_role = "User low-hole/robber defender IF shown by play art"
    elif scheme in ("cover1", "cover2man", "cover0"):
        user_role = "User a non-rushing LB IF available; respect his assigned man"
    elif scheme == "pressure":
        user_role = "User a non-rushing underneath defender IF available"
    elif scheme in ("cover2", "tampa2"):
        user_role = "User middle hook/seam defender IF shown by play art"
    else:
        user_role = "User an underneath hook-curl defender IF available"

    # Format defender read for existing compact single-user-job Madden UI:
    # no arrows or slashes (format_defense strips these separators).
    compact_focus = {
        "cross": "watch crossers after run key",
        "vert": "protect seams and verticals",
        "flood": "watch outside high-low",
        "stack": "watch bunch switch releases",
        "run": "attack your run fit first",
        "rpo": "run key first, then quick throw",
        "scram": "contain QB, then legal plaster",
        "unknown": "run key first, hold assignment",
    }[primary]
    text = user_role + "; " + compact_focus
    return {
        "role": user_role, "first_read": first, "second_read": priority,
        "avoid": structure["caution"], "predicted_concept": primary,
        "predicted_concept_probability": round(confidence, 3),
        "observed_pre_snap": False, "basis": "pre_snap_probability_and_scheme_template",
        "uncertain": not trusted, "compact": text,
        "coverage_structure": structure,
    }
