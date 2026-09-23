"""Madden 27 situation parsing — shared D&D/yardline parser + Madden book names.

Down/distance, yardlines, prev-vs-live markers and `heard:` formatting are the
shared CFB parser. Only the concept/coverage vocabulary is Madden-specific, so
CFB-only play names (Mesh Spot, Deep Flood, …) never leak into Madden hints.
"""

from __future__ import annotations

import re

from cfb_coach.situation import Situation, format_heard, parse_situation

__all__ = ["Situation", "format_heard", "parse_madden_situation", "concept_family"]

# Specific before generic
_CONCEPT_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"texas|y[\s-]*stutter|stutter\s*wheel", re.I), "Texas Y-Stutter Wheel"),
    (re.compile(r"(?:mtn\s*|motion\s*)?shuffle\s*verts?\s*smash|verts?\s*smash", re.I), "Mtn Shuffle Verts Smash"),
    (re.compile(r"mesh\s*rail", re.I), "Mtn Shuffle Mesh Rail"),
    (re.compile(r"mesh\s*post", re.I), "Mesh Post"),
    (re.compile(r"clear\s*deep", re.I), "Clear Deep"),
    (re.compile(r"return\s*bench|\bbench\b", re.I), "Return Bench"),
    (re.compile(r"(?:mtn\s*)?stick\s*wheel", re.I), "Stick Wheel"),
    (re.compile(r"(?:switch\s*)?hb\s*wheel", re.I), "Switch HB Wheel"),
    (re.compile(r"y\s*option(?:\s*wheel)?", re.I), "Y Option Wheel"),
    (re.compile(r"(?:mtn\s*)?double\s*posts?", re.I), "Double Post"),
    (re.compile(r"(?:mtn\s*)?(?:fork\s*)?salem", re.I), "Fork Salem"),
    (re.compile(r"(?:mtn\s*)?(?:post\s*)?swirl", re.I), "Post Swirl"),
    (re.compile(r"hi[\s-]*lo|high[\s-]*low", re.I), "Hi Lo Cross"),
    (re.compile(r"tight\s*cross(?:es)?", re.I), "Tight Crosses"),
    (re.compile(r"triple\s*slants?", re.I), "Mtn Triple Slants"),
    (re.compile(r"ohio(?:\s*return)?|return\s*routes?", re.I), "Y Ohio Return"),
    (re.compile(r"\bflood\b|\bsail\b", re.I), "Flood"),
    (re.compile(r"\bmesh\b", re.I), "Mesh"),
    (re.compile(r"4\s*verts?|four\s*vert(?:ical)?s?|\bverticals\b|\bverts\b|\bseams?\b", re.I), "Four Verticals"),
    (re.compile(r"\bsmash\b", re.I), "Smash"),
    (re.compile(r"\bslants?\b", re.I), "Slants"),
    (re.compile(r"cross(?:ers?|ing)?\b|\bdrags?\b", re.I), "Crossers"),
    (re.compile(r"\bstick\b", re.I), "Stick"),
    (re.compile(r"\bwheel\b", re.I), "Wheel"),
    (re.compile(r"clamp|\bstack\b|\bbunch\b", re.I), "Clamp Stack / Bunch"),
    (re.compile(r"\brpo\b|bubble", re.I), "RPO bubble"),
    (re.compile(r"\bscreen\b", re.I), "Screen"),
    (re.compile(r"scram(?:ble)?|qb\s*run|qb\s*draw", re.I), "QB scramble"),
    (re.compile(r"same\s*side\s*zone", re.I), "Same Side Zone"),
    (re.compile(r"mid\s*zone", re.I), "Mid Zone"),
    (re.compile(r"hb\s*dive|\bdive\b", re.I), "HB Dive"),
    (re.compile(r"inside\s*zone|hb\s*zone|\biz\b|\bduo\b", re.I), "Inside Zone"),
    (re.compile(r"stretch|outside\s*zone|\boz\b|\btoss\b", re.I), "Stretch"),
]

# Madden coverage names the shared table would mislabel (checked first)
_COV_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"buzz\s*match|c3\s*match|cover\s*3\s*match|\bmatch\b", re.I), "Cover 3 Match"),
    (re.compile(r"c2\s*man|cover\s*2\s*man|2\s*man", re.I), "Cover 2 Man"),
    (re.compile(r"cover\s*1\s*(?:hole|robber)|\bhole\b|robber", re.I), "Cover 1"),
    (re.compile(r"sim\s*[23]|\bsim\b|\bstunt|\bloop\b|db\s*fire|\bmug\b|\bfire\b", re.I), "pressure"),
    (re.compile(r"2\s*trap|nickel\s*trap", re.I), "Cover 2"),
]

_OWN_YL = re.compile(r"\b(?:my|our|own)\s*(?:yl|yard\s*line)?\s*\d{1,2}\b", re.I)
_EXPLICIT_RZ = re.compile(r"\b(rz|red\s*zone|gl|goal\s*line|&\s*goal|[1-4]&g(?:oal)?)\b", re.I)


def concept_family(concept: str | None) -> str | None:
    """Madden concept → macro family (vert / flood / cross / stack / scram / run / rpo)."""
    c = (concept or "").lower()
    if not c:
        return None
    if any(k in c for k in ("stick", "flood", "sail", "salem", "bench")):
        return "flood"
    if any(k in c for k in ("vert", "double post", "wheel", "seam", "clear deep")):
        return "vert"
    if any(k in c for k in ("mesh", "cross", "ohio", "return", "swirl", "slant", "hi lo", "option")):
        return "cross"
    if "clamp" in c or "stack" in c or "bunch" in c:
        return "stack"
    if "scram" in c:
        return "scram"
    if "rpo" in c or "bubble" in c:
        return "rpo"
    if any(k in c for k in ("zone", "stretch", "dive")):
        return "run"
    return None


def parse_madden_situation(raw: str, default_side: str = "offense") -> Situation:
    sit = parse_situation(raw, default_side=default_side)
    text = sit.raw
    live = sit.coverage_source == "live" or sit.concept_source == "live" or bool(
        re.search(r"\b(showing|pre[- ]?snap|live|they.?re\s+in|aligned)\b", text, re.I)
    )
    src = "live" if live else "last"

    for rx, label in _COV_HINTS:
        if rx.search(text):
            sit.coverage_hint = label
            sit.coverage_source = src
            break

    # Madden vocabulary replaces any CFB-book concept name
    sit.concept_hint = None
    sit.concept_source = "none"
    for rx, label in _CONCEPT_HINTS:
        if rx.search(text):
            sit.concept_hint = label
            sit.concept_source = src
            break

    # Own-territory yardline (my/our 15) is backed up, not red zone / goal line
    if _OWN_YL.search(text) and not _EXPLICIT_RZ.search(text):
        sit.red_zone = False
        sit.goal_line = False
    return sit
