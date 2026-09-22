"""Play family + optional concept tags (confidence-aware, coarse)."""

from __future__ import annotations

from dataclasses import dataclass, field


PLAY_FAMILIES = (
    "RUN_INSIDE",
    "RUN_OUTSIDE",
    "RPO",
    "SCREEN",
    "QUICK_PASS",
    "DROPBACK_PASS",
    "PLAY_ACTION",
    "QB_RUN",
    "SCRAMBLE",
    "SACK",
    "OTHER",
    "UNKNOWN",
)

CONCEPT_TAGS = (
    "CROSSERS",
    "VERTICALS",
    "FLOOD",
    "MESH",
    "SPACING",
    "SLANTS",
    "SCREEN",
    "DRAW",
    "ZONE",
    "POWER",
    "COUNTER",
    "BOOT",
    "UNKNOWN",
)


@dataclass
class FamilyGuess:
    family: str = "UNKNOWN"
    concept_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


def classify_family(
    *,
    hint: str | None = None,
    formation: str | None = None,
    result_type: str | None = None,
    motion_profile: str | None = None,
    yards: int | None = None,
    confidence: float = 0.5,
) -> FamilyGuess:
    """Coarse family from text hints / result — never overclaim.

    Pure heuristic; CV route grading is out of scope.
    """
    h = (hint or "").lower().strip()
    r = (result_type or "").lower().strip()
    form = (formation or "").lower().strip()
    tags: list[str] = []
    family = "UNKNOWN"
    conf = max(0.0, min(1.0, confidence))
    notes = ""

    # Result-driven hard families
    if r in ("sack",):
        return FamilyGuess("SACK", [], min(0.9, conf + 0.2), "result=sack")
    if r in ("scramble",):
        return FamilyGuess("SCRAMBLE", [], min(0.85, conf + 0.15), "result=scramble")
    if r in ("int", "interception", "turnover"):
        family = "DROPBACK_PASS"
        notes = "turnover on pass look"

    # Lexical hint mapping
    rules: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = [
        (("inside zone", "iz", "duo", "dive", "iso", "power", "counter y", "hb base"), "RUN_INSIDE", ("ZONE", "POWER", "COUNTER")),
        (("outside zone", "stretch", "toss", "sweep", "hb stretch"), "RUN_OUTSIDE", ("ZONE",)),
        (("rpo", "alert", "bubble"), "RPO", ()),
        (("screen", "slip screen", "wr screen"), "SCREEN", ("SCREEN",)),
        (("slant", "quick", "hitch", "stick", "spacing"), "QUICK_PASS", ("SLANTS", "SPACING")),
        (("mesh",), "DROPBACK_PASS", ("MESH",)),
        (("flood", "smash"), "DROPBACK_PASS", ("FLOOD",)),
        (("vertical", "all go", "four verts", "seam"), "DROPBACK_PASS", ("VERTICALS",)),
        (("cross", "crosser", "drag", "shallow"), "DROPBACK_PASS", ("CROSSERS",)),
        (("pa ", "play action", "play-action", "boot", "naked"), "PLAY_ACTION", ("BOOT",)),
        (("qb run", "qb sweep", "qb blast", "qb draw", "scramble designed"), "QB_RUN", ()),
        (("scram", "escape"), "SCRAMBLE", ()),
        (("sack",), "SACK", ()),
        (("draw",), "RUN_INSIDE", ("DRAW",)),
    ]

    for keys, fam, tag_opts in rules:
        if any(k in h for k in keys):
            family = fam
            for t in tag_opts:
                tl = t.lower()
                if tl in h or (t == "ZONE" and "zone" in h) or (t == "MESH" and "mesh" in h):
                    tags.append(t)
                elif t in ("CROSSERS", "VERTICALS", "FLOOD", "MESH", "SLANTS", "SPACING", "SCREEN", "BOOT", "DRAW") and any(
                    x in h for x in (t.lower().rstrip("s"), t.lower())
                ):
                    tags.append(t)
            # Ensure concept tag from key match
            if "cross" in h and "CROSSERS" not in tags:
                tags.append("CROSSERS")
            if "vert" in h and "VERTICALS" not in tags:
                tags.append("VERTICALS")
            if "flood" in h and "FLOOD" not in tags:
                tags.append("FLOOD")
            if "mesh" in h and "MESH" not in tags:
                tags.append("MESH")
            notes = f"hint→{fam}"
            break

    if family == "UNKNOWN" and r in ("run", "rush"):
        family = "RUN_INSIDE"
        conf = min(conf, 0.45)
        notes = "result=run coarse"
    if family == "UNKNOWN" and r in ("completion", "incompletion", "pass"):
        family = "DROPBACK_PASS"
        conf = min(conf, 0.4)
        notes = "result=pass coarse"

    if family == "UNKNOWN" and form in ("empty",) and conf >= 0.3:
        family = "DROPBACK_PASS"
        conf = min(conf, 0.35)
        notes = "empty formation prior"

    # Dedupe tags / validate
    out_tags = []
    for t in tags:
        tu = t.upper()
        if tu in CONCEPT_TAGS and tu not in out_tags:
            out_tags.append(tu)
    if family not in PLAY_FAMILIES:
        family = "UNKNOWN"
    if family == "UNKNOWN":
        conf = min(conf, 0.25)

    return FamilyGuess(family=family, concept_tags=out_tags, confidence=conf, notes=notes)


def family_to_concept_hint(family: str, tags: list[str] | None = None) -> str:
    """Bridge to existing tendency concept strings."""
    tags = tags or []
    if "CROSSERS" in tags:
        return "crossers"
    if "VERTICALS" in tags:
        return "verticals"
    if "FLOOD" in tags:
        return "flood"
    if "MESH" in tags:
        return "mesh"
    m = {
        "RUN_INSIDE": "inside zone",
        "RUN_OUTSIDE": "outside zone",
        "RPO": "rpo",
        "SCREEN": "screen",
        "QUICK_PASS": "quick pass",
        "DROPBACK_PASS": "dropback",
        "PLAY_ACTION": "play action",
        "QB_RUN": "qb run",
        "SCRAMBLE": "scramble",
        "SACK": "sack",
    }
    return m.get(family, family.lower() if family else "")
