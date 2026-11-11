"""Opponent offensive concept classes the defensive coordinator reasons about.

Raw logged labels (e.g. "Four Verticals", "Hi Lo Cross", "RPO bubble") map to 8 classes. Unknown stays
unknown (None); nothing is guessed. Each class maps to the repo's football_knowledge concept ids so the
sourced coverage-vs-concept prior table (football_priors) can be read from the defensive side.
"""
from __future__ import annotations

import re

CLASSES = ("run", "rpo", "screen", "quick", "mesh", "crossers", "flood", "verticals")
PASS_CLASSES = frozenset(CLASSES) - {"run", "rpo"}
_RULES = (  # first match wins; order matters (screen before run, mesh before cross)
    ("screen", r"screen|bubble screen|\bslip\b|jailbreak"),
    ("rpo", r"\brpo\b|read option|option|bubble"),
    ("mesh", r"\bmesh\b|shallow|drive\b"),
    ("verticals", r"vert|seam|\bgo\b|streak|fade|wheel|post wheel|y-stutter|stutter|corner post|\bcorner\b|\bpost\b"),
    ("quick", r"slant|hitch|quick|stick|spot|curl|snag|angle|texas"),
    ("flood", r"flood|sail|fork|bench|levels|smash|slot out|\bout\b"),
    ("crossers", r"cross|drag|\bins\b|dig|over\b|hi lo|dbl ins|deep in"),
    ("quick", r"flat"),
    ("run", r"zone|stretch|dive|power|counter|trap|toss|sweep|iso|duo|draw|sneak|\bhb\b|\bqb run|blast|run"),
)
FP_CONCEPTS = {  # class -> {football_knowledge concept id: weight}
    "run": {"inside_zone": 1.0},
    "rpo": {"inside_zone": 0.5, "slant_flat": 0.5},
    "screen": {"screen": 1.0},
    "quick": {"slant_flat": 0.5, "stick": 0.5},
    "mesh": {"mesh": 1.0},
    "crossers": {"deep_cross": 0.6, "drive": 0.4},
    "flood": {"flood_sail": 1.0},
    "verticals": {"four_verts": 1.0},
}


def classify(raw: str | None) -> str | None:
    t = (raw or "").strip().lower()
    if not t:
        return None
    for cls, pat in _RULES:
        if re.search(pat, t):
            return cls
    from cfb_coach.madden.situation import concept_family
    fam = concept_family(raw)
    return {"run": "run", "rpo": "rpo", "vert": "verticals", "flood": "flood", "cross": "crossers"}.get(fam or "")
