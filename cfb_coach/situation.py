"""Parse live situation shorthand into structured fields."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Situation:
    raw: str
    side: str = "offense"  # offense | defense  (OUR ball side)
    down: int | None = None
    distance: int | None = None
    yardline: int | None = None  # yards from our goal (approx) or absolute
    red_zone: bool = False
    goal_line: bool = False
    two_minute: bool = False
    short_yardage: bool = False
    long_yardage: bool = False
    coverage_hint: str | None = None
    # How coverage entered the situation:
    #   live = pre-snap look this snap; last = previous-snap observation;
    #   none = no coverage signal
    coverage_source: str = "none"  # live | last | none
    concept_hint: str | None = None
    notes: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        parts = []
        if self.down and self.distance is not None:
            parts.append(f"{self.down}&{self.distance}")
        if self.red_zone:
            parts.append("RZ")
        if self.goal_line:
            parts.append("GL")
        if self.two_minute:
            parts.append("2min")
        if self.coverage_hint:
            parts.append(self.coverage_hint)
        if self.concept_hint:
            parts.append(self.concept_hint)
        return " ".join(parts) or self.raw


_DOWN_RE = re.compile(
    r"\b([1-4])(?:st|nd|rd|th)?\s*[&and]+\s*(\d+|inches|goal|inches?)\b",
    re.I,
)
_DOWN_COMPACT = re.compile(r"\b([1-4])&(\d+|g|goal)\b", re.I)
_YL_RE = re.compile(r"\b(?:yl|yardline|on)\s*(\d{1,2})\b", re.I)
_RZ_RE = re.compile(r"\b(rz|red\s*zone|inside\s*the\s*20)\b", re.I)
_GL_RE = re.compile(r"\b(gl|goal\s*line|from\s*the\s*[1-3]|&goal)\b", re.I)
_2MIN_RE = re.compile(r"\b(2\s*min|two\s*minute|hurry|no\s*huddle|tempo)\b", re.I)

_SIDE_O = re.compile(r"\b(o|off|offense|our\s*ball|we\s*have\s*ball)\b", re.I)
_SIDE_D = re.compile(r"\b(d|def|defense|their\s*ball|opp)\b", re.I)

_COV_HINTS = [
    (re.compile(r"\bc6\b|cover\s*6", re.I), "Cover 6"),
    (re.compile(r"\bc9\b|cover\s*9", re.I), "Cover 9"),
    (re.compile(r"\bc4\b|quarters|cover\s*4", re.I), "Cover 4 Quarters"),
    (re.compile(r"palms", re.I), "Cover 4 Palms"),
    (re.compile(r"tampa|tamp\s*2|t2\b", re.I), "Tampa 2"),
    (re.compile(r"\bc3\b|cover\s*3|sky", re.I), "Cover 3 Sky"),
    # Invert before generic C2 so "Cover 2 Invert Hard Flat" matches correctly
    (
        re.compile(
            r"c2\s*invert|cover\s*2\s*invert|invert\s*hard\s*flat|invert",
            re.I,
        ),
        "Cover 2 Invert",
    ),
    (re.compile(r"\bc2\b|cover\s*2", re.I), "Cover 2"),
    (re.compile(r"\bc1\b|cover\s*1|man\b", re.I), "Cover 1"),
    (re.compile(r"\bc0\b|cover\s*0|zero", re.I), "Cover 0"),
    (re.compile(r"two[\s-]*high|split[\s-]*field", re.I), "two-high"),
    (re.compile(r"blitz|pressure|heat", re.I), "pressure"),
]

_CONCEPT_HINTS = [
    (re.compile(r"4\s*verts?|four\s*vert|verticals", re.I), "Four Verticals"),
    (re.compile(r"cross\s*wheels?|crossers?", re.I), "Cross Wheels"),
    (re.compile(r"\bmesh\b", re.I), "Mesh"),
    (re.compile(r"\brpo\b|bubble", re.I), "RPO bubble"),
    (re.compile(r"inside\s*zone|\biz\b|duo", re.I), "Inside Zone"),
    (re.compile(r"scram|scramble|qb\s*run", re.I), "QB scramble"),
    (re.compile(r"flood", re.I), "Flood"),
    (re.compile(r"bunch|cluster", re.I), "Bunch/Cluster"),
]


def parse_situation(raw: str, default_side: str = "offense") -> Situation:
    text = raw.strip()
    sit = Situation(raw=text, side=default_side)

    if _SIDE_D.search(text) and not _SIDE_O.search(text):
        sit.side = "defense"
    elif _SIDE_O.search(text):
        sit.side = "offense"

    m = _DOWN_COMPACT.search(text) or _DOWN_RE.search(text)
    if m:
        sit.down = int(m.group(1))
        dist_raw = m.group(2).lower()
        if dist_raw in ("g", "goal", "inches", "inch"):
            sit.distance = 1
            sit.goal_line = True
            sit.short_yardage = True
        else:
            sit.distance = int(dist_raw)

    yl = _YL_RE.search(text)
    if yl:
        sit.yardline = int(yl.group(1))
        if sit.yardline <= 20:
            sit.red_zone = True
        if sit.yardline <= 5:
            sit.goal_line = True

    if _RZ_RE.search(text):
        sit.red_zone = True
    if _GL_RE.search(text):
        sit.goal_line = True
        sit.red_zone = True
    if _2MIN_RE.search(text):
        sit.two_minute = True

    if sit.distance is not None:
        if sit.distance <= 3:
            sit.short_yardage = True
        # 1st&10 is normal, not "long"; long = late-down distance stress
        if sit.down in (2, 3, 4) and sit.distance >= 8:
            sit.long_yardage = True
        elif sit.down == 1 and sit.distance >= 15:
            sit.long_yardage = True

    last_snap = bool(
        re.search(
            r"\b(last|prev|previous|saw|showed|was)\b",
            text,
            re.I,
        )
    )
    live_mark = bool(
        re.search(
            r"\b(showing|pre[- ]?snap|live|they.?re\s+in|aligned)\b",
            text,
            re.I,
        )
    )

    for rx, label in _COV_HINTS:
        if rx.search(text):
            sit.coverage_hint = label
            if last_snap and not live_mark:
                sit.coverage_source = "last"
            else:
                # Bare coverage in the sit string = soft live look, still not a hard-counter
                sit.coverage_source = "live"
            break

    for rx, label in _CONCEPT_HINTS:
        if rx.search(text):
            sit.concept_hint = label
            break

    return sit
