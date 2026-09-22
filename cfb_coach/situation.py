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
    # Parallel to coverage_source for offense/defense play-name tells
    concept_source: str = "none"  # live | last | none
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
# Bare "opp" / "their 25" are yardlines — only treat opp/their as defense with "ball"
_SIDE_D = re.compile(
    r"\b(d|def|defense|their\s*ball|opp(?:onent)?\s*ball|(?:opp(?:onent)?)(?!\s*\d))\b",
    re.I,
)

# Yardline phrases: my/our N = from own goal; opp/their N = opponent's N → 100-N
_YL_OWN_RE = re.compile(
    r"\b(?:my|our|own)\s*(?:yl|yard\s*line)?\s*(\d{1,2})\b",
    re.I,
)
_YL_OPP_RE = re.compile(
    r"\b(?:opp(?:onent)?s?|their|his|her)\s*(?:yl|yard\s*line)?\s*(\d{1,2})\b",
    re.I,
)
_YL_BALL_ON_RE = re.compile(
    r"\bball\s+on\s+(\d{1,2})\b",
    re.I,
)

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

# Book / seed play-name aliases → concept_hint (specific before generic)
_CONCEPT_HINTS = [
    # Mesh family
    (re.compile(r"mesh\s*spot", re.I), "Mesh Spot"),
    (re.compile(r"mesh\s*traffic", re.I), "Mesh Traffic"),
    (re.compile(r"mesh\s*corner", re.I), "Mesh Corner"),
    (re.compile(r"mesh\s*post", re.I), "Mesh Post"),
    (re.compile(r"irish\s*mesh(?:\s*whip)?", re.I), "Irish Mesh Whip"),
    (re.compile(r"\bmesh\b", re.I), "Mesh"),
    # Flood / verticals / cross
    (re.compile(r"deep\s*flood|flood\s*drive", re.I), "Deep Flood"),
    (re.compile(r"\bflood\b", re.I), "Flood"),
    (re.compile(r"4\s*verts?|four\s*vert(?:ical)?s?|\bverticals\b|\bverts\b", re.I), "Four Verticals"),
    (re.compile(r"cross\s*wheels?|crossers?|cross\s*z\s*post", re.I), "Cross Wheels"),
    # Whip / spot
    (re.compile(r"return\s*whip(?:\s*trail)?|whip\s*trail", re.I), "Return Whip Trail"),
    (re.compile(r"whip\s*double(?:\s*spot)?", re.I), "Whip Double Spot"),
    (re.compile(r"rz\s*pa\s*x\s*whip|pa\s*x\s*whip", re.I), "RZ PA X Whip"),
    (re.compile(r"z\s*spot\s*shake", re.I), "Z Spot Shake"),
    (re.compile(r"z\s*spot(?:\s*goal\s*line)?", re.I), "Z Spot"),
    # RPO / mountain
    (re.compile(r"mtn\s*rpo(?:\s*zone\s*alert)?|mountain\s*rpo|rpo\s*zone\s*alert", re.I), "Mtn RPO Zone Alert"),
    (re.compile(r"mtn\s*cross\s*post", re.I), "Mtn Cross Post"),
    (re.compile(r"mtn\s*hb\s*choice", re.I), "Mtn HB Choice"),
    (re.compile(r"mtn\s*speed\s*dig", re.I), "Mtn Speed Dig Under"),
    (re.compile(r"mtn\s*duo|mountain\s*duo", re.I), "Mtn Duo"),
    (re.compile(r"\brpo\b|bubble", re.I), "RPO bubble"),
    # Run game
    (re.compile(r"inside\s*zone(?:\s*split)?|\biz\b|\bduo\b", re.I), "Inside Zone"),
    (re.compile(r"outside\s*zone|\boz\b|stretch", re.I), "Outside Zone"),
    (re.compile(r"hb\s*base", re.I), "HB Base"),
    (re.compile(r"counter\s*y|\bhb\s*counter\b", re.I), "Counter Y"),
    (re.compile(r"hb\s*dive", re.I), "HB Dive"),
    (re.compile(r"hb\s*mid\s*draw|qb\s*draw", re.I), "HB Mid Draw"),
    (re.compile(r"drive\s*hb\s*under", re.I), "Drive HB Under"),
    (re.compile(r"qb\s*sweep|qb\s*run|scram(?:ble)?", re.I), "QB scramble"),
    # Misc book names
    (re.compile(r"quick\s*slants?", re.I), "Quick Slants"),
    (re.compile(r"\bspacing\b", re.I), "Spacing"),
    (re.compile(r"bunch|cluster", re.I), "Bunch/Cluster"),
]


def format_heard(sit: Situation) -> str:
    """One-line echo of what coach understood from sit> (for Aidan + overlay)."""
    parts: list[str] = []
    if sit.down and sit.distance is not None:
        parts.append(f"{sit.down}&{sit.distance}")
    if sit.yardline is not None:
        parts.append(f"yl{sit.yardline}")
    if sit.red_zone:
        parts.append("RZ")
    if sit.goal_line:
        parts.append("GL")
    if sit.two_minute:
        parts.append("2min")
    if sit.coverage_hint:
        src = sit.coverage_source or "none"
        if src == "last":
            parts.append(f"[prev:{sit.coverage_hint}]")
        elif src == "live":
            parts.append(f"[live:{sit.coverage_hint}]")
        else:
            parts.append(f"[{sit.coverage_hint}]")
    if sit.concept_hint:
        src = getattr(sit, "concept_source", "none") or "none"
        if src == "last":
            parts.append(f"[prev:{sit.concept_hint}]")
        elif src == "live":
            parts.append(f"[live:{sit.concept_hint}]")
        else:
            parts.append(sit.concept_hint)
    body = " ".join(parts) if parts else (sit.raw or "?")
    return f"heard: {body}"


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

    # Prefer explicit own/opp phrases over bare yl/on
    yl_own = _YL_OWN_RE.search(text)
    yl_opp = _YL_OPP_RE.search(text)
    yl_ball = _YL_BALL_ON_RE.search(text)
    yl = _YL_RE.search(text)
    if yl_own:
        sit.yardline = int(yl_own.group(1))
        if sit.yardline <= 20:
            sit.red_zone = True
        if sit.yardline <= 5:
            sit.goal_line = True
    elif yl_opp:
        opp_yl = int(yl_opp.group(1))
        sit.yardline = max(1, min(99, 100 - opp_yl))  # yards from our goal
        if opp_yl <= 20:
            sit.red_zone = True
        if opp_yl <= 5:
            sit.goal_line = True
    elif yl_ball:
        sit.yardline = int(yl_ball.group(1))
        if sit.yardline <= 20:
            sit.red_zone = True
        if sit.yardline <= 5:
            sit.goal_line = True
    elif yl:
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

    # Explicit last markers still count as prev; only live markers force live.
    # Aidan UX (CPU offense): bare play name / coverage = previous snap — no need to say "last".
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
            sit.coverage_source = "live" if live_mark else "last"
            break

    for rx, label in _CONCEPT_HINTS:
        if rx.search(text):
            sit.concept_hint = label
            sit.concept_source = "live" if live_mark else "last"
            break

    return sit
