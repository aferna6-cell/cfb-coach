"""Coarse play result detection — never fabricate yards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RESULT_TYPES = (
    "completion",
    "incompletion",
    "int",
    "sack",
    "scramble",
    "run",
    "td",
    "turnover",
    "punt",
    "fg",
    "unknown",
)


@dataclass
class ResultGuess:
    result_type: str = "unknown"
    yards: int | None = None  # None unless HUD delta confident
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    scramble_dir: str | None = None  # L|R|unknown


def detect_result(
    *,
    hint: str | None = None,
    yards_delta: int | None = None,
    yards_conf: float = 0.0,
    score_us_delta: int | None = None,
    score_them_delta: int | None = None,
    family: str | None = None,
    motion_escape: bool = False,
    turnover_flag: bool = False,
    conf_threshold: float = 0.4,
) -> ResultGuess:
    """Infer result when possible. yards only if yards_conf high enough."""
    reasons: list[str] = []
    h = (hint or "").lower().strip()
    fam = (family or "").upper()
    rt = "unknown"
    conf = 0.0
    scramble_dir = None

    if turnover_flag or any(x in h for x in ("int", "pick", "interception")):
        rt, conf = "int" if "fumble" not in h else "turnover", 0.7
        reasons.append("turnover_signal")
    elif "fumble" in h or "turnover" in h:
        rt, conf = "turnover", 0.65
        reasons.append("turnover_word")
    elif "punt" in h:
        rt, conf = "punt", 0.7
        reasons.append("punt")
    elif "fg" in h or "field goal" in h:
        rt, conf = "fg", 0.7
        reasons.append("fg")
    elif "td" in h or "touchdown" in h or (score_us_delta is not None and score_us_delta >= 6):
        rt, conf = "td", 0.75
        reasons.append("td")
    elif "sack" in h or fam == "SACK":
        rt, conf = "sack", 0.7
        reasons.append("sack")
    elif "scram" in h or fam == "SCRAMBLE" or motion_escape:
        rt, conf = "scramble", 0.55
        reasons.append("scramble")
        if "left" in h or h.endswith(" l"):
            scramble_dir = "L"
        elif "right" in h or h.endswith(" r"):
            scramble_dir = "R"
        else:
            scramble_dir = "unknown"
    elif "incomplete" in h or "incomp" in h:
        rt, conf = "incompletion", 0.65
        reasons.append("incompletion")
    elif "complete" in h or "catch" in h:
        rt, conf = "completion", 0.6
        reasons.append("completion")
    elif "run" in h or fam in ("RUN_INSIDE", "RUN_OUTSIDE", "QB_RUN"):
        rt, conf = "run", 0.5
        reasons.append("run")
    elif fam in ("DROPBACK_PASS", "QUICK_PASS", "PLAY_ACTION", "SCREEN"):
        # family only — weak
        rt, conf = "unknown", 0.25
        reasons.append("family_only")

    yards: int | None = None
    if yards_delta is not None and yards_conf >= conf_threshold:
        yards = int(yards_delta)
        reasons.append("hud_yards")
    # Never fabricate: if low conf, yards stays None even if delta present
    elif yards_delta is not None:
        reasons.append("yards_low_conf_omitted")

    if conf < conf_threshold and rt not in ("td", "int", "punt", "fg", "sack"):
        # Keep weak guesses but mark low
        pass

    if rt not in RESULT_TYPES:
        rt = "unknown"

    return ResultGuess(
        result_type=rt,
        yards=yards,
        confidence=max(0.0, min(1.0, conf)),
        reasons=reasons,
        scramble_dir=scramble_dir,
    )


def result_to_snap_text(g: ResultGuess) -> str:
    """Compact result string for legacy snap logging."""
    if g.result_type == "unknown":
        return ""
    if g.yards is not None:
        sign = "+" if g.yards >= 0 else ""
        return f"{g.result_type} {sign}{g.yards}"
    return g.result_type
