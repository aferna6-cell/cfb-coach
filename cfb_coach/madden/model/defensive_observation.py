"""Structured defensive observation. Fields are independent.

A snap can be Cover 1 and a blitz at the same time. An ambiguous legacy
label such as "man" stays ambiguous: it is not upgraded to Cover 1, a
front, or a box count. Pre-snap, post-snap, historical, and unknown
timings stay separate.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

OBSERVATION_VERSION = "defensive_observation.v1"
TIMINGS = frozenset({"pre_snap", "post_snap", "historical", "unknown"})
SOURCES = frozenset({
    "live", "historical", "legacy_label", "video", "manual", "unknown",
})

_COVERS = (
    ("cover_0", re.compile(r"cover\s*0|\bzero\b", re.I)),
    ("cover_1", re.compile(r"cover\s*1", re.I)),
    ("cover_6", re.compile(r"cover\s*6", re.I)),
    ("cover_4", re.compile(r"cover\s*4|\bquarters\b|\bpalms\b", re.I)),
    ("cover_2", re.compile(r"cover\s*2|\btampa\b", re.I)),
    ("cover_3", re.compile(r"cover\s*3", re.I)),
)
_MAN = re.compile(r"\bman\b", re.I)
_ZONE = re.compile(r"\bzone\b", re.I)
_MATCH = re.compile(r"\bmatch\b|pattern[\s-]*match|zone[\s-]*match", re.I)
_TWO_HIGH = re.compile(r"two[\s-]*high|split[\s-]*field", re.I)
_SINGLE_HIGH = re.compile(r"single[\s-]*high|\bmofc\b", re.I)
_PRESS = re.compile(r"\bpress\b", re.I)
_OFF = re.compile(r"\boff\b|\bbail\b", re.I)
_BLITZ = re.compile(r"\bblitz\b|\bfire\b|\bmug\b", re.I)
_PRESSURE = re.compile(r"pressure|\bblitz\b|\bfire\b|\bmug\b", re.I)
_BOX = re.compile(r"(\d)\s*man\s*box|\bbox\s*(\d)", re.I)
_INSIDE = re.compile(r"inside\s+leverage|\bleverage\s+inside\b", re.I)
_OUTSIDE = re.compile(r"outside\s+leverage|\bleverage\s+outside\b", re.I)
_FRONT = re.compile(r"\b(4-3|3-4|nickel|dime|bear|odd|even)\b", re.I)
_AMBIGUOUS = re.compile(r"^(man|zone|pressure|blitz|two-high|two high|match)$", re.I)


def _shells(text: str) -> list[str]:
    found: list[str] = []
    two_high_only = bool(_TWO_HIGH.search(text)) and not re.search(r"cover\s*[0-9]", text, re.I)
    for name, pattern in _COVERS:
        if not pattern.search(text):
            continue
        if name == "cover_2" and two_high_only:
            continue
        found.append(name)
    return found


def _box(text: str) -> int | None:
    match = _BOX.search(text or "")
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 5 <= value <= 9 else None


def structure_observation(
    text: str | None = None,
    *,
    timing: str = "unknown",
    source: str = "legacy_label",
    confidence: float | None = None,
    fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse one observation. Missing facts stay null. Nothing is invented."""
    raw = (text or "").strip()
    supplied = dict(fields or {})
    shells = _shells(raw) if raw else []
    shell = shells[0] if len(shells) == 1 else None
    if supplied.get("coverage_shell"):
        shell = str(supplied["coverage_shell"])
        shells = [shell]
    contradictory = len(shells) > 1
    man = True if _MAN.search(raw) else None
    zone = True if _ZONE.search(raw) else None
    match = True if _MATCH.search(raw) else None
    if shell == "cover_1":
        man = True
    if shell in ("cover_2", "cover_3", "cover_4", "cover_6"):
        zone = True
    if shell == "cover_0":
        man = True if man is None else man
    no_pressure = bool(re.search(r"\bno\s+pressure\b|\bno\s+blitz\b", raw, re.I))
    blitz = False if re.search(r"\bno\s+blitz\b", raw, re.I) else (True if _BLITZ.search(raw) else None)
    if no_pressure:
        pressure = False
    elif _PRESSURE.search(raw) or shell == "cover_0" or blitz:
        pressure = True
    else:
        pressure = None
    front_match = _FRONT.search(raw)
    front = front_match.group(1).lower() if front_match else None
    if _INSIDE.search(raw):
        leverage = "inside"
    elif _OUTSIDE.search(raw):
        leverage = "outside"
    else:
        leverage = None
    if _TWO_HIGH.search(raw):
        safety = "two_high"
    elif _SINGLE_HIGH.search(raw) or shell in ("cover_0", "cover_1", "cover_3"):
        safety = "single_high"
    elif shell in ("cover_2", "cover_4", "cover_6"):
        safety = "two_high"
    else:
        safety = None
    if _PRESS.search(raw):
        press = True
    elif _OFF.search(raw):
        press = False
    else:
        press = None
    family = None
    if man and not zone and not match:
        family = "man"
    elif zone and not man:
        family = "zone"
    elif match and not man:
        family = "match"
    elif man and zone:
        family = "mixed"
    legacy_ambiguous = bool(raw) and shell is None and bool(_AMBIGUOUS.match(raw))
    if legacy_ambiguous:
        # The word is evidence of a family or pressure, not a verified structure.
        shell = None
        front = None
        safety = safety if raw.lower().startswith("two") else None
    for key in (
        "man_evidence", "zone_evidence", "match_evidence", "pressure", "blitz",
        "front", "box_count", "safety_depth", "press", "leverage", "coverage_family",
    ):
        if key in supplied and supplied[key] is not None:
            if key == "man_evidence":
                man = bool(supplied[key])
            elif key == "zone_evidence":
                zone = bool(supplied[key])
            elif key == "match_evidence":
                match = bool(supplied[key])
            elif key == "pressure":
                pressure = bool(supplied[key])
            elif key == "blitz":
                blitz = bool(supplied[key])
            elif key == "front":
                front = supplied[key]
            elif key == "box_count":
                box = supplied[key]
            elif key == "safety_depth":
                safety = supplied[key]
            elif key == "press":
                press = supplied[key]
            elif key == "leverage":
                leverage = supplied[key]
            elif key == "coverage_family":
                family = supplied[key]
    box = supplied.get("box_count", _box(raw) if "box_count" not in supplied else supplied.get("box_count"))
    if "box_count" not in supplied:
        box = _box(raw)
    unknown = [
        name for name, value in (
            ("coverage_shell", shell),
            ("coverage_family", family),
            ("man_evidence", man),
            ("zone_evidence", zone),
            ("match_evidence", match),
            ("pressure", pressure),
            ("blitz", blitz),
            ("front", front),
            ("box_count", box),
            ("safety_depth", safety),
            ("press", press),
            ("leverage", leverage),
        ) if value is None
    ]
    if contradictory:
        unknown.append("coverage_shell_contradictory")
        shell = None
    timing_value = timing if timing in TIMINGS else "unknown"
    source_value = source if source in SOURCES else "unknown"
    if confidence is None:
        if shell and not legacy_ambiguous:
            confidence = 0.8
        elif raw and not legacy_ambiguous:
            confidence = 0.55
        elif legacy_ambiguous:
            confidence = 0.35
        else:
            confidence = 0.0
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = 0.0
    if legacy_ambiguous:
        confidence_value = min(confidence_value, 0.4)
    return {
        "version": OBSERVATION_VERSION,
        "coverage_shell": shell,
        "coverage_family": family,
        "man_evidence": man,
        "zone_evidence": zone,
        "match_evidence": match,
        "pressure": pressure,
        "blitz": blitz,
        "front": front,
        "box_count": box,
        "safety_depth": safety,
        "press": press,
        "leverage": leverage,
        "timing": timing_value,
        "source": source_value,
        "confidence": round(confidence_value, 3),
        "legacy_ambiguous": legacy_ambiguous,
        "upgraded_from_ambiguous": False,
        "contradictory_shells": contradictory,
        "shells_mentioned": shells,
        "unknown": unknown,
        "raw": raw or None,
    }
