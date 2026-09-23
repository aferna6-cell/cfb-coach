"""Parse snap outcome strings into success/fail + yards.

Accepts live HTML / terminal forms such as:
  gain 13, loss 2, +13, -2, incomplete, sack, TD, INT, stop, convert
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedOutcome:
    raw: str
    kind: str  # gain|loss|incomplete|sack|td|int|stop|convert|unknown
    yards: int | None = None
    success: bool | None = None  # offense-centric; use success_for(side)
    label: str = ""

    def success_for(self, side: str) -> bool | None:
        """Map offense-centric success onto the calling side."""
        if self.success is None:
            return None
        if (side or "offense").lower().startswith("d"):
            return not self.success
        return self.success

    def to_result_text(self) -> str:
        """Canonical result string for snaps.result / display."""
        if self.kind == "gain" and self.yards is not None:
            return f"+{self.yards}"
        if self.kind == "loss" and self.yards is not None:
            return f"-{abs(self.yards)}"
        if self.kind == "td":
            return "td" if self.yards is None else f"td +{self.yards}"
        if self.kind in ("incomplete", "sack", "int", "stop", "convert"):
            return self.kind
        return (self.raw or "").strip() or "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "yards": self.yards,
            "success": self.success,
            "label": self.label or self.to_result_text(),
        }


_GAIN_RE = re.compile(
    r"^(?:gain|gained|pick\s*up|pickup)\s*[:+]?\s*(-?\d+)\b", re.I
)
_LOSS_RE = re.compile(r"^(?:loss|lost|lose)\s*[:+]?\s*(-?\d+)\b", re.I)
_PLUS_RE = re.compile(r"^\+(\d+)\b")
_MINUS_RE = re.compile(r"^-(\d+)\b")
_TD_RE = re.compile(r"\b(td|touchdown|score)\b", re.I)
_INT_RE = re.compile(r"\b(int|intercept(?:ion|ed)?|pick(?:ed)?)\b", re.I)
_SACK_RE = re.compile(r"\b(sack(?:ed)?)\b", re.I)
_INC_RE = re.compile(r"\b(inc|incomp(?:lete)?|incomplete)\b", re.I)
_STOP_RE = re.compile(r"\b(stop|stuff(?:ed)?|hold|punt)\b", re.I)
_CONVERT_RE = re.compile(r"\b(convert(?:ed)?|1st|first\s*down|good)\b", re.I)
_YARDS_INLINE = re.compile(r"([+-]?\d+)\s*(?:yds?|yards?)?\b", re.I)


def parse_outcome(text: str | None) -> ParsedOutcome:
    """Parse a free-text or button outcome into structured fields."""
    raw = (text or "").strip()
    if not raw:
        return ParsedOutcome(raw="", kind="unknown", label="")

    low = raw.lower().strip()

    m = _GAIN_RE.match(low)
    if m:
        y = abs(int(m.group(1)))
        return ParsedOutcome(raw=raw, kind="gain", yards=y, success=True, label=f"+{y}")

    m = _LOSS_RE.match(low)
    if m:
        y = abs(int(m.group(1)))
        return ParsedOutcome(raw=raw, kind="loss", yards=-y, success=False, label=f"-{y}")

    m = _PLUS_RE.match(low)
    if m:
        y = int(m.group(1))
        return ParsedOutcome(raw=raw, kind="gain", yards=y, success=True, label=f"+{y}")

    m = _MINUS_RE.match(low)
    if m:
        y = -int(m.group(1))
        return ParsedOutcome(raw=raw, kind="loss", yards=y, success=False, label=f"{y}")

    if _INT_RE.search(low):
        return ParsedOutcome(raw=raw, kind="int", yards=0, success=False, label="INT")

    if _SACK_RE.search(low):
        y = None
        ym = _YARDS_INLINE.search(low)
        if ym:
            try:
                y = -abs(int(ym.group(1)))
            except ValueError:
                y = None
        return ParsedOutcome(raw=raw, kind="sack", yards=y, success=False, label="sack")

    if _INC_RE.search(low):
        return ParsedOutcome(
            raw=raw, kind="incomplete", yards=0, success=False, label="incomplete"
        )

    if _TD_RE.search(low):
        y = None
        ym = _PLUS_RE.search(low) or _YARDS_INLINE.search(low)
        if ym:
            try:
                y = abs(int(ym.group(1)))
            except ValueError:
                y = None
        return ParsedOutcome(raw=raw, kind="td", yards=y, success=True, label="TD")

    if _STOP_RE.search(low):
        # "stop" is defense-success language; offense-centric = fail
        return ParsedOutcome(raw=raw, kind="stop", yards=0, success=False, label="stop")

    if _CONVERT_RE.search(low):
        return ParsedOutcome(
            raw=raw, kind="convert", yards=None, success=True, label="convert"
        )

    # Fallback: leading digit with sign words already handled; try bare "13"
    bare = re.match(r"^(\d+)\b", low)
    if bare:
        y = int(bare.group(1))
        return ParsedOutcome(raw=raw, kind="gain", yards=y, success=True, label=f"+{y}")

    return ParsedOutcome(raw=raw, kind="unknown", success=None, label=raw)


def outcome_success(result: str | None, side: str) -> bool | None:
    """True/False/None for learning — preferred entry point over ad-hoc string checks."""
    parsed = parse_outcome(result)
    if parsed.kind == "unknown" and parsed.success is None:
        # Preserve legacy crude heuristics for free-text that didn't match
        return _legacy_success(result, side)
    # "stop" stored from defense logging is defense success — if side is defense,
    # success_for flips offense-centric False → True. Good.
    # Special case: defense logging "stop" as their win already encoded offense-fail.
    return parsed.success_for(side)


def _legacy_success(result: str | None, side: str) -> bool | None:
    if not result:
        return None
    r = result.lower().strip()
    if side == "offense":
        if r.startswith("+") or "td" in r or "convert" in r or "good" in r:
            return True
        if any(w in r for w in ("int", "sack", "loss", "fail", "incomp", "pick", "stuff")):
            return False
        if r.startswith("-"):
            return False
        return None
    if any(w in r for w in ("stop", "sack", "stuff", "hold", "punt", "int", "turnover")):
        return True
    if r.startswith("+") or "td" in r or "convert" in r:
        return False
    return None


__all__ = ["ParsedOutcome", "outcome_success", "parse_outcome"]
