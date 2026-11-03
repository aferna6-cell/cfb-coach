"""On-screen adjustments. A route word inside a play name is not one.

Only a short line with an explicit label counts: hot route, audible, an
offensive macro phrase, or a defensive adjustment. A tooltip sentence is
skipped so a play description cannot invent a call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("hot_route", re.compile(r"\bHOT\s*ROUTES?(?:\s+[A-Z][A-Z0-9]+){0,3}", re.I), 0.9),
    ("audible", re.compile(r"\bAUDIBLE(?:\s+TO)?(?:\s+[A-Z0-9][A-Z0-9]*){0,4}", re.I), 0.9),
    (
        "offensive",
        re.compile(r"\b(?:ID\s+THE\s+MIC|SLIDE\s+(?:LEFT|RIGHT)|SMART\s+ROUTE|PASS\s+BLOCK)\b", re.I),
        0.8,
    ),
    (
        "defensive",
        re.compile(
            r"\b(?:PRESS|SHADE\s+(?:INSIDE|OUTSIDE|IN|OUT)|SHOW\s+BLITZ|QB\s+CONTAIN|GUESS\s+PASS|QB\s+SPY)\b",
            re.I,
        ),
        0.82,
    ),
)


@dataclass(frozen=True)
class Adjustment:
    kind: str
    text: str
    confidence: float

    def label(self) -> str:
        return f"{self.kind}: {self.text}"


def read_adjustments(texts: list[str]) -> tuple[list[Adjustment], float]:
    found: list[Adjustment] = []
    seen: set[str] = set()
    for line in texts:
        text = " ".join(str(line).upper().split())
        if not text or len(text) >= 40:
            continue
        for kind, pattern, conf in _PATTERNS:
            for match in pattern.finditer(text):
                label = " ".join(match.group(0).upper().split())
                if label in seen:
                    continue
                seen.add(label)
                found.append(Adjustment(kind, label, conf))
    if not found:
        return [], 0.0
    return found, min(item.confidence for item in found)
