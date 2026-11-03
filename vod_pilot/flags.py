"""Per-fix switches for a VOD label build.

Each flag is off unless the CLI turns it on, so a box can re-label one
gap at a time and keep the previous labels byte-compatible. The payload
records which flags were on under ``source.fixes``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ExtractFlags:
    """One boolean per label fix. Default is the PR #16 reader."""

    read_defense: bool = False
    read_score: bool = False
    read_offense_formation: bool = False
    read_adjustments: bool = False
    yardline_glyphs: bool = False
    read_result: bool = False
    goal_to_go: bool = False
    dd_norm: bool = False
    presnap_clock: bool = False
    flag_menu: bool = False
    field_coverage: bool = False
    book_match: bool = False

    def enabled(self) -> list[str]:
        return [item.name for item in fields(self) if getattr(self, item.name)]

    def needs_boxes(self) -> bool:
        """OCR x/y/conf. Yard line and bare down-and-distance use the slot."""
        return self.yardline_glyphs or self.dd_norm or self.read_score


def flags_from_namespace(args: Any) -> ExtractFlags:
    """Read CLI flags. Missing attributes stay off so flags can land one at a time."""
    data = {item.name: bool(getattr(args, item.name, False)) for item in fields(ExtractFlags)}
    return ExtractFlags(**data)
