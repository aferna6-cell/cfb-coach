"""PlayRecord — one structured play per snap (Milestone 2)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlayRecord:
    game_id: str = ""
    session_id: str = ""
    play_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    opponent_id: str = ""
    side: str = "offense"  # our side perspective: offense|defense
    # Situation
    down: int | None = None
    distance: int | None = None
    yardline: int | None = None
    quarter: int | None = None
    field_zone: str = ""  # open|rz|gl|backed
    score_us: int | None = None
    score_them: int | None = None
    # Pre-snap look
    formation: str = "unknown"
    formation_side: str = ""
    shell: str = "unknown"
    pressure: str = "unknown"
    motion_present: bool | None = None
    motion_direction: str = "UNKNOWN"  # L|R|NO_MOTION|UNKNOWN
    # Classification
    play_family: str = "UNKNOWN"
    concept_tags: list[str] = field(default_factory=list)
    result_type: str = "unknown"
    yards: int | None = None
    scramble_dir: str | None = None
    # Coach
    coach_rec: str = ""
    macro: str = ""
    our_call: str = ""
    # Meta
    confidence: dict[str, float] = field(default_factory=dict)
    snap_confidence: float = 0.0
    end_confidence: float = 0.0
    corrected: bool = False
    notes: str = ""
    ts_snap: float | None = None
    ts_end: float | None = None
    explosive: bool = False

    def conf(self, key: str, default: float = 0.0) -> float:
        return float(self.confidence.get(key, default))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlayRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        if "concept_tags" in kwargs and kwargs["concept_tags"] is None:
            kwargs["concept_tags"] = []
        if "confidence" in kwargs and kwargs["confidence"] is None:
            kwargs["confidence"] = {}
        return cls(**kwargs)

    def situation_bucket(self) -> str:
        dist = self.distance
        if dist is None:
            dd = "unk"
        elif dist <= 2:
            dd = "short"
        elif dist <= 6:
            dd = "med"
        else:
            dd = "long"
        zone = self.field_zone or "open"
        down = str(self.down) if self.down else "x"
        return f"{zone}|{dd}|{down}"

    def score_state(self) -> str:
        if self.score_us is None or self.score_them is None:
            return "unknown"
        diff = self.score_us - self.score_them
        if abs(diff) <= 7:
            return "close"
        if diff > 7:
            return "ahead"
        return "behind"
