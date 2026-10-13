"""Snap rows shaped like ``cfb_coach.db.CoachDB.log_snap``.

Column order matches the INSERT in ``cfb_coach/db.py``. Extra keys are kept
in JSON for the pilot (timestamps, confidence) and omitted from the CSV
columns a future importer would pass to ``log_snap``. Nothing here writes
the coach database.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Same order as CoachDB.log_snap.
SNAP_COLUMNS = (
    "ts",
    "opponent_id",
    "side",
    "down",
    "distance",
    "yardline",
    "quarter",
    "situation_raw",
    "our_call",
    "formation",
    "play",
    "macro",
    "result",
    "coverage_seen",
    "concept_seen",
    "notes",
    "session_id",
)

SCHEMA_ID = "cfb_coach.snaps.v1"


def _ordinal(down: int | None) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(down or 0, "")


def situation_raw(
    *,
    down: int | None,
    distance: int | None,
    quarter: int | None = None,
    clock: str | None = None,
    yardline_phrase: str | None = None,
) -> str:
    """A string ``parse_situation`` can read for down and distance.

    Yard line is included only as ``yl N`` when the caller already converted
    it to the coach's 0-100-from-our-goal convention. A bare on-field number
    is not written here, because own-versus-opp is usually unknown on a VOD.
    """
    parts: list[str] = []
    if down and distance is not None:
        parts.append(f"{_ordinal(down)} & {distance}")
    if yardline_phrase:
        parts.append(yardline_phrase)
    if quarter:
        parts.append(f"q{quarter}")
    if clock:
        parts.append(clock)
    return " ".join(parts)


@dataclass
class SnapRow:
    """One tagged snap. Empty strings mean 'not read', not a guess."""

    ts: str
    opponent_id: str
    side: str = "offense"
    down: int | None = None
    distance: int | None = None
    yardline: int | None = None
    quarter: int | None = None
    situation_raw: str = ""
    our_call: str = ""
    formation: str = ""
    play: str = ""
    macro: str = ""
    result: str = ""
    coverage_seen: str = ""
    concept_seen: str = ""
    notes: str = ""
    session_id: str = ""
    # Pilot extras (JSON only).
    video_id: str = ""
    t_start: float | None = None
    t_end: float | None = None
    game: str = ""
    result_kind: str = ""
    yards: int | None = None
    field_zone: str = ""
    confidence: dict[str, float] = field(default_factory=dict)

    def to_snap_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        out: dict[str, Any] = {}
        for col in SNAP_COLUMNS:
            val = raw[col]
            out[col] = "" if val is None else val
        return out

    def to_full_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(self.to_snap_dict())
        return d


def result_text(*, kind: str, yards: int | None) -> str:
    """Canonical ``snaps.result`` text, compatible with ``parse_outcome``."""
    if kind == "sack" and yards is None:
        return "sack"
    if kind == "incomplete":
        return "incomplete"
    if kind == "int":
        return "int"
    if kind == "fumble":
        return "fumble"
    if kind == "td":
        return "td" if yards is None else f"td +{yards}"
    if kind == "convert" and yards is None:
        return "convert"
    if yards is None:
        return ""
    if yards > 0:
        return f"+{yards}"
    if yards < 0:
        return str(yards)
    return "+0"


def write_outputs(path_stem: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = path_stem.with_suffix(".json")
    csv_path = path_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    snaps = payload.get("snaps") or []
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SNAP_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for snap in snaps:
            row = {col: snap.get(col, "") for col in SNAP_COLUMNS}
            for col, val in row.items():
                if val is None:
                    row[col] = ""
            writer.writerow(row)
    return json_path, csv_path
