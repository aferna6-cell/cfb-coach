"""Training-row export and import. Stub until Data Engineering implements it.

Rows must keep recommendations, verified executions, and outcomes in separate
fields. Missing source columns stay unknown. Do not synthesize snaps.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report schema gaps and leakage. Not implemented.

    The report must not invent counts for files that were not read.
    """
    raise NotImplementedError


def build_rows(
    *,
    db: Any = None,
    extra_paths: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build feature rows from a Madden database and optional existing files.

    Not implemented. Must not fabricate rows when ``db`` is empty.
    """
    raise NotImplementedError


def export_jsonl(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    """Write rows as JSONL. Not implemented."""
    raise NotImplementedError


def import_path(path: str) -> list[dict[str, Any]]:
    """Read CSV, JSON, JSONL, or SQLite into row dicts. Not implemented.

    Older files with missing columns stay readable. Absent fields are unknown.
    """
    raise NotImplementedError
