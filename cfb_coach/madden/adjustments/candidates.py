"""JSON adapter for the Madden ML Lead's ``madden-ml.macro.1`` contract.

The types and ``is_live_selectable`` live in ``cfb_coach.madden.model.schema``.
This module does not redefine them and does not edit that file. ``to_dict`` /
``from_dict`` already stamp ``__schema__`` with ``madden-ml.macro.1``.

``design_record_id`` is a real ``MacroCandidate`` field. Payloads that omit
it still load, as ``None``.
"""

from __future__ import annotations

from typing import Any

from cfb_coach.madden.model.schema import (
    MACRO_SCHEMA_VERSION,
    AdjustmentPrimitive,
    EditorFieldAssignment,
    EditorFieldSource,
    MacroCandidate,
    MacroConflict,
    MacroJointFeatures,
    MacroSide,
    MacroState,
    from_dict,
    is_live_selectable,
    to_dict,
)

WIRE_VERSION = MACRO_SCHEMA_VERSION


def candidate_to_json(candidate: MacroCandidate) -> dict[str, Any]:
    """Thin shim. Unknown scalars stay ``None`` so JSON writes ``null``."""
    if not isinstance(candidate, MacroCandidate):
        raise TypeError("candidate_to_json expects a MacroCandidate")
    payload = to_dict(candidate)
    if payload.get("__type__") != "MacroCandidate" or payload.get("__schema__") != WIRE_VERSION:
        raise ValueError(f"macro JSON schema is not {WIRE_VERSION}")
    return payload


def candidate_from_json(payload: dict[str, Any]) -> MacroCandidate:
    if not isinstance(payload, dict):
        raise TypeError("macro JSON must be an object")
    version = payload.get("__schema__")
    if version != WIRE_VERSION:
        raise ValueError(f"macro JSON schema {version!r} is not {WIRE_VERSION}")
    if payload.get("__type__") not in (None, "MacroCandidate"):
        raise ValueError("macro JSON __type__ must be MacroCandidate")
    loaded = from_dict(payload)
    if not isinstance(loaded, MacroCandidate):
        raise TypeError("macro JSON did not load a MacroCandidate")
    return loaded


__all__ = [
    "WIRE_VERSION",
    "AdjustmentPrimitive",
    "EditorFieldAssignment",
    "EditorFieldSource",
    "MacroCandidate",
    "MacroConflict",
    "MacroJointFeatures",
    "MacroSide",
    "MacroState",
    "candidate_from_json",
    "candidate_to_json",
    "is_live_selectable",
]
