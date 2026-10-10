"""JSON adapter for the Madden ML Lead's ``madden-ml.macro.1`` contract.

The types and ``is_live_selectable`` live in ``cfb_coach.madden.model.schema``.
This module does not redefine them and does not edit that file. ``to_dict`` /
``from_dict`` already stamp ``__schema__`` with ``madden-ml.macro.1``.
"""

from __future__ import annotations

from dataclasses import fields
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

# Not a MacroCandidate field. MacroCandidate in schema.py has no
# design_record_id. This key is carried only on the JSON object so a
# DesignRecord can be found from the wire payload. Downside, catalog
# comparison, and install steps stay off the wire. When the Lead adds
# the field to the contract, to_dict owns it and this shim stops writing it.
DESIGN_RECORD_ID_KEY = "design_record_id"


def contract_has_design_record_id() -> bool:
    """True once MacroCandidate grows a design_record_id field. It does not today."""
    return any(item.name == DESIGN_RECORD_ID_KEY for item in fields(MacroCandidate))


def candidate_to_json(
    candidate: MacroCandidate,
    *,
    design_record_id: str | None = None,
) -> dict[str, Any]:
    """Thin shim. Unknown scalars stay ``None`` so JSON writes ``null``.

    ``design_record_id`` is attached beside the contract fields. It is not
    passed through ``to_dict`` while the contract lacks the field, so a
    future schema addition cannot be double-written.
    """
    if not isinstance(candidate, MacroCandidate):
        raise TypeError("candidate_to_json expects a MacroCandidate")
    if design_record_id is not None and not isinstance(design_record_id, str):
        raise TypeError("design_record_id must be a string or unknown")
    if design_record_id is not None and not design_record_id.strip():
        raise ValueError("design_record_id must be a non-empty string")
    payload = to_dict(candidate)
    if payload.get("__type__") != "MacroCandidate" or payload.get("__schema__") != WIRE_VERSION:
        raise ValueError(f"macro JSON schema is not {WIRE_VERSION}")
    if contract_has_design_record_id():
        contracted = payload.get(DESIGN_RECORD_ID_KEY)
        if design_record_id is not None and contracted not in (None, design_record_id):
            raise ValueError("design_record_id does not match the contract field")
        return payload
    if design_record_id is not None:
        payload[DESIGN_RECORD_ID_KEY] = design_record_id
    return payload


def design_record_id_from_json(payload: dict[str, Any]) -> str | None:
    """Read the shim link. ``None`` means this payload has no design record id."""
    if not isinstance(payload, dict):
        raise TypeError("macro JSON must be an object")
    value = payload.get(DESIGN_RECORD_ID_KEY)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("design_record_id must be a non-empty string or null")
    return value


def candidate_from_json(payload: dict[str, Any]) -> MacroCandidate:
    if not isinstance(payload, dict):
        raise TypeError("macro JSON must be an object")
    version = payload.get("__schema__")
    if version != WIRE_VERSION:
        raise ValueError(f"macro JSON schema {version!r} is not {WIRE_VERSION}")
    if payload.get("__type__") not in (None, "MacroCandidate"):
        raise ValueError("macro JSON __type__ must be MacroCandidate")
    # from_dict reads only MacroCandidate fields. design_record_id stays on
    # the JSON object and is ignored here until the contract has the field.
    design_record_id_from_json(payload)
    loaded = from_dict(payload)
    if not isinstance(loaded, MacroCandidate):
        raise TypeError("macro JSON did not load a MacroCandidate")
    return loaded


__all__ = [
    "DESIGN_RECORD_ID_KEY",
    "WIRE_VERSION",
    "AdjustmentPrimitive",
    "contract_has_design_record_id",
    "design_record_id_from_json",
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
