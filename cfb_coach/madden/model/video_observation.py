"""Versioned video-observation contract for a later film pass.

Accepted rows stay pending validation. They are not verified executions
and this importer does not write a database.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.defensive_observation import (
    TIMINGS,
    structure_observation,
)

VIDEO_OBSERVATION_VERSION = "video_observation.v1"
HUMAN_STATUS = frozenset({"unverified", "pending", "rejected", "verified_human"})
RELATIVE_TIME = frozenset({"pre_snap", "at_snap", "post_snap", "unknown"})


def _reject(row: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "accepted": False,
        "reasons": reasons,
        "schema": VIDEO_OBSERVATION_VERSION,
        "status": "rejected",
        "writes_database": False,
        "verified_execution": False,
        "source_row": dict(row),
    }


def validate_video_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Accept a complete pending observation or reject it with reasons."""
    reasons: list[str] = []
    if not isinstance(row, Mapping) or not row:
        return _reject(row if isinstance(row, Mapping) else {}, ["empty_observation"])
    if row.get("schema") not in (None, VIDEO_OBSERVATION_VERSION):
        reasons.append("unsupported_schema")
    if not (row.get("game_id") or row.get("session_id")):
        reasons.append("missing_game_or_session")
    if not row.get("recording_id"):
        reasons.append("missing_recording_id")
    has_snap = bool(row.get("snap_id"))
    unresolved = row.get("unresolved_snap_association")
    if has_snap and unresolved:
        reasons.append("ambiguous_snap_association")
    if not has_snap and not unresolved:
        reasons.append("missing_snap_association")
    if row.get("video_timestamp") is None and not row.get("frame_interval"):
        reasons.append("missing_video_time")
    relative = row.get("observation_time_relative_to_snap")
    if relative not in RELATIVE_TIME:
        reasons.append("invalid_observation_time")
    if "confidence" not in row or row.get("confidence") is None:
        reasons.append("missing_confidence")
    else:
        try:
            confidence = float(row.get("confidence"))
            if confidence < 0 or confidence > 1:
                reasons.append("confidence_out_of_range")
        except (TypeError, ValueError):
            reasons.append("confidence_not_numeric")
    status = row.get("human_verification")
    if status not in HUMAN_STATUS:
        reasons.append("invalid_human_verification")
    if status == "verified_manual" or row.get("source") == "verified_manual":
        reasons.append("verified_manual_is_not_a_video_status")
    if row.get("verified_execution") or row.get("eligibility") == "verified_execution":
        reasons.append("video_row_cannot_claim_verified_execution")
    alignment = row.get("defensive_alignment")
    state = row.get("observed_game_state")
    if not alignment and not state:
        reasons.append("empty_observation")
    structured = None
    if isinstance(alignment, Mapping):
        if alignment.get("contradictory_shells") or (
            isinstance(alignment.get("shells_mentioned"), list)
            and len(alignment.get("shells_mentioned") or []) > 1
        ):
            reasons.append("contradictory_shells")
        shell = alignment.get("coverage_shell")
        if isinstance(shell, list) and len(shell) > 1:
            reasons.append("contradictory_shells")
        structured = structure_observation(
            alignment.get("raw"),
            timing=alignment.get("timing") or "unknown",
            source="video",
            confidence=row.get("confidence") if isinstance(row.get("confidence"), (int, float)) else None,
            fields=alignment,
        )
        if structured.get("contradictory_shells"):
            reasons.append("contradictory_shells")
        if alignment.get("timing") not in (None, *TIMINGS):
            reasons.append("invalid_alignment_timing")
    elif alignment:
        reasons.append("alignment_not_structured")
    if reasons:
        return _reject(row, sorted(set(reasons)))
    missing = list((structured or {}).get("unknown") or [])
    if not state:
        missing.append("observed_game_state")
    return {
        "accepted": True,
        "schema": VIDEO_OBSERVATION_VERSION,
        "game_id": row.get("game_id") or row.get("session_id"),
        "session_id": row.get("session_id") or row.get("game_id"),
        "snap_id": row.get("snap_id"),
        "unresolved_snap_association": unresolved,
        "recording_id": row.get("recording_id"),
        "video_timestamp": row.get("video_timestamp"),
        "frame_interval": row.get("frame_interval"),
        "observation_time_relative_to_snap": relative,
        "observed_game_state": state,
        "defensive_alignment": structured,
        "source_model": row.get("source_model"),
        "source_version": row.get("source_version"),
        "confidence": float(row["confidence"]),
        "human_verification": status,
        "field_confidence": dict(row.get("field_confidence") or {}),
        "source_layer": row.get("source_layer") or "model",
        "raw_model_observation": row.get("raw_model_observation"),
        "human_label": row.get("human_label"),
        "missing_fields": missing,
        "unknown_fields": missing,
        "available_before_snap": relative != "post_snap",
        "status": "video_pending_validation",
        "verified_execution": False,
        "eligibility": "video_pending_validation",
        "writes_database": False,
        "learned_from_video": False,
    }


def dry_run_import(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate fixtures. Accepted rows are not mixed into verified snaps."""
    accepted = []
    rejected = []
    for row in rows:
        result = validate_video_observation(row)
        if result.get("accepted"):
            accepted.append(result)
        else:
            rejected.append(result)
    return {
        "schema": VIDEO_OBSERVATION_VERSION,
        "dry_run": True,
        "writes_database": False,
        "learned_from_video": False,
        "separate_from_verified_observations": True,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "note": (
            "No video observation has been validated against a real recording. "
            "Pending rows do not update the opponent model."
        ),
    }


def synthetic_video_fixtures() -> list[dict[str, Any]]:
    """Synthetic rows for the dry run. They are not game evidence."""
    good = {
        "schema": VIDEO_OBSERVATION_VERSION,
        "game_id": "synthetic-game",
        "snap_id": "synthetic-snap-1",
        "recording_id": "synthetic-recording",
        "video_timestamp": 12.5,
        "observation_time_relative_to_snap": "pre_snap",
        "observed_game_state": {"down": 3, "distance": 7, "synthetic": True},
        "defensive_alignment": {
            "raw": "showing cover 1 blitz",
            "timing": "pre_snap",
        },
        "source_model": "synthetic_fixture",
        "source_version": "0",
        "confidence": 0.62,
        "human_verification": "pending",
    }
    ambiguous = {
        "schema": VIDEO_OBSERVATION_VERSION,
        "game_id": "synthetic-game",
        "unresolved_snap_association": "unmatched-frame-44",
        "recording_id": "synthetic-recording",
        "frame_interval": [44, 60],
        "observation_time_relative_to_snap": "unknown",
        "observed_game_state": {"synthetic": True},
        "defensive_alignment": {"raw": "man", "timing": "unknown"},
        "source_model": "synthetic_fixture",
        "source_version": "0",
        "confidence": 0.3,
        "human_verification": "unverified",
    }
    return [good, ambiguous]
