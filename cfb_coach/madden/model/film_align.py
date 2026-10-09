"""Associate video segments with coach-log snaps.

A similar game clock is not an identity. Automatic proposals stay unresolved.
A confirmed link requires a person to name a snap that exists in the log.
This module never creates a verified execution.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def align_segments(
    segments: Sequence[Mapping[str, Any]],
    log_snaps: Sequence[Mapping[str, Any]],
    manual_links: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    by_segment = {str(row.get("candidate_id")): row for row in segments}
    by_snap = {str(row.get("snap_id")): row for row in log_snaps if row.get("snap_id")}
    confirmed: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for link in manual_links or []:
        candidate_id = str(link.get("candidate_id") or "")
        snap_id = str(link.get("snap_id") or "")
        segment = by_segment.get(candidate_id)
        snap = by_snap.get(snap_id)
        if segment is None or snap is None:
            rows.append(_row(
                candidate_id, snap_id or None, "unresolved", "manual_target_missing", 0.0,
            ))
            continue
        confirmed[candidate_id] = snap_id
        rows.append(_row(candidate_id, snap_id, "confirmed", "manual_association", 1.0))
    for segment in segments:
        candidate_id = str(segment.get("candidate_id"))
        if candidate_id in confirmed:
            continue
        if segment.get("role") in ("menu", "replay", "menu_or_replay"):
            rows.append(_row(candidate_id, None, "unresolved", "menu_or_replay_not_a_snap", 0.0))
            continue
        if segment.get("role") == "possible_camera_cut":
            rows.append(_row(candidate_id, None, "unresolved", "camera_or_menu_cut_not_a_snap", 0.0))
            continue
        if segment.get("boundary_status") == "unresolved":
            rows.append(_row(candidate_id, None, "unresolved", "ambiguous_boundary", float(segment.get("confidence") or 0)))
            continue
        proposal = _propose(segment, log_snaps, set(confirmed.values()))
        rows.append(proposal)
    matched = {row["snap_id"] for row in rows if row.get("status") == "confirmed" and row.get("snap_id")}
    unmatched_logs = []
    for snap in log_snaps:
        snap_id = snap.get("snap_id")
        if snap_id and str(snap_id) not in matched:
            unmatched_logs.append({
                "snap_id": snap_id,
                "snap_seq": snap.get("snap_seq"),
                "reason": _unmatched_reason(snap, log_snaps, matched),
            })
    return {
        "associations": rows,
        "confirmed_count": sum(1 for row in rows if row["status"] == "confirmed"),
        "unresolved_count": sum(1 for row in rows if row["status"] != "confirmed"),
        "unmatched_log_snaps": unmatched_logs,
        "verified_executions_created": 0,
        "clock_only_matches_confirmed": 0,
        "note": "Confirmed associations are manual. Clock similarity never confirms a snap.",
    }


def _propose(
    segment: Mapping[str, Any],
    log_snaps: Sequence[Mapping[str, Any]],
    used: set[str],
) -> dict[str, Any]:
    scored = []
    for snap in log_snaps:
        snap_id = str(snap.get("snap_id") or "")
        if not snap_id or snap_id in used:
            continue
        score, reasons = _score(segment, snap)
        scored.append((score, reasons, snap_id))
    scored.sort(key=lambda item: (-item[0], item[2]))
    if not scored:
        return _row(str(segment.get("candidate_id")), None, "unresolved", "no_log_snap", 0.0)
    score, reasons, snap_id = scored[0]
    clock_only = reasons == ["displayed_clock"]
    if clock_only:
        return _row(
            str(segment.get("candidate_id")), None, "unresolved",
            "clock_similarity_is_not_enough", score, rejected_snap_id=snap_id,
        )
    return _row(
        str(segment.get("candidate_id")), None, "unresolved",
        "proposed_not_confirmed", score, proposed_snap_id=snap_id, reasons=reasons,
    )


def _score(segment: Mapping[str, Any], snap: Mapping[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    state = segment.get("displayed_state") or {}
    if _same(state.get("down"), snap.get("down")) and _same(state.get("distance"), snap.get("distance")):
        score += 0.45
        reasons.append("down_and_distance")
    if _same(state.get("quarter"), snap.get("quarter")):
        score += 0.15
        reasons.append("quarter")
    if _same(state.get("clock_seconds"), snap.get("clock_seconds")):
        score += 0.1
        reasons.append("displayed_clock")
    if _same(segment.get("suggested_seq"), snap.get("snap_seq")):
        score += 0.2
        reasons.append("sequence_order")
    video_time = segment.get("start_s")
    logged_time = snap.get("video_timestamp_s")
    if video_time is not None and logged_time is not None:
        try:
            if abs(float(video_time) - float(logged_time)) <= 1.0:
                score += 0.1
                reasons.append("video_timestamp")
        except (TypeError, ValueError):
            pass
    return round(score, 3), reasons


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None or left == "" or right == "":
        return False
    return str(left) == str(right)


def _unmatched_reason(
    snap: Mapping[str, Any],
    log_snaps: Sequence[Mapping[str, Any]],
    matched: set[Any],
) -> str:
    seqs = [row.get("snap_seq") for row in log_snaps if row.get("snap_id") in matched and row.get("snap_seq") is not None]
    if seqs and snap.get("snap_seq") is not None and int(snap["snap_seq"]) < min(int(value) for value in seqs):
        return "possible_recording_started_after_kickoff"
    return "skipped_or_unlogged_in_video"


def _row(
    candidate_id: str,
    snap_id: str | None,
    status: str,
    reason: str,
    confidence: float,
    *,
    rejected_snap_id: str | None = None,
    proposed_snap_id: str | None = None,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "snap_id": snap_id,
        "proposed_snap_id": proposed_snap_id,
        "rejected_snap_id": rejected_snap_id,
        "status": status,
        "reason": reason,
        "reasons": reasons or [reason],
        "confidence": confidence,
        "verified_execution": False,
    }
