"""Read-only film report. It does not rewrite gameplay history."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cfb_coach.madden.model.film_evidence import load_admitted_records
from cfb_coach.madden.model.film_review import load_annotations
from cfb_coach.madden.model.opponent_learning import _is_approved_film_observation


def film_report(store: str | Path, game_id: str) -> dict[str, Any]:
    root = Path(store)
    recordings = []
    recording_dir = root / "recordings"
    if recording_dir.is_dir():
        for path in sorted(recording_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if str(payload.get("game_id") or "") == str(game_id):
                recordings.append(payload)
    annotations = load_annotations(root, game_id)
    candidates = list(annotations.get("candidates") or [])
    confirmed = [row for row in candidates if row.get("association_status") == "confirmed"]
    unresolved = [row for row in candidates if row.get("association_status") != "confirmed"]
    reviewed = [row for row in candidates if row.get("review_status") in ("confirmed", "corrected", "rejected")]
    rejected = [row for row in candidates if row.get("review_status") == "rejected"]
    accepted = [row for row in candidates if row.get("review_status") in ("confirmed", "corrected")]
    presnap = []
    conflicts = []
    for row in accepted:
        defense = row.get("defensive_observation") or {}
        if row.get("defense_review") == "confirmed" and row.get("observation_time") in ("pre_snap", "at_snap"):
            presnap.append({
                "candidate_id": row.get("candidate_id"),
                "snap_id": row.get("snap_id"),
                "safety_depth": defense.get("safety_depth"),
                "coverage_shell": defense.get("coverage_shell"),
                "structure_label": defense.get("structure_label"),
                "box_count": defense.get("box_count"),
                "pressure": defense.get("pressure"),
            })
        if row.get("play") and row.get("logged_play") and str(row.get("play")) != str(row.get("logged_play")):
            conflicts.append({
                "candidate_id": row.get("candidate_id"),
                "video_play": row.get("play"),
                "logged_play": row.get("logged_play"),
            })
    admitted = load_admitted_records(root, game_id=game_id)
    contributed = []
    withheld = []
    for row in admitted:
        if _is_approved_film_observation(row):
            contributed.append(row.get("evidence_id"))
        else:
            withheld.append({
                "evidence_id": row.get("evidence_id"),
                "observation_time": row.get("observation_time"),
                "available_before_snap": row.get("available_before_snap"),
                "reason": "not_a_pre_snap_learning_observation",
            })
    duration = None
    if recordings:
        duration = (recordings[0].get("probe") or {}).get("duration_s")
    gaps = []
    if not recordings:
        gaps.append("no_imported_recording_for_this_game")
    if not candidates:
        gaps.append("no_candidate_snaps")
    if unresolved:
        gaps.append("unresolved_snap_associations")
    if not presnap:
        gaps.append("no_human_confirmed_presnap_defense")
    gaps.append("madden_hud_recognition_not_validated")
    return {
        "game_id": game_id,
        "read_only": True,
        "history_modified": False,
        "learned_from_video": False,
        "recognition_accuracy_claim": False,
        "duration_s": duration,
        "recordings": len(recordings),
        "processed_segments": sum(len(row.get("segments") or []) for row in recordings),
        "candidate_offensive_snaps": len(candidates),
        "confirmed_snap_matches": len(confirmed),
        "unresolved_matches": len(unresolved),
        "accepted_observations": len(accepted),
        "rejected_observations": len(rejected),
        "human_reviewed_observations": len(reviewed),
        "presnap_defensive_information": presnap,
        "conflicts_with_manual_log": conflicts,
        "admitted_learning_evidence": [row.get("evidence_id") for row in admitted],
        "contributed_to_learning": contributed,
        "withheld_from_learning": withheld,
        "admitted_are_verified_executions": False,
        "visual_observation_coverage": {
            "candidates": len(candidates),
            "human_reviewed": len(reviewed),
        },
        "verified_training_eligibility": {
            "new_verified_executions_from_video": 0,
            "note": "Video review does not create verified executions.",
        },
        "information_gaps": gaps,
        "identification_claim": False,
        "real_recording_evaluation": real_recording_evaluation(),
    }


def real_recording_evaluation() -> dict[str, Any]:
    """Procedure for labeled Madden footage. Results stay empty until that footage exists."""
    return {
        "procedure": [
            "python3 -m cfb_coach ml film-import /path/to/madden-recording.mp4 --game-id GAME_ID",
            "python3 -m cfb_coach ml film-export-log --game-id GAME_ID --out log-snaps.json",
            "python3 -m cfb_coach ml film-review --game-id GAME_ID --serve",
            "Watch each candidate, correct its boundaries and labels, then confirm or reject it.",
            "python3 -m cfb_coach ml film-approve --game-id GAME_ID -o OPPONENT",
            "python3 -m cfb_coach ml film-report --game-id GAME_ID",
        ],
        "metrics_to_record": [
            "correct_candidate_snap_detections",
            "missed_snaps",
            "false_positives_from_menus_and_replays",
            "human_correction_time",
            "successful_log_associations",
            "verified_presnap_observations",
            "incorrect_or_unsupported_defensive_labels",
            "import_and_review_performance",
        ],
        "results": None,
        "labeled_madden_footage_available": False,
        "recognition_accuracy_claim": False,
        "note": "Synthetic clips do not measure Madden HUD recognition.",
    }
