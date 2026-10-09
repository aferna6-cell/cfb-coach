"""Controlled export from reviewed film into opponent learning.

Approving a defensive look is not confirmation that the play was executed.
A video review cannot create a verified execution the log does not already have.
Rollback withdraws the export. Gameplay history is not rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.defensive_observation import structure_observation

ADMISSION_SCHEMA = "madden.film.admission.v1"


def _path(store: str | Path, game_id: str) -> Path:
    return Path(store) / "admitted" / f"{game_id}.json"


def propose_admission(
    annotations: Mapping[str, Any],
    log_snaps: Sequence[Mapping[str, Any]],
    *,
    opponent_id: str,
) -> dict[str, Any]:
    logs = {str(row.get("snap_id")): row for row in log_snaps if row.get("snap_id")}
    proposals = []
    for row in annotations.get("candidates") or []:
        if row.get("review_status") not in ("confirmed", "corrected"):
            continue
        snap_id = row.get("snap_id")
        if row.get("association_status") != "confirmed" or not snap_id or str(snap_id) not in logs:
            proposals.append({
                "candidate_id": row.get("candidate_id"),
                "admit": False,
                "reason": "unresolved_or_unknown_snap",
                "verified_execution": False,
            })
            continue
        log = logs[str(snap_id)]
        if row.get("defense_review") == "confirmed" and row.get("defensive_observation"):
            proposals.append(_defense_proposal(
                row, log, opponent_id, game_id=str(annotations.get("game_id") or ""),
            ))
        if row.get("execution_review") == "confirmed":
            proposals.append(_execution_proposal(row, log))
    return {
        "schema": ADMISSION_SCHEMA,
        "game_id": annotations.get("game_id"),
        "opponent_id": opponent_id,
        "proposals": proposals,
        "verified_executions_created": 0,
        "history_modified": False,
    }


def _defense_proposal(
    row: Mapping[str, Any], log: Mapping[str, Any], opponent_id: str, *, game_id: str,
) -> dict[str, Any]:
    observed = structure_observation(
        None,
        timing="pre_snap",
        source="video",
        confidence=float(row.get("confidence") or 0.5),
        fields=row.get("defensive_observation") or {},
    )
    if observed.get("safety_depth") == "two_high":
        observed["coverage_shell"] = None
        observed["structure_label"] = "two_high_safety_structure"
    evidence_id = f"{row.get('candidate_id')}:defense"
    return {
        "admit": True,
        "kind": "defensive_observation",
        "evidence_id": evidence_id,
        "candidate_id": row.get("candidate_id"),
        "snap_id": row.get("snap_id"),
        "snap_seq": log.get("snap_seq"),
        "game_id": log.get("game_id") or log.get("session_id") or game_id,
        "opponent_id": opponent_id,
        "down": log.get("down"),
        "distance": log.get("distance"),
        "observation": observed,
        "verified_execution": False,
        "defense_observation_approved": True,
        "human_verification": "verified_human",
        "source": "human_confirmed_film",
        "reason": "human_confirmed_alignment_on_an_existing_snap",
    }


def _execution_proposal(row: Mapping[str, Any], log: Mapping[str, Any]) -> dict[str, Any]:
    verified = str(log.get("executed_verification") or "").lower() == "verified"
    same_play = bool(row.get("play")) and str(row.get("play")) == str(log.get("executed_play") or "")
    if verified and same_play:
        return {
            "admit": False,
            "kind": "execution",
            "candidate_id": row.get("candidate_id"),
            "snap_id": row.get("snap_id"),
            "verified_execution": False,
            "reason": "log_already_verified_execution_not_duplicated",
        }
    return {
        "admit": False,
        "kind": "execution",
        "candidate_id": row.get("candidate_id"),
        "snap_id": row.get("snap_id"),
        "verified_execution": False,
        "reason": "video_cannot_create_verified_execution",
    }


def approve_admission(store: str | Path, proposal: Mapping[str, Any]) -> dict[str, Any]:
    game_id = str(proposal.get("game_id") or "unscoped")
    admitted = [row for row in proposal.get("proposals") or [] if row.get("admit") is True]
    payload = {
        "schema": ADMISSION_SCHEMA,
        "game_id": game_id,
        "opponent_id": proposal.get("opponent_id"),
        "status": "approved",
        "records": admitted,
        "evidence_ids": [row.get("evidence_id") for row in admitted],
        "withdrawn": False,
        "history_modified": False,
        "verified_executions_created": 0,
    }
    path = _path(store, game_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(path), "admitted": len(admitted), "history_modified": False}


def rollback_admission(store: str | Path, game_id: str) -> dict[str, Any]:
    path = _path(store, game_id)
    if not path.is_file():
        return {"ok": True, "withdrawn": True, "admitted": 0, "history_modified": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    payload["status"] = "withdrawn"
    payload["withdrawn"] = True
    payload["records"] = []
    payload["evidence_ids"] = []
    payload["history_modified"] = False
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ok": True, "withdrawn": True, "admitted": 0, "history_modified": False, "path": str(path)}


def load_admitted_records(
    store: str | Path,
    *,
    game_id: str | None = None,
    opponent_id: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(store) / "admitted"
    if not root.is_dir():
        return []
    paths = [root / f"{game_id}.json"] if game_id else sorted(root.glob("*.json"))
    records = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("withdrawn") or payload.get("status") == "withdrawn":
            continue
        for row in payload.get("records") or []:
            if opponent_id is not None and str(row.get("opponent_id") or "") != str(opponent_id):
                continue
            if game_id is not None and str(row.get("game_id") or payload.get("game_id") or "") != str(game_id):
                continue
            if row.get("verified_execution") is True:
                continue
            records.append(dict(row))
    return records
