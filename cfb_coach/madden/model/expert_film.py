"""Expert-match film ingestion and manual snap annotation.

Accepts locally provided footage the user has permission to process.
Does not download, scrape, or redistribute copyrighted broadcasts.
Raw video is never claimed to be automatically understood.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.film_import import (
    FILM_IMPORT_VERSION,
    fingerprint_file,
    import_recording,
    probe_video,
)
from cfb_coach.madden.model.film_review import (
    ANNOTATION_SCHEMA,
    empty_annotations,
    load_annotations,
    save_annotations,
)
from cfb_coach.madden.model.learning_sources import (
    evidence_row,
    retain_evidence,
    utc_now,
)

EXPERT_MANIFEST_SCHEMA = "madden.expert.film_manifest.v1"
EXPERT_ANNOTATION_SCHEMA = "madden.expert.annotations.v1"
PERMISSION_STATUSES = frozenset({
    "user_authorized_local",
    "fixture_synthetic",
    "denied",
    "unknown",
})
UNKNOWN = "unknown"


def default_expert_store(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path.home() / ".cfb-coach" / "film" / "expert"


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_expert_manifest(
    *,
    recording_path: str | Path,
    expert_id: str,
    match_id: str,
    game_version: str = "madden27",
    patch: str | None = None,
    competitive_mode: str = "unknown",
    opponent_type: str = "unknown",
    permission_status: str = "user_authorized_local",
    permission_note: str | None = None,
    source_provenance: Mapping[str, Any] | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    if permission_status not in PERMISSION_STATUSES:
        raise ValueError(f"invalid permission_status: {permission_status}")
    path = Path(recording_path)
    digest = fingerprint or (fingerprint_file(path) if path.is_file() else None)
    return {
        "schema": EXPERT_MANIFEST_SCHEMA,
        "film_import_version": FILM_IMPORT_VERSION,
        "recording_id": None if digest is None else digest[:16],
        "fingerprint": digest,
        "path": str(path),
        "expert_id": expert_id,
        "match_id": match_id,
        "game_version": game_version,
        "patch": patch,
        "competitive_mode": competitive_mode,
        "opponent_type": opponent_type,
        "permission_status": permission_status,
        "permission_note": permission_note or (
            "Local recording supplied by the user with processing permission. "
            "Not scraped or redistributed."
        ),
        "source_provenance": dict(source_provenance or {
            "kind": "local_authorized_recording",
            "automatic_download": False,
            "redistribution": False,
        }),
        "raw_video_automatically_understood": False,
        "created_at": utc_now(),
    }


def import_expert_recording(
    path: str | Path,
    *,
    expert_id: str,
    match_id: str,
    store: str | Path | None = None,
    dry_run: bool = True,
    game_version: str = "madden27",
    patch: str | None = None,
    competitive_mode: str = "unknown",
    opponent_type: str = "unknown",
    permission_status: str = "user_authorized_local",
    permission_note: str | None = None,
    manual_anchors: Sequence[Mapping[str, Any]] | None = None,
    skip_decode: bool = False,
) -> dict[str, Any]:
    """Import an authorized expert recording into the expert film store.

    When ``skip_decode`` is true (fixture manifests without real video), only
    the provenance manifest is written. Decode still requires a real local file.
    """
    root = default_expert_store(store)
    source = Path(path)
    if permission_status == "denied":
        return {
            "ok": False,
            "error": "permission_denied",
            "history_modified": False,
            "downloaded": False,
            "redistributed": False,
        }
    if skip_decode:
        if not source.is_file() and permission_status != "fixture_synthetic":
            return {"ok": False, "error": "recording_missing", "history_modified": False}
        digest = fingerprint_file(source) if source.is_file() else f"fixture:{match_id}"
        manifest = build_expert_manifest(
            recording_path=source,
            expert_id=expert_id,
            match_id=match_id,
            game_version=game_version,
            patch=patch,
            competitive_mode=competitive_mode,
            opponent_type=opponent_type,
            permission_status=permission_status,
            permission_note=permission_note,
            fingerprint=digest if len(digest) == 64 else None,
        )
        if permission_status == "fixture_synthetic":
            manifest["fingerprint"] = digest
            manifest["recording_id"] = digest.replace("fixture:", "fx")[:16]
        result = {
            "ok": True,
            "dry_run": dry_run,
            "skip_decode": True,
            "downloaded": False,
            "redistributed": False,
            "raw_video_automatically_understood": False,
            "manifest": manifest,
            "segments": [],
            "wrote_manifest": False,
            "history_modified": False,
        }
        if not dry_run:
            _persist_expert_import(root, manifest, segments=[])
            result["wrote_manifest"] = True
            result["store"] = str(root)
        return result

    imported = import_recording(
        source,
        game_id=match_id,
        store=root,
        dry_run=dry_run,
        manual_anchors=manual_anchors,
    )
    manifest = build_expert_manifest(
        recording_path=source,
        expert_id=expert_id,
        match_id=match_id,
        game_version=game_version,
        patch=patch,
        competitive_mode=competitive_mode,
        opponent_type=opponent_type,
        permission_status=permission_status,
        permission_note=permission_note,
        fingerprint=imported.get("fingerprint"),
    )
    result = {
        **imported,
        "expert_manifest": manifest,
        "downloaded": False,
        "redistributed": False,
        "raw_video_automatically_understood": False,
        "expert_id": expert_id,
        "match_id": match_id,
    }
    if imported.get("ok") and not dry_run:
        _persist_expert_import(root, manifest, segments=imported.get("segments") or [])
        result["expert_manifest_path"] = str(root / "expert_manifests" / f"{match_id}.json")
    return result


def _persist_expert_import(
    store: Path,
    manifest: Mapping[str, Any],
    *,
    segments: Sequence[Mapping[str, Any]],
) -> None:
    manifests = store / "expert_manifests"
    _write_json(manifests / f"{manifest['match_id']}.json", manifest)
    index_path = store / "expert_index.json"
    index = _read_json(index_path)
    matches = dict(index.get("matches") or {})
    fingerprint = manifest.get("fingerprint")
    previous = None
    if fingerprint:
        for mid, row in matches.items():
            if row.get("fingerprint") == fingerprint and mid != manifest["match_id"]:
                previous = mid
                break
    matches[str(manifest["match_id"])] = {
        "expert_id": manifest.get("expert_id"),
        "recording_id": manifest.get("recording_id"),
        "fingerprint": fingerprint,
        "permission_status": manifest.get("permission_status"),
        "duplicate_of": previous,
        "independent_evidence": previous is None,
    }
    index.update({
        "schema": EXPERT_MANIFEST_SCHEMA,
        "matches": matches,
        "updated_at": utc_now(),
    })
    _write_json(index_path, index)
    annotations = empty_expert_annotations(
        match_id=str(manifest["match_id"]),
        recording_id=manifest.get("recording_id"),
        segments=list(segments),
        expert_id=str(manifest.get("expert_id")),
        game_version=str(manifest.get("game_version") or "madden27"),
        patch=manifest.get("patch"),
        competitive_mode=str(manifest.get("competitive_mode") or "unknown"),
        opponent_type=str(manifest.get("opponent_type") or "unknown"),
    )
    ann_path = store / "annotations" / f"{manifest['match_id']}.json"
    if not ann_path.is_file():
        save_annotations(store, annotations)


def empty_expert_annotations(
    *,
    match_id: str,
    recording_id: str | None,
    segments: Sequence[Mapping[str, Any]],
    expert_id: str,
    game_version: str = "madden27",
    patch: str | None = None,
    competitive_mode: str = "unknown",
    opponent_type: str = "unknown",
) -> dict[str, Any]:
    base = empty_annotations(match_id, recording_id, list(segments))
    candidates = []
    for row in base.get("candidates") or []:
        enriched = dict(row)
        enriched.update({
            "expert_id": expert_id,
            "game_version": game_version,
            "patch": patch,
            "competitive_mode": competitive_mode,
            "opponent_type": opponent_type,
            "concept_family": None,
            "formation_family": None,
            "play_action_family": None,
            "offensive_adjustments": None,
            "postsnap_defense": {},
            "play_outcome": None,
            "outcome_visible": False,
            "observation_confidence": row.get("confidence"),
            "human_verification_status": "unreviewed",
            "invisible_fields": [],
        })
        candidates.append(enriched)
    return {
        **base,
        "schema": EXPERT_ANNOTATION_SCHEMA,
        "base_annotation_schema": ANNOTATION_SCHEMA,
        "expert_id": expert_id,
        "match_id": match_id,
        "game_version": game_version,
        "patch": patch,
        "competitive_mode": competitive_mode,
        "opponent_type": opponent_type,
        "candidates": candidates,
        "raw_video_automatically_understood": False,
    }


def annotate_expert_snap(
    store: str | Path,
    match_id: str,
    candidate_id: str,
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a human-reviewed expert snap annotation. Invisible fields stay unknown."""
    payload = load_annotations(store, match_id)
    if payload.get("schema") not in (EXPERT_ANNOTATION_SCHEMA, ANNOTATION_SCHEMA):
        # Allow upgrading a film annotation file in the expert store.
        payload.setdefault("schema", EXPERT_ANNOTATION_SCHEMA)
    match = next(
        (row for row in payload.get("candidates") or [] if row.get("candidate_id") == candidate_id),
        None,
    )
    if match is None:
        return {"ok": False, "error": "unknown_candidate", "saved": False}

    updates = dict(labels)
    invisible = list(updates.pop("invisible_fields", match.get("invisible_fields") or []))
    for field in (
        "formation", "play", "concept_family", "formation_family",
        "play_action_family", "offensive_adjustments",
    ):
        if updates.get(field) in (None, "", UNKNOWN) or field in invisible:
            updates[field] = None
            if field not in invisible:
                invisible.append(field)

    # Game state
    game_state = dict(match.get("game_state") or {})
    for key in ("score", "quarter", "down", "distance", "clock_seconds"):
        if key in updates:
            game_state[key] = updates.pop(key)
    if game_state:
        match["game_state"] = game_state

    defense = dict(match.get("defensive_observation") or {})
    if "presnap_defense" in updates:
        defense.update(dict(updates.pop("presnap_defense") or {}))
    match["defensive_observation"] = defense

    postsnap = dict(match.get("postsnap_defense") or {})
    if "postsnap_defense" in updates:
        postsnap.update(dict(updates.pop("postsnap_defense") or {}))
    match["postsnap_defense"] = postsnap

    if "play_outcome" in updates:
        outcome = updates.pop("play_outcome")
        if outcome in (None, "", UNKNOWN) or not updates.get("outcome_visible", match.get("outcome_visible")):
            if updates.get("outcome_visible") is False or outcome in (None, "", UNKNOWN):
                match["play_outcome"] = None
                match["outcome_visible"] = False
            else:
                match["play_outcome"] = outcome
                match["outcome_visible"] = True
        else:
            match["play_outcome"] = outcome
            match["outcome_visible"] = bool(updates.get("outcome_visible", True))
    if "outcome_visible" in updates:
        match["outcome_visible"] = bool(updates.pop("outcome_visible"))
        if not match["outcome_visible"]:
            match["play_outcome"] = None

    for key, value in updates.items():
        if key in ("observation_confidence", "confidence"):
            match["observation_confidence"] = value
            match["confidence"] = value
        elif key == "human_verification_status":
            match["human_verification_status"] = value
            if value in ("verified_human", "confirmed"):
                match["review_status"] = "confirmed"
                match["human_label"] = "expert_confirmed"
        else:
            match[key] = value

    match["invisible_fields"] = sorted(set(invisible))
    match["unknown_fields"] = sorted(set(list(match.get("unknown_fields") or []) + invisible))
    # Preserve expert identity from the file header when missing on the row.
    match.setdefault("expert_id", payload.get("expert_id"))
    match.setdefault("game_version", payload.get("game_version"))
    match.setdefault("patch", payload.get("patch"))
    match.setdefault("competitive_mode", payload.get("competitive_mode"))
    match.setdefault("opponent_type", payload.get("opponent_type"))

    path = save_annotations(store, payload)
    return {
        "ok": True,
        "saved": True,
        "path": str(path),
        "candidate_id": candidate_id,
        "invisible_fields": match["invisible_fields"],
        "raw_video_automatically_understood": False,
        "history_modified": False,
    }


def reviewed_expert_snaps(store: str | Path, match_id: str | None = None) -> list[dict[str, Any]]:
    root = Path(store)
    files = []
    if match_id:
        path = root / "annotations" / f"{match_id}.json"
        if path.is_file():
            files = [path]
    else:
        files = sorted((root / "annotations").glob("*.json")) if (root / "annotations").is_dir() else []
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = _read_json(path)
        for candidate in payload.get("candidates") or []:
            status = candidate.get("human_verification_status") or candidate.get("review_status")
            if status not in ("verified_human", "confirmed", "corrected"):
                continue
            rows.append({
                **candidate,
                "match_id": payload.get("match_id") or payload.get("game_id"),
                "expert_id": candidate.get("expert_id") or payload.get("expert_id"),
                "game_version": candidate.get("game_version") or payload.get("game_version"),
                "patch": candidate.get("patch") if candidate.get("patch") is not None else payload.get("patch"),
                "competitive_mode": candidate.get("competitive_mode") or payload.get("competitive_mode"),
                "opponent_type": candidate.get("opponent_type") or payload.get("opponent_type"),
            })
    return rows


def export_expert_evidence(
    film_store: str | Path,
    learning_store: str | Path,
    *,
    match_id: str | None = None,
) -> dict[str, Any]:
    """Export reviewed expert snaps into the independent expert evidence store.

    Observed play selections and verified outcomes are stored separately on
    each row. Unverified outcomes remain null.
    """
    snaps = reviewed_expert_snaps(film_store, match_id=match_id)
    saved = 0
    duplicates = 0
    for snap in snaps:
        observed = {
            "formation": snap.get("formation"),
            "play": snap.get("play"),
            "concept_family": snap.get("concept_family"),
            "formation_family": snap.get("formation_family"),
            "play_action_family": snap.get("play_action_family"),
            "offensive_adjustments": snap.get("offensive_adjustments"),
            "demonstrated_action": True,
            "unchosen_failure_claim": False,
        }
        outcome = None
        if snap.get("outcome_visible") and snap.get("play_outcome") is not None:
            outcome = {
                "visible": True,
                "result": snap.get("play_outcome"),
                "verified": snap.get("human_verification_status") in ("verified_human", "confirmed"),
            }
        row = evidence_row(
            category="expert_evidence",
            evidence_id=f"expert:{snap.get('match_id')}:{snap.get('candidate_id')}",
            match_id=str(snap.get("match_id") or ""),
            situation={
                "down": (snap.get("game_state") or {}).get("down"),
                "distance": (snap.get("game_state") or {}).get("distance"),
                "quarter": (snap.get("game_state") or {}).get("quarter"),
                "score": (snap.get("game_state") or {}).get("score"),
                "opponent_type": snap.get("opponent_type"),
                "competitive_mode": snap.get("competitive_mode"),
                "presnap_defense": snap.get("defensive_observation") or {},
                "postsnap_defense": snap.get("postsnap_defense") or {},
            },
            observed_action=observed,
            verified_outcome=outcome,
            provenance={
                "expert_id": snap.get("expert_id"),
                "game_version": snap.get("game_version"),
                "patch": snap.get("patch"),
                "source": "human_reviewed_expert_film",
                "candidate_id": snap.get("candidate_id"),
                "permission_store": str(film_store),
            },
            confidence=snap.get("observation_confidence", snap.get("confidence")),
            human_verification=str(snap.get("human_verification_status") or "confirmed"),
            executed=False,  # expert demonstration, not the user's execution
        )
        # expert_evidence allows executed=False (demonstrated action).
        result = retain_evidence(learning_store, row)
        if result.get("ok"):
            saved += 1
        elif result.get("error") == "duplicate_evidence":
            duplicates += 1
    return {
        "ok": True,
        "exported": saved,
        "duplicates_rejected": duplicates,
        "reviewed_snaps": len(snaps),
        "category": "expert_evidence",
        "observed_separated_from_outcomes": True,
        "history_modified": False,
    }


def load_expert_manifest(store: str | Path, match_id: str) -> dict[str, Any]:
    return _read_json(Path(store) / "expert_manifests" / f"{match_id}.json")


def expert_film_summary(store: str | Path) -> dict[str, Any]:
    root = Path(store)
    index = _read_json(root / "expert_index.json")
    matches = index.get("matches") or {}
    reviewed = reviewed_expert_snaps(root)
    return {
        "schema": EXPERT_MANIFEST_SCHEMA,
        "store": str(root),
        "matches": len(matches),
        "independent_matches": sum(1 for row in matches.values() if row.get("independent_evidence")),
        "duplicate_matches": sum(1 for row in matches.values() if not row.get("independent_evidence")),
        "reviewed_snaps": len(reviewed),
        "experts": sorted({str(row.get("expert_id")) for row in reviewed if row.get("expert_id")}),
        "raw_video_automatically_understood": False,
        "probe_note": (
            "Import reuses film_import/film_review. Labels require human review; "
            "invisible play names stay unknown."
        ),
    }


def catalog_remote_candidates(
    manifest_path: str | Path,
    *,
    store: str | Path | None = None,
) -> dict[str, Any]:
    """Register researched remote VOD candidates without downloading media.

    Writes provenance stubs into the expert film store. Local media paths stay
    empty until the user supplies an authorized recording.
    """
    path = Path(manifest_path)
    payload = _read_json(path)
    if not payload:
        return {"ok": False, "error": "manifest_unreadable", "path": str(path)}
    root = default_expert_store(store)
    catalog_dir = root / "remote_candidates"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in payload.get("candidates") or []:
        match_id = str(row.get("match_id") or "")
        if not match_id:
            continue
        stub = {
            "schema": EXPERT_MANIFEST_SCHEMA,
            "source_catalog": str(path),
            "match_id": match_id,
            "rank": row.get("rank"),
            "title": row.get("title"),
            "expert_ids": list(row.get("players") or []),
            "tournament": row.get("tournament"),
            "date": row.get("date") or row.get("date_range"),
            "game_version": row.get("madden_version") or "madden27",
            "competitive_mode": row.get("competitive_mode") or "unknown",
            "opponent_type": row.get("opponent_type") or "human",
            "view_links": row.get("view_links") or {},
            "permission_status": row.get("permission_status") or "unknown",
            "download_authorized": bool(row.get("download_authorized")),
            "training_authorized": bool(row.get("training_authorized")),
            "local_recording_path": None,
            "local_authorized_copy_available": False,
            "raw_video_automatically_understood": False,
            "estimated_usable_offensive_snaps": row.get("estimated_usable_offensive_snaps"),
            "annotation_quality": row.get("annotation_quality"),
            "recommended_first_match": bool(row.get("recommended_first_match")),
            "rights_holder": "Electronic Arts Inc.",
            "created_at": utc_now(),
            "note": (
                "Remote candidate only. Import an authorized local file before "
                "annotation or training."
            ),
        }
        out = catalog_dir / f"{match_id}.json"
        _write_json(out, stub)
        written.append({"match_id": match_id, "path": str(out), "rank": row.get("rank")})
    index_path = root / "remote_candidate_index.json"
    _write_json(index_path, {
        "schema": "madden.expert.remote_candidates.v1",
        "source_manifest": str(path),
        "updated_at": utc_now(),
        "candidates": written,
        "downloaded": False,
        "training_authorized": False,
    })
    return {
        "ok": True,
        "store": str(root),
        "registered": len(written),
        "candidates": written,
        "downloaded": False,
        "local_authorized_copy_available": False,
        "permission_status": "unknown",
    }
