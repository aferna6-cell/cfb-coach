"""Explicit learning evidence sources for Sprint 15A.

Expert VODs, personal verified gameplay, and general football knowledge stay
in separate stores. They are never mixed into one undifferentiated training
set. The trained play-success model remains the primary live decision signal;
these sources produce calibrated, auditable priors only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_CATEGORIES = (
    "expert_evidence",
    "personal_evidence",
    "general_football_evidence",
)

SCHEMA = "madden.learning.sources.v1"
EVIDENCE_SCHEMA = "madden.learning.evidence_row.v1"

# Older VOD call-prior path (vod_model.apply_madden_call) must not override
# the model-primary joint decision. Documented for audits.
VOD_PRIOR_INTERACTION = {
    "vod_success": "offline trainer that writes v0.7 beater cells",
    "vod_model": (
        "heuristic-path call prior used by prep/playcaller late swap; "
        "not called by the experimental coordinator joint decision path"
    ),
    "experimental_joint": (
        "model-primary play-success scoring inside choose_joint_action; "
        "expert/personal signals enter only via bounded expert_signal"
    ),
    "double_counting_guard": (
        "expert VOD rows never feed vod_model cells and the expert_signal "
        "prior in the same decision; vod_model remains off the joint path"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_learning_store(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path.home() / ".cfb-coach" / "learning"


def ensure_store(store: str | Path) -> dict[str, Path]:
    root = Path(store)
    paths = {
        "root": root,
        "expert": root / "expert",
        "personal": root / "personal",
        "general": root / "general",
        "artifacts": root / "artifacts",
        "index": root / "index.json",
    }
    for key in ("expert", "personal", "general", "artifacts"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if not paths["index"].is_file():
        paths["index"].write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "created_at": utc_now(),
                    "categories": list(SOURCE_CATEGORIES),
                    "vod_prior_interaction": VOD_PRIOR_INTERACTION,
                    "fingerprints": {},
                    "counts": {name: 0 for name in SOURCE_CATEGORIES},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def fingerprint_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def evidence_row(
    *,
    category: str,
    evidence_id: str,
    match_id: str | None,
    situation: Mapping[str, Any],
    observed_action: Mapping[str, Any] | None,
    verified_outcome: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    confidence: float | None,
    human_verification: str,
    executed: bool,
) -> dict[str, Any]:
    """Build one auditable evidence row. Unexecuted recommendations stay out of personal training."""
    if category not in SOURCE_CATEGORIES:
        raise ValueError(f"unknown evidence category: {category}")
    if category == "personal_evidence" and not executed:
        raise ValueError("personal evidence requires a verified executed play")
    row = {
        "schema": EVIDENCE_SCHEMA,
        "category": category,
        "evidence_id": evidence_id,
        "match_id": match_id,
        "situation": dict(situation),
        "observed_action": None if observed_action is None else dict(observed_action),
        "verified_outcome": None if verified_outcome is None else dict(verified_outcome),
        "provenance": dict(provenance),
        "confidence": confidence,
        "human_verification": human_verification,
        "executed": bool(executed),
        "recorded_at": utc_now(),
    }
    row["fingerprint"] = fingerprint_payload(
        {k: row[k] for k in (
            "category", "evidence_id", "match_id", "situation",
            "observed_action", "verified_outcome", "provenance",
            "confidence", "human_verification", "executed",
        )}
    )
    return row


def retain_evidence(
    store: str | Path,
    row: Mapping[str, Any],
    *,
    allow_duplicate: bool = False,
) -> dict[str, Any]:
    """Persist one evidence row under its category. Duplicates are rejected by default."""
    paths = ensure_store(store)
    category = str(row.get("category") or "")
    if category not in SOURCE_CATEGORIES:
        return {"ok": False, "error": "unknown_category", "saved": False}
    digest = str(row.get("fingerprint") or fingerprint_payload(row))
    index = _read_json(paths["index"])
    fingerprints = dict(index.get("fingerprints") or {})
    if digest in fingerprints and not allow_duplicate:
        return {
            "ok": False,
            "error": "duplicate_evidence",
            "saved": False,
            "duplicate_of": fingerprints[digest],
            "independent_evidence": False,
        }
    folder = {
        "expert_evidence": paths["expert"],
        "personal_evidence": paths["personal"],
        "general_football_evidence": paths["general"],
    }[category]
    destination = folder / f"{row.get('evidence_id') or digest[:16]}.json"
    body = dict(row)
    body["fingerprint"] = digest
    _write_json(destination, body)
    fingerprints[digest] = str(destination)
    index["fingerprints"] = fingerprints
    counts = dict(index.get("counts") or {})
    counts[category] = int(counts.get(category) or 0) + 1
    index["counts"] = counts
    index["schema"] = SCHEMA
    index["updated_at"] = utc_now()
    _write_json(paths["index"], index)
    return {
        "ok": True,
        "saved": True,
        "path": str(destination),
        "category": category,
        "fingerprint": digest,
        "independent_evidence": True,
        "history_modified": False,
    }


def list_evidence(
    store: str | Path,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    paths = ensure_store(store)
    folders = []
    if category is None:
        folders = [paths["expert"], paths["personal"], paths["general"]]
    elif category == "expert_evidence":
        folders = [paths["expert"]]
    elif category == "personal_evidence":
        folders = [paths["personal"]]
    elif category == "general_football_evidence":
        folders = [paths["general"]]
    else:
        return []
    rows: list[dict[str, Any]] = []
    for folder in folders:
        for path in sorted(folder.glob("*.json")):
            payload = _read_json(path)
            if payload:
                rows.append(payload)
    return rows


def source_summary(store: str | Path) -> dict[str, Any]:
    paths = ensure_store(store)
    index = _read_json(paths["index"])
    rows = list_evidence(store)
    by_category = {name: [] for name in SOURCE_CATEGORIES}
    for row in rows:
        by_category.setdefault(str(row.get("category")), []).append(row)
    return {
        "schema": SCHEMA,
        "store": str(paths["root"]),
        "counts": {name: len(by_category.get(name) or []) for name in SOURCE_CATEGORIES},
        "index_counts": index.get("counts") or {},
        "vod_prior_interaction": VOD_PRIOR_INTERACTION,
        "expert_matches": sorted({
            str(row.get("match_id"))
            for row in by_category.get("expert_evidence") or []
            if row.get("match_id")
        }),
        "personal_matches": sorted({
            str(row.get("match_id"))
            for row in by_category.get("personal_evidence") or []
            if row.get("match_id")
        }),
        "general_entries": len(by_category.get("general_football_evidence") or []),
        "retained_independently": True,
        "undifferentiated_mix": False,
    }


def ingest_general_from_football_knowledge(store: str | Path) -> dict[str, Any]:
    """Mirror compiled football knowledge layers as general evidence (read-only copy)."""
    from cfb_coach.madden.model.football_knowledge import LAYERS, SOURCES

    saved = 0
    skipped = 0
    for source_id, meta in SOURCES.items():
        row = evidence_row(
            category="general_football_evidence",
            evidence_id=f"general:{source_id}",
            match_id=None,
            situation={"scope": "compiled_knowledge"},
            observed_action=None,
            verified_outcome=None,
            provenance={
                "source_id": source_id,
                "layer": meta.get("layer"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "game_versions": list(meta.get("game_versions") or []),
                "note": meta.get("note"),
                "available_layers": list(LAYERS),
            },
            confidence=meta.get("confidence"),
            human_verification="compiled_source",
            executed=False,
        )
        result = retain_evidence(store, row)
        if result.get("ok"):
            saved += 1
        else:
            skipped += 1
    return {
        "ok": True,
        "saved": saved,
        "skipped_duplicates": skipped,
        "category": "general_football_evidence",
        "uncertainty_preserved": True,
    }


def personal_rows_from_verified_snaps(
    snaps: Sequence[Mapping[str, Any]],
    *,
    playbook: Mapping[str, Sequence[str]] | None = None,
    roster: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build personal evidence only from verified executed snaps."""
    rows: list[dict[str, Any]] = []
    for snap in snaps:
        if str(snap.get("executed_verification") or "").lower() != "verified":
            continue
        status = str(snap.get("executed_status") or "").lower()
        if status and status not in ("identified", "verified"):
            continue
        play = snap.get("executed_play")
        formation = snap.get("executed_formation")
        if not play or not formation:
            continue
        # Never promote an unexecuted recommendation into personal training.
        if snap.get("recommended_only"):
            continue
        evidence_id = f"personal:{snap.get('snap_id') or snap.get('ml_snap_id')}"
        rows.append(evidence_row(
            category="personal_evidence",
            evidence_id=evidence_id,
            match_id=str(snap.get("game_id") or snap.get("session_id") or ""),
            situation={
                "down": snap.get("down"),
                "distance": snap.get("distance"),
                "quarter": snap.get("quarter"),
                "score_diff": snap.get("score_diff"),
                "opponent_category": snap.get("opponent_id") or snap.get("opponent_type"),
                "playbook_installed": bool(playbook),
                "roster_verified": bool(roster),
            },
            observed_action={
                "formation": formation,
                "play": play,
                "adjustments": snap.get("executed_adjustments") or snap.get("adjustments"),
                "playbook_ref": None if playbook is None else "installed",
                "roster_ref": None if roster is None else "verified",
            },
            verified_outcome={
                "success": snap.get("success"),
                "yards": snap.get("yards"),
                "result": snap.get("result") or snap.get("outcome"),
            },
            provenance={
                "source": "user_verified_gameplay",
                "snap_id": snap.get("snap_id") or snap.get("ml_snap_id"),
                "game_version": snap.get("game_version") or "madden27",
            },
            confidence=1.0,
            human_verification="verified_execution",
            executed=True,
        ))
    return rows
