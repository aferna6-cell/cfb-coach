"""Versioned model artifacts. Owner: Registry.

Entries are :class:`~cfb_coach.madden.model.schema.ModelRegistryEntry` values.
Reading a directory that has no entry raises. It does not return a fake model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

from cfb_coach.madden.model.schema import (
    FEATURE_SCHEMA_VERSION,
    GateResult,
    ModelRegistryEntry,
    from_dict,
    to_dict,
)

ENTRY_FILENAME = "registry_entry.json"
ARTIFACT_FILENAME = "model_artifact.json"
CURRENT_LINK = "CURRENT"


def write_entry(entry: ModelRegistryEntry, directory: str) -> str:
    """Write ``entry`` under ``directory`` and return the path."""
    root = Path(directory)
    version = entry.model_version or "anonymous"
    target = root / version
    target.mkdir(parents=True, exist_ok=True)

    artifact_src = Path(entry.artifact_path) if entry.artifact_path else None
    artifact_dest = target / ARTIFACT_FILENAME
    artifact_sha = entry.artifact_sha256
    if artifact_src is not None and artifact_src.exists():
        if artifact_src.resolve() != artifact_dest.resolve():
            shutil.copy2(artifact_src, artifact_dest)
        raw = artifact_dest.read_bytes()
        artifact_sha = hashlib.sha256(raw).hexdigest()
    elif entry.artifact_path and not (artifact_src and artifact_src.exists()):
        # Keep the entry but do not invent an artifact.
        artifact_dest = target / ARTIFACT_FILENAME

    stored = ModelRegistryEntry(
        feature_schema_version=entry.feature_schema_version,
        model_version=entry.model_version,
        code_version=entry.code_version,
        game_version=entry.game_version,
        data_hashes=entry.data_hashes,
        metrics=entry.metrics,
        gate=entry.gate,
        gate_passed=entry.gate_passed,
        gate_report_path=entry.gate_report_path,
        seed=entry.seed,
        trained_at=entry.trained_at,
        created_ts=entry.created_ts,
        artifact_path=str(artifact_dest) if artifact_dest.exists() else entry.artifact_path,
        artifact_sha256=artifact_sha,
        data_hash=entry.data_hash,
        side=entry.side,
        opponent_type_scope=entry.opponent_type_scope,
        title_update=entry.title_update,
    )
    entry_path = target / ENTRY_FILENAME
    entry_path.write_text(
        json.dumps(to_dict(stored), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current = root / CURRENT_LINK
    current.write_text(version + "\n", encoding="utf-8")
    return str(entry_path)


def read_entry(directory: str) -> ModelRegistryEntry:
    """Load the entry stored in ``directory``.

    ``directory`` may be a version folder, a registry root with CURRENT, or a
    direct path to ``registry_entry.json``.
    """
    path = Path(directory)
    if path.is_file():
        entry_path = path
    elif (path / ENTRY_FILENAME).is_file():
        entry_path = path / ENTRY_FILENAME
    elif (path / CURRENT_LINK).is_file():
        version = (path / CURRENT_LINK).read_text(encoding="utf-8").strip()
        entry_path = path / version / ENTRY_FILENAME
    else:
        raise FileNotFoundError(f"no registry entry under {directory}")

    try:
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupted registry entry: {entry_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"corrupted registry entry: {entry_path}")

    entry = from_dict(payload)
    if not isinstance(entry, ModelRegistryEntry):
        raise ValueError(f"not a ModelRegistryEntry: {entry_path}")
    if entry.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"incompatible feature schema {entry.feature_schema_version!r}; "
            f"expected {FEATURE_SCHEMA_VERSION}"
        )
    if entry.artifact_path:
        art = Path(entry.artifact_path)
        if not art.is_file():
            # Try sibling artifact in the same folder.
            sibling = entry_path.parent / ARTIFACT_FILENAME
            if sibling.is_file():
                entry = replace(entry, artifact_path=str(sibling))
            else:
                raise FileNotFoundError(f"missing model artifact: {entry.artifact_path}")
        if entry.artifact_sha256:
            digest = hashlib.sha256(Path(entry.artifact_path).read_bytes()).hexdigest()
            if digest != entry.artifact_sha256:
                raise ValueError(f"artifact checksum mismatch for {entry.artifact_path}")
    return entry


def promotion_allowed(entry: ModelRegistryEntry) -> bool:
    """Whether hybrid mode may select ``entry``.

    Hybrid stays off unless the evaluation gate is ``passed`` and
    ``gate_passed`` is true. Shadow analysis may still load the artifact.
    """
    if entry.gate is not GateResult.PASSED:
        return False
    if entry.gate_passed is not True:
        return False
    if entry.feature_schema_version != FEATURE_SCHEMA_VERSION:
        return False
    if not entry.artifact_path or not Path(entry.artifact_path).is_file():
        return False
    return True


def list_entries(directory: str) -> list[ModelRegistryEntry]:
    """List readable versioned entries under a registry root."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found: list[ModelRegistryEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        entry_path = child / ENTRY_FILENAME
        if not entry_path.is_file():
            continue
        try:
            found.append(read_entry(str(child)))
        except (OSError, ValueError, FileNotFoundError, TypeError):
            continue
    return found


def set_current(directory: str, model_version: str) -> None:
    root = Path(directory)
    target = root / model_version / ENTRY_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"unknown model version {model_version!r}")
    (root / CURRENT_LINK).write_text(model_version + "\n", encoding="utf-8")
