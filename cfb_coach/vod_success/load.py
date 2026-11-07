"""Read VOD snap batches from CSV, JSON, or JSONL.

The pilot columns (``cfb_coach`` snap fields plus ``video_id``, ``yards``,
``game``) are accepted. ``opponent_type`` is optional. A JSONL manifest can
fill the type and game id when a file omits them. A value already on the
row wins over the manifest.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

_HEADER_RE = re.compile(r"[\s\-]+")

_ALIASES = {
    "opp_type": "opponent_type",
    "opponent_kind": "opponent_type",
    "opponent_type_source": "type_source",
    "off_formation": "off_formation",
    "offense_formation": "off_formation",
    "def_formation": "def_formation",
    "defense_formation": "def_formation",
    "defensive_formation": "def_formation",
}


def load_snaps(path: str | Path, manifest: str | Path | None = None) -> list[dict[str, Any]]:
    """Load every snap under ``path`` (a file or a directory of batch files)."""
    root = Path(path)
    files = _batch_files(root)
    manifest_map = _read_manifest(Path(manifest)) if manifest else {}
    rows: list[dict[str, Any]] = []
    for file in files:
        meta = manifest_map.get(file.name, {})
        for row in _load_file(file):
            rows.append(_apply_manifest(row, meta, file))
    return rows


def load_many(
    paths: Iterable[str | Path],
    manifest: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_snaps(path, manifest=manifest))
    return rows


def _batch_files(root: Path) -> list[Path]:
    if root.is_dir():
        files = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl"}
            and p.name != "manifest.jsonl"
        ]
        return sorted(files)
    if not root.is_file():
        raise FileNotFoundError(root)
    return [root]


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        name = Path(str(obj.get("file") or obj.get("path") or "")).name
        if name:
            out[name] = obj
    return out


def _load_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    return _load_json(path)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for raw in reader:
            row = {_norm_header(k): v for k, v in raw.items() if k}
            rows.append(row)
        return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "counts" in payload and "snaps" not in payload:
        return []
    snaps = _snaps_from_payload(payload)
    return [_norm_row(s) for s in snaps]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(_norm_row(obj))
    return rows


def _snaps_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict)]
    if isinstance(payload, dict):
        snaps = payload.get("snaps")
        if isinstance(snaps, list):
            return [s for s in snaps if isinstance(s, dict)]
        if "play" in payload or "result" in payload:
            return [payload]
    return []


def _norm_header(name: str) -> str:
    key = _HEADER_RE.sub("_", name.strip().lower().lstrip("\ufeff"))
    return _ALIASES.get(key, key)


def _norm_row(raw: dict[str, Any]) -> dict[str, Any]:
    return {_norm_header(str(k)): v for k, v in raw.items()}


def _apply_manifest(
    row: dict[str, Any],
    meta: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    out = dict(row)
    out["_source"] = source.name
    type_value = _blank(out.get("opponent_type"))
    if type_value is None:
        filled = _blank(meta.get("opponent_type"))
        if filled is not None:
            out["opponent_type"] = filled
            out["_type_source"] = "manifest"
        else:
            out["_type_source"] = "missing"
    else:
        out["_type_source"] = "column"
    if _blank(out.get("game_id")) is None:
        game = _blank(meta.get("game_id"))
        if game is not None:
            out["game_id"] = game
    return out


def _blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
