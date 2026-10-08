"""Training-row export and import. Owner: Data Engineering.

Rows keep recommendations, verified executions, and outcomes in separate
fields. Missing source columns stay unknown. Do not synthesize snaps.
A displayed recommendation is never treated as a verified execution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.learning import SUCCESS_NEED
from cfb_coach.madden.model.schema import (
    CONTRACT_VERSION,
    CSV_UNKNOWN,
    LEAKAGE_FIELDS,
    ExecutedStatus,
    Tri,
    Verification,
)
from cfb_coach.outcome import parse_outcome

_RECOMMENDATION_KEYS = (
    "recommended_formation",
    "recommended_play",
    "recommended_macro",
    "our_call",
)
_EXECUTION_KEYS = (
    "executed_formation",
    "executed_play",
    "executed_status",
    "executed_verification",
)
_LABEL_KEYS = ("success", "stop", "yards", "result", "label_available")


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == CSV_UNKNOWN


def _as_tri(value: Any) -> str:
    if value is None or value == "" or value == CSV_UNKNOWN:
        return Tri.UNKNOWN.value
    if isinstance(value, Tri):
        return value.value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "success"):
        return Tri.TRUE.value
    if text in ("0", "false", "no", "n", "fail", "failure"):
        return Tri.FALSE.value
    if text in ("true", "false", "unknown"):
        return text
    return Tri.UNKNOWN.value


def _success_from_yards(
    *,
    side: str,
    down: Any,
    distance: Any,
    yards: int | None,
    kind: str | None,
) -> str:
    if yards is None and not kind:
        return Tri.UNKNOWN.value
    side_l = (side or "offense").strip().lower()
    is_def = side_l.startswith("d")
    if kind in ("int", "interception", "fumble"):
        return Tri.FALSE.value if not is_def else Tri.TRUE.value
    if kind == "sack":
        return Tri.FALSE.value if not is_def else Tri.TRUE.value
    if yards is None:
        if kind == "incomplete":
            return Tri.FALSE.value if not is_def else Tri.TRUE.value
        return Tri.UNKNOWN.value
    try:
        d = int(down) if down is not None and down != "" else None
        dist = int(distance) if distance is not None and distance != "" else None
    except (TypeError, ValueError):
        return Tri.UNKNOWN.value
    if d is None or dist is None:
        return Tri.UNKNOWN.value
    need = SUCCESS_NEED.get(d, 1.0) * float(dist)
    ok = float(yards) >= need
    if is_def:
        return Tri.TRUE.value if not ok else Tri.FALSE.value
    return Tri.TRUE.value if ok else Tri.FALSE.value


def _normalize_row(raw: Mapping[str, Any], *, provenance: str) -> dict[str, Any]:
    row = {str(k): raw[k] for k in raw}
    side = str(row.get("side") or row.get("possession") or "offense").strip().lower()
    if side.startswith("d"):
        side = "defense"
    else:
        side = "offense"

    formation = row.get("formation") or row.get("recommended_formation")
    play = row.get("play") or row.get("recommended_play")
    recommended_formation = row.get("recommended_formation") or formation
    recommended_play = row.get("recommended_play") or play

    executed_status = str(row.get("executed_status") or ExecutedStatus.UNKNOWN.value).lower()
    executed_verification = str(
        row.get("executed_verification") or Verification.UNKNOWN.value
    ).lower()
    executed_formation = row.get("executed_formation")
    executed_play = row.get("executed_play")

    # Never promote a recommendation to a verified execution.
    if executed_status != ExecutedStatus.IDENTIFIED.value:
        executed_formation = None
        executed_play = None
        executed_verification = Verification.UNKNOWN.value
    elif executed_verification != Verification.VERIFIED.value:
        executed_status = ExecutedStatus.UNKNOWN.value
        executed_formation = None
        executed_play = None
        executed_verification = Verification.UNKNOWN.value

    result = row.get("result")
    yards = row.get("yards")
    kind = None
    if yards is None and result not in (None, ""):
        parsed = parse_outcome(str(result))
        yards = parsed.yards
        kind = parsed.kind
    try:
        yards_i = int(yards) if yards is not None and yards != "" else None
    except (TypeError, ValueError):
        yards_i = None

    success = row.get("success")
    if _blank(success):
        success = _success_from_yards(
            side=side,
            down=row.get("down"),
            distance=row.get("distance"),
            yards=yards_i,
            kind=kind,
        )
    else:
        success = _as_tri(success)

    stop = row.get("stop")
    if _blank(stop):
        if side == "defense" and success != Tri.UNKNOWN.value:
            stop = success
        elif side == "offense" and success == Tri.FALSE.value:
            stop = Tri.TRUE.value
        elif side == "offense" and success == Tri.TRUE.value:
            stop = Tri.FALSE.value
        else:
            stop = Tri.UNKNOWN.value
    else:
        stop = _as_tri(stop)

    label_available = success != Tri.UNKNOWN.value or stop != Tri.UNKNOWN.value
    behavior_propensity = row.get("behavior_propensity")
    if _blank(behavior_propensity):
        behavior_propensity = None

    snap_id = row.get("snap_id") or row.get("id")
    game_id = row.get("game_id") or row.get("opponent_id") or row.get("source_game")

    out = {
        "schema_version": CONTRACT_VERSION,
        "provenance": provenance,
        "snap_id": None if _blank(snap_id) else str(snap_id),
        "game_id": None if _blank(game_id) else str(game_id),
        "session_id": None if _blank(row.get("session_id")) else str(row.get("session_id")),
        "opponent_id": None if _blank(row.get("opponent_id")) else str(row.get("opponent_id")),
        "side": side,
        "down": row.get("down"),
        "distance": row.get("distance"),
        "yardline": row.get("yardline"),
        "quarter": row.get("quarter"),
        "situation_raw": row.get("situation_raw") or row.get("raw"),
        "coverage_hint": row.get("coverage_hint") or row.get("coverage_prediction"),
        "concept_hint": row.get("concept_hint") or row.get("concept_prediction"),
        "recommended_formation": None if _blank(recommended_formation) else str(recommended_formation),
        "recommended_play": None if _blank(recommended_play) else str(recommended_play),
        "recommended_macro": None if _blank(row.get("macro") or row.get("recommended_macro")) else str(
            row.get("macro") or row.get("recommended_macro")
        ),
        "our_call": None if _blank(row.get("our_call")) else str(row.get("our_call")),
        "executed_status": executed_status,
        "executed_verification": executed_verification,
        "executed_formation": None if _blank(executed_formation) else str(executed_formation),
        "executed_play": None if _blank(executed_play) else str(executed_play),
        "result": None if _blank(result) else str(result),
        "yards": yards_i,
        "success": success,
        "stop": stop,
        "label_available": bool(label_available),
        "behavior_propensity": behavior_propensity,
        "propensity_method": row.get("propensity_method") or "unknown",
        "coverage_seen": row.get("coverage_seen"),  # post-snap; never a pre-snap feature
        "concept_seen": row.get("concept_seen"),
        "source_path": row.get("source_path"),
    }
    return out


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report schema gaps, duplicates, label coverage, and leakage risks."""
    missing_fields: dict[str, int] = {}
    labeled = 0
    unlabeled = 0
    verified_exec = 0
    recommendation_only = 0
    duplicates = 0
    seen: set[tuple[Any, ...]] = set()
    leakage_hits = 0
    provenance: dict[str, int] = {}

    required_presence = (
        "snap_id",
        "side",
        "recommended_play",
        "success",
        "executed_status",
    )

    for row in rows:
        prov = str(row.get("provenance") or "unknown")
        provenance[prov] = provenance.get(prov, 0) + 1
        for key in required_presence:
            if _blank(row.get(key)):
                missing_fields[key] = missing_fields.get(key, 0) + 1
        if row.get("label_available"):
            labeled += 1
        else:
            unlabeled += 1
        if str(row.get("executed_status") or "") == ExecutedStatus.IDENTIFIED.value:
            verified_exec += 1
        elif not _blank(row.get("recommended_play")):
            recommendation_only += 1
        key = (
            row.get("game_id"),
            row.get("snap_id"),
            row.get("recommended_formation"),
            row.get("recommended_play"),
            row.get("result"),
        )
        if key in seen and key[1] is not None:
            duplicates += 1
        else:
            seen.add(key)
        # Pre-snap feature maps must not include leakage keys as feature inputs.
        feature_map = row.get("features")
        if isinstance(feature_map, Mapping):
            leaked = LEAKAGE_FIELDS.intersection(feature_map)
            if leaked:
                leakage_hits += 1

    return {
        "n_rows": len(rows),
        "labeled": labeled,
        "unlabeled": unlabeled,
        "verified_executions": verified_exec,
        "recommendation_only": recommendation_only,
        "duplicates": duplicates,
        "missing_fields": missing_fields,
        "leakage_feature_rows": leakage_hits,
        "provenance": provenance,
        "recommendation_keys": list(_RECOMMENDATION_KEYS),
        "execution_keys": list(_EXECUTION_KEYS),
        "label_keys": list(_LABEL_KEYS),
    }


def build_rows(
    *,
    db: Any = None,
    extra_paths: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build feature rows from a Madden database and optional existing files.

    Does not fabricate rows when ``db`` is empty. Historical recommendations
    are kept; they are not silently treated as verified executions.
    """
    rows: list[dict[str, Any]] = []
    if db is not None:
        try:
            snap_rows = db.conn.execute(
                "SELECT * FROM snaps ORDER BY id ASC"
            ).fetchall()
        except Exception:  # noqa: BLE001 — empty or non-madden DB
            snap_rows = []
        for snap in snap_rows:
            mapping = dict(snap)
            mapping["source_path"] = str(getattr(db, "path", "coach.db"))
            rows.append(_normalize_row(mapping, provenance="madden_db.snaps"))
        try:
            play_rows = db.conn.execute(
                "SELECT * FROM play_records ORDER BY id ASC"
            ).fetchall()
        except Exception:  # noqa: BLE001
            play_rows = []
        for play in play_rows:
            mapping = dict(play)
            payload = {}
            raw_payload = mapping.get("payload_json")
            if raw_payload:
                try:
                    payload = json.loads(raw_payload)
                except (TypeError, json.JSONDecodeError):
                    payload = {}
            mapping.setdefault("formation", mapping.get("formation") or payload.get("formation"))
            mapping.setdefault("play", payload.get("play") or mapping.get("our_call"))
            mapping.setdefault("result", mapping.get("result_type") or payload.get("result"))
            mapping.setdefault("situation_raw", payload.get("situation_raw"))
            mapping["source_path"] = str(getattr(db, "path", "coach.db"))
            rows.append(_normalize_row(mapping, provenance="madden_db.play_records"))

    for path in extra_paths:
        rows.extend(import_path(path))
    return rows


def export_jsonl(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    """Write rows as JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True, default=str))
            fh.write("\n")


def import_path(path: str) -> list[dict[str, Any]]:
    """Read CSV, JSON, JSONL, or SQLite into row dicts.

    Older files with missing columns stay readable. Absent fields are unknown.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _import_csv(p)
    if suffix == ".jsonl":
        return _import_jsonl(p)
    if suffix == ".json":
        return _import_json(p)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return _import_sqlite(p)
    # Peek at content for extensionless dumps.
    text = p.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        return _import_json(p)
    if "\t" in text.splitlines()[0] if text else False:
        return _import_csv(p)
    return _import_jsonl(p)


def _import_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            cleaned = {
                k: (None if v == CSV_UNKNOWN or v == "" else v)
                for k, v in raw.items()
            }
            cleaned["source_path"] = str(path)
            rows.append(_normalize_row(cleaned, provenance=f"csv:{path.name}"))
    return rows


def _import_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            raw["source_path"] = str(path)
            rows.append(_normalize_row(raw, provenance=f"jsonl:{path.name}"))
    return rows


def _import_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a list of rows or {{'rows': [...]}}")
    rows: list[dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        raw = dict(raw)
        raw["source_path"] = str(path)
        rows.append(_normalize_row(raw, provenance=f"json:{path.name}"))
    return rows


def _import_sqlite(path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        rows: list[dict[str, Any]] = []
        if "snaps" in names:
            for snap in conn.execute("SELECT * FROM snaps ORDER BY id ASC"):
                mapping = dict(snap)
                mapping["source_path"] = str(path)
                rows.append(_normalize_row(mapping, provenance=f"sqlite:{path.name}:snaps"))
        return rows
    finally:
        conn.close()


def row_content_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
