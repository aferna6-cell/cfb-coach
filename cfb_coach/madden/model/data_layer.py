"""Path-explicit Madden data operations.

Migration, the backup helper, and shadow export take every database and
output path from the caller. They do not read ``CFB_COACH_DB``,
``CFB_COACH_MADDEN_DB``, ``data_dir()``, or ``~/.cfb-coach``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import user_game_learning as history
from cfb_coach.madden.model.dataset import ELIGIBILITY_RECOMMENDATION_ONLY
from cfb_coach.madden.model.schema import LEAKAGE_FIELDS

SCHEMA_KEY = "madden_ml_data_schema"
SCHEMA_VERSION = "1"

# Recorded opponent package. ``empty`` means the set was observed and had no
# package. NULL means the set was not recorded. Those are different facts.
OPP_SET_VALUES = (
    "empty",
    "base",
    "nickel",
    "dime",
    "dollar",
    "quarter",
    "goal_line",
    "unknown",
)
OPP_PERSONNEL_VALUES = (
    "00",
    "01",
    "02",
    "10",
    "11",
    "12",
    "13",
    "20",
    "21",
    "22",
    "23",
    "base",
    "nickel",
    "dime",
    "goal_line",
    "unknown",
)
OPP_LOOK_VERIFICATION_VALUES = ("unknown", "unverified", "verified")

_PLAIN_SNAP_COLUMNS = (
    ("ml_snap_id", "TEXT"),
    ("opponent_type", "TEXT"),
    ("executed_status", "TEXT"),
    ("executed_formation", "TEXT"),
    ("executed_play", "TEXT"),
    ("executed_macro", "TEXT"),
    ("executed_verification", "TEXT"),
    ("accepted", "TEXT"),
)
_CHECK_SNAP_COLUMNS = {
    "opp_personnel": OPP_PERSONNEL_VALUES,
    "opp_set": OPP_SET_VALUES,
    "opp_look_verification": OPP_LOOK_VERIFICATION_VALUES,
}

# Decision-time columns only. Post-snap labels and outcomes stay out.
SHADOW_FEATURE_FIELDS = (
    "down",
    "distance",
    "yardline",
    "quarter",
    "side",
    "opponent_id",
    "situation_raw",
    "recommended_formation",
    "recommended_play",
)
SHADOW_FEATURE_DENYLIST = LEAKAGE_FIELDS | frozenset(
    {
        "coverage_seen",
        "concept_seen",
        "executed_formation",
        "executed_status",
        "executed_verification",
        "executed_macro",
        "action_play",
        "action_formation",
        "label_available",
        "accepted",
        "outcome",
        "outcome_json",
        "yards_gained",
        "eligibility",
        "supervised_eligible",
        "penalty",
        "executed_adjustment_id",
        "snap_link",
    }
)
_FEATURE_LEAK = SHADOW_FEATURE_DENYLIST.intersection(SHADOW_FEATURE_FIELDS)
if _FEATURE_LEAK:
    raise RuntimeError(f"shadow features overlap the outcome denylist: {sorted(_FEATURE_LEAK)}")

_GAME_DIR = re.compile(r"[^A-Za-z0-9._-]+")


def _absolute_path(value: Path | str, *, label: str) -> Path:
    """Accept only an absolute caller-supplied path.

    ``expanduser`` is intentionally not applied: a leading ``~`` would resolve
    under the home directory.
    """
    if isinstance(value, Path):
        text = str(value)
        path = value
    else:
        text = str(value)
        path = Path(text)
    if text.startswith("~"):
        raise ValueError(f"{label} must be an explicit absolute path, got {value!r}")
    if not path.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path, got {value!r}")
    return path


def backup_database(source: Path | str, dest_dir: Path | str) -> Path:
    """Backup helper. Consistent SQLite backup of ``source`` into ``dest_dir``.

    The source is opened read-only, so backup never migrates or writes it.
    """
    src = _absolute_path(source, label="source")
    folder = _absolute_path(dest_dir, label="dest_dir")
    if not src.is_file():
        raise FileNotFoundError(f"no database at {src}")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = folder / f"{src.stem}.backup.{stamp}{src.suffix}"
    db = CoachDB.open_read_only(src)
    try:
        return db.backup_to(dest)
    finally:
        db.close()


def backup_manifest_path(backup: Path | str) -> Path:
    """Sidecar written next to a migration backup. The hash is of that copy."""
    path = Path(backup)
    return Path(str(path) + ".sha256.json")


def migrate_database(source: Path | str, backup: Path | str) -> Path:
    """Migrate ``source`` in place.

    The backup file and its sha256 manifest are written before any schema or
    row change. A failed backup leaves ``source`` untouched. A second call,
    once the schema version is recorded, does not open the database for write
    and does not rewrite existing rows. Recommendations are never copied into
    ``executed_*`` or ``accepted``.
    """
    src = _absolute_path(source, label="source")
    backup_path = _absolute_path(backup, label="backup")
    if not src.is_file():
        raise FileNotFoundError(f"no database at {src}")
    if _migration_complete(src):
        return src
    _write_backup_and_manifest(src, backup_path)
    _apply_migration(src)
    return src


def export_shadow(
    source: Path | str,
    out_dir: Path | str,
    *,
    max_games: int = 4,
) -> dict:
    """Read ``source`` read-only and write isolated shadow artifacts under ``out_dir``.

    Feature files are split by game and by side. Their columns are the
    pre-snap allow-list only. Recommendation-only rows and unverified
    ``coverage_seen`` values are flags, not feature inputs. Does not install
    a model or change live mode.
    """
    src = _absolute_path(source, label="source")
    folder = _absolute_path(out_dir, label="out_dir")
    if not src.is_file():
        raise FileNotFoundError(f"no database at {src}")
    db = CoachDB.open_read_only(src)
    try:
        rows = dataset_mod.build_rows(db=db)
    finally:
        db.close()
    feature_files = _write_shadow_features(rows, folder)
    report = history.train_shadow_models(rows, out_dir=folder, max_games=max_games)
    artifacts = report.get("artifacts") or {}
    written = [str(path) for path in artifacts.values()] + feature_files
    report["source"] = str(src)
    report["out_dir"] = str(folder)
    report["feature_files"] = feature_files
    report["paths"] = {
        "source": str(src),
        "out_dir": str(folder),
        "artifacts": written,
    }
    report["installed"] = False
    report["live_mode_changed"] = False
    return report


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _migration_complete(path: Path) -> bool:
    conn = _connect_readonly(path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "meta" not in tables or "snaps" not in tables:
            return False
        if "ml_decisions" not in tables or "ml_outcomes" not in tables:
            return False
        version = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (SCHEMA_KEY,)
        ).fetchone()
        if version is None or version[0] != SCHEMA_VERSION:
            return False
        columns = {row[1] for row in conn.execute("PRAGMA table_info(snaps)")}
        required = {name for name, _typ in _PLAIN_SNAP_COLUMNS} | set(_CHECK_SNAP_COLUMNS)
        if not required.issubset(columns):
            return False
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'snaps'"
        ).fetchone()
        sql = ddl[0] if ddl and ddl[0] else ""
        return all(
            f"{name} TEXT CHECK" in sql and f"{name} IS NULL OR {name} IN" in sql
            for name in _CHECK_SNAP_COLUMNS
        )
    finally:
        conn.close()


def _write_backup_and_manifest(src: Path, backup_path: Path) -> None:
    """Copy ``src`` to ``backup_path`` and record sha256 of that copy.

    Raises if the copy cannot be written. Does not modify ``src``.
    """
    db = CoachDB.open_read_only(src)
    try:
        try:
            db.backup_to(backup_path)
        except sqlite3.Error as exc:
            raise OSError(f"backup failed before migration: {exc}") from exc
    finally:
        db.close()
    digest = hashlib.sha256(backup_path.read_bytes()).hexdigest()
    manifest = backup_manifest_path(backup_path)
    payload = {"backup": backup_path.name, "sha256": digest}
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["sha256"]
    if recorded != hashlib.sha256(backup_path.read_bytes()).hexdigest():
        raise RuntimeError("backup manifest hash does not match the copy")


def _check_clause(column: str, allowed: tuple[str, ...]) -> str:
    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in allowed)
    return f"{column} IS NULL OR {column} IN ({quoted})"


def _add_column(conn: sqlite3.Connection, column: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(snaps)")}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE snaps ADD COLUMN {ddl}")


def _apply_migration(src: Path) -> None:
    """Additive schema only. Does not update existing snap values."""
    conn = sqlite3.connect(src)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ml_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_ts TEXT NOT NULL,
                game_id TEXT,
                snap_id TEXT,
                session_id TEXT,
                snap_seq INTEGER,
                mode TEXT NOT NULL,
                model_version TEXT,
                policy_version TEXT,
                policy_source TEXT,
                heuristic_formation TEXT,
                heuristic_play TEXT,
                final_formation TEXT,
                final_play TEXT,
                shadow_formation TEXT,
                shadow_play TEXT,
                shadow_status TEXT,
                agree INTEGER,
                candidates_json TEXT,
                decision_json TEXT NOT NULL,
                latency_ms_model REAL,
                created_ts TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ml_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snap_id TEXT NOT NULL,
                game_id TEXT,
                decision_id INTEGER,
                executed_status TEXT,
                executed_formation TEXT,
                executed_play TEXT,
                executed_verification TEXT,
                outcome_json TEXT,
                created_ts TEXT NOT NULL
            );
            """
        )
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "snaps" not in tables:
            checks = ", ".join(
                f"{name} TEXT CHECK ({_check_clause(name, allowed)})"
                for name, allowed in _CHECK_SNAP_COLUMNS.items()
            )
            plains = ", ".join(f"{name} {typ}" for name, typ in _PLAIN_SNAP_COLUMNS)
            conn.execute(
                f"""
                CREATE TABLE snaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    opponent_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    {plains},
                    {checks}
                )
                """
            )
        else:
            for name, typ in _PLAIN_SNAP_COLUMNS:
                _add_column(conn, name, f"{name} {typ}")
            for name, allowed in _CHECK_SNAP_COLUMNS.items():
                _add_column(conn, name, f"{name} TEXT CHECK ({_check_clause(name, allowed)})")
        conn.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (SCHEMA_KEY, SCHEMA_VERSION),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _game_dirname(game_id: str) -> str:
    cleaned = _GAME_DIR.sub("_", game_id).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"game id cannot be a path segment: {game_id!r}")
    return cleaned


def _shadow_features(row: dict) -> dict:
    features = {name: row.get(name) for name in SHADOW_FEATURE_FIELDS}
    leaked = SHADOW_FEATURE_DENYLIST.intersection(features)
    if leaked:
        raise RuntimeError(f"shadow feature leaked outcome fields: {sorted(leaked)}")
    return features


def _write_shadow_features(rows: list[dict], folder: Path) -> list[str]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        game_id = str(row.get("game_id") or "unknown-game")
        side = "defense" if str(row.get("side") or "") == "defense" else "offense"
        grouped.setdefault(game_id, {"offense": [], "defense": []})[side].append(row)
    written: list[str] = []
    for game_id, sides in grouped.items():
        game_dir = folder / "by_game" / _game_dirname(game_id)
        game_dir.mkdir(parents=True, exist_ok=True)
        for side in ("offense", "defense"):
            path = game_dir / f"{side}.jsonl"
            lines = []
            for row in sides[side]:
                coverage = row.get("coverage_seen")
                verified = str(row.get("executed_verification") or "") == "verified"
                record = {
                    "snap_id": row.get("snap_id"),
                    "game_id": game_id,
                    "side": side,
                    "features": _shadow_features(row),
                    "recommendation_only": row.get("eligibility") == ELIGIBILITY_RECOMMENDATION_ONLY,
                    "unverified_coverage_seen": bool(str(coverage or "").strip()) and not verified,
                }
                lines.append(json.dumps(record, sort_keys=True, default=str))
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written.append(str(path))
    return written
