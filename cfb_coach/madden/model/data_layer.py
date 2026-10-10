"""Path-explicit Madden data operations.

Migration, the backup helper, and shadow export take every database and
output path from the caller. They do not read ``CFB_COACH_DB``,
``CFB_COACH_MADDEN_DB``, ``data_dir()``, or ``~/.cfb-coach``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden.data import load_seed
from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import user_game_learning as history


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


def migrate_database(source: Path | str, backup: Path | str) -> Path:
    """Migrate ``source`` in place. Refuses to run without an existing backup file."""
    src = _absolute_path(source, label="source")
    backup_path = _absolute_path(backup, label="backup")
    if not src.is_file():
        raise FileNotFoundError(f"no database at {src}")
    if not backup_path.is_file():
        raise ValueError(
            "Refusing to migrate without an existing verified backup file."
        )
    db = CoachDB(src, seed=load_seed(), read_only=False, migrate=True)
    try:
        return src
    finally:
        db.close()


def export_shadow(
    source: Path | str,
    out_dir: Path | str,
    *,
    max_games: int = 4,
) -> dict:
    """Read ``source`` read-only and write isolated shadow artifacts under ``out_dir``.

    Does not install a model, write a registry ``CURRENT`` link, or change live mode.
    An empty database exports no artifact files.
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
    report = history.train_shadow_models(rows, out_dir=folder, max_games=max_games)
    artifacts = report.get("artifacts") or {}
    report["source"] = str(src)
    report["out_dir"] = str(folder)
    report["paths"] = {
        "source": str(src),
        "out_dir": str(folder),
        "artifacts": [str(path) for path in artifacts.values()],
    }
    report["installed"] = False
    report["live_mode_changed"] = False
    return report
