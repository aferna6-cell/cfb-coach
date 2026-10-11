"""Home isolation for the path-explicit data layer.

Migration, the backup helper, and shadow export must take every path from
explicit arguments or the temp fixture. An empty temporary HOME must stay
untouched, including ``~/.cfb-coach``.
"""

from __future__ import annotations

import builtins
import os
import sqlite3
from argparse import Namespace
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from cfb_coach.madden.model.cli import (
    cmd_ml_backup,
    cmd_ml_migrate,
    cmd_ml_shadow_export,
)
from cfb_coach.madden.model.data_layer import (
    backup_database,
    export_shadow,
    migrate_database,
)

_PATH_FUNCS = (
    "open",
    "stat",
    "lstat",
    "mkdir",
    "makedirs",
    "listdir",
    "scandir",
    "access",
    "rename",
    "replace",
    "remove",
    "unlink",
    "link",
    "symlink",
    "chmod",
)


def _path_text(target) -> str | None:
    if isinstance(target, int) or target is None:
        return None
    try:
        raw = os.fspath(target)
    except TypeError:
        return None
    if isinstance(raw, bytes):
        text = os.fsdecode(raw)
    else:
        text = raw
    if text.startswith("file:"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
    return text


def _under_root(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _install_home_guard(monkeypatch: pytest.MonkeyPatch, roots: list[Path]) -> None:
    """Fail the test if any wrapped call names a path under ``~/.cfb-coach``."""
    forbidden = [os.path.normpath(str(root)) for root in roots]

    def touches(target) -> bool:
        text = _path_text(target)
        if not text:
            return False
        path = Path(text)
        if not path.is_absolute():
            path = Path.cwd() / path
        norm = os.path.normpath(str(path))
        return any(norm == root or norm.startswith(root + os.sep) for root in forbidden)

    def reject(target) -> None:
        if touches(target):
            raise AssertionError(f"accessed ~/.cfb-coach path: {target}")

    def wrap(real):
        def inner(*args, **kwargs):
            for arg in args:
                reject(arg)
            for key, value in kwargs.items():
                if key.endswith("_fd"):
                    continue
                reject(value)
            return real(*args, **kwargs)

        return inner

    for name in _PATH_FUNCS:
        monkeypatch.setattr(os, name, wrap(getattr(os, name)))
    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        reject(database)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)


def _refuse_home_default(*_args, **_kwargs):
    raise AssertionError("data operation consulted a home-directory default path")


def _assert_explicit(path: Path, fixture: Path) -> None:
    assert path.is_absolute()
    assert _under_root(path, fixture)
    assert ".cfb-coach" not in path.parts


def _home_coach_snapshot(root: Path, *, stat, scandir) -> dict | None:
    """Names, sizes, and mtimes under ``root``, or None if the directory is absent.

    ``stat`` and ``scandir`` are the unwrapped os functions so the assertion
    can inspect a pre-existing home directory after the access guard is on.
    """
    try:
        dir_stat = stat(root)
    except FileNotFoundError:
        return None
    files: dict[str, tuple[int, int]] = {}

    def walk(directory: Path) -> None:
        with scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    walk(path)
                    continue
                if entry.is_file(follow_symlinks=False):
                    file_stat = stat(path)
                    files[str(path.relative_to(root))] = (
                        file_stat.st_size,
                        file_stat.st_mtime_ns,
                    )

    walk(root)
    return {"mtime_ns": dir_stat.st_mtime_ns, "files": files}


def test_migration_backup_and_shadow_export_use_only_explicit_paths(tmp_path, monkeypatch):
    real_home = Path(os.environ["HOME"])
    home = tmp_path / "empty-home"
    fixture = tmp_path / "fixture"
    home.mkdir()
    fixture.mkdir()
    backup_dir = fixture / "backups"
    shadow_dir = fixture / "shadow"
    source = fixture / "madden27.db"
    # Other tests in this process may already have created the real home
    # directory. Snapshot it before the guard so this test fails only if the
    # data operations create or change it.
    real_cfb = real_home / ".cfb-coach"
    real_stat = os.stat
    real_scandir = os.scandir
    before_real = _home_coach_snapshot(real_cfb, stat=real_stat, scandir=real_scandir)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("CFB_COACH_DB", raising=False)
    monkeypatch.delenv("CFB_COACH_MADDEN_DB", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    _install_home_guard(
        monkeypatch,
        [home / ".cfb-coach", real_home / ".cfb-coach"],
    )
    for target in (
        "cfb_coach.madden.model.cli.madden_db_path",
        "cfb_coach.madden.model.cli.data_dir",
        "cfb_coach.db.default_db_path",
        "cfb_coach.games.madden_db_path",
        "cfb_coach.games.data_dir",
        "cfb_coach.prep_browser.default_prep_dir",
    ):
        monkeypatch.setattr(target, _refuse_home_default)

    sqlite3.connect(source).close()

    backup = backup_database(source, backup_dir)
    migrated = migrate_database(source, backup)
    report = export_shadow(source, shadow_dir, max_games=4)

    _assert_explicit(backup, fixture)
    _assert_explicit(migrated, fixture)
    _assert_explicit(Path(report["source"]), fixture)
    _assert_explicit(Path(report["out_dir"]), fixture)
    for artifact in report["paths"]["artifacts"]:
        _assert_explicit(Path(artifact), fixture)
    assert report["paths"]["source"] == str(source)
    assert report["paths"]["out_dir"] == str(shadow_dir)
    assert report["installed"] is False
    assert report["live_mode_changed"] is False
    assert backup.is_file()
    assert backup.parent == backup_dir

    with sqlite3.connect(source) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "ml_decisions" in tables
    assert "ml_outcomes" in tables

    cli_backup = cmd_ml_backup(Namespace(db=str(source), out=str(backup_dir)))
    cli_migrate = cmd_ml_migrate(Namespace(db=str(source), backup=str(backup)))
    cli_export = cmd_ml_shadow_export(
        Namespace(db=str(source), out=str(shadow_dir), games=4)
    )
    assert cli_backup == 0
    assert cli_migrate == 0
    assert cli_export == 0

    assert os.listdir(home) == []
    assert (
        _home_coach_snapshot(real_cfb, stat=real_stat, scandir=real_scandir)
        == before_real
    )
    written = [path for path in fixture.rglob("*") if path.is_file()]
    assert written
    for path in written:
        _assert_explicit(path, fixture)
