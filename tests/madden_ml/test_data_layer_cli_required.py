"""argv for backup-db and migrate-db must refuse missing paths before any write."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _snapshot(root: Path) -> tuple:
    if not root.exists():
        return (False, ())
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() or path.is_symlink():
            stat = path.stat()
            rows.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return (True, tuple(rows), root.stat().st_mtime_ns)


@pytest.mark.parametrize(
    "argv",
    [
        ["ml", "backup-db", "--out", "{out}"],
        ["ml", "backup-db", "--db", "{db}"],
        ["ml", "migrate-db"],
        ["ml", "migrate-db", "--backup", "x"],
    ],
    ids=[
        "backup-db-without-db",
        "backup-db-without-out",
        "migrate-db-without-db",
        "migrate-db-backup-without-db",
    ],
)
def test_missing_required_flags_exit_2_and_leave_home_untouched(tmp_path, argv):
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "explicit.db"
    out = tmp_path / "out"
    rendered = [part.format(db=db, out=out) for part in argv]
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = str(_REPO)
    env.pop("CFB_COACH_DB", None)
    env.pop("CFB_COACH_MADDEN_DB", None)
    before_home = _snapshot(home)
    before_tmp = _snapshot(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "cfb_coach", *rendered],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert _snapshot(home) == before_home
    assert list(home.iterdir()) == []
    assert _snapshot(tmp_path) == before_tmp
    assert not db.exists()
    assert not out.exists()
    assert not (tmp_path / "x").exists()
