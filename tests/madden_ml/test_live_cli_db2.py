"""More ml write modes require --db; mode switches announce the path first."""

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


def _run(argv: list[str], *, cwd: Path, home: Path, db: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = str(_REPO)
    env.pop("CFB_COACH_DB", None)
    env.pop("CFB_COACH_MADDEN_DB", None)
    if db is not None:
        env["CFB_COACH_MADDEN_DB"] = str(db)
    return subprocess.run(
        [sys.executable, "-m", "cfb_coach", *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["ml", "confirm-execution", "--snap-id", "s", "--formation", "f", "--play", "p", "--evidence", "e"],
        ["ml", "recover-game-execution"],
        ["ml", "train-experimental", "--install"],
        ["ml", "offense-macro-lab", "--stage"],
        ["ml", "offense-design", "--stage", "--text"],
        ["ml", "offense-design", "--confirm-installed", "proposal", "--text"],
        ["ml", "offense-design", "--verify-macro", "NAME", "--text"],
        ["ml", "offense-design", "--unverify-macro", "NAME", "--text"],
        ["ml", "offense-design", "--rollback-design", "proposal", "--text"],
        ["ml", "learning-eval", "--promote"],
        ["ml", "learning-eval", "--rollback"],
    ],
    ids=[
        "confirm-execution-without-db",
        "recover-game-execution-dry-run-without-db",
        "train-experimental-install-without-db",
        "offense-macro-lab-stage-without-db",
        "offense-design-stage-without-db",
        "offense-design-confirm-installed-without-db",
        "offense-design-verify-macro-without-db",
        "offense-design-unverify-macro-without-db",
        "offense-design-rollback-without-db",
        "learning-eval-promote-without-db",
        "learning-eval-rollback-without-db",
    ],
)
def test_missing_required_db_exits_2_and_leaves_home_untouched(tmp_path, argv):
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "explicit.db"
    before_home = _snapshot(home)
    before_tmp = _snapshot(tmp_path)
    result = _run(argv, cwd=tmp_path, home=home)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "--db is required" in result.stderr
    assert _snapshot(home) == before_home
    assert list(home.iterdir()) == []
    assert _snapshot(tmp_path) == before_tmp
    assert not db.exists()


@pytest.mark.parametrize(
    "argv, marker",
    [
        (["ml", "shadow"], "mode: shadow"),
        (["ml", "heuristic"], "mode: heuristic"),
        (["ml", "experimental"], "mode: experimental"),
        (["ml", "experimental-preflight"], "mode remains:"),
    ],
    ids=["shadow", "heuristic", "experimental", "experimental-preflight"],
)
def test_mode_switch_prints_using_db_before_write(tmp_path, argv, marker):
    home = tmp_path / "home"
    home.mkdir()
    db_path = tmp_path / "explicit.db"
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.data import load_seed

    db = CoachDB(db_path, seed=load_seed())
    db.close()
    before = db_path.read_bytes()
    result = _run(argv, cwd=tmp_path, home=home, db=db_path)
    assert result.returncode == 0, result.stderr
    announced = f"Using DB: {db_path}"
    assert result.stdout.startswith(announced + "\n")
    assert result.stdout.find(announced) < result.stdout.find(marker)
    assert db_path.read_bytes() != before
