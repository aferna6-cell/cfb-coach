"""argv for defense write modes must refuse a missing --db before any home write."""

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


def _run(argv: list[str], *, cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = str(_REPO)
    env.pop("CFB_COACH_DB", None)
    env.pop("CFB_COACH_MADDEN_DB", None)
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
        ["ml", "defense-design", "--stage"],
        ["ml", "defense-design", "--confirm-installed", "proposal"],
        ["ml", "defense-experimental", "--enable"],
        ["ml", "defense-experimental", "--disable"],
        ["ml", "defense-experimental", "--retrain", "--out", "{out}"],
        ["ml", "defense-experimental", "--retrain", "--db", "{db}"],
        ["ml", "defense-macro-lab", "--stage"],
        ["ml", "defense-macro-lab", "--verify", "ML-D-1"],
        ["ml", "defense-macro-lab", "--unverify", "ML-D-1"],
    ],
    ids=[
        "defense-design-stage-without-db",
        "defense-design-confirm-installed-without-db",
        "defense-experimental-enable-without-db",
        "defense-experimental-disable-without-db",
        "defense-experimental-retrain-without-db",
        "defense-experimental-retrain-without-out",
        "defense-macro-lab-stage-without-db",
        "defense-macro-lab-verify-without-db",
        "defense-macro-lab-unverify-without-db",
    ],
)
def test_missing_required_flags_exit_2_and_leave_home_untouched(tmp_path, argv):
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "explicit.db"
    out = tmp_path / "out"
    rendered = [part.format(db=db, out=out) for part in argv]
    before_home = _snapshot(home)
    before_tmp = _snapshot(tmp_path)
    result = _run(rendered, cwd=tmp_path, home=home)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert _snapshot(home) == before_home
    assert list(home.iterdir()) == []
    assert _snapshot(tmp_path) == before_tmp
    assert not db.exists()
    assert not out.exists()


def test_defense_design_confirm_installed_cannot_write(tmp_path, monkeypatch):
    """--confirm-installed opens sqlite mode=ro and prints the diff. It does not write."""
    import hashlib
    import json
    from contextlib import redirect_stdout
    from io import StringIO
    from types import SimpleNamespace

    import cfb_coach.db as dbmod
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.data import load_seed
    from cfb_coach.madden.model import defense_coordinator as defense
    from cfb_coach.madden.model.cli import cmd_ml_defense_design

    home = tmp_path / "home"
    home.mkdir()
    db_path = tmp_path / "explicit.db"
    db = CoachDB(db_path, seed=load_seed())
    try:
        proposal_id = defense.stage_design(db, max_formations=2)["proposal_id"]
        pending_before = db.get_meta(defense.PENDING_KEY)
    finally:
        db.close()
    before_hash = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_home = _snapshot(home)

    uris: list[str] = []
    real_connect = dbmod.sqlite3.connect

    def spy(target, *args, **kwargs):
        uris.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(dbmod.sqlite3, "connect", spy)
    saved = {
        key: os.environ.get(key)
        for key in ("HOME", "USERPROFILE", "CFB_COACH_DB", "CFB_COACH_MADDEN_DB")
    }
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    os.environ.pop("CFB_COACH_DB", None)
    os.environ.pop("CFB_COACH_MADDEN_DB", None)
    stdout = StringIO()
    try:
        args = SimpleNamespace(
            stage=False,
            show=False,
            confirm_installed=proposal_id,
            max_formations=15,
            attest="Installed every formation in the Madden editor.",
            db=str(db_path),
        )
        with redirect_stdout(stdout):
            code = cmd_ml_defense_design(args)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    payload = json.loads(stdout.getvalue())
    ro_uris = [uri for uri in uris if "mode=ro" in uri]
    assert code == 0
    assert payload["wrote_book"] is False
    assert payload["diff"]["to"]["formations"]
    assert ro_uris
    assert all("mode=ro" in uri for uri in uris)
    assert any(db_path.resolve().as_posix() in uri for uri in ro_uris)
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_hash
    assert _snapshot(home) == before_home
    assert list(home.iterdir()) == []
    assert list(home.rglob("*")) == []
    opened = CoachDB.open_read_only(db_path)
    try:
        assert opened.get_meta(defense.PENDING_KEY) == pending_before
        assert opened.get_meta(defense.HISTORY_KEY) is None
        assert "diff" not in (opened.get_meta(defense.PENDING_KEY) or "")
    finally:
        opened.close()
