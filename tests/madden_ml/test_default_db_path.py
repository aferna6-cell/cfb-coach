"""CFB_COACH_DB overrides the home default without touching ~/.cfb-coach."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfb_coach.db import CoachDB, default_db_path


def _track_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> list[Path]:
    calls: list[Path] = []
    real_home = Path.home

    def home_tracker() -> Path:
        resolved = real_home()
        calls.append(resolved)
        return resolved

    monkeypatch.setattr(Path, "home", staticmethod(home_tracker))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return calls


def test_set_cfb_coach_db_uses_env_path_and_leaves_home_empty(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    given_parent = tmp_path / "given"
    given = given_parent / "coach.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(given))

    path = default_db_path()
    assert path == given
    assert given_parent.is_dir()
    assert not given.exists()
    assert calls == []
    assert list(home.iterdir()) == []

    mkdirs: list[Path] = []
    real_mkdir = Path.mkdir

    def spy_mkdir(self, *args, **kwargs):
        mkdirs.append(Path(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", spy_mkdir)
    assert default_db_path() == given
    assert mkdirs == []

    db = CoachDB()
    try:
        assert db.path == given
        assert given.is_file()
    finally:
        db.close()
    assert calls == []
    assert list(home.iterdir()) == []


def test_unset_cfb_coach_db_keeps_home_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    calls = _track_home(monkeypatch, home)
    monkeypatch.delenv("CFB_COACH_DB", raising=False)

    expected = home / ".cfb-coach" / "coach.db"
    assert default_db_path() == expected
    assert expected.is_file()
    assert (home / ".cfb-coach").is_dir()
    assert calls == [home]


def test_empty_cfb_coach_db_counts_as_unset(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", "")

    expected = home / ".cfb-coach" / "coach.db"
    assert default_db_path() == expected
    assert expected.is_file()
    assert (home / ".cfb-coach").is_dir()
    assert calls == [home]
