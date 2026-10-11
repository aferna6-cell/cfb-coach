"""CFB_COACH_MADDEN_DB selects the Madden DB without reading CFB_COACH_DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfb_coach.db import default_db_path, madden_db_path


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


def test_set_cfb_coach_madden_db_uses_env_path_and_leaves_home_empty(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    given_parent = tmp_path / "given"
    given = given_parent / "madden27.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.delenv("CFB_COACH_DB", raising=False)
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(given))

    path = madden_db_path()
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
    assert madden_db_path() == given
    assert mkdirs == []
    assert calls == []
    assert list(home.iterdir()) == []


def test_unset_cfb_coach_madden_db_keeps_home_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    calls = _track_home(monkeypatch, home)
    monkeypatch.delenv("CFB_COACH_DB", raising=False)
    monkeypatch.delenv("CFB_COACH_MADDEN_DB", raising=False)

    expected = home / ".cfb-coach" / "madden27.db"
    assert madden_db_path() == expected
    assert expected.parent.is_dir()
    assert not expected.exists()
    assert calls == [home]


def test_empty_cfb_coach_madden_db_counts_as_unset(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    calls = _track_home(monkeypatch, home)
    monkeypatch.delenv("CFB_COACH_DB", raising=False)
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", "")

    expected = home / ".cfb-coach" / "madden27.db"
    assert madden_db_path() == expected
    assert expected.parent.is_dir()
    assert calls == [home]


def test_db_env_vars_stay_on_their_own_paths(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    cfb = tmp_path / "cfb" / "coach.db"
    madden = tmp_path / "madden" / "madden27.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(cfb))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(madden))

    assert default_db_path() == cfb
    assert madden_db_path() == madden
    assert default_db_path() != madden_db_path()
    assert cfb.parent.is_dir()
    assert madden.parent.is_dir()
    assert calls == []
    assert list(home.iterdir()) == []


def test_cfb_coach_db_alone_does_not_change_madden_db_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    cfb_parent = tmp_path / "cfb-only"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(cfb_parent / "coach.db"))
    monkeypatch.delenv("CFB_COACH_MADDEN_DB", raising=False)

    expected = home / ".cfb-coach" / "madden27.db"
    assert madden_db_path() == expected
    assert expected.parent.is_dir()
    assert not (cfb_parent / "madden27.db").exists()
    assert not cfb_parent.exists()
    assert calls == [home]
