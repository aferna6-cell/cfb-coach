"""Madden DB path: dedicated env, else beside CFB_COACH_DB, else home."""

from __future__ import annotations

import warnings
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


def test_cfb_coach_db_alone_places_madden_db_beside_it(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    cfb = tmp_path / "cfb" / "coach.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(cfb))
    monkeypatch.delenv("CFB_COACH_MADDEN_DB", raising=False)

    assert madden_db_path() == cfb.parent / "madden27.db"
    assert calls == []
    assert list(home.iterdir()) == []


def test_cfb_coach_madden_db_wins_when_both_are_set(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    cfb = tmp_path / "cfb" / "coach.db"
    madden = tmp_path / "madden" / "madden27.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(cfb))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(madden))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert madden_db_path() == madden
    assert madden != cfb.parent / "madden27.db"
    assert calls == []
    assert list(home.iterdir()) == []


def test_same_resolved_file_warns(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    folder = tmp_path / "same"
    cfb = folder / "coach.db"
    madden = folder / "madden27.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(cfb))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(madden))

    with pytest.warns(UserWarning, match="CFB_COACH_MADDEN_DB") as caught:
        path = madden_db_path()
    message = str(caught[0].message)
    assert "CFB_COACH_DB" in message
    assert path == madden
    assert calls == []
    assert list(home.iterdir()) == []


def test_both_vars_same_arbitrary_file_warns(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    shared = tmp_path / "shared.db"
    calls = _track_home(monkeypatch, home)
    monkeypatch.setenv("CFB_COACH_DB", str(shared))
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(shared))

    with pytest.warns(UserWarning, match="CFB_COACH_MADDEN_DB") as caught:
        path = madden_db_path()
    message = str(caught[0].message)
    assert "CFB_COACH_DB" in message
    assert path == shared
    assert calls == []
    assert list(home.iterdir()) == []

    other = tmp_path / "other.db"
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", str(other))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert madden_db_path() == other
    assert calls == []
    assert list(home.iterdir()) == []
