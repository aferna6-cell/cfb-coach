"""V2 live CLI guards: the model is never installed or activated by these tests."""
from __future__ import annotations

from types import SimpleNamespace
import json
import sqlite3

import pytest

from cfb_coach.madden.model import defense_v2_live as live
from cfb_coach.madden.model import defense_v2_pilot as pilot


def test_opt_in_requires_explicit_model_plus_both_environment_values(monkeypatch):
    sit = SimpleNamespace(extras={"defense_model": "v2"})
    monkeypatch.delenv(live.MODE_ENV, raising=False)
    monkeypatch.delenv(live.ARTIFACT_ENV, raising=False)
    assert not live.requested(sit)
    monkeypatch.setenv(live.MODE_ENV, "new-franchise")
    monkeypatch.setenv(live.ARTIFACT_ENV, "/tmp/v2-shadow.json")
    assert live.requested(sit)
    assert not live.requested(SimpleNamespace(extras={"defense_model": "v1"}))
    assert not live.requested(SimpleNamespace(extras={}))


def test_strict_model_franchise_cutoff_and_nonzero_verified(tmp_path):
    path = tmp_path / "model.json"
    model = pilot.fit([
        pilot.Snap("old", "cpu-a", "2026-10-01", 1,
                   "Nickel", "Cover 3 Sky", "single_high", "early", "cross", True)
    ])
    model.update(trained_franchise="new-franchise", trained_sessions=["old"],
                 trained_through_started_ts="2026-10-01")
    path.write_text(json.dumps(model), encoding="utf-8")
    assert live._model(str(path), "2026-11-10", "future", "new-franchise") == model
    with pytest.raises(ValueError, match="another franchise"):
        live._model(str(path), "2026-11-10", "future", "primary")
    with pytest.raises(ValueError, match="current game"):
        live._model(str(path), "2026-11-10", "old", "new-franchise")
    with pytest.raises(ValueError, match="current game or future"):
        live._model(str(path), "2026-09-10", "future", "new-franchise")
    model["n_verified"] = 0
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks verified"):
        live._model(str(path), "2026-11-10", "future", "new-franchise")


def test_identity_requires_exact_session_opponent_and_franchise():
    db = sqlite3.connect(":memory:")
    db.executescript("""
    CREATE TABLE game_sessions(session_id TEXT,opponent_id TEXT,dynasty TEXT,started_ts TEXT);
    INSERT INTO game_sessions VALUES ('g','cpu-a','test','2026-11-10');
    """)
    assert live._identity(db, "g", "cpu-a", "test") == "2026-11-10"
    with pytest.raises(ValueError):
        live._identity(db, "g", "cpu-a", "primary")
    with pytest.raises(ValueError):
        live._identity(db, "g", "cpu-b", "test")


def test_installation_lock_requires_exact_book(monkeypatch):
    from cfb_coach.madden import playbook
    monkeypatch.setattr(playbook, "load_books", lambda db: {
        "defense": {"locked_ts": "2026-11-01",
                    "formations": {"Nickel Over": ["Cover 3 Sky", "Cover 2 Man"]}}
    })
    db = SimpleNamespace()
    calls = live._confirmed_book(db, {"Nickel Over": ["Cover 3 Sky", "Cover 2 Man"]})
    assert len(calls) == 2
    assert all(c["installed"] and c["legal"] for c in calls)
    with pytest.raises(ValueError, match="exact installed"):
        live._confirmed_book(db, {"Nickel Over": ["Cover 3 Sky"]})
    with pytest.raises(ValueError, match="exact installed"):
        live._confirmed_book(db, {"Dime": ["Cover 3 Sky"]})
    monkeypatch.setattr(playbook, "load_books", lambda db: {
        "defense": {"formations": {"Nickel Over": ["Cover 3 Sky"]}}
    })
    with pytest.raises(ValueError, match="confirmed installed"):
        live._confirmed_book(db, {"Nickel Over": ["Cover 3 Sky"]})


def test_v2_selector_never_activated_by_default(monkeypatch):
    sit = SimpleNamespace(extras={"session_id": "g", "snap_seq": 1})
    monkeypatch.setenv(live.MODE_ENV, "test")
    monkeypatch.setenv(live.ARTIFACT_ENV, "/nonexistent")
    old = object()
    assert live.try_live(old, sit, "cpu-a", None, {}) is old
