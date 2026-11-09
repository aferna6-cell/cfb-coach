"""Live defensive V2 isolation and actual selected-call tests (small test DB)."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from cfb_coach.madden.model import defense_v2_live as live
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.playcaller import MaddenCall


class FakeDb:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.logged = []
        self.conn.executescript("""
        CREATE TABLE game_sessions (
          session_id TEXT PRIMARY KEY, dynasty TEXT, opponent_id TEXT, started_ts TEXT);
        CREATE TABLE snaps (
          id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, snap_seq INTEGER,
          result TEXT, concept_typed TEXT, concept_seen TEXT);
        INSERT INTO game_sessions VALUES ('new-game','test-franchise','cpu-test','2026-11-09');
        INSERT INTO snaps VALUES (1,'new-game','defense',1,'-2','Mesh',null);
        INSERT INTO snaps VALUES (2,'new-game','defense',2,'-3','Four Verticals',null);
        """)
        self.conn.commit()

    def log_ml_decision(self, decision, agree=None):
        self.logged.append(decision)


def artifact():
    rows = [
        v2.Snap("old-game", "cpu-old", "2026-10-31", 1,
                "Nickel Over", "Cover 3 Sky", "single_high", "long", "cross", False),
    ]
    model = v2.fit(rows)
    model.update({"franchise": "test-franchise",
                  "trained_sessions": ["old-game"],
                  "trained_through_started_ts": "2026-10-31"})
    return model


def situation(seq=3):
    return SimpleNamespace(side="defense", down=3, distance=7,
                           goal_line=False, red_zone=False, two_minute=False,
                           extras={"session_id": "new-game", "snap_seq": seq,
                                   "snap_id": None, "game_id": "new-game"})


def call():
    return MaddenCall("defense", "Nickel Over", "Cover 2 Man", "No adj",
                      "Check alignment", "existing caller")


def test_live_v2_actual_displayed_model_pick_and_logging(tmp_path, monkeypatch):
    db = FakeDb(tmp_path / "test.db")
    monkeypatch.setattr(live, "load_model", lambda _: artifact())
    original = call()
    final = live.maybe_apply_v2(
        original, situation(), "cpu-test", db,
        {"Nickel Over": ["Cover 3 Sky"]},
        model_path="/irrelevant/test-model.json", test_db=str(db.path),
        franchise="test-franchise")
    assert final is not original
    assert final.play == "Cover 3 Sky" and final.formation == "Nickel Over"
    assert final.defense_v2["applied"] is True
    assert final.defense_v2["defender_assignments"] == "UNKNOWN"
    assert final.defense_v2["earlier_opponent_concepts"] == 2
    assert final.macro is None and final.adjustment is None
    assert db.logged
    assert db.logged[-1].policy_version == v2.SCHEMA
    db.conn.close()


def test_live_v2_rejects_other_db_and_franchise_without_mutation(tmp_path, monkeypatch):
    db = FakeDb(tmp_path / "test.db")
    monkeypatch.setattr(live, "load_model", lambda _: artifact())
    original = call()
    for kwargs in (
        {"test_db": str(tmp_path / "WRONG.db"), "franchise": "test-franchise"},
        {"test_db": str(db.path), "franchise": "main"},
    ):
        res = live.maybe_apply_v2(
            original, situation(), "cpu-test", db,
            {"Nickel Over": ["Cover 3 Sky"]},
            model_path="/irrelevant/model.json", **kwargs)
        assert res is original and res.defense_v2["applied"] is False
    assert not db.logged
    db.conn.close()


def test_live_v2_rejects_same_game_and_missing_seq(tmp_path, monkeypatch):
    db = FakeDb(tmp_path / "test.db")
    compromised = artifact()
    compromised["trained_sessions"] = ["new-game"]
    monkeypatch.setattr(live, "load_model", lambda _: compromised)
    res = live.maybe_apply_v2(call(), situation(), "cpu-test", db,
                              {"Nickel Over": ["Cover 3 Sky"]},
                              model_path="/irrelevant/model.json", test_db=str(db.path),
                              franchise="test-franchise")
    assert res.defense_v2["applied"] is False
    safe = artifact()
    monkeypatch.setattr(live, "load_model", lambda _: safe)
    res = live.maybe_apply_v2(call(), situation(seq=0), "cpu-test", db,
                              {"Nickel Over": ["Cover 3 Sky"]},
                              model_path="/irrelevant/model.json", test_db=str(db.path),
                              franchise="test-franchise")
    assert res.defense_v2["applied"] is False
    db.conn.close()


def test_live_v2_book_cap_is_enforced(tmp_path, monkeypatch):
    db = FakeDb(tmp_path / "test.db")
    monkeypatch.setattr(live, "load_model", lambda _: artifact())
    book = {f"Form {n}": ["Cover 3 Sky"] for n in range(16)}
    res = live.maybe_apply_v2(call(), situation(), "cpu-test", db, book,
                              model_path="/irrelevant/model.json",
                              test_db=str(db.path), franchise="test-franchise")
    assert not res.defense_v2["applied"]
    db.conn.close()
