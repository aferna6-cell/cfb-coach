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


def verified_book_hash():
    return live.installed_fingerprint({"Nickel Over": ["Cover 3 Sky"]})


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
        franchise="test-franchise", confirmed_inventory_fingerprint=verified_book_hash())
    assert final is not original
    assert final.play == "Cover 3 Sky" and final.formation == "Nickel Over"
    assert final.defense_v2["applied"] is True
    assert final.defense_v2["defender_assignments"] == "UNKNOWN"
    assert final.defense_v2["earlier_opponent_concepts"] == 2
    assert final.macro is None and final.adjustment is None
    assert not db.logged  # a call without sealed snap_id must not be logged prematurely
    assert final._pending_ml_decision.policy_version == v2.SCHEMA
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
            model_path="/irrelevant/model.json",
            confirmed_inventory_fingerprint=verified_book_hash(), **kwargs)
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
                              franchise="test-franchise", confirmed_inventory_fingerprint=verified_book_hash())
    assert res.defense_v2["applied"] is False
    safe = artifact()
    monkeypatch.setattr(live, "load_model", lambda _: safe)
    res = live.maybe_apply_v2(call(), situation(seq=0), "cpu-test", db,
                              {"Nickel Over": ["Cover 3 Sky"]},
                              model_path="/irrelevant/model.json", test_db=str(db.path),
                              franchise="test-franchise", confirmed_inventory_fingerprint=verified_book_hash())
    assert res.defense_v2["applied"] is False
    db.conn.close()


def test_live_v2_book_cap_is_enforced(tmp_path, monkeypatch):
    db = FakeDb(tmp_path / "test.db")
    monkeypatch.setattr(live, "load_model", lambda _: artifact())
    book = {f"Form {n}": ["Cover 3 Sky"] for n in range(16)}
    res = live.maybe_apply_v2(call(), situation(), "cpu-test", db, book,
                              model_path="/irrelevant/model.json",
                              test_db=str(db.path), franchise="test-franchise", confirmed_inventory_fingerprint=verified_book_hash())
    assert not res.defense_v2["applied"]
    db.conn.close()


def test_v2_refuses_to_open_the_primary_database_on_invalid_cli(monkeypatch):
    from cfb_coach.madden import cli as madden_cli
    monkeypatch.setattr(madden_cli, "_open_opponent", lambda args: (_ for _ in ()).throw(
        AssertionError("unsafe CLI invocation opened a database")))
    args = SimpleNamespace(defense_model="v2", db=None,
                           defense_test_db=None, defense_franchise=None,
                           defense_model_path=None)
    assert madden_cli.cmd_play(args) == 2


def test_verified_action_is_only_shown_with_real_execution_support(tmp_path, monkeypatch):
    import json
    from cfb_coach.madden.model import defense_v2_actions as actions
    from cfb_coach.madden.model.defense_v2_pilot import _key
    db = FakeDb(tmp_path / "test.db")
    model = artifact()
    key = _key("long", "Nickel Over", "Cover 3 Sky", "physical-press")
    model["executed_adjustments"] = {"schema": actions.SCHEMA,
                                      "counts": {key: [0, 300]}, "n": 300}
    monkeypatch.setattr(live, "load_model", lambda _: model)
    confirmed = tmp_path / "actions.json"
    confirmed.write_text(json.dumps({
        "schema": actions.CONFIRM_SCHEMA,
        "confirmed_actions": [{
            "action_id": "def.coverage_hub", "instance_id": "physical-press",
            "physically_verified": True, "console": "Xbox",
            "verified_on": "2026-11-09",
            "verified_buttons": "Hold LT and choose Press",
            "label": "Press coverage", "params": {"option": "Press"},
        }],
    }))
    result = live.maybe_apply_v2(
        call(), situation(), "cpu-test", db,
        {"Nickel Over": ["Cover 3 Sky"]},
        model_path="unused", test_db=str(db.path), franchise="test-franchise",
        confirmed_inventory_fingerprint=verified_book_hash(),
        action_confirmations=str(confirmed),
    )
    assert result.defense_v2["applied"] is True
    assert result.adjustment is not None
    assert result.adjustment["id"] == "physical-press"
    assert result.adjustment["buttons"] == "Hold LT and choose Press"
    # Missing verification file must not silently enable any adjustments.
    stock = live.maybe_apply_v2(
        call(), situation(), "cpu-test", db,
        {"Nickel Over": ["Cover 3 Sky"]},
        model_path="unused", test_db=str(db.path), franchise="test-franchise",
        confirmed_inventory_fingerprint=verified_book_hash(),
    )
    assert stock.adjustment is None
    db.conn.close()
