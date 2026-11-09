"""Near end-to-end V2 testing-franchise live call with installed legal play, advice and action."""
from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from types import SimpleNamespace

from cfb_coach.madden.model import defense_v2_live as live
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.playcaller import MaddenCall


@dataclass
class FakeDecision:
    policy_version: str = "baseline"
    ml_status_detail: str = ""


def test_explicit_test_franchise_live_call_has_optional_adjustment_and_user_read(
    tmp_path, monkeypatch
):
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model import defense_coordinator as old

    sql = sqlite3.connect(":memory:")
    sql.executescript("""
    CREATE TABLE game_sessions (session_id TEXT, opponent_id TEXT, dynasty TEXT, started_ts TEXT);
    CREATE TABLE snaps (id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, snap_seq INTEGER,
                        result TEXT, concept_typed TEXT, concept_seen TEXT);
    INSERT INTO game_sessions VALUES ('current','cpu-a','experiment','2026-11-11');
    INSERT INTO snaps VALUES (1,'current','defense',1,'+4','Four Verticals',NULL);
    """)
    db = SimpleNamespace(conn=sql, path=tmp_path / "isolated_test.db")
    installed = {"Nickel Over": ["Cover 3 Sky"]}
    monkeypatch.setattr(playbook, "load_books", lambda _db: {
        "defense": {"locked_ts": "2026-11-11", "formations": installed}
    })
    monkeypatch.setattr(old, "_model_decision", lambda **kwargs: FakeDecision())
    monkeypatch.setattr(old, "_book_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr(old, "_stamp_decision",
                        lambda call, decision, db, extras:
                        setattr(call, "_pending_ml_decision", decision))

    old_row = v2.Snap("older", "cpu-b", "2026-11-01", 1,
                      "Nickel Over", "Cover 3 Sky", "single_high",
                      "long", "vert", False)
    model = v2.fit([old_row])
    model["trained_franchise"] = "experiment"
    model["trained_sessions"] = ["older"]
    model["trained_through_started_ts"] = "2026-11-01"
    model["concept_counts"] = {"vert": 1000}
    artifact = tmp_path / "v2.json"
    artifact.write_text(json.dumps(model), encoding="utf-8")
    monkeypatch.setenv(live.MODE_ENV, "experiment")
    monkeypatch.setenv(live.ARTIFACT_ENV, str(artifact))

    sit = SimpleNamespace(
        extras={"defense_model": "v2", "session_id": "current", "snap_seq": 2},
        down=3, distance=9, goal_line=False, red_zone=False,
    )
    existing = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "No adj",
                          "User zone", "old")
    actual = live.try_live(existing, sit, "cpu-a", db, installed)
    assert actual.side == "defense"
    assert actual.formation == "Nickel Over"
    assert actual.play == "Cover 3 Sky"
    assert actual.ml_defense_v2["applied"] is True
    assert actual._pending_ml_decision.policy_version == v2.SCHEMA
    assert actual.user_guidance["predicted_concept"] == "vert"
    assert "protect" in actual.read_or_user or "seam" in actual.read_or_user
    assert "USER READ:" in actual.rationale
    assert actual.ml_defense_v2["adjustment_comparison"]["stock"] >= (
        actual.ml_defense_v2["adjustment_comparison"]["adjusted"])
    if actual.adjustment:
        assert actual.adjustment["components"]
        assert "buttons" not in actual.adjustment
        assert "press " not in actual.adjustment_line()


def test_wrong_franchise_and_default_no_opt_in_fail_closed(tmp_path, monkeypatch):
    sit = SimpleNamespace(extras={"defense_model": "v1"})
    monkeypatch.setenv(live.MODE_ENV, "experiment")
    monkeypatch.setenv(live.ARTIFACT_ENV, str(tmp_path / "model.json"))
    call = object()
    assert live.try_live(call, sit, "cpu-a", None, {}) is call
