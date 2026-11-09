"""Defensive V2 shadow contract: verified CPU evidence, franchise isolation, chronology."""
from __future__ import annotations

import sqlite3

import pytest

from cfb_coach.madden.model import defense_v2_pilot as v2


def sample(session: str, seq: int | None, *, opponent: str = "cpu-a",
           concept: str | None = "cross", success: bool = False,
           play: str = "Cover 3 Sky", started: str = "2026-11-01") -> v2.Snap:
    return v2.Snap(session, opponent, started, seq, "Nickel Over", play,
                   "single_high", "long", concept, success)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
    CREATE TABLE game_sessions(session_id TEXT PRIMARY KEY, opponent_id TEXT,
                               dynasty TEXT, started_ts TEXT);
    CREATE TABLE snaps(id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, snap_seq INTEGER,
                       executed_status TEXT, executed_verification TEXT,
                       executed_formation TEXT, executed_play TEXT, result TEXT,
                       down INTEGER, distance INTEGER, concept_typed TEXT, concept_seen TEXT);
    INSERT INTO game_sessions VALUES ('g1','cpu-a','test-franchise','2026-11-01');
    INSERT INTO game_sessions VALUES ('g2','cpu-b','primary','2026-11-02');
    INSERT INTO snaps VALUES
       (1,'g1','defense',1,'identified','verified','Nickel Over','Cover 3 Sky','W',3,7,'Mesh',null),
       (2,'g1','defense',2,'unknown',null,null,null,'L',3,7,'Four Verticals',null),
       (3,'g1','defense',3,'identified','verified','Nickel Over','Cover 3 Sky','L',3,7,'Inside Zone',null),
       (4,'g1','defense',null,'identified','verified','Nickel Over','Cover 3 Sky','L',3,7,'Mesh',null),
       (5,'g2','defense',1,'identified','verified','Nickel Over','Cover 3 Sky','W',3,7,'Mesh',null);
    """)
    return conn


def test_cpu_verified_only_and_franchise_isolation(monkeypatch):
    from cfb_coach.madden.model import defense_v1
    monkeypatch.setattr(defense_v1, "offense_won", lambda result, *_: {"W": True, "L": False}.get(result))
    rows, stats = v2.verified_rows(db(), franchise="test-franchise")
    assert len(rows) == 3
    assert stats["unverified_execution"] == 1
    assert {r.session for r in rows} == {"g1"}
    assert all(r.play == "Cover 3 Sky" for r in rows)
    assert any(r.seq is None for r in rows)
    with pytest.raises(ValueError):
        v2.verified_rows(db(), franchise="")


def test_online_observation_no_same_or_future_and_unverified_execution_allowed():
    conn = db()
    earlier = v2.earlier_opponent_observations(
        conn, franchise="test-franchise", session="g1", opponent="cpu-a", next_seq=3)
    assert [r.seq for r in earlier] == [1, 2]
    assert [r.concept for r in earlier] == ["cross", "vert"]
    assert v2.earlier_opponent_observations(
        conn, franchise="test-franchise", session="g1", opponent="cpu-b", next_seq=3) == []
    with pytest.raises(ValueError):
        v2.earlier_opponent_observations(
            conn, franchise="test-franchise", session="g1", opponent="cpu-a", next_seq=0)


def test_concept_belief_filters_future_session_opponent_and_null_sequence():
    model = v2.fit([sample("old", 1, concept="run")])
    history = [sample("g", 1, concept="cross"), sample("g", 2, concept="cross"),
               sample("g", 3, concept="run"), sample("other", 1, concept="run"),
               sample("g", None, concept="run"),
               sample("g", 1, opponent="cpu-b", concept="run")]
    before = v2.concept_belief(model, history, session="g", opponent="cpu-a", next_seq=3)
    after = v2.concept_belief(model, history, session="g", opponent="cpu-a", next_seq=4)
    assert before["cross"] > after["cross"]
    assert abs(sum(before.values()) - 1.0) < 1e-9


def test_learned_model_no_superiority_without_data():
    rows = [sample("g1", 1, success=True),
            sample("g2", 1, started="2026-11-02", success=False)]
    model = v2.fit(rows)
    assert model["n_games"] == 2 and model["policy"] == "shadow_only"
    report = v2.evaluate(rows, bootstraps=30)
    assert report["prediction_gate"] == "NO_GO"
    assert report["live_activation"] == "NOT_ALLOWED"
    assert report["heldout_games"] == 1


def test_rank_fails_closed_when_calls_uninstalled_or_illegal():
    model = v2.fit([sample("g1", 1)])
    belief = {c: 1.0 / len(v2.CONCEPTS) for c in v2.CONCEPTS}
    ranks = v2.rank(model, context="long", belief=belief, calls=[
        {"formation": "Nickel Over", "play": "Cover 3 Sky", "installed": True, "legal": True},
        {"formation": "Dime", "play": "Cover 1", "installed": True, "legal": False},
        {"formation": "Goal Line", "play": "Cover 0", "installed": False, "legal": True},
    ])
    assert len(ranks["ranked"]) == 1
    assert not ranks["ready_for_live"] and not ranks["applied"]
    with pytest.raises(ValueError):
        v2.rank(model, context="long", belief={"run": 1.2}, calls=[])
