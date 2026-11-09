"""No future-snap leakage in legacy memory; verified outcomes stay separate."""
from __future__ import annotations

import json

from cfb_coach.madden.model import offense_game_memory as memory


class MemoryDB:
    def __init__(self):
        self.meta = {}

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value


def _close(db, session, suffix, coverage, outcome="incomplete", verified=True):
    memory.record_closed_snap(
        db, session_id=session, snap_id=f"{session}-{suffix:04d}",
        snap_seq=None, executed_formation="Gun Test",
        executed_play="Mesh", observed_defense=coverage,
        outcome=outcome, verified=verified,
    )


def test_closed_snaps_recover_sequence_from_exact_session_snap_id():
    db, session = MemoryDB(), "cpu-game-2"
    _close(db, session, 1, "Cover 1")
    data = json.loads(db.meta[f"ml_offense_game_memory.v1:{session}"])
    assert data["events"][0]["snap_seq"] == 1


def test_legacy_null_sequences_are_fenced_for_historical_replay():
    db, session = MemoryDB(), "cpu-game-2"
    events = [
        {
            "snap_id": f"{session}-{i:04d}", "snap_seq": None,
            "executed_play": "Mesh", "observed_defense": "Cover 1" if i == 1 else "Cover 3 Match",
            "outcome": "+3", "verification": "verified_execution",
        }
        for i in (1, 2, 3)
    ]
    # Deliberately inject old Game 1-style records, bypassing new writes.
    db.meta[f"ml_offense_game_memory.v1:{session}"] = json.dumps({
        "schema": memory.SCHEMA, "session_id": session, "events": events
    })
    at_2 = memory.pre_snap_context(db, session_id=session, snap_seq=2)
    assert at_2["events_before_snap"] == 1
    assert at_2["verified_executions"] == 1
    assert at_2["sample_size"] == 1
    at_4 = memory.pre_snap_context(db, session_id=session, snap_seq=4)
    assert at_4["events_before_snap"] == 3
    assert at_4["verified_executions"] == 3


def test_no_sequence_malformed_id_must_not_become_training_future_evidence():
    db, session = MemoryDB(), "cpu-game-2"
    _close(db, session, 1, "Cover 1")
    key = f"ml_offense_game_memory.v1:{session}"
    payload = json.loads(db.meta[key])
    payload["events"].append({
        "snap_id": "other-session-0099", "snap_seq": None,
        "executed_play": "Mesh", "observed_defense": "Cover 3 Match",
        "outcome": "+99", "verification": "verified_execution",
    })
    db.meta[key] = json.dumps(payload)
    at_2 = memory.pre_snap_context(db, session_id=session, snap_seq=2)
    assert at_2["sample_size"] == 1
    assert at_2["events_before_snap"] == 1


def test_snap_id_also_fences_when_explicit_sequence_is_missing():
    db, session = MemoryDB(), "cpu-game-2"
    _close(db, session, 1, "Cover 1")
    _close(db, session, 2, "Cover 3 Match")
    _close(db, session, 3, "Cover 6")
    context = memory.pre_snap_context(
        db, session_id=session, snap_seq=None, snap_id=f"{session}-0002",
    )
    assert context["events_before_snap"] == 1
    assert context["sample_size"] == 1


def test_one_snap_coverage_is_not_current_snap_coverage_or_rigid_decision():
    db, session = MemoryDB(), "cpu-game-2"
    _close(db, session, 1, "Cover 1")
    context = memory.pre_snap_context(db, session_id=session, snap_seq=2)
    assert context["state"] == "unknown"
    assert context["inferred_look"] is None
    assert context["sample_size"] == 1


def test_recommendation_alone_is_not_a_verified_learning_outcome():
    db, session = MemoryDB(), "cpu-game-2"
    memory.remember_recommendation(
        db, session_id=session, snap_id=f"{session}-0001", snap_seq=None,
        formation="Gun Test", play="Mesh",
    )
    context = memory.pre_snap_context(db, session_id=session, snap_seq=2)
    assert context["recommendations"] == 1
    assert context["verified_executions"] == 0
    assert context["sample_size"] == 0
