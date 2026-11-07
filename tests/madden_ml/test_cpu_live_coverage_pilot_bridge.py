"""Integration contract: CPU pilot uses the already logged previous-play coverage."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from cfb_coach.madden.model import offense_pilot as pilot
from cfb_coach.madden.model import offense_structural_model as sm


def test_pilot_consumes_closed_form_history_and_passes_mix(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE snaps(id INTEGER PRIMARY KEY, session_id TEXT, opponent_id TEXT, "
                 "side TEXT, snap_seq INT, result TEXT, coverage_seen TEXT)")
    conn.execute("INSERT INTO snaps(session_id,opponent_id,side,snap_seq,result,coverage_seen) "
                 "VALUES('g1','cpu','offense',1,'+8','Cover 2 Man')")
    conn.execute("INSERT INTO snaps(session_id,opponent_id,side,snap_seq,result,coverage_seen) "
                 "VALUES('g1','cpu','offense',2,'+5','Cover 4 Palms')")
    monkeypatch.setattr(sm, "pre_snap_mix", lambda art, opponent_id: ({"unknown": 1.0}, "none"))
    captured = {}

    def fake_recommend(self, situation, *, game_id="", coverage_context=None):
        captured.update(situation=situation, game_id=game_id, context=coverage_context)
        return {"coverage_history": coverage_context, "primary": {"play": "test"}}

    monkeypatch.setattr(pilot.PilotSession, "recommend", fake_recommend)
    fake = object.__new__(pilot.PilotSession)
    fake.artifact = SimpleNamespace()

    before = conn.total_changes
    result = fake.recommend_from_logged_game(
        conn, {"down": 2, "distance": 7, "yardline": 40,
               "opponent_id": "cpu", "opponent_type": "cpu"},
        session_id="g1", next_snap_seq=2)
    assert conn.total_changes == before
    assert captured["game_id"] == "g1"
    assert captured["context"]["named_n"] == 1
    assert captured["situation"]["mix"]["cover_2_man"] > 0
    assert "cover_4_palms" not in captured["situation"]["mix"]
    assert result["coverage_history"]["next_snap_seq"] == 2
