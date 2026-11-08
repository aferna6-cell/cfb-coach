"""#130 live gate: revealed coverage is accepted after the snap and only feeds the next call.

Uses the real LivePlayController path (call_only → result_and_call) with the
fixture book locked as the applied playbook, so the inventory lock is exercised.
"""
from __future__ import annotations

from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation
from tests.madden_ml.live_book import apply_offense_book

BOOK = {"Gun Bunch": ["Mesh", "Stick"]}
SEED = {"opponents": {"cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu",
                              "confidence": "low", "profile_json": "{}"}}}


def _controller(tmp_path: Path):
    db = CoachDB(tmp_path / "t.db", seed=SEED)
    apply_offense_book(db, BOOK)
    seen: list[dict] = []
    calls: list[MaddenCall] = []

    def make(sit, **kwargs):
        seen.append(dict(kwargs))
        play = "Stick" if kwargs.get("last_coverage") else "Mesh"
        call = MaddenCall("offense", "Gun Bunch", play, "No adj", "read", "r")
        calls.append(call)
        return call

    ctrl = LivePlayController(db=db, opponent_id="cpu", make_call=make,
                              parse_situation=parse_madden_situation,
                              learn_summary=lambda: "ok", brand="Madden 27 Franchise", cpu_only=True)
    ctrl.start()
    return db, ctrl, seen, calls


def test_revealed_coverage_is_taken_after_the_snap_and_only_feeds_the_next_call(tmp_path):
    db, ctrl, seen, calls = _controller(tmp_path)
    try:
        first = ctrl.call_only("1&10 my 25", side="offense")
        assert first["ok"], first
        # Snap 1 is decided with no coverage: nothing is invented pre-snap.
        assert seen[0].get("last_coverage") is None
        shown = (calls[0].formation, calls[0].play, calls[0].adj_or_macro)

        out = ctrl.result_and_call(outcome="+5", sit_raw="2&5 my 30", their="cover 3", side="offense")
        assert out["ok"], out
        # The reveal is accepted after snap 1 and is what snap 2 is decided from.
        assert ctrl.last_coverage and "3" in ctrl.last_coverage
        assert seen[1].get("last_coverage") == ctrl.last_coverage
        assert calls[1].play == "Stick"
        # No same-snap leakage: snap 1's already-shown call is not rewritten.
        assert (calls[0].formation, calls[0].play, calls[0].adj_or_macro) == shown == ("Gun Bunch", "Mesh", "No adj")
        rows = list(db.conn.execute("SELECT * FROM snaps ORDER BY rowid"))
        assert len(rows) == 1
        logged = dict(rows[0])
        # Snap 1's row keeps the call shown before the reveal; the reveal is stored as that snap's outcome data.
        assert (logged["formation"], logged["play"]) == ("Gun Bunch", "Mesh")
        assert logged["coverage_seen"] and "3" in logged["coverage_seen"]
    finally:
        db.close()


def test_snap_without_a_reveal_does_not_fabricate_coverage(tmp_path):
    db, ctrl, seen, _calls = _controller(tmp_path)
    try:
        assert ctrl.call_only("1&10 my 25", side="offense")["ok"]
        assert ctrl.result_and_call(outcome="+3", sit_raw="2&7 my 28", their="", side="offense")["ok"]
        assert seen[1].get("last_coverage") is None
    finally:
        db.close()
