"""Standing user execution attestation: completed snaps only, never displayed calls."""
from __future__ import annotations

import json

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, render_live_html
from cfb_coach.madden.situation import parse_madden_situation


def _seed():
    return {"opponents": {"cpu": {"display_name": "CPU", "team_now": "Test"}}}


class Call:
    side = "offense"
    formation = "Gun Bunch"
    play = "Mesh"
    macro = None
    macro_info = None
    adjustment = None
    reads = ()
    rationale = "synthetic"
    adj_or_macro = "base"

    def format(self):
        return "Gun Bunch — Mesh"


def _start(tmp_path, *, standing=True, action=False):
    db = CoachDB(tmp_path / "madden27.db", seed=_seed())
    def make(sit, **kwargs):
        call = Call()
        if action:
            call.adjustment = {
                "id": "HOT-X", "label": "X slant", "buttons": "X to slant",
                "why": "synthetic matchup",
            }
        return call
    ctrl = LivePlayController(
        db=db, opponent_id="cpu", make_call=make,
        parse_situation=parse_madden_situation, learn_summary=lambda: "synthetic",
        brand="Madden 27 Franchise", cpu_only=True,
        enable_execution_verify=True,
        standing_execution_attestation=standing,
    )
    ctrl.start()
    return db, ctrl


def _snap(db):
    return db.conn.execute(
        "SELECT * FROM snaps WHERE data_source != 'session_close' ORDER BY id LIMIT 1"
    ).fetchone()


def _outcome(db):
    row = db.conn.execute("SELECT * FROM ml_outcomes ORDER BY id LIMIT 1").fetchone()
    return dict(row), json.loads(row["outcome_json"])


def test_untapped_completed_play_is_standing_user_attested(tmp_path):
    db, ctrl = _start(tmp_path)
    try:
        first = ctrl.call_only("1&10 my 25")
        assert first["ok"]
        assert db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0] == 0
        result = ctrl.result_and_call(outcome="+6", sit_raw="2&4 my 31")
        assert result["ok"]
        snap = _snap(db)
        assert snap["executed_play"] == "Mesh"
        assert snap["executed_formation"] == "Gun Bunch"
        assert snap["executed_status"] == "identified"
        assert snap["executed_verification"] == "verified"
        assert snap["accepted"] == "true"
        outcome, payload = _outcome(db)
        assert outcome["executed_verification"] == "verified"
        assert payload["execution_attestation_source"] == "standing_user_attestation"
        assert payload["no_adjustment_explicitly_confirmed"] is True
        assert payload["offense_action_explicitly_confirmed"] is False
        assert ctrl.state()["standing_execution_attestation"] is True
    finally:
        db.close()


def test_per_snap_different_play_overrides_standing_rule(tmp_path):
    db, ctrl = _start(tmp_path)
    try:
        ctrl.call_only("1&10 my 25")
        ctrl.result_and_call(
            outcome="+3", sit_raw="2&7 my 28",
            executed_status="used_different",
            executed_formation="Singleback Ace", executed_play="HB Dive",
            verify_touched=True,
        )
        snap = _snap(db)
        assert snap["executed_status"] == "identified"
        assert snap["executed_verification"] == "verified"
        assert snap["executed_play"] == "HB Dive"
        assert snap["accepted"] == "false"
        _out, payload = _outcome(db)
        assert payload["execution_attestation_source"] == "per_snap_user_confirmation"
    finally:
        db.close()


def test_explicit_unknown_prevents_standing_attestation(tmp_path):
    db, ctrl = _start(tmp_path)
    try:
        ctrl.call_only("1&10 my 25")
        ctrl.result_and_call(
            outcome="+2", sit_raw="2&8 my 27",
            executed_status="unknown", verify_touched=True,
        )
        snap = _snap(db)
        assert snap["executed_verification"] == "unknown"
        assert snap["executed_play"] is None
        _out, payload = _outcome(db)
        assert payload["execution_attestation_source"] == "per_snap_user_confirmation"
    finally:
        db.close()


def test_standing_rule_confirms_recommended_optional_adjustment(tmp_path):
    db, ctrl = _start(tmp_path, action=True)
    try:
        ctrl.call_only("1&10 my 25")
        assert ctrl.state()["pending_offense_action"]["id"] == "HOT-X"
        ctrl.result_and_call(outcome="+9", sit_raw="2&1 my 34")
        snap = _snap(db)
        assert snap["executed_verification"] == "verified"
        _out, payload = _outcome(db)
        assert payload["execution_attestation_source"] == "standing_user_attestation"
        assert payload["offense_action_explicitly_confirmed"] is True
        assert payload["executed_adjustment_id"] == "HOT-X"
        assert payload["no_adjustment_explicitly_confirmed"] is False
    finally:
        db.close()


def test_standing_rule_disabled_keeps_default_unknown(tmp_path):
    db, ctrl = _start(tmp_path, standing=False)
    try:
        ctrl.call_only("1&10 my 25")
        ctrl.result_and_call(outcome="+3", sit_raw="2&7 my 28")
        snap = _snap(db)
        assert snap["executed_verification"] == "unknown"
        assert snap["executed_play"] is None
        _out, payload = _outcome(db)
        assert payload["execution_attestation_source"] == "unknown"
    finally:
        db.close()


def test_end_game_without_outcome_does_not_attest_unplayed_call(tmp_path):
    db, ctrl = _start(tmp_path)
    try:
        ctrl.call_only("1&10 my 25")
        ctrl.end_game(result_wl="win", score="7-0")
        assert db.conn.execute(
            "SELECT COUNT(*) FROM snaps WHERE executed_verification = 'verified'"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_compact_view_hides_confirmation_but_preserves_exception(tmp_path):
    db, ctrl = _start(tmp_path)
    try:
        html = render_live_html(ctrl)
        assert "btn-ran-different" in html
        assert '$("btn-ran-recommended").hidden = !!st.standing_execution_attestation;' in html
        assert "standing_execution_attestation" in html
        assert ctrl.state()["standing_execution_attestation"] is True
    finally:
        db.close()
