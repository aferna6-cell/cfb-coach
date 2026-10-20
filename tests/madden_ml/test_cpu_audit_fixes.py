"""CPU-game audit fixes: drive ids, clock staleness, score reconcile, diagnostics.

Synthetic fixtures only. No real game rows, scores, or opponents.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.last_snap import drive_ended
from cfb_coach.live_server import LivePlayController, fresh_game_clock
from cfb_coach.madden.model import experimental_live
from cfb_coach.madden.model.inference import resolve_mode
from cfb_coach.madden.model.postgame_diagnostics import (
    distance_bucket,
    format_diagnostics,
    session_diagnostics,
)
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.session import parse_score_pair, score_reconcile


def _seed() -> dict:
    return {"opponents": {"cpu": {"display_name": "CPU", "team_now": "AAA", "skill": "cpu", "confidence": "low"}}}


class _Call:
    side = "offense"
    formation = "Synthetic Form"
    play = "Synthetic Play"
    macro = None
    adjustment = None
    adj_or_macro = "base"
    rationale = "synthetic"
    reads = ()
    adjustments = ()

    def format(self) -> str:
        return "Synthetic Form — Synthetic Play"

    def headline(self) -> str:
        return self.format()


def _ctrl(tmp_path: Path):
    db = CoachDB(tmp_path / "madden27.db", seed=_seed())
    ctrl = LivePlayController(
        db=db,
        opponent_id="cpu",
        make_call=lambda sit, **_kw: _Call(),
        parse_situation=parse_madden_situation,
        learn_summary=lambda: "",
        brand="Madden 27 Franchise",
        cpu_only=True,
        enable_execution_verify=True,
    )
    ctrl.start()
    return db, ctrl


def _snaps(db: CoachDB):
    return db.conn.execute(
        "SELECT drive_id, game_clock, result, score_for, score_against, accepted FROM snaps ORDER BY id"
    ).fetchall()


# --- drive_id -------------------------------------------------------------

def test_drive_ids_start_on_first_snap_and_after_possession_change(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=1, score_us=0, score_them=0, score_set=True)
        ctrl.result_and_call(outcome="+5", sit_raw="2&5 my 30", quarter=1)
        ctrl.result_and_call(outcome="td", sit_raw="1&10 my 25", quarter=1,
                             score_us=7, score_them=0, score_set=True)
        ctrl.result_and_call(outcome="+3", sit_raw="2&7 my 28", quarter=1)
        ctrl.result_and_call(outcome="int", sit_raw="1&10 my 20", quarter=2,
                             score_us=7, score_them=7, score_set=True)
        ctrl.result_and_call(outcome="punt", sit_raw="1&10 my 30", quarter=2)
        ctrl.result_and_call(outcome="+4", sit_raw="1&10 my 25", quarter=3)
        ctrl.result_and_call(outcome="+2", sit_raw="2&8 my 27", quarter=3)
        ids = [r["drive_id"] for r in _snaps(db)]
        assert ids == ["auto-d01", "auto-d01", "auto-d02", "auto-d02", "auto-d03", "auto-d04", "auto-d05"]
        assert all(re.fullmatch(r"auto-d\d{2}", i) for i in ids)
    finally:
        db.close()


def test_explicit_drive_id_wins_over_inferred(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25 drive id 9", quarter=1)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29", quarter=1)
        rows = _snaps(db)
        assert rows[0]["drive_id"] == "9"
    finally:
        db.close()


def test_drive_ending_results() -> None:
    assert drive_ended("td") and drive_ended("int") and drive_ended("punt")
    assert drive_ended("fumble lost") and drive_ended("field goal")
    assert not drive_ended("+6") and not drive_ended("incomplete") and not drive_ended("")


def test_old_rows_without_drive_stay_readable(tmp_path: Path) -> None:
    db = CoachDB(tmp_path / "madden27.db", seed=_seed())
    try:
        row = db.log_snap(opponent_id="cpu", side="offense", situation_raw="1st and 10",
                          our_call="X — Y", result="+1", session_id="syn")
        got = db.conn.execute("SELECT drive_id FROM snaps WHERE id = ?", (row,)).fetchone()
        assert got["drive_id"] is None
    finally:
        db.close()


def test_other_team_scoring_starts_a_new_drive(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=1, score_us=0, score_them=0, score_set=True)
        ctrl.result_and_call(outcome="+3", sit_raw="1&10 my 25", quarter=1,
                             score_us=0, score_them=3, score_set=True)
        ctrl.result_and_call(outcome="+3", sit_raw="2&7 my 28", quarter=1)
        assert [r["drive_id"] for r in _snaps(db)] == ["auto-d01", "auto-d02"]
    finally:
        db.close()


def test_undo_restores_drive_break(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=1)
        book = ctrl._book()
        book.close_from_form("td")
        assert book._drive_break is True
        book.undo()
        assert book._drive_break is False  # the TD was undone, so the drive goes on
    finally:
        db.close()


# --- game clock -----------------------------------------------------------

def test_clock_repeat_in_same_quarter_is_stale() -> None:
    assert fresh_game_clock("0:36", quarter=2, previous=(2, "0:36")) is None
    assert fresh_game_clock("0:36", quarter=3, previous=(2, "0:36")) == "0:36"
    assert fresh_game_clock("0:30", quarter=2, previous=(2, "0:36")) == "0:30"
    assert fresh_game_clock(None, quarter=2, previous=(2, "0:36")) is None


def test_live_loop_never_logs_a_stale_clock(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=2, game_clock="0:36")
        for raw in ("2&5 my 30", "3&1 my 34", "1&10 my 40"):
            ctrl.result_and_call(outcome="+5", sit_raw=raw, quarter=2, game_clock="0:36")
        ctrl.result_and_call(outcome="+5", sit_raw="2&5 my 45", quarter=2, game_clock="0:20")
        clocks = [r["game_clock"] for r in _snaps(db)]
        assert clocks == ["0:36", None, None, None]
        assert clocks.count("0:36") == 1
    finally:
        db.close()


def test_html_clears_clock_box_after_submit() -> None:
    import inspect

    from cfb_coach import live_server

    html = inspect.getsource(live_server.render_live_html)
    assert html.count('$("game-clock").value = "";') >= 2
    assert "last_outcome: currentOutcome()" in html


# --- score reconcile ------------------------------------------------------

def test_score_reconcile_records_gap() -> None:
    assert parse_score_pair("21-10") == (21, 10)
    assert parse_score_pair("") is None
    rec = score_reconcile("21-10", (14, 10))
    assert rec == {"session_score": "21-10", "snaplog_score": "14-10", "gap_for": 7,
                   "gap_against": 0, "status": "gap"}
    assert score_reconcile("14-10", (14, 10))["status"] == "match"
    unknown = score_reconcile(None, (14, 10))
    assert unknown["status"] == "unknown" and unknown["gap_for"] is None


def test_end_game_reconciles_snap_log_against_session(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=4, score_us=3, score_them=0, score_set=True)
        ctrl.result_and_call(outcome="+5", sit_raw="2&5 my 30", quarter=4)
        out = ctrl.end_game(result_wl="win", score="10-0")
        assert out["ok"]
        row = db.conn.execute(
            "SELECT snaplog_score, score_gap_for, score_gap_against, score_reconcile "
            "FROM game_sessions WHERE session_id = ?", (ctrl.session_id,)
        ).fetchone()
        assert row["snaplog_score"] == "3-0"
        assert (row["score_gap_for"], row["score_gap_against"]) == (7, 0)
        assert row["score_reconcile"] == "gap"
        assert "not in the snap log" in out["retrain_summary"]
        # The pending call got no snap: nothing was fabricated.
        assert len(_snaps(db)) == 1
    finally:
        db.close()


def test_end_game_logs_typed_final_result(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 opp 10", quarter=4, score_us=0, score_them=0, score_set=True)
        out = ctrl.end_game(result_wl="win", score="6-0", last_outcome="td")
        assert out["ok"]
        rows = _snaps(db)
        assert [r["result"] for r in rows] == ["td"]
        rec = db.conn.execute(
            "SELECT score_reconcile FROM game_sessions WHERE session_id = ?", (ctrl.session_id,)
        ).fetchone()
        assert rec["score_reconcile"] == "match"
    finally:
        db.close()


# --- accepted -------------------------------------------------------------

def test_accepted_only_when_verify_touched(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=1)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29", quarter=1)
        ctrl.result_and_call(outcome="+4", sit_raw="3&2 my 33", quarter=1,
                             executed_status="used_recommended", verify_touched=True)
        acc = [r["accepted"] for r in _snaps(db)]
        assert acc == ["unknown", "true"]
        execd = db.conn.execute("SELECT executed_status FROM snaps ORDER BY id").fetchall()
        assert execd[0]["executed_status"] == "unknown"
        assert resolve_mode(db) is CoachingMode.HEURISTIC
    finally:
        db.close()


# --- diagnostics ----------------------------------------------------------

def _diag_db(path: Path) -> sqlite3.Connection:
    db = CoachDB(path, seed=_seed())
    db.conn.execute(
        "INSERT INTO ml_decisions (decision_ts, game_id, session_id, snap_seq, mode, heuristic_formation, "
        "heuristic_play, final_formation, final_play, agree, decision_json, created_ts) VALUES "
        "('t', 'syn', 'syn', 1, 'experimental', 'H Form', 'H Play', 'M Form', 'M Play', 0, ?, 't')",
        (json.dumps({"experimental_offense": {"probability": 0.5, "uncertainty": 0.1}}),),
    )
    db.conn.execute(
        "INSERT INTO ml_decisions (decision_ts, game_id, session_id, snap_seq, mode, heuristic_formation, "
        "heuristic_play, final_formation, final_play, agree, decision_json, created_ts) VALUES "
        "('t', 'syn', 'syn', 2, 'experimental', 'A', 'B', 'A', 'B', 1, '{}', 't')"
    )
    db.conn.commit()
    for raw, down, dist, result, dec in (
        ("3rd and 2", 3, 2, "+3", 1),
        ("3rd and 8", 3, 8, "incomplete", None),
        ("3rd and 12", 3, 12, "", None),
        ("1st and 10", 1, 10, "int", 1),
    ):
        db.log_snap(opponent_id="cpu", side="offense", situation_raw=raw, our_call="M Form — M Play",
                    formation="M Form", play="M Play", down=down, distance=dist, result=result or None,
                    session_id="syn", ml_decision_id=dec)
    return db.conn


def test_diagnostics_third_down_and_ints(tmp_path: Path) -> None:
    conn = _diag_db(tmp_path / "madden27.db")
    diag = session_diagnostics(conn, "syn")
    third = diag["third_down"]
    assert third["short"]["converted"] == 1 and third["short"]["attempts"] == 1
    assert third["long"]["converted"] == 0
    assert third["very_long"]["unknown"] == 1
    assert len(diag["interceptions"]) == 1
    i = diag["interceptions"][0]
    assert i["model_pick"] == "M Form — M Play" and i["heuristic_pick"] == "H Form — H Play"
    assert i["confidence"] == 0.5 and i["coverage"] is None
    text = format_diagnostics(diag)
    assert "Interceptions: 1" in text and "coverage not logged" in text
    assert distance_bucket(None) is None and distance_bucket(5) == "medium"


def test_model_vs_heuristic_agreement_reported(tmp_path: Path) -> None:
    conn = _diag_db(tmp_path / "madden27.db")
    agree = session_diagnostics(conn, "syn")["model_vs_heuristic"]
    assert agree == {"decisions": 2, "agree": 1, "disagree": 1, "unknown": 0}


# --- macro armed diagnosis ------------------------------------------------

def test_macro_weights_armed_is_not_a_live_gate(tmp_path: Path) -> None:
    """``macro_weights.armed`` is only counted for the log; it does not gate a macro."""
    db = CoachDB(tmp_path / "madden27.db", seed=_seed())
    try:
        db.bump_macro_weight("cpu", "SYN", 0.2)
        assert db.conn.execute("SELECT armed FROM macro_weights WHERE macro='SYN'").fetchone()[0] == 0
        assert experimental_live._weights_armed_count(db, "cpu") == "0"
        import inspect

        src = inspect.getsource(experimental_live._resolve_model_adjustment)
        assert "macro_weights.armed`` is not this gate" in src
        assert "armed" not in inspect.getsource(experimental_live._heuristic_catalog_macro).replace(
            "arm", ""
        )
    finally:
        db.close()
