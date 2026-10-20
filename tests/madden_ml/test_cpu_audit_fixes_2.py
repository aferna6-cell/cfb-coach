"""PR #80 follow-up: clock staleness signal, countdown guard, drive_id_source.

Synthetic fixtures only. No real game rows, sessions, or scores.
"""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

from cfb_coach import live_server
from cfb_coach.db import CoachDB
from cfb_coach.live_server import check_game_clock, fresh_game_clock
from cfb_coach.madden.model.call_outcome_log import (
    context_from_situation,
    drive_id_is_verified,
    drive_id_source_for_log,
)

from tests.madden_ml.test_cpu_audit_fixes import _ctrl, _seed


def _rows(db: CoachDB):
    return db.conn.execute(
        "SELECT drive_id, drive_id_source, game_clock, game_clock_null_reason, "
        "game_clock_suspect_reason "
        "FROM snaps ORDER BY id"
    ).fetchall()


# --- clock: pure function ---------------------------------------------------

def test_equal_clock_alone_is_not_stale() -> None:
    assert check_game_clock("4:12", quarter=2, previous=(2, "4:12"), fresh=True) == ("4:12", None)
    # No flag (older client): an equal value alone never counts as stale.
    assert check_game_clock("4:12", quarter=2, previous=(2, "4:12")) == ("4:12", None)


def test_resubmitted_clock_without_new_entry_is_stale() -> None:
    assert check_game_clock("4:12", quarter=2, previous=(2, "4:12"), fresh=False) == (
        None, "stale_resubmit")


def test_clock_going_up_in_same_quarter_is_rejected() -> None:
    assert check_game_clock("6:10", quarter=2, previous=(2, "5:00"), fresh=True) == (
        None, "clock_went_up")
    assert fresh_game_clock("4:59", quarter=2, previous=(2, "5:00"), fresh=True) == "4:59"


def test_new_quarter_or_ot_resets_checks() -> None:
    assert check_game_clock("15:00", quarter=3, previous=(2, "0:12"), fresh=True) == ("15:00", None)
    assert check_game_clock("10:00", quarter="OT", previous=(4, "0:00"), fresh=False) == ("10:00", None)
    assert check_game_clock(None, quarter=2, previous=(2, "5:00")) == (None, None)


# --- clock: live loop -------------------------------------------------------

def test_live_loop_keeps_legit_repeat_and_nulls_stale_and_backwards(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=2, game_clock="5:00", game_clock_fresh=True)
        # Incompletion: the clock really stopped, and the box was typed again.
        ctrl.result_and_call(outcome="incomplete", sit_raw="2&10 my 25", quarter=2,
                             game_clock="5:00", game_clock_fresh=True)
        # Same value sent with no new entry: stale.
        ctrl.result_and_call(outcome="+4", sit_raw="3&6 my 29", quarter=2,
                             game_clock="5:00", game_clock_fresh=False)
        # Freshly typed clock went up within the quarter: stored, flagged suspect.
        ctrl.result_and_call(outcome="+7", sit_raw="1&10 my 36", quarter=2,
                             game_clock="6:10", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+3", sit_raw="2&7 my 39", quarter=2,
                             game_clock="4:40", game_clock_fresh=True)
        # New quarter resets the countdown check.
        ctrl.result_and_call(outcome="+2", sit_raw="1&10 my 25", quarter=3,
                             game_clock="15:00", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+1", sit_raw="2&9 my 26", quarter=3)
        got = [(r["game_clock"], r["game_clock_null_reason"], r["game_clock_suspect_reason"])
               for r in _rows(db)]
        assert got == [
            ("5:00", None, None),
            ("5:00", None, None),
            (None, "stale_resubmit", None),
            ("6:10", None, "clock_went_up"),
            ("4:40", None, None),
            ("15:00", None, None),
        ]
    finally:
        db.close()


def test_page_sends_fresh_flag_and_resets_it() -> None:
    html = inspect.getsource(live_server.render_live_html)
    assert html.count("game_clock_fresh: clockTyped") == 2
    assert html.count("clockTyped = false;") >= 2
    assert '$("game-clock").addEventListener("input"' in html


def test_fresh_flag_only_accepts_real_bools() -> None:
    assert live_server._fresh_flag(True) is True
    assert live_server._fresh_flag(False) is False
    assert live_server._fresh_flag("false") is None
    assert live_server._fresh_flag(None) is None


# --- drive_id_source --------------------------------------------------------

def test_drive_id_source_explicit_inferred_and_typed_auto(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=1)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29 drive id 7", quarter=1)
        ctrl.result_and_call(outcome="+4", sit_raw="3&2 my 33 drive id auto-d03", quarter=1)
        ctrl.result_and_call(outcome="+4", sit_raw="1&10 my 37", quarter=1)
        got = [(r["drive_id"], r["drive_id_source"]) for r in _rows(db)]
        assert got[0] == ("auto-d01", "inferred")
        assert got[1] == ("7", "explicit")
        assert got[2] == ("auto-d03", "explicit")  # hand-typed, so explicit
        dec = db.conn.execute(
            "SELECT drive_id_source FROM ml_decisions WHERE drive_id IS NOT NULL ORDER BY id"
        ).fetchall()
        assert all(r[0] in ("explicit", "inferred") for r in dec)
    finally:
        db.close()


def test_source_helpers() -> None:
    assert drive_id_source_for_log("inferred") == "inferred"
    assert drive_id_source_for_log(None) == "explicit"
    assert drive_id_is_verified("auto-d03", "explicit") is True
    assert drive_id_is_verified("auto-d01", "inferred") is False
    assert drive_id_is_verified("d1", None) is False  # older row, source unknown
    assert drive_id_is_verified(None, "explicit") is False

    class _S:
        raw = "1&10 drive id 4"
        extras: dict = {}
        score_us = score_them = None

    assert context_from_situation(_S())["drive_id_source"] == "explicit"
    assert context_from_situation(None)["drive_id_source"] is None


def test_migration_is_additive_and_older_rows_stay_null(tmp_path: Path) -> None:
    path = tmp_path / "madden27.db"
    db = CoachDB(path, seed=_seed())
    try:
        row = db.log_snap(opponent_id="cpu", side="offense", situation_raw="1st and 10",
                          our_call="X — Y", result="+1", session_id="syn", drive_id="d1")
        got = db.conn.execute(
            "SELECT drive_id, drive_id_source, game_clock_null_reason FROM snaps WHERE id = ?",
            (row,),
        ).fetchone()
        assert got["drive_id"] == "d1"
        assert got["drive_id_source"] is None and got["game_clock_null_reason"] is None
    finally:
        db.close()
    db = CoachDB(path, seed=_seed())  # reopening re-runs the migration: idempotent
    db.close()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(snaps)")}
    assert {"drive_id_source", "game_clock_null_reason", "game_clock_suspect_reason"} <= cols
    for table in ("ml_decisions", "ml_outcomes"):
        tcols = {r[1] for r in sqlite3.connect(path).execute(f"PRAGMA table_info({table})")}
        if tcols:
            assert "game_clock_suspect_reason" in tcols


def _clocks(db: CoachDB):
    return [(r["game_clock"], r["game_clock_null_reason"], r["game_clock_suspect_reason"])
            for r in _rows(db)]


def test_typo_low_clock_does_not_lock_out_quarter(tmp_path: Path) -> None:
    """QA HOLD-1: 1:12 typed for 11:12, then real clocks are all kept."""
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=2, game_clock="11:30", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29", quarter=2,
                             game_clock="1:12", game_clock_fresh=True)
        for i, clk in enumerate(("10:50", "10:30", "10:05")):
            ctrl.result_and_call(outcome="+3", sit_raw=f"1&10 my {40 + i * 5}", quarter=2,
                                 game_clock=clk, game_clock_fresh=True)
        ctrl.result_and_call(outcome="+1", sit_raw="2&9 my 60", quarter=2)
        assert _clocks(db) == [
            ("11:30", None, None),
            ("1:12", None, None),
            ("10:50", None, "clock_went_up"),
            ("10:30", None, None),
            ("10:05", None, None),
        ]
    finally:
        db.close()


def test_missed_quarter_rollover_keeps_new_quarter_clocks(tmp_path: Path) -> None:
    """QA HOLD-2: 15:00 after 0:20 with the quarter not updated."""
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=1, game_clock="0:20", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29", quarter=1,
                             game_clock="15:00", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+4", sit_raw="3&2 my 33", quarter=1,
                             game_clock="14:31", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+4", sit_raw="1&10 my 37", quarter=1,
                             game_clock="14:02", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+1", sit_raw="2&9 my 38", quarter=1)
        assert _clocks(db) == [
            ("0:20", None, None),
            ("15:00", None, "clock_went_up"),
            ("14:31", None, None),
            ("14:02", None, None),
        ]
    finally:
        db.close()


def test_stale_resubmit_and_unknown_flag_keep_old_behavior(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=2, game_clock="5:00", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29", quarter=2,
                             game_clock="5:00", game_clock_fresh=False)
        # Outside client with no flag: a higher clock is still dropped.
        ctrl.result_and_call(outcome="+4", sit_raw="3&2 my 33", quarter=2,
                             game_clock="6:00")
        ctrl.result_and_call(outcome="+1", sit_raw="1&10 my 40", quarter=2)
        assert _clocks(db) == [
            ("5:00", None, None),
            (None, "stale_resubmit", None),
            (None, "clock_went_up", None),
        ]
    finally:
        db.close()
