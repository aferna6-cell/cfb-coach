"""A clock typed into the situation text is current-snap evidence.

Regression for 9297ada, which cleared ``clock_seconds`` whenever the clock box
was blank, discarding a clock the user typed as ``... clock 5:15``. A clock that
WAS posted and rejected (stale) must still be dropped everywhere.

Synthetic fixtures only. No real game rows, sessions, or scores.
"""
from __future__ import annotations

from pathlib import Path

from tests.madden_ml.test_cpu_audit_fixes import _ctrl
from tests.madden_ml.test_cpu_audit_fixes_2 import _rows


def test_text_clock_is_kept_when_clock_box_is_blank(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 40 clock 5:15", quarter=3)
        assert ctrl.last_sit.extras.get("clock_seconds") == 315
        ctrl.result_and_call(outcome="+3", sit_raw="2&7 my 43", quarter=3)
        # The second snap names no clock at all: unknown, never carried over.
        ctrl.result_and_call(outcome="+2", sit_raw="3&4 my 46", quarter=3)
        rows = _rows(db)
        assert [r["game_clock"] for r in rows] == ["5:15", None]
        assert rows[0]["game_clock_null_reason"] is None
    finally:
        db.close()


def test_rejected_stale_posted_clock_clears_text_clock_too(tmp_path: Path) -> None:
    db, ctrl = _ctrl(tmp_path)
    try:
        ctrl.call_only("1&10 my 25", quarter=2, game_clock="5:00", game_clock_fresh=True)
        ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29 clock 5:00", quarter=2,
                             game_clock="5:00", game_clock_fresh=False)
        # The stale snap is the pending one: nothing about its clock may survive.
        assert ctrl.last_sit.extras.get("clock_seconds") is None
        assert ctrl.last_sit.extras.get("game_clock") is None
        ctrl.result_and_call(outcome="+2", sit_raw="3&4 my 31", quarter=2,
                             game_clock="4:30", game_clock_fresh=True)
        last = _rows(db)[-1]
        assert last["game_clock"] is None
        assert last["game_clock_null_reason"] == "stale_resubmit"
    finally:
        db.close()
