"""Close a live game_sessions row, including after a crash.

Synthetic rows only. These tests never open a real Madden database.
"""

from __future__ import annotations

import selectors
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model.inference import resolve_mode
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.session import (
    abandon_owner,
    close_armed_sessions,
    disarm_session_exit,
    recover_unclosed_sessions,
)
from cfb_coach.situation import parse_situation

_CHILD = """
import sys
import time
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.session import arm_session_exit, start_session

path, us, them = sys.argv[1:]
db = CoachDB(Path(path))
sess = start_session("cpu", db=db, notes="synthetic-live")
db.log_snap(
    opponent_id="cpu",
    side="offense",
    situation_raw="1&10 my 25",
    our_call="synthetic",
    session_id=sess.session_id,
    score_for=int(us),
    score_against=int(them),
)
arm_session_exit(db, sess.session_id)
print(sess.session_id, flush=True)
time.sleep(60)
"""


class _Call:
    def __init__(self) -> None:
        self.side = "offense"
        self.formation = "Gun Bunch"
        self.play = "Mesh"
        self.macro = None
        self.rationale = "synthetic"

    def format(self) -> str:
        return f"{self.formation} — {self.play}"


def _seed() -> dict:
    return {
        "opponents": {
            "cpu": {
                "display_name": "CPU",
                "team_now": "CPU",
                "skill": "cpu",
                "confidence": "high",
                "profile_json": "{}",
            }
        }
    }


def _controller(db: CoachDB) -> LivePlayController:
    return LivePlayController(
        db=db,
        opponent_id="cpu",
        make_call=lambda sit, **k: _Call(),
        parse_situation=parse_situation,
        learn_summary=lambda: "synthetic",
        brand="Madden 27 Franchise",
        play_cmd="x",
        cpu_only=True,
    )


def _row(db: CoachDB, session_id: str):
    return db.conn.execute(
        "SELECT ended_ts, result_wl, score, play_count, recovered, notes "
        "FROM game_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()


class SessionCloseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._tmp.name) / "synthetic.db", seed=_seed())
        self.assertIs(resolve_mode(self.db), CoachingMode.HEURISTIC)

    def tearDown(self) -> None:
        sid = getattr(self, "_open_sid", None)
        if sid:
            disarm_session_exit(sid)
        try:
            self.db.close()
        except sqlite3.ProgrammingError:
            pass
        self._tmp.cleanup()

    def test_normal_end_writes_result_score_and_play_count(self) -> None:
        ctrl = _controller(self.db)
        ctrl.start()
        self._open_sid = ctrl.session_id
        ctrl.set_live_score(7, 3)
        self.assertTrue(ctrl.call_only("1&10 my 25")["ok"])
        logged = ctrl.result_and_call(outcome="+4", sit_raw="2&6 my 29")
        self.assertTrue(logged["ok"])
        ended = ctrl.end_game(result_wl="win", score="24-17")
        self.assertTrue(ended["ok"])

        row = _row(self.db, ctrl.session_id)
        self.assertTrue(row["ended_ts"])
        self.assertEqual(row["result_wl"], "win")
        self.assertEqual(row["score"], "24-17")
        self.assertEqual(int(row["play_count"]), 1)
        self.assertEqual(int(row["recovered"] or 0), 0)
        self.assertNotIn("recovered", (row["notes"] or "").split())

        # The exit handler must not replace a game that already closed.
        close_armed_sessions(ctrl.session_id)
        again = _row(self.db, ctrl.session_id)
        self.assertEqual(again["score"], "24-17")
        self.assertEqual(again["result_wl"], "win")
        self.assertEqual(again["ended_ts"], row["ended_ts"])

        nxt = _controller(self.db)
        nxt.start()
        self._open_sid = nxt.session_id
        self.assertNotEqual(nxt.session_id, ctrl.session_id)
        kept = _row(self.db, ctrl.session_id)
        self.assertEqual(kept["score"], "24-17")
        self.assertEqual(int(kept["recovered"] or 0), 0)
        fresh = _row(self.db, nxt.session_id)
        self.assertFalse(fresh["ended_ts"])
        self.assertIs(resolve_mode(self.db), CoachingMode.HEURISTIC)

    def test_crash_then_next_start_recovers_last_logged_score(self) -> None:
        ctrl = _controller(self.db)
        ctrl.start()
        abandoned = ctrl.session_id
        self.assertTrue(
            ctrl.call_only("1&10 my 25", score_us=10, score_them=7, score_set=True)["ok"]
        )
        self.assertTrue(
            ctrl.result_and_call(
                outcome="+6",
                sit_raw="2&4 my 31",
                score_us=17,
                score_them=21,
                score_set=True,
            )["ok"]
        )
        self.assertTrue(ctrl.result_and_call(outcome="incomplete", sit_raw="3&4 my 31")["ok"])
        # A later snap with no score must not erase the last one that had one.
        self.db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1&10 my 40",
            our_call="no score on this snap",
            session_id=abandoned,
        )
        before = _row(self.db, abandoned)
        self.assertFalse(before["ended_ts"])
        self.assertEqual(int(before["play_count"]), 0)

        # Crash: the owner is gone and the exit handler never runs.
        abandon_owner(self.db, abandoned)
        nxt = _controller(self.db)
        nxt.start()
        self._open_sid = nxt.session_id

        row = _row(self.db, abandoned)
        self.assertTrue(row["ended_ts"])
        self.assertEqual(row["score"], "17-21")
        self.assertEqual(row["result_wl"], "loss")
        self.assertEqual(int(row["play_count"]), 3)
        self.assertEqual(int(row["recovered"]), 1)
        self.assertIn("recovered", (row["notes"] or "").split())
        fresh = _row(self.db, nxt.session_id)
        self.assertFalse(fresh["ended_ts"])
        self.assertEqual(int(fresh["recovered"] or 0), 0)
        self.assertIs(resolve_mode(self.db), CoachingMode.HEURISTIC)

    def test_exit_handler_closes_from_notes_without_recovered_flag(self) -> None:
        ctrl = _controller(self.db)
        ctrl.start()
        sid = ctrl.session_id
        self._open_sid = sid
        self.db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1&10 my 25",
            our_call="early",
            session_id=sid,
            score_for=3,
            score_against=0,
        )
        self.db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="2&7 my 40",
            our_call="late",
            session_id=sid,
            notes="score_us=24 score_them=21",
        )
        close_armed_sessions(sid)
        row = _row(self.db, sid)
        self.assertTrue(row["ended_ts"])
        self.assertEqual(row["score"], "24-21")
        self.assertEqual(row["result_wl"], "win")
        self.assertEqual(int(row["play_count"]), 2)
        self.assertEqual(int(row["recovered"] or 0), 0)
        self.assertNotIn("recovered", (row["notes"] or "").split())
        self.assertIs(resolve_mode(self.db), CoachingMode.HEURISTIC)

    def test_live_owner_is_not_recovered_by_another_process(self) -> None:
        proc = self._spawn_holder(7, 0)
        try:
            sid = self._read_child_session(proc)
            recovered = recover_unclosed_sessions(self.db)
            self.assertNotIn(sid, recovered)
            row = _row(self.db, sid)
            self.assertFalse(row["ended_ts"])
            self.assertEqual(int(row["recovered"] or 0), 0)
            self.assertIsNone(row["score"])
            proc.kill()
            proc.wait(timeout=5)
            recovered = recover_unclosed_sessions(self.db)
            self.assertIn(sid, recovered)
            closed = _row(self.db, sid)
            self.assertTrue(closed["ended_ts"])
            self.assertEqual(closed["score"], "7-0")
            self.assertEqual(closed["result_wl"], "win")
            self.assertEqual(int(closed["recovered"]), 1)
        finally:
            self._stop_child(proc)

    def test_owner_close_supersedes_recovered_row(self) -> None:
        ctrl = _controller(self.db)
        ctrl.start()
        self._open_sid = ctrl.session_id
        self.db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1&10 my 25",
            our_call="synthetic",
            session_id=ctrl.session_id,
            score_for=7,
            score_against=0,
        )
        premature = "2026-01-01T00:00:00+00:00"
        self.db.conn.execute(
            "UPDATE game_sessions SET ended_ts = ?, result_wl = ?, score = ?, "
            "play_count = ?, recovered = 1, notes = ? WHERE session_id = ?",
            (premature, "win", "7-0", 1, "html-live recovered", ctrl.session_id),
        )
        self.db.conn.commit()
        ended = ctrl.end_game(result_wl="win", score="14-7")
        self.assertTrue(ended["ok"])
        row = _row(self.db, ctrl.session_id)
        self.assertEqual(row["score"], "14-7")
        self.assertEqual(row["result_wl"], "win")
        self.assertEqual(int(row["recovered"] or 0), 0)
        self.assertNotEqual(row["ended_ts"], premature)
        self.assertNotIn("recovered", (row["notes"] or "").split())
        self.assertIs(resolve_mode(self.db), CoachingMode.HEURISTIC)

    def test_exit_handler_lock_leaves_session_recoverable(self) -> None:
        ctrl = _controller(self.db)
        ctrl.start()
        sid = ctrl.session_id
        self._open_sid = sid
        self.db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1&10 my 25",
            our_call="synthetic",
            session_id=sid,
            score_for=9,
            score_against=3,
        )
        holder = sqlite3.connect(self.db.path, timeout=5)
        try:
            holder.execute("BEGIN EXCLUSIVE")
            started = time.perf_counter()
            close_armed_sessions(sid)
            elapsed = time.perf_counter() - started
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        self.assertLess(elapsed, 1.0)
        still = _row(self.db, sid)
        self.assertFalse(still["ended_ts"])
        self.assertEqual(int(still["recovered"] or 0), 0)
        abandon_owner(self.db, sid)
        self.assertIn(sid, recover_unclosed_sessions(self.db))
        closed = _row(self.db, sid)
        self.assertTrue(closed["ended_ts"])
        self.assertEqual(closed["score"], "9-3")
        self.assertEqual(int(closed["recovered"]), 1)

    def test_sigterm_closes_session(self) -> None:
        proc = self._spawn_holder(14, 7)
        try:
            sid = self._read_child_session(proc)
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
            row = _row(self.db, sid)
            self.assertTrue(row["ended_ts"])
            self.assertEqual(row["score"], "14-7")
            self.assertEqual(row["result_wl"], "win")
            self.assertEqual(int(row["recovered"] or 0), 0)
            self.assertNotIn("recovered", (row["notes"] or "").split())
        finally:
            self._stop_child(proc)

    def _spawn_holder(self, us: int, them: int) -> subprocess.Popen[str]:
        self.db.close()
        return subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(self.db.path), str(us), str(them)],
            cwd="/workspace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _read_child_session(self, proc: subprocess.Popen[str]) -> str:
        assert proc.stdout is not None
        watcher = selectors.DefaultSelector()
        watcher.register(proc.stdout, selectors.EVENT_READ)
        if not watcher.select(10):
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read()
            self.fail(f"holder produced no session id: {err}")
        sid = proc.stdout.readline().strip()
        self.assertTrue(sid)
        self.db = CoachDB(self.db.path)
        return sid

    def _stop_child(self, proc: subprocess.Popen[str] | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.kill()
        proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
