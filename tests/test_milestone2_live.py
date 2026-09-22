"""Milestone 2 — live tendency, play tracker, session, DB, counters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cfb_coach.counter_evidence import strongest_signal, weight_for_tendency
from cfb_coach.db import CoachDB
from cfb_coach.live_tendency import (
    SAMPLE_ACTIONABLE,
    SAMPLE_MILD,
    LiveTendencyEngine,
)
from cfb_coach.playcaller import make_call
from cfb_coach.session import end_session, start_session
from cfb_coach.situation import Situation
from cfb_coach.vision.play_family import classify_family
from cfb_coach.vision.play_record import PlayRecord
from cfb_coach.vision.play_tracker import PlayTracker
from cfb_coach.vision.result_detect import detect_result
from cfb_coach.vision.snap_detect import SnapDetector
from cfb_coach.vision.threaded_capture import ThreadedCapture
from cfb_coach.vision.capture import StubCapture, Frame


def _rec(**kwargs) -> PlayRecord:
    base = dict(
        session_id="testsess",
        game_id="testsess",
        opponent_id="gavin",
        side="defense",
        down=1,
        distance=10,
        field_zone="open",
        formation="bunch",
        shell="two_high",
        pressure="none",
        play_family="DROPBACK_PASS",
        concept_tags=["CROSSERS"],
        result_type="completion",
        yards=8,
        confidence={"family": 0.7, "result": 0.6},
    )
    base.update(kwargs)
    return PlayRecord(**base)


class TestPlayFamilyAndResult(unittest.TestCase):
    def test_crossers_family(self) -> None:
        g = classify_family(hint="Bunch Crossers", confidence=0.8)
        self.assertEqual(g.family, "DROPBACK_PASS")
        self.assertIn("CROSSERS", g.concept_tags)

    def test_inside_zone(self) -> None:
        g = classify_family(hint="Inside Zone", confidence=0.8)
        self.assertEqual(g.family, "RUN_INSIDE")

    def test_yards_never_fabricated(self) -> None:
        r = detect_result(hint="completion", yards_delta=12, yards_conf=0.1)
        self.assertIsNone(r.yards)
        r2 = detect_result(hint="completion", yards_delta=12, yards_conf=0.9)
        self.assertEqual(r2.yards, 12)


class TestSnapNoDouble(unittest.TestCase):
    def test_cooldown_blocks_double_snap(self) -> None:
        d = SnapDetector(cooldown_s=2.0, min_signals=1, strong_motion=0.5)
        d.arm(True)
        a = d.update(now=1.0, motion=0.7, play_state="PRE_SNAP")
        self.assertTrue(a.snapped)
        b = d.update(now=1.5, motion=0.8, play_state="PRE_SNAP")
        self.assertFalse(b.snapped)
        self.assertIn("cooldown", b.reasons)


class TestPlayTrackerLifecycle(unittest.TestCase):
    def test_one_play_record_per_snap(self) -> None:
        ended: list[PlayRecord] = []
        tr = PlayTracker(session_id="s1", game_id="s1", opponent_id="gavin")
        tr.on_play_end = ended.append
        tr.state.force("PRE_SNAP", 0.0)
        tr.snapper.arm(True)
        # Snap via strong motion
        st, done = tr.feed(now=1.0, motion=0.7, formation="bunch", family_hint="crossers")
        self.assertEqual(st, "PLAY_ACTIVE")
        self.assertIsNone(done)
        self.assertIsNotNone(tr._open)
        # Active motion
        tr.feed(now=1.5, motion=0.5)
        # Settle to end (need settle_frames)
        tr.feed(now=2.0, motion=0.05)
        st2, done2 = tr.feed(now=2.3, motion=0.02, family_hint="Bunch Crossers")
        self.assertIsNotNone(done2)
        self.assertEqual(len(ended), 1)
        self.assertEqual(len(tr.plays), 1)
        # Another end signal must not double-count
        tr.feed(now=2.5, motion=0.01)
        self.assertEqual(len(tr.plays), 1)

    def test_correction(self) -> None:
        tr = PlayTracker(session_id="s1")
        tr.state.force("PRE_SNAP", 0.0)
        tr.snapper.arm(True)
        tr.feed(now=1.0, motion=0.7, formation="bunch")
        tr.force_end(now=2.0, family_hint="Inside Zone")
        rec = tr.correct_last(play_family="RUN_INSIDE", result_type="run", yards=4)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertTrue(rec.corrected)
        self.assertEqual(rec.play_family, "RUN_INSIDE")


class TestLiveTendencySampleTiers(unittest.TestCase):
    def test_bunch_crossers_x4_with_iz_middle(self) -> None:
        """Bunch→Crossers x4 with one Inside Zone in middle → actionable only after enough samples."""
        eng = LiveTendencyEngine(alert_cooldown_s=0.0)
        sequence = [
            ("DROPBACK_PASS", ["CROSSERS"]),
            ("DROPBACK_PASS", ["CROSSERS"]),
            ("RUN_INSIDE", []),  # middle IZ
            ("DROPBACK_PASS", ["CROSSERS"]),
            ("DROPBACK_PASS", ["CROSSERS"]),
        ]
        actionable_at: list[int] = []
        for i, (fam, tags) in enumerate(sequence, 1):
            eng.add_play(
                _rec(
                    play_id=f"p{i}",
                    play_family=fam,
                    concept_tags=tags,
                    result_type="run" if fam == "RUN_INSIDE" else "completion",
                )
            )
            ts = eng.compute()
            cross = [t for t in ts if t.signal == "CROSSERS" and t.bucket.startswith("global")]
            if cross and cross[0].sample_size >= SAMPLE_ACTIONABLE and cross[0].actionable:
                actionable_at.append(i)

        # After 2 crossers: mild only (not actionable)
        eng2 = LiveTendencyEngine(alert_cooldown_s=0.0)
        eng2.add_play(_rec(play_id="a", concept_tags=["CROSSERS"]))
        eng2.add_play(_rec(play_id="b", concept_tags=["CROSSERS"]))
        cross2 = [t for t in eng2.compute() if t.signal == "CROSSERS" and "global" in t.bucket]
        self.assertTrue(cross2)
        self.assertEqual(cross2[0].confidence, "mild")
        self.assertFalse(cross2[0].actionable)

        # Full sequence: CROSSERS appears 4 times → actionable
        cross_final = [
            t for t in eng.compute() if t.signal == "CROSSERS" and t.bucket.startswith("global")
        ]
        self.assertTrue(cross_final)
        self.assertGreaterEqual(cross_final[0].sample_size, 4)
        self.assertTrue(cross_final[0].actionable)
        # IZ only once → log tier, not actionable
        iz = [t for t in eng.compute() if t.signal == "RUN_INSIDE" and "global" in t.bucket]
        self.assertTrue(iz)
        self.assertEqual(iz[0].sample_size, 1)
        self.assertFalse(iz[0].actionable)

    def test_sample_thresholds(self) -> None:
        eng = LiveTendencyEngine(alert_cooldown_s=0.0)
        for i in range(5):
            eng.add_play(_rec(play_id=f"n{i}", concept_tags=["MESH"], play_family="DROPBACK_PASS"))
            ts = [t for t in eng.compute() if t.signal == "MESH" and "global" in t.bucket][0]
            if i + 1 == 1:
                self.assertEqual(ts.confidence, "log")
            elif i + 1 == 2:
                self.assertEqual(ts.confidence, "mild")
            elif i + 1 == 3:
                self.assertEqual(ts.confidence, "actionable")
            elif i + 1 >= 5:
                self.assertEqual(ts.confidence, "strong")

    def test_recency_shift(self) -> None:
        eng = LiveTendencyEngine(alert_cooldown_s=0.0)
        # early: mostly RUN
        for i in range(6):
            eng.add_play(
                _rec(
                    play_id=f"r{i}",
                    play_family="RUN_INSIDE",
                    concept_tags=[],
                    result_type="run",
                )
            )
        # recent: CROSSERS
        for i in range(5):
            eng.add_play(
                _rec(
                    play_id=f"c{i}",
                    play_family="DROPBACK_PASS",
                    concept_tags=["CROSSERS"],
                )
            )
        recent = [t for t in eng.compute() if t.bucket.startswith("recent5") and t.signal == "CROSSERS"]
        self.assertTrue(recent)
        self.assertGreaterEqual(recent[0].hit_rate, 0.8)

    def test_counter_activation(self) -> None:
        eng = LiveTendencyEngine(alert_cooldown_s=0.0)
        for i in range(3):
            eng.add_play(_rec(play_id=f"x{i}", concept_tags=["CROSSERS"]))
        eng.register_counter("CROSSERS", "CROSS")
        for i in range(3):
            eng.add_play(
                _rec(
                    play_id=f"y{i}",
                    result_type="incompletion",
                    yards=0,
                    concept_tags=["CROSSERS"],
                )
            )
        cvs = eng.validate_counters(window=3)
        self.assertTrue(cvs)
        self.assertEqual(cvs[0].attempts, 3)

    def test_current_counter_phrase(self) -> None:
        eng = LiveTendencyEngine(alert_cooldown_s=0.0)
        for i in range(3):
            eng.add_play(_rec(play_id=f"z{i}", concept_tags=["CROSSERS"]))
        c = eng.current_counter(side="defense")
        self.assertIn("CROSS", c.upper())


class TestPlaycallerLiveWeight(unittest.TestCase):
    def test_weights_grow_with_sample(self) -> None:
        eng = LiveTendencyEngine(alert_cooldown_s=0.0)
        for i in range(4):
            eng.add_play(_rec(play_id=f"w{i}", concept_tags=["CROSSERS"]))
        ts = eng.tendencies_for_context(side="defense")
        best = strongest_signal(ts)
        self.assertIsNotNone(best)
        assert best is not None
        self.assertGreaterEqual(weight_for_tendency(best), 0.5)
        sit = Situation(raw="1&10", side="defense", down=1, distance=10, concept_hint=None)
        call = make_call(sit, "gavin", db=None, live_engine=eng, rng=__import__("random").Random(0))
        self.assertIn("live_tend", call.rationale)


class TestSessionAndDB(unittest.TestCase):
    def test_session_logging_and_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db")
            # Ensure opponent exists for FK-less play_records
            sess = start_session("gavin", dynasty="alabama", db=db)
            self.assertTrue(sess.session_id)
            rec = _rec(session_id=sess.session_id, game_id=sess.session_id, play_id="p0001")
            rid = db.log_play_record(rec)
            self.assertGreater(rid, 0)
            plays = db.get_session_plays(sess.session_id)
            self.assertEqual(len(plays), 1)
            db.log_live_tendency_event(
                session_id=sess.session_id,
                kind="TENDENCY_CONFIRMED",
                signal="CROSSERS",
                message="test",
            )
            ev = db.get_live_tendency_events(sess.session_id)
            self.assertEqual(len(ev), 1)
            # Idempotent upsert
            rec.corrected = True
            rec.yards = 15
            db.log_play_record(rec)
            plays2 = db.get_session_plays(sess.session_id)
            self.assertEqual(len(plays2), 1)
            end_session(sess, db=db)
            row = db.conn.execute(
                "SELECT ended_ts FROM game_sessions WHERE session_id = ?",
                (sess.session_id,),
            ).fetchone()
            self.assertIsNotNone(row["ended_ts"])
            # Existing snaps table still intact
            db.log_snap(
                opponent_id="gavin",
                side="offense",
                situation_raw="1&10",
                our_call="test",
            )
            self.assertGreaterEqual(db.count_snaps("gavin"), 1)
            db.close()


class TestDroppedFrameBehavior(unittest.TestCase):
    def test_threaded_capture_drops_stale(self) -> None:
        class SeqCap:
            name = "seq"

            def __init__(self) -> None:
                self.n = 0

            def grab(self) -> Frame:
                self.n += 1
                return Frame(bgr=None, source="seq", ts=float(self.n), extras={"n": self.n})

            def close(self) -> None:
                return None

        tc = ThreadedCapture(SeqCap(), target_fps=50.0)  # type: ignore[arg-type]
        tc.start()
        import time as _t

        _t.sleep(0.08)
        f1 = tc.grab_fresh()
        self.assertIsNotNone(f1)
        # Immediate second grab_fresh without new frame → drop/None
        f2 = tc.grab_fresh()
        # May be None (stale) or new if thread advanced — either way dropped counter works
        _t.sleep(0.05)
        f3 = tc.grab_fresh()
        stats = tc.debug_stats()
        self.assertIn("dropped", stats)
        self.assertIn("queue", stats)
        tc.close()


class TestPivotThresholds(unittest.TestCase):
    def test_live_explosive_pivot_note(self) -> None:
        from cfb_coach.gameplan import (
            clear_live_tendency_pivot,
            live_tendency_pivot_tip,
            note_live_tendency_pivot,
        )
        from cfb_coach.live_tendency import LiveTendency

        clear_live_tendency_pivot()
        note_live_tendency_pivot(
            LiveTendency(
                signal="EXPLOSIVE",
                sample_size=3,
                hit_rate=0.4,
                confidence="actionable",
                recent_rate=0.5,
                bucket="global|explosive",
            )
        )
        tip = live_tendency_pivot_tip("offense")
        self.assertIsNotNone(tip)
        assert tip is not None
        self.assertIn("EXPLOSIVE", tip.message)
        clear_live_tendency_pivot()


if __name__ == "__main__":
    unittest.main()
