"""Live watch UX: heartbeat, status cadence, non-blocking stdin (v1.9.1)."""

from __future__ import annotations

import io
import unittest
from unittest import mock

from cfb_coach.vision.look import DefenseLook
from cfb_coach.watch import (
    StdinCommandQueue,
    _CAPTURE_OK,
    _dispatch_live_command,
    _status_line,
    _waiting_frames_msg,
    _TROUBLESHOOT_NO_FRAMES,
)


class TestHeartbeatAndStatus(unittest.TestCase):
    def test_waiting_frames_includes_capture_name(self) -> None:
        msg = _waiting_frames_msg("threaded:dxcam")
        self.assertIn("waiting for frames", msg)
        self.assertIn("Remote Play", msg)
        self.assertIn("threaded:dxcam", msg)

    def test_status_line_has_fps_state_look(self) -> None:
        line = _status_line(
            "PRE | BUNCH | 2-HIGH",
            fps=7.5,
            play_state="PRE_SNAP",
            look_label="front=nickel shell=cover3 pressure=none conf=0.85",
        )
        self.assertIn("fps=7.5", line)
        self.assertIn("state=PRE_SNAP", line)
        self.assertIn("look=front=nickel", line)

    def test_troubleshoot_mentions_recalibrate_and_region(self) -> None:
        self.assertIn("5s", _TROUBLESHOOT_NO_FRAMES)
        self.assertIn("calibrate", _TROUBLESHOOT_NO_FRAMES.lower())
        self.assertIn("--screen-region", _TROUBLESHOOT_NO_FRAMES)
        self.assertEqual(_CAPTURE_OK, "capture OK — LIVE tips below")


class TestStdinCommandQueue(unittest.TestCase):
    def test_push_poll(self) -> None:
        q = StdinCommandQueue()
        self.assertIsNone(q.poll())
        q.push("tips")
        q.push("q")
        self.assertEqual(q.poll(), "tips")
        self.assertEqual(q.poll(), "q")
        self.assertIsNone(q.poll())
        q.close()

    def test_start_noop_when_not_tty(self) -> None:
        q = StdinCommandQueue()
        with mock.patch("sys.stdin") as fake:
            fake.isatty.return_value = False
            q.start()
        self.assertFalse(q.enabled)
        self.assertIsNone(q._thread)
        q.close()


class TestDispatchLiveCommand(unittest.TestCase):
    def test_quit_and_help_and_look(self) -> None:
        emitted: list[DefenseLook] = []

        def emit(look: DefenseLook, **_kw) -> None:
            emitted.append(look)

        look = DefenseLook(front="even", shell="two_high", pressure="none", confidence=0.5)
        tips = ["stay calm"]

        action, _ = _dispatch_live_command(
            "q",
            emit=emit,
            look=look,
            tips=tips,
            play_tracker=None,
            live_engine=None,
            db=None,
            overlay_path=None,
            inventory=None,
            opponent=None,
        )
        self.assertEqual(action, "quit")

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            action, _ = _dispatch_live_command(
                "help",
                emit=emit,
                look=look,
                tips=tips,
                play_tracker=None,
                live_engine=None,
                db=None,
                overlay_path=None,
                inventory=None,
                opponent=None,
            )
        self.assertEqual(action, "ok")
        self.assertIn("Typed commands", buf.getvalue())

        action, _ = _dispatch_live_command(
            "look nickel c3 blitz_left",
            emit=emit,
            look=look,
            tips=tips,
            play_tracker=None,
            live_engine=None,
            db=None,
            overlay_path=None,
            inventory=None,
            opponent=None,
        )
        self.assertEqual(action, "ok")
        self.assertTrue(emitted)
        self.assertEqual(emitted[-1].front, "nickel")


class TestDemoStillWorks(unittest.TestCase):
    def test_demo_once_exits_clean(self) -> None:
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-m", "cfb_coach", "watch", "--demo", "--once"],
            cwd="/workspace/cfb-coach",
            capture_output=True,
            text=True,
            timeout=30,
            env={**dict(**__import__("os").environ), "PYTHONPATH": "/workspace/cfb-coach"},
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
        self.assertIn("CO-PILOT", r.stdout)
        # Must not enter pipeline / waiting-for-frames path
        self.assertNotIn("waiting for frames", r.stdout)
        self.assertNotIn("Pipeline capture=", r.stdout)


if __name__ == "__main__":
    unittest.main()
