"""Live watch UX: heartbeat cadence, window match, non-blocking stdin (v1.9.2)."""

from __future__ import annotations

import argparse
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from cfb_coach.vision.capture import window_search_needles
from cfb_coach.vision.look import DefenseLook
from cfb_coach.watch import (
    StdinCommandQueue,
    _CAPTURE_OK,
    _HEARTBEAT_AFTER_TROUBLE_S,
    _HEARTBEAT_INTERVAL_S,
    _dispatch_live_command,
    _heartbeat_interval,
    _should_emit_heartbeat,
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

    def test_heartbeat_intervals(self) -> None:
        self.assertEqual(_HEARTBEAT_INTERVAL_S, 5.0)
        self.assertEqual(_HEARTBEAT_AFTER_TROUBLE_S, 15.0)
        self.assertEqual(_heartbeat_interval(False), 5.0)
        self.assertEqual(_heartbeat_interval(True), 15.0)

    def test_should_emit_heartbeat_gates(self) -> None:
        # Too early
        self.assertFalse(
            _should_emit_heartbeat(
                now=3.0,
                loop_start=0.0,
                last_heartbeat=0.0,
                last_frame_wall=0.0,
                troub_printed=False,
                typing=False,
            )
        )
        # First at 5s
        self.assertTrue(
            _should_emit_heartbeat(
                now=5.0,
                loop_start=0.0,
                last_heartbeat=0.0,
                last_frame_wall=0.0,
                troub_printed=False,
                typing=False,
            )
        )
        # After troubleshoot: 15s gap
        self.assertFalse(
            _should_emit_heartbeat(
                now=10.0,
                loop_start=0.0,
                last_heartbeat=5.0,
                last_frame_wall=0.0,
                troub_printed=True,
                typing=False,
            )
        )
        self.assertTrue(
            _should_emit_heartbeat(
                now=20.0,
                loop_start=0.0,
                last_heartbeat=5.0,
                last_frame_wall=0.0,
                troub_printed=True,
                typing=False,
            )
        )
        # Never mid-keystroke
        self.assertFalse(
            _should_emit_heartbeat(
                now=20.0,
                loop_start=0.0,
                last_heartbeat=5.0,
                last_frame_wall=0.0,
                troub_printed=True,
                typing=True,
            )
        )
        # Recent frame → no heartbeat
        self.assertFalse(
            _should_emit_heartbeat(
                now=20.0,
                loop_start=0.0,
                last_heartbeat=0.0,
                last_frame_wall=19.5,
                troub_printed=False,
                typing=False,
            )
        )

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

    def test_troubleshoot_mentions_list_windows_and_calib(self) -> None:
        self.assertIn("5s", _TROUBLESHOOT_NO_FRAMES)
        self.assertIn("--list-windows", _TROUBLESHOOT_NO_FRAMES)
        self.assertIn("calibrate", _TROUBLESHOOT_NO_FRAMES.lower())
        self.assertIn("--screen-region", _TROUBLESHOOT_NO_FRAMES)
        self.assertIn("vision_calib.json", _TROUBLESHOOT_NO_FRAMES)
        self.assertEqual(_CAPTURE_OK, "capture OK — LIVE tips below")


class TestWindowNeedles(unittest.TestCase):
    def test_xbox_expands_aliases(self) -> None:
        needles = window_search_needles("Xbox")
        self.assertEqual(needles[0], "Xbox")
        joined = " | ".join(n.lower() for n in needles)
        self.assertIn("remote play", joined)
        self.assertIn("game bar", joined)

    def test_custom_title_stays_primary(self) -> None:
        needles = window_search_needles("My Custom Capture")
        self.assertEqual(needles, ["My Custom Capture"])

    def test_case_preserved_primary(self) -> None:
        needles = window_search_needles("remote play")
        self.assertEqual(needles[0], "remote play")
        self.assertTrue(any("Xbox" == n for n in needles))


class TestStdinCommandQueue(unittest.TestCase):
    def test_push_poll(self) -> None:
        q = StdinCommandQueue()
        self.assertIsNone(q.poll())
        q.push("tips")
        q.push("q")
        self.assertEqual(q.poll(), "tips")
        self.assertEqual(q.poll(), "q")
        self.assertIsNone(q.poll())
        self.assertFalse(q.has_partial_input())
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


class TestCliFlags(unittest.TestCase):
    def test_list_windows_and_screen_region_nargs(self) -> None:
        from cfb_coach.cli import build_parser

        p = build_parser()
        args = p.parse_args(["watch", "--list-windows"])
        self.assertTrue(args.list_windows)

        args = p.parse_args(["watch", "--screen-region"])
        self.assertEqual(args.screen_region, "calib")

        args = p.parse_args(["watch", "--screen-region", "10,20,800,600"])
        self.assertEqual(args.screen_region, "10,20,800,600")


class TestScreenRegionCalib(unittest.TestCase):
    def test_calib_flag_prefers_mss(self) -> None:
        from cfb_coach.vision.pipeline import build_capture_from_args

        calib = {
            "screen_region": {"left": 1, "top": 2, "width": 100, "height": 50},
            "crop": {"left": 1, "top": 2, "width": 100, "height": 50},
        }
        args = SimpleNamespace(
            image=None,
            video=None,
            window=None,
            screen_region="calib",
            device=None,
            live=False,
        )
        with mock.patch("cfb_coach.vision.pipeline.load_calib", return_value=calib):
            with mock.patch(
                "cfb_coach.vision.pipeline.MssRegionCapture"
            ) as mss_cls:
                mss_cls.return_value = mock.Mock(name="mss")
                # Dxcam may or may not import — prefer_mss should hit mss first
                with mock.patch(
                    "cfb_coach.vision.pipeline.DxcamWindowCapture",
                    side_effect=ImportError("skip"),
                ):
                    cap = build_capture_from_args(args)
        mss_cls.assert_called_once()
        self.assertIs(cap, mss_cls.return_value)


class TestDemoStillWorks(unittest.TestCase):
    def test_demo_once_exits_clean(self) -> None:
        import os
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-m", "cfb_coach", "watch", "--demo", "--once"],
            cwd="/workspace/cfb-coach",
            capture_output=True,
            text=True,
            timeout=30,
            env={**dict(os.environ), "PYTHONPATH": "/workspace/cfb-coach"},
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
        self.assertIn("CO-PILOT", r.stdout)
        self.assertIn("v1.9.2", r.stdout)
        # Must not enter pipeline / waiting-for-frames path
        self.assertNotIn("waiting for frames", r.stdout)
        self.assertNotIn("Pipeline capture=", r.stdout)


if __name__ == "__main__":
    unittest.main()
