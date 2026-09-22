"""Live watch UX: heartbeat cadence, window match, non-blocking stdin (v1.9.4)."""

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
        self.assertIn("auto-fallback", _TROUBLESHOOT_NO_FRAMES)
        self.assertIn("capture OK", _CAPTURE_OK)
        self.assertIn("PLAY", _CAPTURE_OK)


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
        buf = io.StringIO()
        with mock.patch("cfb_coach.vision.pipeline.load_calib", return_value=calib):
            with mock.patch(
                "cfb_coach.vision.pipeline.MssRegionCapture"
            ) as mss_cls:
                mss_inst = mock.Mock()
                mss_inst.name = "mss"
                mss_cls.return_value = mss_inst
                with mock.patch("sys.stdout", buf):
                    cap = build_capture_from_args(args)
        mss_cls.assert_called_once()
        self.assertIs(cap, mss_cls.return_value)
        self.assertIn("capture=mss", buf.getvalue())

    def test_window_mode_uses_dxcam_mss_fallback(self) -> None:
        from cfb_coach.vision.pipeline import build_capture_from_args

        args = SimpleNamespace(
            image=None,
            video=None,
            window="Xbox",
            screen_region=None,
            device=None,
            live=False,
        )
        with mock.patch("cfb_coach.vision.pipeline.load_calib", return_value={}):
            with mock.patch(
                "cfb_coach.vision.pipeline.DxcamMssFallbackCapture"
            ) as fb_cls:
                fb_cls.return_value = mock.Mock(name="fallback", matched_title="Xbox App")
                cap = build_capture_from_args(args)
        fb_cls.assert_called_once()
        self.assertIs(cap, fb_cls.return_value)


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
        self.assertIn("v1.9.4", r.stdout)
        # Must not enter pipeline / waiting-for-frames path
        self.assertNotIn("waiting for frames", r.stdout)
        self.assertNotIn("Pipeline capture=", r.stdout)




class TestLivePlayAdjustHike(unittest.TestCase):
    """Unknown look still yields a non-empty playcaller PLAY; ADJUST/HIKE helpers."""

    def test_unknown_look_still_yields_nonempty_call(self) -> None:
        from cfb_coach.watch import compute_live_call, format_play_block

        look = DefenseLook(
            front="unknown",
            shell="unknown",
            pressure="unknown",
            confidence=0.17,
            source="capture",
            notes="low conf",
        )
        call, sit = compute_live_call(look, opponent="gavin", short_line="1&10 | ? | ?-HIGH")
        text = call.format()
        self.assertTrue(text.strip())
        self.assertIn("—", text)
        self.assertIn("|", text)
        # Must look like playcaller syntax, not generic tip
        self.assertNotIn("base look", text.lower())
        self.assertNotIn("wait for clearer", text.lower())
        self.assertEqual(sit.down, 1)
        self.assertEqual(sit.distance, 10)
        # coverage not injected at low conf
        self.assertTrue(sit.coverage_source in ("none", "") or sit.coverage_hint is None or float(look.confidence) < 0.4)

        block = format_play_block(text, short_line="1&10 | UNKNOWN | ?-HIGH | PRESSURE ?")
        self.assertIn("PLAY", block)
        self.assertIn(text.split("|")[0].strip().split("—")[0].strip().split()[0], block)

    def test_confident_look_injects_coverage(self) -> None:
        from cfb_coach.watch import build_live_situation

        look = DefenseLook(
            front="nickel", shell="cover3", pressure="none", confidence=0.85, source="hotkey"
        )
        sit = build_live_situation(look, short_line="2&7")
        self.assertEqual(sit.coverage_source, "live")
        self.assertIn("Cover 3", sit.coverage_hint or "")

    def test_adjust_and_audible(self) -> None:
        from cfb_coach.watch import adjust_lines_from_look, audible_warranted, format_play_block

        mild = DefenseLook("nickel", "two_high", "blitz_left", 0.8, "demo")
        ads = adjust_lines_from_look(mild)
        self.assertTrue(ads)
        self.assertTrue(any("slide L" in a or "left" in a.lower() for a in ads))
        self.assertFalse(audible_warranted(mild))

        heat = DefenseLook("nickel", "man", "all_out", 0.95, "demo")
        self.assertTrue(audible_warranted(heat))

        block = format_play_block(
            "Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag",
            adjusts=["pressure left — slide L, hot ready"],
            hike=True,
        )
        self.assertIn("PLAY", block)
        self.assertIn("ADJUST  pressure left", block)
        self.assertIn("HIKE — go", block)

    def test_cli_no_overlay_flag(self) -> None:
        from cfb_coach.cli import build_parser

        p = build_parser()
        args = p.parse_args(["watch", "--window", "Xbox", "--no-overlay"])
        self.assertTrue(args.no_overlay)
        args = p.parse_args(["watch", "--overlay"])
        self.assertEqual(args.overlay, "auto")



if __name__ == "__main__":
    unittest.main()
