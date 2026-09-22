"""Regression tests for the live-capture RCA (docs/vision-capture-rca.md).

Each test pins one root cause so it cannot silently come back:
  - dxcam "Invalid Region" from snapped windows (-7px invisible border)
  - capture exceptions swallowed by ThreadedCapture → endless "waiting for frames"
  - cloaked / minimized / terminal windows matched as "Xbox"
  - image-y vs field-y inversion in the classifiers
  - WGC backend contract (copy frame, stale → None)
"""

from __future__ import annotations

import sys
import time
import types
import unittest
from unittest import mock

from cfb_coach.vision.capture import Frame
from cfb_coach.vision.winenum import (
    WindowInfo,
    clamp_ltrb,
    enumerate_windows,
    rank_matches,
    to_output_local,
)

try:
    import cv2  # noqa: F401
    import numpy as np

    HAS_CV = True
except ImportError:  # pragma: no cover — vision extras optional
    HAS_CV = False


def _win(title: str, client=(0, 0, 100, 100), **kw) -> WindowInfo:
    return WindowInfo(hwnd=kw.pop("hwnd", 1), title=title, class_name=kw.pop("cls", "X"),
                      rect=client, client=client, **kw)


class TestRegionMath(unittest.TestCase):
    def test_snapped_left_window_clamped_for_dxcam(self) -> None:
        # Win11 snap-left: invisible border puts left at -7 → dxcam rejected it.
        self.assertEqual(
            to_output_local((-7, 0, 967, 1047), (0, 0), (1920, 1080)), (0, 0, 967, 1047)
        )

    def test_snapped_right_window_clamped_for_dxcam(self) -> None:
        self.assertEqual(
            to_output_local((953, 0, 1927, 1047), (0, 0), (1920, 1080)),
            (953, 0, 1920, 1047),
        )

    def test_window_on_second_monitor_is_none(self) -> None:
        self.assertIsNone(to_output_local((1920, 0, 3840, 1080), (0, 0), (1920, 1080)))

    def test_clamp_empty(self) -> None:
        self.assertIsNone(clamp_ltrb((10, 10, 10, 20), (0, 0, 100, 100)))


class TestWindowRanking(unittest.TestCase):
    def test_cloaked_minimized_terminal_never_match(self) -> None:
        wins = [
            _win("Xbox", cloaked=True),  # suspended UWP Xbox app
            _win("Xbox Game Bar", minimized=True),
            _win("cfb-coach watch --window Xbox", is_terminal=True),
        ]
        self.assertEqual(rank_matches(wins, "Xbox"), [])

    def test_exact_then_prefix_then_contains_then_area(self) -> None:
        small_contains = _win("Play on XBOX PC app - Edge", client=(0, 0, 10, 10), hwnd=1)
        big_contains = _win("XBOX PC app | Remote Play - Microsoft Edge", client=(0, 0, 900, 500), hwnd=2)
        exact = _win("XBOX", client=(0, 0, 50, 50), hwnd=3)
        got = rank_matches([small_contains, big_contains, exact], "xbox")
        self.assertEqual([w.hwnd for w in got], [3, 2, 1])

    def test_enumerate_is_none_off_windows(self) -> None:
        if sys.platform != "win32":
            self.assertIsNone(enumerate_windows())


class _FakeCam:
    def __init__(self) -> None:
        self.width, self.height = 1920, 1080
        self.grab_regions: list = []

    def grab(self, region=None):
        self.grab_regions.append(region)
        return np.zeros((10, 10, 3), np.uint8) if HAS_CV else None


class TestDxcamRegion(unittest.TestCase):
    def _fake_dxcam(self, cam: _FakeCam) -> types.ModuleType:
        mod = types.ModuleType("dxcam")
        mod.create = mock.Mock(return_value=cam)  # type: ignore[attr-defined]
        return mod

    def test_create_without_region_then_grab_clamped(self) -> None:
        from cfb_coach.vision.capture import DxcamWindowCapture

        cam = _FakeCam()
        fake = self._fake_dxcam(cam)
        with mock.patch.dict(sys.modules, {"dxcam": fake}):
            cap = DxcamWindowCapture(window_substring=None, region=(-7, 0, 967, 1047))
            cap.grab()
        kwargs = fake.create.call_args.kwargs
        self.assertNotIn("region", kwargs)  # full-output camera; never "Invalid Region"
        self.assertEqual(cam.grab_regions[-1], (0, 0, 967, 1047))

    def test_off_primary_raises_clear_error(self) -> None:
        from cfb_coach.vision.capture import DxcamWindowCapture

        fake = self._fake_dxcam(_FakeCam())
        with mock.patch.dict(sys.modules, {"dxcam": fake}):
            cap = DxcamWindowCapture(window_substring=None, region=(2000, 0, 3000, 500))
            with self.assertRaises(RuntimeError) as ctx:
                cap.grab()
        self.assertIn("primary monitor", str(ctx.exception))


class TestThreadedCaptureErrors(unittest.TestCase):
    def test_exception_surfaces_as_last_error(self) -> None:
        from cfb_coach.vision.threaded_capture import ThreadedCapture

        class Boom:
            name = "dxcam"

            def grab(self):
                raise ValueError("Invalid Region: Region should be in 1920x1080")

            def close(self):
                return None

        tc = ThreadedCapture(Boom(), target_fps=100.0)  # type: ignore[arg-type]
        tc.start()
        time.sleep(0.05)
        tc.close()
        self.assertIn("Invalid Region", tc.last_error or "")
        self.assertGreaterEqual(tc.debug_stats()["errors"], 1)

    def test_heartbeat_shows_last_error(self) -> None:
        from cfb_coach.watch import _waiting_frames_msg

        msg = _waiting_frames_msg("threaded:dxcam", "ValueError: Invalid Region")
        self.assertIn("waiting for frames", msg)
        self.assertIn("Invalid Region", msg)
        self.assertNotIn("last error", _waiting_frames_msg("threaded:dxcam"))


class TestWgcBackend(unittest.TestCase):
    def _fake_module(self):
        holder: dict = {}

        class FakeCtl:
            def stop(self):
                holder["stopped"] = True

        class FakeWindowsCapture:
            def __init__(self, **kw):
                holder["kwargs"] = kw
                holder["cap"] = self
                self.frame_handler = None
                self.closed_handler = None

            def start_free_threaded(self):
                return FakeCtl()

        mod = types.ModuleType("windows_capture")
        mod.WindowsCapture = FakeWindowsCapture  # type: ignore[attr-defined]
        return mod, holder

    @unittest.skipUnless(HAS_CV, "numpy needed")
    def test_copies_frame_and_stale_is_none(self) -> None:
        from cfb_coach.vision.wgc_capture import WgcWindowCapture

        mod, holder = self._fake_module()
        with mock.patch.dict(sys.modules, {"windows_capture": mod}):
            cap = WgcWindowCapture(hwnd=42, title="XBOX")
            self.assertIsNone(cap.grab())  # starts session, nothing yet
            self.assertEqual(holder["kwargs"]["window_hwnd"], 42)
            self.assertFalse(holder["kwargs"]["draw_border"])
            buf = np.full((4, 6, 4), 7, np.uint8)
            holder["cap"].frame_handler(types.SimpleNamespace(frame_buffer=buf, width=6, height=4), None)
            buf[:] = 0  # native buffer reused → our frame must be a copy
            f = cap.grab()
            self.assertIsNotNone(f)
            self.assertEqual(f.bgr.shape, (4, 6, 3))
            self.assertEqual(int(f.bgr.max()), 7)
            self.assertIsNone(cap.grab())  # no new frame → None (frozen stream)
            cap.close()
        self.assertTrue(holder.get("stopped"))

    def test_build_capture_wgc_uses_hwnd(self) -> None:
        from cfb_coach.vision import pipeline

        info = _win("XBOX", hwnd=77)
        args = types.SimpleNamespace(
            image=None, video=None, window="XBOX", screen_region=None, device=None,
            capture="wgc",
        )
        with mock.patch.object(pipeline, "load_calib", return_value={}), mock.patch(
            "cfb_coach.vision.capture.find_window", return_value=info
        ), mock.patch("sys.stdout"):
            cap = pipeline.build_capture_from_args(args)
        self.assertEqual(cap.name, "wgc")
        self.assertEqual(cap.hwnd, 77)


@unittest.skipUnless(HAS_CV, "opencv needed")
class TestFieldOrientation(unittest.TestCase):
    def _frame(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        img[:] = (40, 140, 50)
        off = [(560 + i * 32, 470) for i in range(5)] + [(640, 510), (640, 560), (200, 462), (1080, 462)]
        de = [(575 + i * 40, 420) for i in range(4)] + [(520, 380), (760, 380), (460, 230), (820, 230)]
        for x, y in off:
            cv2.rectangle(img, (x - 7, y - 16), (x + 7, y + 16), (30, 30, 170), -1)
        for x, y in de:
            cv2.rectangle(img, (x - 6, y - 12), (x + 6, y + 12), (245, 245, 245), -1)
        return img

    def test_image_to_field_flips_y(self) -> None:
        from cfb_coach.vision.pipeline import image_to_field

        (x, y), = image_to_field([(0.25, 0.9)])
        self.assertEqual(x, 0.25)
        self.assertAlmostEqual(y, 0.1)

    def test_two_deep_safeties_at_top_of_frame_read_two_high(self) -> None:
        from cfb_coach.vision.pipeline import VisionPipeline

        rois = {"field": {"x": 0.05, "y": 0.15, "w": 0.90, "h": 0.70}}
        img = self._frame()
        obs = VisionPipeline(calib={"rois": rois}).process_frame(
            Frame(bgr=img, width=img.shape[1], height=img.shape[0])
        )
        self.assertEqual(obs.shell, "two_high")

    def test_no_grass_is_the_0_17_signature(self) -> None:
        # conf≈0.17 seen live == mean(0.15, 0.20, 0.15) == zero player blobs.
        from cfb_coach.vision.pipeline import VisionPipeline

        img = np.zeros((720, 1280, 3), np.uint8)
        look = VisionPipeline(calib={}).process_frame(
            Frame(bgr=img, width=1280, height=720)
        ).to_defense_look()
        self.assertAlmostEqual(look.confidence, 0.1667, places=3)
        self.assertEqual(look.extras.get("n_centroids"), 0)


@unittest.skipUnless(HAS_CV, "opencv needed")
class TestProbeVerdicts(unittest.TestCase):
    def test_verdicts(self) -> None:
        from cfb_coach.vision.probe import frame_stats, verdict

        black = np.zeros((90, 160, 3), np.uint8)
        grass = np.zeros((90, 160, 3), np.uint8)
        grass[:] = (40, 140, 50)
        grass2 = grass.copy()
        grass2[:45] = (60, 160, 70)
        menu = np.full((90, 160, 3), (120, 60, 30), np.uint8)
        menu2 = menu.copy()
        menu2[:30] = (200, 200, 200)

        self.assertTrue(verdict(0, None, "ValueError: Invalid Region").startswith("NO FRAMES"))
        self.assertTrue(verdict(5, frame_stats(black), None).startswith("BLACK"))
        self.assertTrue(verdict(5, frame_stats(grass, grass), None).startswith("FROZEN"))
        self.assertTrue(verdict(5, frame_stats(menu2, menu), None).startswith("NO FIELD"))
        self.assertTrue(verdict(5, frame_stats(grass2, grass), None).startswith("OK"))


if __name__ == "__main__":
    unittest.main()
