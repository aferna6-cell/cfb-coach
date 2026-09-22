"""CaptureBackend stubs without dxcam; graceful ImportError; offline image."""

from __future__ import annotations

import unittest
from pathlib import Path

from cfb_coach.vision.capture import (
    Frame,
    ImageFileCapture,
    StubCapture,
    frame_to_bytes,
)
from cfb_coach.vision.look import naive_roi_heuristic


FIX_PNG = Path(__file__).parent / "fixtures" / "synthetic_field.png"


class TestCapture(unittest.TestCase):
    def test_stub(self) -> None:
        c = StubCapture()
        self.assertIsNone(c.grab())
        c.close()

    def test_image_file(self) -> None:
        self.assertTrue(FIX_PNG.is_file())
        c = ImageFileCapture(FIX_PNG)
        frame = c.grab()
        self.assertIsNotNone(frame)
        assert isinstance(frame, Frame)
        self.assertTrue(frame.available)
        self.assertTrue(frame.png_bytes)
        # Legacy heuristic still works on bytes
        look = naive_roi_heuristic(frame.png_bytes or b"")
        self.assertEqual(look.source, "image")
        c.close()

    def test_dxcam_import_error_message(self) -> None:
        from cfb_coach.vision.capture import DxcamWindowCapture

        cap = DxcamWindowCapture(window_substring="Xbox")
        # On Linux without dxcam, _ensure raises ImportError
        try:
            import dxcam  # noqa: F401

            has = True
        except ImportError:
            has = False
        if not has:
            with self.assertRaises(ImportError) as ctx:
                cap.grab()
            self.assertIn("dxcam", str(ctx.exception).lower())

    def test_package_import_without_vision_extras(self) -> None:
        import cfb_coach
        from cfb_coach import vision

        self.assertTrue(hasattr(vision, "DefenseLook"))
        self.assertTrue(hasattr(vision, "GameObservation"))
        self.assertTrue(hasattr(cfb_coach, "__version__"))



class TestDxcamMssFallback(unittest.TestCase):
    def test_ltrb_to_mss_region(self) -> None:
        from cfb_coach.vision.capture import ltrb_to_mss_region

        self.assertEqual(
            ltrb_to_mss_region((10, 20, 110, 80)),
            {"left": 10, "top": 20, "width": 100, "height": 60},
        )

    def test_fallback_after_timeout(self) -> None:
        from cfb_coach.vision.capture import (
            DXCAM_NO_FRAMES_FALLBACK_MSG,
            DxcamMssFallbackCapture,
            Frame,
        )
        from unittest import mock

        fake_frame = Frame(bgr=object(), width=4, height=4, source="mss")

        with mock.patch(
            "cfb_coach.vision.capture.DxcamWindowCapture"
        ) as dx_cls:
            dx = mock.Mock()
            dx.matched_title = "Xbox App"
            dx._resolved_region = (0, 0, 100, 50)
            dx.grab.return_value = None
            dx_cls.return_value = dx

            with mock.patch(
                "cfb_coach.vision.capture.MssRegionCapture"
            ) as mss_cls:
                mss = mock.Mock()
                mss.grab.return_value = fake_frame
                mss_cls.return_value = mss

                cap = DxcamMssFallbackCapture(
                    window_substring="Xbox",
                    fallback_after_s=0.05,
                )
                self.assertEqual(cap.name, "dxcam")
                self.assertEqual(cap.matched_title, "Xbox App")

                # Before timeout: still None / dxcam
                self.assertIsNone(cap.grab())
                self.assertEqual(cap.name, "dxcam")

                import time

                time.sleep(0.06)
                buf = __import__("io").StringIO()
                with mock.patch("sys.stdout", buf):
                    frame = cap.grab()
                self.assertIs(frame, fake_frame)
                self.assertEqual(cap.name, "mss")
                self.assertIn(DXCAM_NO_FRAMES_FALLBACK_MSG, buf.getvalue())
                mss_cls.assert_called_once()
                # Message prints only once
                with mock.patch("sys.stdout", buf):
                    cap.grab()
                self.assertEqual(buf.getvalue().count(DXCAM_NO_FRAMES_FALLBACK_MSG), 1)
                cap.close()

    def test_resolve_prefers_calib(self) -> None:
        from cfb_coach.vision.capture import resolve_mss_region_for_window

        calib = {"left": 5, "top": 6, "width": 7, "height": 8}
        with __import__("unittest").mock.patch(
            "cfb_coach.vision.capture.find_window_region",
            return_value=(0, 0, 100, 100),
        ):
            reg = resolve_mss_region_for_window("Xbox", calib_region=calib)
        self.assertEqual(reg, calib)


if __name__ == "__main__":
    unittest.main()
