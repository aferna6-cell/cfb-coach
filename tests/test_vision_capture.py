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


if __name__ == "__main__":
    unittest.main()
