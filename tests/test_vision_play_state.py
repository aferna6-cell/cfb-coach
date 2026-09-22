"""Play-state transitions."""

from __future__ import annotations

import unittest

from cfb_coach.vision.play_state import PlayStateTracker


class TestPlayState(unittest.TestCase):
    def test_between_to_presnap_to_active_to_ended(self) -> None:
        t = PlayStateTracker()
        t.force("BETWEEN_PLAYS", 0.0)
        self.assertEqual(t.update(now=1.0, hud_visible=True, motion=0.01), "PRE_SNAP")
        self.assertEqual(t.update(now=2.0, motion=0.6), "PLAY_ACTIVE")
        self.assertEqual(t.update(now=3.0, motion=0.02), "PLAY_ENDED")
        self.assertEqual(t.update(now=4.0, motion=0.01), "BETWEEN_PLAYS")

    def test_menu(self) -> None:
        t = PlayStateTracker()
        t.force("PRE_SNAP", 0.0)
        self.assertEqual(t.update(now=1.0, menu_like=True), "MENU")

    def test_invalid_forced_to_other(self) -> None:
        t = PlayStateTracker()
        self.assertEqual(t.force("NOPE", 0.0), "OTHER")


if __name__ == "__main__":
    unittest.main()
