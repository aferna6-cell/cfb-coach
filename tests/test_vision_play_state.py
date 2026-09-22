"""Play-state transitions (Milestone 2)."""

from __future__ import annotations

import unittest

from cfb_coach.vision.play_state import PlayStateTracker, canonicalize, to_compat


class TestPlayState(unittest.TestCase):
    def test_between_to_presnap_to_active_to_ending(self) -> None:
        t = PlayStateTracker()
        t.force("BETWEEN_PLAYS", 0.0)
        self.assertEqual(t.update(now=1.0, hud_visible=True, motion=0.01), "PRE_SNAP")
        self.assertEqual(t.update(now=2.0, motion=0.6), "PLAY_ACTIVE")
        self.assertIsNotNone(t.snap_time)
        self.assertEqual(t.update(now=3.0, motion=0.02), "PLAY_ENDING")
        self.assertIsNotNone(t.play_end_time)
        self.assertEqual(t.update(now=4.0, motion=0.01), "POST_PLAY")
        self.assertEqual(t.update(now=5.0, motion=0.01), "BETWEEN_PLAYS")

    def test_menu(self) -> None:
        t = PlayStateTracker()
        t.force("PRE_SNAP", 0.0)
        self.assertEqual(t.update(now=1.0, menu_like=True), "MENU")

    def test_invalid_forced_to_unknown(self) -> None:
        t = PlayStateTracker()
        self.assertEqual(t.force("NOPE", 0.0), "UNKNOWN")

    def test_compat_aliases(self) -> None:
        self.assertEqual(canonicalize("OTHER"), "UNKNOWN")
        self.assertEqual(canonicalize("PLAY_ENDED"), "PLAY_ENDING")
        self.assertEqual(to_compat("PLAY_ENDING"), "PLAY_ENDED")
        self.assertEqual(to_compat("UNKNOWN"), "OTHER")

    def test_snap_signal(self) -> None:
        t = PlayStateTracker()
        t.force("PRE_SNAP", 0.0)
        self.assertEqual(t.update(now=1.0, snap_signal=True), "PLAY_ACTIVE")

    def test_timestamps_on_cycle(self) -> None:
        t = PlayStateTracker()
        t.force("PRE_SNAP", 1.0)
        self.assertEqual(t.pre_snap_start, 1.0)
        t.update(now=2.0, snap_signal=True)
        self.assertEqual(t.snap_time, 2.0)
        t.update(now=3.0, end_signal=True)
        self.assertEqual(t.play_end_time, 3.0)


if __name__ == "__main__":
    unittest.main()
