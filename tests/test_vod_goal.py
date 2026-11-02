"""Goal-to-go stays goal. Distance 1 is not the stand-in."""

from __future__ import annotations

import unittest

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.parse_hud import goal_down
from vod_pilot.segment import build_snaps, classify_frame

_ON = ExtractFlags(goal_to_go=True)


def _lex() -> Lexicon:
    return Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])


class GoalTokenTests(unittest.TestCase):
    def test_goal_words(self) -> None:
        self.assertEqual(goal_down("1STGOAL"), 1)
        self.assertEqual(goal_down("2ND&GOAL"), 2)
        self.assertEqual(goal_down("1STSGOAL"), 1)
        self.assertEqual(goal_down("3rd & Goal"), 3)
        self.assertIsNone(goal_down("1ST10"))
        self.assertIsNone(goal_down("1st 2:00"))


class GoalSnapTests(unittest.TestCase):
    def test_spots_replace_a_fake_plus_zero(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(10, ["1STGOAL", "opp 8", "1st 2:00"], 0.6, lex, _ON),
            classify_frame(20, ["2NDGOAL", "opp 4", "1st 1:40"], 0.6, lex, _ON),
        ]
        self.assertTrue(frames[0].goal_to_go)
        self.assertEqual(frames[0].distance, 8)
        self.assertNotEqual(frames[0].distance, 1)
        snaps = build_snaps(frames, video_id="g", channel="SGU", game="madden27", extract=_ON)
        self.assertTrue(snaps[0].goal_to_go)
        self.assertIn("Goal", snaps[0].situation_raw)
        self.assertNotIn("& 1", snaps[0].situation_raw)
        self.assertEqual(snaps[0].result, "+4")
        self.assertEqual(snaps[0].yards, 4)

    def test_and_one_is_goal_to_go(self) -> None:
        lex = _lex()
        frame = classify_frame(10, ["1ST1", "opp 6"], 0.6, lex, _ON)
        self.assertTrue(frame.goal_to_go)
        self.assertEqual(frame.distance, 6)

    def test_no_spot_is_not_plus_zero(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(10, ["1STGOAL", "1st 2:00"], 0.6, lex, _ON),
            classify_frame(20, ["2NDGOAL", "1st 1:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(frames, video_id="g", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].result, "")
        self.assertEqual(snaps[0].result_flag, "goal_no_spot")
        self.assertIsNone(snaps[0].distance)

    def test_flag_off_still_maps_goal_to_one(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(10, ["1STGOAL", "1st 2:00"], 0.6, lex),
            classify_frame(20, ["2NDGOAL", "1st 1:40"], 0.6, lex),
        ]
        self.assertEqual(frames[0].distance, 1)
        self.assertFalse(frames[0].goal_to_go)
        snaps = build_snaps(frames, video_id="g", channel="SGU", game="madden27")
        self.assertEqual(snaps[0].result, "+0")


if __name__ == "__main__":
    unittest.main()
