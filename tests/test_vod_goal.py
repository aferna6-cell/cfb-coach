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

    def test_and_one_inside_the_ten_keeps_its_distance(self) -> None:
        lex = _lex()
        # 1st & 1 at the 6 is not goal-to-go: the parsed 1 is shorter than the 6.
        short = classify_frame(10, ["1ST1", "opp 6"], 0.6, lex, _ON)
        self.assertFalse(short.goal_to_go)
        self.assertEqual(short.distance, 1)
        # 2nd & 5 at the 3 is goal-to-go, and the 5 stays 5.
        near = classify_frame(12, ["2ND85", "opp 3"], 0.6, lex, _ON)
        self.assertTrue(near.goal_to_go)
        self.assertEqual(near.distance, 5)

    def test_and_one_away_from_the_goal_keeps_v096(self) -> None:
        """Box rows whose 'and 1' was rewritten as 100 - yardline."""
        lex = _lex()
        # RsOG3JCF7Wc:88.1 own 44, 3rd & 1. v0.96 distance 1, result -7.
        third = [
            classify_frame(88.1, ["3RD81", "OWN 44", "1st 5:00"], 0.6, lex, _ON),
            classify_frame(96.0, ["4TH88", "1st 4:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(third, video_id="RsOG3JCF7Wc", channel="SGU", game="madden27", extract=_ON)
        self.assertFalse(snaps[0].goal_to_go)
        self.assertEqual(snaps[0].distance, 1)
        self.assertEqual(snaps[0].yardline, 44)
        self.assertEqual(snaps[0].result, "-7")
        # The neighbour-motion fill on :104.1 reads this distance. It must stay 1, not 56.
        self.assertNotEqual(snaps[0].distance, 56)

        # RsOG3JCF7Wc:68.1 was +9. The next 'and 1' at own 44 had become distance 56.
        gain = [
            classify_frame(68.1, ["1ST810", "1st 8:00"], 0.6, lex, _ON),
            classify_frame(76.0, ["2ND81", "OWN 44", "1st 7:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(gain, video_id="RsOG3JCF7Wc", channel="SGU", game="madden27", extract=_ON)
        self.assertFalse(snaps[1].goal_to_go)
        self.assertEqual(snaps[1].distance, 1)
        self.assertEqual(snaps[0].result, "+9")

        # RsOG3JCF7Wc:1045.0 was +3. The next 'and 1' at opp 30 had become distance 30.
        plus3 = [
            classify_frame(1045.0, ["1ST84", "1st 3:00"], 0.6, lex, _ON),
            classify_frame(1052.0, ["2ND81", "opp 30", "1st 2:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(plus3, video_id="RsOG3JCF7Wc", channel="SGU", game="madden27", extract=_ON)
        self.assertFalse(snaps[1].goal_to_go)
        self.assertEqual(snaps[1].distance, 1)
        self.assertEqual(snaps[1].yardline, 70)
        self.assertEqual(snaps[0].result, "+3")

        # 4_-atoyAQzA:9772.0 yard line 74, 1st & 1. v0.96 distance 1, result +0.
        mid = [
            classify_frame(9772.0, ["1ST81", "opp 26", "1st 6:00"], 0.6, lex, _ON),
            classify_frame(9780.0, ["2ND81", "1st 5:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(mid, video_id="4_-atoyAQzA", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].yardline, 74)
        self.assertFalse(snaps[0].goal_to_go)
        self.assertEqual(snaps[0].distance, 1)
        self.assertEqual(snaps[0].result, "+0")

        # zs19MRWF9Bc:880.9 yard line 56 is not goal-to-go.
        far = classify_frame(880.9, ["2ND81", "opp 44", "1st 4:00"], 0.6, lex, _ON)
        self.assertEqual(far.yardline, 56)
        self.assertFalse(far.goal_to_go)
        self.assertEqual(far.distance, 1)

    def test_no_spot_keeps_distance_empty_and_still_resolves(self) -> None:
        """Missing spots stay None. Yards still use the painted GOAL distance."""
        lex = _lex()
        frames = [
            classify_frame(10, ["1STGOAL", "1st 2:00"], 0.6, lex, _ON),
            classify_frame(20, ["2NDGOAL", "1st 1:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(frames, video_id="g", channel="SGU", game="madden27", extract=_ON)
        self.assertIsNone(snaps[0].distance)
        self.assertEqual(snaps[0].baseline_distance, 1)
        self.assertTrue(snaps[0].goal_to_go)
        self.assertEqual(snaps[0].yards, 0)
        self.assertEqual(snaps[0].result, "+0")
        self.assertEqual(snaps[0].result_flag, "")

    def test_goal_then_a_normal_down_uses_the_painted_distance(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(10, ["1STGOAL", "1st 2:00"], 0.6, lex, _ON),
            classify_frame(20, ["2ND86", "1st 1:40"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(frames, video_id="g", channel="SGU", game="madden27", extract=_ON)
        self.assertIsNone(snaps[0].distance)
        self.assertEqual(snaps[0].yards, -5)
        self.assertEqual(snaps[0].result, "-5")

    def test_goal_touchdown_survives_a_missing_distance(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(10, ["1STGOAL", "TOUCHDOWN", "1st 2:00"], 0.6, lex, _ON),
            classify_frame(20, ["1ST810", "1st 1:20"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(frames, video_id="g", channel="SGU", game="madden27", extract=_ON)
        self.assertIsNone(snaps[0].distance)
        self.assertEqual(snaps[0].result, "td")
        self.assertEqual(snaps[0].result_kind, "td")

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
