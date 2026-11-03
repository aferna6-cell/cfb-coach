"""Result kind. A +0 is not an incomplete, and a moved ball is not +0."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.segment import FrameObs, _extra_result_markers, build_snaps, classify_frame

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "results.json"
_ON = ExtractFlags(read_result=True)


def _live(t: float, down: int, dist: int, yard: int | None = None, texts: list[str] | None = None) -> FrameObs:
    return FrameObs(
        t=t,
        kind="live",
        down=down,
        distance=dist,
        yardline=yard,
        texts=list(texts or []),
        green=0.6,
        quarter=1,
        clock=300,
    )


class ResultMarkerFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 6)
        for case in cases:
            frame = FrameObs(t=0, kind="live", texts=case["texts"])
            self.assertEqual(sorted(_extra_result_markers([frame])), sorted(case["markers"]), case["id"])


class ResultSnapTests(unittest.TestCase):
    def test_same_spot_is_a_real_zero(self) -> None:
        frames = [_live(10, 1, 10, 25), _live(20, 2, 10, 25)]
        snaps = build_snaps(frames, video_id="r", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].result, "+0")
        self.assertEqual(snaps[0].result_kind, "no_gain")
        self.assertEqual(snaps[0].result_flag, "")

    def test_moved_ball_is_not_plus_zero(self) -> None:
        frames = [_live(10, 1, 10, 25), _live(20, 2, 10, 40)]
        snaps = build_snaps(frames, video_id="r", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].result, "")
        self.assertEqual(snaps[0].result_kind, "unknown")
        self.assertEqual(snaps[0].result_flag, "result_spot_conflict")
        self.assertIn("result_spot_conflict", snaps[0].notes)

    def test_missing_spot_still_uses_the_distance(self) -> None:
        frames = [_live(10, 1, 10), _live(20, 2, 10)]
        snaps = build_snaps(frames, video_id="r", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].result, "+0")
        self.assertEqual(snaps[0].result_flag, "")

    def test_banners(self) -> None:
        def first(frames: list[FrameObs]):
            return build_snaps(frames, video_id="r", channel="SGU", game="madden27", extract=_ON)[0]

        self.assertEqual(
            first([_live(10, 1, 10, texts=["PASS INCOMPLETE"]), _live(20, 2, 10)]).result_kind,
            "incomplete",
        )
        self.assertEqual(first([_live(30, 1, 8, texts=["TOUCH DOWN"]), _live(40, 1, 10)]).result, "td")
        self.assertEqual(first([_live(50, 2, 7, texts=["TURNOVER"]), _live(60, 1, 10)]).result, "turnover")

    def test_new_set_of_downs_is_a_conversion(self) -> None:
        """3rd down into 1st & 10 takes the convert branch.

        That branch reads the next snap's distance. It used to raise
        UnboundLocalError because dist1 was only set when the down advanced
        by one.
        """
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        specs = [
            (10, ["3RD84", "1st 5:00"]),
            (20, ["1ST810", "1st 4:40"]),
        ]
        flagged = ExtractFlags(yardline_glyphs=True, goal_to_go=True, read_result=True, dd_norm=True)
        for extract in (ExtractFlags(), flagged):
            frames = [classify_frame(t, texts, 0.6, lex, extract) for t, texts in specs]
            snaps = build_snaps(frames, video_id="r", channel="SGU", game="madden27", extract=extract)
            self.assertEqual(frames[0].down, 3)
            self.assertEqual(frames[0].distance, 4)
            self.assertEqual(frames[1].down, 1)
            self.assertEqual(frames[1].distance, 10)
            self.assertEqual(snaps[0].result_kind, "convert")
            self.assertEqual(snaps[0].result, "convert")

        goal = [
            classify_frame(10, ["2NDGOAL", "1st 5:00"], 0.6, lex, flagged),
            classify_frame(20, ["1ST810", "1st 4:40"], 0.6, lex, flagged),
        ]
        snaps = build_snaps(goal, video_id="r", channel="SGU", game="madden27", extract=flagged)
        self.assertEqual(snaps[0].result_kind, "convert")
        self.assertEqual(snaps[0].result, "convert")

    def test_flag_off_keeps_distance_plus_zero(self) -> None:
        frames = [_live(10, 1, 10, 25), _live(20, 2, 10, 40)]
        snaps = build_snaps(frames, video_id="r", channel="SGU", game="madden27")
        self.assertEqual(snaps[0].result, "+0")
        self.assertEqual(snaps[0].result_flag, "")


if __name__ == "__main__":
    unittest.main()
