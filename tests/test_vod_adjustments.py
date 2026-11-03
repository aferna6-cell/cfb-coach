"""Adjustment reads. A play name that merely contains a route is not one."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.adjustments import read_adjustments
from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.segment import build_snaps, classify_frame

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "adjustments.json"
_ON = ExtractFlags(read_adjustments=True)


class AdjustmentFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 5)
        for case in cases:
            found, conf = read_adjustments(case["texts"])
            labels = [item.label() for item in found]
            self.assertEqual(labels, case["expect"], case["id"])
            if case["expect"]:
                self.assertGreaterEqual(conf, case["min_conf"], case["id"])
            else:
                self.assertEqual(conf, 0.0, case["id"])


class AdjustmentSnapTests(unittest.TestCase):
    def test_call_screen_adjustments_land_on_the_snap(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["Inside Zone"], coverages=["Cover 3"])
        specs = [
            (36, ["HOT ROUTE SLANT", "SLIDE LEFT", "INSIDE ZONE"], 0.12),
            (48, ["1ST810", "1st 4:40"], 0.6),
            (56, ["1ST810"], 0.62),
        ]
        frames = [classify_frame(t, texts, green, lex, _ON) for t, texts, green in specs]
        snaps = build_snaps(frames, video_id="a", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(
            snaps[0].adjustments,
            ["hot_route: HOT ROUTE SLANT", "offensive: SLIDE LEFT"],
        )
        self.assertGreaterEqual(snaps[0].confidence["adjustments"], 0.8)

    def test_flag_off_leaves_adjustments_empty(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["Inside Zone"], coverages=["Cover 3"])
        frame = classify_frame(30, ["HOT ROUTE SLANT", "1ST810"], 0.2, lex)
        self.assertEqual(frame.adjustments, [])


if __name__ == "__main__":
    unittest.main()
