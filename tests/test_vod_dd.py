"""Down-and-distance normalizer, including 360p misreads of 3rd."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.parse_hud import norm_dd
from vod_pilot.segment import classify_frame
from vod_pilot.tokens import Tok

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "dd.json"
_ON = ExtractFlags(dd_norm=True)


class NormDdFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 30)
        for case in cases:
            self.assertEqual(norm_dd(case["text"], bare=case["bare"]), case["expect"], case["text"])


class NormDdSlotTests(unittest.TestCase):
    def test_bare_digits_need_the_madden_slot(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        outside = classify_frame(10, ["32", "1st 5:00"], 0.6, lex, _ON)
        self.assertIsNone(outside.down)
        inside = classify_frame(10, [Tok("32", x=0.87, y=0.95), "1st 5:00"], 0.6, lex, _ON)
        self.assertEqual((inside.down, inside.distance), (3, 2))

    def test_ordinal_misread_does_not_need_a_slot(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        frame = classify_frame(10, ["2ND64", "1st 5:00"], 0.6, lex, _ON)
        self.assertEqual((frame.down, frame.distance), (2, 4))

    def test_cfb_play_clock_is_not_a_down(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        frame = classify_frame(
            10,
            [Tok("128", x=0.50, y=0.90), "1st 5:00"],
            0.6,
            lex,
            _ON,
            game="cfb27",
        )
        self.assertIsNone(frame.down)

    def test_flag_off_leaves_2nd64_unread(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        frame = classify_frame(10, ["2ND64"], 0.6, lex)
        self.assertIsNone(frame.down)


if __name__ == "__main__":
    unittest.main()
