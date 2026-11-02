"""Ball-spot glyphs. The arrow is not part of the yard line."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.segment import build_snaps, classify_frame
from vod_pilot.tokens import Tok
from vod_pilot.vodx_field import spot_token

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "spots.json"
_ON = ExtractFlags(yardline_glyphs=True)


def _lex() -> Lexicon:
    return Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])


class SpotFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 12)
        for case in cases:
            got = spot_token(case["text"], case["conf"])
            if case["expect"] is None:
                self.assertIsNone(got, case["text"])
            else:
                self.assertEqual(list(got), case["expect"], case["text"])


class YardlineSnapTests(unittest.TestCase):
    def test_arrow_becomes_a_coach_yardline(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(50, ["1ST810", "1st 5:10", "￥46"], 0.6, lex, _ON),
            classify_frame(58, ["1ST810", Tok("a35", conf=0.57)], 0.6, lex, _ON),
        ]
        # Two different spots in one snap disagree, so the row does not pick one.
        snaps = build_snaps(frames, video_id="y", channel="SGU", game="madden27", extract=_ON)
        self.assertIsNone(snaps[0].yardline)
        one = classify_frame(50, ["1ST810", "1st 5:10", Tok("^35", conf=0.57)], 0.6, lex, _ON)
        self.assertEqual(one.yardline, 35)
        self.assertEqual(one.yard_phrase, "own 35")
        snaps = build_snaps(
            [one, classify_frame(58, ["1ST810", Tok("^35", conf=0.61)], 0.6, lex, _ON)],
            video_id="y",
            channel="SGU",
            game="madden27",
            extract=_ON,
        )
        self.assertEqual(snaps[0].yardline, 35)
        self.assertIn("own 35", snaps[0].situation_raw)
        self.assertGreaterEqual(snaps[0].confidence["yardline"], 0.5)

    def test_cfb_layout_does_not_invent_a_spot(self) -> None:
        lex = _lex()
        frame = classify_frame(
            50, ["2ND62", "127", "a35"], 0.6, lex, _ON, game="cfb27"
        )
        self.assertIsNone(frame.yardline)

    def test_low_confidence_glyph_is_not_a_spot(self) -> None:
        lex = _lex()
        frame = classify_frame(50, ["1ST810", Tok("a35", conf=0.49)], 0.6, lex, _ON)
        self.assertIsNone(frame.yardline)

    def test_flag_off_leaves_a_bare_glyph_alone(self) -> None:
        lex = _lex()
        frame = classify_frame(50, ["1ST810", "￥46"], 0.6, lex)
        self.assertIsNone(frame.yardline)


if __name__ == "__main__":
    unittest.main()
