"""Circled OCR digits must not raise, and must not invent a yard line."""

from __future__ import annotations

import unittest

from vod_pilot.digits import normalize_digits, safe_int
from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.parse_hud import norm_dd, parse_clock, parse_down_distance_token
from vod_pilot.scorebug import parse_scorebug
from vod_pilot.segment import classify_frame
from vod_pilot.vodx_field import spot_token
from vodx_pr88 import as_int, rows_from_frames

_ON = ExtractFlags(yardline_glyphs=True, dd_norm=True, read_score=True, goal_to_go=True, flag_menu=True)


def _lex() -> Lexicon:
    return Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])


class CircledDigitTests(unittest.TestCase):
    def test_normalize_examples(self) -> None:
        self.assertEqual(normalize_digits("③"), "3")
        self.assertEqual(normalize_digits("1③"), "13")
        self.assertEqual(normalize_digits("⑩"), "10")
        self.assertEqual(normalize_digits("⑳"), "20")
        self.assertEqual(normalize_digits("⓵"), "1")
        self.assertEqual(normalize_digits("❶"), "1")
        self.assertEqual(normalize_digits("➀"), "1")
        self.assertEqual(normalize_digits("❿"), "10")
        self.assertEqual(safe_int("③"), 3)
        self.assertEqual(safe_int("1③"), 13)
        self.assertEqual(safe_int("⑩"), 10)
        self.assertIsNone(safe_int("garbage"))
        self.assertIsNone(safe_int("not-a-yard"))
        self.assertEqual(as_int("③"), 3)
        self.assertEqual(as_int("1③"), 13)
        self.assertEqual(as_int("⑩"), 10)
        self.assertIsNone(as_int("garbage"))

    def test_parsers_do_not_raise(self) -> None:
        lex = _lex()
        for text in ("③", "1③", "⑩", "garbage", "not-a-yard", "v③", "￥⑩"):
            spot_token(text, 0.8)
            parse_down_distance_token(text)
            norm_dd(text, bare=True)
            parse_clock([text])
            parse_scorebug([text])
            classify_frame(10, [text, "1ST810", "1st 5:00"], 0.6, lex, _ON)
        self.assertEqual(spot_token("v③", 0.8), (3, "own"))
        self.assertEqual(spot_token("￥⑩", 0.8), (10, "own"))
        self.assertIsNone(spot_token("③", 0.8))
        self.assertIsNone(spot_token("1③", 0.8))
        self.assertIsNone(spot_token("garbage", 0.8))
        self.assertEqual(norm_dd("2ND③"), "2ND&3")
        self.assertEqual(parse_clock(["1st ③:⑩"]), (1, 3 * 60 + 10))

    def test_unparsed_yard_stays_empty(self) -> None:
        lex = _lex()
        frames = [
            {
                "t": 10.0,
                "g": 0.62,
                "tokens": [["1ST810", 0.9, 0.5, 0.95], ["③", 0.8, 0.2, 0.5], ["garbage", 0.4, 0.1, 0.1]],
            },
            {"t": 18.0, "g": 0.62, "tokens": [["1ST810", 0.9, 0.5, 0.95], ["1③", 0.8, 0.3, 0.5]]},
        ]
        rows = rows_from_frames(
            frames,
            video_id="circ",
            channel="SGU",
            game="madden27",
            extract=ExtractFlags(yardline_glyphs=True),
            lexicon=lex,
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["yardline"])
        self.assertLessEqual(rows[0]["conf"].get("yardline", 0.0), 0.2)
        sided = rows_from_frames(
            [{"t": 10.0, "g": 0.62, "tokens": [["1ST810", 0.9, 0.5, 0.95], ["v③", 0.8, 0.96, 0.95]]}],
            video_id="circ",
            channel="SGU",
            game="madden27",
            extract=ExtractFlags(yardline_glyphs=True),
            lexicon=lex,
        )
        self.assertIsNone(sided[0]["yardline"])
        self.assertEqual(sided[0]["glyph_yardline"], 3)
        self.assertLessEqual(sided[0]["conf"].get("yardline", 0.0), 0.2)
        sided[0]["field_spot"] = 3
        from vod_pilot.segment import gate_glyph_yardlines

        gate_glyph_yardlines(sided)
        self.assertEqual(sided[0]["yardline"], 3)


if __name__ == "__main__":
    unittest.main()
