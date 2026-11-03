"""Ball-spot glyphs. The arrow is not part of the yard line."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.segment import build_snaps, classify_frame, gate_glyph_yardlines
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
        # A glyph with no field spot and no neighboring snaps stays empty.
        self.assertIsNone(snaps[0].yardline)
        self.assertEqual(snaps[0].glyph_yardline, 35)
        self.assertLessEqual(snaps[0].confidence["yardline"], 0.2)
        snaps[0].field_spot = 35
        gate_glyph_yardlines(snaps)
        self.assertEqual(snaps[0].yardline, 35)
        self.assertIn("own 35", snaps[0].situation_raw)
        self.assertGreaterEqual(snaps[0].confidence["yardline"], 0.5)

    def test_glyph_does_not_override_or_blank_a_phrase(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(10, ["1ST810", "1st 5:10", Tok("￥40", conf=0.9)], 0.6, lex, _ON),
            classify_frame(14, ["1ST810", "OWN 22", "1st 5:02"], 0.6, lex, _ON),
        ]
        self.assertEqual(frames[0].yardline, 40)
        self.assertTrue(frames[0].yard_from_glyph)
        self.assertEqual(frames[1].yardline, 22)
        self.assertFalse(frames[1].yard_from_glyph)
        snaps = build_snaps(frames, video_id="y", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].yardline, 22)
        self.assertFalse(snaps[0].yard_from_glyph)
        self.assertIn("own 22", snaps[0].situation_raw)
        snaps[0].field_spot = 40
        gate_glyph_yardlines(snaps)
        self.assertEqual(snaps[0].yardline, 22)

    def test_glyph_needs_field_spot_or_both_neighbors(self) -> None:
        lex = _lex()
        rejected = build_snaps(
            [
                classify_frame(10, ["1ST810", "1st 5:10", Tok("^35", conf=0.8)], 0.6, lex, _ON),
                classify_frame(30, ["2ND810", "1st 4:40"], 0.6, lex, _ON),
            ],
            video_id="y",
            channel="SGU",
            game="madden27",
            extract=_ON,
        )
        self.assertIsNone(rejected[0].yardline)
        self.assertEqual(rejected[0].glyph_yardline, 35)
        self.assertLessEqual(rejected[0].confidence["yardline"], 0.2)
        rejected[0].field_spot = 37
        gate_glyph_yardlines(rejected)
        self.assertEqual(rejected[0].yardline, 35)
        rejected[0].field_spot = 22
        gate_glyph_yardlines(rejected)
        self.assertIsNone(rejected[0].yardline)
        self.assertLessEqual(rejected[0].confidence["yardline"], 0.2)

        confirmed = build_snaps(
            [
                classify_frame(10, ["1ST810", "OWN 25", "1st 8:00"], 0.6, lex, _ON),
                classify_frame(20, ["2ND85", "1st 7:40", Tok("^30", conf=0.8)], 0.6, lex, _ON),
                classify_frame(30, ["3RD81", "OWN 34", "1st 7:20"], 0.6, lex, _ON),
            ],
            video_id="y",
            channel="SGU",
            game="madden27",
            extract=_ON,
        )
        self.assertEqual([snap.yardline for snap in confirmed], [25, 30, 34])
        self.assertTrue(confirmed[1].yard_from_glyph)
        self.assertGreaterEqual(confirmed[1].confidence["yardline"], 0.5)

        missed = build_snaps(
            [
                classify_frame(10, ["1ST810", "OWN 25", "1st 8:00"], 0.6, lex, _ON),
                classify_frame(20, ["2ND85", "1st 7:40", Tok("^40", conf=0.8)], 0.6, lex, _ON),
                classify_frame(30, ["3RD81", "OWN 34", "1st 7:20"], 0.6, lex, _ON),
            ],
            video_id="y",
            channel="SGU",
            game="madden27",
            extract=_ON,
        )
        self.assertEqual(missed[0].yardline, 25)
        self.assertIsNone(missed[1].yardline)
        self.assertEqual(missed[1].glyph_yardline, 40)
        self.assertLessEqual(missed[1].confidence["yardline"], 0.2)
        self.assertEqual(missed[2].yardline, 34)
        missed[1].field_spot = 40
        gate_glyph_yardlines(missed)
        self.assertEqual(missed[1].yardline, 40)

    def test_glyph_does_not_flip_side_or_drop_an_earlier_spot(self) -> None:
        """The three box rows: neighbour 37, opp-30 vs ytg 70, own-22 vs opp territory."""
        rows = [
            {
                "snap_id": "RsOG3JCF7Wc:104.1",
                "yard_from_glyph": True,
                "glyph_yardline": 37,
                "glyph_phrase": "own 37",
                "glyph_conf": 0.51,
                "yardline": 37,
                "yards_to_goal": 63,
                "territory": "own",
                "field_spot": 37,
                "field_sources": {"yardline": "scorebug_spot+motion_from_neighbour"},
                "conf": {"yardline": 0.62},
            },
            {
                "snap_id": "RsOG3JCF7Wc:1061.1",
                "yard_from_glyph": True,
                "glyph_yardline": 70,
                "glyph_phrase": "opp 30",
                "glyph_conf": 0.55,
                "yardline": 30,
                "yards_to_goal": 70,
                "territory": "own",
                "field_spot": 30,
                "field_sources": {"yardline": "scorebug_spot+motion_from_neighbour"},
                "conf": {"yardline": 0.6},
            },
            {
                "snap_id": "cOPnN8CBQwI:1683.7",
                "yard_from_glyph": True,
                "glyph_yardline": 22,
                "glyph_phrase": "own 22",
                "glyph_conf": 0.51,
                "yardline": 78,
                "yards_to_goal": 22,
                "territory": "opp",
                "field_spot": 22,
                "field_sources": {"yardline": "scorebug_spot+motion_from_neighbour"},
                "conf": {"yardline": 0.6},
            },
        ]
        gate_glyph_yardlines(rows)
        self.assertEqual(rows[0]["yardline"], 37)
        self.assertEqual(rows[0]["yards_to_goal"], 63)
        self.assertEqual(rows[0]["territory"], "own")
        self.assertEqual(rows[1]["yardline"], 30)
        self.assertEqual(rows[1]["yards_to_goal"], 70)
        self.assertEqual(rows[1]["territory"], "own")
        self.assertEqual(rows[2]["yardline"], 78)
        self.assertEqual(rows[2]["territory"], "opp")
        self.assertEqual(rows[2]["yards_to_goal"], 22)
        # A bare 30 is not a license to write opp 30, and it must not clear ytg.
        flip = {
            "yard_from_glyph": True,
            "glyph_yardline": 70,
            "glyph_phrase": "opp 30",
            "glyph_conf": 0.51,
            "yardline": None,
            "yards_to_goal": 70,
            "territory": "own",
            "field_spot": 30,
            "conf": {},
        }
        own_on_opp = {
            "yard_from_glyph": True,
            "glyph_yardline": 22,
            "glyph_phrase": "own 22",
            "glyph_conf": 0.51,
            "yardline": None,
            "yards_to_goal": 22,
            "territory": "opp",
            "field_spot": 22,
            "conf": {},
        }
        gate_glyph_yardlines([flip, own_on_opp])
        self.assertIsNone(flip["yardline"])
        self.assertEqual(flip["yards_to_goal"], 70)
        self.assertEqual(flip["territory"], "own")
        self.assertIsNone(own_on_opp["yardline"])
        self.assertEqual(own_on_opp["territory"], "opp")

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
