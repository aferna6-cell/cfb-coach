"""Scorebug reads. The parenthesis in front of a club is its record."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.scorebug import parse_scorebug
from vod_pilot.segment import build_snaps, classify_frame
from vod_pilot.tokens import Tok

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "score.json"
_ON = ExtractFlags(read_score=True)


def _lex() -> Lexicon:
    return Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])


class ScoreFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 7)
        for case in cases:
            got = parse_scorebug(case["texts"])
            self.assertEqual(got.team_left, case["team_left"], case["id"])
            self.assertEqual(got.team_right, case["team_right"], case["id"])
            self.assertEqual(got.score_left, case["score_left"], case["id"])
            self.assertEqual(got.score_right, case["score_right"], case["id"])
            self.assertEqual(got.record_left, case["record_left"], case["id"])
            self.assertEqual(got.record_right, case["record_right"], case["id"])
            if case["team_left"]:
                self.assertGreaterEqual(got.confidence, case["min_conf"], case["id"])
            else:
                self.assertEqual(got.confidence, 0.0, case["id"])


class ScoreSnapTests(unittest.TestCase):
    def test_paren_team_survives_onto_the_snap(self) -> None:
        lex = _lex()
        bar = ["(0-0) ATL", "7", "14", "(1-2) TB", "1ST810", "1st 5:10"]
        frames = [
            classify_frame(50, bar, 0.6, lex, _ON),
            classify_frame(58, bar, 0.6, lex, _ON),
        ]
        self.assertEqual(frames[0].team_left, "ATL")
        self.assertEqual(frames[0].score_left, 7)
        snaps = build_snaps(frames, video_id="s", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].team_left, "ATL")
        self.assertEqual(snaps[0].team_right, "TB")
        self.assertEqual(snaps[0].score_left, 7)
        self.assertEqual(snaps[0].score_right, 14)
        self.assertEqual(snaps[0].record_left, "0-0")
        self.assertGreaterEqual(snaps[0].confidence["score"], 0.86)

    def test_x_order_puts_the_left_club_on_the_left(self) -> None:
        got = parse_scorebug(
            [
                Tok("(0-0) TB", x=0.86),
                Tok("14", x=0.72),
                Tok("7", x=0.22),
                Tok("(0-0) ATL", x=0.08),
            ]
        )
        self.assertEqual((got.team_left, got.score_left), ("ATL", 7))
        self.assertEqual((got.team_right, got.score_right), ("TB", 14))

    def test_flag_off_leaves_the_score_empty(self) -> None:
        lex = _lex()
        frame = classify_frame(50, ["(0-0) ATL", "7", "14", "(0-0) TB", "1ST810"], 0.6, lex)
        self.assertEqual(frame.team_left, "")
        self.assertIsNone(frame.score_left)

    def test_disagreeing_frames_do_not_pick_one(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(50, ["(0-0) ATL", "7", "14", "(0-0) TB", "1ST810"], 0.6, lex, _ON),
            classify_frame(58, ["(0-0) ATL", "7", "21", "(0-0) TB", "1ST810"], 0.6, lex, _ON),
        ]
        snaps = build_snaps(frames, video_id="s", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].team_left, "")
        self.assertIsNone(snaps[0].score_left)


if __name__ == "__main__":
    unittest.main()
