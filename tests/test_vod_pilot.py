"""Parser and snap-grouping tests for the VOD pilot. No video required."""

from __future__ import annotations

import unittest

from cfb_coach.outcome import parse_outcome
from cfb_coach.situation import parse_situation

from vod_pilot.accuracy import score
from vod_pilot.lexicon import Lexicon
from vod_pilot.parse_hud import parse_clock, parse_down_distance, parse_down_distance_token
from vod_pilot.schema import SNAP_COLUMNS, result_text, situation_raw
from vod_pilot.segment import FrameObs, build_snaps, classify_frame


def _lex() -> Lexicon:
    return Lexicon(
        formations=[
            "Gun Bunch Str Nasty",
            "Gun Doubles Clamp Stack",
            "Gun Cluster",
            "Singleback Deuce Close",
        ],
        plays=["HB Slam", "Inside Zone", "Mesh Spot", "Cover 1 Contain"],
        coverages=["Cover 1", "Cover 1 Contain", "Cover 3", "Cover 4 Quarters"],
    )


class ParseHudTests(unittest.TestCase):
    def test_ampersand_read_as_eight(self) -> None:
        self.assertEqual(parse_down_distance_token("1ST810"), (1, 10))
        self.assertEqual(parse_down_distance_token("2ND814"), (2, 14))
        self.assertEqual(parse_down_distance_token("3RD814"), (3, 14))
        self.assertEqual(parse_down_distance_token("4TH822"), (4, 22))
        self.assertEqual(parse_down_distance_token("1ST815"), (1, 15))

    def test_real_distance_without_false_ampersand(self) -> None:
        self.assertEqual(parse_down_distance_token("4TH22"), (4, 22))
        self.assertEqual(parse_down_distance_token("1ST8"), (1, 8))
        self.assertEqual(parse_down_distance_token("1st & 10"), (1, 10))

    def test_mashed_own_yardline_follows_the_current_down(self) -> None:
        from vod_pilot.parse_hud import parse_yardline_phrase

        texts = ["Illegal Man Downfield", "1st&15onown22", "2nd&10onown27"]
        yl, phrase = parse_yardline_phrase(texts, (1, 15))
        self.assertEqual(yl, 22)
        self.assertEqual(phrase, "own 22")

    def test_clock_is_not_a_down(self) -> None:
        self.assertIsNone(parse_down_distance_token("1st3:56"))
        self.assertEqual(parse_clock(["1st3:56"]), (1, 3 * 60 + 56))
        self.assertEqual(parse_down_distance(["1st3:56", "1ST810"]), (1, 10))

    def test_situation_string_round_trips_into_the_coach(self) -> None:
        raw = situation_raw(down=2, distance=7, quarter=1, clock="3:56")
        sit = parse_situation(raw)
        self.assertEqual(sit.down, 2)
        self.assertEqual(sit.distance, 7)

    def test_result_strings_round_trip_into_the_coach(self) -> None:
        self.assertEqual(parse_outcome(result_text(kind="gain", yards=4)).yards, 4)
        self.assertEqual(parse_outcome(result_text(kind="loss", yards=-8)).kind, "loss")
        self.assertEqual(parse_outcome(result_text(kind="sack", yards=None)).kind, "sack")
        self.assertEqual(parse_outcome(result_text(kind="incomplete", yards=0)).kind, "incomplete")
        self.assertEqual(parse_outcome(result_text(kind="int", yards=0)).kind, "int")
        self.assertEqual(parse_outcome(result_text(kind="convert", yards=None)).kind, "convert")


class SegmentTests(unittest.TestCase):
    def test_sgu_style_drive_yards(self) -> None:
        """1st&15 -> 2nd&14 (+1), 2nd&14 -> 3rd&14 (0), 3rd&14 -> 4th&22 sack (-8)."""
        lex = _lex()
        specs = [
            (32, ["KICKOFF", "1st6:00", "35"], 0.3),
            (56, ["KICKOFF", "1st5:59"], 0.7),
            (80, ["1ST810", "1st5:57"], 0.55),
            (92, ["1ST810", "1st5:50"], 0.7),
            (116, ["1ST815", "1st5:48"], 0.7),
            (128, ["1ST815"], 0.72),
            (140, ["2ND814", "1st5:42"], 0.3),
            (152, ["2ND814", "1st5:19"], 0.7),
            (164, ["3RD814"], 0.3),
            (176, ["3RD814"], 0.62),
            (188, ["4TH822", "1st5:06", "1SACK"], 0.75),
            (200, ["4TH22", "1st4:54"], 0.2),
        ]
        frames = [classify_frame(t, texts, green, lex) for t, texts, green in specs]
        snaps = build_snaps(frames, video_id="demo", channel="SGU", game="madden27")
        by_dd = [(s.down, s.distance, s.result, s.play) for s in snaps]
        self.assertIn((None, None, "", "Kickoff"), [(s.down, s.distance, s.result, s.play) for s in snaps][:1] or by_dd)
        results = {(s.down, s.distance): s.result for s in snaps}
        self.assertEqual(results[(1, 15)], "+1")
        self.assertEqual(results[(2, 14)], "+0")
        self.assertEqual(results[(3, 14)], "-8")
        sack = next(s for s in snaps if s.down == 3 and s.distance == 14)
        self.assertEqual(sack.result_kind, "sack")

    def test_play_call_menu_does_not_pick_one_formation_of_many(self) -> None:
        lex = _lex()
        texts = [
            "BUNCH STR NASTY",
            "CLUSTER",
            "DOUBLES CLAMP STACK",
            "HB Slam in 11 personnel under center attacks the middle. Great for short yardage.",
        ]
        frame = classify_frame(44, texts, 0.22, lex)
        self.assertEqual(frame.kind, "play_call")
        self.assertEqual(frame.formation, "")
        self.assertEqual(frame.play, "HB Slam")

    def test_single_coverage_on_a_defensive_call(self) -> None:
        lex = _lex()
        frame = classify_frame(128, ["COVER1CONTAIN", "COACH SUGGESTIONS", "1st2:39"], 0.06, lex)
        self.assertEqual(frame.kind, "play_call")
        self.assertEqual(frame.coverage, "Cover 1 Contain")

    def test_mashed_sack_token(self) -> None:
        from vod_pilot.parse_hud import parse_flags

        self.assertIn("sack", parse_flags(["1SACK", "Nick Bosa"]))
        self.assertNotIn("sack", parse_flags(["Sack the QB-0/3", "Unstoppable Force"]))

    def test_menu_with_shared_formation_tails_does_not_pick_one_row(self) -> None:
        lex = Lexicon(
            formations=["Gun Cluster", "Gun Bunch Str Nasty", "Shotgun Bunch Str Nasty"],
            plays=["HB Slam"],
            coverages=["Cover 3"],
        )
        texts = [
            "BUNCHSTRNASTY",
            "CLUSTER",
            "HB Slam in 11 personnel under center attacks the middle. Great for short yardage.",
        ]
        frame = classify_frame(44, texts, 0.2, lex)
        self.assertEqual(frame.formation, "")
        self.assertEqual(frame.formation_hits, [])
        self.assertEqual(frame.play, "HB Slam")

    def test_previous_play_and_touchdown_attach_to_the_snap_that_ended(self) -> None:
        lex = _lex()
        specs = [
            (100, ["1ST810", "1st3:10"], 0.6),
            (110, ["1ST810", "1st3:04"], 0.6),
            (118, ["PREVIOUS PLAY", "COVER1CONTAIN", "TOUCHDOWN", "1st2:39"], 0.05),
            (130, ["KICKOFF", "1st2:30"], 0.4),
            (140, ["KICKOFF"], 0.5),
        ]
        frames = [classify_frame(t, texts, green, lex) for t, texts, green in specs]
        self.assertEqual(frames[2].coverage, "")
        self.assertEqual(frames[2].previous_coverage, "Cover 1 Contain")
        snaps = build_snaps(frames, video_id="demo", channel="Era", game="madden27")
        scoring = next(s for s in snaps if s.down == 1)
        self.assertEqual(scoring.result, "td")
        self.assertEqual(scoring.coverage_seen, "Cover 1 Contain")
        self.assertEqual(scoring.side, "offense")
        kick = next(s for s in snaps if s.play == "Kickoff")
        self.assertEqual(kick.coverage_seen, "")
        self.assertEqual(kick.formation, "")

    def test_penalty_screen_does_not_donate_a_formation(self) -> None:
        lex = _lex()
        specs = [
            (150, ["1ST810", "1st2:10"], 0.6),
            (160, ["1ST810"], 0.6),
            (168, ["PENALTY", "COVER 1", "Gun Cluster", "ACCEPT"], 0.1),
        ]
        frames = [classify_frame(t, texts, green, lex) for t, texts, green in specs]
        self.assertEqual(frames[2].formation, "")
        self.assertEqual(frames[2].coverage, "")
        self.assertEqual(frames[2].play, "")
        snaps = build_snaps(frames, video_id="demo", channel="Era", game="madden27")
        self.assertEqual(snaps[0].formation, "")
        self.assertEqual(snaps[0].coverage_seen, "")

    def test_select_a_play_keeps_the_tooltip_and_drops_the_tab(self) -> None:
        lex = _lex()
        specs = [
            (36, [
                "SELECT A PLAY",
                "BUNCH STR NASTY",
                "CLUSTER",
                "HB Slam in 11 personnel under center attacks the middle.",
            ], 0.1),
            (50, ["1ST810", "1st3:50"], 0.6),
            (60, ["1ST810"], 0.6),
        ]
        frames = [classify_frame(t, texts, green, lex) for t, texts, green in specs]
        snaps = build_snaps(frames, video_id="demo", channel="Era", game="madden27")
        self.assertEqual(snaps[0].play, "HB Slam")
        self.assertEqual(snaps[0].formation, "")

    def test_dynasty_menu_is_not_a_snap(self) -> None:
        lex = _lex()
        frames = [
            classify_frame(t, ["Dynasty Central", "Recruiting", "Play Game", "Create Savepoint"], 0.65, lex)
            for t in (40, 52, 64, 76)
        ]
        snaps = build_snaps(frames, video_id="cfb", channel="Gamer Ability", game="cfb27")
        self.assertEqual(snaps, [])

    def test_csv_columns_match_log_snap(self) -> None:
        self.assertIn("coverage_seen", SNAP_COLUMNS)
        self.assertIn("situation_raw", SNAP_COLUMNS)
        self.assertNotIn("confidence", SNAP_COLUMNS)


class AccuracyTests(unittest.TestCase):
    def test_false_fill_is_not_recall(self) -> None:
        preds = [
            {
                "t_start": 10,
                "video_id": "v",
                "down": 1,
                "distance": 10,
                "quarter": 1,
                "yardline": "",
                "formation": "Gun Bunch",
                "play": "",
                "coverage_seen": "",
                "result": "+4",
            }
        ]
        labels = [
            {
                "t_start": 11,
                "down": 1,
                "distance": 10,
                "quarter": 1,
                "yardline": None,
                "formation": None,
                "play": None,
                "coverage_seen": None,
                "result": "+4",
            }
        ]
        report = score(preds, labels)
        self.assertEqual(report["fields"]["down"]["recall"], 1.0)
        self.assertEqual(report["fields"]["formation"]["false_fills"], 1)
        self.assertEqual(report["fields"]["formation"]["recall"], None)
        self.assertEqual(report["fields"]["result"]["recall"], 1.0)


class FrameObsShape(unittest.TestCase):
    def test_dataclass_used(self) -> None:
        obs = FrameObs(t=1.0, kind="live", down=1, distance=10)
        self.assertEqual(obs.dd, (1, 10))


if __name__ == "__main__":
    unittest.main()
