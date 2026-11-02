"""Field coverage replaces book coverage. book_match never overwrites the read."""

from __future__ import annotations

import unittest

from vod_pilot.field_coverage import coverage_report, exact_book_match
from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.schema import SnapRow
from vod_pilot.segment import build_snaps, classify_frame


class BookMatchTests(unittest.TestCase):
    def test_exact_match_does_not_rewrite_a_near_miss(self) -> None:
        names = ["Inside Zone", "Inside Zone Weak", "HB Slam"]
        self.assertEqual(exact_book_match("INSIDE ZONE", names), "Inside Zone")
        self.assertEqual(exact_book_match("INSIDE ZNE", names), "")
        self.assertEqual(exact_book_match("ZONE", names), "")

    def test_snap_play_stays_the_read(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam", "Inside Zone"], coverages=["Cover 3"])
        frames = [
            classify_frame(36, ["SELECT A PLAY", "HB Slam in 11 personnel under center attacks the middle."], 0.1, lex),
            classify_frame(50, ["1ST810", "1st 3:50"], 0.6, lex),
            classify_frame(60, ["1ST810"], 0.6, lex),
        ]
        before = [frame.play for frame in frames]
        snaps = build_snaps(
            frames,
            video_id="b",
            channel="Era",
            game="madden27",
            extract=ExtractFlags(book_match=True),
        )
        self.assertEqual([frame.play for frame in frames], before)
        self.assertEqual(snaps[0].play, "HB Slam")
        self.assertEqual(snaps[0].book_match, "HB Slam")


class CoverageReportTests(unittest.TestCase):
    def test_report_has_the_label_fields_and_not_a_book_rate(self) -> None:
        rows = [
            SnapRow(
                ts="t",
                opponent_id="vod:x",
                side="offense",
                formation="Gun Trips",
                play="Inside Zone",
                def_formation="Nickel Over",
                coverage_seen="Cover 3",
                adjustments=["hot_route: HOT ROUTE SLANT"],
                score_left=7,
                score_right=14,
                yardline=35,
                result="+4",
                result_kind="gain",
            ),
            SnapRow(
                ts="t2",
                opponent_id="vod:x",
                side="defense",
                formation="Nickel Wide",
                play="Cover 2",
            ),
        ]
        report = coverage_report(rows)
        self.assertNotIn("book", report)
        self.assertNotIn("book_coverage", report)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["offensive_formation"], 0.5)
        self.assertEqual(report["defensive_formation"], 1.0)
        self.assertEqual(report["defensive_call"], 0.5)
        self.assertEqual(report["adjustments"], 0.5)
        self.assertEqual(report["score"], 0.5)
        self.assertEqual(report["yardline"], 0.5)
        self.assertEqual(report["result"], 0.5)
        self.assertEqual(report["by_side"]["offense"]["rows"], 1)
        self.assertEqual(report["by_side"]["defense"]["offensive_formation"], 0.0)
        self.assertEqual(report["by_side"]["defense"]["defensive_formation"], 1.0)


if __name__ == "__main__":
    unittest.main()
