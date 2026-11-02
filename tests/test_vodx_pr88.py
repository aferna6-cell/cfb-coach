"""Box glue: cache tokens, flags, and confident-reader wins."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
import vodx_pr88
from vodx_pr88 import flags_from_fixes, rows_from_frames, run_cache, stage_write


def _lex() -> Lexicon:
    return Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])


def _frames() -> list[dict]:
    return [
        {
            "t": 10.0,
            "g": 0.62,
            "tokens": [
                ["1ST810", 0.9, 0.4, 0.95],
                ["1st 5:10", 0.8, 0.2, 0.2],
                ["(0-0) ATL", 0.9, 0.08, 0.93],
                ["7", 0.9, 0.22, 0.93],
                ["14", 0.9, 0.72, 0.93],
                ["(0-0) TB", 0.9, 0.88, 0.93],
            ],
        },
        {
            "t": 12.0,
            "g": 0.62,
            "tokens": [["2ND64", 0.8, 0.87, 0.96]],
        },
        {
            "t": 14.0,
            "g": 0.62,
            "tokens": [["1ST810", 0.9, 0.4, 0.95]],
        },
    ]


class FixesTests(unittest.TestCase):
    def test_comma_list_and_all(self) -> None:
        flags = flags_from_fixes("read-defense, dd-norm")
        self.assertTrue(flags.read_defense)
        self.assertTrue(flags.dd_norm)
        self.assertFalse(flags.read_score)
        everything = flags_from_fixes("all")
        self.assertTrue(everything.field_coverage)
        self.assertFalse(everything.book_match)

    def test_unknown_flag_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            flags_from_fixes("not-a-flag")


class CacheRebuildTests(unittest.TestCase):
    def test_positions_reach_the_row_and_do_not_add_a_snap(self) -> None:
        plain = rows_from_frames(
            _frames(),
            video_id="vid",
            channel="SGU",
            game="madden27",
            extract=ExtractFlags(),
            lexicon=_lex(),
        )
        fixed = rows_from_frames(
            _frames(),
            video_id="vid",
            channel="SGU",
            game="madden27",
            extract=flags_from_fixes(
                "read-score,dd-norm,read-defense,yardline-glyphs,goal-to-go,read-result,presnap-clock"
            ),
            lexicon=_lex(),
        )
        self.assertEqual([row["snap_id"] for row in fixed], [row["snap_id"] for row in plain])
        self.assertEqual(fixed[0]["snap_id"], "vid:10.0")
        self.assertEqual(fixed[0]["team_left"], "ATL")
        self.assertEqual(fixed[0]["score_left"], 7)
        self.assertEqual(fixed[0]["score_right"], 14)
        self.assertEqual(fixed[0]["score"], "7-14")
        self.assertIn("5:10", fixed[0]["clock"])

    def test_cache_only_writes_fixes_and_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "ocr"
            cache.mkdir()
            lines = [
                json.dumps({"t": 10, "g": 0.6, "k": [["1ST810", 0.9, 0.5, 0.95], ["1st 5:00", 0.8, 0.2, 0.2]]}),
                json.dumps({"t": 18, "g": 0.6, "k": [["1ST810", 0.9, 0.5, 0.95]]}),
            ]
            (cache / "abc.fps0.5.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            out = root / "labels" / "v0.97"
            manifest = run_cache(
                cache,
                out_dir=out,
                fixes="field-coverage,read-defense",
                game="madden27",
                channel="SGU",
                video_ids=["abc"],
                version="v0.97",
            )
            self.assertTrue(manifest["source"]["cache_only"])
            self.assertEqual(manifest["source"]["fixes"], ["read_defense", "field_coverage"])
            self.assertIn("offensive_formation", manifest["field_coverage"])
            self.assertNotIn("book_coverage", manifest["field_coverage"])
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["source"]["fixes"], manifest["source"]["fixes"])
            self.assertEqual(summary["field_coverage"]["rows"], manifest["counts"]["snaps_labeled"])
            # The cache file was the only input. No video path is required.
            self.assertTrue((out / "snaps.jsonl").exists())


class StageConfidenceTests(unittest.TestCase):
    def test_later_stage_loses_when_the_new_reader_is_more_confident(self) -> None:
        rows = rows_from_frames(
            _frames(),
            video_id="vid",
            channel="SGU",
            game="madden27",
            extract=flags_from_fixes("read-score"),
            lexicon=_lex(),
        )
        row = rows[0]
        self.assertFalse(stage_write(row, "score_left", 3, 0.4, "vodx_score"))
        self.assertEqual(row["score_left"], 7)
        self.assertFalse(stage_write(row, "score_left", 7, 0.99, "vodx_score"))
        self.assertTrue(stage_write(row, "score_left", 10, 0.95, "vodx_score"))
        self.assertEqual(row["score_left"], 10)

    def test_adjustment_fields_are_carried(self) -> None:
        frames = [
            {
                "t": 30.0,
                "g": 0.6,
                "tokens": [["1ST810", 0.9, 0.5, 0.95], ["HOT ROUTE SLANT", 0.9, 0.4, 0.5]],
            },
            {"t": 36.0, "g": 0.6, "tokens": [["1ST810", 0.9, 0.5, 0.95]]},
        ]
        rows = rows_from_frames(
            frames,
            video_id="vid",
            channel="SGU",
            game="madden27",
            extract=flags_from_fixes("read-adjustments"),
            lexicon=_lex(),
        )
        row = rows[0]
        self.assertEqual(row["hot_routes"], "HOT ROUTE SLANT")
        self.assertIn("hot_route: HOT ROUTE SLANT", row["adjustments"])
        self.assertFalse(stage_write(row, "hot_routes", "OTHER", 0.2, "vodx_refine"))
        self.assertEqual(row["hot_routes"], "HOT ROUTE SLANT")


if __name__ == "__main__":
    unittest.main()
