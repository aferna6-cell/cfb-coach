"""Box glue: cache tokens, flags, and confident-reader wins."""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
import vodx_pr88
from vodx_pr88 import flags_from_fixes, rows_from_frames, run_cache, stage_write, stamp_manifest_row


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


class GoalDistanceTests(unittest.TestCase):
    def test_refine_rows_accepts_a_goal_to_go_blank_distance(self) -> None:
        """--fixes goal-to-go used to crash apply_quality: yy = dist - ndist."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from vodx_refine import refine_rows

        lex = _lex()
        frames = [
            {"t": 10.0, "g": 0.6, "tokens": [["1STGOAL", 0.9, 0.5, 0.95], ["1st 2:00", 0.8, 0.2, 0.2]]},
            {"t": 20.0, "g": 0.6, "tokens": [["2NDGOAL", 0.9, 0.5, 0.95], ["1st 1:40", 0.8, 0.2, 0.2]]},
        ]
        rows = rows_from_frames(
            frames,
            video_id="hGyL",
            channel="CFB",
            game="cfb27",
            extract=flags_from_fixes("goal-to-go"),
            lexicon=lex,
        )
        self.assertTrue(rows[0]["goal_to_go"])
        self.assertIsInstance(rows[0]["down"], int)
        self.assertIsNone(rows[0]["distance"])
        # The box row that crashed: empty string distance, int down, goal flag set.
        crashed = {
            "snap_id": "hGyL:10.0",
            "down": 1,
            "distance": "",
            "goal_to_go": True,
            "yards": None,
            "result": "",
        }
        nxt = {"snap_id": "hGyL:20.0", "down": 2, "distance": 8, "goal_to_go": True, "yards": None, "result": ""}
        refine_rows([crashed, nxt], [], "cfb27")
        refine_rows(rows, [], "cfb27")
        self.assertIsNone(crashed["distance"])
        self.assertTrue(crashed["goal_to_go"])
        self.assertIsNone(rows[0]["distance"])
        self.assertTrue(rows[0]["goal_to_go"])

    def test_missing_goal_distance_still_resolves_and_keeps_a_score(self) -> None:
        """None used to make apply_quality skip the row. \"\" must not be subtracted."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from vodx_quality import apply_quality

        goal = {
            "down": 1,
            "distance": None,
            "goal_to_go": True,
            "baseline_distance": 1,
            "yards": None,
            "result": "",
            "result_kind": "",
            "t_start": 10.0,
            "t_end": 18.0,
            "clock": "2:00",
        }
        nxt = {
            "down": 2,
            "distance": 6,
            "yards": None,
            "result": "",
            "result_kind": "",
            "t_start": 20.0,
            "t_end": 28.0,
            "clock": "1:40",
        }
        apply_quality([goal, nxt])
        self.assertIsNone(goal["distance"])
        self.assertEqual(goal["yards"], -5)
        self.assertEqual(goal["result"], "-5")

        normal = {
            "down": 1,
            "distance": 10,
            "yards": None,
            "result": "",
            "result_kind": "",
            "t_start": 10.0,
            "t_end": 18.0,
            "clock": "5:00",
        }
        second = {
            "down": 2,
            "distance": 6,
            "yards": None,
            "result": "",
            "result_kind": "",
            "t_start": 20.0,
            "t_end": 28.0,
            "clock": "4:40",
        }
        apply_quality([normal, second])
        self.assertEqual(normal["yards"], 4)
        self.assertEqual(normal["result"], "+4")

        scored = {
            "down": 1,
            "distance": "",
            "goal_to_go": True,
            "yards": None,
            "result": "td",
            "result_kind": "td",
        }
        kick = {"down": 1, "distance": 10, "yards": None, "result": "", "result_kind": ""}
        apply_quality([scored, kick])
        self.assertIsNone(scored["distance"])
        self.assertEqual(scored["result"], "td")
        self.assertEqual(scored["result_kind"], "td")

        blank = {"down": 1, "distance": "", "goal_to_go": True, "yards": None, "result": "", "result_kind": ""}
        also = {"down": 2, "distance": "", "goal_to_go": True, "yards": None, "result": "", "result_kind": ""}
        apply_quality([blank, also])
        self.assertIsNone(blank["distance"])
        self.assertIsNone(blank["yards"])

    def test_restore_keeps_a_neighbour_motion_yardline(self) -> None:
        from vodx_pr88 import restore_rows

        row = {
            "snap_id": "RsOG3JCF7Wc:104.1",
            "yard_from_glyph": True,
            "yardline": 37,
            "yards_to_goal": 63,
            "territory": "own",
            "field_sources": {"yardline": "scorebug_spot+motion_from_neighbour"},
            "conf": {"yardline": 0.62},
            "_pr88_conf": {},
        }
        restore_rows(
            [row],
            {"RsOG3JCF7Wc:104.1": {"values": {"yardline": None}, "conf": {"yardline": 0.15}}},
        )
        self.assertEqual(row["yardline"], 37)
        self.assertEqual(row["yards_to_goal"], 63)
        self.assertEqual(row["territory"], "own")


class ManifestMetaTests(unittest.TestCase):
    def test_opponent_and_side_stay(self) -> None:
        row = {"side": "defense", "opponent_type": "unknown", "game": "madden27"}
        stamp_manifest_row(row, {"opponent_type": "human", "game": "cfb27", "video_id": "D1xGdV6Yv2I"})
        self.assertEqual(row["opponent_type"], "human")
        self.assertEqual(row["opponent_type_source"], "scout_manifest")
        self.assertEqual(row["side"], "defense")
        self.assertEqual(row["game"], "cfb27")


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
            with gzip.open(out / "abc.jsonl.gz", "rt", encoding="utf-8") as fh:
                saved = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(saved), manifest["counts"]["snaps_labeled"])


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
