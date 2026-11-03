"""Defensive formation reads. A list of fronts is not a call."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.screen_read import read_defensive_formation
from vod_pilot.segment import build_snaps, classify_frame

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "defense.json"
_ON = ExtractFlags(read_defense=True)


def _lex() -> Lexicon:
    return Lexicon(
        formations=["Gun Bunch Str Nasty", "Nickel Over", "Nickel Wide", "Dime 3-2 Odd"],
        plays=["HB Slam", "Cover 3 Buzz"],
        coverages=["Cover 3", "Cover 3 Buzz"],
    )


class DefenseFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 8)
        for case in cases:
            name, conf = read_defensive_formation(case["texts"])
            self.assertEqual(name, case["expect"], case["id"])
            if case["expect"]:
                self.assertGreaterEqual(conf, case["min_conf"], case["id"])
            else:
                self.assertEqual(conf, 0.0, case["id"])


class DefenseSnapTests(unittest.TestCase):
    def test_unique_front_becomes_the_defense_row(self) -> None:
        lex = _lex()
        specs = [
            (40, ["NICKELOVER", "COVER 3 BUZZ"], 0.12),
            (50, ["1ST810", "1st 5:10"], 0.6),
            (58, ["1ST810", "1st 5:02"], 0.6),
        ]
        frames = [classify_frame(t, texts, green, lex, _ON) for t, texts, green in specs]
        self.assertEqual(frames[0].def_formation, "Nickel Over")
        self.assertGreaterEqual(frames[0].def_formation_conf, 0.9)
        snaps = build_snaps(frames, video_id="d", channel="SGU", game="madden27", extract=_ON)
        row = next(s for s in snaps if s.down == 1)
        self.assertEqual(row.side, "defense")
        self.assertEqual(row.formation, "Nickel Over")
        self.assertEqual(row.def_formation, "Nickel Over")
        self.assertGreater(row.confidence["def_formation"], 0)

    def test_previous_play_front_stays_on_the_offense_snap(self) -> None:
        lex = _lex()
        specs = [
            (100, ["1ST810", "1st 3:10"], 0.6),
            (110, ["1ST810", "1st 3:04"], 0.6),
            (118, ["PREVIOUS PLAY", "NICKEL OVER", "COVER 3"], 0.05),
        ]
        frames = [classify_frame(t, texts, green, lex, _ON) for t, texts, green in specs]
        self.assertEqual(frames[2].formation, "")
        self.assertEqual(frames[2].previous_def_formation, "Nickel Over")
        snaps = build_snaps(frames, video_id="d", channel="Era", game="madden27", extract=_ON)
        row = next(s for s in snaps if s.down == 1)
        self.assertEqual(row.side, "offense")
        self.assertEqual(row.formation, "")
        self.assertEqual(row.def_formation, "Nickel Over")
        self.assertGreaterEqual(row.confidence["def_formation"], 0.8)

    def test_flag_off_does_not_invent_a_front(self) -> None:
        lex = _lex()
        frame = classify_frame(40, ["NICKELOVER"], 0.1, lex)
        snaps = build_snaps([frame], video_id="d", channel="SGU", game="madden27")
        # No live down, so no snap. The frame itself must not carry the new field.
        self.assertEqual(snaps, [])
        self.assertEqual(frame.def_formation, "")
        self.assertEqual(frame.previous_def_formation, "")


if __name__ == "__main__":
    unittest.main()
