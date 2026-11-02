"""Menu, lobby, splash, play-select, and replay flags. Gameplay is not guessed."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.menu_flag import menu_flag
from vod_pilot.segment import build_snaps, classify_frame

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "menu.json"
_ON = ExtractFlags(flag_menu=True)


class MenuFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 8)
        for case in cases:
            flagged, conf = menu_flag(case["texts"])
            self.assertEqual(flagged, case["flag"], case["id"])
            if case["flag"]:
                self.assertGreaterEqual(conf, case["min_conf"], case["id"])
            else:
                self.assertEqual(conf, 0.0, case["id"])


class MenuSnapTests(unittest.TestCase):
    def test_pause_over_a_scorebug_flags_the_row(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        frames = [
            classify_frame(10, ["PAUSE MENU", "1ST810", "1st 5:00"], 0.5, lex, _ON),
            classify_frame(16, ["PAUSE MENU", "1ST810"], 0.5, lex, _ON),
        ]
        self.assertTrue(frames[0].is_menu_or_lobby)
        snaps = build_snaps(frames, video_id="m", channel="SGU", game="madden27", extract=_ON)
        self.assertTrue(snaps[0].is_menu_or_lobby)
        self.assertGreaterEqual(snaps[0].menu_confidence, 0.9)

    def test_flag_off_leaves_the_row_clear(self) -> None:
        lex = Lexicon(formations=["Gun Bunch"], plays=["HB Slam"], coverages=["Cover 3"])
        frame = classify_frame(10, ["PAUSE MENU", "1ST810"], 0.5, lex)
        self.assertFalse(frame.is_menu_or_lobby)
        self.assertEqual(frame.menu_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
