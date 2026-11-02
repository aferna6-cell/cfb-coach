"""Offensive formation reads. A shared Gun/Shotgun set is still one name."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.screen_read import read_offensive_formation
from vod_pilot.segment import build_snaps, classify_frame

_FIXTURE = Path(__file__).parent / "fixtures" / "vod" / "offense_formation.json"
_ON = ExtractFlags(read_offense_formation=True)


def _lex() -> Lexicon:
    return Lexicon(
        formations=["Gun Deuce Close", "Singleback Deuce Close", "Pistol Deuce Close", "Gun Bunch"],
        plays=["Inside Zone", "HB Slam"],
        coverages=["Cover 3"],
    )


class OffenseFormationFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(payload["cases"]), 6)
        for case in payload["cases"]:
            name, conf = read_offensive_formation(case["texts"], payload["names"])
            self.assertEqual(name, case["expect"], case["id"])
            if case["expect"]:
                self.assertGreaterEqual(conf, case["min_conf"], case["id"])
            else:
                self.assertEqual(conf, 0.0, case["id"])


class OffenseFormationSnapTests(unittest.TestCase):
    def test_shared_set_is_kept_when_the_flag_is_on(self) -> None:
        lex = _lex()
        specs = [
            (36, ["DEUCE CLOSE", "SELECT A PLAY"], 0.15),
            (50, ["1ST810", "1st 4:10"], 0.6),
            (60, ["1ST810"], 0.6),
        ]
        frames = [classify_frame(t, texts, green, lex, _ON) for t, texts, green in specs]
        self.assertEqual(frames[0].formation, "Deuce Close")
        self.assertGreaterEqual(frames[0].off_formation_conf, 0.84)
        snaps = build_snaps(frames, video_id="o", channel="SGU", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].formation, "Deuce Close")
        self.assertEqual(snaps[0].side, "offense")
        self.assertGreaterEqual(snaps[0].confidence["formation"], 0.84)

    def test_flag_off_still_abstains_on_a_shared_tail(self) -> None:
        lex = _lex()
        frame = classify_frame(36, ["DEUCE CLOSE", "SELECT A PLAY"], 0.15, lex)
        self.assertEqual(frame.formation, "")

    def test_previous_play_set_attaches_to_the_snap_that_ended(self) -> None:
        lex = _lex()
        specs = [
            (100, ["1ST810", "1st 3:10"], 0.6),
            (110, ["1ST810"], 0.6),
            (118, ["PREVIOUS PLAY", "DEUCE CLOSE"], 0.05),
        ]
        frames = [classify_frame(t, texts, green, lex, _ON) for t, texts, green in specs]
        self.assertEqual(frames[2].formation, "")
        self.assertEqual(frames[2].previous_off_formation, "Deuce Close")
        snaps = build_snaps(frames, video_id="o", channel="Era", game="madden27", extract=_ON)
        self.assertEqual(snaps[0].formation, "Deuce Close")
        self.assertEqual(snaps[0].side, "offense")


if __name__ == "__main__":
    unittest.main()
