"""Flags fill fields. They do not create or split snaps.

Snap ids are ``video:t_start``. v0.96 ids stay comparable: the only snaps that
disappear are menu rows dropped by --flag-menu.
"""

from __future__ import annotations

import unittest

from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon
from vod_pilot.segment import build_snaps, classify_frame
_ALL = ExtractFlags(
    read_defense=True,
    read_score=True,
    read_offense_formation=True,
    read_adjustments=True,
    yardline_glyphs=True,
    read_result=True,
    goal_to_go=True,
    dd_norm=True,
    presnap_clock=True,
    flag_menu=False,
)


def _lex() -> Lexicon:
    return Lexicon(
        formations=["Gun Deuce Close", "Singleback Deuce Close", "Nickel Over", "Nickel Wide"],
        plays=["HB Slam", "Inside Zone"],
        coverages=["Cover 3"],
    )


def _starts(frames, extract: ExtractFlags) -> list[float]:
    snaps = build_snaps(frames, video_id="seg", channel="SGU", game="madden27", extract=extract)
    return [snap.t_start for snap in snaps]


class SegmentationTests(unittest.TestCase):
    def test_recovered_downs_do_not_split_or_add_snaps(self) -> None:
        """2ND64 and a slotted 32 would each be a new down with --dd-norm.

        The baseline reader does not see them, so they must not open a snap
        or cut the 1st-and-10 run they sit inside.
        """
        lex = _lex()
        specs = [
            (10, ["1ST810", "1st 5:10", "(0-0) ATL", "7", "14", "(0-0) TB"], 0.62),
            (12, ["2ND64", "￥46"], 0.62),
            (14, ["1ST810", "1st 5:02", "DEUCE CLOSE"], 0.62),
            (40, ["2ND814", "1st 4:40", "NICKEL OVER"], 0.7),
            (48, ["2ND814", "HOT ROUTE SLANT"], 0.7),
        ]
        frames = [classify_frame(t, texts, green, lex, _ALL) for t, texts, green in specs]
        base_frames = [classify_frame(t, texts, green, lex) for t, texts, green in specs]
        self.assertEqual(_starts(frames, _ALL), _starts(base_frames, ExtractFlags()))
        self.assertEqual(_starts(frames, _ALL), [10.0, 40.0])
        # The recovered down is on the frame, and it is not its own snap.
        self.assertEqual((frames[1].down, frames[1].distance), (2, 4))
        self.assertEqual(frames[1].kind, "other")

    def test_menu_flag_only_removes_menu_snaps(self) -> None:
        lex = _lex()
        specs = [
            (10, ["1ST810", "1st 5:10"], 0.6),
            (18, ["1ST810"], 0.6),
            (50, ["PAUSE MENU", "1ST810", "1st 4:00"], 0.55),
            (56, ["PAUSE MENU", "1ST810"], 0.55),
            (90, ["2ND810", "1st 3:20"], 0.6),
        ]
        base = [classify_frame(t, texts, green, lex) for t, texts, green in specs]
        flagged = [
            classify_frame(t, texts, green, lex, ExtractFlags(flag_menu=True))
            for t, texts, green in specs
        ]
        base_ids = _starts(base, ExtractFlags())
        flagged_ids = _starts(flagged, ExtractFlags(flag_menu=True))
        self.assertEqual(base_ids, [10.0, 50.0, 90.0])
        self.assertEqual(flagged_ids, [10.0, 90.0])
        self.assertTrue(set(flagged_ids).issubset(base_ids))


if __name__ == "__main__":
    unittest.main()
