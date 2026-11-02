"""The snap clock is the one on the screen before the ball is snapped."""

from __future__ import annotations

import unittest

from vod_pilot.flags import ExtractFlags
from vod_pilot.segment import FrameObs, build_snaps

_ON = ExtractFlags(presnap_clock=True)


def _live(t: float, clock: int | None) -> FrameObs:
    return FrameObs(t=t, kind="live", down=1, distance=10, quarter=1, clock=clock, green=0.6)


class PresnapClockTests(unittest.TestCase):
    def test_play_call_clock_beats_the_post_play_clock(self) -> None:
        frames = [
            FrameObs(t=8, kind="play_call", quarter=1, clock=3 * 60 + 56, play="Inside Zone"),
            _live(10, None),
            _live(18, 3 * 60 + 40),
        ]
        snaps = build_snaps(frames, video_id="c", channel="SGU", game="madden27", extract=_ON)
        self.assertIn("3:56", snaps[0].situation_raw)
        self.assertNotIn("3:40", snaps[0].situation_raw)

    def test_flag_off_uses_the_first_live_clock(self) -> None:
        frames = [
            FrameObs(t=8, kind="play_call", quarter=1, clock=3 * 60 + 56, play="Inside Zone"),
            _live(10, None),
            _live(18, 3 * 60 + 40),
        ]
        snaps = build_snaps(frames, video_id="c", channel="SGU", game="madden27")
        self.assertIn("3:40", snaps[0].situation_raw)
        self.assertNotIn("3:56", snaps[0].situation_raw)


if __name__ == "__main__":
    unittest.main()
