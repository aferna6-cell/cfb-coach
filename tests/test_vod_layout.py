"""No-flag snap counts follow the v0.96 path: layout, then plain-text classify.

YJZksbAu6jE and D1xGdV6Yv2I grew when pre-game and ticker tokens were classified
with no gameplay window and no mask. The window and mask run first.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from vod_pilot.lexicon import Lexicon
import vodx_pr88
from vodx_pr88 import apply_layout, baseline_snap_ids, layout_for, rows_from_frames

_LAYOUTS = Path(__file__).resolve().parents[1] / "tools" / "layouts.json"


def _lex() -> Lexicon:
    return Lexicon(formations=["Gun Bunch"], plays=["Inside Zone"], coverages=["Cover 3"])


def _bug(text: str, x: float, y: float, conf: float = 0.9) -> list:
    return [text, conf, x, y]


def _frame(t: float, tokens: list) -> dict:
    return {"t": t, "g": 0.62, "k": tokens}


class LayoutCountTests(unittest.TestCase):
    def test_layouts_file_has_the_box_masks_and_windows(self) -> None:
        data = json.loads(_LAYOUTS.read_text(encoding="utf-8"))
        videos = data["videos"]
        self.assertGreaterEqual(sum(1 for body in videos.values() if body.get("gameplay_window")), 268)
        self.assertGreaterEqual(sum(1 for body in videos.values() if body.get("mask")), 190)
        self.assertEqual(videos["YJZksbAu6jE"]["gameplay_window"], [138, 4247])
        self.assertEqual(videos["D1xGdV6Yv2I"]["gameplay_window"], [13600, 18545])

    def test_no_flag_count_matches_baseline_after_window_and_mask(self) -> None:
        data = json.loads(_LAYOUTS.read_text(encoding="utf-8"))
        cases = {
            "YJZksbAu6jE": (10.0, 200.0, 4300.0),
            "D1xGdV6Yv2I": (1000.0, 14000.0, 19000.0),
        }
        for video_id, (before, during, after) in cases.items():
            layout = layout_for({"video_id": video_id, "creator": ""}, data)
            frames = [
                _frame(before, [_bug("1ST810", 0.5, 0.93), _bug("1st 5:00", 0.2, 0.2)]),
                _frame(before + 8, [_bug("1ST810", 0.5, 0.93)]),
                _frame(
                    during,
                    [
                        _bug("1ST810", 0.5, 0.93),
                        _bug("1st 4:10", 0.2, 0.2),
                        _bug("CHAT 1ST810", 0.8, 0.5, 0.98),
                    ],
                ),
                _frame(during + 8, [_bug("1ST810", 0.5, 0.93)]),
                _frame(after, [_bug("2ND810", 0.5, 0.93), _bug("1st 1:00", 0.2, 0.2)]),
            ]
            laid = apply_layout(frames, layout)
            lex = _lex()
            flags_off = rows_from_frames(
                laid, video_id=video_id, channel="box", game="madden27", extract=vodx_pr88.flags_from_fixes(""), lexicon=lex
            )
            baseline = baseline_snap_ids(laid, video_id=video_id, channel="box", game="madden27", lexicon=lex)
            self.assertEqual([row["snap_id"] for row in flags_off], baseline, video_id)
            self.assertTrue(all(during <= row["t_start"] < after for row in flags_off), video_id)
            raw = rows_from_frames(
                frames, video_id=video_id, channel="box", game="madden27", extract=vodx_pr88.flags_from_fixes(""), lexicon=lex
            )
            self.assertGreater(len(raw), len(flags_off), video_id)
        yjz = layout_for({"video_id": "YJZksbAu6jE"}, data)
        ticker = apply_layout(
            [_frame(200, [_bug("1ST810", 0.5, 0.93), _bug("CHAT 1ST810", 0.5, 0.98, 0.8)])],
            yjz,
        )
        self.assertEqual([token[0] for token in ticker[0]["k"]], ["1ST810"])

    def test_mask_drops_a_replay_token_on_the_cfb_facecam(self) -> None:
        data = json.loads(_LAYOUTS.read_text(encoding="utf-8"))
        layout = layout_for({"video_id": "hGyLMPWurMg"}, data)
        frame = _frame(
            4000,
            [
                _bug("1ST810", 0.5, 0.9),
                _bug("REPLAY", 0.95, 0.05, 0.8),
            ],
        )
        laid = apply_layout([frame], layout)
        texts = [token[0] for token in laid[0]["k"]]
        self.assertIn("1ST810", texts)
        self.assertNotIn("REPLAY", texts)


if __name__ == "__main__":
    unittest.main()
