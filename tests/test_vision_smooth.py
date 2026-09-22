"""Temporal smoother — pure Python."""

from __future__ import annotations

import unittest

from cfb_coach.vision.smooth import TemporalSmoother, majority_vote


class TestSmooth(unittest.TestCase):
    def test_requires_n_of_m(self) -> None:
        s = TemporalSmoother(window=5, threshold=3, fields=("shell",))
        # first frames adopt latest while history short
        self.assertEqual(s.update({"shell": "one_high"})["shell"], "one_high")
        self.assertEqual(s.update({"shell": "one_high"})["shell"], "one_high")
        # flicker to two_high once — should stay one_high once window settled
        s.update({"shell": "one_high"})
        out = s.update({"shell": "two_high"})
        self.assertEqual(out["shell"], "one_high")
        # three two_high votes flip
        s.update({"shell": "two_high"})
        out = s.update({"shell": "two_high"})
        self.assertEqual(out["shell"], "two_high")

    def test_majority_vote(self) -> None:
        self.assertEqual(majority_vote(["a", "a", "b"], threshold=2), "a")
        self.assertIsNone(majority_vote(["a", "b"], threshold=2))


if __name__ == "__main__":
    unittest.main()
