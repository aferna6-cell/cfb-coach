"""Formation from synthetic player coordinates — pure Python."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cfb_coach.vision.formation import classify_formation

FIX = Path(__file__).parent / "fixtures" / "player_coords.json"


class TestFormation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coords = json.loads(FIX.read_text(encoding="utf-8"))

    def test_bunch_r(self) -> None:
        form, side, conf = classify_formation(self.coords["bunch_r"])
        self.assertEqual(form, "bunch")
        self.assertEqual(side, "R")
        self.assertGreaterEqual(conf, 0.5)

    def test_trips_l(self) -> None:
        form, side, conf = classify_formation(self.coords["trips_l"])
        self.assertEqual(form, "trips")
        self.assertEqual(side, "L")

    def test_empty(self) -> None:
        form, side, conf = classify_formation(self.coords["empty"])
        self.assertEqual(form, "empty")

    def test_2x2(self) -> None:
        form, side, conf = classify_formation(self.coords["two_by_two"])
        self.assertEqual(form, "2x2")

    def test_too_few_unknown(self) -> None:
        form, side, conf = classify_formation([(0.5, 0.3)])
        self.assertEqual(form, "unknown")
        self.assertLess(conf, 0.35)


if __name__ == "__main__":
    unittest.main()
