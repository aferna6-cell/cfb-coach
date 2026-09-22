"""Shell / pressure classifiers from synthetic coords."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cfb_coach.vision.pressure import classify_pressure, classify_shell

FIX = Path(__file__).parent / "fixtures" / "player_coords.json"


class TestPressureShell(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coords = json.loads(FIX.read_text(encoding="utf-8"))

    def test_two_high(self) -> None:
        shell, conf = classify_shell(self.coords["defense_two_high"])
        self.assertEqual(shell, "two_high")
        self.assertGreaterEqual(conf, 0.5)

    def test_pressure_unknown_few(self) -> None:
        p, conf = classify_pressure([(0.5, 0.5)])
        self.assertEqual(p, "unknown")


if __name__ == "__main__":
    unittest.main()
