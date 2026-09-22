"""GameObservation normalize + DefenseLook bridge + confidence thresholds."""

from __future__ import annotations

import unittest

from cfb_coach.vision.observation import GameObservation, SituationHUD


class TestGameObservation(unittest.TestCase):
    def test_normalize_aliases(self) -> None:
        obs = GameObservation(
            formation="bunch_r",
            shell="1h",
            pressure="l",
            confidence={"formation": 0.9, "shell": 0.9, "pressure": 0.9},
        )
        n = obs.normalized()
        self.assertEqual(n.formation, "bunch")
        self.assertEqual(n.shell, "one_high")
        self.assertEqual(n.pressure, "left")

    def test_low_confidence_becomes_unknown(self) -> None:
        obs = GameObservation(
            formation="trips",
            shell="two_high",
            pressure="show",
            confidence={"formation": 0.1, "shell": 0.2, "pressure": 0.1},
        )
        n = obs.normalized(conf_threshold=0.35)
        self.assertEqual(n.formation, "unknown")
        self.assertEqual(n.shell, "unknown")
        self.assertEqual(n.pressure, "unknown")

    def test_to_defense_look_shell_pressure(self) -> None:
        obs = GameObservation(
            shell="one_high",
            pressure="left",
            confidence={"shell": 0.8, "pressure": 0.8},
            source="capture",
        )
        look = obs.to_defense_look()
        self.assertEqual(look.shell, "single_high")
        self.assertEqual(look.pressure, "blitz_left")

    def test_short_line(self) -> None:
        obs = GameObservation(
            situation=SituationHUD(down=3, distance=7),
            formation="bunch",
            formation_side="R",
            shell="two_high",
            pressure="left",
            confidence={"formation": 0.9, "shell": 0.9, "pressure": 0.9},
        )
        line = obs.short_line()
        self.assertIn("3&7", line)
        self.assertIn("BUNCH R", line)
        self.assertIn("2-HIGH", line)
        self.assertIn("PRESSURE L", line)


if __name__ == "__main__":
    unittest.main()
