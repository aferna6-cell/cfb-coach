"""Smoke test for the coordinator demo script."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_REPORT = _ROOT / "docs" / "madden_ml" / "coordinator_demo_report.md"
_SCRIPT = _ROOT / "scripts" / "madden_coordinator_demo.py"


class TestCoordinatorDemo(unittest.TestCase):
    def test_demo_script_writes_the_report(self) -> None:
        checked_in_before = _REPORT.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder) / "coordinator_demo_report.md"
            completed = subprocess.run(
                [sys.executable, str(_SCRIPT), "--report", str(report_path)],
                cwd=_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            report = report_path.read_text(encoding="utf-8")
        self.assertEqual(report, completed.stdout)
        self.assertEqual(_REPORT.read_bytes(), checked_in_before)
        checked_in = checked_in_before.decode("utf-8")
        self.assertIn("Calls timed: 50.", checked_in)
        self.assertIn("TEST FIXTURE - not an Aidan-verified macro", report)
        self.assertIn("Offense lines unchanged: 5/5.", report)
        self.assertIn("Draft line (macro absent):", report)
        self.assertIn("Armed line:", report)
        self.assertIn("ml.latency_budget_ms stays 150.", report)
        self.assertIn("Calls timed: 50.", report)
        self.assertIn("Formation — Play | Adjustment | Read 1 → Read 2", report)
        self.assertIn("Formation — Play | Adjustment/Macro | User responsibility", report)
        self.assertIn("Dime 3-2 Odd", report)
        self.assertIn("DB Fire 2", report)
        self.assertIn("Pistol Trips", report)
        self.assertIn("HB Dive", report)
        self.assertIn("No VOD prior is enabled", report)
        self.assertIn("synthetic", report)
        self.assertIn("MLStatus counts:", report)
        self.assertIn("Fallback count:", report)
