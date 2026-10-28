"""Live offense inference stays inside the budget on a synthetic book.

The fixture is generated in the test. It is not a game log, a database, or an
audit row. The p95 assertion uses four times the 150 ms budget so a slow CI
runner does not flake. Set ``CFB_COACH_SKIP_LATENCY_BENCH=1`` to skip it.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.experimental_live import (
    apply_experimental_offense,
    set_artifact_path,
)
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS
from cfb_coach.madden.offense_macros import pairs_in_book
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.situation import Situation

# Four times the live budget. The per-play book scan this replaces is slower
# than this bound on the synthetic book below; a correct call is not.
_GENEROUS_P95_MS = ML_LATENCY_BUDGET_MS * 4
_MACROS = [
    "MATCH", "C2", "C3", "MAN", "O-RUN", "RZ", "SHOT", "ZERO", "PROT", "O-HEAT", "O-RPO",
]
_PLAYS = [
    "Mesh", "Inside Zone", "Flood", "Four Verticals", "HB Slip Screen",
    "PA Boot", "Stick", "Smash", "Slants", "HB Dive", "Power O", "Stretch",
    "Y Cross", "Curl Flat", "Spacing", "Bench",
]


def _book() -> dict[str, list[str]]:
    return {f"Gun Form {index}": list(_PLAYS) for index in range(15)}


def _rows(book: dict[str, list[str]]) -> list[dict]:
    flat = [(form, play) for form, plays in book.items() for play in plays]
    rows = []
    for index in range(52):
        form, play = flat[index % len(flat)]
        rows.append({
            "side": "offense",
            "success": "true" if index % 3 else "false",
            "action_play": play,
            "action_formation": form,
            "down": (index % 4) + 1,
            "distance": 10,
            "yardline": 25 + (index % 50),
            "opponent_type": "cpu",
            "opponent_id": "cpu",
            "coverage_seen": "Cover 3",
            "supervised_eligible": True,
        })
    return rows


def _situation(index: int) -> Situation:
    if index % 5 == 0:
        return Situation(
            raw="3rd and 12", down=3, distance=12, yardline=45, side="offense",
            long_yardage=True, coverage_hint="Cover 3", coverage_source="live",
        )
    if index % 5 == 1:
        return Situation(
            raw="1st and 10 cover 3", down=1, distance=10, yardline=35, side="offense",
            coverage_hint="Cover 3", coverage_source="live",
        )
    return Situation(raw="2nd and 7", down=2, distance=7, yardline=40, side="offense")


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[rank]


class OffenseInferenceLatencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(
            Path(self.tmp.name) / "coach.db",
            seed={"opponents": {
                "cpu": {
                    "display_name": "CPU",
                    "team_now": "DET",
                    "skill": "cpu",
                    "confidence": "low",
                    "profile_json": "{}",
                },
            }},
        )
        self.book = _book()
        artifact = experimental_model.train_experimental(_rows(self.book), side="offense")
        self.artifact_path = Path(self.tmp.name) / "offense.json"
        experimental_model.save_artifact(artifact, self.artifact_path)
        set_artifact_path(self.db, self.artifact_path)
        self.heuristic = MaddenCall(
            "offense", "Gun Form 0", _PLAYS[0], "No adj", "Primary", "heuristic",
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_macro_pairs_are_scanned_once_per_macro(self) -> None:
        ranked = [
            {
                "formation": form,
                "play": play,
                "probability": 0.5,
                "uncertainty": 0.4,
                "evidence_quality": "empirical",
                "play_concept": "mesh",
                "selection_score": 0.5,
            }
            for form, plays in self.book.items()
            for play in plays
        ]
        with mock.patch(
            "cfb_coach.madden.offense_macros.pairs_in_book",
            wraps=pairs_in_book,
        ) as scanned:
            choose_joint_action(
                ranked=ranked,
                anchor=ranked[0],
                sit=_situation(1),
                book=self.book,
                active=_MACROS,
                db=self.db,
                opponent_id="cpu",
            )
        # One full-book scan per armed macro, plus the detail sheet's short
        # scans. Not one scan per play.
        self.assertGreater(scanned.call_count, 0)
        self.assertLessEqual(scanned.call_count, len(_MACROS) * 5)
        self.assertLess(scanned.call_count, len(ranked))

    @unittest.skipIf(
        os.environ.get("CFB_COACH_SKIP_LATENCY_BENCH") == "1",
        "latency bench skipped on slow CI",
    )
    def test_synthetic_p95_stays_under_a_generous_budget(self) -> None:
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        handler = logging.Handler()
        messages: list[str] = []

        def _emit(record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

        handler.emit = _emit  # type: ignore[method-assign]
        log = logging.getLogger("cfb_coach.madden.model.inference")
        previous = log.level
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        samples: list[float] = []
        try:
            for index in range(21):
                _call, decision = apply_experimental_offense(
                    heuristic_call=self.heuristic,
                    sit=_situation(index),
                    opponent_id="cpu",
                    db=self.db,
                    book=self.book,
                    active=_MACROS,
                    session_id="synthetic-game",
                    snap_seq=index + 1,
                    commit=False,
                )
                info = getattr(_call, "ml_experimental", None) or {}
                self.assertIn("latency_ms", info)
                self.assertIn("timed_out", info)
                self.assertIn("slowest_stage", info)
                timed_out = bool(
                    decision.fell_back
                    and decision.fallback_reason is not None
                    and decision.fallback_reason.value == "timeout"
                )
                self.assertEqual(info["timed_out"], timed_out)
                self.assertFalse(info["timed_out"])
                self.assertFalse(timed_out)
                if index == 0:
                    continue
                self.assertIsNotNone(decision.latency_ms_model)
                samples.append(float(decision.latency_ms_model))
        finally:
            log.removeHandler(handler)
            log.setLevel(previous)
        self.assertEqual(len(messages), 21)
        for line in messages:
            self.assertIn("latency_ms=", line)
            self.assertIn("timed_out=false", line)
            self.assertIn("slowest_stage=", line)
        p95 = _p95(samples)
        self.assertLess(p95, _GENEROUS_P95_MS)


class ModelCacheKeyTests(unittest.TestCase):
    def test_inode_changes_the_cache_key_when_mtime_and_size_do_not(self) -> None:
        from cfb_coach.madden.model.inference import (
            cached_parsed_model,
            clear_parsed_model_cache,
            file_revision,
        )

        clear_parsed_model_cache()
        tmp = tempfile.TemporaryDirectory()
        try:
            path = Path(tmp.name) / "model.json"
            path.write_text('{"v":1}', encoding="utf-8")
            before = path.stat()
            first = file_revision(path)
            self.assertIsNotNone(first)
            self.assertEqual(len(first[0]), 4)
            self.assertEqual(first[0][3], before.st_ino)
            loads = {"n": 0}

            def loader() -> str:
                loads["n"] += 1
                return path.read_text(encoding="utf-8")

            self.assertEqual(cached_parsed_model(("synthetic",) + first, loader), '{"v":1}')
            self.assertEqual(cached_parsed_model(("synthetic",) + first, loader), '{"v":1}')
            self.assertEqual(loads["n"], 1)
            replacement = Path(tmp.name) / "model-next.json"
            replacement.write_text('{"v":2}', encoding="utf-8")
            os.replace(replacement, path)
            os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))
            after = path.stat()
            second = file_revision(path)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(after.st_size, before.st_size)
            self.assertNotEqual(after.st_ino, before.st_ino)
            self.assertNotEqual(first, second)
            self.assertEqual(cached_parsed_model(("synthetic",) + second, loader), '{"v":2}')
            self.assertEqual(loads["n"], 2)
            self.assertEqual(cached_parsed_model(("synthetic",) + first, loader), '{"v":1}')
            self.assertEqual(loads["n"], 2)
        finally:
            clear_parsed_model_cache()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
