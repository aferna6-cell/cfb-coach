"""Synthetic human games versus John.

No real franchise rows. The product default stays heuristic. These games opt
into experimental offense the same way ``play --ml-human-control`` does, and
they keep the established defense caller.
"""

from __future__ import annotations

import json
import random
import tempfile
import time
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import data as mdata
from cfb_coach.madden.human_opponents import ensure_madden_opponent
from cfb_coach.madden.macros import store_selection
from cfb_coach.madden.model import experimental_live, experimental_model, inference
from cfb_coach.madden.model.schema import CoachingMode, ML_LATENCY_BUDGET_MS
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.vod_model.flags import use_cli_flags

_ACTIVE = {
    "offense": ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "PROT"],
    "defense": [],
}
_BOOK = {
    "offense": {"Gun Bunch": ["Inside Zone"]},
    "defense": {"Nickel Over": ["Cover 4 Quarters", "Cover 3 Sky"]},
}
_CHAIN = [
    ("1&10 my 25 showing cover 4", "+5", "cover 3", False, None),
    ("2&6 my 40", "+3", "", False, None),
    ("3&2 opp 42", "+4", "", False, None),
    ("3&12 my 28 showing cover 2", "incomplete", "", False, None),
    ("1&10 opp 18", "+7", "", False, None),
    ("1&goal opp 3", "td", "", False, None),
    ("d 1&10", "incomplete", "mesh", False, None),
    ("d 2&7 my 40", "+2", "", False, None),
    ("d 3&8", "sack", "", False, None),
    ("4&1 opp 40", "+2", "", False, None),
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    cursor = (len(ordered) - 1) * fraction
    low = int(cursor)
    high = min(low + 1, len(ordered) - 1)
    weight = cursor - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def _tendencies(db: CoachDB, oid: str) -> list[tuple]:
    return [
        (row["bucket"], row["key"], int(row["count"]), int(row["success"]))
        for row in db.get_tendencies(oid)
    ]


class WildcardJohnGamesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "synthetic.db", seed=mdata.load_seed())
        ensure_madden_opponent(self.db, "john")
        store_selection(self.db, "john", _ACTIVE)
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.HEURISTIC)
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        artifact = experimental_model.train_experimental([], side="offense")
        dest = Path(self._td.name) / "offense.json"
        experimental_model.save_artifact(artifact, dest)
        experimental_live.set_artifact_path(self.db, dest)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)

    def tearDown(self) -> None:
        self.db.close()
        self._td.cleanup()

    def _make(self, sit, **kwargs):
        return make_call(
            sit, "john", self.db,
            playbook=_BOOK,
            active_macros=_ACTIVE,
            allow_human_ml=True,
            live_macros=True,
            rng=random.Random(0),
            last_coverage=kwargs.get("last_coverage"),
            last_concept=kwargs.get("last_concept"),
        )

    def _game(self, *, result_wl: str, score: str, quarter: int, verify_index: int | None, drive_index: int | None) -> dict:
        ctrl = LivePlayController(
            db=self.db,
            opponent_id="john",
            make_call=self._make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "synthetic",
            brand="Madden 27 Franchise",
            enable_execution_verify=True,
        )
        ctrl.start()
        wall: list[float] = []
        started = time.perf_counter()
        first = ctrl.call_only("1&10 my 30 showing cover 4")
        wall.append((time.perf_counter() - started) * 1000.0)
        self.assertTrue(first["ok"])
        started = time.perf_counter()
        second = ctrl.call_only("2&8 my 34")
        wall.append((time.perf_counter() - started) * 1000.0)
        self.assertTrue(second["ok"])
        for index, (sit_raw, outcome, their, _unused, _drive) in enumerate(_CHAIN):
            raw = sit_raw
            if drive_index is not None and index == drive_index:
                raw = f"{sit_raw} drive id WD2"
            started = time.perf_counter()
            logged = ctrl.result_and_call(
                outcome=outcome,
                sit_raw=raw,
                their=their,
                quarter=quarter,
                score_us=14,
                score_them=10,
                score_set=True,
                verify_touched=verify_index is not None and index == verify_index,
                executed_status="used_recommended" if verify_index is not None and index == verify_index else "unknown",
            )
            wall.append((time.perf_counter() - started) * 1000.0)
            self.assertTrue(logged["ok"], logged)
        started = time.perf_counter()
        # default_side sticks after the defensive series. Name offense so this
        # seal is an experimental decision with no later result.
        pending = ctrl.call_only("1&10 opp 22", side="offense")
        wall.append((time.perf_counter() - started) * 1000.0)
        self.assertTrue(pending["ok"])
        ended = ctrl.end_game(result_wl=result_wl, score=score)
        self.assertTrue(ended["ok"], ended)
        return {"session_id": ctrl.session_id, "wall_ms": wall}

    def test_three_synthetic_games_write_the_game2_fields(self) -> None:
        gavin_before = _tendencies(self.db, "gavin")
        tiano_before = _tendencies(self.db, "tiano")
        self.assertEqual(_tendencies(self.db, "john"), [])
        with use_cli_flags(no_vod_prior=True):
            games = [
                self._game(result_wl="win", score="24-17", quarter=1, verify_index=None, drive_index=None),
                self._game(result_wl="loss", score="14-21", quarter=4, verify_index=1, drive_index=1),
                self._game(result_wl="win", score="17-14", quarter=4, verify_index=None, drive_index=None),
            ]
        self.assertEqual(len(games), 3)
        sessions = list(self.db.conn.execute(
            "SELECT session_id, ended_ts, result_wl, score, play_count FROM game_sessions ORDER BY started_ts"
        ))
        self.assertEqual(len(sessions), 3)
        for row in sessions:
            self.assertTrue(row["ended_ts"])
            self.assertIn(row["result_wl"], {"win", "loss"})
            self.assertTrue(row["score"])
            self.assertGreater(int(row["play_count"]), 0)

        decisions = list(self.db.conn.execute(
            "SELECT id, decision_json, orphan_reason, shadow_status, latency_ms_model "
            "FROM ml_decisions ORDER BY id"
        ))
        self.assertGreater(len(decisions), 0)
        macros = []
        model_ms: list[float] = []
        timeouts = 0
        reasons = set()
        for row in decisions:
            payload = json.loads(row["decision_json"] or "{}")
            macro = payload.get("macro") or {}
            if isinstance(macro, dict) and macro.get("macro_id"):
                macros.append(macro["macro_id"])
            if row["latency_ms_model"] is not None:
                model_ms.append(float(row["latency_ms_model"]))
            if row["shadow_status"] == "timeout" or payload.get("fallback_reason") == "timeout":
                timeouts += 1
                self.assertTrue(payload.get("fell_back"))
            if row["orphan_reason"]:
                reasons.add(row["orphan_reason"])
            if payload.get("inference_budget_ms") is not None:
                self.assertEqual(int(payload["inference_budget_ms"]), 150)
            if row["latency_ms_model"] is not None and float(row["latency_ms_model"]) > 150:
                self.assertTrue(payload.get("fell_back"))
        self.assertTrue(macros, "a qualifying macro was not written on decision_json")
        self.assertIn("superseded", reasons)
        self.assertIn("no_result", reasons)

        snaps = list(self.db.conn.execute("SELECT accepted, drive_id, situation_raw FROM snaps"))
        self.assertGreater(len(snaps), 0)
        explicit = [row for row in snaps if row["drive_id"] and not str(row["drive_id"]).startswith("auto-d")]
        plain = [row for row in snaps if not row["drive_id"]]
        inferred = [row for row in snaps if str(row["drive_id"] or "").startswith("auto-d")]
        self.assertTrue(inferred or plain)
        self.assertTrue(explicit)
        self.assertTrue(all("drive id" in (row["situation_raw"] or "").lower() for row in explicit))
        self.assertTrue(all(row["drive_id"] is None for row in plain))
        # Inferred ids carry the auto- prefix and never pose as explicit ones.
        outcomes = list(self.db.conn.execute("SELECT accepted FROM ml_outcomes"))
        accepted = [row["accepted"] for row in outcomes]
        self.assertIn("unknown", accepted)
        self.assertIn("true", accepted)
        self.assertTrue(all(value in {"unknown", "true", "false"} for value in accepted if value is not None))
        unknown = [value for value in accepted if value == "unknown"]
        self.assertGreater(len(unknown), len(accepted) - len(unknown))

        john_rows = _tendencies(self.db, "john")
        self.assertTrue(any(
            bucket == "their_coverage" and "cover 3" in key.lower()
            for bucket, key, _count, _success in john_rows
        ))
        self.assertFalse(any(bucket.endswith("_seed") for bucket, _key, _count, _success in john_rows))
        self.assertEqual(_tendencies(self.db, "gavin"), gavin_before)
        self.assertEqual(_tendencies(self.db, "tiano"), tiano_before)

        wall = [sample for game in games for sample in game["wall_ms"]]
        self.assertTrue(wall)
        report = {
            "wall_ms": {
                "p50": round(_percentile(wall, 0.50), 3),
                "p95": round(_percentile(wall, 0.95), 3),
                "max": round(max(wall), 3),
            },
            "model_ms": {
                "p50": round(_percentile(model_ms, 0.50), 3) if model_ms else None,
                "p95": round(_percentile(model_ms, 0.95), 3) if model_ms else None,
                "max": round(max(model_ms), 3) if model_ms else None,
                "n": len(model_ms),
            },
            "timeouts": timeouts,
            "macros": sorted(set(macros)),
            "games": len(sessions),
            "decisions": len(decisions),
            "budget_ms": ML_LATENCY_BUDGET_MS,
        }
        print("WILDCARD_JOHN_LATENCY " + json.dumps(report, sort_keys=True))
        self.assertEqual(report["budget_ms"], 150)
        self.assertGreaterEqual(report["timeouts"], 0)
