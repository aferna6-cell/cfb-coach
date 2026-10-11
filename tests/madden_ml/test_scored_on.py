"""Hybrid refuses a model whose training report is not reviewed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model import adapters, inference
from cfb_coach.madden.model.offense_postgame import evaluate_saved_games
from cfb_coach.madden.model.schema import CoachingMode, MLStatus, Possession, candidate_in_book
from cfb_coach.situation import Situation


def _artifact(scored: str | None) -> dict:
    payload = {"model_type": "intercept_prior", "weights": [], "intercept": 0.0}
    if scored is not None:
        payload["training_report"] = {"tags": {"scored_on": scored}}
    return payload


def _decision(mode: CoachingMode, artifact: dict | None):
    sit = Situation(raw="1&10 my 25", side="offense", down=1, distance=10, yardline=25)
    state, obs = adapters.situation_to_state(sit, game_id="g", snap_id="s")
    book = {"Gun Bunch": ["Mesh"]}
    cand = candidate_in_book(
        book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh",
    )
    return inference.rank_live(
        state,
        obs,
        [cand],
        None,
        None,
        (),
        mode=mode,
        heuristic_pick=cand,
        model_artifact=artifact,
        registry_dir="/tmp/madden-ml-no-registry",
    )


class TestScoredOn(unittest.TestCase):
    def test_hybrid_refused_for_unreviewed(self) -> None:
        decision = _decision(CoachingMode.HYBRID, _artifact("unreviewed"))
        self.assertEqual(decision.effective_mode, CoachingMode.HEURISTIC)
        self.assertTrue(decision.fell_back)
        self.assertEqual(
            decision.mode_downgrade_reason,
            "hybrid refused: scored_on is unreviewed",
        )

    def test_hybrid_refused_for_missing(self) -> None:
        decision = _decision(CoachingMode.HYBRID, _artifact(None))
        self.assertEqual(decision.effective_mode, CoachingMode.HEURISTIC)
        self.assertTrue(decision.fell_back)
        self.assertEqual(
            decision.mode_downgrade_reason,
            "hybrid refused: scored_on is unknown",
        )
        self.assertEqual(inference.read_scored_on({}), "unknown")
        self.assertEqual(inference.read_scored_on(None), "unknown")

    def test_shadow_ok_for_unreviewed(self) -> None:
        decision = _decision(CoachingMode.SHADOW, _artifact("unreviewed"))
        self.assertEqual(decision.mode, CoachingMode.SHADOW)
        self.assertFalse(decision.fell_back)
        self.assertNotIn("hybrid refused", decision.mode_downgrade_reason or "")
        self.assertIn(
            decision.shadow_status,
            (MLStatus.OK, MLStatus.LOW_CONFIDENCE),
        )
        self.assertEqual(decision.final_pick.play, "Mesh")

    def test_postgame_and_live_line_show_scored_on(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "offense.json"
            artifact.write_text(
                json.dumps(_artifact("reviewed")),
                encoding="utf-8",
            )
            db = CoachDB(root / "madden27.db")
            try:
                db.set_meta("ml_experimental_artifact", str(artifact))
                report = evaluate_saved_games(db)
            finally:
                db.close()
        self.assertEqual(report["scored_on"], "reviewed")

        class _Call:
            def format(self) -> str:
                return "Gun Bunch — Mesh | No adj | Spot → Drag"

            ml_experimental = {
                "explanation": "heuristic kept",
                "evidence_quality": "prior_driven",
                "heuristic_formation": "Gun Bunch",
                "heuristic_play": "Mesh",
                "ml_formation": "Gun Bunch",
                "ml_play": "Mesh",
                "final_formation": "Gun Bunch",
                "final_play": "Mesh",
                "scored_on": "unreviewed",
            }

        text = LivePlayController._format_call_text(_Call())
        self.assertIn("model scored_on unreviewed", text)

        class _Ctrl:
            last_call = _Call()
            ended = False

        state = LivePlayController._ml_experimental_state(_Ctrl())
        self.assertEqual(state["scored_on"], "unreviewed")
