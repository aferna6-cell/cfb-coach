"""Sprint 6B: model-primary offense and explicit offensive action provenance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.last_snap import LastSnapBook
from cfb_coach.live_server import LivePlayController, render_live_html
from cfb_coach.madden.model import experimental_live, inference
from cfb_coach.madden.model.offense_action_policy import choose_offense_action
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import Situation


def _seed() -> dict:
    return {"opponents": {
        "cpu": {
            "display_name": "CPU", "team_now": "DET", "skill": "cpu",
            "confidence": "low", "profile_json": "{}"
        }
    }}


class ModelPrimaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "test.db", seed=_seed())
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        self.sit = Situation(raw="2&8 my 35", side="offense", down=2, distance=8, yardline=35)
        self.book = {"Gun Bunch": ["Mesh", "Flood", "Inside Zone"]}

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _ranked(self, play: str) -> list[dict]:
        return [{
            "formation": "Gun Bunch", "play": play, "probability": 0.52,
            "uncertainty": 0.7, "play_concept": "flood",
            "evidence_quality": "prior_driven",
        }]

    def _artifact(self) -> SimpleNamespace:
        return SimpleNamespace(
            evidence_quality="prior_driven",
            model_version="test.primary", knowledge_version="test",
            knowledge_origin="cache", data_version="test",
        )

    def test_ml_overrides_heuristic_even_when_score_margin_is_small(self) -> None:
        heuristic = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "r1", "heuristic")
        rebuilt = MaddenCall("offense", "Gun Bunch", "Flood", "No adj", "r2", "model")
        with (
            mock.patch.object(experimental_live, "situational_offense_candidates",
                              return_value=([("Gun Bunch", "Mesh"), ("Gun Bunch", "Flood")],
                                            {("Gun Bunch", "Mesh"): 10.0})),
            mock.patch.object(experimental_live, "resolve_artifact_path",
                              return_value=Path("test.json")),
            mock.patch.object(experimental_live.exp_mod, "load_artifact",
                              return_value=self._artifact()),
            mock.patch.object(experimental_live.exp_mod, "rank_candidates",
                              return_value=self._ranked("Flood")) as rank,
            mock.patch.object(experimental_live, "rebuild_offense_attachments",
                              return_value=rebuilt) as rebuild,
        ):
            selected, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heuristic, sit=self.sit, opponent_id="cpu",
                db=self.db, book=self.book, commit=False,
            )
        self.assertIs(selected, rebuilt)
        self.assertEqual(dec.final_pick.play, "Flood")
        self.assertEqual(dec.effective_mode, CoachingMode.EXPERIMENTAL)
        kwargs = rank.call_args.kwargs
        self.assertIsNone(kwargs["heuristic"])
        self.assertEqual(kwargs["heuristic_bonuses"], {})
        self.assertEqual(kwargs["near_tie_margin"], 0.0)
        self.assertTrue(rebuild.call_args.kwargs["model_action_policy"])

    def test_model_rebuilds_even_on_same_play(self) -> None:
        heuristic = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "r1", "heuristic")
        rebuilt = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "r2", "model")
        with (
            mock.patch.object(experimental_live, "situational_offense_candidates",
                              return_value=([("Gun Bunch", "Mesh")], {})),
            mock.patch.object(experimental_live, "resolve_artifact_path",
                              return_value=Path("test.json")),
            mock.patch.object(experimental_live.exp_mod, "load_artifact",
                              return_value=self._artifact()),
            mock.patch.object(experimental_live.exp_mod, "rank_candidates",
                              return_value=self._ranked("Mesh")),
            mock.patch.object(experimental_live, "rebuild_offense_attachments",
                              return_value=rebuilt) as rebuild,
        ):
            selected, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heuristic, sit=self.sit, opponent_id="cpu",
                db=self.db, book=self.book,
            )
        self.assertIs(selected, rebuilt)
        self.assertIsNot(selected, heuristic)
        self.assertEqual(dec.effective_mode, CoachingMode.EXPERIMENTAL)
        rebuild.assert_called_once()

    def test_invalid_model_play_triggers_heuristic_fallback(self) -> None:
        heuristic = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "r1", "heuristic")
        with (
            mock.patch.object(experimental_live, "situational_offense_candidates",
                              return_value=([("Gun Bunch", "Mesh")], {})),
            mock.patch.object(experimental_live, "resolve_artifact_path",
                              return_value=Path("test.json")),
            mock.patch.object(experimental_live.exp_mod, "load_artifact",
                              return_value=self._artifact()),
            mock.patch.object(experimental_live.exp_mod, "rank_candidates",
                              return_value=self._ranked("OUT OF BOOK")),
        ):
            selected, dec = experimental_live.apply_experimental_offense(
                heuristic_call=heuristic, sit=self.sit, opponent_id="cpu",
                db=self.db, book=self.book,
            )
        self.assertIs(selected, heuristic)
        self.assertEqual(dec.effective_mode, CoachingMode.HEURISTIC)
        self.assertTrue(dec.fell_back)


class ActionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sit = Situation(raw="2&8", side="offense", down=2, distance=8)
        self.book = {"Gun Bunch": ["Mesh"]}
        self.pred = {"probability": 0.65, "uncertainty": 0.2}

    def test_no_invented_actions_or_last_snap_prediction(self) -> None:
        no_look = choose_offense_action(
            formation="Gun Bunch", play="Mesh", sit=self.sit,
            book=self.book, active=[], prediction=self.pred,
        )
        self.assertEqual(no_look["kind"], "none")

        self.sit.coverage_hint = "Cover 1"
        self.sit.coverage_source = "last"
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[{"id": "HR", "kind": "hot_route", "sources": ["research"],
                           "label": "Hot route", "buttons": "Y"}],
        ) as adj:
            no_prediction = choose_offense_action(
                formation="Gun Bunch", play="Mesh", sit=self.sit,
                book=self.book, active=[], prediction=self.pred,
                repeated=False,
            )
            adj.assert_not_called()
        self.assertEqual(no_prediction["kind"], "none")

    def test_model_ranks_supported_hot_route_not_guessed_audible(self) -> None:
        self.sit.coverage_hint = "Cover 1"
        self.sit.coverage_source = "live"
        with mock.patch(
            "cfb_coach.madden.adjustments.offense_adjustment_candidates",
            return_value=[
                {"id": "HR1", "kind": "hot_route", "sources": ["research"],
                 "label": "Hot route WR1", "buttons": "Y"},
                {"id": "AUDIBLE", "kind": "audible", "sources": ["research"],
                 "label": "Audible run", "buttons": "X"},
                {"id": "UNKNOWN", "kind": "hot_route", "sources": [],
                 "label": "Guess", "buttons": "VERIFY"},
            ],
        ):
            choice = choose_offense_action(
                formation="Gun Bunch", play="Mesh", sit=self.sit,
                book=self.book, active=[], prediction=self.pred,
            )
        self.assertEqual(choice["kind"], "adjustment")
        self.assertEqual(choice["id"], "HR1")
        self.assertIn("not_learned_action_effect", choice["scores_are"])

    def test_rejects_out_of_book_even_with_valid_research(self) -> None:
        choice = choose_offense_action(
            formation="Gun Bunch", play="NOT INSTALLED", sit=self.sit,
            book=self.book, active=["MAN"], prediction=self.pred,
        )
        self.assertEqual(choice["kind"], "none")


class ActionVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "actions.db", seed=_seed())

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _snap(self, approved: bool | None) -> dict:
        book = LastSnapBook(
            db=self.db, opponent_id="cpu",
            parse_situation=parse_madden_situation, session_id="game",
        )
        call = MaddenCall(
            "offense", "Gun Bunch", "Mesh", "No adj", "read", "rationale",
            macro="MAN",
        )
        sit = Situation(raw="1&10", side="offense", down=1, distance=10, yardline=25)
        book.remember_call(call, sit)
        return book.close_from_form(
            "+7", executed_status="used_recommended",
            applied_recommended_macro=approved,
        )

    def test_play_confirmation_does_not_automatically_verify_macro(self) -> None:
        closed = self._snap(False)
        self.assertEqual(closed["executed_status"], "identified")
        self.assertEqual(closed["executed_verification"], "verified")
        self.assertIsNone(closed["executed_macro"])

    def test_explicit_applied_action_can_record_macro(self) -> None:
        closed = self._snap(True)
        self.assertEqual(closed["executed_macro"], "MAN")

    def test_madden_drawer_collapsed_and_confirmation_unchecked(self) -> None:
        ctrl = LivePlayController(
            db=self.db, opponent_id="cpu",
            make_call=lambda sit, **kwargs: MaddenCall(
                "offense", "Gun Bunch", "Mesh", "No adj", "", "",
                macro="MAN", macro_info={
                    "id": "MAN", "name": "MAN", "buttons": "LB → MAN",
                }
            ),
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise", cpu_only=True,
            enable_execution_verify=True,
        )
        ctrl.start()
        ctrl.call_only("1&10 my 25")
        html = render_live_html(ctrl)
        self.assertIn('id="offense-action-panel" hidden', html)
        self.assertIn('id="action-applied"', html)
        self.assertIn("applied_recommended_action:", html)
        self.assertIn("Gun Bunch — Mesh", html)


if __name__ == "__main__":
    unittest.main()
