"""Sprint 6A: the in-game coaching pad shows only formation and play.

Full coaching explanations must remain available in API state and SQLite for
analysis, but must not crowd the live call. Outcome entry remains functional.
"""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, _call_main, render_live_html
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.model import experimental_live, inference
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.situation import Situation
from cfb_coach.madden.situation import parse_madden_situation


def _seed() -> dict:
    return {
        "opponents": {
            "cpu": {
                "display_name": "CPU",
                "team_now": "DET",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            }
        }
    }


class CompactLiveCoachTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.temp.name) / "coach.db", seed=_seed())
        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="cpu",
            make_call=lambda sit, **kwargs: MaddenCall(
                "offense", "Gun Doubles Clamp Stack", "Texas Y-Stutter Wheel",
                "No adj", "Read 1 -> Read 2", "ML experimental explanation",
            ),
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            cpu_only=True,
            enable_execution_verify=True,
        )

    def tearDown(self) -> None:
        self.db.close()
        self.temp.cleanup()

    def test_only_final_formation_and_play_are_displayed(self) -> None:
        self.ctrl.last_call = MaddenCall(
            "offense", "Gun Doubles Clamp Stack", "Texas Y-Stutter Wheel",
            "Hot route", "Read 1 -> Read 2", "ML probability=0.55",
        )
        self.ctrl.call_text = "PLAY: long diagnostic text with ML explanations"
        self.assertEqual(
            _call_main(self.ctrl),
            "Gun Doubles Clamp Stack — Texas Y-Stutter Wheel",
        )
        html = render_live_html(self.ctrl)
        self.assertIn(
            '<div class="call" id="call" aria-live="polite">'
            "Gun Doubles Clamp Stack — Texas Y-Stutter Wheel</div>",
            html,
        )
        self.assertIn("const rec = st.pending_recommendation;", html)
        self.assertIn('renderMacro(null);', html)
        self.assertIn('<div class="heard" id="ml-experimental" hidden>', html)
        self.assertIn('<details id="live-admin">', html)

    def test_logging_and_verification_controls_remain_usable(self) -> None:
        html = render_live_html(self.ctrl)
        self.assertIn('id="btn-submit"', html)
        self.assertIn('id="btn-undo"', html)
        self.assertIn('id="btn-call-only"', html)
        self.assertIn('value="used_recommended" checked', html)
        self.assertIn('id="exec-diff"', html)
        self.assertIn('id="btn-end"', html)

    def test_no_pending_and_ended_states(self) -> None:
        self.assertEqual(_call_main(self.ctrl), "Waiting for next play")
        self.ctrl.ended = True
        self.assertEqual(_call_main(self.ctrl), "GAME OVER")

    def test_human_experimental_control_requires_per_session_opt_in(self) -> None:
        """CPU behavior unchanged; human experimental must be explicitly opted in."""
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        heuristic = MaddenCall(
            "offense", "Gun Bunch", "Mesh", "No adj", "r1", "heuristic"
        )
        ml = MaddenCall(
            "offense", "Gun Doubles", "Flood", "No adj", "r2", "experimental"
        )
        sit = Situation(raw="2&8", side="offense", down=2, distance=8)
        book = {"Gun Bunch": ["Mesh"], "Gun Doubles": ["Flood"]}
        with mock.patch(
            "cfb_coach.madden.model.experimental_live.apply_experimental_offense",
            return_value=(ml, object()),
        ) as apply:
            no_opt = experimental_live.maybe_apply_experimental(
                call=heuristic, sit=sit, opponent_id="gavin", db=self.db,
                book=book,
            )
            self.assertIs(no_opt, heuristic)
            apply.assert_not_called()

            opted_in = experimental_live.maybe_apply_experimental(
                call=heuristic, sit=sit, opponent_id="gavin", db=self.db,
                book=book, allow_human_ml=True,
            )
            self.assertIs(opted_in, ml)
            apply.assert_called_once()

            # Session flag is never written to the DB; subsequent no-flag call
            # must stay heuristic despite global experimental mode.
            apply.reset_mock()
            again = experimental_live.maybe_apply_experimental(
                call=heuristic, sit=sit, opponent_id="gavin", db=self.db,
                book=book,
            )
            self.assertIs(again, heuristic)
            apply.assert_not_called()

            cpu = experimental_live.maybe_apply_experimental(
                call=heuristic, sit=sit, opponent_id="cpu", db=self.db,
                book=book,
            )
            self.assertIs(cpu, ml)
            apply.assert_called_once()


if __name__ == "__main__":
    unittest.main()
