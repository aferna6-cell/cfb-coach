"""Model-path offensive calls carry Aidan's catalog macros.

The experimental coordinator picks the play. The adjustment is the same
Custom Adjustment the heuristic would arm for that play, from the real
catalog only. Synthetic situations — no franchise rows.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.model import experimental_live, experimental_model
from cfb_coach.madden.model.inference import exact_offense_line
from cfb_coach.madden.offense_macros import clean_ids, known, suggest_for_snap, xbox_name
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import Situation

# Eight confirmed offense Custom Adjustments. PROT is in this set; O-HEAT is
# not, so a live blitz on a mesh resolves to PROT rather than HEAT.
ACTIVE8 = ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "PROT"]
INVENTED = "NOT-A-REAL-MACRO"

_NONE = {
    "kind": "none",
    "id": None,
    "macro": None,
    "adjustment": None,
    "reason": "run the play unchanged",
}


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


def _sit(**kwargs) -> Situation:
    base = dict(
        raw="1&10",
        side="offense",
        down=1,
        distance=10,
        yardline=35,
        coverage_hint=None,
        coverage_source="none",
    )
    base.update(kwargs)
    return Situation(**base)


class ModelPathAdjustmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _rebuild(self, formation: str, play: str, sit: Situation, *, decision: dict | None = None, active: list[str] | None = None):
        heuristic = experimental_live.rebuild_offense_attachments(
            formation=formation,
            play=play,
            sit=sit,
            book={formation: [play]},
            active=list(ACTIVE8 if active is None else active),
            db=self.db,
            opponent_id="cpu",
            rationale="heuristic",
        )
        model = experimental_live.rebuild_offense_attachments(
            formation=formation,
            play=play,
            sit=sit,
            book={formation: [play]},
            active=list(ACTIVE8 if active is None else active),
            db=self.db,
            opponent_id="cpu",
            rationale="model",
            model_action_policy=True,
            preset_decision=dict(decision or _NONE),
        )
        return heuristic, model

    def test_catalog_validation_still_rejects_invented_names(self) -> None:
        self.assertEqual(clean_ids(ACTIVE8 + [INVENTED, "secret"]), ACTIVE8)
        self.assertTrue(all(known(mid) for mid in ACTIVE8))
        self.assertFalse(known(INVENTED))
        self.assertFalse(known(""))

    def test_model_path_matches_heuristic_macros(self) -> None:
        cases = (
            ("Gun Bunch", "Inside Zone", _sit(coverage_hint="Cover 4", coverage_source="live"), "O-RUN"),
            ("Gun Trips", "Deep Flood", _sit(coverage_hint="Cover 3", coverage_source="live"), "C3"),
            ("Gun Bunch", "Mesh Spot", _sit(coverage_hint="Blitz", coverage_source="live", down=2, distance=8), "PROT"),
        )
        for formation, play, sit, expected in cases:
            with self.subTest(play=play, expected=expected):
                heuristic, model = self._rebuild(formation, play, sit)
                direct = suggest_for_snap(
                    zone="open",
                    play=play,
                    coverage=sit.coverage_hint,
                    coverage_source=sit.coverage_source,
                    active=ACTIVE8,
                    down=sit.down,
                    book={formation: [play]},
                )
                self.assertEqual(heuristic.macro, expected)
                self.assertEqual(model.macro, expected)
                self.assertEqual(direct["id"], expected)
                self.assertIn(model.macro, ACTIVE8)
                self.assertTrue(known(model.macro))
                self.assertEqual(model.macro_state, "armed")
                self.assertIsNone(model.adjustment)
                self.assertIn(f"MACRO: {xbox_name(expected)}", model.format())
                line = exact_offense_line(
                    model.formation,
                    model.play,
                    None if model.adj_or_macro == "No adj" else model.adj_or_macro,
                    "Primary",
                    "Checkdown",
                    macro_name=model.macro,
                    macro_state=model.macro_state,
                )
                self.assertIn(f"| {expected} |", line)
                self.assertNotIn("No adj", line)
                action = model.ml_offense_action
                self.assertEqual(action["kind"], "macro")
                self.assertEqual(action["id"], expected)

    def test_live_pad_receives_the_model_macro(self) -> None:
        _heuristic, model = self._rebuild(
            "Gun Bunch",
            "Inside Zone",
            _sit(coverage_hint="Cover 4", coverage_source="live"),
        )
        ctrl = LivePlayController(
            db=self.db,
            opponent_id="cpu",
            make_call=lambda sit, **kwargs: model,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            cpu_only=True,
        )
        ctrl.last_call = model
        pending = ctrl._pending_offense_action()
        self.assertIsNotNone(pending)
        self.assertEqual(pending["kind"], "macro")
        self.assertEqual(pending["id"], "O-RUN")
        self.assertEqual(ctrl._pending_recommendation()["macro"], "O-RUN")

    def test_invented_macro_never_replaces_or_survives(self) -> None:
        invented = {
            "kind": "macro",
            "id": INVENTED,
            "reason": "made up",
            "macro": {
                "id": INVENTED,
                "name": INVENTED,
                "buttons": f"LB → {INVENTED}",
                "why": "not a real adjustment",
            },
            "adjustment": None,
        }
        _heuristic, armed = self._rebuild(
            "Gun Bunch",
            "Inside Zone",
            _sit(coverage_hint="Cover 4", coverage_source="live"),
            decision=invented,
        )
        self.assertEqual(armed.macro, "O-RUN")
        self.assertNotIn(INVENTED, armed.format())

        _heuristic, quiet = self._rebuild(
            "Gun Bunch",
            "Mesh Spot",
            _sit(),
            decision=invented,
        )
        self.assertIsNone(quiet.macro)
        self.assertEqual(quiet.adj_or_macro, "No adj")
        self.assertNotIn(INVENTED, quiet.format())
        self.assertEqual(quiet.ml_offense_action["kind"], "none")

    def test_no_macro_when_the_heuristic_would_not_arm_one(self) -> None:
        heuristic, model = self._rebuild("Gun Bunch", "Mesh Spot", _sit())
        self.assertIsNone(heuristic.macro)
        self.assertIsNone(model.macro)
        self.assertEqual(model.adj_or_macro, "No adj")
        line = exact_offense_line(
            model.formation, model.play, model.adj_or_macro, "Spot", "Drag",
            macro_name=model.macro, macro_state=getattr(model, "macro_state", None),
        )
        self.assertIn("| No adj |", line)

    def test_no_macros_flag_withholds_the_model_macro(self) -> None:
        sit = _sit(coverage_hint="Cover 3", coverage_source="live")
        sit.extras = {"live_macros": False}
        heuristic, model = self._rebuild("Gun Trips", "Deep Flood", sit)
        self.assertIsNone(heuristic.macro)
        self.assertIsNone(model.macro)

    def test_experimental_coordinator_call_carries_the_macro(self) -> None:
        book = {"Gun Bunch": ["Inside Zone"]}
        art = experimental_model.train_experimental([], side="offense")
        dest = Path(self.tmp.name) / "offense.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        sit = _sit(coverage_hint="Cover 4", coverage_source="live")
        heuristic = MaddenCall("offense", "Gun Bunch", "Inside Zone", "No adj", "Front → Cutback", "heuristic")
        call, decision = experimental_live.apply_experimental_offense(
            heuristic_call=heuristic,
            sit=sit,
            opponent_id="cpu",
            db=self.db,
            book=book,
            active=ACTIVE8,
            budget_ms=5000,
            commit=False,
        )
        self.assertFalse(decision.fell_back)
        self.assertEqual(call.formation, "Gun Bunch")
        self.assertEqual(call.play, "Inside Zone")
        self.assertEqual(call.macro, "O-RUN")
        self.assertTrue(known(call.macro))
        self.assertIn(f"MACRO: {xbox_name('O-RUN')}", call.format())
        shown = exact_offense_line(
            call.formation,
            call.play,
            None,
            "Front",
            "Cutback",
            macro_name=call.macro,
            macro_state=call.macro_state,
        )
        self.assertIn("| O-RUN |", shown)
        self.assertNotIn("No adj", shown)


if __name__ == "__main__":
    unittest.main()
