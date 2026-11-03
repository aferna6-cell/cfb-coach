"""Opt-in defense v1 logistic caller. Synthetic artifact only; no game rows."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.cli import build_parser
from cfb_coach.db import CoachDB
from cfb_coach.madden import data as mdata
from cfb_coach.madden.human_opponents import ensure_madden_opponent
from cfb_coach.madden.macros import store_selection
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model import defense_v1, inference
from cfb_coach.madden.model.dual_control import install_primary_lions_books
from cfb_coach.madden.model.experimental_live import commit_experimental_decision
from cfb_coach.madden.model.schema import CoachingMode, MLStatus
from cfb_coach.madden.playbook import load_books
from cfb_coach.madden.playcaller import MaddenCall, make_call
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import Situation

_ACTIVE = {
    "offense": ["MATCH", "C2", "O-RUN", "C3", "MAN", "RZ", "ZERO", "PROT"],
    "defense": ["SAFE DEEP", "TAMPA MABLE", "QTRS OVERTOP", "SKY CONTAIN"],
}


def _artifact(best: tuple[str, str], other: tuple[str, str], **overrides) -> dict:
    payload = {
        "schema": defense_v1.SCHEMA,
        "policy": defense_v1.POLICY,
        "target": defense_v1.TARGET,
        "intercept": 0.0,
        "features": ["red_zone"],
        "coefficients": {"red_zone": 0.0},
        "call_families": ["shell", "pressure"],
        "packages": ["base"],
        "family_coefficients": {"shell": -2.0, "pressure": 2.0},
        "package_coefficients": {"base": 0.0},
        "candidates": [
            {"formation": best[0], "play": best[1], "family": "shell", "package": "base"},
            {"formation": other[0], "play": other[1], "family": "pressure", "package": "base"},
        ],
    }
    payload.update(overrides)
    return payload


class DefenseV1SchemaTests(unittest.TestCase):
    def test_synthetic_fixture_names_the_unverified_policy(self) -> None:
        art = defense_v1.validate_artifact(_artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0")))
        self.assertEqual(art["policy"], "logged_unverified_v1")
        self.assertEqual(art["schema"], "madden.defense.logistic.v1")
        self.assertEqual(art["target"], "offense_success")
        self.assertEqual(art["features"], ["red_zone"])

    def test_rejects_a_policy_that_claims_verification(self) -> None:
        raw = _artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0"))
        raw["policy"] = "verified_champion"
        with self.assertRaises(defense_v1.DefenseV1Error) as caught:
            defense_v1.validate_artifact(raw)
        self.assertEqual(caught.exception.reason, "invalid")

    def test_rejects_bool_coefficients_unknown_features_and_duplicates(self) -> None:
        raw = _artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0"))
        raw["intercept"] = True
        with self.assertRaises(defense_v1.DefenseV1Error):
            defense_v1.validate_artifact(raw)
        raw = _artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0"))
        raw["features"] = ["yards"]
        raw["coefficients"] = {"yards": 1.0}
        with self.assertRaises(defense_v1.DefenseV1Error):
            defense_v1.validate_artifact(raw)
        raw = _artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0"))
        raw["candidates"].append(dict(raw["candidates"][0]))
        with self.assertRaises(defense_v1.DefenseV1Error):
            defense_v1.validate_artifact(raw)

    def test_lower_offense_success_wins_and_a_missing_number_does_not_become_zero(self) -> None:
        art = defense_v1.validate_artifact(_artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0")))
        sit = Situation(raw="d 1&10", side="defense", down=1, distance=10)
        shell = defense_v1.offense_success_probability(art, sit, art["candidates"][0])
        pressure = defense_v1.offense_success_probability(art, sit, art["candidates"][1])
        self.assertLess(shell, pressure)
        bare = Situation(raw="d", side="defense", down=1, distance=None)
        missing = defense_v1.validate_artifact({
            **art,
            "features": ["distance"],
            "coefficients": {"distance": 0.1},
        })
        with self.assertRaises(defense_v1.DefenseV1Error) as caught:
            defense_v1.offense_success_probability(missing, bare, art["candidates"][0])
        self.assertEqual(caught.exception.reason, "missing_feature")
        flagged = defense_v1.validate_artifact({
            **art,
            "features": ["distance", "distance_missing"],
            "coefficients": {"distance": 0.0, "distance_missing": 1.0},
        })
        values = defense_v1.situation_feature_vector(bare, flagged["features"])
        self.assertEqual(values["distance"], 0.0)
        self.assertEqual(values["distance_missing"], 1.0)

    def test_missing_file_and_flag_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            with self.assertRaises(defense_v1.DefenseV1Error) as caught:
                defense_v1.load_artifact(missing)
            self.assertEqual(caught.exception.reason, "missing")
            dest = Path(tmp) / "defense_v1.json"
            dest.write_text(json.dumps(_artifact(("Nickel", "Cover 3"), ("Dime", "Cover 0"))), encoding="utf-8")
            loaded = defense_v1.load_artifact(dest)
            self.assertEqual(loaded["policy"], "logged_unverified_v1")
            self.assertEqual(defense_v1.resolve_artifact_path(str(dest)), dest)
        self.assertEqual(
            defense_v1.default_artifact_path(),
            Path.home() / ".cfb-coach" / "models" / "defense_v1.json",
        )


class DefenseV1LiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.home = Path(self._td.name)
        self.db = CoachDB(self.home / "madden27.db", seed=mdata.load_seed())
        ensure_madden_opponent(self.db, "ryan")
        store_selection(self.db, "ryan", _ACTIVE)
        install_primary_lions_books(self.db)
        self.book = load_books(self.db)["defense"]["formations"]
        pairs = [(formation, play) for formation, plays in self.book.items() for play in plays]
        self.assertGreaterEqual(len(pairs), 2)
        self.best = pairs[0]
        self.other = pairs[1]
        self.path = self.home / "defense_v1.json"
        self.path.write_text(json.dumps(_artifact(self.best, self.other)), encoding="utf-8")
        self.heuristic = MaddenCall(
            "defense", self.other[0], self.other[1], "No adj", "User hook", "established",
        )

    def tearDown(self) -> None:
        self.db.close()
        self._td.cleanup()

    def _sit(self, raw: str = "d 1&10") -> Situation:
        sit = parse_madden_situation(raw, default_side="defense")
        sit.extras["defense_model"] = "v1"
        sit.extras["defense_model_path"] = str(self.path)
        return sit

    def test_explicit_v1_shows_the_legal_pick_without_verified_snaps_or_experimental_mode(self) -> None:
        self.assertEqual(defense.mode(self.db), "off")
        self.assertIsNone(defense.load_model(None))
        call = defense.maybe_apply_defense(self.heuristic, self._sit(), "ryan", self.db, self.book)
        self.assertEqual((call.formation, call.play), self.best)
        self.assertEqual(call.call_label, "EXPERIMENTAL")
        self.assertIn("LABEL: EXPERIMENTAL", call.format())
        self.assertNotIn("LABEL: HEURISTIC", call.format())
        self.assertIn(call.play, self.book[call.formation])
        decision = call._pending_ml_decision
        self.assertFalse(decision.fell_back)
        self.assertEqual(decision.model_version, "logged_unverified_v1")
        self.assertEqual(decision.policy_version, defense_v1.SCHEMA)
        self.assertEqual((decision.shadow_pick.formation, decision.shadow_pick.play), self.best)
        self.assertEqual((decision.heuristic_pick.formation, decision.heuristic_pick.play), self.other)
        self.assertEqual((decision.final_pick.formation, decision.final_pick.play), self.best)
        log = call.defense_v1_log
        self.assertEqual(log["model_pick"]["play"], self.best[1])
        self.assertEqual(log["heuristic_pick"]["play"], self.other[1])
        self.assertEqual(log["executed_pick"]["play"], self.best[1])
        self.assertFalse(log["fell_back"])
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.HEURISTIC)

        row_id = commit_experimental_decision(
            self.db, call, game_id="synthetic-v1", snap_id="snap-v1", snap_seq=1, session_id="synthetic-v1",
        )
        self.assertIsNotNone(row_id)
        row = self.db.conn.execute("SELECT * FROM ml_decisions WHERE id = ?", (row_id,)).fetchone()
        self.assertEqual(row["shadow_formation"], self.best[0])
        self.assertEqual(row["shadow_play"], self.best[1])
        self.assertEqual(row["heuristic_formation"], self.other[0])
        self.assertEqual(row["heuristic_play"], self.other[1])
        self.assertEqual(row["final_formation"], self.best[0])
        self.assertEqual(row["final_play"], self.best[1])
        self.assertEqual(row["model_version"], "logged_unverified_v1")
        payload = json.loads(row["decision_json"])
        config = {key: value for key, value in payload["config_source"]}
        self.assertEqual(config["policy"], "logged_unverified_v1")
        self.assertEqual(config["model_play"], self.best[1])
        self.assertEqual(config["executed_play"], self.best[1])
        self.assertEqual(config["heuristic_play"], self.other[1])

    def test_illegal_call_timeout_exception_and_invalid_file_fall_back(self) -> None:
        illegal = _artifact(("Not A Formation", "Not A Play"), self.other)
        self.path.write_text(json.dumps(illegal), encoding="utf-8")
        call = defense.maybe_apply_defense(self.heuristic, self._sit(), "ryan", self.db, self.book)
        self.assertEqual((call.formation, call.play), self.other)
        self.assertEqual(call.call_label, "HEURISTIC")
        self.assertIn("LABEL: HEURISTIC", call.format())
        self.assertNotIn("LABEL: EXPERIMENTAL", call.format())
        self.assertIs(call._pending_ml_decision.fallback_reason, MLStatus.INVALID_OUTPUT)
        self.assertIn("fallback_source=select_defense", call._pending_ml_decision.ml_status_detail)
        self.assertIn("illegal play", call._pending_ml_decision.ml_status_detail)
        self.assertEqual(call.defense_v1_log["model_pick"]["play"], "Not A Play")
        self.assertEqual(call.defense_v1_log["executed_pick"]["play"], self.other[1])
        self.assertIsNone(call._pending_ml_decision.shadow_pick)
        self.assertIs(call, self.heuristic)

        self.path.write_text(json.dumps(_artifact(self.best, self.other)), encoding="utf-8")
        with mock.patch.object(
            defense_v1, "_check_budget",
            side_effect=defense_v1.DefenseV1Error("timeout", "defense v1 exceeded 150 ms"),
        ):
            timed = defense.maybe_apply_defense(self.heuristic, self._sit(), "ryan", self.db, self.book)
        self.assertEqual((timed.formation, timed.play), self.other)
        self.assertIs(timed._pending_ml_decision.fallback_reason, MLStatus.TIMEOUT)
        self.assertEqual(timed.call_label, "HEURISTIC")

        with mock.patch.object(defense_v1, "offense_success_probability", side_effect=RuntimeError("boom")):
            exploded = defense.maybe_apply_defense(self.heuristic, self._sit(), "ryan", self.db, self.book)
        self.assertEqual((exploded.formation, exploded.play), self.other)
        self.assertIs(exploded._pending_ml_decision.fallback_reason, MLStatus.EXCEPTION)
        self.assertEqual(exploded.call_label, "HEURISTIC")

        self.path.write_text("{", encoding="utf-8")
        invalid = defense.maybe_apply_defense(self.heuristic, self._sit(), "ryan", self.db, self.book)
        self.assertIs(invalid._pending_ml_decision.fallback_reason, MLStatus.INVALID_OUTPUT)
        self.assertEqual(invalid.call_label, "HEURISTIC")
        self.assertIsNone(invalid.defense_v1_log["model_pick"])

        sit = self._sit()
        sit.extras["defense_model_path"] = str(self.home / "absent.json")
        missing = defense.maybe_apply_defense(self.heuristic, sit, "ryan", self.db, self.book)
        self.assertIs(missing._pending_ml_decision.fallback_reason, MLStatus.MODEL_MISSING)
        self.assertIn("defense v1 artifact missing", missing._pending_ml_decision.ml_status_detail)
        self.assertEqual(missing.call_label, "HEURISTIC")
        self.assertIn("fallback_reason: model_missing", missing.format())

    def test_empty_prior_uses_v1_and_a_verified_file_does_not(self) -> None:
        defense.set_mode(self.db, "experimental")
        sit = parse_madden_situation("d 2&7", default_side="defense")
        sit.extras["require_trained_defense_artifact"] = True
        sit.extras["defense_model_path"] = str(self.path)
        call = defense.maybe_apply_defense(self.heuristic, sit, "ryan", self.db, self.book)
        self.assertEqual((call.formation, call.play), self.best)
        self.assertEqual(call.call_label, "EXPERIMENTAL")
        self.assertEqual(call.defense_v1_log["policy"], "logged_unverified_v1")

        row = {
            "supervised_eligible": True, "side": "defense", "executed_play": "Cover 3",
            "success": "true", "executed_verification": "verified",
            "executed_status": "identified", "label_available": True,
            "opponent_type": "human", "opponent_id": "ryan", "game_id": "synthetic-unit",
        }
        art = defense.train_defense([row])
        dest = self.home / "coordinator.json"
        defense.save_model(art, dest)
        self.db.set_meta(defense.MODEL_META, str(dest))
        self.assertGreater(int(art["supervised_defensive_snaps"]), 0)
        with mock.patch("cfb_coach.madden.model.defense_v1.load_artifact", side_effect=AssertionError("v1")):
            kept = defense.maybe_apply_defense(self.heuristic, sit, "ryan", self.db, self.book)
        self.assertFalse(hasattr(kept, "defense_v1_log"))
        sit.extras["defense_model"] = "v1"
        forced = defense.maybe_apply_defense(self.heuristic, sit, "ryan", self.db, self.book)
        self.assertEqual(forced.call_label, "EXPERIMENTAL")
        self.assertEqual((forced.formation, forced.play), self.best)
        self.assertEqual(forced._pending_ml_decision.model_version, "logged_unverified_v1")

    def test_make_call_opts_in_and_leaves_offense_alone(self) -> None:
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        call = make_call(
            parse_madden_situation("d 1&10", default_side="defense"),
            "ryan", self.db, active_macros=_ACTIVE, live_macros=False,
            defense_model="v1", defense_model_path=str(self.path), rng=random.Random(0),
        )
        self.assertEqual(call.side, "defense")
        self.assertIn(call.call_label, ("EXPERIMENTAL", "HEURISTIC"))
        self.assertIn(f"LABEL: {call.call_label}", call.format())
        if call.call_label == "EXPERIMENTAL":
            self.assertEqual((call.formation, call.play), self.best)
            self.assertEqual(call.defense_v1_log["executed_pick"]["play"], self.best[1])
        else:
            self.assertTrue(call.defense_v1_log["fell_back"])
            self.assertIn("fallback_reason:", call.format())
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)

        with mock.patch(
            "cfb_coach.madden.model.defense_v1.load_artifact",
            side_effect=AssertionError("defense artifact loaded on offense"),
        ):
            offense = make_call(
                parse_madden_situation("1&10 my 25", default_side="offense"),
                "ryan", self.db, active_macros=_ACTIVE, live_macros=False,
                allow_human_ml=False, defense_model="v1",
                defense_model_path=str(self.path), rng=random.Random(0),
            )
        self.assertEqual(offense.side, "offense")
        self.assertFalse(hasattr(offense, "defense_v1_log"))
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)

        with mock.patch.object(defense_v1, "load_artifact", side_effect=RuntimeError("disk")):
            fallen = make_call(
                parse_madden_situation("d 1&10", default_side="defense"),
                "ryan", self.db, active_macros=_ACTIVE, live_macros=False,
                defense_model="v1", defense_model_path=str(self.path), rng=random.Random(0),
            )
        self.assertEqual(fallen.call_label, "HEURISTIC")
        self.assertIs(fallen._pending_ml_decision.fallback_reason, MLStatus.EXCEPTION)
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.EXPERIMENTAL)
        self.assertEqual(defense.mode(self.db), "off")

    def test_play_flags_parse(self) -> None:
        args = build_parser().parse_args([
            "play", "--game", "madden27", "-o", "ryan",
            "--defense-model", "v1",
            "--defense-model-path", str(self.path),
            "--once", "d 1&10", "--terminal", "--no-overlay",
        ])
        self.assertEqual(args.defense_model, "v1")
        self.assertEqual(args.defense_model_path, str(self.path))
        self.assertFalse(args.ml_defense_control)
        self.assertFalse(args.ml_human_control)


def _interaction(**overrides: float) -> dict[str, float]:
    row = {name: 0.0 for name in defense_v1.INTERACTION_FEATURES}
    row.update(overrides)
    return row


def _v2_artifact(candidates: list[dict[str, str]], family_bias: dict[str, dict[str, float]], **extra) -> dict:
    families: list[str] = []
    packages: list[str] = []
    for row in candidates:
        if row["family"] not in families:
            families.append(row["family"])
        if row["package"] not in packages:
            packages.append(row["package"])
    payload = {
        "schema": defense_v1.SCHEMA_V2,
        "policy": defense_v1.POLICY,
        "target": defense_v1.TARGET,
        "intercept": 0.0,
        "features": ["red_zone"],
        "coefficients": {"red_zone": 0.0},
        "call_families": families,
        "packages": packages,
        "family_coefficients": {name: 0.0 for name in families},
        "package_coefficients": {name: 0.0 for name in packages},
        "family_situation_coefficients": {
            name: family_bias.get(name, _interaction()) for name in families
        },
        "package_situation_coefficients": {name: _interaction() for name in packages},
        "candidates": candidates,
        "temperature": 0.0,
        "top_k": 4,
        "max_repeat": 2,
        "family_spread": 0.0,
    }
    payload.update(extra)
    return defense_v1.validate_artifact(payload)


class DefenseV2VarietyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {"formation": "Nickel", "play": "Buzz", "family": "cover3", "package": "nickel"},
            {"formation": "Dime", "play": "Tampa", "family": "cover2", "package": "dime"},
            {"formation": "Bear", "play": "Goal", "family": "goal", "package": "base"},
            {"formation": "Nickel", "play": "Man", "family": "man", "package": "nickel"},
        ]
        self.book = {"Nickel": ["Buzz", "Man"], "Dime": ["Tampa"], "Bear": ["Goal"]}
        self.artifact = _v2_artifact(self.candidates, {
            "cover3": _interaction(yardline_mid=-4.0),
            "cover2": _interaction(long_yardage=-5.0),
            "goal": _interaction(short_yardage=-4.0),
            "man": _interaction(yardline_own=-4.0),
        })

    def _pick(self, sit: Situation, **kwargs):
        return defense_v1.pick_call(self.artifact, sit, self.book, seed=1, **kwargs)["pick"]["play"]

    def test_v2_rankings_change_across_four_situations(self) -> None:
        situations = [
            Situation(raw="short", side="defense", down=3, distance=2, yardline=90, short_yardage=True),
            Situation(raw="long", side="defense", down=3, distance=12, yardline=40, long_yardage=True),
            Situation(raw="own", side="defense", down=1, distance=10, yardline=15),
            Situation(raw="mid", side="defense", down=2, distance=7, yardline=55),
        ]
        plays = [self._pick(sit) for sit in situations]
        self.assertEqual(plays, ["Goal", "Tampa", "Man", "Buzz"])
        self.assertGreaterEqual(len(set(plays)), 4)
        self.assertEqual(self.artifact["schema"], "madden.defense.logistic.v2")
        self.assertEqual(
            set(self.artifact["family_situation_coefficients"]["cover3"]),
            set(defense_v1.INTERACTION_FEATURES),
        )

    def test_no_call_repeats_more_than_twice_and_the_seed_repeats(self) -> None:
        sit = Situation(raw="flat", side="defense", down=1, distance=10, yardline=55)
        greedy = _v2_artifact(self.candidates, {}, family_spread=0.0, temperature=0.0)
        varied = _v2_artifact(self.candidates, {}, family_spread=0.0, temperature=0.4)

        def sequence(artifact: dict, seed: int) -> list[str]:
            recent: list[tuple[str, str]] = []
            plays = []
            for _ in range(9):
                picked = defense_v1.pick_call(artifact, sit, self.book, recent=recent, seed=seed)["pick"]
                plays.append(picked["play"])
                recent.append((picked["formation"], picked["play"]))
            return plays

        capped = sequence(greedy, 7)
        self.assertEqual(capped[:3], ["Goal", "Goal", "Tampa"])
        seeded = sequence(varied, 7)
        self.assertEqual(seeded, sequence(varied, 7))
        self.assertNotEqual(seeded, sequence(varied, 8))
        for plays in (capped, seeded, sequence(varied, 8)):
            for index in range(len(plays) - 2):
                self.assertFalse(plays[index] == plays[index + 1] == plays[index + 2])

    def test_family_spread_rotates_plays_inside_the_family(self) -> None:
        candidates = [
            {"formation": "Nickel", "play": "Buzz", "family": "cover3", "package": "first"},
            {"formation": "Nickel", "play": "Sky", "family": "cover3", "package": "second"},
            {"formation": "Dime", "play": "Tampa", "family": "cover2", "package": "dime"},
        ]
        artifact = _v2_artifact(
            candidates,
            {},
            family_spread=0.5,
            temperature=0.0,
            package_coefficients={"first": -2.0, "second": -1.0, "dime": 0.0},
        )
        book = {"Nickel": ["Buzz", "Sky"], "Dime": ["Tampa"]}
        sit = Situation(raw="flat", side="defense", down=1, distance=10, yardline=55)
        first = defense_v1.pick_call(artifact, sit, book, seed=1)["pick"]
        self.assertEqual(first["play"], "Buzz")
        second = defense_v1.pick_call(
            artifact, sit, book, recent=[("Nickel", "Buzz")], seed=1,
        )["pick"]
        self.assertEqual(second["play"], "Sky")
        self.assertEqual(second["family"], "cover3")

    def test_v2_still_refuses_an_illegal_best_call(self) -> None:
        candidates = [
            {"formation": "Not A Formation", "play": "Not A Play", "family": "cover3", "package": "nickel"},
            {"formation": "Nickel", "play": "Buzz", "family": "cover2", "package": "dime"},
        ]
        artifact = _v2_artifact(candidates, {"cover3": _interaction(short_yardage=-4.0)})
        sit = Situation(raw="short", side="defense", down=3, distance=2, yardline=90, short_yardage=True)
        book = {"Nickel": ["Buzz"]}
        with self.assertRaises(defense_v1.DefenseV1Error) as caught:
            defense_v1.pick_call(artifact, sit, book, seed=1)
        self.assertEqual(caught.exception.reason, "illegal")
        self.assertEqual(caught.exception.pick["play"], "Not A Play")

    def test_goal_to_go_distance_uses_the_named_side(self) -> None:
        from cfb_coach.madden.situation import parse_madden_situation
        from cfb_coach.situation import parse_situation

        own = parse_situation("1st and goal my 8")
        self.assertEqual((own.down, own.distance, own.yardline), (1, 8, 8))
        madden = parse_madden_situation("1st and goal my 8")
        self.assertEqual((madden.down, madden.distance, madden.yardline), (1, 8, 8))
        self.assertTrue(madden.goal_line)
        opp = parse_situation("1&goal opp 6")
        self.assertEqual((opp.down, opp.distance, opp.yardline), (1, 6, 94))


class DefenseV1EmptyPriorMessageTests(unittest.TestCase):
    def test_missing_substitute_keeps_the_established_missing_artifact_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = CoachDB(home / "madden27.db", seed=mdata.load_seed())
            try:
                ensure_madden_opponent(db, "ryan")
                install_primary_lions_books(db)
                book = load_books(db)["defense"]["formations"]
                defense.set_mode(db, "experimental")
                formation, plays = next(iter(book.items()))
                heuristic = MaddenCall("defense", formation, plays[0], "No adj", "User hook", "h")
                sit = parse_madden_situation("d 1&10", default_side="defense")
                sit.extras["require_trained_defense_artifact"] = True
                call = defense.maybe_apply_defense(heuristic, sit, "ryan", db, book)
                self.assertEqual(call.call_label, "HEURISTIC")
                self.assertIs(call._pending_ml_decision.fallback_reason, MLStatus.MODEL_MISSING)
                self.assertIn(
                    "missing defense artifact or zero verified snaps",
                    call._pending_ml_decision.ml_status_detail,
                )
                self.assertIn("fallback_source=select_defense", call._pending_ml_decision.ml_status_detail)
                self.assertIsNone(call.defense_v1_log["model_pick"])
                self.assertEqual(call.defense_v1_log["executed_pick"]["play"], plays[0])
            finally:
                db.close()
