"""Decision-quality scoring. The harness is stubbed; production modules stay untouched."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import pwd
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "eval_scoring",
    _ROOT / "scripts" / "madden_ml" / "eval_scoring.py",
)
scoring = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(scoring)

from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.zones import field_zone


def _call(formation: str, play: str, adjustment=None, state=None, mode: str = "heuristic"):
    return {
        "formation": formation,
        "play": play,
        "adjustment_id": adjustment,
        "macro_state": state,
        "display_line": f"DISPLAY {formation} / {play}",
        "mode": mode,
    }


def _harness(new, old, **extra):
    row = {
        "new": new,
        "old": old,
        "fallback_reason": extra.get("fallback_reason"),
        "model_version": extra.get("model_version"),
        "scored_on": extra.get("scored_on"),
        "latency_ms": extra.get("latency_ms", {"whole": 10.0, "model": 4.0}),
        "budget_ms": extra.get("budget_ms", 150),
        "over_budget": extra.get("over_budget", False),
        "legal": extra.get("legal", True),
        "propensity": extra.get("propensity"),
        "opp_look": extra.get("opp_look"),
        "seed": extra.get("seed"),
    }
    return row


class SyntheticGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = scoring.build_synthetic_identities()

    def test_grid_is_balanced_and_tagged(self):
        self.assertEqual(len(self.rows), 72)
        self.assertEqual(sum(row["side"] == "offense" for row in self.rows), 36)
        self.assertEqual(sum(row["side"] == "defense" for row in self.rows), 36)
        self.assertTrue(all(row["tags"] == ["synthetic"] for row in self.rows))
        self.assertTrue(all(row["source"] == "synthetic" for row in self.rows))
        self.assertTrue(all(row["default_filled"] == [] for row in self.rows))
        self.assertTrue(all(row["look_observed"] is False for row in self.rows))

    def test_every_factor_level_appears_on_both_sides(self):
        for side in ("offense", "defense"):
            chosen = [row for row in self.rows if row["side"] == side]
            self.assertEqual({row["distance_bucket"] for row in chosen}, set(scoring.DISTANCE_BUCKETS))
            self.assertEqual({row["field_zone"] for row in chosen}, set(scoring.FIELD_ZONES))
            self.assertEqual({row["clock_bucket"] for row in chosen}, set(scoring.CLOCK_BUCKETS))
            self.assertEqual({row["score_bucket"] for row in chosen}, set(scoring.SCORE_BUCKETS))
            self.assertEqual({row["down"] for row in chosen}, {1, 2, 3, 4})
            for row in chosen:
                self.assertEqual(row["distance_bucket"], scoring.distance_bucket(row["distance"]))
                self.assertEqual(
                    row["score_bucket"],
                    scoring.score_bucket(row["score_us"], row["score_them"]),
                )

    def test_null_and_empty_looks_are_distinct_on_every_row(self):
        for row in self.rows:
            looks = {(look["opp_personnel"], look["opp_set"]) for look in row["looks"]}
            self.assertIn((None, None), looks)
            self.assertIn(("empty", "empty"), looks)
            self.assertGreaterEqual(len(row["looks"]), 3)
            self.assertFalse(any(look["opp_personnel"] == "" or look["opp_set"] == "" for look in row["looks"]))

    def test_distance_buckets_and_zones_match_the_written_cuts(self):
        self.assertEqual(scoring.distance_bucket(2), "short")
        self.assertEqual(scoring.distance_bucket(3), "medium")
        self.assertEqual(scoring.distance_bucket(6), "medium")
        self.assertEqual(scoring.distance_bucket(7), "long")
        codes = {
            "own_deep": "open",
            "midfield": "open",
            "plus_territory": "open",
            "red_zone": "rz",
            "goal_line": "gl",
        }
        for row in self.rows:
            if row["field_zone"] not in codes:
                continue
            distance = row["distance"] if row["field_zone"] != "red_zone" else min(row["distance"], 5)
            # Red-zone yardline 88 is 12 yards from the goal, so a distance of 10
            # would be goal-to-go. The stored zone is the grid label; the code
            # zone below uses a distance that stays inside that label.
            if row["field_zone"] == "red_zone":
                distance = 5
            self.assertEqual(field_zone(row["yardline"], distance), codes[row["field_zone"]])

    def test_raw_line_parses_without_turning_4_min_into_two_minute(self):
        for row in self.rows:
            parsed = parse_madden_situation(row["raw"], default_side=row["side"])
            self.assertEqual(parsed.side, row["side"])
            self.assertEqual(parsed.down, row["down"])
            self.assertEqual(parsed.distance, row["distance"])
            self.assertEqual(parsed.yardline, row["yardline"])
            self.assertEqual(parsed.extras.get("quarter"), row["quarter"])
            self.assertEqual(parsed.score_us, row["score_us"])
            self.assertEqual(parsed.score_them, row["score_them"])
            self.assertEqual(parsed.two_minute, row["clock_bucket"] == "2-minute")
            if row["clock_bucket"] == "4-min protect":
                self.assertIn("4-min protect", row["raw"])
                self.assertNotIn("two minute", row["raw"])
                self.assertFalse(parsed.two_minute)

    def test_checked_in_fixture_matches_the_builder(self):
        on_disk = json.loads(scoring.SYNTHETIC_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, self.rows)


class ObservedLoaderTests(unittest.TestCase):
    def test_loader_keeps_the_csv_integers_and_assigns_looks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "presnap_observed_cpu.csv"
            path.write_text(
                "side,down,distance,yardline,quarter,game_id\n"
                "offense,3,1,97,4,g1\n"
                "D,2,8,22,1,g4\n",
                encoding="utf-8",
            )
            rows = scoring.load_observed_csv(path)
        self.assertEqual(len(rows), 2)
        first, second = rows
        self.assertEqual(first["tags"], ["observed"])
        self.assertEqual((first["down"], first["distance"], first["yardline"], first["quarter"]), (3, 1, 97, 4))
        self.assertEqual(first["game_id"], "g1")
        self.assertEqual(first["default_filled"], ["score", "clock"])
        self.assertEqual((first["score_us"], first["score_them"]), (0, 0))
        self.assertIsNone(first["clock_seconds"])
        self.assertEqual(first["clock_bucket"], "normal")
        self.assertFalse(first["two_minute"])
        self.assertFalse(first["look_observed"])
        self.assertEqual(len(first["looks"]), 3)
        self.assertEqual(first["looks"][0]["opp_personnel"], None)
        self.assertEqual(first["looks"][1]["opp_set"], "empty")
        expanded = scoring.expand_decision_rows(rows)
        self.assertEqual(len(expanded), 6)
        self.assertTrue(all(row["tags"] == ["observed_situation", "assigned_look"] for row in expanded))
        self.assertTrue(all(row["look_observed"] is False for row in expanded))
        self.assertNotIn("nickel", first["raw"])
        self.assertNotIn("empty", first["raw"])
        self.assertEqual(second["side"], "defense")
        self.assertEqual(second["game_id"], "g4")
        self.assertEqual(second["yardline"], 22)

    def test_blank_look_is_null_and_empty_stays_empty(self):
        self.assertIsNone(scoring.normalize_opp_look(""))
        self.assertIsNone(scoring.normalize_opp_look("   "))
        self.assertIsNone(scoring.normalize_opp_look(None))
        self.assertEqual(scoring.normalize_opp_look("empty"), "empty")
        self.assertEqual(
            scoring.normalize_opp_look({"opp_personnel": "", "opp_set": "empty"}),
            {"opp_personnel": None, "opp_set": "empty"},
        )

    def test_attached_csv_is_not_invented(self):
        with scoring.OBSERVED_CSV.open(newline="", encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
        self.assertEqual(len(source), 161)
        self.assertEqual(
            list(source[0].keys()),
            ["row", "game", "side", "down", "distance", "yardline", "quarter", "tag"],
        )
        loaded = scoring.load_observed_csv(scoring.OBSERVED_CSV)
        self.assertEqual(len(loaded), 161)
        for raw, row in zip(source, loaded):
            self.assertEqual(raw["tag"], "observed")
            self.assertEqual(row["side"], raw["side"])
            self.assertEqual(row["down"], int(raw["down"]))
            self.assertEqual(row["distance"], int(raw["distance"]))
            self.assertEqual(row["yardline"], int(raw["yardline"]))
            self.assertEqual(row["quarter"], int(raw["quarter"]))
            self.assertEqual(row["game_id"], raw["game"])
            self.assertEqual(row["tags"], ["observed"])
            self.assertEqual(row["default_filled"], ["score", "clock"])
            self.assertEqual((row["score_us"], row["score_them"]), (0, 0))
            self.assertIsNone(row["clock_seconds"])
            self.assertEqual(row["clock_bucket"], "normal")
            self.assertFalse(row["two_minute"])
            self.assertFalse(row["look_observed"])
            self.assertEqual(len(row["looks"]), 3)
            self.assertNotIn("opp_personnel", raw)
            self.assertNotIn("nickel", row["raw"])
            self.assertNotIn("empty", row["raw"])
        expanded = scoring.expand_decision_rows(loaded)
        self.assertEqual(len(expanded), 161 * 3)
        self.assertTrue(
            all(row["tags"] == ["observed_situation", "assigned_look"] for row in expanded)
        )
        self.assertTrue(all(row["look_observed"] is False for row in expanded))
        identities, note = scoring.load_situation_identities()
        counts = note["fixture_counts"]
        self.assertEqual(len(identities), 72 + 161)
        self.assertEqual(counts["identity_tags_by_side"]["synthetic"], {"offense": 36, "defense": 36})
        self.assertEqual(counts["identity_tags_by_side"]["observed"], {"offense": 161, "defense": 0})
        self.assertEqual(
            counts["decision_tags_by_side"]["synthetic"],
            {"offense": 144, "defense": 144},
        )
        self.assertEqual(
            counts["decision_tags_by_side"]["observed_situation,assigned_look"],
            {"offense": 483, "defense": 0},
        )
        self.assertEqual(counts["identities"], 233)
        self.assertGreaterEqual(counts["identities"], 120)


class BookAndRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmp.name) / "madden27.db"
        cls.book = scoring.load_installed_book(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_book_is_the_installed_five_and_five(self):
        self.assertEqual(
            set(self.book["offense"]),
            {
                "Gun Doubles Clamp Stack",
                "Gun 5WR Tight",
                "Gun Trips X Nasty",
                "Pistol Deuce Close",
                "Pistol Trips",
            },
        )
        self.assertEqual(
            set(self.book["defense"]),
            {
                "Dime 3-2 Odd",
                "Nickel Double Mug",
                "Nickel Single Mug",
                "4-3 Over",
                "Nickel Over",
            },
        )
        self.assertIn("HB Dive", self.book["offense"]["Pistol Trips"])
        self.assertIn("Verticals", self.book["offense"]["Pistol Trips"])
        self.assertIn("Cover 1 Contain", self.book["defense"]["4-3 Over"])

    def test_legality_is_book_plus_macro_state(self):
        book = self.book
        legal, reasons = scoring.decision_legal(
            "offense", "Pistol Trips", "HB Dive", None, None, book
        )
        self.assertTrue(legal)
        self.assertEqual(reasons, [])
        legal, reasons = scoring.decision_legal(
            "offense", "Pistol Trips", "HB Dive", "MACRO-1", "armed", book
        )
        self.assertTrue(legal)
        legal, reasons = scoring.decision_legal(
            "offense", "Pistol Trips", "HB Dive", "MACRO-1", "draft", book
        )
        self.assertFalse(legal)
        self.assertEqual(reasons, ["unarmed_macro"])
        legal, reasons = scoring.decision_legal(
            "offense", "Not A Formation", "HB Dive", None, None, book
        )
        self.assertFalse(legal)
        self.assertIn("outside_book", reasons)

    def test_context_rules_match_the_written_list(self):
        rules = scoring.scoring_rules()
        self.assertEqual(
            [rule["id"] for rule in rules],
            [
                "no_deep_shot_on_short_or_goal_line",
                "two_minute_trailing_not_clock_kill_run",
                "no_kneel_or_run_clock_when_trailing_late",
                "goal_line_short_not_dime",
                "trailing_late_no_prevent",
            ],
        )
        self.assertIn(
            "passed its defined contextual rules, which cover a limited set of decisions.",
            scoring.CONTEXT_RULES_NOTE,
        )
        short_goal = {
            "side": "offense",
            "down": 3,
            "distance_bucket": "short",
            "field_zone": "goal_line",
            "clock_bucket": "normal",
            "score_bucket": "close",
            "default_filled": [],
        }
        deep = scoring.apply_context_rules(short_goal, "Pistol Trips", "Verticals")
        dive = scoring.apply_context_rules(short_goal, "Pistol Trips", "HB Dive")
        self.assertFalse(_passed(deep, "no_deep_shot_on_short_or_goal_line"))
        self.assertTrue(_passed(dive, "no_deep_shot_on_short_or_goal_line"))

        trailing = {
            "side": "offense",
            "down": 1,
            "distance_bucket": "long",
            "field_zone": "own_deep",
            "clock_bucket": "2-minute",
            "score_bucket": "trailing_big",
            "default_filled": [],
        }
        self.assertFalse(_passed(scoring.apply_context_rules(trailing, "Pistol Trips", "HB Dive"), "two_minute_trailing_not_clock_kill_run"))
        self.assertTrue(
            _passed(
                scoring.apply_context_rules(trailing, "Pistol Trips", "HB Slip Screen"),
                "two_minute_trailing_not_clock_kill_run",
            )
        )
        self.assertFalse(
            _passed(
                scoring.apply_context_rules(trailing, "Pistol Trips", "QB Kneel"),
                "no_kneel_or_run_clock_when_trailing_late",
            )
        )
        filled = dict(trailing, default_filled=["score", "clock"])
        self.assertIsNone(_passed(scoring.apply_context_rules(filled, "Pistol Trips", "HB Dive"), "two_minute_trailing_not_clock_kill_run"))

        goal_d = {
            "side": "defense",
            "down": 3,
            "distance_bucket": "short",
            "field_zone": "goal_line",
            "clock_bucket": "normal",
            "score_bucket": "close",
            "default_filled": [],
        }
        self.assertFalse(_passed(scoring.apply_context_rules(goal_d, "Dime 3-2 Odd", "Cover 1 Robber Press"), "goal_line_short_not_dime"))
        self.assertTrue(_passed(scoring.apply_context_rules(goal_d, "4-3 Over", "Cover 1 Contain"), "goal_line_short_not_dime"))
        late_d = {
            "side": "defense",
            "down": 2,
            "distance_bucket": "long",
            "field_zone": "midfield",
            "clock_bucket": "4-min protect",
            "score_bucket": "trailing_big",
            "default_filled": [],
        }
        self.assertFalse(_passed(scoring.apply_context_rules(late_d, "4-3 Over", "Prevent"), "trailing_late_no_prevent"))
        self.assertTrue(_passed(scoring.apply_context_rules(late_d, "4-3 Over", "Cover 1 Contain"), "trailing_late_no_prevent"))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.book = scoring.load_installed_book(Path(cls._tmp.name) / "madden27.db")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_stub_contract_scores_agreement_adaptation_and_latency(self):
        identities = scoring.build_synthetic_identities()

        def runner(sit, *, mode, opp_look, side, seed):
            package = (opp_look or {}).get("opp_set")
            if side == "offense":
                play = "HB Dive" if package == "nickel" else "HB Slip Screen"
                formation = "Pistol Trips"
                old_play = "HB Dive"
            else:
                formation, play, old_play = "4-3 Over", "Cover 1 Contain", "Cover 1 Contain"
            return _harness(
                _call(formation, play, mode=mode),
                _call(formation, old_play, mode="heuristic"),
                opp_look=opp_look,
                seed=seed,
                latency_ms={"whole": 12.0, "model": 5.0},
                propensity=None,
                model_version=None,
                scored_on=None,
            )

        report = scoring.evaluate(identities, runner, self.book, mode="heuristic")
        self.assertEqual(report["disclaimer"], scoring.DISCLAIMER)
        self.assertNotIn("share_cap", report)
        offense = report["summary"]["by_side"]["offense"]["synthetic"]
        defense = report["summary"]["by_side"]["defense"]["synthetic"]
        self.assertEqual(offense["n"], 36 * 4)
        self.assertEqual(defense["n"], 36 * 4)
        self.assertEqual(offense["formation_agree"], 36 * 4)
        self.assertEqual(offense["play_agree"], 36)
        self.assertEqual(defense["play_agree"], 36 * 4)
        self.assertEqual(offense["hard_fail"], 0)
        self.assertEqual(offense["adjustment_legal"], 36 * 4)
        self.assertNotIn("top_share_flag", offense["diversity"])
        self.assertGreater(offense["diversity"]["top_call_share"], 0.25)
        self.assertTrue(offense["diversity"]["formation_usage"])
        adapt = report["summary"]["adaptation"]
        self.assertEqual(adapt["offense:synthetic"]["adapted"], 36)
        self.assertEqual(adapt["offense:synthetic"]["seed_noise"], 0)
        self.assertEqual(adapt["defense:synthetic"]["unchanged_across_looks"], 36)
        self.assertEqual(report["summary"]["latency"]["whole"]["p50"], 12.0)
        self.assertEqual(report["summary"]["latency"]["over_budget"], 0)
        self.assertTrue(all(row["look_observed"] is False for row in report["decisions"]))
        self.assertIn("not effectiveness claims", render := scoring.render_markdown(report))
        self.assertIn("HB Slip Screen", render)
        self.assertIn("Pistol Trips", render)

    def test_same_seed_mismatch_is_noise_not_adaptation(self):
        identity = scoring.build_synthetic_identities()[0]
        identity = dict(identity)
        identity["looks"] = identity["looks"][:3]
        calls = {"n": 0}

        def runner(sit, *, mode, opp_look, side, seed):
            calls["n"] += 1
            play = "HB Dive" if calls["n"] % 2 else "HB Slip Screen"
            return _harness(
                _call("Pistol Trips", play),
                _call("Pistol Trips", play),
                opp_look=opp_look,
                seed=seed,
            )

        report = scoring.evaluate([identity], runner, self.book)
        block = report["summary"]["adaptation"]["offense:synthetic"]
        self.assertEqual(block["seed_noise"], 1)
        self.assertEqual(block["adapted"], 0)

    def test_usage_shares_are_reported_without_a_penalty(self):
        pairs = [
            ("Pistol Trips", "HB Dive"),
            ("Pistol Deuce Close", "HB Dive"),
            ("Gun Trips X Nasty", "Inside Zone"),
            ("Gun Doubles Clamp Stack", "Inside Zone"),
        ]
        prepared = []
        for index, (formation, play) in enumerate(pairs):
            request = _request(f"cap-{index}", "offense", "synthetic")
            decision = _call(formation, play)
            prepared.append(
                {
                    "request": request,
                    "first": _harness(decision, decision, opp_look=request["opp_look"], seed=1),
                    "second": _harness(decision, decision, opp_look=request["opp_look"], seed=1),
                }
            )
        report = scoring.build_report(prepared, self.book)
        diversity = report["summary"]["by_side"]["offense"]["synthetic"]["diversity"]
        self.assertEqual(diversity["top_call_share"], 0.25)
        self.assertNotIn("top_share_flag", diversity)
        self.assertNotIn("share_cap", diversity)
        self.assertIsNone(diversity.get("penalty"))
        self.assertEqual(diversity["call_entropy_bits"], 2.0)
        self.assertEqual(sum(item["n"] for item in diversity["formation_usage"]), 4)

        prepared[1]["first"]["new"] = _call("Pistol Trips", "HB Dive")
        prepared[1]["second"]["new"] = _call("Pistol Trips", "HB Dive")
        crowded = scoring.build_report(prepared, self.book)
        crowded_div = crowded["summary"]["by_side"]["offense"]["synthetic"]["diversity"]
        self.assertGreater(crowded_div["top_call_share"], 0.25)
        self.assertNotIn("top_share_flag", crowded_div)

    def test_outside_book_and_unarmed_macro_are_hard_fails(self):
        request = _request("bad-1", "offense", "synthetic")
        illegal = _call("Gun Bunch", "Mesh")
        first = _harness(illegal, illegal, legal=True, opp_look=request["opp_look"], seed=1)
        report = scoring.build_report(
            [{"request": request, "first": first, "second": first}],
            self.book,
        )
        row = report["decisions"][0]
        self.assertTrue(row["hard_fail"])
        self.assertIn("outside_book", row["legal_reasons"])
        self.assertTrue(row["legal_disagreement"])

        request = _request("bad-2", "defense", "synthetic")
        unarmed = _call("4-3 Over", "Cover 1 Contain", adjustment="MACRO-9", state="installed")
        first = _harness(unarmed, unarmed, legal=True, opp_look=request["opp_look"], seed=2)
        report = scoring.build_report(
            [{"request": request, "first": first, "second": first}],
            self.book,
        )
        row = report["decisions"][0]
        self.assertTrue(row["hard_fail"])
        self.assertEqual(row["legal_reasons"], ["unarmed_macro"])
        self.assertEqual(row["new"]["adjustment_id"], "MACRO-9")

    def test_latency_percentiles_and_fallback_counts_come_from_the_rows(self):
        self.assertIsNone(scoring.nearest_rank([], 50))
        self.assertEqual(scoring.nearest_rank([10, 20, 30, 40], 50), 20)
        self.assertEqual(scoring.nearest_rank([10, 20, 30, 40], 95), 40)
        prepared = []
        samples = [10.0, 20.0, 30.0, 40.0]
        statuses = [None, "timeout", "ok", "timeout"]
        for index, (whole, status) in enumerate(zip(samples, statuses)):
            request = _request(f"lat-{index}", "defense", "observed")
            decision = _call("Nickel Over", "Cover 4 Quarters")
            over = whole > 150
            first = _harness(
                decision,
                decision,
                latency_ms={"whole": whole, "model": whole / 2},
                fallback_reason=status,
                over_budget=over,
                opp_look=request["opp_look"],
                seed=index,
            )
            if index == 3:
                first["latency_ms"] = {"whole": 200.0, "model": 10.0}
                first["over_budget"] = False
            prepared.append({"request": request, "first": first, "second": first})
        report = scoring.build_report(prepared, self.book)
        latency = report["summary"]["latency"]
        self.assertEqual(latency["fallback_counts"], {"timeout": 2})
        self.assertEqual(latency["over_budget"], 1)
        self.assertEqual(latency["whole"]["max"], 200.0)
        self.assertEqual(report["decisions"][3]["over_budget"], True)

    def test_adapter_is_the_only_harness_door_and_it_is_absent(self):
        source = (_ROOT / "scripts" / "madden_ml" / "eval_scoring.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed: set[int] = set()
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                allowed.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            if isinstance(node, ast.FunctionDef) and node.name in {
                "_harness_module",
                "adapt_run_situation",
                "adapt_score_offline_model",
            }:
                allowed.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        needles = ("eval_harness", '"run_situation"', "'run_situation'", '"score_offline_model"', "'score_offline_model'")
        for lineno, line in enumerate(source.splitlines(), start=1):
            if any(needle in line for needle in needles):
                self.assertIn(lineno, allowed, line)
        self.assertNotIn("nothing ignored the game situation", source)
        with self.assertRaises(RuntimeError):
            scoring.adapt_score_offline_model(
                "1&10",
                side="offense",
                model_dir=".",
                strata=["madden27/cpu/offense"],
                offline_ok=False,
            )
        with self.assertRaises(ValueError):
            scoring.adapt_score_offline_model(
                "1&10",
                side="offense",
                model_dir=".",
                strata=["cfb27/cpu/offense"],
                offline_ok=True,
            )
        harness = _ROOT / "scripts" / "madden_ml" / "eval_harness.py"
        if harness.is_file():
            harness_text = harness.read_text(encoding="utf-8")
            self.assertIn("def run_situation", harness_text)
            self.assertIn("def score_offline_model", harness_text)
            with tempfile.TemporaryDirectory() as folder:
                model = Path(folder) / "model"
                _write_bundle(model, "v0.21")
                payload = scoring.adapt_score_offline_model(
                    "1&10",
                    side="offense",
                    model_dir=model,
                    strata=["madden27/cpu/offense"],
                    offline_ok=True,
                )
            self.assertEqual(payload["gate_bypassed"], "offline_eval")
            self.assertEqual(payload["model_version"], "v0.21")
            self.assertTrue(payload.get("play"))
            self.assertTrue((payload.get("heuristic") or {}).get("play"))
            fixture = _ROOT / "tests" / "madden_ml" / "fixtures" / "item3" / "v0.21"
            total = sum(path.stat().st_size for path in fixture.iterdir() if path.is_file())
            self.assertLess(total, 200_000)
            stored = json.loads((fixture / "model_params.json").read_text(encoding="utf-8"))
            groups = stored["strata"]["madden27/cpu/offense"]["model"]["coef"]
            self.assertIn("call", groups)
            self.assertIn("look", groups)
            self.assertIn("look_family", groups)
            self.assertLessEqual(len(groups["call"]), 8)
        else:
            with self.assertRaises(FileNotFoundError):
                scoring.adapt_run_situation({}, mode="heuristic", opp_look=None, side="offense", seed=1)
            with self.assertRaises(FileNotFoundError):
                scoring.adapt_score_offline_model(
                    "1&10",
                    side="offense",
                    model_dir=".",
                    strata=["madden27/cpu/offense"],
                    offline_ok=True,
                )

    def test_adapter_coerces_csv_text_and_rejects_a_missing_assigned_look(self):
        sit = {
            "raw": "1&10 my 50",
            "down": "1",
            "distance": "10",
            "yardline": "  ",
            "look_id": "11-nickel",
        }
        with self.assertRaises(RuntimeError):
            scoring.adapt_score_offline_model(
                sit,
                side="offense",
                model_dir=".",
                strata=["madden27/cpu/offense"],
                opp_look=None,
                offline_ok=True,
            )
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model"
            _write_bundle(model, "v0.21")
            payload = scoring.adapt_score_offline_model(
                sit,
                side="offense",
                model_dir=model,
                strata=["madden27/cpu/offense"],
                opp_look={"opp_personnel": "11", "opp_set": "nickel"},
                offline_ok=True,
            )
        diag = payload["input_diagnostics"]
        self.assertEqual(diag["down_type"], "str")
        self.assertEqual(diag["distance_type"], "str")
        self.assertEqual(diag["yardline_type"], "str")
        self.assertEqual(diag["down"], 1)
        self.assertEqual(diag["distance"], 10)
        self.assertIsNone(diag["yardline"])
        self.assertEqual(diag["opp_look"], {"opp_personnel": "11", "opp_set": "nickel"})
        self.assertEqual(diag["opp_look_type"], "dict")
        self.assertEqual(diag["opp_look_raw"], repr(diag["opp_look"]))
        blank = scoring._prepare_hook_call(
            {"down": None, "distance": "", "yardline": "12", "look_id": "null"},
            {"opp_personnel": None, "opp_set": None},
            "offense",
        )
        self.assertEqual(blank["diagnostics"]["down_type"], "NoneType")
        self.assertIsNone(blank["diagnostics"]["distance"])
        self.assertEqual(blank["diagnostics"]["yardline"], 12)

    def test_report_files_do_not_invent_an_observed_count(self):
        request = _request("one", "offense", "synthetic")
        decision = _call("Pistol Trips", "HB Slip Screen")
        first = _harness(decision, decision, opp_look=request["opp_look"], seed=1)
        report = scoring.build_report(
            [{"request": request, "first": first, "second": first}],
            self.book,
            inventory_note={"observed_csv_present": False},
        )
        with tempfile.TemporaryDirectory() as folder:
            json_path, md_path, jsonl_path = scoring.write_report(report, Path(folder))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            text = md_path.read_text(encoding="utf-8")
            self.assertTrue(jsonl_path.is_file())
        self.assertNotIn("share_cap", payload)
        self.assertEqual(payload["disclaimer"], scoring.DISCLAIMER)
        self.assertIn("passed its defined contextual rules, which cover a limited set of decisions.", text)
        self.assertNotIn("nothing ignored the game situation", text)
        self.assertIn("no_deep_shot_on_short_or_goal_line", text)
        self.assertNotIn("161", text)


class ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        madden = Path(cls._tmp.name) / "madden27.db"
        coach = Path(cls._tmp.name) / "coach.db"
        cls.resources = scoring.set_shadow_databases(madden, coach)
        cls.book = cls.resources.book

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_hook_keeps_refusal_folds_and_versions_apart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_bundle(root, "v0.21")
            _write_bundle(root / "folds" / "foldA", "v0.24", fold_id="foldA")
            _write_bundle(
                root / "folds" / "foldB",
                "v0.24",
                fold_id="foldB",
                returned_version="v0.21",
            )
            offense = scoring.build_synthetic_identities()[0]
            offense = dict(offense)
            offense["looks"] = [offense["looks"][0], offense["looks"][1]]
            empty = dict(offense)
            empty["situation_id"] = "syn-empty"
            empty["looks"] = [offense["looks"][1]]
            refused = _defense_fold_identity("3&1 my 12", "missing-fold", "obs-def-refuse")
            matched = _defense_fold_identity("3&2 my 12", "foldA", "obs-def-match")
            mismatched = _defense_fold_identity("2&8 my 40", "foldB", "obs-def-mismatch")
            report = scoring.evaluate_shadow(
                [offense, empty, refused, matched, mismatched],
                scoring.adapt_score_offline_model,
                self.book,
                model_dir=root,
                requested_version="v0.21",
                seed_count=2,
            )
        self.assertTrue(all(row["gate_bypassed"] == "offline_eval" for row in report["decisions"]))
        self.assertTrue(all(row["scored_on"] == "unreviewed" for row in report["decisions"]))
        self.assertTrue(all(row["offensive_personnel"] is None for row in report["decisions"]))
        personnel = {row["opp_personnel"] for row in report["decisions"] if row["situation_id"] == offense["situation_id"]}
        self.assertIn(None, personnel)
        self.assertIn("empty", personnel)
        versions = report["by_model_version"]
        self.assertEqual(report["summary_grouped_by"], "loaded_manifest_label_version")
        self.assertIn("v0.21", versions)
        self.assertIn("v0.24", versions)
        self.assertNotIn("v0.23", versions)
        versioned = [row for row in report["decisions"] if row.get("model_version")]
        self.assertTrue(versioned)
        self.assertTrue(all(row["model_version"] == row["manifest_model_version"] for row in versioned))
        fold_rows = [row for row in report["decisions"] if row.get("fold_id") in {"foldA", "foldB"}]
        self.assertTrue(fold_rows)
        self.assertTrue(all(row["model_version"] == "v0.24" for row in fold_rows))
        matched_rows = [row for row in fold_rows if row["situation_id"] == "obs-def-match"]
        mismatch_rows = [row for row in fold_rows if row["situation_id"] == "obs-def-mismatch"]
        self.assertTrue(matched_rows)
        self.assertTrue(all(row["summary_bucket"] == "observed" for row in matched_rows))
        self.assertTrue(all(row["returned_model_version"] == "v0.24" for row in matched_rows))
        self.assertTrue(all(str(row["model_dir"]).endswith("folds/foldA") for row in matched_rows))
        self.assertTrue(all(str(row["model_dir"]).endswith("folds/foldB") for row in mismatch_rows))
        self.assertTrue(mismatch_rows)
        self.assertTrue(all(row["summary_bucket"] == "indicative" for row in mismatch_rows))
        self.assertTrue(all(row["returned_model_version"] == "v0.21" and row["version_mismatch"] for row in mismatch_rows))
        v21_synth = versions["v0.21"]["by_side"]["offense"]["synthetic"]["by_stratum"]
        self.assertGreater(v21_synth["madden27/cpu/offense"]["disagreement"]["n"], 0)
        v24_defense = versions["v0.24"]["by_side"]["defense"]
        v24_observed = v24_defense["observed"]["by_stratum"]
        self.assertGreater(v24_observed["madden27/cpu/defense"]["disagreement"]["n"], 0)
        self.assertGreater(v24_defense["indicative"]["n"], 0)
        self.assertEqual(v24_defense["synthetic"]["n"], 0)
        for bucket in ("synthetic", "observed", "indicative", "refused"):
            self.assertEqual(versions["v0.21"]["by_side"]["defense"][bucket]["n"], 0)
            self.assertEqual(versions["v0.24"]["by_side"]["offense"][bucket]["n"], 0)
        refused_rows = [row for row in report["decisions"] if row["outcome"] == "refused"]
        self.assertTrue(refused_rows)
        self.assertTrue(all(row["outcome"] != "same" for row in refused_rows))
        self.assertIn("no_matching_fold", versions["refused_without_version"]["by_reason"])
        cpu = v21_synth["madden27/cpu/offense"]
        self.assertEqual(cpu["disagreement"]["k"] + cpu["same"], cpu["disagreement"]["n"])
        self.assertLessEqual(cpu["disagreement"]["low"], cpu["disagreement"]["rate"])
        self.assertGreaterEqual(cpu["disagreement"]["high"], cpu["disagreement"]["rate"])
        self.assertEqual(cpu["seed_noise"]["model"]["seed_shift_n"], 0)
        self.assertEqual(cpu["seed_noise"]["model"]["seeds_compared"], 2)
        self.assertIn("heuristic", cpu["seed_noise"])
        self.assertIsNone(cpu["penalty"])
        quality = cpu["by_down_and_distance_quality"]
        self.assertTrue(quality)
        self.assertIn("formation_usage", quality[0]["heuristic"])
        self.assertIn("formation_usage", quality[0]["model"])
        self.assertIn("rules", quality[0]["heuristic"])
        self.assertIsNone(quality[0]["penalty"])
        text = scoring.render_markdown(report)
        self.assertIn("passed its defined contextual rules, which cover a limited set of decisions.", text)
        self.assertIn("grouped by the model_version on the manifest that was loaded", text)
        self.assertIn("## Model version v0.24", text)
        self.assertIn("## Model version v0.21", text)
        self.assertNotIn("nothing ignored the game situation", text)
        self.assertNotIn("v0.23", json.dumps(versions["v0.21"]))
        with tempfile.TemporaryDirectory() as out:
            _json_path, _md_path, jsonl_path = scoring.write_report(report, Path(out))
            lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), len(report["decisions"]))
        recounted = sum(1 for row in lines if row["outcome"] == "disagree" and row["model_version"] == "v0.21" and row["summary_bucket"] == "synthetic" and row["stratum"] == "madden27/cpu/offense")
        self.assertEqual(recounted, cpu["disagree"])

    def test_v021_rank_matches_the_logistic_model(self):
        identity = dict(scoring.build_synthetic_identities()[0])
        identity["looks"] = identity["looks"][:1]
        report = scoring.evaluate_shadow(
            [identity],
            scoring.adapt_score_offline_model,
            self.book,
            model_dir=_ROOT / "tests" / "madden_ml" / "fixtures" / "item3" / "v0.21",
            requested_version="v0.21",
            seed_count=2,
        )
        rows = [
            row
            for row in report["decisions"]
            if row["stratum"] == "madden27/cpu/offense" and row["outcome"] != "refused"
        ]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["manifest_model_version"], "v0.21")
        self.assertEqual(row["scored_on"], "unreviewed")
        self.assertEqual(row["gate_bypassed"], "offline_eval")
        self.assertIsNone(row["offensive_personnel"])
        self.assertEqual(row["model_version"], "v0.21")
        direct = scoring.adapt_score_offline_model(
            identity["raw"],
            side="offense",
            model_dir=_ROOT / "tests" / "madden_ml" / "fixtures" / "item3" / "v0.21",
            strata=["madden27/cpu/offense"],
            opp_look=identity["looks"][0],
            seed=identity["seed"],
            offline_ok=True,
        )
        self.assertEqual(row["model_play"], direct["play"])
        self.assertEqual(row["model_formation"], direct["formation"])
        self.assertEqual(direct["gate_bypassed"], "offline_eval")
        self.assertTrue(row["model_play"])
        self.assertTrue(row["model_formation"])
        self.assertNotEqual(row["model_formation"], row["offensive_personnel"])
        self.assertIn(row["outcome"], {"same", "disagree"})
        noise = report["by_model_version"]["v0.21"]["by_side"]["offense"]["synthetic"]["by_stratum"]["madden27/cpu/offense"]["seed_noise"]
        self.assertEqual(noise["model"]["seed_shift_n"], 0)
        self.assertIn("heuristic", noise)
        defense = [
            item
            for item in report["decisions"]
            if item["side"] == "defense"
        ]
        self.assertEqual(defense, [])


def _write_bundle(
    directory: Path,
    version: str,
    fold_id: str | None = None,
    returned_version: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"label_version": version}
    if returned_version:
        manifest["model_version"] = returned_version
    if fold_id:
        manifest["fold_id"] = fold_id
    strata = {}
    for name in (
        "madden27/cpu/offense",
        "madden27/cpu/defense",
        "madden27/human/offense",
        "madden27/human/defense",
    ):
        strata[name] = {
            "model": {
                "intercept": 0.0,
                "coef": {"call": {"hb dive": 0.1}},
                "hyperparams": {
                    "lambda_situation": 1.0,
                    "lambda_entity": 1.0,
                    "lambda_interaction": 1.0,
                },
            }
        }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "model_params.json").write_text(
        json.dumps({"label_version": version, "strata": strata}),
        encoding="utf-8",
    )


def _defense_fold_identity(raw: str, fold_id: str, situation_id: str) -> dict:
    identity = scoring._identity(
        situation_id=situation_id,
        side="defense",
        tags=["observed"],
        source="observed",
        down=3,
        distance=2,
        yardline=12,
        field_zone="own_deep",
        clock_name="normal",
        score_name="close",
        game_id=None,
        default_filled=["score", "clock"],
        looks=[{"look_id": "file", "opp_personnel": "11", "opp_set": "nickel"}],
        look_observed=True,
        quarter=None,
        clock_seconds=None,
        two_minute=False,
        score_us=0,
        score_them=0,
    )
    identity["raw"] = raw
    identity["looks_from_file"] = True
    identity["source_tag"] = "film"
    identity["fold_id"] = fold_id
    identity["offensive_personnel"] = None
    return identity


class IsolationTests(unittest.TestCase):
    def test_scoring_does_not_write_home_or_call_path_home(self):
        source = (_ROOT / "scripts" / "madden_ml" / "eval_scoring.py").read_text(encoding="utf-8")
        self.assertNotIn("Path.home", source)
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        real_tree = real_home / ".cfb-coach"
        before = _snapshot(real_tree)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = root / "empty-home"
            empty.mkdir()
            previous = {key: os.environ.get(key) for key in ("HOME", "CFB_COACH_DB", "CFB_COACH_MADDEN_DB")}
            os.environ["HOME"] = str(empty)
            os.environ["CFB_COACH_DB"] = str(root / "coach.db")
            os.environ["CFB_COACH_MADDEN_DB"] = str(root / "madden27.db")
            try:
                book = scoring.load_installed_book(root / "madden27.db")
                request = _request("iso", "offense", "synthetic")
                decision = _call("Pistol Trips", "HB Dive")
                first = _harness(decision, decision, opp_look=request["opp_look"], seed=1)
                scoring.build_report(
                    [{"request": request, "first": first, "second": first}],
                    book,
                )
                self.assertEqual(list(empty.iterdir()), [])
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        self.assertEqual(_snapshot(real_tree), before)


def _passed(results, rule_id):
    match = next(item for item in results if item["id"] == rule_id)
    if not match["applies"]:
        return None
    return match["passed"]


def _request(row_id: str, side: str, source: str) -> dict:
    tags = ["synthetic"] if source == "synthetic" else ["observed_situation", "assigned_look"]
    return {
        "row_id": row_id,
        "situation_id": row_id,
        "side": side,
        "source": source,
        "tags": tags,
        "game_id": None,
        "down": 1,
        "distance": 10,
        "distance_bucket": "long",
        "yardline": 50,
        "field_zone": "midfield",
        "quarter": 2,
        "clock_bucket": "normal",
        "score_bucket": "close",
        "default_filled": [],
        "opp_look": {"opp_personnel": None, "opp_set": None},
        "seed": 1,
    }


def _snapshot(path: Path) -> tuple:
    if not path.exists():
        return ()
    return tuple(
        sorted(
            (item.relative_to(path).as_posix(), item.stat().st_size)
            for item in path.rglob("*")
        )
    )
