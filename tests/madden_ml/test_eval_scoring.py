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
                "offense,3,1,97,4,syn-game-01\n"
                "D,2,8,22,1,syn-game-02\n",
                encoding="utf-8",
            )
            rows = scoring.load_observed_csv(path)
        self.assertEqual(len(rows), 2)
        first, second = rows
        self.assertEqual(first["tags"], ["observed"])
        self.assertEqual((first["down"], first["distance"], first["yardline"], first["quarter"]), (3, 1, 97, 4))
        self.assertEqual(first["game_id"], "syn-game-01")
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
        self.assertEqual(second["game_id"], "syn-game-02")
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
        self.assertEqual(len(source), 160)
        self.assertEqual(
            list(source[0].keys()),
            ["row", "game", "side", "down", "distance", "yardline", "quarter", "tag"],
        )
        loaded = scoring.load_observed_csv(scoring.OBSERVED_CSV)
        self.assertEqual(len(loaded), 160)
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
        self.assertEqual(len(expanded), 160 * 3)
        self.assertTrue(
            all(row["tags"] == ["observed_situation", "assigned_look"] for row in expanded)
        )
        self.assertTrue(all(row["look_observed"] is False for row in expanded))
        identities, note = scoring.load_situation_identities()
        counts = note["fixture_counts"]
        self.assertEqual(len(identities), 72 + 160)
        self.assertEqual(counts["identity_tags_by_side"]["synthetic"], {"offense": 36, "defense": 36})
        self.assertEqual(counts["identity_tags_by_side"]["observed"], {"offense": 160, "defense": 0})
        self.assertEqual(
            counts["decision_tags_by_side"]["synthetic"],
            {"offense": 144, "defense": 144},
        )
        self.assertEqual(
            counts["decision_tags_by_side"]["observed_situation,assigned_look"],
            {"offense": 480, "defense": 0},
        )
        self.assertEqual(counts["identities"], 232)
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

    def test_adapter_coerces_csv_text_and_keeps_a_package_look_as_no_coverage(self):
        sit = {
            "raw": "1&10 my 50",
            "down": "1",
            "distance": "10",
            "yardline": "  ",
            "look_id": "11-nickel",
        }
        missing_columns = scoring._prepare_hook_call(sit, None, "offense")
        self.assertIsNone(missing_columns["opp_look"])
        self.assertTrue(missing_columns["diagnostics"]["no_coverage_in_source"])
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
            covered = scoring.adapt_score_offline_model(
                {
                    "raw": "1&10 my 50",
                    "down": "1",
                    "distance": "10",
                    "yardline": "50",
                    "coverage_seen": "Cover 3",
                },
                side="offense",
                model_dir=model,
                strata=["madden27/cpu/offense"],
                opp_look=None,
                offline_ok=True,
            )
        diag = payload["input_diagnostics"]
        self.assertEqual(diag["down_type"], "str")
        self.assertEqual(diag["distance_type"], "str")
        self.assertEqual(diag["yardline_type"], "str")
        self.assertEqual(diag["down"], 1)
        self.assertEqual(diag["distance"], 10)
        self.assertIsNone(diag["yardline"])
        self.assertIsNone(diag["opp_look"])
        self.assertEqual(diag["opp_look_type"], "NoneType")
        self.assertTrue(diag["no_coverage_in_source"])
        self.assertEqual(diag["opp_look_assigned"], {"opp_personnel": "11", "opp_set": "nickel"})
        self.assertIsNone(payload["features"]["groups"]["look"]["key"])
        self.assertEqual(covered["input_diagnostics"]["opp_look"], "cover 3")
        self.assertEqual(covered["input_diagnostics"]["prep_look_family"], "cover_3")
        self.assertEqual(covered["features"]["groups"]["look_family"]["key"], "cover_3")
        blank = scoring._prepare_hook_call(
            {"down": None, "distance": "", "yardline": "12", "look_id": "null"},
            {"opp_personnel": None, "opp_set": None},
            "offense",
        )
        self.assertIsNone(blank["opp_look"])
        self.assertTrue(blank["diagnostics"]["no_coverage_in_source"])
        package = scoring._prepare_hook_call(
            {"raw": "1&10 my 50", "down": 1, "distance": 10, "yardline": 50, "look_id": "11-nickel"},
            {"opp_personnel": None, "opp_set": None},
            "offense",
        )
        self.assertIsNone(package["opp_look"])
        self.assertTrue(package["diagnostics"]["no_coverage_in_source"])
        defense = scoring._prepare_hook_call(
            {"raw": "d 1&10 my 50", "down": 1, "distance": 10, "yardline": 50, "look_id": "11-nickel"},
            {"opp_personnel": None, "opp_set": None},
            "defense",
        )
        self.assertIsNone(defense["opp_look"])
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
        self.assertNotIn("160", text)


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

    def test_assigned_look_reaches_the_hook_and_flips_on_the_fixture(self):
        fixture = _ROOT / "tests" / "madden_ml" / "fixtures" / "item3" / "v0.21"
        synthetic = next(
            row for row in scoring.build_synthetic_identities() if row["situation_id"] == "syn-offense-028"
        )
        self.assertEqual({look["look_id"] for look in synthetic["looks"]}, {"null", "empty", "11-nickel", "12-base"})
        observed = scoring.load_observed_csv(scoring.OBSERVED_CSV)[0]
        self.assertTrue(any(look["look_id"] not in {"", "null"} for look in observed["looks"]))
        covered = _paired_identity(
            "sit-cover",
            "offense",
            1,
            10,
            50,
            "midfield",
            [
                {
                    "look_id": "cover",
                    "opp_personnel": None,
                    "opp_set": None,
                    "def_call": "Nickel Double Mug",
                    "coverage_seen": "Cover 3",
                    "def_overlay": "Cover 2",
                }
            ],
        )
        report = scoring.evaluate_shadow(
            [synthetic, observed, covered],
            scoring.adapt_score_offline_model,
            self.book,
            model_dir=fixture,
            requested_version="v0.21",
            seed_count=1,
        )

        def cpu(situation_id, look_id):
            matches = [
                row
                for row in report["decisions"]
                if row["situation_id"] == situation_id
                and row["look_id"] == look_id
                and row["stratum"] == "madden27/cpu/offense"
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        nickel = cpu("syn-offense-028", "11-nickel")
        self.assertIsNone(nickel["hook_look"])
        self.assertIsNone(nickel["features"]["groups"]["look"]["key"])
        self.assertFalse(nickel["features"]["groups"]["look"]["matched"])
        self.assertFalse(nickel["look_key_matched"])
        self.assertFalse(nickel["look_family_key_matched"])
        self.assertTrue(nickel["no_coverage_in_source"])
        self.assertIsNone(nickel["input_diagnostics"]["opp_look"])
        base = cpu("syn-offense-028", "12-base")
        self.assertIsNone(base["hook_look"])
        self.assertFalse(base["look_key_matched"])
        self.assertFalse(base["look_family_key_matched"])
        empty = cpu("syn-offense-028", "empty")
        self.assertIsNone(empty["hook_look"])
        self.assertFalse(empty["look_key_matched"])
        missing = cpu("syn-offense-028", "null")
        self.assertIsNone(missing["hook_look"])
        self.assertIsNone(missing["features"]["groups"]["look"]["key"])
        self.assertEqual(missing["features"]["groups"]["look_family"]["key"], "none")
        self.assertFalse(missing["look_family_key_matched"])
        self.assertEqual(missing["model_pick"], nickel["model_pick"])

        observed_live = [
            cpu(observed["situation_id"], look["look_id"])
            for look in observed["looks"]
            if look["look_id"] not in {"", "null"}
        ]
        self.assertGreaterEqual(len(observed_live), 1)
        for row in observed_live:
            self.assertIsNone(row["hook_look"])
            self.assertTrue(row["no_coverage_in_source"])
            self.assertFalse(row["look_key_matched"])
            self.assertFalse(row["look_family_key_matched"])

        without = report["no_coverage_in_source"]
        self.assertGreater(without["n"], 0)
        self.assertTrue(any(str(row_id).endswith("__null__madden27/cpu/offense") for row_id in without["row_ids"]))
        self.assertTrue(any("11-nickel" in str(row_id) for row_id in without["row_ids"]))
        self.assertTrue(any(str(row_id).endswith("__empty__madden27/cpu/offense") for row_id in without["row_ids"]))

        cover = cpu("sit-cover", "cover")
        self.assertEqual(cover["hook_look"], "cover 3")
        self.assertEqual(cover["prep_look"], "cover 3")
        self.assertEqual(cover["prep_look_family"], "cover_3")
        self.assertEqual(cover["features"]["groups"]["look_family"]["key"], "cover_3")
        self.assertTrue(cover["features"]["groups"]["look_family"]["matched"])
        self.assertFalse(cover["look_key_matched"])
        self.assertTrue(cover["look_family_key_matched"])
        self.assertFalse(cover["no_coverage_in_source"])
        self.assertNotIn("sit-cover", "".join(str(row_id) for row_id in without["row_ids"]))

        summary = report["by_model_version"]["v0.21"]["by_side"]["offense"]["synthetic"]["by_stratum"]["madden27/cpu/offense"]["paired_summary"]
        self.assertEqual(summary["look_flip_count"], 0)
        self.assertEqual(summary["look_key_matches"], 0)
        self.assertEqual(summary["look_family_key_matches"], 1)
        self.assertEqual(summary["look_family_key_match_share"], 1 / summary["look_match_n"])

        self.assertEqual(nickel["field_zone"], "midfield")
        self.assertEqual(nickel["yardline"], 50)
        self.assertEqual(nickel["trainer_zone"], "opp")
        self.assertIn("midfield", nickel["zone_note"])
        self.assertIn("opp", nickel["zone_note"])
        self.assertIn("Trainer zone buckets", nickel["zone_note"])
        self.assertIn("yardline passed to the hook is unchanged", nickel["zone_note"])


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
                "coef": {
                    "call": {"hb dive": 0.1},
                    "look_family": {
                        "cover_3": 0.0,
                        "cover_1": 0.0,
                        "none": 0.0,
                        "other": 0.0,
                    },
                },
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


def _paired_identity(situation_id, side, down, distance, yardline, zone, looks):
    return scoring._identity(
        situation_id=situation_id,
        side=side,
        tags=["synthetic"],
        source="synthetic",
        down=down,
        distance=distance,
        yardline=yardline,
        field_zone=zone,
        clock_name="normal",
        score_name="close",
        game_id=None,
        default_filled=[],
        looks=looks,
        look_observed=False,
    )


_NULL_LOOK = {"look_id": "null", "opp_personnel": None, "opp_set": None}
_EMPTY_LOOK = {"look_id": "empty", "opp_personnel": "empty", "opp_set": "empty"}
_SENTINEL_FEATURES = {
    "intercept": 0.0,
    "groups": {
        "not_a_real_group": {"key": "sentinel-key", "matched": True, "weight": 0.25},
    },
    "call_matches": ["sentinel-key"],
}


def _scripted_payload(seed, model_play, heuristic_play, *, weights=None, features=None):
    heuristic = {
        "formation": "Gun Trips",
        "play": heuristic_play,
        "adjustment_id": None,
        "mode": "heuristic",
    }
    if weights is not None:
        heuristic["candidate_probabilities"] = weights
    return {
        "formation": "Pistol Trips",
        "play": model_play,
        "adjustment_id": None,
        "mode": "offline",
        "model_version": "v0.21",
        "scored_on": "reviewed",
        "legal": True,
        "gate_bypassed": "offline_eval",
        "heuristic": heuristic,
        "seed": seed,
        "features": features if features is not None else _SENTINEL_FEATURES,
    }


def _scripted_runner(sit, *, side, model_dir, strata, opp_look, seed, offline_ok):
    del model_dir, strata, opp_look
    if offline_ok is not True:
        raise AssertionError("offline_ok")
    seed = int(seed)
    down = int(sit["down"])
    distance = int(sit["distance"])
    look_id = sit["look_id"]
    if side == "defense":
        model = "Cover 1" if seed < 9 else "Cover 2"
        return _scripted_payload(seed, model, "Cover 4")
    if (down, distance) == (1, 10):
        heuristic = "alpha" if seed < 5 else "zeta"
        return _scripted_payload(seed, "alpha", heuristic)
    if (down, distance) == (2, 4):
        plays = ["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"]
        return _scripted_payload(seed, "a", plays[seed])
    if (down, distance) == (3, 10):
        return _scripted_payload(
            seed,
            "screen",
            "dive",
            weights={"screen": 0.15, "dive": 0.85},
        )
    if (down, distance) == (4, 10):
        return _scripted_payload(seed, "never", "dive")
    if (down, distance) == (1, 2):
        model = "dive" if look_id == "null" else "screen"
        return _scripted_payload(seed, model, "dive")
    if (down, distance) == (1, 4):
        return _scripted_payload(seed, "dive", "dive")
    raise AssertionError(f"unexpected row {side} {down} {distance} {look_id}")


class PairedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        madden = root / "madden27.db"
        coach = root / "coach.db"
        cls.resources = scoring.set_shadow_databases(madden, coach)
        bundle = root / "bundle"
        _write_bundle(bundle, "v0.21")
        looks_one = [_NULL_LOOK]
        looks_two = [_NULL_LOOK, _EMPTY_LOOK]
        looks_flip = [
            {**_NULL_LOOK, "coverage_seen": "Cover 3"},
            {**_EMPTY_LOOK, "coverage_seen": "Cover 1"},
        ]
        identities = [
            _paired_identity("sit-tie", "offense", 1, 10, 50, "midfield", looks_one),
            _paired_identity("sit-unstable", "offense", 2, 4, 50, "midfield", looks_one),
            _paired_identity("sit-weights", "offense", 3, 10, 50, "midfield", looks_one),
            _paired_identity("sit-zero", "offense", 4, 10, 50, "midfield", looks_one),
            _paired_identity("sit-flip", "offense", 1, 2, 50, "midfield", looks_flip),
            _paired_identity("sit-same", "offense", 1, 4, 50, "midfield", looks_two),
            _paired_identity("sit-def", "defense", 3, 2, 12, "own_deep", looks_one),
        ]
        cls.report = scoring.evaluate_shadow(
            identities,
            _scripted_runner,
            cls.resources.book,
            model_dir=bundle,
            requested_version="v0.21",
            seed_count=10,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _cpu(self, situation_id, look_id="null", side="offense"):
        stratum = f"madden27/cpu/{side}"
        matches = [
            row
            for row in self.report["decisions"]
            if row["situation_id"] == situation_id
            and row["look_id"] == look_id
            and row["stratum"] == stratum
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_columns_tie_break_probability_and_defense(self):
        self.assertEqual(self.report["seeds"], list(range(10)))
        self.assertIn("lexicographic play name", self.report["heuristic_tie_break"])
        tie = self._cpu("sit-tie")
        self.assertEqual(tie["down"], 1)
        self.assertEqual(tie["distance"], 10)
        self.assertEqual(tie["field_zone"], "midfield")
        self.assertEqual(tie["look_id"], "null")
        self.assertEqual(tie["stratum"], "madden27/cpu/offense")
        self.assertEqual(tie["model_pick"], "alpha")
        self.assertTrue(tie["model_seed_stable"])
        self.assertEqual([item["seed"] for item in tie["heuristic_picks"]], list(range(10)))
        self.assertEqual(
            [item["play"] for item in tie["heuristic_picks"]],
            ["alpha"] * 5 + ["zeta"] * 5,
        )
        self.assertEqual(tie["heuristic_majority_pick"], "alpha")
        self.assertEqual(tie["heuristic_majority_share"], 5 / 10)
        self.assertAlmostEqual(tie["heuristic_seed_change_rate"], 1 / 9)
        self.assertTrue(tie["stable_heuristic"])
        self.assertEqual(tie["agreement"], "agree")
        self.assertIs(tie["agree_with_majority"], True)
        self.assertEqual(tie["model_pick_heuristic_prob"], 0.5)
        self.assertEqual(tie["prob_source"], "seed_frequency_estimate")
        self.assertEqual(tie["features"]["groups"]["not_a_real_group"]["key"], "sentinel-key")
        self.assertIs(tie["features"]["groups"]["not_a_real_group"]["matched"], True)
        self.assertTrue(tie["evaluable"])
        self.assertIsNone(tie["evaluable_reason"])

        unstable = self._cpu("sit-unstable")
        self.assertEqual(unstable["heuristic_majority_pick"], "a")
        self.assertEqual(unstable["heuristic_majority_share"], 4 / 10)
        self.assertFalse(unstable["stable_heuristic"])
        self.assertEqual(unstable["agreement"], "no_stable_heuristic_pick")
        self.assertIsNone(unstable["agree_with_majority"])
        self.assertAlmostEqual(unstable["heuristic_seed_change_rate"], 3 / 9)
        self.assertEqual(unstable["prob_source"], "seed_frequency_estimate")
        self.assertEqual(unstable["model_pick_heuristic_prob"], 0.4)

        weights = self._cpu("sit-weights")
        self.assertEqual(weights["model_pick_heuristic_prob"], 0.15)
        self.assertEqual(weights["prob_source"], "heuristic_weights")
        self.assertEqual(weights["agreement"], "disagree")
        self.assertIs(weights["agree_with_majority"], False)
        self.assertEqual(weights["heuristic_majority_pick"], "dive")
        self.assertTrue(weights["stable_heuristic"])

        zero = self._cpu("sit-zero")
        self.assertEqual(zero["model_pick_heuristic_prob"], "<0.1")
        self.assertEqual(zero["prob_source"], "seed_frequency_estimate")
        self.assertEqual(zero["agreement"], "disagree")

        other_weight_keys = [
            {"seed": 0, "play": "screen", "heuristic": {"play": "dive", "weights": {"screen": 0.2}}},
            {"seed": 0, "play": "screen", "heuristic": {"play": "dive", "sampling_weights": {"screen": 0.4}}},
        ]
        prob, source = scoring._model_pick_probability("screen", ["dive"], other_weight_keys[:1])
        self.assertEqual((prob, source), (0.2, "heuristic_weights"))
        prob, source = scoring._model_pick_probability("screen", ["dive"], other_weight_keys[1:])
        self.assertEqual((prob, source), (0.4, "heuristic_weights"))
        missing, source = scoring._model_pick_probability(
            "absent",
            ["dive"],
            [{"seed": 0, "play": "absent", "heuristic": {"play": "dive", "weights": {"dive": 1.0}}}],
        )
        self.assertEqual((missing, source), (0.0, "heuristic_weights"))

        defense = self._cpu("sit-def", side="defense")
        self.assertFalse(defense["evaluable"])
        self.assertEqual(defense["evaluable_reason"], scoring.DEFENSE_PENDING_REASON)
        self.assertEqual(defense["model_pick"], "Cover 1")
        self.assertFalse(defense["model_seed_stable"])
        self.assertEqual(len(defense["heuristic_picks"]), 10)
        defense_summary = self.report["by_model_version"]["v0.21"]["by_side"]["defense"]["synthetic"]["by_stratum"]["madden27/cpu/defense"]["paired_summary"]
        self.assertFalse(defense_summary["evaluable"])
        self.assertEqual(defense_summary["reason"], "defense path pending (call keyed by offense play)")
        self.assertNotIn("agreement_vs_majority", defense_summary)
        self.assertNotIn("n_situations", defense_summary)
        self.assertNotIn("n_stable", defense_summary)
        self.assertGreater(defense_summary["n_rows_excluded"], 0)

        flip_null = self._cpu("sit-flip", "null")
        flip_empty = self._cpu("sit-flip", "empty")
        self.assertEqual(flip_null["prep_look_family"], "cover_3")
        self.assertEqual(flip_empty["prep_look_family"], "cover_1")
        self.assertTrue(flip_null["look_family_key_matched"])
        self.assertTrue(flip_empty["look_family_key_matched"])
        self.assertFalse(flip_null["look_key_matched"])
        self.assertNotEqual(flip_null["model_pick"], flip_empty["model_pick"])

        summary = self.report["by_model_version"]["v0.21"]["by_side"]["offense"]["synthetic"]["by_stratum"]["madden27/cpu/offense"]["paired_summary"]
        self.assertTrue(summary["evaluable"])
        self.assertEqual(summary["n_situations"], 8)
        self.assertEqual(summary["n_stable"], 7)
        self.assertEqual(summary["agreement_vs_majority"], scoring.wilson_interval(4, 7))
        self.assertAlmostEqual(summary["mean_heuristic_seed_change_rate"], (1 / 9 + 1 / 3) / 8)
        self.assertAlmostEqual(summary["median_model_pick_heuristic_prob"], 0.45)
        self.assertEqual(summary["look_flip_count"], 1)
        self.assertEqual(summary["situation_changes_pick_within_look"], 2)
        human = self.report["by_model_version"]["v0.21"]["by_side"]["offense"]["synthetic"]["by_stratum"]["madden27/human/offense"]["paired_summary"]
        self.assertEqual(human["n_situations"], summary["n_situations"])
        self.assertEqual(human["agreement_vs_majority"], summary["agreement_vs_majority"])
        text = scoring.render_markdown(self.report)
        self.assertIn("lexicographic play name", text)
        self.assertIn("defense path pending (call keyed by offense play)", text)
        self.assertIn("paired summary excluded", text)

    def test_trimmed_fixture_carries_hook_features_and_seed_zero_pick(self):
        source = (_ROOT / "scripts" / "madden_ml" / "eval_scoring.py").read_text(encoding="utf-8")
        self.assertNotIn("feature_map", source)
        self.assertNotIn("ALL_GROUPS", source)
        self.assertNotIn("snap_from_row", source)
        fixture = _ROOT / "tests" / "madden_ml" / "fixtures" / "item3" / "v0.21"
        size = sum(path.stat().st_size for path in fixture.rglob("*") if path.is_file())
        self.assertLess(size, 20_000)
        identity = dict(scoring.build_synthetic_identities()[0])
        identity["looks"] = [identity["looks"][0]]
        report = scoring.evaluate_shadow(
            [identity],
            scoring.adapt_score_offline_model,
            self.resources.book,
            model_dir=fixture,
            requested_version="v0.21",
            seed_count=10,
        )
        row = next(
            item
            for item in report["decisions"]
            if item["stratum"] == "madden27/cpu/offense"
        )
        direct = scoring.adapt_score_offline_model(
            {
                "raw": identity["raw"],
                "down": identity["down"],
                "distance": identity["distance"],
                "yardline": identity["yardline"],
                "look_id": identity["looks"][0]["look_id"],
            },
            side="offense",
            model_dir=fixture,
            strata=["madden27/cpu/offense"],
            opp_look=identity["looks"][0],
            seed=0,
            offline_ok=True,
        )
        self.assertEqual(row["model_pick"], direct["play"])
        self.assertTrue(row["model_seed_stable"])
        self.assertEqual([item["seed"] for item in row["heuristic_picks"]], list(range(10)))
        self.assertEqual(row["features"], direct["features"])
        self.assertTrue(row["features"]["groups"])
        for group in row["features"]["groups"].values():
            self.assertIn("key", group)
            self.assertIn("matched", group)
            self.assertIsInstance(group["matched"], bool)
        self.assertEqual(row["prob_source"], "seed_frequency_estimate")
        self.assertIn(
            row["agreement"],
            {"agree", "disagree", "no_stable_heuristic_pick"},
        )
        if row["stable_heuristic"]:
            self.assertIsInstance(row["agree_with_majority"], bool)
        else:
            self.assertEqual(row["agreement"], "no_stable_heuristic_pick")
            self.assertIsNone(row["agree_with_majority"])
        share = row["heuristic_majority_share"]
        self.assertAlmostEqual(share * 10, round(share * 10))
        summary = report["by_model_version"]["v0.21"]["by_side"]["offense"]["synthetic"]["by_stratum"]["madden27/cpu/offense"]["paired_summary"]
        self.assertEqual(summary["n_situations"], 1)
        defense_summary = report["by_model_version"]["v0.21"]["by_side"]["defense"]["synthetic"]["by_stratum"]
        self.assertEqual(defense_summary, {})


class LookConstructionTests(unittest.TestCase):
    def test_prep_order_uses_the_trainer_family_and_hard_fails_only_on_dropped_coverage(self):
        prep = scoring.build_prep_look(
            {
                "def_call": "Nickel Double Mug",
                "coverage_seen": "Cover 3",
                "def_overlay": "Cover 2",
            },
            {"opp_personnel": "11", "opp_set": "nickel"},
        )
        # def_call is never a look or family source (recipe f7c17d16 addendum).
        self.assertEqual(prep["look"], "cover 3")
        self.assertEqual(prep["family"], "cover_3")
        call_only = scoring.build_prep_look({"def_call": "Cover 1", "coverage_seen": ""}, None)
        self.assertIsNone(call_only["look"])
        self.assertEqual(call_only["status"], "blank_coverage")
        self.assertEqual(prep["status"], "coverage")
        seen_only = scoring.build_prep_look({"coverage_seen": "Cover 3"}, None)
        self.assertEqual(seen_only["look"], "cover 3")
        self.assertEqual(seen_only["family"], scoring._look_family("Cover 3"))
        overlay = scoring.build_prep_look({"def_overlay": "Cover 1"}, None)
        self.assertEqual(overlay["look"], "cover 1")
        self.assertEqual(overlay["family"], "cover_1")
        blank = scoring.build_prep_look({"def_call": "  ", "coverage_seen": None}, None)
        self.assertIsNone(blank["look"])
        self.assertEqual(blank["status"], "blank_coverage")
        self.assertTrue(blank["columns_present"])
        missing = scoring.build_prep_look(
            {"look_id": "11-nickel"},
            {"opp_personnel": "11", "opp_set": "nickel"},
        )
        self.assertIsNone(missing["look"])
        self.assertEqual(missing["status"], "no_coverage_in_source")
        scoring._raise_if_coverage_dropped(columns_present=False, nonblank=False, look=None)
        scoring._raise_if_coverage_dropped(columns_present=True, nonblank=False, look=None)
        with self.assertRaises(RuntimeError) as caught:
            scoring.build_prep_look({"coverage_seen": {"dropped": "Cover 3"}}, None)
        self.assertIn("coverage columns", str(caught.exception))
        self.assertIn("empty", str(caught.exception))
        self.assertFalse(scoring._is_trained_match("nickel", {"nickel", "base", "empty"}, exclude_packages=True))
        self.assertFalse(scoring._is_trained_match("none", {"none", "cover_3"}, exclude_packages=False))
        self.assertFalse(scoring._is_trained_match("other", {"other", "cover_3"}, exclude_packages=False))
        self.assertTrue(scoring._is_trained_match("cover_3", {"cover_3", "none"}, exclude_packages=False))
        fallback = [
            {
                "situation_id": "s",
                "look_id": "nickel",
                "model_pick": "Inside Zone",
                "look_key_matched": False,
                "look_family_key_matched": False,
            },
            {
                "situation_id": "s",
                "look_id": "null",
                "model_pick": "HB Dive",
                "look_key_matched": False,
                "look_family_key_matched": False,
            },
        ]
        self.assertEqual(scoring._look_flip_count(fallback), 0)
        real = [
            {
                "situation_id": "s",
                "look_id": "c3",
                "model_pick": "dive",
                "look_key_matched": False,
                "look_family_key_matched": True,
            },
            {
                "situation_id": "s",
                "look_id": "c1",
                "model_pick": "screen",
                "look_key_matched": False,
                "look_family_key_matched": True,
            },
        ]
        self.assertEqual(scoring._look_flip_count(real), 1)

    def test_label_snaps_report_match_share_without_hardcoded_ids(self):
        source = (_ROOT / "scripts" / "madden_ml" / "eval_scoring.py").read_text(encoding="utf-8")
        self.assertNotIn("four_game_ids", source)
        self.assertNotIn("F8Hm324SMtg", source)
        fixture = _ROOT / "tests" / "madden_ml" / "fixtures" / "item3" / "v0.21"
        snaps = [
            {
                "video_id": "synthetic-off",
                "side": "offense",
                "stratum": "madden27/cpu/offense",
                "defense_look": {
                    "def_call": "cover 3 buzz match",
                    "coverage_seen": "Cover 3",
                    "def_overlay": None,
                },
            },
            {
                "video_id": "synthetic-def",
                "side": "defense",
                "stratum": "madden27/cpu/defense",
                "defense_look": {
                    "def_call": "cover 3 buzz match",
                    "coverage_seen": "Cover 3",
                    "def_overlay": None,
                },
            },
            {
                "video_id": "F8Hm324SMtg",
                "side": "offense",
                "stratum": "madden27/cpu/offense",
                "defense_look": {"coverage_seen": "Cover 3"},
            },
            {
                "game_id": "audit-game-1",
                "side": "offense",
                "stratum": "madden27/human/offense",
                "defense_look": {"coverage_seen": "Cover 1"},
            },
            {
                "video_id": "synthetic-package",
                "side": "offense",
                "stratum": "madden27/cpu/offense",
                "defense_look": {"opp_personnel": "11", "opp_set": "nickel"},
            },
        ]
        exclude = ["F8Hm324SMtg", "audit-game-1", "audit-game-2", "audit-game-3", "audit-game-4"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snaps.json"
            path.write_text(json.dumps(snaps), encoding="utf-8")
            loaded = scoring.load_label_snaps(path)
        self.assertEqual(len(loaded), 5)
        check = scoring.label_look_key_check(loaded, fixture, exclude)
        self.assertEqual(check["label"], "training rows, key-construction check, not held-out")
        self.assertEqual(check["excluded_n"], 2)
        offense = check["by_stratum"]["madden27/cpu/offense"]
        defense = check["by_stratum"]["madden27/cpu/defense"]
        self.assertEqual(offense["side"], "offense")
        self.assertEqual(defense["side"], "defense")
        self.assertEqual(offense["n"], 2)
        self.assertEqual(defense["n"], 1)
        self.assertEqual(offense["look_key_matches"], 0)
        self.assertEqual(offense["look_family_key_matches"], 1)
        self.assertEqual(offense["look_family_key_match_share"], 0.5)
        self.assertEqual(defense["look_family_key_matches"], 1)
        self.assertEqual(defense["look_family_key_match_share"], 1.0)
        self.assertNotIn("madden27/human/offense", check["by_stratum"])
        text = scoring.render_markdown({"by_model_version": {}, "label_look_key_check": check, "disclaimer": scoring.DISCLAIMER, "no_coverage_in_source": {"n": 0}, "vod_derived_excluded": [], "requested_model_version": "v0.21"})
        self.assertIn("training rows, key-construction check, not held-out", text)


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
