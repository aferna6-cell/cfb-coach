"""Every installed Madden offensive play stays tracked and selectable by ML."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.situation import Situation
from cfb_coach.madden import playbook
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import offense_inventory as inventory
from cfb_coach.madden.model import offense_design_browser as browser
from cfb_coach.madden.model.experimental_model import ExperimentalArtifact, rank_candidates
from cfb_coach.madden.model.experimental_live import situational_offense_candidates
from cfb_coach.madden.model.offense_selection_policy import choose_model_play


CATALOG = {
    "Buccaneers": {
        "Gun Bunch": [
            "Mesh", "Inside Zone", "Flood", "Four Verticals",
            "Quick Slants", "HB Slip Screen", "PA Boot", "HB Draw",
        ],
        "Gun Doubles": ["Slants", "Texas Y-Stutter Wheel", "Inside Zone", "Bench"],
    },
    "Lions": {
        "Gun Tight": ["Mesh", "HB Dive", "Smash", "Slants", "Flood", "Corner Post"],
    },
}


class FullInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = CoachDB(
            Path(self.temp.name) / "coach.db",
            seed={"opponents": {
                "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu",
                        "confidence": "low", "profile_json": "{}"}
            }},
        )
        self.old = {
            "side": "offense", "mode": "custom", "name": "Buccaneers",
            "rev": 2, "source_book": "Buccaneers",
            "trimmed": True, "formations": {"Gun Bunch": ["Mesh"]},
            "audibles": {}, "core": ["Gun Bunch"], "locked_ts": "test",
        }
        playbook._save_state(
            self.db, {"applied": {"offense": self.old}, "pending": {}}
        )
        self.patch = mock.patch.object(
            designer.catalog, "book_formations",
            side_effect=lambda side, source: CATALOG.get(source, {}),
        )
        self.patch.start()
        self.art = ExperimentalArtifact(
            model_version="inventory-test", n_supervised=10,
            evidence_quality="empirical_light", global_rate=0.52,
        )

    def tearDown(self):
        self.patch.stop()
        self.db.close()
        self.temp.cleanup()

    def plan(self):
        return designer.design_offense(
            self.db, opponent_id="cpu", max_formations=3,
            artifact=self.art, catalogue=CATALOG,
        )

    def install(self):
        pending = designer.stage_design(self.db, self.plan())
        result = designer.confirm_installed(
            self.db, proposal_id=pending["proposal_id"],
            attestation="I built and checked each complete stock-sourced formation in Madden.",
        )
        return pending, result

    def test_staging_does_not_replace_installed_playbook(self):
        draft = designer.stage_design(self.db, self.plan())
        active = inventory.installed_inventory(self.db)
        self.assertEqual(active["formations"], self.old["formations"])
        self.assertEqual(active["play_count"], 1)
        self.assertNotEqual(active["inventory_id"], draft["inventory_id"])
        self.assertEqual(inventory.inventory_report(self.db)["unused_plays"], 1)

    def test_confirmed_book_includes_every_stock_play_and_fingerprint(self):
        pending, result = self.install()
        active = inventory.installed_inventory(self.db)
        self.assertTrue(active["installed"])
        self.assertEqual(active["formations"], pending["book"]["formations"])
        self.assertEqual(active["inventory_id"], pending["inventory_id"])
        self.assertEqual(result["inventory_id"], active["inventory_id"])
        self.assertEqual(result["plays"], active["play_count"])
        for name, plays in active["formations"].items():
            source = active["formation_sources"][name]
            self.assertEqual(plays, CATALOG[source][name])
        self.assertEqual(
            len(inventory.pairs_in_inventory(active["formations"])),
            active["play_count"],
        )

    def test_catalog_change_incomplete_full_formation_cannot_be_confirmed(self):
        plan = designer.stage_design(self.db, self.plan())
        target = next(iter(plan["book"]["formations"]))
        source = plan["book"]["formation_sources"][target]
        source_data = {k: dict(v) for k, v in CATALOG.items()}
        source_data[source][target] = source_data[source][target][:-1]
        with mock.patch.object(
            designer.catalog, "book_formations",
            side_effect=lambda side, name: source_data.get(name, {}),
        ):
            with self.assertRaisesRegex(ValueError, "incomplete or diverged"):
                designer.confirm_installed(
                    self.db, proposal_id=plan["proposal_id"],
                    attestation="I installed the whole formation and checked each play.",
                )
        self.assertEqual(
            inventory.installed_inventory(self.db)["formations"],
            self.old["formations"],
        )

    def test_live_pool_and_ranker_consider_every_eligible_installed_play(self):
        self.install()
        locked = playbook.eligible(playbook.active_books(self.db, ("offense",)))["offense"]
        sit = Situation(
            raw="1&10 own 40", down=1, distance=10, yardline=40,
            side="offense",
        )
        pool, _ = situational_offense_candidates(sit, locked)
        all_pairs = inventory.pairs_in_inventory(locked)
        self.assertEqual(set(pool), set(all_pairs))
        rankings = rank_candidates(self.art, pool, down=1, distance=10,
                                   opponent_id="cpu", opponent_type="cpu")
        self.assertEqual(len(rankings), len(all_pairs))
        _, audit = choose_model_play(rankings, sit=sit, session_id="g", snap_seq=1)
        self.assertEqual(audit["full_eligible_population_count"], len(all_pairs))

    def test_large_full_playbook_not_cut_to_24_or_two_per_concept(self):
        candidates = [
            {"formation": f"Gun Formation {i//12}", "play": f"Pass Play {i:02d}",
             "probability": 0.65 - i * .001, "play_concept": "pass",
             "uncertainty": .7, "evidence_quality": "empirical_light"}
            for i in range(60)
        ]
        selections = []
        for n in range(1, 251):
            ranked, audit = choose_model_play(
                candidates, opponent_type="cpu",
                sit=Situation(raw="1&10", down=1, distance=10),
                session_id="all-plays-g", snap_seq=n,
            )
            selections.append(ranked[0]["play"])
            self.assertEqual(audit["full_eligible_population_count"], 60)
            self.assertEqual(audit["shortlist_count"], 60)
        self.assertTrue(
            any(int(play.rsplit(" ", 1)[1]) >= 24 for play in selections),
            "No selection beyond the former top-24 limitation",
        )
        self.assertGreaterEqual(len(set(selections)), 20)

    def test_coverage_report_counts_recommendations_but_not_executions(self):
        self.install()
        forms = inventory.installed_inventory(self.db)["formations"]
        form, play = next((f, p) for f, ps in forms.items() for p in ps)
        for idx in range(2):
            self.db.log_snap(
                opponent_id="cpu", side="offense", situation_raw="1&10",
                our_call=f"{form} — {play}", formation=form, play=play,
                session_id="test-g", result="+5",
                executed_status="unknown", executed_verification="unknown",
            )
        report = inventory.inventory_report(self.db, opponent_id="cpu")
        self.assertEqual(report["used_plays"], 1)
        self.assertEqual(report["unused_plays"], report["play_count"] - 1)
        entry = next(r for r in report["play_usage"]
                     if (r["formation"], r["play"]) == (form, play))
        self.assertEqual(entry["recommended_calls"], 2)
        self.assertIn("not proof of execution", report["counting_note"])
        scoped = inventory.inventory_report(self.db, game_id="other-g")
        self.assertEqual(scoped["used_plays"], 0)
        self.assertEqual(scoped["unused_plays"], scoped["play_count"])

    def test_browser_lists_all_plays_and_recommendation_coverage(self):
        self.install()
        actual = browser.installed_design(self.db, "cpu")
        html = browser.render_html(self.db, actual, mode="installed")
        report = inventory.installed_inventory(self.db)
        self.assertIn(report["inventory_id"], html)
        self.assertIn(f"0 / {report['play_count']}", html)
        for _form, plays in report["formations"].items():
            for play in plays:
                self.assertIn(play, html)
        self.assertIn("calls recommended", html)


if __name__ == "__main__":
    unittest.main()
