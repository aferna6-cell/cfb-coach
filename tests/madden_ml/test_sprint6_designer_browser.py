"""Browser preview and installation-state tests for the Madden ML designer."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import offense_design_browser as browser
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model import cli as ml_cli


BOOK = {
    "Buccaneers": {
        "Gun Bunch": ["Mesh", "Inside Zone", "Flood", "Four Verticals"],
        "Singleback Wing": ["Stretch", "PA Boot", "Quick Slants"],
    },
    "Lions": {
        "Gun Tight": ["Mesh", "HB Dive", "Smash", "Slants"],
        "Gun Trips": ["Inside Zone", "Verticals", "PA Cross"],
    },
}


class DesignerBrowserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed={
            "opponents": {"cpu": {
                "display_name": "CPU", "team_now": "DET", "skill": "cpu",
                "confidence": "low", "profile_json": "{}",
            }}
        })
        self.old = {
            "side": "offense", "mode": "custom", "name": "Buccaneers",
            "source_book": "Buccaneers", "rev": 1, "trimmed": True,
            "formations": {"Gun Bunch": ["Mesh", "Inside Zone"]},
            "audibles": {}, "core": ["Gun Bunch"], "locked_ts": "test",
        }
        playbook._save_state(self.db, {"applied": {"offense": self.old}, "pending": {}})
        self.book_patch = mock.patch.object(
            designer.catalog, "book_formations",
            side_effect=lambda side, src: BOOK.get(src, {})
        )
        self.book_patch.start()
        self.catalog_patch = mock.patch.object(
            browser.catalog, "books", return_value={
                "Buccaneers": {"url": "https://huddle.gg/27/playbooks/buccaneers-off/"},
                "Lions": {"url": "https://huddle.gg/27/playbooks/lions-off/"}
            }
        )
        self.catalog_patch.start()
        self.src_patch = mock.patch.object(browser.research_db, "sources", return_value={
            "prodigy-beat-man": {
                "id": "prodigy-beat-man", "title": "Madden Prodigy",
                "url": "https://www.maddenprodigy.com/route-guide/",
            },
            "realsport-controls": {
                "id": "realsport-controls", "title": "Controller Guide",
                "url": "https://realsport101.com/article/madden-27-controls-guide",
            },
        })
        self.src_patch.start()
        self.artifact = experimental_model.ExperimentalArtifact(
            model_version="browser-test-model", n_supervised=8,
            evidence_quality="mixed", global_rate=0.5,
        )

    def tearDown(self) -> None:
        self.src_patch.stop()
        self.catalog_patch.stop()
        self.book_patch.stop()
        self.db.close()
        self.tmp.cleanup()

    def design(self) -> dict:
        return designer.design_offense(
            self.db, opponent_id="cpu", max_formations=3,
            artifact=self.artifact, catalogue=BOOK
        )

    def test_browser_shows_formations_plays_macro_settings_and_sources(self) -> None:
        plan = self.design()
        markup = browser.render_html(self.db, plan, mode="preview", opponent_id="cpu")
        self.assertIn("AI Offensive Playbook Designer", markup)
        self.assertIn("PREVIEW — NO CHANGES", markup)
        self.assertIn("browser-test-model", markup)
        self.assertIn("Stock source:", markup)
        self.assertIn('href="https://huddle.gg/27/playbooks/', markup)
        self.assertIn("<th>Setting</th>", markup)
        self.assertIn("Research:", markup)
        self.assertIn("Madden Prodigy", markup)
        self.assertIn("source", markup.lower())
        self.assertIn("Custom Adjustments", markup)
        self.assertIn("Draft — not callable", markup)
        self.assertIn("build-check", markup)
        self.assertIn("complete formation", markup)
        self.assertIn("all", markup)
        self.assertNotIn("ADD_PLAY", markup)
        self.assertNotIn("REMOVE_PLAY", markup)
        self.assertIn("Not callable".lower(), markup.lower())
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)

    def test_staged_browser_only_unlocks_copy_after_checks(self) -> None:
        plan = designer.stage_design(self.db, self.design())
        markup = browser.render_html(self.db, plan, mode="staged", opponent_id="cpu")
        self.assertIn("STAGED — NOT CALLABLE", markup)
        self.assertIn('id="confirm-copy"', markup)
        self.assertIn("disabled", markup)
        self.assertIn(plan["proposal_id"], markup)
        self.assertIn("checkboxes are local to this browser", markup.lower())
        self.assertEqual(playbook.load_books(self.db)["offense"], self.old)

    def test_installed_browser_only_after_attestation(self) -> None:
        staged = designer.stage_design(self.db, self.design())
        self.assertFalse(
            browser.render_html(self.db, staged, mode="staged").count("INSTALLED — CALLABLE")
        )
        designer.confirm_installed(
            self.db, proposal_id=staged["proposal_id"],
            attestation="I installed every formation and play individually inside Madden.",
        )
        saved = browser.installed_design(self.db, "cpu")
        self.assertIsNotNone(saved)
        markup = browser.render_html(self.db, saved, mode="installed", opponent_id="cpu")
        self.assertIn("INSTALLED — CALLABLE", markup)
        self.assertIn("Installed &amp; callable", markup)
        self.assertIn("Macro", markup)
        self.assertIn("Draft — not callable", markup)
        self.assertIn('id="confirm-copy"', browser.render_html(
            self.db, staged, mode="staged", opponent_id="cpu"
        ))
        approved = designer.verified_created_macros(self.db, "cpu")
        self.assertEqual(approved, [])

    def test_removed_macro_becomes_uncallable_for_opponent(self) -> None:
        staged = designer.stage_design(self.db, self.design())
        designer.confirm_installed(
            self.db, proposal_id=staged["proposal_id"],
            attestation="I installed every formation and play in Madden's custom editor.",
        )
        m = staged["macro_blueprints"][0]
        designer.verify_created_macro(
            self.db, name=m["name"], opponent_id="cpu",
            attestation="I built this named macro with the specified settings and armed the slot inside Madden.",
        )
        installed = browser.installed_design(self.db, "cpu")
        markup = browser.render_html(self.db, installed, mode="installed", opponent_id="cpu")
        self.assertIn("Verified and armed", markup)
        self.assertIn("--unverify-macro", markup)
        from cfb_coach.madden.model.offense_action_policy import choose_offense_action
        pair = m["base_pairs"][0]
        context = SimpleNamespace(
            coverage_hint="Cover 1" if m["coverage"] == "man" else "Blitz",
            coverage_source="live", goal_line=False, red_zone=False, down=2,
        )
        state = dict(
            formation=pair["formation"], play=pair["play"], sit=context,
            book=playbook.load_books(self.db)["offense"]["formations"],
            active=[], db=self.db, opponent_id="cpu",
            prediction={"probability": 0.6, "uncertainty": 0.2},
        )
        before = choose_offense_action(**state)
        self.assertIn(m["name"], [item["id"] for item in before["candidates"]])
        result = designer.unverify_created_macro(self.db, name=m["name"], opponent_id="cpu")
        self.assertFalse(result["now_callable"])
        self.assertEqual(designer.verified_created_macros(self.db, "cpu"), [])
        after = choose_offense_action(**state)
        self.assertNotIn(m["name"], [item["id"] for item in after["candidates"]])
        markup = browser.render_html(self.db, installed, mode="installed", opponent_id="cpu")
        self.assertIn("Draft — not callable", markup)

    def test_render_escapes_untrusted_names_and_unsafe_source_urls(self) -> None:
        plan = self.design()
        plan["book"]["formations"]["<script>alert(1)</script>"] = ["<img src=x onerror=alert(1)>"]
        markup = browser.render_html(self.db, plan, mode="preview", opponent_id="cpu")
        self.assertNotIn("<script>alert(1)</script>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertNotIn("<img src=x onerror=alert(1)>", markup)

    def test_html_file_opens_in_browser_unless_disabled(self) -> None:
        plan = self.design()
        out = Path(self.tmp.name) / "designer.html"
        with mock.patch("cfb_coach.prep_browser.open_prep_html") as opened:
            actual = browser.write_and_open(
                self.db, plan, path=out, opponent_id="cpu", open_browser=True
            )
        self.assertEqual(out, actual)
        opened.assert_called_once_with(out, open_browser=True)
        self.assertIn("<!doctype html>", out.read_text())
        with mock.patch("cfb_coach.prep_browser.open_prep_html") as opened:
            browser.write_and_open(
                self.db, plan, path=out, opponent_id="cpu", open_browser=False
            )
        opened.assert_not_called()

    def test_cli_default_browser_json_still_available_via_text(self) -> None:
        with mock.patch.object(ml_cli, "open_madden_db", return_value=self.db):
            with mock.patch.object(designer, "design_offense", return_value=self.design()):
                with mock.patch.object(browser, "write_and_open", return_value=Path("designer.html")) as written:
                    args = argparse.Namespace(
                        show=False, stage=False, confirm_installed=None,
                        rollback_design=None, verify_macro=None, unverify_macro=None,
                        retire_existing=None,
                        opponent="cpu", max_formations=3,
                        attest=None, text=False, no_open=True,
                    )
                    self.assertEqual(ml_cli.cmd_ml_offense_design(args), 0)
                    self.assertFalse(written.call_args.kwargs["open_browser"])
                    written.reset_mock()
                    args.text = True
                    self.assertEqual(ml_cli.cmd_ml_offense_design(args), 0)
                    written.assert_not_called()


if __name__ == "__main__":
    unittest.main()
