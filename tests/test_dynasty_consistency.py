"""v1.15.1: one dynasty end to end. `play` follows the latest prep for that opponent
(explicit --dynasty overrides; the global dynasty_mode meta is only the fallback), and
the live window / terminal show the same custom book, rev, formations and sources as
the prep page."""

from __future__ import annotations

import contextlib
import io
from unittest import mock

from cfb_coach import cfb_playbook as cp
from cfb_coach import cli
from cfb_coach import macros as mc
from cfb_coach import playcaller
from cfb_coach.dynasty import last_prep_dynasty, record_prep_dynasty, resolve_dynasty
from tests.test_cfb_playbook import NAMED_PTRIPS, PTRIPS, _DBCase


class TestResolveDynasty(_DBCase):
    def test_order_explicit_then_latest_prep_then_configured(self):
        self.db.set_meta("dynasty_mode", "alabama")
        self.assertEqual(resolve_dynasty(self.db, "cpu", None),
                         ("alabama", "configured default (no prep for this opponent yet)"))
        record_prep_dynasty(self.db, "cpu", "ohio_state", book="OSU LAB O (custom)", rev=2)
        dyn, src = resolve_dynasty(self.db, "cpu", None)
        self.assertEqual(dyn, "ohio_state")
        self.assertTrue(src.startswith("latest prep vs cpu") and src.endswith("ET"), src)
        self.assertEqual(resolve_dynasty(self.db, "cpu", "alabama"), ("alabama", "--dynasty"))
        self.assertEqual(resolve_dynasty(self.db, "someone_else", None)[0], "alabama")  # per opponent
        self.db.set_meta("dynasty_mode", "")
        self.assertEqual(resolve_dynasty(self.db, "nobody", None)[0], "alabama")  # built-in default


class _PreppedOSU(_DBCase):
    """Prep vs cpu as Ohio State (confirmed rev 1 + pending Pistol Trips), then a
    different prep flips the global dynasty_mode to Alabama."""

    def setUp(self):
        super().setUp()
        from tests.test_cfb_playbook import TestPrepPage

        self.osu_games()
        cp.plan_book(self.db, "ohio_state", {"mode": "live", "named_signals": {}})
        cp.apply_pending(self.db, "ohio_state")  # rev 1 confirmed
        cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)  # rev 2 pending: + Pistol Trips
        self.plan = TestPrepPage._plan(self)
        self.db.set_meta("dynasty_mode", "alabama")  # e.g. a later prep for another opponent

    def _play(self, argv, lines=("2&7 yl40 showing cover 1", "q")):
        buf = io.StringIO()
        seen = []
        real = playcaller.make_call

        def spy(*a, **kw):
            seen.append(kw.get("dynasty"))
            return real(*a, **kw)

        with mock.patch("builtins.input", side_effect=list(lines)), mock.patch.object(cli, "make_call", spy), \
                contextlib.redirect_stdout(buf):
            self.assertEqual(cli.main(argv), 0)
        return buf.getvalue(), seen


class TestPlayFollowsPrep(_PreppedOSU):
    def test_prep_recorded_dynasty_and_book(self):
        rec = last_prep_dynasty(self.db, "cpu")
        self.assertEqual(rec["dynasty"], "ohio_state")
        self.assertEqual(rec["book"], self.plan["cfb_book"]["name"])
        self.assertEqual(rec["book"], cp.live_book_info(self.db, "ohio_state")["name"])

    def test_terminal_play_without_flag_uses_prep_dynasty(self):
        out, seen = self._play(["play", "-o", "cpu", "--terminal", "--no-overlay"])
        name = cp.live_book_info(self.db, "ohio_state")["name"]
        self.assertTrue(name)
        self.assertIn("Dynasty: Ohio State", out)
        self.assertIn("— latest prep vs cpu", out)
        self.assertIn(f"Playbook (ohio_state): {name} — locked to rev 1", out)
        self.assertIn("pending formation change(s) (rev 2)", out)
        self.assertNotIn("UNCONFIRMED", out)
        self.assertNotIn("alabama", out.lower().split("doctrine")[0])  # header never mentions Alabama
        self.assertEqual(set(seen), {"ohio_state"})

    def test_explicit_dynasty_overrides(self):
        out, seen = self._play(["play", "-o", "cpu", "--terminal", "--no-overlay", "--dynasty", "alabama"], lines=("q",))
        self.assertIn("— --dynasty", out)
        self.assertIn("Playbook (alabama)", out)

    def test_html_controller_same_dynasty_and_apply_targets_it(self):
        from cfb_coach.live_server import render_live_html

        cp.plan_book(self.db, "alabama", None)  # an Alabama pending build must stay untouched
        bama_before = cp.pending_rev(self.db, "alabama")
        name = cp.live_book_info(self.db, "ohio_state")["name"]
        box = {}

        def fake_server(ctrl, **kw):  # runs while play's db is still open
            box["dyn"], box["state"], box["html"] = ctrl.dynasty, ctrl.state(), render_live_html(ctrl)
            box["apply"] = ctrl.apply_book(None)
            box["after"] = cp.callable_book(ctrl.db, "ohio_state")["formations"]
            box["bama"] = cp.pending_rev(ctrl.db, "alabama")
            return 0

        with mock.patch("cfb_coach.live_server.run_live_server", fake_server), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["play", "-o", "cpu"]), 0)
        self.assertEqual(box["dyn"], "ohio_state")
        self.assertEqual(box["state"]["dynasty"], "ohio_state")
        self.assertIn("latest prep vs cpu", box["state"]["dynasty_source"])
        self.assertEqual(box["state"]["book"]["name"], name)
        self.assertIn("Dynasty: <b>Ohio State", box["html"])
        self.assertIn(f"Book: <b>{name} rev 1 (applied)</b>", box["html"])
        self.assertTrue(box["apply"]["ok"])
        self.assertEqual((box["apply"]["dynasty"], box["apply"]["applied_rev"]), ("ohio_state", 2))
        self.assertIn(PTRIPS, box["after"])
        self.assertEqual(box["bama"]["rev"], bama_before["rev"])

    def test_live_book_matches_prep_formations_sources_counts(self):
        bk = self.plan["cfb_book"]
        live = cp.live_book_info(self.db, "ohio_state")
        in_book = {r["formation"]: r for r in bk["formation_list"] if r["status"] in ("applied", "changed")}
        self.assertTrue(in_book)
        self.assertEqual(set(in_book), set(live["formations"]))
        for f, r in in_book.items():
            self.assertEqual(r["n_plays"], len(live["formations"][f]), f)
            self.assertEqual(r["source_book"], live["sources"].get(f), f)
        new = [r["formation"] for r in bk["formation_list"] if r["status"] == "new"]
        self.assertEqual(new, [PTRIPS])  # pending add, callable only after book apply

    def test_active8_stored_per_dynasty(self):
        import json

        stored = json.loads(self.db.get_meta("active_macros_o:cpu"))
        self.assertEqual(stored["dynasty"], "ohio_state")
        self.assertEqual(mc.active_offense_macros(self.db, "cpu", "ohio_state"), stored["offense"])
        mc.store_active_offense_macros(self.db, "cpu", ["SHOT"], dynasty="alabama")
        self.assertNotIn("SHOT", mc.active_offense_macros(self.db, "cpu", "ohio_state"))  # other dynasty's list ignored
        self.assertEqual(mc.active_offense_macros(self.db, "cpu", "alabama"), ["SHOT"])
