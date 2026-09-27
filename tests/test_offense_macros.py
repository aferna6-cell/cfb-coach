"""v1.15: offense custom-adjustment drill-down on the prep page (Aidan's settings verbatim,
book pairs, fire trigger, research notes only on the details page) and the live caller's
macro suggestion (Active 8 only, only when it adds value, rendered in the live window
and terminal, stored with the snap, capped macro learning)."""

from __future__ import annotations

import contextlib
import io
import itertools
import random
import re
from unittest import mock

from cfb_coach import macros as mc
from cfb_coach.playcaller import Call, _attach_offense_macro
from cfb_coach.situation import parse_situation
from tests.test_cfb_playbook import BUNCH, CLUSTER, _DBCase

ACTIVE8 = ["RZ", "O-RUN", "MATCH", "C2", "MAN", "O-RPO", "C3", "ZERO"]


def sug(play, cov=None, src="live", zone="open", active=ACTIVE8, **kw):
    return mc.suggest_offense_macro(zone=zone, play=play, coverage=cov, coverage_source=src, active=active, **kw)


def _catalog_routes(mid: str) -> list[tuple[str, str]]:
    """His route_assignments string, split exactly as he wrote it."""
    import json
    from pathlib import Path

    cat = json.loads((Path(mc.__file__).parent / "data" / "macro_catalog.json").read_text())
    val = cat["macros"][mid]["full_settings"]["route_assignments"]
    assert val["status"] == "confirmed"
    return [tuple(p.strip().split(" ", 1)) for p in val["value"].split(";")]


class TestSettingsData(_DBCase):
    def test_settings_are_aidans_verbatim_no_invented_rows(self):
        allowed = {"Route assignments", "Protection", "Blocking"}
        for mid in ACTIVE8 + ["O-HEAT", "SHOT"]:
            rows = mc.aidan_offense_settings(mid)
            self.assertTrue(rows, mid)
            self.assertTrue({r["section"] for r in rows} <= allowed, mid)
            fs = mc.get_macro(mid)["full_settings"]
            for r in rows:  # every value appears verbatim in one of his confirmed fields
                self.assertEqual(fs[r["field"]]["status"], "confirmed")
                self.assertIn(r["value"], fs[r["field"]]["value"], (mid, r))
        self.assertEqual(mc.aidan_offense_settings("ZERO")[-1], {"section": "Protection", "setting": "Protection",
                                                                  "value": "Max protect", "field": "protection"})
        self.assertEqual(mc.macro_key_settings("O-RUN"),
                         "Blocking: Base OL technique; highest-OVR double team; Mike ID; aggressive blocking; conservative ball carrier")
        self.assertEqual(mc.aidan_settings_gaps("PROT"), ["protection: Max protect / slide to pressure side [approx UI]"])
        self.assertEqual([m for m in ACTIVE8 if not mc.aidan_offense_settings(m)], [])  # no Active-8 gaps

    def test_man_c2_match_render_his_routes_verbatim_everywhere(self):
        from cfb_coach.prep_browser import _render_macro_accordion

        book = {BUNCH: ["Mesh Spot", "Inside Zone", "Return Whip Trail"], CLUSTER: ["Z Spot Shake"]}
        for mid in ("MAN", "C2", "MATCH"):
            want = _catalog_routes(mid)
            rows = [(r["setting"], r["value"]) for r in mc.aidan_offense_settings(mid)]
            self.assertEqual(rows, want, mid)
            d = mc.offense_macro_detail(mid, book)
            cb = mc.offense_copy_block(d)
            for slot, route in want:
                self.assertIn(f"  [ ] {slot}: {route}\n", cb)
            self.assertIn("[ ] Everything else: Default", cb)
            self.assertEqual(d["key"], " · ".join(f"{a} {b}" for a, b in want))
            card = mc.attach_offense_detail({"id": mid, "name": mid, "side": "offense", "slot": "active"}, book)
            html = _render_macro_accordion([card], {}, offense_only=True)
            for slot, route in want:
                self.assertIn(f"<td class='ca-k'>{slot}</td><td class='ca-v'>{route}</td>", html)
            s = sug({"MAN": "Mesh Spot", "C2": "Mesh Spot", "MATCH": "Mesh Spot"}[mid],
                    {"MAN": "Cover 1", "C2": "Cover 2", "MATCH": "Cover 4 Quarters"}[mid])
            self.assertEqual(s["id"], mid)
            self.assertEqual([(r["setting"], r["value"]) for r in s["settings"]], want)
        self.assertEqual(_catalog_routes("MAN"), [("WR1", "deep cross"), ("WR2", "zig"), ("WR3", "short cross"), ("TE", "wheel"), ("HB", "Texas")])
        self.assertEqual(_catalog_routes("C2"), [("WR1", "streak"), ("WR2", "corner"), ("WR3", "post"), ("TE", "slot fade"), ("HB", "flat")])

    def test_pairs_with_book_plays(self):
        book = {BUNCH: ["Mesh Spot", "Inside Zone", "Return Whip Trail"], CLUSTER: ["Z Spot Shake"]}
        d = mc.offense_macro_detail("MAN", book)
        self.assertEqual(d["pairs_with"][:2], [f"Mesh Spot ({BUNCH})", f"Return Whip Trail ({BUNCH})"])
        self.assertNotIn("Inside Zone", " ".join(d["pairs_with"]))  # a run never pairs with a route macro


class TestPrepDrillDown(_DBCase):
    def _plan(self):
        from tests.test_cfb_playbook import TestPrepPage

        self.osu_games()
        return TestPrepPage._plan(self)

    def test_prep_page_rows_expand_to_exact_settings_collapsed_clean(self):
        from cfb_coach.prep_browser import render_prep_details_html, render_prep_html

        plan = self._plan()
        cards = plan["macro_cards"]
        self.assertTrue(cards and all(c.get("ingame") for c in cards))
        html = render_prep_html(plan)
        macros = html[html.index('<section id="macros">'):]
        n = len(cards)
        self.assertEqual(macros.count('<details class="macro-card'), n)
        self.assertNotIn('data-macro="MAN" open', macros)
        self.assertEqual(len(re.findall(r'<details class="macro-card[^>]*\bopen\b', macros)), 0)  # collapsed by default
        for want in ("Settings — Create &amp; Share › Custom Adjustments › Offense", "Fire it when:",
                     "Pairs with (your book):", "<td class='ca-k'>WR1</td><td class='ca-v'>deep cross</td>",
                     "<td class='ca-k'>Everything else</td><td class='ca-v'>Default</td>", 'class="copy-btn"'):
            self.assertIn(want, macros, want)
        for bad in ("assumed", "hot-route menu", "D-pad", "Deep Over"):  # no remaps / research on the main page
            self.assertNotIn(bad.lower(), macros.lower(), bad)
        det = render_prep_details_html(plan)
        self.assertIn("nothing was remapped", det)
        self.assertIn("not in the CFB 26 outside hot-route menu", det)  # research note only, value unchanged
        self.assertIn("collegefootball.gg/all-of-the-new-hot-routes-in-cfb-26", det)

    def test_prep_stores_active_offense_macros_for_live(self):
        plan = self._plan()
        ids = [c["id"] for c in plan["macro_cards"]]
        self.assertEqual(mc.active_offense_macros(self.db, "cpu"), ids)


class TestLiveSuggestion(_DBCase):
    def test_right_situations(self):
        self.assertEqual(sug("Mesh Spot", "Cover 1")["id"], "MAN")
        self.assertEqual(sug("Mesh Spot", "Cover 2 Invert")["id"], "C2")
        self.assertEqual(sug("Deep Flood", "Cover 3 Sky")["id"], "C3")
        self.assertEqual(sug("Mesh Spot", "Cover 4 Quarters")["id"], "MATCH")
        self.assertEqual(sug("Inside Zone", "Cover 6")["id"], "O-RUN")
        self.assertEqual(sug("Mtn RPO Zone Alert", "Cover 4 Quarters")["id"], "O-RPO")
        self.assertEqual(sug("Mesh Spot", "Cover 0")["id"], "ZERO")
        self.assertEqual(sug("Mesh Spot", "pressure")["id"], "ZERO")  # no HEAT/PROT in the Active 8
        self.assertEqual(sug("Mesh Spot", None, src="none", zone="rz")["id"], "RZ")
        self.assertEqual(sug("Mesh Spot", "Cover 1", zone="rz")["id"], "MAN")  # coverage answer beats generic RZ

    def test_not_every_snap(self):
        self.assertIsNone(sug("Mesh Spot", None, src="none"))  # open field, no look
        self.assertIsNone(sug("Mesh Spot", "Cover 1", src="last"))  # last snap alone never fires
        self.assertEqual(sug("Mesh Spot", "Cover 1", src="last", repeated=True)["id"], "MAN")
        self.assertIsNone(sug("Inside Zone", "Cover 1"))  # route macro on a run adds nothing
        self.assertIsNone(sug("Inside Zone", None, src="none", zone="rz"))
        self.assertIsNone(sug("Return Whip Trail", "Cover 2 Invert"))  # never whip into the hard flat
        self.assertIsNone(sug("RZ PA X Whip", "Cover 2 Invert", zone="rz"))
        self.assertIsNone(sug("Deep Flood", "Cover 6"))
        self.assertIsNone(sug("Mesh Spot", "Cover 1", weights={"MAN": -0.3}))  # learned: it keeps failing

    def test_never_outside_active8(self):
        self.assertIsNone(sug("Mesh Spot", "Cover 1", active=["RZ", "O-RUN"]))
        self.assertIsNone(sug("Mesh Spot", "Cover 1", active=[]))
        plays = ["Mesh Spot", "Inside Zone", "Deep Flood", "Return Whip Trail", "Mtn RPO Zone Alert", "RZ PA X Whip", "PA Read", "Quick Slants"]
        covs = [None, "Cover 0", "Cover 1", "Cover 2", "Cover 2 Invert", "Cover 3 Sky", "Cover 4 Quarters", "Cover 6", "pressure", "two-high"]
        rng = random.Random(7)
        pool = list(mc.load_offense_settings()["macros"])
        for play, cov, zone, src in itertools.product(plays, covs, ("open", "rz", "gl"), ("live", "last")):
            active = rng.sample(pool, 4)
            s = sug(play, cov, src=src, zone=zone, active=active, down=rng.choice([1, 2, 3]))
            if s:
                self.assertIn(s["id"], active)

    def test_attach_uses_stored_active8_and_live_look(self):
        mc.store_active_offense_macros(self.db, "cpu", ["RZ", "MAN"])
        call = Call("offense", BUNCH, "Mesh Spot", "No adj", "Spot → Drag", "base")
        _attach_offense_macro(call, parse_situation("3&6 my 40 showing cover 1"), {"_id": "cpu"}, self.db)
        self.assertEqual(call.macro, "MAN")
        self.assertEqual(call.headline(), f"PLAY: Mesh Spot ({BUNCH}) + MACRO: MAN")
        self.assertIn("MACRO: MAN — LB → MAN | WR1 deep cross · WR2 zig · WR3 short cross · TE wheel · HB Texas", call.format())
        c2 = Call("offense", BUNCH, "Mesh Spot", "No adj", "Spot → Drag", "base")
        _attach_offense_macro(c2, parse_situation("3&6 my 40 showing cover 2"), {"_id": "cpu"}, self.db)
        self.assertIsNone(c2.macro)  # C2 not in this Active list
        self.assertNotIn("MACRO:", c2.format())

    def test_make_call_macros_only_with_a_look_and_only_active(self):
        from cfb_coach.playcaller import make_call

        mc.store_active_offense_macros(self.db, "cpu", ACTIVE8)
        seen = set()
        for i in range(60):
            c = make_call(parse_situation("2&7 my 40 showing cover 1"), "cpu", self.db, rng=random.Random(i))
            if c.macro:
                self.assertIn(c.macro, ACTIVE8)
                seen.add(c.macro)
            n = make_call(parse_situation("1&10 my 30"), "cpu", self.db, rng=random.Random(i))
            self.assertIsNone(n.macro, n.format())  # no look, open field: no macro
        self.assertIn("MAN", seen)


class _FakeCtrlMixin:
    def ctrl(self, call):
        from cfb_coach.live_server import LivePlayController

        c = LivePlayController(db=self.db, opponent_id="cpu", make_call=lambda sit, **k: call,
                               parse_situation=parse_situation, learn_summary=lambda: "", cpu_only=True)
        c.start()
        return c


class TestLiveWindowAndSnap(_FakeCtrlMixin, _DBCase):
    def _call(self):
        mc.store_active_offense_macros(self.db, "cpu", ACTIVE8)
        call = Call("offense", BUNCH, "Mesh Spot", "No adj", "Spot → Drag", "base")
        _attach_offense_macro(call, parse_situation("3&6 my 40 showing cover 1"), {"_id": "cpu"}, self.db)
        return call

    def test_live_html_shows_play_plus_macro_and_settings(self):
        from cfb_coach.live_server import render_live_html

        ctrl = self.ctrl(self._call())
        res = ctrl.call_only("3&6 my 40 showing cover 1")
        st = res["state"]["macro"]
        self.assertEqual(st["headline"], f"PLAY: Mesh Spot ({BUNCH}) + MACRO: MAN")
        self.assertEqual(st["key"], "WR1 deep cross · WR2 zig · WR3 short cross · TE wheel · HB Texas")
        html = render_live_html(ctrl)
        self.assertIn(f"PLAY: Mesh Spot ({BUNCH}) + MACRO: MAN", html)
        self.assertIn('<div id="macro-box">', html)
        self.assertIn("<td class='s'>WR2</td><td>zig</td>", html)  # his exact settings, expandable right there
        self.assertIn("renderMacro(st.macro)", html)

    def test_no_macro_box_when_no_macro(self):
        from cfb_coach.live_server import render_live_html

        ctrl = self.ctrl(Call("offense", BUNCH, "Inside Zone", "No adj", "Front → Cutback", "base"))
        ctrl.call_only("1&10 my 30")
        self.assertIsNone(ctrl.state()["macro"])
        self.assertIn('<div id="macro-box" hidden>', render_live_html(ctrl))

    def test_macro_stored_with_snap_and_learned_capped(self):
        from cfb_coach.learning import CAP_FAMILY, compute_all

        ctrl = self.ctrl(self._call())
        ctrl.call_only("3&6 my 40 showing cover 1")
        for out in ("gain 12", "gain 9", "incomplete"):
            ctrl.result_and_call(outcome=out, sit_raw="3&6 my 40 showing cover 1")
        rows = self.db.conn.execute("SELECT side, play, macro FROM snaps ORDER BY id").fetchall()
        self.assertEqual([tuple(r) for r in rows], [("offense", "Mesh Spot", "MAN")] * 3)
        w = compute_all(self.db.conn)["macros"]["cpu"]
        self.assertIn("MAN", w)
        self.assertLessEqual(abs(w["MAN"]), CAP_FAMILY)


class TestTerminal(_DBCase):
    def test_terminal_prints_macro_line_and_logs_it(self):
        from cfb_coach import cli

        mc.store_active_offense_macros(self.db, "cpu", ACTIVE8)

        def fake_make_call(sit, oid, db, **kw):
            call = Call("offense", BUNCH, "Mesh Spot", "No adj", "Spot → Drag", "base")
            _attach_offense_macro(call, sit, {"_id": oid}, db)
            return call

        feed = iter(["2&7 my 40 showing cover 1", "result gain 9", "q"])
        buf = io.StringIO()
        with mock.patch.object(cli, "make_call", fake_make_call), mock.patch("builtins.input", lambda *_: next(feed)), \
                contextlib.redirect_stdout(buf):
            self.assertEqual(cli.main(["play", "-o", "cpu", "--terminal", "--no-overlay"]), 0)
        out = buf.getvalue()
        self.assertIn("MACRO: MAN — LB → MAN | WR1 deep cross · WR2 zig · WR3 short cross · TE wheel · HB Texas", out)
        self.db.conn.commit()
        from cfb_coach.db import CoachDB

        db2 = CoachDB(self.path)
        try:
            self.assertEqual(db2.conn.execute("SELECT macro FROM snaps WHERE side='offense'").fetchone()[0], "MAN")
        finally:
            db2.close()


class TestLiveWindowScript(_FakeCtrlMixin, _DBCase):
    def test_live_window_script_parses(self):
        """Guard: the inline JS (renderMacro etc.) must be valid — checked with node when available."""
        import shutil
        import subprocess
        import tempfile
        import unittest

        from cfb_coach.live_server import render_live_html

        node = shutil.which("node")
        if not node:
            raise unittest.SkipTest("node not installed")
        mc.store_active_offense_macros(self.db, "cpu", ACTIVE8)
        call = Call("offense", BUNCH, "Mesh Spot", "No adj", "Spot → Drag", "base")
        _attach_offense_macro(call, parse_situation("3&6 my 40 showing cover 1"), {"_id": "cpu"}, self.db)
        ctrl = self.ctrl(call)
        ctrl.call_only("3&6 my 40 showing cover 1")
        html = render_live_html(ctrl)
        self.assertNotIn("MACRO: MAN — LB", html.split('<div class="call" id="call">', 1)[1].split("</div>", 1)[0])
        js = html.split("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        r = subprocess.run([node, "--check", fh.name], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
