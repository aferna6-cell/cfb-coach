"""v1.13: autonomous CFB custom playbook — seeding, decisions + hysteresis, versioning,
pending-edits flow, live caller locked to the book, live-window hooks, CLI."""

from __future__ import annotations

import contextlib
import io
import os
import random
import tempfile
import unittest
from pathlib import Path

from cfb_coach import cfb_playbook as cp
from cfb_coach.cfb_catalog import canonical_pair, formations, zone_fit
from cfb_coach.db import CoachDB
from cfb_coach.session import start_session

BUNCH, CLUSTER, DEUCE = "Gun Bunch X Nasty", "Gun Cluster", "Singleback Deuce Close"
LAB = [(BUNCH, "Post Wheel Shallow"), (BUNCH, "Y Flat GoalLine"), (BUNCH, "Z Mesh GoalLine")]


def _log(db: CoachDB, sid: str, form: str, play: str, results: list[str], yl: int = 40, down: int = 1, dist: int = 10) -> None:
    for r in results:
        db.log_snap(opponent_id="cpu", side="offense", situation_raw=f"{down}&{dist}", our_call=f"{form} {play}",
                    formation=form, play=play, down=down, distance=dist, yardline=yl, result=r, session_id=sid)


class _DBCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self._old = os.environ.get("CFB_COACH_DB")
        self.path = Path(self.td.name) / "coach.db"
        os.environ["CFB_COACH_DB"] = str(self.path)
        self.db = CoachDB(self.path)
        self.db.set_meta("dynasty_mode", "ohio_state")

    def tearDown(self):
        self.db.close()
        if self._old is None:
            os.environ.pop("CFB_COACH_DB", None)
        else:
            os.environ["CFB_COACH_DB"] = self._old
        self.td.cleanup()

    def osu_games(self) -> None:
        sid = start_session("cpu", dynasty="ohio_state", db=self.db).session_id
        _log(self.db, sid, BUNCH, "Inside Zone", ["gain 7", "gain 6", "gain 5", "gain 9", "gain 4", "gain 12", "gain 3", "gain 8"])
        _log(self.db, sid, BUNCH, "Mesh Spot", ["gain 8", "gain 11", "gain 6", "incomplete", "gain 9", "gain 7", "gain 14"])
        # off-meta-ish but working for him (Cluster HB Mid Draw has no seed prior)
        _log(self.db, sid, CLUSTER, "HB Mid Draw", ["gain 9", "gain 7", "gain 12", "gain 6", "gain 8", "gain 10", "gain 7"])
        # failing, meta cold (Drive HB Under has only a tiny seed prior)
        _log(self.db, sid, BUNCH, "Drive HB Under", ["incomplete", "loss 3", "incomplete", "gain 1", "incomplete", "sack", "incomplete"])
        # failing but the meta likes it (seed prior +0.30 in RZ/GL)
        _log(self.db, sid, BUNCH, "RZ PA X Whip", ["incomplete", "int", "incomplete", "sack", "incomplete", "incomplete", "int"], yl=88, dist=8)
        # logged with a sloppy name -> canonicalized
        _log(self.db, sid, "singleback deuce close", "hb dive", ["td", "gain 3"], yl=97, dist=3)


class TestCatalog(unittest.TestCase):
    def test_canonical_names_and_zone_fit(self):
        self.assertEqual(canonical_pair("gun bunch x nasty", "mesh spot")[:2], (BUNCH, "Mesh Spot"))
        self.assertTrue(canonical_pair("singleback deuce close", "hb dive")[2])
        for f, p in LAB:
            self.assertIn(p, formations()[f])
        self.assertFalse(zone_fit("Y Flat GoalLine", "open"))
        self.assertFalse(zone_fit("Deep Flood", "gl"))
        self.assertTrue(zone_fit("Inside Zone", "gl"))


class TestSeedingAndDecisions(_DBCase):
    def test_seed_from_logged_snaps_then_meta_adds_pending(self):
        self.osu_games()
        r = cp.plan_book(self.db, "ohio_state", None)
        self.assertTrue(r["seeded_now"])
        cur = cp.current_rev(self.db, "ohio_state")
        self.assertEqual(cur["rev"], 1)
        self.assertIn("HB Dive", cur["book"]["formations"][DEUCE])  # canonicalized from "hb dive"
        self.assertEqual(r["status"], "pending")
        added = {(e["formation"], e["play"]) for e in r["edits"] if e["op"] == "add_play"}
        self.assertTrue(set(LAB) & added, added)  # lab candidates added for the OSU lab
        self.assertTrue(all(e.get("reason") for e in r["edits"]))
        self.assertIn("+ ADD", r["edit_text"])
        # working play kept regardless of meta; meta play failing -> demoted with reasoning
        self.assertEqual(r["flags"][f"{CLUSTER}::HB Mid Draw"]["flag"], "working")
        self.assertEqual(r["flags"][f"{BUNCH}::RZ PA X Whip"]["flag"], "demoted")
        self.assertIn("keeps failing for you", r["flags"][f"{BUNCH}::RZ PA X Whip"]["why"])
        self.assertNotIn(f"{BUNCH}::RZ PA X Whip", r["active8"])
        self.assertLessEqual(len(r["active8"]), 8)
        # full book within the practical caps
        self.assertLessEqual(len(r["target"]), cp.PRACTICAL["max_formations"])
        self.assertTrue(all(len(v) <= cp.PRACTICAL["max_plays_per_formation"] for v in r["target"].values()))

    def test_no_churn_same_inputs_same_pending_and_no_diff_after_apply(self):
        self.osu_games()
        r1 = cp.plan_book(self.db, "ohio_state", None)
        r2 = cp.plan_book(self.db, "ohio_state", None)
        self.assertEqual(r1["pending"]["rev"], r2["pending"]["rev"])  # same proposal kept, no new revision
        self.assertEqual(len(cp.history(self.db, "ohio_state")), 2)
        cp.apply_pending(self.db, "ohio_state")
        r3 = cp.plan_book(self.db, "ohio_state", None)
        self.assertEqual(r3["status"], "no_change")
        self.assertEqual(r3["edits"], [])
        self.assertEqual(r3["edit_text"], "")

    def test_pending_order_differences_are_not_churn(self):
        import json as _json

        self.osu_games()
        r1 = cp.plan_book(self.db, "ohio_state", None)
        rev = r1["pending"]["rev"]
        pend = cp.get_rev(self.db, "ohio_state", rev)
        book = pend["book"]
        book["formations"] = {f: list(reversed(ps)) for f, ps in reversed(list(book["formations"].items()))}
        self.db.conn.execute(f"UPDATE {cp.TABLE} SET book_json = ? WHERE dynasty = ? AND rev = ?",
                             (_json.dumps(book), "ohio_state", rev))
        self.db.conn.commit()
        r2 = cp.plan_book(self.db, "ohio_state", None)
        self.assertEqual(r2["pending"]["rev"], rev)
        self.assertTrue(cp.same_book({"A": ["x", "y"]}, {"A": ["y", "x"]}))
        self.assertFalse(cp.same_book({"A": ["x"]}, {"A": ["y"]}))

    def test_cut_needs_two_strikes_with_new_data_between(self):
        stats = {f"{BUNCH}::Drive HB Under": {"n": 7, "own": -0.3, "conf": 0.5, "succ_rate": 0.1, "turnovers": 0, "verified": True}}
        meta = {f"{BUNCH}::Drive HB Under": {"meta": 0.02, "specific": 0.02, "reasons": [], "cites": []}}
        book = {BUNCH: ["Inside Zone", "Drive HB Under"]}
        st = {"strikes": {}, "fade": {}, "cooldown": {}, "flags": {}, "prep_count": 0}
        d1 = cp.decide(book, dynasty="ohio_state", stats=stats, meta=meta, state=st, snap_count=50)
        self.assertIn("Drive HB Under", d1["target"][BUNCH])
        self.assertEqual(d1["flags"][f"{BUNCH}::Drive HB Under"]["flag"], "on_notice")
        d2 = cp.decide(book, dynasty="ohio_state", stats=stats, meta=meta, state=d1["state"], snap_count=50)
        self.assertIn("Drive HB Under", d2["target"][BUNCH])  # no new snaps -> no second strike
        d3 = cp.decide(book, dynasty="ohio_state", stats=stats, meta=meta, state=d2["state"], snap_count=60)
        self.assertNotIn("Drive HB Under", d3["target"][BUNCH])
        self.assertIn("CUT", d3["reasons"][f"{BUNCH}::Drive HB Under"])

    def test_overwhelming_failure_cut_now_and_working_off_meta_kept(self):
        stats = {
            f"{BUNCH}::Drive HB Under": {"n": 12, "own": -0.6, "conf": 0.6, "succ_rate": 0.1, "turnovers": 2, "verified": True},
            f"{CLUSTER}::HB Mid Draw": {"n": 9, "own": 0.3, "conf": 0.6, "succ_rate": 0.7, "turnovers": 0, "verified": True},
        }
        meta = {f"{CLUSTER}::HB Mid Draw": {"meta": -0.2, "specific": -0.2, "reasons": [], "cites": []}}
        book = {BUNCH: ["Inside Zone", "Drive HB Under"], CLUSTER: ["HB Mid Draw"]}
        st = {"strikes": {}, "fade": {}, "cooldown": {}, "flags": {}, "prep_count": 0}
        d = cp.decide(book, dynasty="ohio_state", stats=stats, meta=meta, state=st, snap_count=10)
        self.assertNotIn("Drive HB Under", d["target"][BUNCH])
        self.assertIn("HB Mid Draw", d["target"][CLUSTER])
        self.assertEqual(d["flags"][f"{CLUSTER}::HB Mid Draw"]["flag"], "working")

    def test_add_hysteresis_and_swap_margin(self):
        k = f"{BUNCH}::Post Wheel Shallow"
        bar = cp.ADD_MIN["ohio_state"]
        meta = {k: {"meta": bar - 0.05, "specific": bar - 0.05, "reasons": ["x"], "cites": []}}
        st = {"strikes": {}, "fade": {}, "cooldown": {}, "flags": {}, "prep_count": 0}
        book = {BUNCH: ["Inside Zone"]}
        d = cp.decide(book, dynasty="ohio_state", stats={}, meta=meta, state=st, snap_count=0)
        self.assertNotIn("Post Wheel Shallow", d["target"][BUNCH])  # below the bar
        d = cp.decide(book, dynasty="ohio_state", stats={}, meta=meta, state=st, snap_count=0,
                      pending={BUNCH: ["Inside Zone", "Post Wheel Shallow"]})
        self.assertIn("Post Wheel Shallow", d["target"][BUNCH])  # already proposed -> lower bar to stay
        # full formation: swap only with a clear margin over the weakest incumbent
        full = {BUNCH: [p for p in formations()[BUNCH] if p != "Post Wheel Shallow"][: cp.PRACTICAL["max_plays_per_formation"]]}
        meta2 = {k: {"meta": 0.30, "specific": 0.30, "reasons": ["x"], "cites": []}}
        for p in full[BUNCH]:
            meta2[f"{BUNCH}::{p}"] = {"meta": 0.20, "specific": 0.20, "reasons": [], "cites": []}
        d = cp.decide(full, dynasty="ohio_state", stats={}, meta=meta2, state=st, snap_count=0)
        self.assertNotIn("Post Wheel Shallow", d["target"][BUNCH])  # 0.30 < 0.20 + margin
        meta2[k]["meta"] = meta2[k]["specific"] = 0.6
        d = cp.decide(full, dynasty="ohio_state", stats={}, meta=meta2, state=st, snap_count=0)
        self.assertIn("Post Wheel Shallow", d["target"][BUNCH])
        self.assertEqual(len(d["target"][BUNCH]), cp.PRACTICAL["max_plays_per_formation"])

    def test_named_research_can_add_a_new_formation_with_two_plays(self):
        scout = {"named_signals": {"pairs": {
            "Gun Trips TE::Quick Slant": {"score": 6.0, "docs": 4, "sources": [{"label": "YT", "url": "u", "date": "2026-09-20"}]},
            "Gun Trips TE::Deep In": {"score": 6.0, "docs": 4, "sources": []},
        }, "plays": {}, "formations": {"Gun Trips TE": {"score": 5.0}}}}
        meta = cp.meta_scores(scout, dynasty="ohio_state")
        self.assertGreaterEqual(meta["Gun Trips TE::Quick Slant"]["meta"], 0.3)
        st = {"strikes": {}, "fade": {}, "cooldown": {}, "flags": {}, "prep_count": 0}
        d = cp.decide({BUNCH: ["Inside Zone"]}, dynasty="ohio_state", stats={}, meta=meta, state=st, snap_count=0)
        self.assertIn("Gun Trips TE", d["target"])
        self.assertGreaterEqual(len(d["target"]["Gun Trips TE"]), 2)
        edits = cp.diff_books({BUNCH: ["Inside Zone"]}, d["target"])
        self.assertTrue(any(e["op"] == "add_formation" and e["formation"] == "Gun Trips TE" for e in edits))

    def test_alabama_starter_is_pending_and_skips_lab_failures(self):
        self.osu_games()
        r = cp.plan_book(self.db, "alabama", None)
        self.assertTrue(r["first_build"])
        self.assertIsNone(cp.current_rev(self.db, "alabama"))
        pend = cp.pending_rev(self.db, "alabama")
        self.assertIsNotNone(pend)
        plays = {(f, p) for f, ps in pend["book"]["formations"].items() for p in ps}
        self.assertIn((BUNCH, "Inside Zone"), plays)
        self.assertNotIn((BUNCH, "RZ PA X Whip"), plays)  # failed in the OSU lab
        self.assertNotIn((BUNCH, "Drive HB Under"), plays)
        self.assertFalse(set(LAB) & plays)  # lab-only candidates stay in the lab
        self.assertIn("CREATE custom offense playbook", r["edit_text"])


class TestVersioning(_DBCase):
    def test_apply_and_rollback_with_cooldown(self):
        self.osu_games()
        r = cp.plan_book(self.db, "ohio_state", None)
        rev2 = r["pending"]["rev"]
        self.assertIsNone(cp.apply_pending(self.db, "ohio_state", rev=rev2 + 5))  # stale rev refused
        rec = cp.apply_pending(self.db, "ohio_state", rev=rev2)
        self.assertEqual(rec["status"], "current")
        self.assertEqual(cp.get_rev(self.db, "ohio_state", 1)["status"], "superseded")
        rb = cp.rollback(self.db, "ohio_state", 1)
        self.assertEqual(rb["kind"], "rollback")
        self.assertEqual(cp.current_rev(self.db, "ohio_state")["book"]["formations"],
                         cp.get_rev(self.db, "ohio_state", 1)["book"]["formations"])
        self.assertTrue(any(e["op"] == "remove_play" for e in rb["edits"]))
        # cooldown: the next prep must not immediately re-add what was rolled back
        r2 = cp.plan_book(self.db, "ohio_state", None)
        self.assertEqual(r2["status"], "no_change")
        self.assertIn("rollback", cp.format_history(self.db, "ohio_state"))


class TestLiveCallerLockedToBook(_DBCase):
    SITS = ["1&10 my 25", "2&7 my 40", "3&8 opp 45", "3&2 opp 30", "1&10 opp 15", "2&goal opp 4", "1&goal opp 2",
            "3&12 my 20", "2&3 opp 8", "4&1 opp 1"]

    def _calls(self, n_seeds: int = 12):
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        out = []
        for seed in range(n_seeds):
            for raw in self.SITS:
                sit = parse_situation(raw, default_side="offense")
                out.append(make_call(sit, "cpu", self.db, rng=random.Random(seed)))
        return out

    def test_never_recommends_a_play_outside_the_applied_book(self):
        self.osu_games()
        cp.plan_book(self.db, "ohio_state", None)  # rev1 current (his plays) + rev2 pending (lab adds)
        cur = cp.current_rev(self.db, "ohio_state")["book"]["formations"]
        allowed = {(f, p) for f, ps in cur.items() for p in ps}
        calls = self._calls()
        for c in calls:
            self.assertIn((c.formation, c.play), allowed, c.rationale)
        self.assertTrue(any("pending edit(s) not callable" in c.rationale for c in calls))
        # after Aidan confirms, the new (lab) plays become callable
        cp.apply_pending(self.db, "ohio_state")
        cur2 = cp.current_rev(self.db, "ohio_state")["book"]["formations"]
        allowed2 = {(f, p) for f, ps in cur2.items() for p in ps}
        calls2 = self._calls(20)
        for c in calls2:
            self.assertIn((c.formation, c.play), allowed2)
        self.assertTrue({(c.formation, c.play) for c in calls2} & set(LAB))

    def test_unconfirmed_first_build_is_marked(self):
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        self.db.set_meta("dynasty_mode", "alabama")
        self.osu_games()
        cp.plan_book(self.db, "alabama", None)
        c = make_call(parse_situation("1&10 my 30", default_side="offense"), "cpu", self.db, rng=random.Random(1))
        pend = cp.pending_rev(self.db, "alabama")["book"]["formations"]
        self.assertIn(c.play, pend.get(c.formation, []))
        self.assertIn("UNCONFIRMED", c.rationale)

    def test_no_book_keeps_legacy_behavior(self):
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        c = make_call(parse_situation("1&10 my 30", default_side="offense"), "cpu", self.db, rng=random.Random(1))
        self.assertNotIn("book rev", c.rationale)


class TestLiveWindowAndCli(_DBCase):
    def test_live_controller_book_hooks(self):
        from cfb_coach.live_server import LivePlayController, render_live_html
        from cfb_coach.situation import parse_situation

        self.osu_games()
        cp.plan_book(self.db, "ohio_state", None)
        ctrl = LivePlayController(
            db=self.db, opponent_id="cpu", make_call=lambda sit, **k: None, parse_situation=parse_situation,
            learn_summary=lambda: "", dynasty="ohio_state", cpu_only=True,
            book_info=lambda: cp.live_book_info(self.db, "ohio_state"),
            book_apply=lambda rev: cp.apply_pending(self.db, "ohio_state", rev),
        )
        st = ctrl.state()["book"]
        self.assertEqual(st["callable_rev"], 1)
        self.assertTrue(st["pending_rev"])
        self.assertIn("+ ADD", st["pending_text"])
        self.assertIn("book-panel", render_live_html(ctrl))
        res = ctrl.apply_book(st["pending_rev"])
        self.assertTrue(res["ok"])
        self.assertEqual(res["state"]["book"]["callable_rev"], st["pending_rev"])
        self.assertFalse(ctrl.apply_book(None)["ok"])  # nothing pending any more
        # Madden-style controller without hooks: no book panel data
        plain = LivePlayController(db=self.db, opponent_id="cpu", make_call=lambda sit, **k: None,
                                   parse_situation=parse_situation, learn_summary=lambda: "")
        self.assertIsNone(plain.state()["book"])

    def test_cli_book_commands(self):
        from cfb_coach.cli import main

        self.osu_games()
        cp.plan_book(self.db, "ohio_state", None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["book", "show", "--dynasty", "ohio_state"]), 0)
            self.assertEqual(main(["book", "diff", "--dynasty", "ohio_state"]), 0)
            self.assertEqual(main(["book", "apply", "--dynasty", "ohio_state", "--rev", "99"]), 1)
            self.assertEqual(main(["book", "apply", "--dynasty", "ohio_state"]), 0)
            self.assertEqual(main(["book", "history", "--dynasty", "ohio_state"]), 0)
            self.assertEqual(main(["book", "rollback", "--dynasty", "ohio_state", "--to", "1"]), 0)
        out = buf.getvalue()
        self.assertIn("PENDING", out)
        self.assertIn("Applied rev 2", out)
        self.assertIn("Rolled back", out)


class TestPrepPage(_DBCase):
    def test_prep_page_shows_book_edit_list_and_loud_fallback(self):
        from unittest import mock

        from cfb_coach import meta_scout as ms
        from cfb_coach.install_sheet import build_prep_plan, format_delta_text
        from cfb_coach.prep_browser import render_prep_html

        self.osu_games()

        def down(url, timeout):
            src = ms.MetaSource(url=url, label="x", kind="")
            src.error = "URLError: offline"
            return src, ""

        with mock.patch.object(ms, "_fetch_one", down):
            plan = build_prep_plan("cpu", {}, db=self.db, persist=True, dynasty="ohio_state")
        html = render_prep_html(plan)
        self.assertIn("Custom playbook", html)
        self.assertIn("EDIT LIST", html)
        self.assertIn('data-target="book-edits"', html)
        self.assertIn('data-target="book-full"', html)
        self.assertIn("book apply --dynasty ohio_state", html)
        self.assertIn("LIVE META RESEARCH DID NOT RUN", html)
        self.assertIn("Custom playbook limits", html)
        self.assertEqual(plan["meta_scout"]["research_status"], "failed")
        txt = format_delta_text(plan)
        self.assertIn("EDIT LIST", txt)
        self.assertIn("RESULTING BOOK", txt)


if __name__ == "__main__":
    unittest.main()
