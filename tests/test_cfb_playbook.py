"""v1.14: formation-level autonomous CFB custom playbook — catalog per stock book,
seeding, decisions + hysteresis, migration of v1.13 per-play revisions, versioning,
pending-edits flow, live caller ranking over every play in the applied formations
(with meta), live-window hooks, CLI, minimal prep page vs details page."""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach import cfb_playbook as cp
from cfb_coach.cfb_catalog import (
    book_plays,
    canonical_pair,
    formation_books,
    formations,
    load_catalog,
    logged_additions,
    zone_fit,
)
from cfb_coach.db import CoachDB
from cfb_coach.session import start_session

BUNCH, CLUSTER, DEUCE, PTRIPS = "Gun Bunch X Nasty", "Gun Cluster", "Singleback Deuce Close", "Pistol Trips"
LAB = [(BUNCH, "Post Wheel Shallow"), (BUNCH, "Y Flat GoalLine"), (BUNCH, "Z Mesh GoalLine")]
ST0 = {"strikes": {}, "fade": {}, "cooldown": {}, "flags": {}, "prep_count": 0}
NAMED_PTRIPS = {"mode": "live", "named_signals": {"pairs": {}, "plays": {}, "formations": {
    PTRIPS: {"score": 6.0, "docs": 4, "sources": [{"label": "YT: CFB 27 run meta", "url": "https://youtu.be/x", "date": "2026-09-25"}]}}}}


def _log(db: CoachDB, sid: str, form: str, play: str, results: list[str], yl: int = 40, down: int = 1, dist: int = 10) -> None:
    for r in results:
        db.log_snap(opponent_id="cpu", side="offense", situation_raw=f"{down}&{dist}", our_call=f"{form} {play}",
                    formation=form, play=play, down=down, distance=dist, yardline=yl, result=r, session_id=sid)


def _fs(n: int, own: float) -> dict:
    return {"n": n, "own": own, "conf": n / (n + cp.FORM_MIN_N), "succ_rate": 0.3, "turnovers": 0}


def _fm(meta: float, book: str | None = "Ohio State") -> dict:
    return {"meta": meta, "specific": meta, "book": book, "reasons": ["x"], "cites": [], "short": "x", "lab": 0.0}


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
        _log(self.db, sid, CLUSTER, "HB Mid Draw", ["gain 9", "gain 7", "gain 12", "gain 6", "gain 8", "gain 10", "gain 7"])
        _log(self.db, sid, BUNCH, "Drive HB Under", ["incomplete", "loss 3", "incomplete", "gain 1", "incomplete", "sack", "incomplete"])
        _log(self.db, sid, BUNCH, "RZ PA X Whip", ["incomplete", "int", "incomplete", "sack", "incomplete", "incomplete", "int"], yl=88, dist=8)
        _log(self.db, sid, "singleback deuce close", "hb dive", ["td", "gain 3"], yl=97, dist=3)

    def cur_forms(self, dyn: str = "ohio_state") -> dict:
        return cp.current_rev(self.db, dyn)["book"]["formations"]


class TestCatalog(unittest.TestCase):
    def test_canonical_names_and_zone_fit(self):
        self.assertEqual(canonical_pair("gun bunch x nasty", "mesh spot")[:2], (BUNCH, "Mesh Spot"))
        self.assertTrue(canonical_pair("singleback deuce close", "hb dive")[2])
        for f, p in LAB:
            self.assertIn(p, formations()[f])
        self.assertFalse(zone_fit("Y Flat GoalLine", "open"))
        self.assertFalse(zone_fit("Deep Flood", "gl"))
        self.assertTrue(zone_fit("Inside Zone", "gl"))

    def test_per_book_play_lists_logged_additions_and_citations(self):
        osu_bunch = book_plays(BUNCH, "Ohio State")
        self.assertEqual(len(osu_bunch), 21)
        self.assertIn("Post Wheel Shallow", osu_bunch)
        self.assertNotIn("Mesh Spot", osu_bunch)  # stock OSU list has Return Mesh Spot only
        self.assertIn("Mesh Spot", logged_additions(BUNCH))
        self.assertEqual(len(book_plays(CLUSTER, "Ohio State")), 12)
        osu_deuce = book_plays(DEUCE, "Ohio State")
        self.assertEqual(len(osu_deuce), 18)
        self.assertNotIn("HB Dive", osu_deuce)
        self.assertIn("HB Dive", logged_additions(DEUCE))
        self.assertIn("Washington State", formation_books(PTRIPS))
        self.assertIn("Strong Power", book_plays(PTRIPS, "Washington State"))
        cat = load_catalog()
        urls = " ".join(c.get("url", "") for c in cat.get("citations") or [])
        self.assertIn("cfb.fan", urls)
        self.assertIn("maddenturf.com", urls)
        self.assertTrue(cat["formations"][BUNCH]["by_book"]["Ohio State"]["source"].startswith("https://cfb.fan/27/playbooks/ohio-state-off/"))


class TestSeedingAndDecisions(_DBCase):
    def test_seed_from_logged_formations_every_play_callable(self):
        self.osu_games()
        r = cp.plan_book(self.db, "ohio_state", None)
        self.assertTrue(r["seeded_now"])
        forms = self.cur_forms()
        self.assertEqual(set(forms), {BUNCH, CLUSTER, DEUCE})
        for f in forms:
            self.assertEqual(forms[f]["source_book"], "Ohio State")
        # every play in the formation comes with it + his logged plays (ground truth)
        self.assertTrue(set(book_plays(BUNCH, "Ohio State")) <= set(forms[BUNCH]["plays"]))
        self.assertIn("Mesh Spot", forms[BUNCH]["plays"])
        self.assertIn("HB Dive", forms[DEUCE]["plays"])
        cb = cp.callable_book(self.db, "ohio_state")
        self.assertEqual(len(cb["formations"][BUNCH]), 22)
        self.assertIn("Post Wheel Shallow", cb["formations"][BUNCH])  # lab play callable without per-play edits
        # no per-play edits anywhere
        for h in cp.history(self.db, "ohio_state"):
            self.assertTrue(all(e["op"] in ("add_formation", "remove_formation", "change_source") for e in h["edits"]))
        self.assertLessEqual(len(r["target"]), cp.PRACTICAL["max_formations"])
        self.assertEqual(set(r["audibles"]), set(r["target"]))
        for f, aud in r["audibles"].items():
            self.assertEqual(len(aud), 4)
            self.assertTrue(set(aud) <= set(r["target_plays"][f]))

    def test_play_flags_working_and_demoted_never_removed(self):
        self.osu_games()
        r = cp.plan_book(self.db, "ohio_state", None)
        self.assertEqual(r["flags"][f"{CLUSTER}::HB Mid Draw"]["flag"], "working")
        self.assertEqual(r["flags"][f"{BUNCH}::RZ PA X Whip"]["flag"], "demoted")
        self.assertIn("keeps failing for you", r["flags"][f"{BUNCH}::RZ PA X Whip"]["why"])
        self.assertIn("RZ PA X Whip", r["target_plays"][BUNCH])  # formation unit: demoted, not removed
        self.assertNotIn("RZ PA X Whip", r["audibles"][BUNCH])

    def test_no_churn_same_inputs_same_pending_and_no_diff_after_apply(self):
        self.osu_games()
        r1 = cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        self.assertEqual(r1["status"], "pending")
        r2 = cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        self.assertEqual(r1["pending"]["rev"], r2["pending"]["rev"])  # same proposal kept
        self.assertEqual(len(cp.history(self.db, "ohio_state")), 2)
        cp.apply_pending(self.db, "ohio_state")
        r3 = cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        self.assertEqual(r3["status"], "no_change")
        self.assertEqual(r3["edits"], [])
        self.assertEqual(r3["edit_text"], "")
        self.assertEqual(r3["apply_cmd"], "")
        self.assertTrue(cp.same_book({"A": {"source_book": "X", "plays": ["a", "b"]}}, {"A": {"source_book": "X", "plays": ["b"]}}))
        self.assertFalse(cp.same_book({"A": {"source_book": "X", "plays": []}}, {"A": {"source_book": "Y", "plays": []}}))

    def test_cut_needs_two_strikes_with_new_data_between(self):
        book = {BUNCH: "Ohio State", "Gun Trips TE": "Alabama"}
        fstats = {"Gun Trips TE": _fs(14, -0.3)}
        fmeta = {BUNCH: _fm(0.5), "Gun Trips TE": _fm(0.02, "Alabama")}
        d1 = cp.decide(book, dynasty="ohio_state", fstats=fstats, fmeta=fmeta, state=ST0, snap_count=50)
        self.assertIn("Gun Trips TE", d1["target"])
        self.assertEqual(d1["flags"]["Gun Trips TE"]["flag"], "on_notice")
        d2 = cp.decide(book, dynasty="ohio_state", fstats=fstats, fmeta=fmeta, state=d1["state"], snap_count=50)
        self.assertIn("Gun Trips TE", d2["target"])  # no new snaps -> no second strike
        d3 = cp.decide(book, dynasty="ohio_state", fstats=fstats, fmeta=fmeta, state=d2["state"], snap_count=60)
        self.assertNotIn("Gun Trips TE", d3["target"])
        self.assertIn("CUT", d3["reasons"]["Gun Trips TE"])

    def test_overwhelming_failure_cut_now_working_off_meta_kept_meta_failing_demoted(self):
        book = {BUNCH: "Ohio State", CLUSTER: "Ohio State", "Gun Trips TE": "Alabama"}
        fstats = {"Gun Trips TE": _fs(25, -0.6), CLUSTER: _fs(15, 0.2), BUNCH: _fs(40, -0.3)}
        fmeta = {BUNCH: _fm(0.5), CLUSTER: _fm(-0.2), "Gun Trips TE": _fm(0.0, "Alabama")}
        d = cp.decide(book, dynasty="ohio_state", fstats=fstats, fmeta=fmeta, state=ST0, snap_count=10)
        self.assertNotIn("Gun Trips TE", d["target"])
        self.assertEqual(d["flags"][CLUSTER]["flag"], "working")
        self.assertEqual(d["flags"][BUNCH]["flag"], "demoted")
        self.assertIn(BUNCH, d["target"])

    def test_add_bar_hysteresis_one_per_prep_and_swap_margin(self):
        bar = cp.ADD_MIN["ohio_state"]
        book = {BUNCH: "Ohio State"}
        fmeta = {BUNCH: _fm(0.5), PTRIPS: _fm(bar - 0.05, "Washington State"), "Gun Power I Tight": _fm(0.6, "West Virginia"),
                 "Pistol U Off Trips": _fm(0.55, "Washington State")}
        d = cp.decide(book, dynasty="ohio_state", fstats={}, fmeta=fmeta, state=ST0, snap_count=0)
        self.assertEqual(d["adds"], ["Gun Power I Tight"])  # one formation per prep, best first
        self.assertEqual(d["target"]["Gun Power I Tight"], "West Virginia")
        fmeta2 = {BUNCH: _fm(0.5), PTRIPS: _fm(bar - 0.05, "Washington State")}
        d = cp.decide(book, dynasty="ohio_state", fstats={}, fmeta=fmeta2, state=ST0, snap_count=0)
        self.assertNotIn(PTRIPS, d["target"])  # below the bar
        d = cp.decide(book, dynasty="ohio_state", fstats={}, fmeta=fmeta2, state=ST0, snap_count=0,
                      pending={BUNCH: {"source_book": "Ohio State"}, PTRIPS: {"source_book": "Washington State"}})
        self.assertIn(PTRIPS, d["target"])  # already proposed -> lower bar to stay
        # full book: swap only with a clear margin over the weakest incumbent
        names = [f for f in formations() if f != PTRIPS][: cp.PRACTICAL["max_formations"]]
        full = {f: "Ohio State" for f in names}
        fm = {f: _fm(0.20) for f in names}
        fm[PTRIPS] = _fm(0.30, "Washington State")
        d = cp.decide(full, dynasty="ohio_state", fstats={}, fmeta=fm, state=ST0, snap_count=0)
        self.assertNotIn(PTRIPS, d["target"])  # 0.30 < 0.20 + margin
        fm[PTRIPS] = _fm(0.6, "Washington State")
        d = cp.decide(full, dynasty="ohio_state", fstats={}, fmeta=fm, state=ST0, snap_count=0)
        self.assertIn(PTRIPS, d["target"])
        self.assertEqual(len(d["target"]), cp.PRACTICAL["max_formations"])

    def test_named_research_adds_whole_formation_from_its_source_book(self):
        self.osu_games()
        fm = cp.formation_meta_scores(NAMED_PTRIPS, dynasty="ohio_state")
        self.assertGreaterEqual(fm[PTRIPS]["meta"], cp.ADD_MIN["ohio_state"])
        self.assertEqual(fm[PTRIPS]["book"], "Washington State")  # research names the WSU book
        self.assertGreater(fm[BUNCH]["meta"], fm[PTRIPS]["meta"])
        r = cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        add = [e for e in r["edits"] if e["op"] == "add_formation"]
        self.assertEqual([e["formation"] for e in add], [PTRIPS])
        self.assertEqual(add[0]["source_book"], "Washington State")
        self.assertEqual(add[0]["plays"], book_plays(PTRIPS, "Washington State"))  # every play comes with it
        self.assertIn(f"+ ADD FORMATION  {PTRIPS}  (from the Washington State playbook, 18 plays)", r["edit_text"])
        new = [x for x in r["formation_list"] if x["status"] == "new"]
        self.assertEqual([x["formation"] for x in new], [PTRIPS])
        self.assertLessEqual(len(new[0]["note"]), 100)
        self.assertTrue(r["apply_cmd"].endswith("book apply --dynasty ohio_state"))

    def test_meta_added_formation_fades_when_support_disappears(self):
        book = {BUNCH: "Ohio State", PTRIPS: "Washington State"}
        st = dict(ST0, origin={PTRIPS: "meta"})
        fmeta = {BUNCH: _fm(0.5), PTRIPS: _fm(0.0, "Washington State")}
        d1 = cp.decide(book, dynasty="ohio_state", fstats={}, fmeta=fmeta, state=st, snap_count=0)
        self.assertEqual(d1["flags"][PTRIPS]["flag"], "fading")
        d2 = cp.decide(book, dynasty="ohio_state", fstats={}, fmeta=fmeta, state=d1["state"], snap_count=0)
        self.assertNotIn(PTRIPS, d2["target"])

    def test_alabama_starter_is_pending_and_skips_lab_failures(self):
        self.osu_games()
        sid = start_session("cpu", dynasty="ohio_state", db=self.db).session_id
        _log(self.db, sid, "Gun Trips TE", "Quick Slant", ["int", "incomplete", "sack", "incomplete", "int", "loss 2",
                                                         "incomplete", "incomplete", "int", "sack", "incomplete"])
        r = cp.plan_book(self.db, "alabama", None)
        self.assertTrue(r["first_build"])
        self.assertIsNone(cp.current_rev(self.db, "alabama"))
        pend = cp.pending_rev(self.db, "alabama")["book"]["formations"]
        self.assertIn(BUNCH, pend)
        self.assertNotIn("Gun Trips TE", pend)  # failed in the OSU lab
        self.assertIn("CREATE custom offense playbook", r["edit_text"])
        self.assertTrue(all(x["status"] == "new" for x in r["formation_list"]))


class TestMigration(_DBCase):
    def _insert_v1(self, rev: int, status: str, forms: dict, edits: list) -> None:
        cp.ensure_table(self.db.conn)
        book = {"name": "OSU LAB O (custom)", "dynasty": "ohio_state", "formations": forms, "active8": ["x"],
                "audibles": {f: ps[:4] for f, ps in forms.items()}, "flags": {}}
        self.db.conn.execute(
            f"INSERT INTO {cp.TABLE} (dynasty, rev, status, kind, created_ts, applied_ts, parent_rev, book_json, edits_json, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ohio_state", rev, status, "seed" if rev == 1 else "prep", "2026-09-26T20:00:00+00:00", None, None,
             json.dumps(book), json.dumps(edits), f"v1 rev {rev}"))
        self.db.conn.commit()

    def test_v1_per_play_rows_migrate_to_formations_idempotently(self):
        v1_cur = {BUNCH: ["Inside Zone", "Mesh Spot"], DEUCE: ["HB Dive", "Mtn Duo"]}
        self._insert_v1(1, "current", v1_cur, [{"op": "add_formation", "formation": BUNCH, "plays": ["Inside Zone"]},
                                               {"op": "add_play", "formation": BUNCH, "play": "Inside Zone"}])
        self._insert_v1(2, "pending", {BUNCH: ["Inside Zone", "Mesh Spot", "Post Wheel Shallow"], DEUCE: ["HB Dive", "Mtn Duo"]},
                        [{"op": "add_play", "formation": BUNCH, "play": "Post Wheel Shallow", "reason": "lab"}])
        self._insert_v1(3, "discarded", {BUNCH: ["Inside Zone"], "Gun Trips TE": ["Quick Slant"]},
                        [{"op": "add_formation", "formation": "Gun Trips TE", "plays": ["Quick Slant"]}])
        self.db.set_meta(cp.SCHEMA_KEY, "1")
        n = cp.migrate_v1(self.db.conn)
        self.assertGreaterEqual(n, 3)
        cur = cp.current_rev(self.db, "ohio_state")
        f = cur["book"]["formations"]
        self.assertEqual(f[BUNCH]["source_book"], "Ohio State")
        self.assertTrue(set(book_plays(BUNCH, "Ohio State")) | {"Mesh Spot"} <= set(f[BUNCH]["plays"]))
        self.assertIn("HB Dive", f[DEUCE]["plays"])
        self.assertEqual(cur["book"]["legacy_formations"], v1_cur)
        self.assertNotIn("active8", cur["book"])
        self.assertEqual([e["op"] for e in cur["edits"]], ["add_formation"])  # per-play edits dropped
        self.assertTrue(cur["book"]["legacy_edits"])
        # the v1 pending only changed plays -> superseded by the formation-level book
        self.assertIsNone(cp.pending_rev(self.db, "ohio_state"))
        self.assertEqual(cp.get_rev(self.db, "ohio_state", 2)["status"], "discarded")
        self.assertIn("superseded", cp.get_rev(self.db, "ohio_state", 2)["summary"])
        r3 = cp.get_rev(self.db, "ohio_state", 3)
        self.assertEqual(r3["edits"][0]["op"], "add_formation")
        self.assertEqual(self.db.get_meta(cp.SCHEMA_KEY), "2")
        before = self.db.conn.execute(f"SELECT book_json FROM {cp.TABLE} ORDER BY rev").fetchall()
        self.assertEqual(cp.migrate_v1(self.db.conn, force=True), 0)  # idempotent
        after = self.db.conn.execute(f"SELECT book_json FROM {cp.TABLE} ORDER BY rev").fetchall()
        self.assertEqual([tuple(r) for r in before], [tuple(r) for r in after])
        # the migrated book works end to end
        r = cp.plan_book(self.db, "ohio_state", None)
        self.assertTrue(all(e["op"] in ("add_formation", "remove_formation") for e in r["edits"]))
        self.assertLessEqual(len(r["adds"]), 1)
        self.assertIn("Post Wheel Shallow", cp.callable_book(self.db, "ohio_state")["formations"][BUNCH])

    def test_v1_pending_with_new_formation_stays_pending_as_formation_edit(self):
        self._insert_v1(1, "current", {BUNCH: ["Inside Zone"]}, [])
        self._insert_v1(2, "pending", {BUNCH: ["Inside Zone"], PTRIPS: ["HB Stretch", "Strong Power"]},
                        [{"op": "add_formation", "formation": PTRIPS, "plays": ["HB Stretch", "Strong Power"]},
                         {"op": "add_play", "formation": PTRIPS, "play": "HB Stretch"}])
        self.db.set_meta(cp.SCHEMA_KEY, "1")
        cp.migrate_v1(self.db.conn)
        pend = cp.pending_rev(self.db, "ohio_state")
        self.assertIsNotNone(pend)
        self.assertEqual([(e["op"], e["formation"]) for e in pend["edits"]], [("add_formation", PTRIPS)])
        self.assertIn("ADD FORMATION", cp.format_edit_list(pend["edits"]))
        state_raw = {"strikes": {f"{BUNCH}::X": {"n": 1}}, "origin": {f"{BUNCH}::X": "meta"}}
        self.db.set_meta(cp.STATE_KEY.format(d="ohio_state"), json.dumps(state_raw))
        cp.migrate_v1(self.db.conn, force=True)
        st = cp.load_state(self.db, "ohio_state")
        self.assertEqual(st["strikes"], {})
        self.assertEqual(st["schema"], 2)


class TestVersioning(_DBCase):
    def test_apply_and_rollback_with_cooldown(self):
        self.osu_games()
        r = cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        rev2 = r["pending"]["rev"]
        self.assertIsNone(cp.apply_pending(self.db, "ohio_state", rev=rev2 + 5))  # stale rev refused
        rec = cp.apply_pending(self.db, "ohio_state", rev=rev2)
        self.assertEqual(rec["status"], "current")
        self.assertIn(PTRIPS, rec["book"]["formations"])
        self.assertEqual(cp.get_rev(self.db, "ohio_state", 1)["status"], "superseded")
        rb = cp.rollback(self.db, "ohio_state", 1)
        self.assertEqual(rb["kind"], "rollback")
        self.assertEqual(set(self.cur_forms()), {BUNCH, CLUSTER, DEUCE})
        self.assertEqual([(e["op"], e["formation"]) for e in rb["edits"]], [("remove_formation", PTRIPS)])
        # cooldown: the next prep must not immediately re-add the rolled-back formation
        r2 = cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        self.assertEqual(r2["status"], "no_change")
        hist = cp.format_history(self.db, "ohio_state")
        self.assertIn("rollback", hist)
        self.assertIn(f"- {PTRIPS}", hist)


class TestLiveCallerUsesFormationsAndResearch(_DBCase):
    SITS = ["1&10 my 25", "2&7 my 40", "3&8 opp 45", "3&2 opp 30", "1&10 opp 15", "2&goal opp 4", "1&goal opp 2",
            "3&12 my 20", "2&3 opp 8", "4&1 opp 1", "1&10 my 35", "2&5 opp 40"]

    def _calls(self, n_seeds: int = 12):
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        out = []
        for seed in range(n_seeds):
            for raw in self.SITS:
                sit = parse_situation(raw, default_side="offense")
                out.append(make_call(sit, "cpu", self.db, rng=random.Random(seed)))
        return out

    def test_never_recommends_a_play_outside_the_applied_formations(self):
        self.osu_games()
        cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)  # rev1 current (his formations) + rev2 pending (+Pistol Trips)
        allowed = {(f, p) for f, ps in cp.callable_book(self.db, "ohio_state")["formations"].items() for p in ps}
        calls = self._calls()
        for c in calls:
            self.assertIn((c.formation, c.play), allowed, c.rationale)
            self.assertNotEqual(c.formation, PTRIPS)
        self.assertTrue(any("pending edit(s) not callable" in c.rationale for c in calls))
        # the whole formation is callable, not just the plays he logged or the old menus
        logged = {"Inside Zone", "Mesh Spot", "HB Mid Draw", "Drive HB Under", "RZ PA X Whip", "HB Dive"}
        self.assertTrue({c.play for c in calls} - logged)

    def test_newly_added_untested_formation_gets_meta_prior_and_real_calls(self):
        from cfb_coach.meta_align import MetaPriors

        self.osu_games()
        (self.path.parent / "meta_cache.json").write_text(json.dumps({"result": NAMED_PTRIPS}), encoding="utf-8")
        mp = MetaPriors.load_cached()
        pr = mp.prior("open", PTRIPS, "Strong Power")
        self.assertGreater(pr["prior"], 0.05)  # formation prior (seed WSU run shift + named this prep)
        self.assertIn("named", pr["why"])
        cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        cp.apply_pending(self.db, "ohio_state")
        calls = self._calls(15)
        share = sum(c.formation == PTRIPS for c in calls) / len(calls)
        self.assertGreater(share, 0.05, share)
        self.assertTrue(all(c.play in book_plays(PTRIPS, "Washington State") for c in calls if c.formation == PTRIPS))

    def test_rank_pool_is_every_zone_fit_play_with_situational_bonus(self):
        from cfb_coach.playcaller import _book_menu
        from cfb_coach.situation import parse_situation

        self.osu_games()
        cp.plan_book(self.db, "ohio_state", None)
        book = cp.callable_book(self.db, "ohio_state")
        sit = parse_situation("3&12 my 20", default_side="offense")
        pool, bonus = _book_menu(sit, [(BUNCH, "Mesh Spot")], book)
        n_open = sum(1 for f, ps in book["formations"].items() for p in ps if zone_fit(p, "open"))
        self.assertGreaterEqual(len(pool), n_open - 6)  # only GL-only / short-yardage plays drop out
        self.assertGreater(bonus[(BUNCH, "Mesh Spot")], bonus.get((BUNCH, "Inside Zone"), 0.0))
        self.assertLess(bonus.get((BUNCH, "Inside Zone"), 0.0), 0.0)  # 3rd & long: runs down
        short = parse_situation("3&1 opp 30", default_side="offense")
        _, b2 = _book_menu(short, [], book)
        self.assertGreater(b2.get((BUNCH, "Inside Zone"), 0.0), 0.0)

    def test_unconfirmed_first_build_is_marked(self):
        from cfb_coach.playcaller import make_call
        from cfb_coach.situation import parse_situation

        self.db.set_meta("dynasty_mode", "alabama")
        self.osu_games()
        cp.plan_book(self.db, "alabama", None)
        c = make_call(parse_situation("1&10 my 30", default_side="offense"), "cpu", self.db, rng=random.Random(1))
        pend = cp.callable_book(self.db, "alabama")["formations"]
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
        cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
        ctrl = LivePlayController(
            db=self.db, opponent_id="cpu", make_call=lambda sit, **k: None, parse_situation=parse_situation,
            learn_summary=lambda: "", dynasty="ohio_state", cpu_only=True,
            book_info=lambda: cp.live_book_info(self.db, "ohio_state"),
            book_apply=lambda rev: cp.apply_pending(self.db, "ohio_state", rev),
        )
        st = ctrl.state()["book"]
        self.assertEqual(st["callable_rev"], 1)
        self.assertTrue(st["pending_rev"])
        self.assertIn("+ ADD FORMATION", st["pending_text"])
        self.assertIn("book-panel", render_live_html(ctrl))
        res = ctrl.apply_book(st["pending_rev"])
        self.assertTrue(res["ok"])
        self.assertEqual(res["state"]["book"]["callable_rev"], st["pending_rev"])
        self.assertFalse(ctrl.apply_book(None)["ok"])
        self.assertIn("4 formations", cp.live_status_line(self.db, "ohio_state"))
        plain = LivePlayController(db=self.db, opponent_id="cpu", make_call=lambda sit, **k: None,
                                   parse_situation=parse_situation, learn_summary=lambda: "")
        self.assertIsNone(plain.state()["book"])

    def test_cli_book_commands(self):
        from cfb_coach.cli import main

        self.osu_games()
        cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)
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
        self.assertIn("[Ohio State playbook]", out)
        self.assertIn("ADD FORMATION  Pistol Trips", out)


REMOVED_FROM_MAIN = [
    "Every callable play", "Revision history", "Custom playbook limits", "Zone plan", "Meta vs your data",
    "Current meta — freshness", "Recent headlines", "Retrain constants", "Plays named in current sources",
    "Formations — why each", "Add candidates", "Playbook adjustments", "Call emphasis", "Inventory",
]


class TestPrepPage(_DBCase):
    def _plan(self, scout_fail: bool = True):
        from cfb_coach import meta_scout as ms
        from cfb_coach.install_sheet import build_prep_plan

        def down(url, timeout):
            src = ms.MetaSource(url=url, label="x", kind="")
            src.error = "URLError: offline"
            return src, ""

        with mock.patch.object(ms, "_fetch_one", down):
            return build_prep_plan("cpu", {}, db=self.db, persist=True, dynasty="ohio_state")

    def test_minimal_prep_page_only_formations_audibles_macros(self):
        from cfb_coach.prep_browser import render_prep_details_html, render_prep_html

        self.osu_games()
        cp.plan_book(self.db, "ohio_state", NAMED_PTRIPS)  # leaves a pending Pistol Trips add
        plan = self._plan()
        html = render_prep_html(plan)
        for want in ("Formations — OSU LAB O (custom)", BUNCH, CLUSTER, DEUCE, "Audibles (4 per formation)", "Macros",
                     "LIVE META RESEARCH DID NOT RUN", "Meta research FAILED", 'data-target="book-apply-cmd"',
                     "book apply --dynasty ohio_state", ">NEW<", "Copy block", "copy-btn"):
            self.assertIn(want, html, want)
        for gone in REMOVED_FROM_MAIN:
            self.assertNotIn(gone, html, gone)
        self.assertEqual(html.count("copy-btn\" data-target=\"book-apply-cmd"), 1)  # a single apply command
        det = render_prep_details_html(plan)
        for want in ("Every callable play", "Revision history", "Custom playbook limits", "Zone plan",
                     "Current meta — freshness", "Formations — why each", "LIVE META RESEARCH DID NOT RUN",
                     "Plays named in current sources", "Retrain constants"):
            self.assertIn(want, det, want)
        self.assertEqual(plan["meta_scout"]["research_status"], "failed")

    def test_cpu_macros_are_offense_custom_adjustments_with_exact_settings(self):
        from cfb_coach.install_sheet import format_delta_text, format_prep_minimal_text

        self.osu_games()
        plan = self._plan()
        cards = plan["macro_cards"]
        self.assertTrue(0 < len(cards) <= 8)
        self.assertTrue(all(c["side"] == "offense" and c["copy_block"] and c["full_settings"] for c in cards))
        ids = [c["id"] for c in cards]
        self.assertIn("RZ", ids)
        self.assertIn("O-RUN", ids)
        self.assertEqual(plan["loadout"]["meter"], f"Active {len(cards)}/8 (O-only)")
        txt = format_prep_minimal_text(plan)
        self.assertIn("## Formations", txt)
        self.assertIn("## Audibles", txt)
        self.assertIn("## Macros — Active", txt)
        self.assertIn("MACRO: RZ (offense)", txt)
        self.assertIn("  [ ] WR1: fade", txt)  # Aidan's RZ route, verbatim
        self.assertNotIn("RESULTING BOOK", txt)
        full = format_delta_text(plan)
        self.assertIn("RESULTING BOOK", full)
        self.assertIn("FORMATIONS (why)", full)

    def test_prep_writes_details_page_not_opened_and_cli_views(self):
        from cfb_coach.cli import main

        self.osu_games()
        buf = io.StringIO()
        with mock.patch("cfb_coach.prep_browser.open_prep_html") as op, contextlib.redirect_stdout(buf):
            self.assertEqual(main(["prep", "-o", "cpu", "--dynasty", "ohio_state", "--offline", "--no-open"]), 0)
        det = self.path.parent / "prep_details_cpu.html"
        self.assertTrue(det.is_file())
        self.assertTrue((self.path.parent / "prep_cpu.html").is_file())
        self.assertTrue(det.with_suffix(".txt").is_file())
        opened = [str(c.args[0]) for c in op.call_args_list]
        self.assertFalse(any("prep_details" in o for o in opened))
        self.assertIn("Details (reasons, research, sources, history)", buf.getvalue())
        b2 = io.StringIO()
        with contextlib.redirect_stdout(b2):
            main(["prep", "-o", "cpu", "--dynasty", "ohio_state", "--offline", "--text"])
        self.assertIn("## Audibles", b2.getvalue())
        self.assertNotIn("RESULTING BOOK", b2.getvalue())
        b3 = io.StringIO()
        with contextlib.redirect_stdout(b3):
            main(["prep", "-o", "cpu", "--dynasty", "ohio_state", "--offline", "--text", "--details"])
        self.assertIn("RESULTING BOOK", b3.getvalue())


if __name__ == "__main__":
    unittest.main()
