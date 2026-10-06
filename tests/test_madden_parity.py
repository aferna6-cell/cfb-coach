"""Madden 27 ↔ CFB parity (v1.16): research-driven books, per-book catalog, audibles,
Active 8 with Aidan's verbatim settings, PLAY + MACRO calls, CFB retrain into madden27.db."""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timezone
from unittest import mock

from cfb_coach.madden import catalog
from cfb_coach.madden import playbook as pb
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.prep import build_prep_plan
from cfb_coach.madden.situation import parse_madden_situation
from tests.test_madden27 import _Isolated

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)


def _live_research(o_books=None, d_books=None, o_named=None, d_named=None):
    return {"mode": "live", "named": {"offense": o_named or {}, "defense": d_named or {}},
            "books": {"offense": {b: {"score": s, "docs": 4} for b, s in (o_books or {}).items()},
                      "defense": {b: {"score": s, "docs": 4} for b, s in (d_books or {}).items()}}}


class TestSeedFixesAndCatalog(_Isolated):
    def test_seed_errors_fixed(self) -> None:
        from cfb_coach.madden.data import load_seed

        seed = load_seed()
        pistol = seed["playbooks"]["offense_formations"]["Pistol Trips"]
        blob = json.dumps({k: pistol.get(k) for k in ("core", "verified", "audibles")})
        self.assertNotIn("Stick", blob)
        self.assertNotIn("Flood", blob)
        mug = seed["playbooks"]["defense_packages"]["Nickel Single Mug"]["calls"]
        self.assertNotIn("Cover 3 Match", mug)
        self.assertIn("DT Mike Loop 3", mug)
        # the Bucs Clamp Stack is the book with Texas Y-Stutter Wheel + Same Side Zone
        bucs = catalog.book_formations("offense", "Buccaneers")["Gun Doubles Clamp Stack"]
        self.assertIn("Texas Y-Stutter Wheel", bucs)
        self.assertIn("Same Side Zone", bucs)
        self.assertNotIn("Texas Y-Stutter Wheel", catalog.book_formations("offense", "Lions")["Gun Doubles Clamp Stack"])
        self.assertNotIn("Gun 5WR Tight", catalog.book_formations("offense", "Shotgun Classic"))
        self.assertIn("Gun 5WR Trio", catalog.book_formations("offense", "Shotgun Classic"))
        self.assertEqual(catalog.formation_family("offense", "Buccaneers", "Gun Doubles Clamp Stack"), "Shotgun")

    def test_every_book_record_is_real_plays_from_that_book(self) -> None:
        for side in ("offense", "defense"):
            for b in catalog.book_names(side):
                rec = pb.make_record(side, "stock", b)
                real = catalog.book_formations(side, b)
                self.assertTrue(rec["formations"], (side, b))
                for f, plays in rec["formations"].items():
                    self.assertEqual(plays, real[f], (side, b, f))
                for f, auds in (rec.get("audibles") or {}).items():
                    self.assertEqual(len(auds), 4, (b, f))
                    self.assertTrue(set(auds) <= set(real[f]), (b, f, auds))

    def test_stock_chiefs_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pb.parse_choice("offense", "stock:chiefs")
        self.assertEqual(pb.parse_choice("offense", "stock:lions"), ("stock", "Lions"))


class TestResearchPicksBooks(_Isolated):
    def test_default_start_without_live_research(self) -> None:
        rec = pb.recommend_book("offense", current=None, research=None, team="Detroit Lions")
        self.assertEqual(rec["book"], "Buccaneers")
        rec = pb.recommend_book("defense", current=None, research=None, team="Detroit Lions")
        self.assertEqual(rec["book"], "49ers")
        # strong signal but research did not run live → no switch
        cached = _live_research(o_books={"Texans": 9.0})
        cached["mode"] = "cache"
        self.assertEqual(pb.recommend_book("offense", current=None, research=cached, team=None)["book"], "Buccaneers")

    def test_live_research_can_switch_offense_and_defense(self) -> None:
        # the #1-ranked 49ers D needs book AND formation evidence to lose (Titans-only 46 Bear)
        res = _live_research(o_books={"Texans": 9.0}, d_books={"Titans": 9.0},
                             o_named={"formations": {"Gun Tight": {"score": 6.0, "docs": 5}}},
                             d_named={"formations": {"46 Bear": {"score": 10.0, "docs": 6}}})
        self.assertEqual(pb.recommend_book("defense", current=None, research=_live_research(d_books={"Titans": 9.0}),
                                           team=None)["book"], "49ers")
        o = pb.recommend_book("offense", current=None, research=res, team="Detroit Lions")
        self.assertEqual(o["book"], "Texans")
        self.assertIn("recommends Texans", o["reason"])
        d = pb.recommend_book("defense", current=None, research=res, team="Detroit Lions")
        self.assertEqual(d["book"], "Titans")

    def test_hysteresis_and_cooldown(self) -> None:
        weak = _live_research(o_books={"Texans": 0.6})
        self.assertEqual(pb.recommend_book("offense", current=None, research=weak, team=None)["book"], "Buccaneers")
        strong = _live_research(o_books={"Texans": 9.0})
        cur = pb.make_record("offense", "stock", "Buccaneers")
        rec = pb.recommend_book("offense", current=cur, research=strong, team=None, preps_since_switch=0)
        self.assertEqual(rec["book"], "Buccaneers")
        self.assertIn("cooldown", rec["reason"])

    def test_prep_uses_research_for_book_and_focus(self) -> None:
        from cfb_coach.madden.meta_scout import MetaScoutResult

        fake = MetaScoutResult(available=True, mode="live", research_status="live",
                               named_signals={"offense": {"formations": {"Gun Tight": {"score": 6.0, "docs": 5}},
                                                          "pairs": {}, "plays": {}},
                                              "defense": {"formations": {}, "pairs": {}, "plays": {}},
                                              "books": {"offense": {"Texans": {"score": 9.0, "docs": 6}},
                                                        "defense": {}}})
        db = self.madden_db()
        try:
            with mock.patch("cfb_coach.madden.meta_scout.run_madden_scout", return_value=fake):
                plan = build_prep_plan("gavin", db=db)
            off = plan["playbook"]["offense"]
            self.assertEqual(off["record"]["name"], "Texans")
            self.assertIn("Gun Tight", off["record"]["formations"])
            self.assertEqual(plan["playbook"]["defense"]["record"]["name"], "49ers")
            # explicit config pin beats research
            from cfb_coach.madden.franchise import save_config

            save_config(offense_book="stock:Buccaneers")
            with mock.patch("cfb_coach.madden.meta_scout.run_madden_scout", return_value=fake):
                plan = build_prep_plan("gavin", db=db)
            self.assertEqual(plan["playbook"]["offense"]["record"]["name"], "Buccaneers")
        finally:
            db.close()

    def test_madden_youtube_profile_and_vocab(self) -> None:
        from cfb_coach import yt_research as yt
        from cfb_coach.madden.meta_scout import book_mentions, named_signals

        prof = yt.madden_profile()
        mk = lambda t, d="2026-09-20": yt.Video(video_id=t[:8], title=t, published=d)  # noqa: E731
        self.assertTrue(yt.classify_video(mk("Best Madden 27 defense playbook"), now=NOW, profile=prof)[0])
        self.assertFalse(yt.classify_video(mk("College Football 27 best offense"), now=NOW, profile=prof)[0])
        self.assertFalse(yt.classify_video(mk("Madden 26 best offense"), now=NOW, profile=prof)[0])
        self.assertFalse(yt.classify_video(mk("Best Madden 27 offense", "2026-07-01"), now=NOW, profile=prof)[0])
        # CFB default profile unchanged
        self.assertFalse(yt.classify_video(mk("Best Madden 27 defense playbook"), now=NOW)[0])
        hits = book_mentions("The Buccaneers playbook is the best offense in Madden 27. "
                             "On the other side of the ball, the 49ers defense playbook is elite.")
        self.assertIn("Buccaneers", hits["offense"])
        self.assertIn("49ers", hits["defense"])
        ns = named_signals([{"label": "yt", "kind": "youtube_transcript", "date": "2026-09-25",
                             "text": "run texas y stutter wheel out of shotgun doubles clamp stack every down"}], now=NOW)
        self.assertIn("Gun Doubles Clamp Stack::Texas Y-Stutter Wheel", ns["offense"]["pairs"])


class TestMacrosResearchBuilt(_Isolated):
    def test_every_field_researched_or_default_and_cli_is_read_only(self) -> None:
        # v1.17 owner spec: Madden defense macro settings are research-built, not typed in
        from cfb_coach.madden import research_db as rdb
        from cfb_coach.madden.macros import copy_block, macro_detail

        d = macro_detail("TAMPA MABLE")
        self.assertEqual(d["n_fields"], len(d["settings"]))
        self.assertGreater(d["n_fields"], 40)  # the whole editor, not just the researched rows
        by = {(r["section"], r["setting"]): r for r in d["settings"]}
        self.assertEqual((by[("Zone Drops", "Flats")]["value"], by[("Zone Drops", "Flats")]["source"]), ("25", "civil-def-macros"))
        self.assertEqual((by[("General", "Show Blitz")]["value"], by[("General", "Show Blitz")]["source"]), ("Default", "default"))
        blk = copy_block(d)
        self.assertIn("  [ ] Flats: 25   <- civil-def-macros", blk)
        self.assertIn("In game: LB → TAMPA MABLE", blk)
        rc, out = self.run_cli(["macro-settings", "--game", "madden27"])
        self.assertEqual(rc, 0)
        self.assertIn("TAMPA MABLE", out)
        self.assertIn("qb contain", out)
        with self.assertRaises(SystemExit):
            self.run_cli(["macro-settings", "--game", "madden27", "TAMPA MABLE", "--set", "Zone Drops: Flats = 10"])
        self.assertEqual(rdb.validate(rdb.load()), [])

    def test_prep_stores_defense_ten_used_by_play(self) -> None:
        from cfb_coach.madden.macros import load_selection

        self.run_cli(["prep", "--game", "madden27", "-o", "quen", "--offline", "--text", "--franchise", "lab"])
        db = self.madden_db()
        try:
            sel = load_selection(db, "quen")
        finally:
            db.close()
        self.assertEqual((len(sel["offense"]), len(sel["defense"])), (8, 8))
        rc, out = self.run_cli(["play", "--game", "madden27", "-o", "quen", "--once", "1&10", "--no-overlay"])
        self.assertEqual(rc, 0)
        line = out.split("Active macros (from last prep):", 1)[1].split("\n", 1)[0]
        self.assertIn(sel["defense"][0], line)
        self.assertIn(sel["offense"][0], line)
        self.assertIn("PLAY: ", out)


class TestPlayMacroCalls(_Isolated):
    def _books(self):
        return {"offense": pb.make_record("offense", "stock", "Buccaneers")["formations"],
                "defense": pb.make_record("defense", "stock", "49ers")["formations"]}

    def test_offense_adjustment_with_buttons_only_when_look_calls_for_it(self) -> None:
        db = self.madden_db()
        try:
            prev = make_call(parse_madden_situation("3&8 cover 1"), "gavin", db, rng=random.Random(1),
                             playbook=self._books())
            self.assertIsNone(prev.adjustment)  # one previous-snap look: nothing
            self.assertIsNone(prev.macro)  # offense never carries a macro
            hit = None
            for seed in range(20):
                c = make_call(parse_madden_situation("3&8 showing cover 1"), "gavin", db, rng=random.Random(seed),
                              playbook=self._books())
                self.assertIsNone(c.macro)
                if c.adjustment:
                    hit = c
                    break
            self.assertIsNotNone(hit)
            self.assertEqual(hit.adjustment["label"], "Hot route WR1 → Slant")
            self.assertEqual(hit.headline(), f"PLAY: {hit.play} ({hit.formation}) + ADJ: Hot route WR1 → Slant")
            self.assertIn("press Y → tap WR1's icon button → pick Slant", hit.format())
            self.assertIn(hit.play, self._books()["offense"][hit.formation])
        finally:
            db.close()

    def test_defense_macro_has_buttons_and_live_window_shows_it(self) -> None:
        from cfb_coach.live_server import LivePlayController

        db = self.madden_db()
        try:
            for _ in range(3):
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="x", our_call="x", formation="x",
                            play="x", result="+20", concept_seen="Four Verticals")
            sit = parse_madden_situation("d 2&6 showing 4 verts", default_side="defense")
            call = make_call(sit, "gavin", db, rng=random.Random(0), playbook=self._books(),
                             active_macros={"defense": ["QTRS OVERTOP", "TAMPA MABLE"]})
            self.assertEqual(call.macro, "QTRS OVERTOP")
            self.assertEqual(call.play, "Cover 4 Quarters")  # the researched base play, in the book
            self.assertEqual(call.macro_info["buttons"], "LB → QTRS OVERTOP")
            ctrl = LivePlayController(db=db, opponent_id="gavin", make_call=lambda s, **k: call,
                                      parse_situation=parse_madden_situation, learn_summary=lambda: "")
            ctrl.last_call = call
            st = ctrl.macro_state()
            self.assertEqual(st["headline"], f"PLAY: {call.play} ({call.formation}) + MACRO: QTRS OVERTOP")
            self.assertEqual(st["press"], "LB → QTRS OVERTOP")
        finally:
            db.close()

    def test_ranker_calls_any_in_book_play_not_just_menu(self) -> None:
        books = {"offense": pb.make_record("offense", "stock", "Cardinals")["formations"]}
        seen = set()
        for seed in range(30):
            c = make_call(parse_madden_situation("1&10 my 30"), "gavin", None, rng=random.Random(seed), playbook=books)
            self.assertIn(c.play, books["offense"][c.formation])
            seen.add((c.formation, c.play))
        self.assertGreater(len(seen), 5)


class TestRetrainParity(_Isolated):
    def test_postgame_rebuilds_per_opponent_zone_weights_in_madden_db_only(self) -> None:
        from cfb_coach.madden.postgame import summary

        db = self.madden_db()
        try:
            for i, (yl, res) in enumerate([(92, "td"), (90, "+8"), (95, "td"), (40, "+12"), (45, "int")]):
                db.log_snap(opponent_id="gavin", side="offense", situation_raw="x", our_call="x",
                            formation="Gun Doubles Clamp Stack", play="Texas Y-Stutter Wheel" if i < 3 else "Inside Zone",
                            result=res, down=1, distance=10 if yl < 90 else 5, yardline=yl)
            out = summary(db, "gavin")
            keys = [r[0] for r in db.conn.execute(
                "SELECT key FROM gameplan_weights WHERE opponent_id='gavin' AND side='offense'").fetchall()]
        finally:
            db.close()
        self.assertIn("Retrain", out)
        self.assertTrue(any(k.startswith("zone::") and "Texas Y-Stutter Wheel" in k for k in keys), keys)
        self.assertTrue(any("Inside Zone" in k for k in keys))
        cfb = self.dir / "coach.db"
        if cfb.exists():
            n = sqlite3.connect(cfb).execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
            self.assertEqual(n, 0)


class TestConfigBooks(_Isolated):
    def test_config_book_flags(self) -> None:
        rc, out = self.run_cli(["config", "--game", "madden27", "--o-book", "stock:Shotgun Classic", "--d-book", "auto"])
        self.assertEqual(rc, 0)
        self.assertIn("offense book: stock:Shotgun Classic", out)
        self.assertIn("defense book: auto", out)
        with self.assertRaises(SystemExit):
            self.run_cli(["config", "--game", "madden27", "--o-book", "stock:chiefs"])


class TestMaddenScoutLiveNoNetwork(_Isolated):
    def test_live_scout_parses_books_formations_without_network(self) -> None:
        from cfb_coach import meta_scout as ms
        from cfb_coach.madden.meta_scout import research_from_scout, run_madden_scout

        page = ("<html><title>Best Madden 27 playbooks</title><body><p>The Texans playbook is the best offense in "
                "Madden 27 after the patch: Gun Tight Tight Crosses beats man.</p><p>On defense the Titans playbook "
                "and its 46 Bear front are the meta defense.</p></body></html>")

        def fake(url, timeout):
            return ms.MetaSource(url=url, fetched=True, status=200, title="Best Madden 27 playbooks"), page

        with mock.patch.object(ms, "_fetch_one", fake):
            r = run_madden_scout(youtube=False)
        self.assertEqual(r.mode, "live")
        res = research_from_scout(r)
        self.assertIn("Texans", res["books"]["offense"])
        self.assertIn("Titans", res["books"]["defense"])
        self.assertIn("Gun Tight", res["named"]["offense"]["formations"])
        # offline afterwards → loud cache fallback, never silently "live"
        r2 = run_madden_scout(offline=True)
        self.assertEqual(r2.mode, "cache")
