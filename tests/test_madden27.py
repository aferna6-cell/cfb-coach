"""Madden 27 Franchise typed coach — seed, meta label, CLI (user + CPU), config, CFB regressions."""

from __future__ import annotations

import io
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cfb_coach.cli import build_parser, main
from cfb_coach.db import CoachDB
from cfb_coach.games import GAMES, MADDEN27, madden_db_path, normalize_game
from cfb_coach.madden import data as mdata
from cfb_coach.madden.franchise import (
    load_config,
    profile_config,
    resolve_nfl_team,
    save_config,
)
from cfb_coach.madden.macros import USER_ACTIVE_CAP, split_loadout
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.prep import build_prep_plan
from cfb_coach.madden.prep_browser import render_prep_html
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.seed import load_seed as load_cfb_seed


class _Isolated(unittest.TestCase):
    """Every test gets its own data dir (CFB DB, Madden DB, config, overlay)."""

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        env = {
            "CFB_COACH_DB": str(self.dir / "coach.db"),
            "HOME": str(self.dir),
        }
        self._env = mock.patch.dict(os.environ, env, clear=False)
        self._env.start()
        for k in ("CFB_COACH_MADDEN_DB", "CFB_COACH_MADDEN_PRIMARY_TEAM", "CFB_COACH_MADDEN_LAB_TEAM"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        self._env.stop()
        self._td.cleanup()

    def madden_db(self) -> CoachDB:
        return CoachDB(madden_db_path(), seed=mdata.load_seed())

    def run_cli(self, argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch("cfb_coach.prep_browser.open_prep_html"):
            rc = main(argv)
        return rc, buf.getvalue()


class TestMaddenSeedAndMeta(unittest.TestCase):
    def test_seed_loads_scheme_pack_and_primary_tbd(self) -> None:
        seed = mdata.load_seed()
        self.assertEqual(seed["game"], "Madden 27")
        self.assertEqual(seed["mode"], "franchise")
        self.assertIsNone(seed["league"]["franchise_profiles"]["primary"]["team"])
        self.assertEqual(seed["league"]["franchise_profiles"]["primary"]["id"], "franchise_primary")
        self.assertEqual(seed["league"]["franchise_profiles"]["lab"]["id"], "franchise_lab")
        forms = seed["playbooks"]["offense_formations"]
        self.assertIn("Gun Doubles Clamp Stack", forms)
        self.assertIn("Mtn Shuffle Verts Smash", forms["Gun Doubles Clamp Stack"]["core"])
        self.assertIn("Texas Y-Stutter Wheel", forms["Gun Doubles Clamp Stack"]["core"])
        self.assertIn("Pistol Trips", forms)
        self.assertNotIn("Pistol Trips RZ", forms)
        self.assertIn("Nickel Over", seed["playbooks"]["defense_packages"])
        self.assertNotIn("Nickel Mug", seed["playbooks"]["defense_packages"])
        self.assertIn("NOT Aidan's Franchise team", seed["scheme_pack"]["note"])

    def test_meta_label(self) -> None:
        bl = mdata.load_meta_baseline()
        self.assertEqual(bl["version"], "madden27-2026-09")
        self.assertEqual(mdata.META_VERSION, "madden27-2026-09")
        self.assertEqual(GAMES[MADDEN27].meta_version, "madden27-2026-09")
        self.assertEqual(bl["validated"], "2026-09-23")
        notes = mdata.validation_notes()
        self.assertIn("COMMUNITY-DERIVED", notes)
        self.assertIn("DRIFTS", notes)
        self.assertIn("CONTESTED", notes)

    def test_shared_personas_no_cfb_playbook_leak(self) -> None:
        cfb = load_cfb_seed()["opponents"]
        madden = mdata.load_seed()["opponents"]
        self.assertEqual(set(cfb), set(madden))  # same cast, no Madden-only personas
        self.assertEqual(madden["gavin"]["display_name"], "Gavin")
        self.assertEqual(madden["gavin"]["archetype"], "split_field_zone")
        self.assertEqual(madden["cpu"]["display_name"], "CPU Franchise")
        for prof in madden.values():
            self.assertNotIn("offense", prof)  # CFB concept lists never carried over
            self.assertNotIn("sample_snaps", prof)

    def test_macro_catalog_cap_and_status(self) -> None:
        cat = mdata.load_macro_catalog()
        active = cat["franchise_profiles"]["primary"]["default_active"]
        self.assertEqual(len(active), USER_ACTIVE_CAP)
        split = split_loadout(active)
        self.assertEqual((len(split["defense"]), len(split["offense"])), (6, 2))
        for name, m in cat["macros"].items():
            self.assertEqual(m["validated_status"], "meta_grounded", name)
            self.assertTrue(m["copy_block"])
            self.assertTrue(all(v["status"] == "approx" for v in m["full_settings"].values()))

    def test_every_menu_call_is_stocked_and_verified(self) -> None:
        """Validated meta: playcaller only calls plays that exist in the stocked formation."""
        seed = mdata.load_seed()
        bl = mdata.load_meta_baseline()
        from cfb_coach.madden.playbook import formation_catalog

        o = formation_catalog("offense")
        d = seed["playbooks"]["defense_packages"]
        og = bl["offense_gameplan"]
        entries = [e for menu in og["situations"].values() for e in menu]
        entries += [e for menu in og["coverage_answers"].values() for e in menu]
        entries += [{"formation": f["formation"], "play": p}
                    for f in og["pivot_families"].values() for p in f["plays"]]
        for e in entries:
            self.assertIn(e["play"], o[e["formation"]], e)  # catalog = verified plays only
        dg = bl["defense_gameplan"]
        calls = [(dg["home_package"], c) for c in dg["home_rotation"]]
        calls += [(v["package"], c) for v in dg["situational"].values() for c in v["calls"]]
        for pkg, call in calls:
            self.assertIn(call, d[pkg]["calls"], (pkg, call))
        for m in mdata.load_macro_catalog()["macros"].values():
            pkg, _, call = m["shell_pair"].partition(" — ")
            if pkg in d:
                self.assertIn(call.split(" / ")[0], d[pkg]["calls"], m["name"])

    def test_custom_adjustments_path_is_ea_documented(self) -> None:
        cat = mdata.load_macro_catalog()
        self.assertEqual(cat["xbox_path"][:2], ["Create & Share", "Custom Adjustments"])
        self.assertIn("10 active", cat["active_cap"]["note"])

    def test_game_aliases(self) -> None:
        self.assertEqual(normalize_game("madden"), MADDEN27)
        self.assertEqual(normalize_game(None), "cfb27")
        with self.assertRaises(ValueError):
            normalize_game("mut")


class TestMaddenSituation(unittest.TestCase):
    def test_madden_concepts_prev_and_live(self) -> None:
        sit = parse_madden_situation("2&7 my 35 stick wheel")
        self.assertEqual((sit.down, sit.distance, sit.yardline), (2, 7, 35))
        self.assertEqual(sit.concept_hint, "Stick Wheel")
        self.assertEqual(sit.concept_source, "last")
        live = parse_madden_situation("1&10 showing cover 3 match")
        self.assertEqual(live.coverage_hint, "Cover 3 Match")
        self.assertEqual(live.coverage_source, "live")

    def test_cfb_book_names_do_not_leak(self) -> None:
        sit = parse_madden_situation("1&10 mesh spot")
        self.assertEqual(sit.concept_hint, "Mesh")
        self.assertEqual(parse_madden_situation("2&6 texas y stutter").concept_hint, "Texas Y-Stutter Wheel")
        self.assertEqual(parse_madden_situation("1&10 showing nickel sim 2").coverage_hint, "pressure")

    def test_own_territory_is_not_red_zone(self) -> None:
        self.assertFalse(parse_madden_situation("1&10 my 12").red_zone)
        self.assertTrue(parse_madden_situation("1&10 opp 12").red_zone)


class TestMaddenPlaycaller(_Isolated):
    def test_cpu_is_offense_only(self) -> None:
        sit = parse_madden_situation("d 3&8", default_side="defense")
        call = make_call(sit, "cpu", None, rng=random.Random(1))
        self.assertEqual(call.side, "offense")
        self.assertIn("offense-only", call.rationale)
        self.assertIn(call.formation, mdata.load_seed()["playbooks"]["offense_formations"])

    def test_user_defense_uses_madden_packages_and_format(self) -> None:
        sit = parse_madden_situation("d 1&10", default_side="defense")
        call = make_call(sit, "gavin", None, rng=random.Random(2))
        self.assertEqual(call.side, "defense")
        self.assertIn(call.formation, mdata.load_seed()["playbooks"]["defense_packages"])
        line = call.format().splitlines()[0]
        self.assertRegex(line, r"^.+ — .+ \| .+ \| User .+$")

    def test_offense_format_two_reads(self) -> None:
        call = make_call(parse_madden_situation("1&10"), "gavin", None, rng=random.Random(4))
        self.assertIn(" → ", call.format())
        self.assertEqual(call.format().count(" | "), 2)

    def test_one_tell_no_macro_repeated_arms(self) -> None:
        db = self.madden_db()
        try:
            one = make_call(parse_madden_situation("d 2&6 showing 4 verts"), "gavin", db, rng=random.Random(5))
            self.assertIsNone(one.macro)
            self.assertIn("MATCH-4", one.suggest_macro or "")
            for _ in range(2):
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="d 2&6",
                            our_call="x", formation="Nickel Mug", play="Cover 4 Quarters",
                            result="+20", concept_seen="Four Verticals")
            rep = make_call(parse_madden_situation("d 2&6 showing 4 verts"), "gavin", db, rng=random.Random(5))
            self.assertEqual(rep.macro, "MATCH-4")
            self.assertIn("MATCH-4", rep.format())
            prev = make_call(parse_madden_situation("d 2&6 4 verts"), "gavin", db, rng=random.Random(5))
            self.assertIsNone(prev.macro)  # previous-snap tell never arms (CFB parity)
        finally:
            db.close()


class TestMaddenPrep(_Isolated):
    def test_cpu_prep_offense_only(self) -> None:
        plan = build_prep_plan("cpu", offline=True, persist=False)
        self.assertTrue(plan["offense_only"])
        self.assertTrue(all(c["side"] == "offense" for c in plan["macro_cards"]))
        self.assertIn("offense only", plan["loadout"]["meter"])
        self.assertEqual(plan["version"], "madden27-2026-09")
        self.assertEqual(plan["meta_scout"]["baseline_fallback"], "madden27-2026-09")

    def test_user_prep_eight_cap_and_html(self) -> None:
        plan = build_prep_plan("gavin", offline=True, persist=False)
        self.assertFalse(plan["offense_only"])
        self.assertEqual(plan["loadout"]["total"], USER_ACTIVE_CAP)
        self.assertTrue(plan["shown_deltas"])
        self.assertTrue(all(d["validated_status"] == "meta_grounded" for d in plan["shown_deltas"]))
        html = render_prep_html(plan)
        self.assertIn("Playbook of record", html)
        self.assertIn("Show full playbook", html)
        self.assertNotIn("assumed stocked", html)
        self.assertIn("Madden 27 Franchise", html)
        self.assertIn("TBD", html)
        self.assertIn("madden27-2026-09", html)
        self.assertNotIn("cfb27-2026-09", html)

    def test_lab_add_swaps_to_stay_at_cap(self) -> None:
        plan = build_prep_plan("quen", offline=True, persist=False, profile="lab")
        adds = [d for d in plan["shown_deltas"] if d["kind"] == "macro" and d["action"] == "ADD"]
        self.assertEqual([d["target"] for d in adds], ["HEAT"])
        self.assertTrue(adds[0].get("swap_plan"))
        self.assertIn("HEAT", plan["active_after"])
        self.assertEqual(len(plan["active_after"]), USER_ACTIVE_CAP)
        self.assertTrue(plan["replacing_lines"])

    def test_thin_persona_picks_stock_book_nothing_to_build(self) -> None:
        plan = build_prep_plan("ryan", offline=True, persist=False)
        off = plan["playbook"]["offense"]
        self.assertEqual((off["record"]["mode"], off["record"]["name"]), ("stock", "Buccaneers"))
        self.assertEqual(off["checklist"], [])
        self.assertEqual({d["action"] for d in plan["shown_deltas"]}, {"USE STOCK"})


class TestPlaybookOfRecord(_Isolated):
    """Owner contract: stock vs custom every prep; custom = full list then formation ADD/REMOVE only;
    switch any time; live calls hard-locked to the book."""

    def _prep(self, oid: str, **kw):
        db = self.madden_db()
        try:
            return build_prep_plan(oid, db=db, offline=True, **kw)
        finally:
            db.close()

    def test_first_custom_lists_every_formation_then_formation_diffs_only(self) -> None:
        first = self._prep("gavin")
        off = first["playbook"]["offense"]
        self.assertEqual(off["record"]["mode"], "custom")
        self.assertEqual(off["change"], "first_custom")
        listed = [i["formation"] for i in off["checklist"]]
        self.assertEqual(listed, list(off["record"]["formations"]))
        self.assertIn("Pistol Deuce Close", listed)

        nxt = self._prep("quen")
        off2 = nxt["playbook"]["offense"]
        self.assertEqual(off2["change"], "diff")
        self.assertEqual(off2["checklist"], [])  # no full rebuild
        acts = {(d["action"], d["target"]) for d in off2["deltas"]}
        self.assertEqual(acts, {("ADD", "Gun Tight"), ("REMOVE", "Pistol Deuce Close")})
        book_deltas = [d for d in nxt["shown_deltas"] if d["kind"] == "playbook"]
        self.assertTrue(all(d["action"] in ("ADD", "REMOVE") for d in book_deltas))
        self.assertIn("Gun Tight", off2["record"]["formations"])  # full list still available

        again = self._prep("quen")
        self.assertEqual(again["playbook"]["offense"]["change"], "none")

    def test_switch_custom_to_stock_and_back(self) -> None:
        self._prep("gavin")
        stock = self._prep("gavin", o_book="stock:Shotgun Classic")
        rec = stock["playbook"]["offense"]["record"]
        self.assertEqual((rec["mode"], rec["name"]), ("stock", "Shotgun Classic"))
        self.assertEqual(set(rec["formations"]), {"Gun Doubles Clamp Stack", "Gun 5WR Tight"})
        back = self._prep("ryan", o_book="custom")
        self.assertEqual(back["playbook"]["offense"]["change"], "switch")
        self.assertTrue(back["playbook"]["offense"]["checklist"])  # full list on switch-to-custom

    def test_no_audible_edits_as_install_steps(self) -> None:
        plan = self._prep("gavin")
        self.assertFalse([d for d in plan["shown_deltas"] if d["field"] == "Audibles"])

    def test_live_calls_hard_locked_to_book(self) -> None:
        from cfb_coach.madden.playbook import eligible, make_record

        books = [
            {"offense": make_record("offense", "stock", "Shotgun Classic")["formations"],
             "defense": {"Nickel Over": ["Cover 4 Quarters", "Tampa 2"]}},
            {"offense": make_record("offense", "stock", "Buccaneers")["formations"],
             "defense": make_record("defense", "stock", "49ers")["formations"]},
            {"offense": make_record("offense", "custom", None,
                                    formations=["Gun Tight", "Pistol Deuce Close"])["formations"],
             "defense": {"Dime 3-2 Odd": ["Cover 4 Quarters"]}},
        ]
        sits = ["1&10", "2&7", "3&1", "3&12", "4&goal", "1&10 opp 8", "2&5 2min",
                "2&7 showing cover 1", "2&7 showing cover 3", "2&7 showing blitz", "3&8 showing cover 4",
                "d 1&10", "d 3&1", "d 3&14", "d 4&goal", "d 2&6 showing 4 verts", "d 2&6 showing mesh"]
        db = self.madden_db()
        try:
            for i in range(3):  # repeated live tells unlock coverage answers + macros
                db.log_snap(opponent_id="gavin", side="offense", situation_raw="x", our_call="x",
                            formation="x", play="x", result="-2", coverage_seen="Cover 1")
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="x", our_call="x",
                            formation="x", play="x", result="+20", concept_seen="Four Verticals")
            pivots = {"offense": 0, "defense": 0}
            for book in books:
                for oid in ("gavin", "quen", "cpu"):
                    for raw in sits:
                        for seed in range(4):
                            sit = parse_madden_situation(raw, default_side="defense" if raw.startswith("d ") else "offense")
                            call = make_call(sit, oid, db, rng=random.Random(seed), playbook=book)
                            side_book = book[call.side]
                            self.assertIn(call.formation, side_book, (raw, oid, call.format()))
                            self.assertIn(call.play, side_book[call.formation], (raw, oid, call.format()))
                            if "PIVOT" in call.rationale:
                                pivots[call.side] += 1
            # gavin's logged fail streaks (3 per side) must drive the pivot path too
            self.assertGreater(pivots["offense"], 0)
            self.assertGreater(pivots["defense"], 0)
            # default (no explicit playbook) = the DB-locked book
            locked = eligible(__import__("cfb_coach.madden.playbook", fromlist=["x"]).active_books(db))
            call = make_call(parse_madden_situation("4&goal"), "gavin", db, rng=random.Random(0))
            self.assertIn(call.play, locked["offense"][call.formation])
        finally:
            db.close()

    def test_playbook_cli_and_flag_guards(self) -> None:
        self.run_cli(["prep", "--game", "madden27", "-o", "quen", "--offline", "--text"])
        rc, out = self.run_cli(["playbook"])
        self.assertEqual(rc, 0)
        self.assertIn("Gun Tight", out)
        self.assertIn("[custom]", out)
        rc, out = self.run_cli(["prep", "--game", "madden27", "-o", "quen", "--offline", "--text",
                                "--o-book", "stock:bucs"])
        self.assertIn("Buccaneers [stock]", out)
        with self.assertRaises(SystemExit):
            self.run_cli(["prep", "--game", "madden27", "-o", "quen", "--o-book", "stock:chiefs", "--text"])
        with self.assertRaises(SystemExit):
            self.run_cli(["prep", "-o", "gavin", "--o-book", "custom", "--offline", "--text"])


class TestPrimaryTeamConfig(_Isolated):
    def test_default_tbd_then_set_and_clear(self) -> None:
        self.assertIsNone(load_config()["primary_team"])
        self.assertIsNone(profile_config("primary")["team"])
        cfg, warn = save_config(primary_team="bucs")
        self.assertEqual(cfg["primary_team"], "Tampa Bay Buccaneers")
        self.assertEqual(warn, [])
        plan = build_prep_plan("ryan", offline=True, persist=False)
        self.assertIn("matches primary team", plan["playbook"]["offense"]["reason"])
        cfg, _ = save_config(clear_primary=True)
        self.assertIsNone(cfg["primary_team"])

    def test_team_resolution_and_env_override(self) -> None:
        self.assertEqual(resolve_nfl_team("49ers"), ("San Francisco 49ers", True))
        self.assertEqual(resolve_nfl_team("KC"), ("Kansas City Chiefs", True))
        self.assertFalse(resolve_nfl_team("Los Angeles")[1])  # ambiguous city
        self.assertFalse(resolve_nfl_team("London Monarchs")[1])  # kept as custom
        with mock.patch.dict(os.environ, {"CFB_COACH_MADDEN_PRIMARY_TEAM": "Eagles"}):
            self.assertEqual(load_config()["primary_team"], "Philadelphia Eagles")


class TestMaddenCli(_Isolated):
    def test_play_once_cpu_and_user(self) -> None:
        overlay = self.dir / "ov.html"
        rc, out = self.run_cli(["play", "--game", "madden27", "--opponent", "cpu",
                                "--once", "1&10 my 35 stick wheel", "--overlay", str(overlay)])
        self.assertEqual(rc, 0)
        self.assertIn("OFFENSE-ONLY", out)
        self.assertIn("heard: 1&10 yl35 [prev:Stick Wheel]", out)
        html = overlay.read_text(encoding="utf-8")
        self.assertIn("Madden Coach", html)
        self.assertIn("PLAY", html)

        rc, out = self.run_cli(["play", "--game", "madden", "-o", "gavin", "--once", "d 3&8", "--no-overlay"])
        self.assertEqual(rc, 0)
        self.assertIn("User game", out)
        self.assertRegex(out, r"\| User ")

    def test_prep_text_user_and_cpu(self) -> None:
        rc, out = self.run_cli(["prep", "--game", "madden27", "-o", "gavin", "--offline", "--text"])
        self.assertEqual(rc, 0)
        self.assertIn("Madden 27 Franchise", out)
        self.assertIn("(D 6 + O 2)", out)
        rc, out = self.run_cli(["prep", "--game", "madden27", "-o", "cpu", "--offline", "--text"])
        self.assertIn("N/A — offense only", out)

    def test_prep_browser_writes_madden_file(self) -> None:
        rc, out = self.run_cli(["prep", "--game", "madden27", "-o", "tiano", "--offline", "--no-open"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.dir / "prep_madden27_tiano.html").is_file())
        self.assertFalse((self.dir / "prep_tiano.html").exists())

    def test_opponents_config_and_flag_guards(self) -> None:
        rc, out = self.run_cli(["opponents", "--game", "madden27"])
        self.assertIn("shared personas", out)
        rc, out = self.run_cli(["config", "--primary-team", "Buccaneers"])
        self.assertIn("Tampa Bay Buccaneers", out)
        with self.assertRaises(SystemExit):
            self.run_cli(["prep", "-o", "gavin", "--franchise", "lab", "--offline", "--text"])
        with self.assertRaises(SystemExit):
            self.run_cli(["prep", "--game", "madden27", "-o", "gavin", "--dynasty", "alabama"])

    def test_postgame_proven_and_lab_promotion(self) -> None:
        db = self.madden_db()
        try:
            for _ in range(3):
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="d 2&6",
                            our_call="x", formation="Nickel Mug", play="Cover 4 Quarters",
                            macro="MATCH-4", result="stop")
        finally:
            db.close()
        rc, out = self.run_cli(["postgame", "--game", "madden27", "-o", "gavin", "--franchise", "lab"])
        self.assertEqual(rc, 0)
        self.assertIn("MATCH-4 → proven", out)
        self.assertIn("promotion candidate", out)
        rc, out = self.run_cli(["promote", "--game", "madden27", "--accept-all"])
        self.assertIn("Accepted", out)


class TestCfbUnchanged(_Isolated):
    def test_default_game_is_cfb(self) -> None:
        args = build_parser().parse_args(["play", "--opponent", "gavin"])
        self.assertEqual(args.game, "cfb27")

    def test_cfb_play_and_prep_still_cfb(self) -> None:
        rc, out = self.run_cli(["play", "-o", "cpu", "--once", "1&10 my 35", "--no-overlay"])
        self.assertEqual(rc, 0)
        cfb_forms = load_cfb_seed()["playbooks"]["offense_formations"]
        self.assertTrue(any(f in out for f in cfb_forms))
        self.assertNotIn("Madden", out)
        rc, out = self.run_cli(["prep", "-o", "cpu", "--offline", "--text"])
        self.assertEqual(rc, 0)
        self.assertIn("cfb27-2026-09", out)
        # Games keep separate DBs
        self.assertTrue((self.dir / "coach.db").exists())
        self.assertFalse((self.dir / "madden27.db").exists())


if __name__ == "__main__":
    unittest.main()
