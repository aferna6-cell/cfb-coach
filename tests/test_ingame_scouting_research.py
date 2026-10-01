"""In-game benching, opponent scouting, the Madden defense mix, live quarter / 'they ran'
logging, and the daily AI research file prep reads."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from cfb_coach import ai_research, ingame, scouting
from cfb_coach.db import CoachDB
from cfb_coach.madden import catalog
from cfb_coach.madden import defense_select as ds
from cfb_coach.madden import playbook as pb
from cfb_coach.madden.situation import concept_family, parse_madden_situation
from cfb_coach.outcome import outcome_success

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def _snap(db, sid, side, form, play, result, *, quarter=None, concept=None, coverage=None, down=1, dist=10, yl=40,
          oid="gavin"):
    db.log_snap(opponent_id=oid, side=side, situation_raw="x", our_call=f"{form} {play}", formation=form, play=play,
                down=down, distance=dist, yardline=yl, quarter=quarter, result=result, concept_seen=concept,
                coverage_seen=coverage, session_id=sid)


class _DB(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.td.name) / "t.db")

    def tearDown(self):
        self.db.close()
        self.td.cleanup()


class TestOutcomeAssumptions(unittest.TestCase):
    def test_result_strings_used_below(self):
        self.assertFalse(outcome_success("+20", "defense"))
        self.assertTrue(outcome_success("-2", "defense"))
        self.assertTrue(outcome_success("+20", "offense"))
        self.assertFalse(outcome_success("-2", "offense"))


class TestBench(_DB):
    def test_three_fails_bench_until_halftime_then_clean(self):
        for _ in range(3):
            _snap(self.db, "g1", "offense", "Gun Trips", "Mesh", "-2", quarter=1)
        rep = ingame.bench_report(self.db, "g1", "offense", 2)
        self.assertTrue(rep.is_benched("Gun Trips", "Mesh"))
        self.assertIn("halftime", rep.note())
        self.assertFalse(ingame.bench_report(self.db, "g1", "offense", 3).benched)  # 2nd half starts clean

    def test_wins_offset_fails_and_other_side_and_games_ignored(self):
        for r in ("-2", "-2", "-2", "+12", "+12", "+12"):
            _snap(self.db, "g1", "offense", "Gun Trips", "Mesh", r, quarter=1)
        _snap(self.db, "g2", "offense", "Gun Trips", "Stick", "-2", quarter=1)
        for _ in range(3):
            _snap(self.db, "g1", "defense", "Nickel", "Cover 3", "-2", quarter=1)  # stops = successes on D
        self.assertFalse(ingame.bench_report(self.db, "g1", "offense", 1).benched)
        self.assertFalse(ingame.bench_report(self.db, "g1", "defense", 1).benched)
        self.assertFalse(ingame.bench_report(self.db, None, "offense", 1).benched)

    def test_no_quarter_uses_last_snaps_window(self):
        for _ in range(3):
            _snap(self.db, "g1", "offense", "Gun Trips", "Mesh", "-2")
        self.assertTrue(ingame.bench_report(self.db, "g1", "offense").is_benched("Gun Trips", "Mesh"))
        for _ in range(ingame.WINDOW_SNAPS):
            _snap(self.db, "g1", "offense", "I Form", "Power", "+6")
        self.assertFalse(ingame.bench_report(self.db, "g1", "offense").benched)

    def test_family_penalty_and_drop_benched_never_empties(self):
        for p in ("Cover 4 Quarters", "Cover 6", "Cover 4 Drop", "Cover 4 Palms"):
            _snap(self.db, "g1", "defense", "Nickel", p, "+20", quarter=3)
        rep = ingame.bench_report(self.db, "g1", "defense", 3, family_of=ds.call_family)
        self.assertEqual(rep.family_penalty, {"two_high": ingame.FAMILY_PENALTY})
        rep.benched[("A", "x")] = "0/3"
        self.assertEqual(ingame.drop_benched([{"formation": "A", "play": "x"}], rep), [{"formation": "A", "play": "x"}])
        rows = [{"formation": "A", "play": "x"}, {"formation": "B", "play": "y"}]
        self.assertEqual(ingame.drop_benched(rows, rep), rows[1:])


class TestScouting(_DB):
    def test_buckets_and_expected_families(self):
        for _ in range(5):
            _snap(self.db, "g1", "defense", "Nickel", "Cover 3", "+15", concept="Four Verticals", down=3, dist=9)
        _snap(self.db, "g1", "defense", "Nickel", "Cover 3", "-1", concept="HB Dive", down=1, dist=10)
        for _ in range(3):
            _snap(self.db, "g1", "offense", "Gun", "Mesh", "+8", coverage="Cover 6", down=3, dist=8)
        rep = scouting.scout_opponent(self.db, "gavin", family_of=concept_family,
                                      coverage_class_of=ds.coverage_seen_family)
        self.assertEqual(rep["snaps"], 9)
        self.assertEqual(rep["their_offense"]["buckets"]["3rd_long"]["n"], 5)
        fams, scope, n = scouting.expected_families(rep, parse_madden_situation("d 3&9", default_side="defense"))
        self.assertEqual((fams, n), ({"vert": 1.0}, 5))
        self.assertIn("long", scope)
        fams, scope, _ = scouting.expected_families(rep, parse_madden_situation("d 2&5", default_side="defense"))
        self.assertEqual(scope, "all downs")  # thin bucket borrows the overall mix
        cov, _, _ = scouting.expected_coverage(rep, parse_madden_situation("3&8"))
        self.assertEqual(cov, {"two_high": 1.0})
        self.assertEqual(rep["their_defense"]["our_plays"][0]["play"], "Mesh")
        text = "\n".join(scouting.scouting_lines(rep))
        self.assertIn("Four Verticals", text)
        html = scouting.opponent_study_html(scouting.scouting_lines(rep), ["D vs Mesh: Cover 1"])
        self.assertIn("id='scouting'", html)
        self.assertIn("counters for him", html)
        self.assertEqual(scouting.opponent_study_html([], []), "")

    def test_down_bucket(self):
        self.assertEqual(scouting.down_bucket(2, 2), "2nd_short")
        self.assertEqual(scouting.down_bucket(4, 5), "3rd_medium")
        self.assertEqual(scouting.down_bucket(1, 10, 85), "red_zone")


def _d_book() -> dict[str, list[str]]:
    for b in catalog.book_names("defense"):
        book = pb.make_record("defense", "stock", b)["formations"]
        fams = {ds.call_family(p) for plays in book.values() for p in plays}
        if {"two_high", "single_high", "man", "pressure"} <= fams:
            return book
    raise AssertionError("no stock defensive book covers the families")


class TestDefenseSelect(_DB):
    def test_call_family(self):
        cases = {"Cover 4 Quarters": "two_high", "Cover 6": "two_high", "Tampa 2": "cover2", "Cover 2 Invert": "cover2",
                 "Cover 3 Match": "single_high", "Cover 1 Robber": "man", "2 Man Under": "man",
                 "Mike Sim Pressure": "pressure", "Cover 0 Blitz": "pressure", "DT Mike Loop 3": "pressure"}
        for name, fam in cases.items():
            self.assertEqual(ds.call_family(name), fam, name)
        self.assertEqual(ds.coverage_seen_family("pressure"), "pressure")

    def test_variety_stays_in_book_and_never_three_families_in_a_row(self):
        book = _d_book()
        rng = random.Random(7)
        fams, calls, forms = Counter(), Counter(), Counter()
        for i in range(60):
            sit = parse_madden_situation(f"d {1 + i % 3}&{(3, 7, 10)[i % 3]}", default_side="defense")
            sit.extras.update(session_id="g1", quarter=1 + i // 15)
            pick = ds.select_defense(sit, "gavin", self.db, book, rng)
            self.assertIn(pick.play, book[pick.formation])
            fams[pick.family] += 1
            calls[pick.play] += 1
            forms[pick.formation] += 1
            _snap(self.db, "g1", "defense", pick.formation, pick.play, "+4" if i % 2 else "-1", quarter=1 + i // 15)
        self.assertGreaterEqual(len(fams), 4, fams)
        self.assertGreaterEqual(len(calls), 12, calls)
        self.assertGreaterEqual(len(forms), 2, forms)
        seq = [ds.call_family(r["play"]) for r in self.db.get_session_snaps("g1")]
        self.assertFalse(any(seq[i] == seq[i + 1] == seq[i + 2] for i in range(len(seq) - 2)), seq)

    def test_scouted_verts_pull_two_high_and_benched_call_never_picked(self):
        book = _d_book()
        for _ in range(12):
            _snap(self.db, "old", "defense", "x", "x", "+20", concept="Four Verticals", down=3, dist=10)
        rep = scouting.scout_opponent(self.db, "gavin", family_of=concept_family)
        sit = parse_madden_situation("d 3&10", default_side="defense")
        rng = random.Random(1)
        scouted = Counter(ds.select_defense(sit, "gavin", None, book, rng, scouting=rep).family for _ in range(300))
        plain = Counter(ds.select_defense(sit, "gavin", None, book, rng).family for _ in range(300))
        self.assertGreater(scouted["two_high"], plain["two_high"])
        two_high = [(f, p) for f, ps in book.items() for p in ps if ds.call_family(p) == "two_high"]
        f, p = two_high[0]
        for _ in range(3):
            _snap(self.db, "g2", "defense", f, p, "+20", quarter=1)
        sit.extras.update(session_id="g2", quarter=2)
        for _ in range(40):
            pick = ds.select_defense(sit, "gavin", self.db, book, rng, scouting=rep)
            self.assertNotEqual((pick.formation, pick.play), (f, p))

    def test_exclude_family_for_pivot(self):
        book = _d_book()
        sit = parse_madden_situation("d 1&10", default_side="defense")
        for s in range(30):
            pick = ds.select_defense(sit, "gavin", None, book, random.Random(s), exclude_families={"two_high"})
            self.assertNotEqual(pick.family, "two_high")


class TestLiveLogging(_DB):
    def _ctrl(self, call_side="defense"):
        from cfb_coach.live_server import LivePlayController

        class Call:
            side = call_side
            formation, play, macro = "Nickel", "Cover 4 Quarters", None

            def format(self):
                return "Nickel — Cover 4 Quarters"

        return LivePlayController(db=self.db, opponent_id="gavin", make_call=lambda sit, **k: Call(),
                                  parse_situation=parse_madden_situation, learn_summary=lambda: "", session_id="g1")

    def test_quarter_and_their_concept_logged_and_bench_shown(self):
        ctrl = self._ctrl()
        ctrl.call_only("d 3&8", side="defense", quarter=2)
        self.assertEqual(ctrl.last_sit.extras["quarter"], 2)
        self.assertEqual(ctrl.last_sit.extras["session_id"], "g1")
        for _ in range(3):
            ctrl.result_and_call(outcome="+20", sit_raw="d 1&10", side="defense", quarter=2, their="Mesh")
        rows = self.db.get_session_snaps("g1")
        self.assertEqual([r["quarter"] for r in rows], [2, 2, 2])
        self.assertTrue(all("mesh" in (r["concept_seen"] or "").lower() for r in rows))
        st = ctrl.state()
        self.assertEqual(st["quarter"], 2)
        self.assertIn("Cover 4 Quarters", st["benched"]["defense"][0])
        ctrl.result_and_call(outcome="-1", sit_raw="d 1&10", side="defense", quarter=3)
        self.assertEqual(ctrl.state()["benched"], {})

    def test_live_html_has_quarter_and_they_ran(self):
        from cfb_coach.live_server import render_live_html

        html = render_live_html(self._ctrl())
        for want in ('id="quarter"', 'id="their"', "bench-panel"):
            self.assertIn(want, html)


def _doc(game="madden27", hours_old=2.0, **extra):
    doc = {
        "schema": 1, "game": game,
        "researched_at": (NOW - timedelta(hours=hours_old)).isoformat(),
        "summary": "Cover 4 Quarters with a user robber shuts down Cross Wheels.",
        "patch": {"version": "1.04", "date": "2026-09-30", "url": "https://example.com/p", "notes": ["Zone drops tightened"]},
        "findings": [{"claim": "Gun Bunch Ace Cross Wheels is the top pass concept online", "side": "offense",
                      "source": "Guide", "url": "https://example.com/a", "published": "2026-09-29", "confidence": "high"}],
        "meta_offense": ["Cross Wheels from Gun Bunch"], "meta_defense": ["Cover 4 Quarters"],
        "suggestions": [{"side": "defense", "label": "Robber", "tip": "User the hook", "macro_hint": None,
                         "source_url": "https://example.com/a"}],
        "opponents": {"gavin": {
            "notes": ["Spams verticals on 3rd and long"],
            "defense_counters": [{"vs": "Four Verticals", "vs_family": "vert", "families": ["two_high"],
                                  "calls": ["Cover 4 Quarters"], "why": "caps both seams", "source_url": "https://example.com/a"}],
            "offense_counters": [{"vs": "Cover 6", "vs_class": "two_high", "plays": ["Mesh Spot"], "why": "x"}]}},
        "sources": [{"title": "Guide", "url": "https://example.com/a", "kind": "guide", "published": "2026-09-29",
                     "accessed": "2026-10-01"}],
    }
    doc.update(extra)
    return doc


class TestAIResearch(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.patches = [
            mock.patch.dict(os.environ, {"CFB_COACH_AI_RESEARCH": "on", "CFB_COACH_DATA_DIR": str(self.dir)}),
            mock.patch.object(ai_research, "REPO_DIR", self.dir / "repo"),
            mock.patch.object(ai_research, "cache_path", lambda g: self.dir / f"ai_research_{g}.json"),
        ]
        for p in self.patches:
            p.start()
        (self.dir / "repo").mkdir()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.td.cleanup()

    def test_validate(self):
        self.assertEqual(ai_research.validate(_doc(), "madden27"), [])
        self.assertTrue(ai_research.validate(_doc(), "cfb27"))
        self.assertTrue(ai_research.validate(_doc(findings=[]), "madden27"))
        bad = _doc()
        bad["opponents"]["gavin"]["defense_counters"][0]["families"] = ["cover7"]
        self.assertTrue(any("cover7" in e for e in ai_research.validate(bad)))
        self.assertTrue(ai_research.validate(_doc(researched_at="2026-10-01T08:00:00")))  # no timezone

    def test_load_prefers_freshest_and_caches_github(self):
        (self.dir / "repo" / "madden27.json").write_text(json.dumps(_doc(hours_old=30)))
        fresh = _doc(hours_old=1, summary="fresh")
        doc, origin = ai_research.load_research("madden27", fetch=lambda url, t: json.dumps(fresh))
        self.assertEqual((origin, doc["summary"]), ("github", "fresh"))
        self.assertTrue((self.dir / "ai_research_madden27.json").is_file())
        doc, origin = ai_research.load_research("madden27", offline=True)
        self.assertEqual((origin, doc["summary"]), ("cache", "fresh"))

        def boom(url, t):
            raise OSError("offline")

        (self.dir / "ai_research_madden27.json").unlink()
        self.assertEqual(ai_research.load_research("madden27", fetch=boom)[1], "repo")

    def test_to_scout_result_and_staleness(self):
        r = ai_research.to_scout_result(_doc(), "madden27", origin="github", now=NOW)
        self.assertEqual((r.mode, r.research_status), ("ai", "ai"))
        self.assertTrue(r.available)
        self.assertIn("1 findings", r.message)
        self.assertEqual(r.meta_defense, ["Cover 4 Quarters"])
        old = ai_research.to_scout_result(_doc(hours_old=48), "madden27", origin="cache", now=NOW)
        self.assertEqual(old.research_status, "ai-stale")
        self.assertIn("STALE", old.message)
        cfb = ai_research.to_scout_result(_doc(game="cfb27"), "cfb27", origin="repo", now=NOW)
        self.assertEqual(cfb.patch_notes[0]["version"], "1.04")

    def test_prep_research_uses_file_not_scraper(self):
        with mock.patch("cfb_coach.madden.meta_scout.run_madden_scout") as scraper, \
                mock.patch("cfb_coach.madden.meta_scout._save_cache") as save:
            r = ai_research.prep_research("madden27", fetch=lambda url, t: json.dumps(_doc(hours_old=0)))
        scraper.assert_not_called()
        save.assert_called_once()
        self.assertEqual(r.mode, "ai")
        with mock.patch("cfb_coach.madden.meta_scout.run_madden_scout") as scraper:
            ai_research.prep_research("madden27", live_scout=True)
        scraper.assert_called_once_with(offline=False)

    def test_opponent_counters(self):
        (self.dir / "repo" / "madden27.json").write_text(json.dumps(_doc()))
        self.assertEqual(ai_research.opponent_counters("madden27", "gavin", "defense")[0]["calls"], ["Cover 4 Quarters"])
        self.assertEqual(ai_research.opponent_counters("madden27", "nobody", "defense"), [])
        lines = ai_research.opponent_lines("madden27", "gavin")
        self.assertIn("Spams verticals on 3rd and long", lines)
        self.assertTrue(any(ln.startswith("D vs Four Verticals: Cover 4 Quarters") for ln in lines))

    def test_validator_script(self):
        import subprocess
        import sys

        p = self.dir / "madden27.json"
        p.write_text(json.dumps(_doc(hours_old=1, opponents={})))
        root = Path(__file__).resolve().parent.parent
        ok = subprocess.run([sys.executable, str(root / "scripts" / "validate_ai_research.py"), str(p)],
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        p.write_text(json.dumps(_doc(opponents={"not_a_real_opp": {}})))
        bad = subprocess.run([sys.executable, str(root / "scripts" / "validate_ai_research.py"), str(p)],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1, bad.stdout)


if __name__ == "__main__":
    unittest.main()
