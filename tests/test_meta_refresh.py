"""v1.12 meta refresh: live fetch (mocked), same-day TTL, --refresh-meta, fallbacks, diff, alignment."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from cfb_coach import meta_scout as ms
from cfb_coach.meta_align import MetaPriors, alignment, live_boosts

GUIDE = """<html><head><title>CFB 27 Offense Guide</title>
<meta name="description" content="College Football 27 offense guide"></head><body>
<h2>Red Zone</h2><p>In the red zone use compressed sets. If linebackers sit inside, hit the flats
and quick outs; if the defense overplays quick passes, run inside zone. Versus press man near the
goal line, mesh and whip concepts from bunch win.</p>
<p>The College Football 27 meta is run-first after passing was toned down.</p></body></html>"""
PATCH = """<html><head><title>College Football 27 Update 1.013 Patch Notes</title></head><body>
<h2>Gameplay</h2><ul><li>Improved run-action blocking threat detection on play action</li>
<li>Fixed an issue where contain custom adjustments were not respected</li></ul>
<p>Update 1.013 September 30, 2026</p></body></html>"""
RSS = """<?xml version="1.0"?><rss><channel><title>news</title>
<item><title>College Football 27: the new goal line meta (TE flat)</title><link>https://x/1</link>
<pubDate>{d}</pubDate><description>College Football 27 red zone: TE flat and mesh from bunch</description></item>
<item><title>Some other game news</title><link>https://x/2</link><pubDate>{d}</pubDate></item>
</channel></rss>"""


def _fake_fetch_factory(calls: list[str], *, fail: bool = False, rss_title_suffix: str = ""):
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def fake(url: str, timeout: float):
        calls.append(url)
        src = ms.MetaSource(url=url, label=(ms._SOURCE_BY_URL.get(url) or {}).get("label", ""),
                            kind=(ms._SOURCE_BY_URL.get(url) or {}).get("kind", ""))
        if fail:
            src.error = "URLError: offline"
            return src, ""
        src.fetched, src.status = True, 200
        kind = src.kind
        if kind == "rss":
            body = RSS.format(d=now).replace("(TE flat)", "(TE flat)" + rss_title_suffix)
        elif kind == "patch":
            body = PATCH
        elif kind == "index":
            body = '<a href="/games/ea-sports-college-football/college-football-27/news/title-update-september-30th-2026">x</a>'
        else:
            body = GUIDE
        src.title = "t"
        return src, body

    return fake


class _EnvCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self._old = os.environ.get("CFB_COACH_DB")
        os.environ["CFB_COACH_DB"] = str(Path(self.td.name) / "coach.db")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("CFB_COACH_DB", None)
        else:
            os.environ["CFB_COACH_DB"] = self._old
        self.td.cleanup()


class TestMetaRefresh(_EnvCase):
    def test_live_fetch_builds_signals_headlines_and_cache(self):
        calls: list[str] = []
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls)):
            r = ms.run_meta_scout()
        self.assertEqual(r.mode, "live")
        self.assertTrue(r.available)
        self.assertGreaterEqual(len(calls), len(ms.TRUSTED_URLS))
        self.assertTrue(any("title-update-september-30th-2026" in c for c in calls))  # discovered
        self.assertIn("mesh", r.rz_signals)
        self.assertIn("spot_flat", r.rz_signals)
        self.assertTrue(any("goal line meta" in h["title"] for h in r.headlines))
        # Google News items must mention CFB 27 (r/NCAAFBseries posts are all series posts)
        self.assertFalse(any("other game" in h["title"] and "Google" in h["source"] for h in r.headlines))
        self.assertTrue(ms.cache_path().is_file())
        self.assertIn("First meta fetch", r.changes_since_last[0])

    def test_same_day_ttl_reuses_cache_and_refresh_forces(self):
        calls: list[str] = []
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls)):
            ms.run_meta_scout()
            n = len(calls)
            r2 = ms.run_meta_scout()
            self.assertEqual(len(calls), n)  # cache hit, no network
            self.assertEqual(r2.mode, "cache")
            r3 = ms.run_meta_scout(refresh=True)
            self.assertGreater(len(calls), n)
            self.assertEqual(r3.mode, "live")

    def test_cache_from_yesterday_is_stale(self):
        now = datetime.now(timezone.utc)
        self.assertTrue(ms._cache_fresh({"cached_at": (now - timedelta(minutes=5)).isoformat()}, now=now)
                        or now.astimezone().hour == 0)
        self.assertFalse(ms._cache_fresh({"cached_at": (now - timedelta(hours=25)).isoformat()}, now=now))
        self.assertFalse(ms._cache_fresh({"cached_at": (now - timedelta(hours=7)).isoformat()}, now=now))

    def test_changes_since_last(self):
        calls: list[str] = []
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls)):
            ms.run_meta_scout()
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls, rss_title_suffix=" v2")):
            r = ms.run_meta_scout(refresh=True)
        self.assertTrue(r.previous_fetched_at)
        self.assertTrue(any(c.startswith("New:") and "v2" in c for c in r.changes_since_last))

    def test_offline_falls_back_to_cache_then_seed(self):
        r = ms.run_meta_scout(offline=True)
        self.assertEqual(r.mode, "seed")
        self.assertTrue(r.concept_signals)  # seed research still yields priors
        calls: list[str] = []
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls)):
            ms.run_meta_scout()
        r = ms.run_meta_scout(offline=True)
        self.assertEqual(r.mode, "cache")
        self.assertTrue(r.offline)

    def test_network_failure_uses_last_good_cache(self):
        calls: list[str] = []
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls)):
            ms.run_meta_scout()
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls, fail=True)):
            r = ms.run_meta_scout(refresh=True)
        self.assertEqual(r.mode, "cache")
        self.assertIn("Live fetch failed", r.message)

    def test_network_failure_without_cache_uses_seed(self):
        calls: list[str] = []
        with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls, fail=True)):
            r = ms.run_meta_scout(refresh=True)
        self.assertEqual(r.mode, "seed")
        self.assertFalse(ms.cache_path().is_file() and json.loads(ms.cache_path().read_text()).get("result", {}).get("mode") == "live")


class TestMetaAlignment(unittest.TestCase):
    def test_seed_research_is_cited_and_in_formations(self):
        from cfb_coach.gameplan import load_baseline

        res = load_baseline()["meta_research"]
        self.assertTrue(res["updated"])
        for f in res["findings"]:
            self.assertTrue(f["url"].startswith("https://"))
            self.assertTrue(f.get("accessed"))
        allowed = {"Gun Bunch X Nasty", "Gun Cluster", "Singleback Deuce Close"}
        for p in res["priors"] + res["lab_candidates"]:
            self.assertIn(p["formation"], allowed)

    def test_live_boost_capped(self):
        b = live_boosts({"mesh": 50}, {"mesh": 50})
        for zones in b.values():
            self.assertTrue(all(v <= 0.12 + 1e-9 for v in zones.values()))

    def test_conflict_when_meta_likes_what_fails_for_you(self):
        from cfb_coach.learning import LearnedWeights

        lw = LearnedWeights(
            {"zone::rz::play::Gun Bunch X Nasty::RZ PA X Whip": -2.4, "zone::gl::play::Gun Bunch X Nasty::RZ PA X Whip": -2.4,
             "play::Gun Bunch X Nasty::RZ PA X Whip": -2.0},
            {"zone::rz::play::Gun Bunch X Nasty::RZ PA X Whip": {"n": 12, "n_obs": 20.0, "mean": -1.5},
             "zone::gl::play::Gun Bunch X Nasty::RZ PA X Whip": {"n": 8, "n_obs": 15.0, "mean": -1.5}},
        )
        al = alignment(lw, MetaPriors.build(None), dynasty="ohio_state")
        msgs = [c for c in al["conflicts"] if c["play"] == "RZ PA X Whip"]
        self.assertTrue(msgs)
        self.assertIn("keeps failing for you", msgs[0]["message"])
        self.assertTrue(al["lab_candidates"])  # Ohio State lab only
        al2 = alignment(lw, MetaPriors.build(None), dynasty="alabama")
        self.assertFalse(al2["lab_candidates"])


if __name__ == "__main__":
    unittest.main()


class TestPrepPageRenders(_EnvCase):
    def test_prep_page_shows_freshness_sources_zone_plan_conflicts(self):
        from cfb_coach import learning as L
        from cfb_coach.db import CoachDB
        from cfb_coach.install_sheet import build_prep_plan
        from cfb_coach.prep_browser import render_prep_html

        db = CoachDB(Path(os.environ["CFB_COACH_DB"]))
        try:
            for i in range(6):
                db.log_snap(opponent_id="cpu", side="offense", situation_raw="1&goal opp 5", our_call="x",
                            formation="Gun Bunch X Nasty", play="RZ PA X Whip", down=1, distance=5,
                            yardline=95, result="int" if i % 2 else "incomplete", session_id="g")
            L.rebuild_all(db)
            calls: list[str] = []
            with mock.patch.object(ms, "_fetch_one", _fake_fetch_factory(calls)):
                plan = build_prep_plan("cpu", {}, db=db, persist=False, dynasty="ohio_state", refresh_meta=True)
            html = render_prep_html(plan)
        finally:
            db.close()
        self.assertIn("Current meta — freshness", html)
        self.assertIn("LIVE — fetched this prep", html)
        self.assertIn("MaddenTurf", html)
        self.assertIn("Zone plan", html)
        self.assertIn("keeps failing for you", html)
        self.assertIn("Ohio State lab candidates", html)
        za = plan["zone_alignment"]
        self.assertLessEqual(len(plan["inventory"].get("macros_active") or []), 8)
        self.assertTrue(za["zone_plan"]["gl"])
