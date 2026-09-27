"""v1.13: YouTube discovery + transcript parsing (fixtures, no network), named-entity
extraction, graceful degradation, transcript cache."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cfb_coach import yt_research as yt
from cfb_coach.meta_entities import aggregate, extract_doc, recency_weight

FIX = Path(__file__).parent / "fixtures" / "youtube"
NOW = datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


class TestParsing(unittest.TestCase):
    def test_search_page_and_filter(self):
        vids = yt.parse_search_html(_read("search.html"), now=NOW, query="q")
        self.assertEqual([v.video_id for v in vids], ["AAAAAAAAAA1", "AAAAAAAAAA2", "AAAAAAAAAA3"])
        self.assertEqual(vids[0].channel, "Test Creator")
        self.assertEqual(vids[0].published, "2026-09-22")
        keep = [v.video_id for v in vids if yt.classify_video(v, now=NOW)[0]]
        self.assertEqual(keep, ["AAAAAAAAAA1"])  # CFB 26 and Madden filtered out
        self.assertEqual(yt.classify_video(vids[2], now=NOW)[1], "older title / Madden")

    def test_channel_feed(self):
        vids = yt.parse_channel_feed(_read("feed.xml"), channel="Feed Creator")
        self.assertEqual(len(vids), 2)
        self.assertEqual(vids[0].published, "2026-09-20")
        self.assertTrue(vids[0].published_exact)
        self.assertIn("&", vids[0].title)
        self.assertTrue(yt.classify_video(vids[0], now=NOW)[0])
        self.assertFalse(yt.classify_video(vids[1], now=NOW)[0])

    def test_transcript_formats(self):
        t = yt.parse_json3(_read("transcript.json3"))
        self.assertIn("gun bunch x nasty we run mesh spot", t.lower())
        self.assertIn("y flat goalline", t.lower())
        x = yt.parse_timedtext_xml(_read("transcript.xml"))
        self.assertIn("Z Spot Shake", x)
        self.assertEqual(yt.parse_json3("not json"), "")
        tr = yt.pick_caption_track([{"languageCode": "es"}, {"languageCode": "en", "kind": "asr", "baseUrl": "a"},
                                    {"languageCode": "en", "baseUrl": "m"}])
        self.assertEqual(tr["baseUrl"], "m")  # manual English preferred over ASR

    def test_relative_dates(self):
        self.assertEqual(yt.parse_relative_date("Streamed 2 weeks ago", NOW), "2026-09-13")
        self.assertEqual(yt.parse_relative_date("21 hours ago", NOW), "2026-09-26")


class TestEntities(unittest.TestCase):
    def test_extracts_formation_attributed_plays(self):
        e = extract_doc(yt.parse_json3(_read("transcript.json3")))
        self.assertIn("Gun Bunch X Nasty::Mesh Spot", e["pairs"])
        self.assertIn("Gun Bunch X Nasty::Y Flat GoalLine", e["pairs"])
        self.assertIn("Singleback Deuce Close::HB Dive", e["pairs"])
        # a generic play with no formation nearby isn't attributed
        self.assertEqual(extract_doc("just run inside zone all day")["pairs"], {})

    def test_aggregate_recency_and_sources(self):
        docs = [
            {"label": "YT A", "kind": "youtube_transcript", "url": "a", "date": "2026-09-25",
             "text": "gun bunch x nasty mesh spot. bunch x nasty mesh spot again"},
            {"label": "Old blog", "kind": "guide", "url": "b", "date": "2026-07-10", "text": "Gun Bunch X Nasty Mesh Spot"},
        ]
        agg = aggregate(docs, now=NOW)
        p = agg["pairs"]["Gun Bunch X Nasty::Mesh Spot"]
        self.assertEqual(p["docs"], 2)
        self.assertEqual(len(p["sources"]), 2)
        self.assertGreater(recency_weight("2026-09-25", NOW), recency_weight("2026-07-10", NOW))


class TestOrchestration(unittest.TestCase):
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

    @staticmethod
    def _fetch(url, timeout):
        if "results?search_query" in url:
            return 200, _read("search.html")
        if "feeds/videos.xml" in url:
            return 200, _read("feed.xml")
        return 404, ""

    def test_transcripts_fetched_cached_and_used(self):
        calls = []

        def ok(vid, to):
            calls.append(vid)
            return yt.parse_json3(_read("transcript.json3"))

        r = yt.run_youtube_research(now=NOW, budget_s=5, fetch=self._fetch, transcript_fetchers=[("fake", ok)])
        self.assertEqual(r.found, 2)  # 1 search + 1 RSS video survive the CFB 27 filter
        self.assertEqual(r.transcripts, 2)
        self.assertEqual(r.fetched_now, 2)
        self.assertTrue(any(d["kind"] == "youtube_transcript" and "mesh spot" in d["text"].lower() for d in r.docs))
        self.assertNotIn("text", r.to_dict()["docs"][0])  # persisted copy stays small
        r2 = yt.run_youtube_research(now=NOW, budget_s=5, fetch=self._fetch, transcript_fetchers=[("fake", ok)])
        self.assertEqual(len(calls), 2)  # second run served from the per-video cache
        self.assertEqual(r2.from_cache, 2)

    def test_blocked_degrades_gracefully(self):
        def blocked(vid, to):
            raise yt.Blocked("HTTP 429")

        r = yt.run_youtube_research(now=NOW, budget_s=5, fetch=self._fetch, transcript_fetchers=[("fake", blocked)])
        self.assertEqual(r.found, 2)
        self.assertEqual(r.transcripts, 0)
        self.assertTrue(r.blocked)
        self.assertTrue(any("blocked" in n.lower() for n in r.notes))
        self.assertTrue(r.docs)  # titles/descriptions still count as (weaker) signals

    def test_network_down_and_offline(self):
        r = yt.run_youtube_research(now=NOW, budget_s=3, fetch=lambda url, timeout: (0, ""), transcript_fetchers=[])
        self.assertEqual(r.found, 0)
        self.assertEqual(r.search_ok + r.rss_ok, 0)
        self.assertTrue(yt.run_youtube_research(offline=True).notes)


class TestScoutIntegration(unittest.TestCase):
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

    def test_youtube_transcripts_feed_named_signals_and_web_failure_still_live(self):
        from unittest import mock

        from cfb_coach import meta_scout as ms

        def fake_fetch_one(url, timeout):
            src = ms.MetaSource(url=url, label="x", kind=(ms._SOURCE_BY_URL.get(url) or {}).get("kind", ""))
            src.error = "offline"
            return src, ""

        def runner(**kw):
            return yt.run_youtube_research(now=NOW, budget_s=4, fetch=TestOrchestration._fetch,
                                           transcript_fetchers=[("fake", lambda v, t: yt.parse_json3(_read("transcript.json3")))])

        with mock.patch.object(ms, "_fetch_one", fake_fetch_one):
            r = ms.run_meta_scout(yt_runner=runner)
        self.assertEqual(r.mode, "live")
        self.assertEqual(r.youtube["transcripts"], 2)
        self.assertIn("Gun Bunch X Nasty::Y Flat GoalLine", r.named_signals["pairs"])
        srcs = r.named_signals["pairs"]["Gun Bunch X Nasty::Y Flat GoalLine"]["sources"]
        self.assertTrue(any("youtube.com/watch" in s["url"] for s in srcs))


if __name__ == "__main__":
    unittest.main()
