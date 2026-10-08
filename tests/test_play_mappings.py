"""Play mappings turn a raw VOD name into a formation the book can call.

Fixtures and temp databases only. The models directory is whatever
``CFB_COACH_VOD_MODELS`` points at for that test.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden.prep import build_prep_plan, format_delta_text
from cfb_coach.vod_model.book import AUDIT_KEY, read_audit
from cfb_coach.vod_model.mappings import load_mappings, lookup
from tests.test_madden27 import _Isolated
from tests.test_vod_model import FIXTURE, _calls, _cover3, _forms, _write_model

MAPPED = "Mtnbunchcrossdig"
MAPPED_PLAY = "Mtn Bunch Cross Dig"
MAPPED_FORM = "Gun Tight Flex Wk"


def _records(*raw_names: str) -> list[dict]:
    payload = json.loads((FIXTURE / "play_mappings" / "v1.json").read_text(encoding="utf-8"))
    wanted = set(raw_names)
    return [dict(row) for row in payload["mappings"] if row["raw_name"] in wanted]


def _install(root: Path, records: list[dict] | None, *, latest: str | None = None, body: str | None = None) -> None:
    folder = root / "play_mappings"
    folder.mkdir(parents=True, exist_ok=True)
    if latest is not None:
        (folder / "LATEST.json").write_text(latest, encoding="utf-8")
    else:
        (folder / "LATEST.json").write_text(json.dumps({"version": "v1", "file": "v1.json"}), encoding="utf-8")
    if body is not None:
        (folder / "v1.json").write_text(body, encoding="utf-8")
    elif records is not None:
        (folder / "v1.json").write_text(json.dumps({
            "schema": "play_mappings.v1", "version": "v1", "mappings": records,
        }), encoding="utf-8")


def _model(root: Path, play: str, *, n: int = 20, n_vods: int = 4, tentative: bool = False) -> None:
    _write_model(
        root, game="madden27", opponent_type="human", look="cover_3", play=play,
        n=n, n_vods=n_vods, tentative=tentative,
    )


class TestMappingLoader(_Isolated):
    def test_real_v1_keeps_one_formation_at_high_or_med(self) -> None:
        index = load_mappings(FIXTURE)
        assert index is not None
        self.assertEqual(index.version_label, "play_mappings v1")
        hit = lookup(index, "madden27", "Mtnbunchcrossdig")
        assert hit is not None
        self.assertEqual(hit.play_name, "Mtn Bunch Cross Dig")
        self.assertEqual(hit.formation, "Gun Tight Flex Wk")
        self.assertEqual(hit.playbook_name, "Raiders")
        self.assertGreaterEqual(len(hit.plays), 2)
        self.assertIn("Mtn Bunch Cross Dig", hit.plays)
        sailed = lookup(index, "madden27", "Sail Dig")
        assert sailed is not None
        same = lookup(index, "madden27", "Saildig")
        assert same is not None
        self.assertEqual(sailed.formation, same.formation)
        self.assertEqual(sailed.play_name, "Sail Dig")
        verticals = lookup(index, "Madden27", "Four Verticals")
        assert verticals is not None
        self.assertEqual(verticals.confidence, "med")
        self.assertEqual(verticals.formation, "Gun Doubles HB Wk")
        for ignored in ("Zone WK", "HB Dive", "HB Power O", "Inside Zone", "Dagger", "Curl Combo"):
            self.assertIsNone(lookup(index, "madden27", ignored), ignored)
        self.assertEqual(
            hit.display("Mtnbunchcrossdig"),
            "Mtnbunchcrossdig → Raiders / Gun Tight Flex Wk / Mtn Bunch Cross Dig",
        )

    def test_missing_corrupt_and_escaped_pointers_are_noop(self) -> None:
        self.assertIsNone(load_mappings(Path("/tmp/does-not-exist-vod-mappings")))
        root = self.dir / "models"
        root.mkdir()
        self.assertIsNone(load_mappings(root))
        _install(root, None, latest="{")
        self.assertIsNone(load_mappings(root))
        _install(root, None, latest=json.dumps({"file": "missing.json"}))
        self.assertIsNone(load_mappings(root))
        _install(root, _records(MAPPED), body="{")
        self.assertIsNone(load_mappings(root))
        outside = self.dir / "elsewhere.json"
        outside.write_text(json.dumps({"version": "v9", "mappings": []}), encoding="utf-8")
        _install(root, None, latest=json.dumps({"file": str(outside)}))
        self.assertIsNone(load_mappings(root))
        _install(root, None, latest=json.dumps({"file": "../v1.json"}))
        self.assertIsNone(load_mappings(root))
        _install(root, _records(MAPPED), latest=json.dumps({"version": "v1"}))
        loaded = load_mappings(root)
        assert loaded is not None
        self.assertEqual(loaded.version_label, "play_mappings v1")
        self.assertIsNotNone(lookup(loaded, "madden27", MAPPED))


class TestMappingEdits(_Isolated):
    def setUp(self) -> None:
        super().setUp()
        os.environ.pop("CFB_COACH_VOD_MODELS", None)
        os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
        os.environ.pop("CFB_COACH_VOD_FREEZE_BOOK", None)

    def tearDown(self) -> None:
        os.environ.pop("CFB_COACH_VOD_MODELS", None)
        os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
        os.environ.pop("CFB_COACH_VOD_FREEZE_BOOK", None)
        super().tearDown()

    def _prep(self, models: Path, *, prior_off: bool = False, freeze: bool = False) -> dict:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        env = {
            "HOME": td.name,
            "CFB_COACH_DB": str(Path(td.name) / "coach.db"),
            "CFB_COACH_VOD_MODELS": str(models),
            "CFB_COACH_NO_VOD_PRIOR": "1" if prior_off else "",
            "CFB_COACH_VOD_FREEZE_BOOK": "1" if freeze else "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            for key in ("CFB_COACH_MADDEN_DB", "CFB_COACH_MADDEN_PRIMARY_TEAM", "CFB_COACH_MADDEN_LAB_TEAM"):
                os.environ.pop(key, None)
            if not prior_off:
                os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
            if not freeze:
                os.environ.pop("CFB_COACH_VOD_FREEZE_BOOK", None)
            from cfb_coach.games import madden_db_path
            from cfb_coach.madden import data as mdata

            db = CoachDB(madden_db_path(), seed=mdata.load_seed())
            try:
                _cover3(db, "james")
                plan = build_prep_plan("james", db=db, offline=True, persist=True)
                text = format_delta_text(plan)
                vod = text[text.find("## VOD beaters"):] if "## VOD beaters" in text else ""
                return {
                    "forms": _forms(db),
                    "calls": _calls(db, "james", ["1&10", "3&8", "1&10 cover 3"]),
                    "text": text,
                    "vod": vod,
                    "plan": plan,
                    "audit": read_audit(db),
                    "meta_audit": db.get_meta(AUDIT_KEY),
                }
            finally:
                db.close()

    def test_ocr_med_cell_adds_the_mapped_formation(self) -> None:
        root = self.dir / "ocr"
        _model(root, MAPPED)
        _install(root, _records(MAPPED))
        snap = self._prep(root)
        forms = snap["forms"]
        self.assertIn(MAPPED_FORM, forms)
        self.assertIn(MAPPED_PLAY, forms[MAPPED_FORM])
        evidence = _records(MAPPED)[0]["formation_plays"][MAPPED_FORM]
        self.assertEqual(forms[MAPPED_FORM], evidence)
        change = snap["plan"]["vod_report"]["changes"][0]
        self.assertEqual(change["mapping_version"], "play_mappings v1")
        self.assertEqual(change["mapped"], f"{MAPPED} → Raiders / {MAPPED_FORM} / {MAPPED_PLAY}")
        self.assertEqual(change["tier"], "med")
        self.assertEqual(change["action"], "swap_formation")
        self.assertTrue(change.get("replaced"))
        self.assertNotIn(change["replaced"], forms)
        self.assertEqual(len(forms), 5)
        self.assertIn("play_mappings v1", snap["text"])
        self.assertIn(change["mapped"], snap["text"])
        self.assertEqual(snap["audit"][-1]["mapping_version"], "play_mappings v1")
        self.assertEqual(snap["audit"][-1]["mapped"], change["mapped"])

    def test_uncatalogued_name_adds_the_mapped_formation_not_the_resolver_default(self) -> None:
        bare = self.dir / "verticals-bare"
        mapped = self.dir / "verticals-mapped"
        _model(bare, "Four Verticals")
        _model(mapped, "Four Verticals")
        _install(mapped, _records("Four Verticals"))
        without = self._prep(bare)
        with_map = self._prep(mapped)
        evidence = _records("Four Verticals")[0]["formation_plays"]["Gun Doubles HB Wk"]
        self.assertIn("Gun Doubles HB Wk", with_map["forms"])
        self.assertEqual(with_map["forms"]["Gun Doubles HB Wk"], evidence)
        self.assertNotEqual(without["forms"].get("Gun Doubles HB Wk"), evidence)
        self.assertNotIn("Gun Doubles HB Wk", without["forms"])
        self.assertEqual(with_map["audit"][-1]["source_book"], "Bills")
        self.assertIn("Four Verticals → Bills / Gun Doubles HB Wk / Four Verticals", with_map["text"])
        self.assertNotIn("→ Bills", without["vod"])

    def test_low_and_ambiguous_change_nothing(self) -> None:
        bare = self.dir / "ignore-bare"
        low = self.dir / "ignore-low"
        ambiguous = self.dir / "ignore-ambiguous"
        _model(bare, MAPPED)
        _model(low, MAPPED)
        _model(ambiguous, MAPPED)
        low_rows = _records(MAPPED)
        low_rows[0]["confidence"] = "low"
        amb_rows = _records(MAPPED)
        amb_rows[0]["status"] = "resolved_book_formation_ambiguous"
        amb_rows[0]["confidence"] = "med"
        amb_rows[0]["formation"] = None
        amb_rows[0]["formation_candidates"] = ["Gun Tight Flex Wk", "Gun Trips", "Gun Bunch"]
        _install(low, low_rows)
        _install(ambiguous, amb_rows)
        base = self._prep(bare)
        low_snap = self._prep(low)
        amb_snap = self._prep(ambiguous)
        self.assertNotIn(MAPPED_FORM, base["forms"])
        self.assertEqual(low_snap["forms"], base["forms"])
        self.assertEqual(amb_snap["forms"], base["forms"])
        self.assertEqual(low_snap["calls"], base["calls"])
        self.assertEqual(amb_snap["calls"], base["calls"])
        self.assertEqual(base["plan"]["vod_report"]["changes"], [])
        self.assertEqual(low_snap["plan"]["vod_report"]["changes"], [])
        self.assertEqual(amb_snap["plan"]["vod_report"]["changes"], [])
        self.assertNotIn("→", low_snap["vod"])
        self.assertNotIn("→", amb_snap["vod"])
        self.assertNotIn(MAPPED_PLAY, low_snap["vod"])

    def test_missing_or_corrupt_mappings_match_no_mapping(self) -> None:
        bare = self.dir / "same-bare"
        corrupt = self.dir / "same-corrupt"
        missing_pointer = self.dir / "same-missing"
        _model(bare, MAPPED)
        _model(corrupt, MAPPED)
        _model(missing_pointer, MAPPED)
        _install(corrupt, None, latest="{")
        _install(missing_pointer, None, latest=json.dumps({"file": "/workspace/vod_data/models/play_mappings/v1.json"}))
        base = self._prep(bare)
        bad = self._prep(corrupt)
        pointed = self._prep(missing_pointer)
        self.assertEqual(bad["forms"], base["forms"])
        self.assertEqual(pointed["forms"], base["forms"])
        self.assertEqual(bad["calls"], base["calls"])
        self.assertEqual(pointed["calls"], base["calls"])
        self.assertEqual(bad["vod"], base["vod"])
        self.assertEqual(pointed["vod"], base["vod"])
        self.assertNotIn("play_mappings", base["vod"])
        self.assertNotIn(MAPPED_FORM, base["forms"])

        catalogued = self.dir / "catalogued"
        catalogued_bad = self.dir / "catalogued-bad"
        _model(catalogued, "Cross Drag")
        _model(catalogued_bad, "Cross Drag")
        _install(catalogued_bad, None, latest="{")
        plain = self._prep(catalogued)
        broken = self._prep(catalogued_bad)
        self.assertEqual(broken["forms"], plain["forms"])
        self.assertEqual(broken["calls"], plain["calls"])
        self.assertEqual(broken["vod"], plain["vod"])
        self.assertTrue(plain["plan"]["vod_report"]["changes"])

    def test_stub_play_list_stays_display_only(self) -> None:
        root = self.dir / "stub"
        _model(root, MAPPED)
        rows = _records(MAPPED)
        rows[0]["formation_plays"] = {MAPPED_FORM: [MAPPED_PLAY]}
        _install(root, rows)
        snap = self._prep(root)
        self.assertNotIn(MAPPED_FORM, snap["forms"])
        self.assertEqual(snap["plan"]["vod_report"]["changes"], [])
        self.assertIn(f"{MAPPED} → Raiders / {MAPPED_FORM} / {MAPPED_PLAY}", snap["vod"])

    def test_no_vod_prior_disables_mappings(self) -> None:
        root = self.dir / "flag"
        _model(root, MAPPED)
        _install(root, _records(MAPPED))
        os.environ["CFB_COACH_VOD_MODELS"] = str(root)
        db = self.madden_db()
        try:
            _cover3(db, "james")
        finally:
            db.close()
        rc, out = self.run_cli([
            "prep", "-o", "james", "--game", "madden27", "--text", "--offline", "--no-vod-prior",
        ])
        self.assertEqual(rc, 0)
        self.assertNotIn("## VOD beaters", out)
        self.assertNotIn("play_mappings", out)
        self.assertNotIn(MAPPED_PLAY, out)
        db = self.madden_db()
        try:
            self.assertNotIn(MAPPED_FORM, _forms(db))
            self.assertIsNone(db.get_meta(AUDIT_KEY))
        finally:
            db.close()
        rc, out = self.run_cli([
            "prep", "-o", "james", "--game", "madden27", "--text", "--offline",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("play_mappings v1", out)
        self.assertIn(f"{MAPPED} → Raiders / {MAPPED_FORM} / {MAPPED_PLAY}", out)
        db = self.madden_db()
        try:
            self.assertIn(MAPPED_PLAY, _forms(db).get(MAPPED_FORM) or [])
        finally:
            db.close()

    def test_freeze_and_sample_bar_still_block_the_edit(self) -> None:
        frozen_root = self.dir / "frozen"
        thin_root = self.dir / "thin"
        _model(frozen_root, MAPPED)
        _model(thin_root, MAPPED, n=10, n_vods=2, tentative=True)
        _install(frozen_root, _records(MAPPED))
        _install(thin_root, _records(MAPPED))
        frozen = self._prep(frozen_root, freeze=True)
        thin = self._prep(thin_root)
        self.assertNotIn(MAPPED_FORM, frozen["forms"])
        self.assertNotIn(MAPPED_FORM, thin["forms"])
        self.assertEqual(frozen["plan"]["vod_report"]["changes"], [])
        self.assertTrue(frozen["plan"]["vod_report"]["frozen"])
        self.assertEqual(thin["plan"]["vod_report"]["changes"], [])
        self.assertIsNone(frozen["meta_audit"])
        self.assertIsNone(thin["meta_audit"])
        self.assertIn("Book frozen", frozen["text"])
        self.assertIn("play_mappings v1", frozen["text"])

    def test_alternate_spelling_resolves_without_editing_a_frozen_book(self) -> None:
        root = self.dir / "spell"
        _model(root, "Meshocr")
        _install(root, [{
            "game": "madden27",
            "raw_name": "Meshocr",
            "status": "resolved",
            "confidence": "high",
            "playbook": "buccaneers-off",
            "playbook_name": "Buccaneers",
            "playbook_in_coach_catalog": "Buccaneers",
            "formation": "Gun 5WR Tight",
            "formation_candidates": ["Gun 5WR Tight"],
            "play_name": "Mesh",
        }])
        hit = lookup(load_mappings(root), "madden27", "Meshocr")
        assert hit is not None
        self.assertEqual((hit.formation, hit.play_name), ("Gun 5WR Tight", "Mesh"))
        snap = self._prep(root, freeze=True)
        self.assertIn("Mesh", snap["forms"].get("Gun 5WR Tight") or [])
        self.assertEqual(snap["audit"], [])
        self.assertEqual(snap["plan"]["vod_report"]["changes"], [])
