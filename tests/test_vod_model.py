"""VOD model prior: v0.7 is display-only, and a synthetic med cell may edit the book.

Live play treats that edit as already installed. The next snap's candidates and
macro pairs include the swapped-in formation and omit the one that was removed.
"""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from pathlib import Path
from unittest import mock

from cfb_coach.cli import build_parser
from cfb_coach.db import CoachDB
from cfb_coach.madden.playbook import load_books
from cfb_coach.madden.playcaller import MaddenCall, make_call
from cfb_coach.madden.prep import build_prep_plan, format_delta_text
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.vod_model.adapter import VodCell
from cfb_coach.vod_model.book import AUDIT_KEY, read_audit, revert_last
from cfb_coach.vod_model.loader import load_model
from cfb_coach.vod_model.live import book_for_next_snap, macro_candidates, play_candidates
from cfb_coach.vod_model.prior import cell_tier, influence, select_cell
from tests.test_madden27 import _Isolated

FIXTURE = Path(__file__).parent / "fixtures" / "vod_model"
CPU_SNAPS = Path(__file__).parent / "fixtures" / "cpu_19-0_offense_snaps.csv"
JAMES_SITS = ("1&10", "3&8", "1&10 cover 3", "2&6 cover 2", "1&goal opp 8")


def _forms(db: CoachDB) -> dict[str, list[str]]:
    rec = (load_books(db).get("offense") or {})
    return {f: list(ps) for f, ps in (rec.get("formations") or {}).items()}


def _calls(db: CoachDB, oid: str, raws: list[str]) -> list[tuple[str, str, str | None]]:
    out = []
    for i, raw in enumerate(raws):
        sit = parse_madden_situation(raw)
        call = make_call(sit, oid, db, rng=random.Random(i))
        out.append((call.formation, call.play, call.macro))
    return out


def _write_model(root: Path, *, game: str, opponent_type: str, look: str, play: str,
                 n: int = 20, n_vods: int = 4, tentative: bool = False,
                 shrunk: float = 0.82, lower: float = 0.61, base: float = 0.45,
                 version: str = "v9.9", omit_shrunk: bool = False) -> None:
    version_dir = root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    cell = {
        "call": play,
        "call_key": play.lower(),
        "call_class": "run",
        "n": n,
        "successes": int(round(shrunk * n)),
        "raw_success": shrunk,
        "lower_bound_90": lower,
        "n_vods": n_vods,
        "tentative": tentative,
        "confidence": "tentative" if tentative else "ok",
    }
    if not omit_shrunk:
        cell["shrunk_success"] = shrunk
    payload = {
        "label_version": version,
        "tentative_n": 15,
        "strata": {
            f"{game}/{opponent_type}": {
                "stratum_base_rate_all_known": base,
                "views": {
                    "all_clean_pairs": {
                        "stratum_clean_success_rate": base,
                        "look_family": {"looks": {look: {"look": look, "n": n + 5, "calls": [cell]}}},
                        "look": {"looks": {}},
                    }
                },
            }
        },
    }
    (version_dir / "beaters.json").write_text(json.dumps(payload), encoding="utf-8")
    (version_dir / "model_params.json").write_text("{}", encoding="utf-8")
    (root / "LAST_TRAINED.json").write_text(json.dumps({
        "label_version": version,
        "model_dir": "/workspace/vod_data/models/" + version,
    }), encoding="utf-8")


def _cover3(db: CoachDB, oid: str, n: int = 4) -> None:
    for _ in range(n):
        db.log_snap(
            opponent_id=oid, side="offense", situation_raw="1&10 cover 3", our_call="HB Dive",
            down=1, distance=10, yardline=25, result="+4", coverage_seen="Cover 3",
        )


def _cell(**overrides: object) -> VodCell:
    base = dict(
        game="madden27", opponent_type="human", look="cover_3", look_grain="family",
        call="Mesh", call_key="mesh", call_class="pass", n=32, n_vods=5, successes=26,
        raw_success=0.8, shrunk_success=0.8, lower_bound=0.7, baseline=0.45, lift=0.35,
        tentative=False, confidence="ok", model_version="vtest",
    )
    base.update(overrides)
    return VodCell(**base)  # type: ignore[arg-type]


class TestVodLoaderAndPrior(_Isolated):
    def test_v07_fixture_loads_without_using_model_dir(self) -> None:
        model = load_model(FIXTURE)
        assert model is not None
        self.assertEqual(model.version, "v0.7")
        last = json.loads((FIXTURE / "LAST_TRAINED.json").read_text(encoding="utf-8"))
        self.assertTrue(str(last["model_dir"]).startswith("/workspace/vod_data"))
        self.assertFalse(Path(last["model_dir"]).is_dir())
        cells = [c for key, rows in model.cells.items() if key[2] == "family" for c in rows]
        self.assertTrue(cells)
        self.assertTrue(all(c.n < 15 for c in cells))
        self.assertTrue(all(cell_tier(c, tentative_n=model.tentative_n) == "none" for c in cells))
        self.assertNotIn(("cfb27", "unknown"), {(k[0], k[1]) for k in model.cells})

    def test_missing_and_corrupt_are_noop(self) -> None:
        self.assertIsNone(load_model(Path("/tmp/does-not-exist-vod")))
        root = self.dir / "bad"
        root.mkdir()
        (root / "v0.1").mkdir()
        (root / "v0.1" / "beaters.json").write_text("{", encoding="utf-8")
        (root / "LAST_TRAINED.json").write_text(json.dumps({"label_version": "v0.1"}), encoding="utf-8")
        self.assertIsNone(load_model(root))

    def test_corrupt_params_and_version_fallback(self) -> None:
        root = self.dir / "models"
        _write_model(root, game="madden27", opponent_type="human", look="cover_3", play="Mesh", version="v0.2")
        (root / "v0.2" / "model_params.json").write_text("{", encoding="utf-8")
        (root / "LAST_TRAINED.json").write_text(json.dumps({
            "label_version": "v0.2", "model_dir": "/workspace/vod_data/models/v0.2",
        }), encoding="utf-8")
        self.assertIsNone(load_model(root))
        _write_model(root, game="madden27", opponent_type="human", look="cover_3", play="Mesh", version="v0.9")
        (root / "LAST_TRAINED.json").write_text(json.dumps({
            "label_version": "v9.9", "model_dir": "/workspace/vod_data/models/v9.9",
        }), encoding="utf-8")
        loaded = load_model(root)
        assert loaded is not None
        self.assertEqual(loaded.version, "v0.9")

    def test_shrinkage_and_tier_weight_grows(self) -> None:
        root = self.dir / "shrink"
        _write_model(
            root, game="madden27", opponent_type="human", look="cover_3", play="Mesh",
            n=10, n_vods=2, omit_shrunk=True, shrunk=0.8, lower=0.4, base=0.5,
        )
        model = load_model(root)
        assert model is not None
        cell = model.cells_for("madden27", "human", "family", "cover_3")[0]
        # (successes + 5 * 0.5) / (10 + 5). successes was round(0.8 * 10) = 8.
        self.assertAlmostEqual(cell.shrunk_success, (8 + 5 * 0.5) / 15, places=4)
        self.assertEqual(cell_tier(cell, tentative_n=15), "low")
        self.assertEqual(influence("none"), 0.0)
        self.assertEqual(influence("low"), 0.0)
        self.assertEqual(influence("med"), 0.65)
        self.assertEqual(influence("high"), 1.0)
        low = _cell(n=10, n_vods=2, tentative=True, confidence="tentative", shrunk_success=0.9, lower_bound=0.8)
        self.assertEqual(cell_tier(low, tentative_n=15), "low")
        med = _cell(n=20, n_vods=4, tentative=False, confidence="ok", shrunk_success=0.8, lower_bound=0.6, baseline=0.45)
        self.assertEqual(cell_tier(med, tentative_n=15), "med")
        high = _cell(n=32, n_vods=5, lower_bound=0.7, baseline=0.45)
        self.assertEqual(cell_tier(high, tentative_n=15), "high")

    def test_logs_nudge_a_near_tie_and_do_not_override(self) -> None:
        # Both cells clear the high bar. The gap (0.40) is wider than the nudge,
        # so logs that favor B cannot pull it into the band.
        strong = _cell(call="A", shrunk_success=0.90, lower_bound=0.70, baseline=0.45)
        weak = _cell(call="B", shrunk_success=0.50, lower_bound=0.52, baseline=0.45)
        picked = select_cell([strong, weak], lambda c: -1.0 if c.call == "A" else 1.0, tentative_n=15)
        assert picked is not None
        self.assertEqual(picked.call, "A")
        near_a = _cell(call="A", shrunk_success=0.60, lower_bound=0.55, baseline=0.45)
        near_b = _cell(call="B", shrunk_success=0.55, lower_bound=0.52, baseline=0.45)
        nudged = select_cell([near_a, near_b], lambda c: -1.0 if c.call == "A" else 1.0, tentative_n=15)
        assert nudged is not None
        self.assertEqual(nudged.call, "B")

    def test_flags_parse_and_no_macros_stays(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["play", "-o", "james", "--game", "madden27", "--no-macros", "--no-vod-prior", "--freeze-vod-book"])
        self.assertTrue(args.no_macros)
        self.assertTrue(args.no_vod_prior)
        self.assertTrue(args.freeze_vod_book)


class TestV07Replay(_Isolated):
    def _prep_and_calls(self, oid: str, raws: list[str], *, models: bool, prior_off: bool) -> tuple[dict, list, str]:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        env = {
            "HOME": td.name,
            "CFB_COACH_DB": str(Path(td.name) / "coach.db"),
            "CFB_COACH_NO_VOD_PRIOR": "1" if prior_off else "",
            "CFB_COACH_VOD_FREEZE_BOOK": "",
            "CFB_COACH_VOD_MODELS": str(FIXTURE) if models else "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            for key in ("CFB_COACH_MADDEN_DB", "CFB_COACH_MADDEN_PRIMARY_TEAM", "CFB_COACH_MADDEN_LAB_TEAM"):
                os.environ.pop(key, None)
            if not models:
                os.environ.pop("CFB_COACH_VOD_MODELS", None)
            if not prior_off:
                os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
            os.environ.pop("CFB_COACH_VOD_FREEZE_BOOK", None)
            from cfb_coach.games import madden_db_path
            from cfb_coach.madden import data as mdata

            db = CoachDB(madden_db_path(), seed=mdata.load_seed())
            try:
                plan = build_prep_plan(oid, db=db, offline=True, persist=True)
                return _forms(db), _calls(db, oid, raws), format_delta_text(plan)
            finally:
                db.close()

    def test_james_and_cpu_match_baseline(self) -> None:
        cpu_raws = [row["situation_raw"] for row in csv.DictReader(CPU_SNAPS.open(encoding="utf-8"))]
        self.assertEqual(len(cpu_raws), 35)
        for oid, raws in (("james", list(JAMES_SITS)), ("cpu", cpu_raws)):
            base_forms, base_calls, base_text = self._prep_and_calls(oid, raws, models=False, prior_off=False)
            on_forms, on_calls, on_text = self._prep_and_calls(oid, raws, models=True, prior_off=False)
            off_forms, off_calls, off_text = self._prep_and_calls(oid, raws, models=True, prior_off=True)
            self.assertEqual(on_forms, base_forms, oid)
            self.assertEqual(on_calls, base_calls, oid)
            self.assertEqual(off_calls, base_calls, oid)
            self.assertEqual(off_forms, base_forms, oid)
            self.assertIn("## VOD beaters", on_text)
            self.assertIn("## Playbook changes", on_text)
            self.assertIn("No playbook changes", on_text)
            self.assertNotIn("## VOD beaters", base_text)
            self.assertNotIn("## VOD beaters", off_text)
            self.assertNotIn("CALL:", on_text)


class TestBookChangeGate(_Isolated):
    def setUp(self) -> None:
        super().setUp()
        os.environ.pop("CFB_COACH_VOD_MODELS", None)
        os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
        os.environ.pop("CFB_COACH_VOD_FREEZE_BOOK", None)
        self.models = self.dir / "models"

    def _target_play(self, forms: dict[str, list[str]]) -> str:
        from cfb_coach.madden.catalog import book_formations

        stock = book_formations("offense", "Buccaneers")["Singleback Bunch TE"]
        have = {play for plays in forms.values() for play in plays}
        for play in stock:
            if play not in have:
                return play
        self.fail("Singleback Bunch TE plays are already in the trimmed book")
        return stock[0]

    def test_thin_model_does_not_write_audit(self) -> None:
        _write_model(self.models, game="madden27", opponent_type="human", look="cover_3",
                     play="Inside Zone", n=5, n_vods=1, tentative=True, shrunk=0.9, lower=0.2)
        os.environ["CFB_COACH_VOD_MODELS"] = str(self.models)
        db = self.madden_db()
        try:
            _cover3(db, "james")
            before = db.get_meta(AUDIT_KEY)
            plan = build_prep_plan("james", db=db, offline=True, persist=True)
            self.assertIsNone(before)
            self.assertIsNone(db.get_meta(AUDIT_KEY))
            self.assertIsNone(db.get_meta("vod_book_history"))
            self.assertEqual(plan["vod_report"]["changes"], [])
        finally:
            db.close()

    def test_prior_off_and_freeze_block_the_edit(self) -> None:
        db = self.madden_db()
        try:
            _cover3(db, "james")
            base = build_prep_plan("james", db=db, offline=True, persist=True)
            base_forms = _forms(db)
            play = self._target_play(base_forms)
            _write_model(self.models, game="madden27", opponent_type="human", look="cover_3", play=play)
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.models)
            os.environ["CFB_COACH_NO_VOD_PRIOR"] = "1"
            build_prep_plan("james", db=db, offline=True, persist=True)
            self.assertEqual(_forms(db), base_forms)
            self.assertIsNone(db.get_meta(AUDIT_KEY))
            self.assertNotIn("vod_report", base)
            os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
            os.environ["CFB_COACH_VOD_FREEZE_BOOK"] = "1"
            frozen = build_prep_plan("james", db=db, offline=True, persist=True)
            self.assertEqual(_forms(db), base_forms)
            self.assertIsNone(db.get_meta(AUDIT_KEY))
            self.assertTrue(frozen["vod_report"]["frozen"])
            self.assertEqual(frozen["vod_report"]["changes"], [])
        finally:
            db.close()

    def test_swap_is_live_immediately_for_calls_and_macros(self) -> None:
        db = self.madden_db()
        try:
            _cover3(db, "james")
            os.environ["CFB_COACH_NO_VOD_PRIOR"] = "1"
            build_prep_plan("james", db=db, offline=True, persist=True)
            cached = play_candidates(db, "offense")
            self.assertTrue(cached)
            play = self._target_play(cached)
            os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
            _write_model(self.models, game="madden27", opponent_type="human", look="cover_3", play=play)
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.models)
            plan = build_prep_plan("james", db=db, offline=True, persist=True)
            changes = plan["vod_report"]["changes"]
            self.assertEqual(len(changes), 1)
            change = changes[0]
            self.assertIn(change["action"], ("swap_formation", "add_formation"))
            self.assertEqual(change["n"], 20)
            self.assertAlmostEqual(change["success"], 0.82)
            self.assertAlmostEqual(change["lower_bound"], 0.61)
            self.assertEqual(change["tier"], "med")
            self.assertEqual(change["model_version"], "v9.9")
            added = change["changed"].split(" / ")[0]
            removed = change.get("replaced") or ""
            text = format_delta_text(plan)
            self.assertIn("## Playbook changes", text)
            self.assertIn(added, text)
            self.assertIn("n=20", text)
            audit = read_audit(db)
            self.assertEqual(audit[-1]["action"], change["action"])
            self.assertEqual(audit[-1]["replaced"], removed)
            # A session that started on the old book picks up the edit on the next snap.
            fresh = book_for_next_snap(db, "offense", cached=cached)
            self.assertIn(added, fresh)
            self.assertNotEqual(fresh, cached)
            if removed:
                self.assertNotIn(removed, fresh)
            from cfb_coach.madden.catalog import book_formations

            stock = book_formations("offense", "Buccaneers")[added]
            self.assertEqual(set(fresh[added]), set(stock))
            for i in range(12):
                call = make_call(parse_madden_situation("1&10"), "james", db, rng=random.Random(i))
                self.assertIn(call.formation, fresh)
                self.assertNotEqual(call.formation, removed)
                self.assertIn(call.play, fresh[call.formation])
            macros = macro_candidates(db, "james")
            self.assertTrue(any(row["formation"] == added for row in macros))
            self.assertTrue(all(row["formation"] != removed for row in macros))
            self.assertTrue(any(row["kind"] == "package" and row["formation"] != removed for row in macros))
            packages = [row for row in macros if row["kind"] == "package"]
            self.assertTrue(packages)
            self.assertTrue(all(row["formation"] in fresh and row["play"] in fresh[row["formation"]] for row in packages))
        finally:
            db.close()

    def test_audit_revert_restores_the_previous_book(self) -> None:
        db = self.madden_db()
        try:
            _cover3(db, "james")
            os.environ["CFB_COACH_NO_VOD_PRIOR"] = "1"
            build_prep_plan("james", db=db, offline=True, persist=True)
            before = _forms(db)
            play = self._target_play(before)
            os.environ.pop("CFB_COACH_NO_VOD_PRIOR", None)
            _write_model(self.models, game="madden27", opponent_type="human", look="cover_3", play=play)
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.models)
            build_prep_plan("james", db=db, offline=True, persist=True)
            self.assertNotEqual(_forms(db), before)
            row = revert_last(db, game="madden27")
            assert row is not None
            self.assertEqual(row["action"], "revert")
            self.assertEqual(_forms(db), before)
            self.assertEqual(read_audit(db)[-1]["action"], "revert")
            fresh = play_candidates(db, "offense")
            self.assertEqual(set(fresh), set(before))
        finally:
            db.close()

    def test_in_book_vod_can_switch_the_call_without_editing_the_book(self) -> None:
        from cfb_coach.vod_model.live import apply_madden_call

        db = self.madden_db()
        try:
            build_prep_plan("james", db=db, offline=True, persist=True)
            forms = _forms(db)
            self.assertIn("Mesh", forms.get("Gun 5WR Tight") or [])
            _write_model(self.models, game="madden27", opponent_type="human", look="cover_3", play="Mesh")
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.models)
            os.environ["CFB_COACH_VOD_FREEZE_BOOK"] = "1"
            sit = parse_madden_situation("1&10 showing cover 3")
            call = MaddenCall("offense", "Pistol Trips", "HB Dive", "No adj", "", "base", macro="O-RUN")
            switched = apply_madden_call(call, sit, "james", db, {"offense": forms})
            self.assertEqual((switched.formation, switched.play), ("Gun 5WR Tight", "Mesh"))
            self.assertEqual(_forms(db), forms)
            os.environ["CFB_COACH_NO_VOD_PRIOR"] = "1"
            again = MaddenCall("offense", "Pistol Trips", "HB Dive", "No adj", "", "base")
            stayed = apply_madden_call(again, sit, "james", db, {"offense": forms})
            self.assertEqual((stayed.formation, stayed.play), ("Pistol Trips", "HB Dive"))
        finally:
            db.close()

    def test_vod_override_builds_reads_and_macro_for_the_final_play(self) -> None:
        from cfb_coach.madden import adjustments as adjmod
        from cfb_coach.madden import offense_macros as om
        from cfb_coach.madden.data import reads_for
        from cfb_coach.madden.offense_macros import pairs_in_book

        db = self.madden_db()
        try:
            build_prep_plan("james", db=db, offline=True, persist=True)
            forms = _forms(db)
            self.assertIn("Mesh", forms.get("Gun 5WR Tight") or [])
            self.assertTrue(any("HB Dive" in plays for plays in forms.values()))
            _write_model(self.models, game="madden27", opponent_type="human", look="cover_3", play="Mesh")
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.models)
            labeled: list[str] = []
            real_suggest = om.suggest_for_snap
            real_adjust = adjmod.offense_adjustment

            def _suggest(**kwargs):
                labeled.append(kwargs["play"])
                return real_suggest(**kwargs)

            def _adjust(**kwargs):
                labeled.append(kwargs["play"])
                return real_adjust(**kwargs)

            def _sample(rows, rng):
                for row in rows:
                    if row["play"] == "HB Dive":
                        return row
                self.fail("HB Dive was not in the pre-VOD pool")

            sit = parse_madden_situation("1&10 showing cover 3")
            with (
                mock.patch("cfb_coach.playcaller._sample", _sample),
                mock.patch("cfb_coach.madden.offense_macros.suggest_for_snap", _suggest),
                mock.patch("cfb_coach.madden.adjustments.offense_adjustment", _adjust),
            ):
                call = make_call(sit, "james", db, rng=random.Random(0), active_macros=[], live_macros=True)
            self.assertEqual((call.formation, call.play), ("Gun 5WR Tight", "Mesh"))
            self.assertEqual(call.read_or_user, reads_for(call.play))
            self.assertNotEqual(call.read_or_user, reads_for("HB Dive"))
            self.assertTrue(labeled)
            self.assertTrue(all(play == call.play for play in labeled))
            if call.macro:
                paired = pairs_in_book(call.macro, forms, cap=500)
                self.assertTrue(
                    any(call.play.lower() == item.split(" (", 1)[0].lower() for item in paired)
                )
        finally:
            db.close()


class TestCfbBook(_Isolated):
    def test_cfb_edit_is_current_and_reverts(self) -> None:
        from cfb_coach.cfb_catalog import book_plays, formations
        from cfb_coach.cfb_playbook import _insert, callable_book, current_rev, ensure_table, pending_rev
        from cfb_coach.vod_model.book import consider_cfb

        from cfb_coach.cfb_catalog import formations_with_play

        names = list(formations())
        keep = names[0]
        add_play = ""
        add = ""
        for formation in names[1:]:
            for play in formations()[formation]:
                owners = formations_with_play(play)
                if owners and owners[0] == formation and formation != keep:
                    add, add_play = formation, play
                    break
            if add_play:
                break
        self.assertTrue(add_play, "no CFB play is unique to a formation outside the seed")
        db = CoachDB(Path(os.environ["CFB_COACH_DB"]))
        try:
            ensure_table(db.conn)
            _insert(
                db, "alabama", status="current", kind="seed",
                book={
                    "name": "BAMA META O (custom)", "dynasty": "alabama", "schema": 2,
                    "formations": {keep: {"source_book": None, "plays": list(book_plays(keep, None) or formations()[keep])}},
                },
                edits=[], summary="seed", parent_rev=None, applied=True,
            )
            self.assertIsNone(db.get_meta(AUDIT_KEY))
            _cover3(db, "james")
            _write_model(
                self.dir / "cfb-models", game="cfb27", opponent_type="human", look="cover_3",
                play=add_play, version="v9.9",
            )
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.dir / "cfb-models")
            model = load_model()
            assert model is not None
            # No look match against a cpu-only file would no-op; this file is human.
            rows = consider_cfb(db, "james", model, dynasty="alabama", persist=True)
            self.assertEqual(len(rows), 1)
            book = callable_book(db, "alabama")
            assert book is not None
            self.assertIn(add, book["formations"])
            self.assertTrue(book["confirmed"])
            self.assertEqual(current_rev(db, "alabama")["status"], "current")
            self.assertIsNone(pending_rev(db, "alabama"))
            reverted = revert_last(db, game="cfb27")
            assert reverted is not None
            self.assertEqual(reverted["action"], "revert")
            after = callable_book(db, "alabama")
            assert after is not None
            self.assertNotIn(add, after["formations"])
            self.assertIn(keep, after["formations"])
        finally:
            db.close()

    def test_cfb_without_a_book_is_noop(self) -> None:
        from cfb_coach.vod_model.book import consider_cfb

        db = CoachDB(Path(os.environ["CFB_COACH_DB"]))
        try:
            _cover3(db, "james")
            _write_model(self.dir / "cfb-models", game="cfb27", opponent_type="human", look="cover_3", play="Mesh")
            os.environ["CFB_COACH_VOD_MODELS"] = str(self.dir / "cfb-models")
            model = load_model()
            assert model is not None
            self.assertEqual(consider_cfb(db, "james", model, dynasty="alabama", persist=True), [])
            self.assertIsNone(db.get_meta(AUDIT_KEY))
        finally:
            db.close()

