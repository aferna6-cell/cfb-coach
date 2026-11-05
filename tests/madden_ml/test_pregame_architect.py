"""Pregame playbook architect: catalog, portfolio search, install workflow, live scope.

Temporary databases only. Formations and play lists come from the parsed
catalog, so a confirmed book is always a real source-book list.
"""
from __future__ import annotations

import copy
import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.madden import catalog_full, playbook
from cfb_coach.madden.model import defense_coordinator, defense_v1, experimental_model, game_evidence, game_prep
from cfb_coach.madden.model import playbook_architect as arch
from cfb_coach.madden.model import playbook_install as install
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import Situation

ATTEST = "I built this exact book in Madden 27 and selected it on Gameday."


def _seed() -> dict:
    return {"opponents": {
        "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low", "profile_json": "{}"},
        "ryan": {"display_name": "Ryan", "team_now": "CIN", "skill": "human", "confidence": "low",
                 "profile_json": "{}"},
    }}


def _manual_design(side: str, n: int = 3, skip: int = 0) -> dict:
    """A design made of real catalog formations with their complete source play lists."""
    books = catalog_full.architect_books(side)
    forms, sources = {}, {}
    for book in sorted(books):
        for form, rec in sorted((books[book].get("formations") or {}).items()):
            if form in forms or not rec.get("plays"):
                continue
            if skip:
                skip -= 1
                continue
            forms[form] = list(dict.fromkeys(catalog_full.source_plays(side, book, form)))
            sources[form] = book
            if len(forms) == n:
                break
        if len(forms) == n:
            break
    book = {"name": arch.BOOK_NAMES[side], "custom": True, "formations": forms, "formation_sources": sources,
            "source_book": next(iter(sources.values()))}
    return {"side": side, "book": book, "fingerprint": arch.fingerprint(side, forms, sources)}


def _cand(name: str, value: float, *, runs: int = 4, passes: int = 6, group: str = "11",
          concepts: tuple[str, ...] = ("mesh",), family: str | None = None) -> dict:
    plays = [f"{name} run {i}" for i in range(runs)] + [f"{name} pass {i}" for i in range(passes)]
    return {"id": f"{name} | Book", "formation": name, "family": family or name, "source_book": "Book",
            "plays": plays, "values": {s: value for s in arch.OFFENSE_WEIGHTS}, "good_calls": {},
            "concepts": list(concepts), "group": group, "runs": runs, "evidence": {"n": 0}}


def _objective(kind: str = "human") -> arch.Objective:
    return arch.Objective("offense", arch.situation_weights("offense", {}), kind)


class _DBCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "madden.db", seed=_seed())
        self.root = Path(self.tmp.name) / "game_prep"

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def stage(self, side: str, opponent: str = "ryan", **kw) -> dict:
        design = _manual_design(side, **kw)
        applied = playbook.load_books(self.db).get(side)
        p = install.build_proposal(design, opponent_id=opponent, applied=applied)
        install.save_proposals(self.root, opponent, {side: p})
        return p

    def confirm(self, p: dict, **kw) -> dict:
        args = dict(root=self.root, opponent_id=p["opponent_id"], side=p["side"], proposal_id=p["proposal_id"],
                    typed_fingerprint=p["fingerprint"], attestation=ATTEST)
        args.update(kw)
        return install.confirm(self.db, **args)

    def meta(self) -> str | None:
        return self.db.get_meta(playbook.META_KEY)


class CatalogTests(unittest.TestCase):
    def test_full_catalog_is_parsed_and_every_indexed_formation_is_covered(self) -> None:
        audit = catalog_full.completeness_audit()
        self.assertTrue(audit["full_catalog_present"])
        self.assertEqual(audit["parse_failures"], [])
        for side in ("offense", "defense"):
            s = audit["sides"][side]
            self.assertGreater(s["formation_index_size"], 0)
            self.assertEqual(s["formations_missing_from_parsed_books"], [])
            self.assertEqual(s["formation_index_covered"], s["formation_index_size"])

    def test_every_record_keeps_its_own_source_book_play_list(self) -> None:
        for side in ("offense", "defense"):
            for rec in catalog_full.formation_records(side)[:200]:
                self.assertEqual(list(rec["plays"]),
                                 list(dict.fromkeys(catalog_full.source_plays(side, rec["source_book"],
                                                                             rec["formation"]))))
                self.assertTrue(rec["source_book"])
                self.assertIn("verification", rec)

    def test_defense_design_inspects_every_catalogued_formation(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        db = CoachDB(Path(tmp.name) / "m.db", seed=_seed())
        try:
            d = arch.design_side("defense", opponent_id="ryan", db=db, applied=None,
                                 evidence=game_evidence.defense_evidence(db, "ryan"))
        finally:
            db.close()
            tmp.cleanup()
        unique = catalog_full.completeness_audit()["sides"]["defense"]["unique_formations"]
        self.assertEqual(d["catalog"]["formations_inspected"], unique)
        self.assertLessEqual(d["limits"]["formations"], 15)
        self.assertLessEqual(d["limits"]["plays"], arch.MAX_PLAYS)
        self.assertEqual(install.validate_book("defense", d["book"]), [])


class PortfolioSearchTests(unittest.TestCase):
    def test_cap_and_play_limit_are_never_exceeded(self) -> None:
        pool = [_cand(f"F{i}", 0.3 + i * 0.01, concepts=(f"c{i}",), group=str(i % 7)) for i in range(40)]
        members, _ = arch.search(pool, [], _objective(), cap=15, threshold=1e-9)
        self.assertLessEqual(len(members), 15)
        self.assertTrue(arch.feasible(members, 15))
        start = pool[:20]
        trimmed, moves = arch.search(pool, start, _objective(), cap=15, threshold=0.012)
        self.assertLessEqual(len(trimmed), 15)
        self.assertTrue(any("editor limits" in str(m.get("why")) for m in moves))

    def test_fifteen_is_a_ceiling_not_a_quota(self) -> None:
        pool = [_cand(f"F{i}", 0.5) for i in range(30)]
        members, _ = arch.search(pool, [], _objective("cpu"), cap=15, threshold=1e-9)
        self.assertLess(len(members), 15)

    def test_an_unchanged_optimal_book_stays_unchanged(self) -> None:
        pool = [_cand("A", 0.6), _cand("B", 0.55, group="12", concepts=("flood",)), _cand("C", 0.3)]
        ideal, _ = arch.search(pool, [], _objective(), cap=15, threshold=1e-9)
        again, moves = arch.search(pool, ideal, _objective(), cap=15, threshold=0.012)
        self.assertEqual(moves, [])
        self.assertEqual({c["id"] for c in again}, {c["id"] for c in ideal})

    def test_a_superior_formation_enters(self) -> None:
        a, b, c = _cand("A", 0.40), _cand("B", 0.42, group="12"), _cand("C", 0.75, group="10")
        final, moves = arch.search([a, b, c], [a, b], _objective(), cap=15, threshold=0.012)
        self.assertIn(c["id"], {m["id"] for m in final})
        self.assertTrue(any(m.get("added") == c["id"] for m in moves))

    def test_a_failing_formation_leaves_for_a_better_one(self) -> None:
        a, f, g = _cand("A", 0.6), _cand("F", 0.15), _cand("G", 0.55)
        final, moves = arch.search([a, f, g], [a, f], _objective(), cap=2, threshold=0.012)
        self.assertNotIn(f["id"], {m["id"] for m in final})
        self.assertTrue(any(m.get("removed") == f["id"] for m in moves))

    def test_small_gains_do_not_churn_the_book(self) -> None:
        a, b = _cand("A", 0.50), _cand("B", 0.505)
        final, moves = arch.search([a, b], [a], _objective(), cap=1, threshold=0.012)
        self.assertEqual([m["id"] for m in final], [a["id"]])
        self.assertEqual(moves, [])

    def test_the_design_is_deterministic_with_no_timer_input(self) -> None:
        pool = [_cand(f"F{i}", 0.3 + (i % 5) * 0.05, group=str(i % 4), concepts=(f"c{i % 6}",)) for i in range(20)]
        one = arch.search(pool, pool[:3], _objective(), cap=15, threshold=0.012)
        with mock.patch("time.time", return_value=time.time() + 86400 * 30):
            two = arch.search(pool, pool[:3], _objective(), cap=15, threshold=0.012)
        self.assertEqual([c["id"] for c in one[0]], [c["id"] for c in two[0]])
        self.assertIn("timer", arch.no_schedule_statement())

    def test_installed_formations_stay_until_the_book_is_full(self) -> None:
        recs = [r for r in catalog_full.formation_records("defense") if r["source_book"] == "49ers"][:4]
        weak = recs[0]
        applied = {"name": "Old", "formations": {weak["formation"]: list(weak["plays"])},
                   "formation_sources": {weak["formation"]: "49ers"}}
        bad = {"formation": {weak["formation"]: {"delta": -0.8, "n": 80}}, "verified_executions": 80}
        loose = arch.design_side("defense", opponent_id="ryan", db=None, applied=applied, evidence=bad,
                                 evidence_quality="empirical", remove_only_when_full=False)
        kept = arch.design_side("defense", opponent_id="ryan", db=None, applied=applied, evidence=bad,
                                evidence_quality="empirical")
        self.assertIn(weak["formation"], loose["changes"]["remove"])
        self.assertEqual(kept["changes"]["remove"], [])
        self.assertIn(weak["formation"], kept["book"]["formations"])
        self.assertIn("15", kept["objective"]["removal_policy"])
        full = arch.design_side("defense", opponent_id="ryan", db=None, applied=applied, evidence=bad,
                                evidence_quality="empirical", max_formations=1)
        self.assertNotIn(weak["formation"], full["book"]["formations"])

    def test_switching_a_formation_to_a_better_source_book_is_not_an_add_and_remove(self) -> None:
        from cfb_coach.madden.defense_select import FAMILIES

        def rec(book: str, fams: dict) -> dict:
            plays = [f"{book} call {i}" for i in range(8)]
            return {"side": "defense", "formation": "Nickel Over", "family": "Nickel", "set": "Over",
                    "source_book": book, "plays": plays, "package": "nickel", "book_kind": "team",
                    "profile": {"coverage_families": fams, "match_calls": 0},
                    "verification": {"status": "test"}}

        rich = rec("BookA", {f: 3 for f in FAMILIES})
        thin = rec("BookB", {})
        applied = {"name": "Old", "formations": {"Nickel Over": thin["plays"]},
                   "formation_sources": {"Nickel Over": "BookB"}}
        d = arch.design_side("defense", opponent_id="ryan", db=None, applied=applied, evidence={},
                             records=[rich, thin], max_formations=1)
        self.assertEqual(d["changes"]["new_source"], ["Nickel Over"])
        self.assertEqual(d["changes"]["add"], [])
        self.assertEqual(d["changes"]["remove"], [])
        self.assertEqual(d["formations"][0]["previous_source_book"], "BookB")
        self.assertEqual(d["book"]["formation_sources"], {"Nickel Over": "BookA"})

    def test_failed_defensive_evidence_lowers_that_formation_only(self) -> None:
        recs = catalog_full.formation_records("defense")[:6]
        target = recs[0]["formation"]
        base = {c["id"]: c["values"] for c in arch.defense_candidates(None, "ryan", {}, recs)}
        bad = {c["id"]: c["values"] for c in arch.defense_candidates(
            None, "ryan", {"formation": {target: {"delta": -0.4, "n": 40}}}, recs)}
        for cid, vals in base.items():
            for sit, v in vals.items():
                expected = v - 0.1 if cid.startswith(target + " |") else v
                self.assertAlmostEqual(bad[cid][sit], expected, places=5)


class InstallWorkflowTests(_DBCase):
    def test_generation_and_staging_leave_the_applied_books_untouched(self) -> None:
        before = self.meta()
        self.stage("offense")
        self.stage("defense")
        self.assertEqual(self.meta(), before)
        self.assertEqual(install.side_status(self.db, self.root, "ryan", "offense")["state"], "awaiting_install")

    def test_confirm_writes_only_that_side_and_marks_it_ready(self) -> None:
        off = self.stage("offense")
        de = self.stage("defense")
        self.assertNotEqual(off["proposal_id"], de["proposal_id"])
        self.assertNotEqual(off["fingerprint"], de["fingerprint"])
        result = self.confirm(de)
        self.assertTrue(result["confirmed"])
        books = playbook.load_books(self.db)
        self.assertEqual(books["defense"]["formations"], de["book"]["formations"])
        self.assertEqual(books["defense"]["architect_proposal_id"], de["proposal_id"])
        self.assertNotIn("offense", books)
        self.assertEqual(install.side_status(self.db, self.root, "ryan", "defense")["state"], "ready")
        self.assertEqual(install.side_status(self.db, self.root, "ryan", "offense")["state"], "awaiting_install")
        self.confirm(off)
        self.assertEqual(playbook.load_books(self.db)["defense"]["formations"], de["book"]["formations"])

    def test_confirmation_cannot_be_fabricated(self) -> None:
        p = self.stage("defense")
        before = self.meta()
        cases = {
            "wrong fingerprint": dict(typed_fingerprint="DEF-0000-0000"),
            "short attestation": dict(attestation="done"),
            "other operator": dict(operator="Ryan"),
            "unknown proposal": dict(proposal_id="deadbeefdeadbeef"),
            "wrong side": dict(side="offense"),
        }
        for label, kw in cases.items():
            with self.subTest(label), self.assertRaises(install.InstallError):
                self.confirm(p, **kw)
        self.assertEqual(self.meta(), before)

    def test_tampered_superseded_and_stale_proposals_are_refused(self) -> None:
        p = self.stage("defense")
        path = install.proposal_dir(self.root, "ryan") / f"defense-{p['proposal_id']}.json"
        raw = json.loads(path.read_text())
        first = next(iter(raw["book"]["formations"]))
        raw["book"]["formations"][first] = raw["book"]["formations"][first][:1]
        path.write_text(json.dumps(raw))
        with self.assertRaises(install.InstallError):
            self.confirm(p)

        old = self.stage("defense")
        self.stage("defense", skip=1)
        with self.assertRaisesRegex(install.InstallError, "superseded"):
            self.confirm(old)

        current = self.stage("defense", skip=2)
        state = playbook._load_state(self.db)
        state["applied"]["defense"] = playbook.make_record("defense", "stock", "49ers")
        playbook._save_state(self.db, state)
        with self.assertRaisesRegex(install.InstallError, "changed since"):
            self.confirm(current)

    def test_a_book_that_drifts_from_its_source_list_is_refused(self) -> None:
        design = _manual_design("defense")
        first = next(iter(design["book"]["formations"]))
        design["book"]["formations"][first] = design["book"]["formations"][first][:-1]
        design["fingerprint"] = arch.fingerprint("defense", design["book"]["formations"],
                                                 design["book"]["formation_sources"])
        p = install.build_proposal(design, opponent_id="ryan", applied=None)
        install.save_proposals(self.root, "ryan", {"defense": p})
        with self.assertRaisesRegex(install.InstallError, "complete"):
            self.confirm(p)
        self.assertIsNone(playbook.load_books(self.db).get("defense"))

    def test_a_failed_install_leaves_the_previous_book_intact(self) -> None:
        first = self.stage("defense")
        self.confirm(first)
        before = self.meta()
        second = self.stage("defense", skip=3)
        with mock.patch.object(install, "_snapshot", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.confirm(second)
        self.assertEqual(self.meta(), before)
        with self.assertRaises(Exception):
            install._write_atomically(self.db, {playbook.META_KEY: "{}", install.HISTORY_KEY: object()})
        self.assertEqual(self.meta(), before)

    def test_rollback_restores_each_side_independently(self) -> None:
        stock = playbook.make_record("defense", "stock", "49ers")
        playbook._save_state(self.db, {"applied": {"defense": copy.deepcopy(stock)}, "pending": {}})
        de = self.stage("defense")
        off = self.stage("offense")
        self.confirm(de)
        self.confirm(off)
        with self.assertRaises(install.InstallError):
            install.rollback(self.db, root=self.root, side="defense", attestation="short")
        out = install.rollback(self.db, root=self.root, side="defense", attestation=ATTEST)
        self.assertTrue(out["rolled_back"])
        books = playbook.load_books(self.db)
        self.assertEqual(books["defense"]["formations"], stock["formations"])
        self.assertEqual(books["offense"]["architect_proposal_id"], off["proposal_id"])
        install.rollback(self.db, root=self.root, side="offense", attestation=ATTEST)
        self.assertNotIn("offense", playbook.load_books(self.db))
        with self.assertRaises(install.InstallError):
            install.rollback(self.db, root=self.root, side="offense", attestation=ATTEST)

    def test_macro_approvals_survive_a_book_that_only_adds(self) -> None:
        key = "ml_offense_verified_macros.v1:cpu"
        first = self.stage("offense", n=2)
        self.confirm(first)
        self.db.set_meta(key, "approved")
        grown = _manual_design("offense", n=3)
        p = install.build_proposal(grown, opponent_id="ryan", applied=playbook.load_books(self.db)["offense"])
        install.save_proposals(self.root, "ryan", {"offense": p})
        out = self.confirm(p)
        self.assertTrue(out["macro_approvals"].startswith("kept"))
        self.assertEqual(self.db.get_meta(key), "approved")
        shrunk = self.stage("offense", n=2, skip=3)
        out = self.confirm(shrunk)
        self.assertTrue(out["macro_approvals"].startswith("revoked"))
        self.assertEqual(self.db.get_meta(key), "")

    def test_an_unknown_human_starts_from_a_pooled_prior(self) -> None:
        from cfb_coach.madden.model.opponent_tendencies import pooled_human_profile

        self.assertEqual(pooled_human_profile(self.db, None, exclude="dillon")["n"], 0)
        self.db.bump_tendency("ryan", "offense_concept", "Inside Zone", amount=12)
        self.db.bump_tendency("ryan", "offense_concept", "Mesh", amount=4)
        self.db.bump_tendency("cpu", "offense_concept", "Four Verticals", amount=40)
        self.db.bump_tendency("dillon", "offense_concept", "Stick", amount=40)
        pooled = pooled_human_profile(self.db, None, exclude="dillon")
        self.assertEqual(pooled["n"], 16)
        self.assertAlmostEqual(pooled["run_share"], 0.75)
        self.assertLessEqual(pooled["shrink"], 0.4)
        self.assertIn("league prior", pooled["summary"])

    def test_ordinary_prep_does_not_rewrite_a_confirmed_architect_book(self) -> None:
        p = self.stage("defense")
        self.confirm(p)
        current = playbook.load_books(self.db)["defense"]
        plan = playbook.plan_side("defense", current, opp={}, choice=None, team=None)
        self.assertEqual(plan["record"]["formations"], current["formations"])
        self.assertEqual(plan["status"], "applied")


class GamePrepTests(_DBCase):
    def test_game_prep_designs_both_sides_without_touching_the_live_book(self) -> None:
        art = experimental_model.ExperimentalArtifact(model_version="test.model", n_supervised=3,
                                                      evidence_quality="mixed", global_rate=0.5)
        before = self.meta()
        with mock.patch.object(game_prep, "_load_offense_artifact", return_value=(art, "mixed")):
            one = game_prep.run_game_prep(self.db, opponent_id="ryan", root=self.root, now="2026-11-05T10:00:00+00:00")
            two = game_prep.run_game_prep(self.db, opponent_id="ryan", root=self.root, now="2026-11-12T10:00:00+00:00")
        self.assertEqual(self.meta(), before)
        for side in ("offense", "defense"):
            s1, s2 = one["sides"][side], two["sides"][side]
            self.assertEqual(s1["fingerprint"], s2["fingerprint"], "a week passing must not change the book")
            self.assertNotEqual(s1["proposal_id"], s2["proposal_id"])
            self.assertLessEqual(s1["design"]["limits"]["formations"], 15)
            for f in s1["design"]["formations"]:
                self.assertEqual(f["plays"], list(dict.fromkeys(
                    catalog_full.source_plays(side, f["source_book"], f["formation"]))))
                self.assertTrue(f["why"])
        self.assertFalse(two["game_ready"]["ready"])
        self.assertEqual(two["game_ready"]["needed"], ["offense", "defense"])
        old = one["sides"]["defense"]
        with self.assertRaisesRegex(install.InstallError, "superseded"):
            install.confirm(self.db, root=self.root, opponent_id="ryan", side="defense",
                            proposal_id=old["proposal_id"], typed_fingerprint=old["fingerprint"], attestation=ATTEST)
        md = game_prep.render_markdown(two)
        html = game_prep.render_html(two)
        self.assertIn(two["sides"]["defense"]["fingerprint"], md)
        self.assertIn(two["sides"]["offense"]["fingerprint"], html)


class EvidenceTests(_DBCase):
    def test_score_correction_applies_only_to_the_stored_wrong_score(self) -> None:
        fixed = game_evidence.corrected_result({"session_id": "3d9b477c10df4827", "score": "42-7", "result_wl": "W"})
        self.assertEqual(fixed["score"], "35-7")
        self.assertTrue(fixed["corrected"])
        same = game_evidence.corrected_result({"session_id": "3d9b477c10df4827", "score": "35-7", "result_wl": "W"})
        self.assertFalse(same["corrected"])
        other = game_evidence.corrected_result({"session_id": "other", "score": "42-7", "result_wl": "W"})
        self.assertEqual(other["score"], "42-7")

    def _snap(self, ts: str, verified: str = "verified") -> None:
        self.db.conn.execute(
            "INSERT INTO snaps (ts, opponent_id, side, down, distance, yardline, formation, play, result, "
            "session_id, executed_status, executed_verification, executed_formation, executed_play, yards) "
            "VALUES (?, 'ryan', 'offense', 1, 10, 30, 'Gun Bunch', 'Mesh', 'gain', 's1', 'identified', ?, "
            "'Gun Bunch', 'Mesh', 12)", (ts, verified))
        self.db.conn.commit()

    def test_no_future_rows_and_only_verified_executions(self) -> None:
        self._snap("2026-11-01T00:00:00")
        self._snap("2026-11-01T00:01:00", verified="unverified")
        self._snap("2026-11-20T00:00:00")
        rows = game_evidence.verified_offense_rows(self.db, "ryan", before_ts="2026-11-05T00:00:00")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(game_evidence.verified_offense_rows(self.db, "ryan")), 2)
        self.assertEqual(game_evidence.defense_evidence(self.db, "ryan")["verified_executions"], 0)


def _artifact(best: tuple[str, str], other: tuple[str, str], gap: float = 2.0) -> dict:
    return defense_v1.validate_artifact({
        "schema": defense_v1.SCHEMA, "policy": defense_v1.POLICY, "target": defense_v1.TARGET,
        "intercept": 0.0, "features": ["red_zone"], "coefficients": {"red_zone": 0.0},
        "call_families": ["shell", "pressure"], "packages": ["base"],
        "family_coefficients": {"shell": -gap, "pressure": gap}, "package_coefficients": {"base": 0.0},
        "candidates": [
            {"formation": best[0], "play": best[1], "family": "shell", "package": "base"},
            {"formation": other[0], "play": other[1], "family": "pressure", "package": "base"},
        ],
    })


class LiveScopeTests(_DBCase):
    def test_installed_scope_never_calls_an_uninstalled_formation(self) -> None:
        art = _artifact(("4-3 Over", "Cover 3"), ("Nickel 2-4-5", "Cover 1"))
        book = {"Nickel 2-4-5": ["Cover 1"]}
        sit = Situation(raw="2&7", side="defense", down=2, distance=7)
        with self.assertRaises(defense_v1.DefenseV1Error) as caught:
            defense_v1.pick_call(art, sit, book)
        self.assertEqual(caught.exception.reason, "illegal")
        out = defense_v1.pick_call(art, sit, book, installed_scope=True)
        self.assertEqual(out["pick"]["formation"], "Nickel 2-4-5")
        with self.assertRaises(defense_v1.DefenseV1Error) as gap:
            defense_v1.pick_call(art, sit, {"Dime 1-3-7": ["Spy Contain"]}, installed_scope=True)
        self.assertEqual(gap.exception.reason, "installed_gap")

    def test_untrained_installed_plays_are_scored_by_coverage_family_and_package(self) -> None:
        art = _artifact(("Nickel 2-4-5", "Cover 3"), ("Dime 1-3-7", "Cover 1"))
        art = dict(art, packages=["base", "four_three", "nickel", "dime"],
                   package_coefficients={"base": 0.0, "four_three": 0.0, "nickel": 0.0, "dime": 0.0})
        art["call_families"] = ["shell", "pressure", "cover3", "one_high_man"]
        art["family_coefficients"] = {"shell": 0.0, "pressure": 0.0, "cover3": -1.0, "one_high_man": 1.0}
        book = {"46 Normal": ["Cover 3 Sky", "Cover 1 Robber"], "Nickel 2-4-5": ["Cover 3"]}
        cands = defense_v1.installed_candidates(art, book)
        mapped = {(c["formation"], c["play"]): c for c in cands if c.get("mapped")}
        self.assertEqual(mapped[("46 Normal", "Cover 3 Sky")]["family"], "cover3")
        self.assertEqual(mapped[("46 Normal", "Cover 3 Sky")]["package"], "four_three")
        sit = Situation(raw="3&1", side="defense", down=3, distance=1, short_yardage=True)
        out = defense_v1.pick_call(art, sit, book, installed_scope=True)
        self.assertEqual(out["pick"]["formation"], "46 Normal")

    def test_common_madden_concept_names_reach_the_live_adjustment(self) -> None:
        from cfb_coach.madden.situation import concept_family, parse_madden_situation

        expected = {
            "Power": "run", "Counter": "run", "HB Draw": "run", "Duo": "run", "Toss": "run", "Run": "run",
            "Read Option": "run", "QB Draw": "scram", "Smash": "flood", "Curl Flat": "flood",
            "Drive": "cross", "Mesh": "cross", "Go": "vert", "Four Verticals": "vert",
            "Mtn Shuffle Verts Smash": "vert", "Mtn RPO Zone Alert": "rpo",
        }
        for name, family in expected.items():
            self.assertEqual(concept_family(name), family, name)
        self.assertIsNone(concept_family("Goal line"))
        for raw, hint in [("2&6 own 34 power", "Power"), ("3&4 HB draw", "HB Draw"), ("2&7 read option", "Read Option")]:
            self.assertEqual(parse_madden_situation(raw, default_side="defense").concept_hint, hint)
        self.assertEqual(parse_madden_situation("2&7 cover 2 trap").coverage_hint, "Cover 2")

    def test_a_thin_human_profile_hands_off_from_the_league_prior(self) -> None:
        from cfb_coach.madden.model import opponent_tendencies as ot

        prior = {"n": 160, "run_share": 0.4, "concepts": {"run": 0.4, "cross": 0.6}, "shrink": 0.4}
        own = {"n": 12, "run_share": 1.0, "concepts": {"run": 1.0}, "shrink": 0.3}
        with mock.patch.object(ot, "pooled_human_profile", return_value=prior):
            blended = ot.with_pooled_prior(None, own, None, "dillon")
            self.assertAlmostEqual(blended["run_share"], 0.7)
            self.assertIn("50% his own", blended["summary"])
            self.assertIs(ot.with_pooled_prior(None, {}, None, "dillon"), prior)
            seasoned = dict(own, n=60)
            self.assertEqual(ot.with_pooled_prior(None, seasoned, None, "dillon")["run_share"], 1.0)

    def test_a_run_heavy_game_moves_the_defense_out_of_light_boxes(self) -> None:
        art = _artifact(("Dime 3-2 Odd", "Cover 3"), ("Dime 3-2 Odd", "Cover 1"), gap=0.02)
        art = dict(art, packages=["base", "dime", "four_three"],
                   package_coefficients={"base": 0.0, "dime": -0.05, "four_three": 0.0})
        art["call_families"] = ["shell", "pressure", "cover3"]
        art["family_coefficients"] = {"shell": -0.02, "pressure": 0.02, "cover3": 0.0}
        book = {"Dime 3-2 Odd": ["Cover 3", "Cover 1"], "4-3 Over": ["Cover 3 Sky"]}
        sit = Situation(raw="1&10", side="defense", down=1, distance=10)
        quiet = defense_v1.pick_call(art, sit, book, installed_scope=True)
        self.assertEqual(quiet["pick"]["formation"], "Dime 3-2 Odd")
        runs = ["Inside Zone", "Stretch", "Inside Zone", "Outside Zone", "Mesh"]
        loud = defense_v1.pick_call(art, sit, book, installed_scope=True, recent_concepts=runs, run_heavy=False)
        self.assertEqual(loud["pick"]["formation"], "4-3 Over")
        self.assertIn("run 80%", loud["live_summary"])

    def test_heavy_front_downs_fall_back_when_no_known_heavy_front_is_installed(self) -> None:
        art = _artifact(("Nickel 2-4-5", "Cover 3"), ("Dime 1-3-7", "Cover 1"))
        sit = Situation(raw="3&1", side="defense", down=3, distance=1, short_yardage=True)
        with self.assertRaises(defense_v1.DefenseV1Error) as caught:
            defense_v1.pick_call(art, sit, {"Nickel 2-4-5": ["Cover 3"]}, installed_scope=True)
        self.assertEqual(caught.exception.reason, "installed_gap")

    def test_in_game_concentration_lets_a_close_installed_call_through(self) -> None:
        book = {"Dime 3-2 Odd": ["Cover 3"], "Nickel Over": ["Cover 1"]}
        sit = Situation(raw="2&7", side="defense", down=2, distance=7)
        recent = [("Dime 3-2 Odd", "Cover 3")] * 8 + [("Nickel Over", "Cover 1")] * 2
        close = _artifact(("Dime 3-2 Odd", "Cover 3"), ("Nickel Over", "Cover 1"), gap=0.05)
        self.assertEqual(defense_v1.pick_call(close, sit, book)["pick"]["formation"], "Dime 3-2 Odd")
        self.assertEqual(defense_v1.pick_call(close, sit, book, recent=recent, installed_scope=True)
                         ["pick"]["formation"], "Nickel Over")
        clear = _artifact(("Dime 3-2 Odd", "Cover 3"), ("Nickel Over", "Cover 1"), gap=2.0)
        self.assertEqual(defense_v1.pick_call(clear, sit, book, recent=recent, installed_scope=True)
                         ["pick"]["formation"], "Dime 3-2 Odd")

    def test_scope_turns_on_only_for_a_confirmed_architect_book(self) -> None:
        self.assertFalse(defense_coordinator._installed_scope(self.db, {}))
        self.confirm(self.stage("defense"))
        self.assertTrue(defense_coordinator._installed_scope(self.db, {}))
        self.assertFalse(defense_coordinator._installed_scope(self.db, {"defense_v1_installed_scope": False}))

    def test_scoring_stays_inside_the_150ms_budget_without_clock_or_showing(self) -> None:
        cands = [{"formation": f"F{i}", "play": f"P{i}", "family": "shell" if i % 2 else "pressure",
                  "package": "base"} for i in range(400)]
        raw = _artifact(("A", "a"), ("B", "b"))
        raw = dict(raw, candidates=cands)
        book = {f"F{i}": [f"P{i}"] for i in range(0, 400, 3)}
        sit = Situation(raw="3&6", side="defense", down=3, distance=6)
        self.assertFalse(hasattr(sit, "showing"))
        out = defense_v1.pick_call(raw, sit, book, installed_scope=True)
        self.assertLess(out["elapsed_ms"], 150.0)
        self.assertIn(out["pick"]["formation"], book)
        with self.assertRaises(defense_v1.DefenseV1Error) as caught:
            defense_v1.pick_call(raw, sit, book, budget_ms=0.0, installed_scope=True)
        self.assertEqual(caught.exception.reason, "timeout")

    def test_live_calls_come_only_from_the_confirmed_inventory(self) -> None:
        de = self.stage("defense", n=4)
        off = self.stage("offense", n=4)
        self.confirm(de)
        self.confirm(off)
        books = {s: r["formations"] for s, r in playbook.active_books(self.db).items()}
        rng = random.Random(7)
        for text in ("1&10", "3&8", "3&1", "2&5 opp 15", "1&G opp 3", "4&2"):
            call = make_call(parse_madden_situation(text), "ryan", None, rng=rng, playbook=books)
            self.assertIn(call.formation, books["offense"])
            dcall = make_call(parse_madden_situation(f"d {text}", default_side="defense"), "ryan", None,
                              rng=rng, playbook=books)
            self.assertIn(dcall.formation, books["defense"])


def _stock_books() -> dict:
    return {"offense": playbook.make_record("offense", "stock", "Buccaneers")["formations"],
            "defense": playbook.make_record("defense", "stock", "49ers")["formations"]}


class CpuDefenseOptInTests(_DBCase):
    def test_cpu_stays_offense_only_by_default(self) -> None:
        sit = parse_madden_situation("d 3&8", default_side="defense")
        call = make_call(sit, "cpu", None, rng=random.Random(1), playbook=_stock_books())
        self.assertEqual(call.side, "offense")
        self.assertIn("offense-only", call.rationale)

    def test_cpu_defense_flag_calls_defense_with_the_established_caller(self) -> None:
        sit = parse_madden_situation("d 3&8", default_side="defense")
        with mock.patch("cfb_coach.madden.model.defense_coordinator.maybe_apply_defense") as ml:
            call = make_call(sit, "cpu", None, rng=random.Random(1), playbook=_stock_books(), cpu_defense=True)
        self.assertEqual(call.side, "defense")
        self.assertIn(call.formation, _stock_books()["defense"])
        ml.assert_not_called()

    def test_cpu_defense_with_an_explicit_model_reaches_the_defensive_model(self) -> None:
        sit = parse_madden_situation("d 3&8", default_side="defense")
        with mock.patch("cfb_coach.madden.model.defense_coordinator.maybe_apply_defense",
                        side_effect=lambda call, *a, **k: call) as ml:
            make_call(sit, "cpu", None, rng=random.Random(1), playbook=_stock_books(),
                      cpu_defense=True, defense_model="v1")
        self.assertTrue(ml.called)
        self.assertTrue(ml.call_args.kwargs.get("allow_cpu"))

    def test_maybe_apply_defense_ignores_cpu_unless_allowed(self) -> None:
        sentinel = object()
        sit = parse_madden_situation("d 3&8", default_side="defense")
        self.assertIs(defense_coordinator.maybe_apply_defense(sentinel, sit, "cpu", self.db, {}), sentinel)

    def test_play_parser_exposes_the_opt_in(self) -> None:
        from cfb_coach.cli import build_parser

        args = build_parser().parse_args(["play", "--game", "madden27", "-o", "cpu"])
        self.assertFalse(args.cpu_defense)
        args = build_parser().parse_args(["play", "--game", "madden27", "-o", "cpu", "--cpu-defense"])
        self.assertTrue(args.cpu_defense)

    def test_game_prep_can_register_a_new_human(self) -> None:
        from cfb_coach.cli import build_parser
        from cfb_coach.madden.model.cli import cmd_ml_game_prep

        db_path = Path(self.tmp.name) / "madden.db"
        args = build_parser().parse_args(["ml", "game-prep", "--opponent", "dillon", "--new-opponent", "Dillon",
                                          "--db", str(db_path), "--status", "--out", str(self.root)])
        self.assertEqual(cmd_ml_game_prep(args), 0)
        from cfb_coach.madden.human_opponents import is_cold_start_opponent, resolve_madden_opponent

        self.assertTrue(is_cold_start_opponent("dillon", self.db))
        self.assertEqual(resolve_madden_opponent("dillon", self.db), "dillon")


if __name__ == "__main__":
    unittest.main()
