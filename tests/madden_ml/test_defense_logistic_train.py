"""Defense logistic v2 trainer: provenance admission, fit, held-out gate, and the CLI's file safety."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cfb_coach.madden.model import defense_logistic_train as train
from cfb_coach.madden.model import defense_v1

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["down", "down_missing", "distance", "distance_missing",
            "yardline", "yardline_missing", "short_yardage", "long_yardage"]
ZERO_IX = {name: 0.0 for name in defense_v1.INTERACTION_FEATURES}


def _template() -> dict:
    return {
        "schema": defense_v1.SCHEMA_V2, "policy": defense_v1.POLICY, "target": defense_v1.TARGET,
        "intercept": 0.0, "features": FEATURES, "coefficients": {name: 0.0 for name in FEATURES},
        "call_families": ["cover3", "pressure"], "packages": ["nickel", "four_three"],
        "family_coefficients": {"cover3": 0.0, "pressure": 0.0},
        "package_coefficients": {"nickel": 0.0, "four_three": 0.0},
        "family_situation_coefficients": {"cover3": dict(ZERO_IX), "pressure": dict(ZERO_IX)},
        "package_situation_coefficients": {"nickel": dict(ZERO_IX), "four_three": dict(ZERO_IX)},
        "candidates": [
            {"formation": "Nickel Over", "play": "Cover 3 Sky", "family": "cover3", "package": "nickel"},
            {"formation": "4-3 Over", "play": "Fire Zone", "family": "pressure", "package": "four_three"},
        ],
    }


def _db(games: list[tuple[str, str, str, list[tuple]]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE game_sessions (session_id TEXT, opponent_id TEXT, started_ts TEXT)")
    conn.execute(
        "CREATE TABLE snaps (id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, snap_seq INTEGER, "
        "situation_raw TEXT, formation TEXT, play TEXT, result TEXT, down INTEGER, distance INTEGER, "
        "executed_verification TEXT, executed_formation TEXT, executed_play TEXT)"
    )
    for session, opponent, started, snaps in games:
        conn.execute("INSERT INTO game_sessions VALUES (?,?,?)", (session, opponent, started))
        for seq, (formation, play, result, *rest) in enumerate(snaps):
            verification, ex_form, ex_play = (rest + [None, None, None])[:3]
            conn.execute(
                "INSERT INTO snaps (session_id, side, snap_seq, situation_raw, formation, play, result, "
                "down, distance, executed_verification, executed_formation, executed_play) "
                "VALUES (?, 'defense', ?, '1&10 my 30', ?, ?, ?, 1, 10, ?, ?, ?)",
                (session, seq, formation, play, result, verification, ex_form, ex_play),
            )
    conn.commit()
    return conn


def _signal_game(n: int = 20, *, flip_every: int = 5) -> list[tuple]:
    """Pressure gives up gains and Cover 3 stops, with one exception in every ``flip_every`` snaps."""
    snaps = []
    for i in range(n):
        pressure = i % 2 == 0
        offense_wins = pressure != (i % flip_every == 0)
        call = ("4-3 Over", "Fire Zone") if pressure else ("Nickel Over", "Cover 3 Sky")
        snaps.append((*call, "+12" if offense_wins else "incomplete", "verified", *call))
    return snaps


class CollectRowsTest(unittest.TestCase):
    def test_provenance_and_admission(self) -> None:
        conn = _db([
            ("old", "jaxon", "2026-10-01T00:00:00+00:00", [
                ("Nickel Over", "Cover 3 Sky", "+12"),
                ("Nickel Over", "Cover 3 Sky", "+12", "verified", "4-3 Over", "Fire Zone"),
                ("Nickel Over", "Cover 3 Sky", "???"),
            ]),
            ("ryan", "ryan", "2026-11-03T17:00:00+00:00", [("4-3 Over", "Fire Zone", "incomplete")]),
            ("dillon", "dillon", "2026-11-05T21:00:00+00:00", [("4-3 Over", "Fire Zone", "td")]),
            ("cpu1", "cpu", "2026-11-05T22:00:00+00:00", [("4-3 Over", "Fire Zone", "td")]),
        ])
        rows = train.collect_rows(conn, attested={"dillon": "user confirmed"},
                                  unattested_after="2026-11-03T00:00:00+00:00")
        got = [(r.session_id, r.formation, r.play, r.provenance, r.admitted, r.excluded_reason) for r in rows]
        self.assertEqual(got, [
            ("old", "Nickel Over", "Cover 3 Sky", train.PROVENANCE_LOGGED, False, "execution_unverified"),
            ("old", "4-3 Over", "Fire Zone", train.PROVENANCE_VERIFIED, True, ""),
            ("ryan", "4-3 Over", "Fire Zone", train.PROVENANCE_LOGGED, False, "execution_unverified"),
            ("dillon", "4-3 Over", "Fire Zone", train.PROVENANCE_ATTESTED, False, "execution_unverified"),
        ])
        self.assertEqual([r.offense_success for r in rows], [True, True, False, True])

    def test_without_cutoff_unattested_logged_snap_stays_excluded(self) -> None:
        conn = _db([("ryan", "ryan", "2026-11-03T17:00:00+00:00", [("4-3 Over", "Fire Zone", "+12")])])
        (row,) = train.collect_rows(conn)
        self.assertFalse(row.admitted)
        self.assertEqual(row.excluded_reason, "execution_unverified")
        self.assertEqual(row.provenance, train.PROVENANCE_LOGGED)

    def test_summary_records_attestation_and_exclusions(self) -> None:
        conn = _db([
            ("ryan", "ryan", "2026-11-03T17:00:00+00:00", [("4-3 Over", "Fire Zone", "+12")]),
            ("dillon", "dillon", "2026-11-05T21:00:00+00:00", [("4-3 Over", "Fire Zone", "+12")]),
        ])
        # Historical comparison only: do not use this opt-out for model promotion.
        rows = train.collect_rows(conn, attested={"dillon": "confirmed"},
                                  unattested_after="2026-11-03", verified_only=False)
        summary = train.training_summary(rows, {"dillon": "confirmed"}, "2026-11-03", 10.0)
        self.assertEqual(summary["provenance"], {train.PROVENANCE_ATTESTED: 1})
        self.assertEqual(summary["sessions"]["dillon"]["attestation"], "confirmed")
        self.assertEqual(summary["sessions"]["ryan"]["excluded_reason"], "unattested_after_cutoff")


class FitTest(unittest.TestCase):
    def test_fit_is_a_valid_v2_artifact_that_learns_the_family_signal(self) -> None:
        conn = _db([(f"g{i}", f"opp{i}", f"2026-10-0{i + 1}", _signal_game()) for i in range(4)]
                   + [("g9", "opp9", "2026-10-09", [("Nickel Over", "Cover 3 Buzz", "incomplete")])])
        rows = train.collect_rows(conn)
        template = _template()
        labels = train.label_calls(rows, template)
        self.assertEqual(labels[("Nickel Over", "Cover 3 Buzz")], {"family": "cover3", "package": "nickel"})
        art = defense_v1.validate_artifact(train.fit_artifact(rows, labels, template, 1.0))
        self.assertIn({"formation": "Nickel Over", "play": "Cover 3 Buzz", "family": "cover3", "package": "nickel"},
                      art["candidates"])
        pressure = next(r for r in rows if r.play == "Fire Zone")
        cover3 = next(r for r in rows if r.play == "Cover 3 Sky")
        self.assertGreater(train.predict(art, pressure, labels), 0.6)
        self.assertLess(train.predict(art, cover3, labels), 0.4)

    def test_leave_one_game_out_beats_a_flat_current_and_passes_the_gate(self) -> None:
        conn = _db([(f"g{i}", f"opp{i}", f"2026-10-0{i + 1}", _signal_game()) for i in range(5)])
        rows = train.collect_rows(conn)
        template = _template()
        labels = train.label_calls(rows, template)
        flat = defense_v1.validate_artifact(template)
        report = train.leave_one_game_out(rows, labels, flat, template, current_in_sample={"g0"})
        self.assertEqual([g["current_in_sample"] for g in report["per_game"]], [True] + [False] * 4)
        self.assertEqual(report["current_out_of_sample"]["games"], 4)
        verdict = train.gate(report)
        self.assertEqual(verdict["result"], "PASSED", verdict)


class GateTest(unittest.TestCase):
    @staticmethod
    def _report(games: int, n: int, cand: float, cur: float, *, oos_cand: float = 0.5, oos_cur: float = 0.6) -> dict:
        def side(ll: float, nn: int) -> dict:
            return {"n": nn, "log_loss": ll, "brier": ll / 3, "auc": None}
        return {
            "all_games": {"games": games, "candidate": side(cand, n), "current": side(cur, n)},
            "current_out_of_sample": {"games": 1, "candidate": side(oos_cand, 10), "current": side(oos_cur, 10)},
        }

    def test_too_few_games_or_snaps_is_insufficient(self) -> None:
        self.assertEqual(train.gate(self._report(3, 100, 0.5, 0.7))["result"], "INSUFFICIENT")
        self.assertEqual(train.gate(self._report(5, 29, 0.5, 0.7))["result"], "INSUFFICIENT")

    def test_needs_the_full_log_loss_margin(self) -> None:
        self.assertEqual(train.gate(self._report(5, 100, 0.59, 0.60))["result"], "FAILED")
        self.assertEqual(train.gate(self._report(5, 100, 0.57, 0.60))["result"], "PASSED")

    def test_losing_on_games_current_never_saw_fails(self) -> None:
        report = self._report(5, 100, 0.50, 0.60, oos_cand=0.70, oos_cur=0.60)
        verdict = train.gate(report)
        self.assertEqual(verdict["result"], "FAILED")
        self.assertFalse(verdict["checks"]["unseen_games_not_worse"])


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "train_defense_logistic", ROOT / "scripts" / "madden_ml" / "train_defense_logistic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "m.db"
        disk = sqlite3.connect(self.db)
        _db([(f"g{i}", f"opp{i}", f"2026-10-0{i + 1}T00:00:00+00:00", _signal_game()) for i in range(4)]
            + [("late", "dillon", "2026-11-05T21:00:00+00:00", _signal_game())]).backup(disk)
        disk.close()
        self.current = self.tmp / "current.json"
        self.current.write_text(json.dumps(_template()), encoding="utf-8")
        self.script = _load_script()

    def _run(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return self.script.main(["--db", str(self.db), "--current", str(self.current), *extra])

    def test_writes_candidate_with_provenance_and_leaves_current_and_db_alone(self) -> None:
        before = (self.current.read_bytes(), self.db.read_bytes())
        out = self.tmp / "cand.json"
        self._run("--out", str(out), "--unattested-after", "2026-11-01",
                  "--attest", "late=user confirmed logged results",
                  "--current-trained-before", "2026-11-01")
        self.assertEqual((self.current.read_bytes(), self.db.read_bytes()), before)
        art = json.loads(out.read_text(encoding="utf-8"))
        defense_v1.validate_artifact(art)
        training = art["training"]
        self.assertEqual(training["provenance"], {train.PROVENANCE_VERIFIED: 100})
        self.assertEqual(training["sessions"]["late"]["attestation"], "user confirmed logged results")
        self.assertEqual(training["evaluation"]["current_in_sample_sessions"], ["g0", "g1", "g2", "g3"])
        self.assertIn(training["evaluation"]["gate"]["result"], ("PASSED", "FAILED", "INSUFFICIENT"))

    def test_refuses_unverified_only_database_without_creating_artifact(self) -> None:
        no_verified = self.tmp / "unverified.db"
        conn = sqlite3.connect(no_verified)
        _db([
            ("g1", "jack", "2026-10-01", [("Nickel Over", "Cover 3 Sky", "+12")]),
            ("g2", "jill", "2026-10-02", [("4-3 Over", "Fire Zone", "td")]),
        ]).backup(conn)
        conn.close()
        out = self.tmp / "should_not_exist.json"
        with self.assertRaisesRegex(SystemExit, "verified defensive executions"):
            self._run("--db", str(no_verified), "--out", str(out),
                      "--attest", "g2=I confirm the score")
        self.assertFalse(out.exists())

    def test_refuses_to_overwrite(self) -> None:
        with self.assertRaises(SystemExit):
            self._run("--out", str(self.current))
        existing = self.tmp / "exists.json"
        existing.write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self._run("--out", str(existing))
        self.assertEqual(existing.read_text(encoding="utf-8"), "{}")

    def test_unknown_attested_session_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._run("--out", str(self.tmp / "c.json"), "--attest", "nope=confirmed")
        self.assertFalse((self.tmp / "c.json").exists())


if __name__ == "__main__":
    unittest.main()
