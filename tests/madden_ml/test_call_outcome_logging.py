"""Synthetic Madden call/outcome logging.

These tests build databases in a temp directory. They do not read or commit
a real game, a database file, or an audit row.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import subprocess
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, render_live_html
from cfb_coach.madden.model import inference
from cfb_coach.madden.model.call_outcome_log import (
    LINKED,
    NO_RESULT,
    NO_SNAP_RECORDED,
    POST_SNAP_LOG_FIELDS,
    SCHEMA_META_KEY,
    SCHEMA_VERSION,
    SUPERSEDED,
    apply_seal_logging,
    playbook_version_of,
    record_session_run_config,
    yards_for_log,
)
from cfb_coach.madden.model.inference import resolve_mode
from cfb_coach.madden.model.schema import ML_LATENCY_BUDGET_MS, CoachingMode
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.madden.model.data_layer import SHADOW_FEATURE_FIELDS
from cfb_coach.madden.model.features import pre_snap_feature_names
from cfb_coach.madden.model.identity import LiveDecisionTracker
from cfb_coach.outcome import parse_outcome
from cfb_coach.situation import Situation, parse_situation
from cfb_coach.vod_model.flags import use_cli_flags

SESSION = "synthetic-cpu-001"


def _seed() -> dict:
    return {
        "opponents": {
            "cpu": {
                "display_name": "CPU",
                "team_now": "DET",
                "skill": "cpu",
                "confidence": "low",
            }
        }
    }


def _open(path: Path) -> CoachDB:
    return CoachDB(path, seed=_seed())


def _columns(path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _create_old_madden_db(path: Path) -> None:
    """A pre-call-outcome Madden file: decisions and snaps, no new columns."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('seeded', 'old');
        CREATE TABLE snaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            opponent_id TEXT NOT NULL,
            side TEXT NOT NULL,
            down INTEGER,
            distance INTEGER,
            yardline INTEGER,
            quarter INTEGER,
            situation_raw TEXT,
            our_call TEXT,
            formation TEXT,
            play TEXT,
            macro TEXT,
            result TEXT,
            coverage_seen TEXT,
            concept_seen TEXT,
            notes TEXT,
            session_id TEXT
        );
        CREATE TABLE ml_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_ts TEXT NOT NULL,
            game_id TEXT,
            snap_id TEXT,
            session_id TEXT,
            snap_seq INTEGER,
            mode TEXT NOT NULL,
            model_version TEXT,
            policy_version TEXT,
            policy_source TEXT,
            heuristic_formation TEXT,
            heuristic_play TEXT,
            final_formation TEXT,
            final_play TEXT,
            shadow_formation TEXT,
            shadow_play TEXT,
            shadow_status TEXT,
            agree INTEGER,
            candidates_json TEXT,
            decision_json TEXT NOT NULL,
            latency_ms_model REAL,
            created_ts TEXT NOT NULL
        );
        CREATE TABLE ml_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snap_id TEXT NOT NULL,
            game_id TEXT,
            decision_id INTEGER,
            executed_status TEXT,
            executed_formation TEXT,
            executed_play TEXT,
            executed_verification TEXT,
            outcome_json TEXT,
            created_ts TEXT NOT NULL
        );
        INSERT INTO snaps (
            ts, opponent_id, side, down, distance, yardline, quarter,
            situation_raw, our_call, formation, play, result, notes, session_id
        ) VALUES (
            '2026-01-01T00:00:00+00:00', 'cpu', 'offense', 1, 10, 25, 1,
            '1st and 10', 'Gun Bunch — Mesh', 'Gun Bunch', 'Mesh', 'TD',
            'score_us=7 score_them=0', 'old-session'
        );
        INSERT INTO ml_decisions (
            decision_ts, game_id, snap_id, session_id, snap_seq, mode,
            final_formation, final_play, decision_json, created_ts
        ) VALUES (
            '2026-01-01T00:00:00+00:00', 'old-session', 'old-session-0001',
            'old-session', 1, 'heuristic', 'Gun Bunch', 'Mesh',
            '{"final_pick":{"formation":"Gun Bunch","play":"Mesh"}}',
            '2026-01-01T00:00:00+00:00'
        );
        INSERT INTO ml_outcomes (
            snap_id, game_id, decision_id, executed_status, outcome_json, created_ts
        ) VALUES (
            'old-session-0001', 'old-session', 1, 'unknown',
            '{"result":"TD","yards":0}', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    conn.commit()
    conn.close()


def _result_for(index: int) -> str:
    return {
        0: "TD",
        1: "TD +8",
        2: "sack",
        3: "sack -6",
        4: "INT",
        5: "INT return 15",
        6: "+4",
    }[index % 7]


def _expected_yards(result: str) -> int | None:
    return yards_for_log(result)


def test_yards_unknown_stays_null_and_parser_placeholder_is_unchanged() -> None:
    assert parse_outcome("INT").yards == 0
    assert parse_outcome("fumble").yards == 0
    assert parse_outcome("incomplete").yards == 0
    assert yards_for_log("TD") is None
    assert yards_for_log("sack") is None
    assert yards_for_log("INT") is None
    assert yards_for_log("fumble") is None
    assert yards_for_log("incomplete") is None
    assert yards_for_log("stop") is None
    assert yards_for_log("TD +8") == 8
    assert yards_for_log("sack -6") == -6
    assert yards_for_log("INT return 15") == 15
    assert yards_for_log("sack 0") == 0


def test_post_snap_fields_stay_out_of_pre_snap_features() -> None:
    features = set(pre_snap_feature_names())
    assert POST_SNAP_LOG_FIELDS.isdisjoint(features)
    assert POST_SNAP_LOG_FIELDS.isdisjoint(set(SHADOW_FEATURE_FIELDS))
    assert "executed_adjustment_id" not in features
    assert "penalty" not in features
    assert "yards" not in features
    assert "behavior_propensity" not in features


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "madden27.db"
    db = _open(path)
    try:
        before = {table: _columns(path, table) for table in ("ml_decisions", "ml_outcomes", "snaps")}
        decision_id = db.log_ml_decision(
            {
                "decision_ts": "2026-01-01T00:00:00+00:00",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": f"{SESSION}-0001",
                "snap_seq": 1,
                "mode": "heuristic",
                "accepted": "unknown",
                "behavior_propensity": None,
                "propensity_method": "unknown",
                "propensity_override": False,
                "macro": {"macro_id": "HOT"},
            }
        )
        stored = dict(
            db.conn.execute("SELECT * FROM ml_decisions WHERE id = ?", (decision_id,)).fetchone()
        )
        db._migrate_madden_call_outcome()
        db._migrate_madden_call_outcome()
        again = dict(
            db.conn.execute("SELECT * FROM ml_decisions WHERE id = ?", (decision_id,)).fetchone()
        )
        assert db.get_meta(SCHEMA_META_KEY) == SCHEMA_VERSION
    finally:
        db.close()
    assert {table: _columns(path, table) for table in before} == before
    assert again == stored
    assert stored["accepted"] == "unknown"
    assert stored["behavior_propensity"] is None
    assert stored["recommended_adjustment_id"] == "HOT"
    assert "executed_adjustment_id" not in stored
    reopened = CoachDB.open_read_only(path)
    try:
        row = reopened.conn.execute(
            "SELECT accepted, recommended_adjustment_id FROM ml_decisions WHERE id = ?",
            (decision_id,),
        ).fetchone()
        assert row["accepted"] == "unknown"
        assert row["recommended_adjustment_id"] == "HOT"
    finally:
        reopened.close()


def test_old_db_upgrade_stays_readable(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    _create_old_madden_db(source)
    readable = tmp_path / "legacy-ro.db"
    shutil.copy(source, readable)
    reader = CoachDB.open_read_only(readable)
    try:
        snap = reader.conn.execute("SELECT result, notes, play FROM snaps").fetchone()
        decision = reader.conn.execute(
            "SELECT final_play, decision_json FROM ml_decisions"
        ).fetchone()
        outcome = reader.conn.execute("SELECT outcome_json FROM ml_outcomes").fetchone()
        assert snap["result"] == "TD"
        assert snap["notes"] == "score_us=7 score_them=0"
        assert snap["play"] == "Mesh"
        assert decision["final_play"] == "Mesh"
        assert "Mesh" in decision["decision_json"]
        assert json.loads(outcome["outcome_json"])["result"] == "TD"
        assert "yards" not in _columns(readable, "ml_outcomes")
    finally:
        reader.close()

    db = _open(source)
    try:
        snap = db.conn.execute("SELECT * FROM snaps").fetchone()
        decision = db.conn.execute("SELECT * FROM ml_decisions").fetchone()
        outcome = db.conn.execute("SELECT * FROM ml_outcomes").fetchone()
        assert snap["result"] == "TD"
        assert snap["play"] == "Mesh"
        assert snap["notes"] == "score_us=7 score_them=0"
        assert snap["yards"] is None
        assert snap["penalty"] is None
        assert snap["drive_id"] is None
        assert snap["score_for"] is None
        assert decision["final_play"] == "Mesh"
        assert decision["accepted"] is None
        assert decision["behavior_propensity"] is None
        assert decision["snap_link"] is None
        assert outcome["yards"] is None
        assert outcome["executed_adjustment_id"] is None
        assert json.loads(outcome["outcome_json"])["yards"] == 0
        db._migrate_madden_call_outcome()
        assert db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0] == 1
        assert db.conn.execute("SELECT COUNT(*) FROM ml_decisions").fetchone()[0] == 1
    finally:
        db.close()
    assert _columns(source, "ml_decisions").count("accepted") == 1
    assert _columns(source, "ml_outcomes").count("yards") == 1


def test_fifty_seven_calls_and_forty_two_snaps_leave_no_orphans(tmp_path: Path) -> None:
    path = tmp_path / "madden27.db"
    db = _open(path)
    try:
        for index in range(1, 58):
            db.log_ml_decision(
                {
                    "decision_ts": "2026-01-01T00:00:00+00:00",
                    "game_id": SESSION,
                    "session_id": SESSION,
                    "snap_id": f"{SESSION}-{index:04d}",
                    "snap_seq": index,
                    "mode": "heuristic",
                    "accepted": "true" if index % 5 == 0 else "unknown",
                    "behavior_propensity": 0.25 if index % 2 == 0 else None,
                    "propensity_method": "exact" if index % 2 == 0 else "unknown",
                    "propensity_override": False,
                    "drive_id": f"d{(index - 1) // 6 + 1}",
                    "quarter": ((index - 1) % 4) + 1,
                    "game_clock": f"{index % 15}:00",
                    "score_for": index % 5,
                    "score_against": index % 3,
                    "final_pick": {
                        "formation": "Gun Bunch",
                        "play": "Mesh",
                        "adjustment_id": "HOT",
                    },
                    "macro": {"macro_id": "HOT"},
                }
            )
        for index in range(1, 43):
            snap_id = f"{SESSION}-{index:04d}"
            result = "TD +8, penalty declined" if index == 8 else _result_for(index)
            confirmed = index == 11
            db.log_snap(
                opponent_id="cpu",
                side="offense",
                situation_raw="1st and 10",
                our_call="Gun Bunch — Mesh",
                formation="Gun Bunch",
                play="Mesh",
                result=result,
                session_id=SESSION,
                ml_snap_id=snap_id,
                snap_seq=index,
                quarter=((index - 1) % 4) + 1,
                drive_id=f"d{(index - 1) // 6 + 1}",
                game_clock=f"{index % 15}:00",
                score_for=index % 5,
                score_against=index % 3,
                recommended_adjustment_id="HOT",
                executed_adjustment_id="SLIDE" if confirmed else "HOT",
                adjustment_confirmed=confirmed,
            )
            decision_id = db.conn.execute(
                "SELECT id FROM ml_decisions WHERE snap_id = ?",
                (snap_id,),
            ).fetchone()[0]
            db.log_ml_outcome(
                snap_id=snap_id,
                game_id=SESSION,
                decision_id=int(decision_id),
                executed_status="identified" if confirmed else "unknown",
                executed_formation="Gun Bunch" if confirmed else None,
                executed_play="Mesh" if confirmed else None,
                executed_verification="verified" if confirmed else "unknown",
                outcome={"result": result, "yards": 0},
                recommended_adjustment_id="HOT",
                executed_adjustment_id="HOT" if not confirmed else "SLIDE",
                adjustment_confirmed=confirmed,
                drive_id=f"d{(index - 1) // 6 + 1}",
                quarter=((index - 1) % 4) + 1,
                game_clock=f"{index % 15}:00",
                score_for=index % 5,
                score_against=index % 3,
                penalty="declined" if index == 8 else None,
            )
        first = db.reconcile_call_links(SESSION)
        second = db.reconcile_call_links(SESSION)
        assert first == second
        assert first == {
            "calls": 57,
            "linked": 42,
            "no_snap_recorded": 15,
            "orphans": 0,
        }
        counts = dict(
            db.conn.execute(
                "SELECT snap_link, COUNT(*) FROM ml_decisions WHERE session_id = ? "
                "GROUP BY snap_link",
                (SESSION,),
            ).fetchall()
        )
        assert counts[LINKED] == 42
        assert counts[NO_SNAP_RECORDED] == 15
        assert None not in counts
        reasons = dict(
            db.conn.execute(
                "SELECT orphan_reason, COUNT(*) FROM ml_decisions WHERE session_id = ? "
                "AND snap_link = ? GROUP BY orphan_reason",
                (SESSION, NO_SNAP_RECORDED),
            ).fetchall()
        )
        assert reasons == {SUPERSEDED: 14, NO_RESULT: 1}
        linked_without_decision = db.conn.execute(
            """
            SELECT d.id FROM ml_decisions d
            WHERE d.session_id = ?
              AND NOT EXISTS (SELECT 1 FROM snaps s WHERE s.ml_decision_id = d.id)
              AND d.snap_link = ?
            """,
            (SESSION, LINKED),
        ).fetchall()
        assert linked_without_decision == []
        orphans = db.conn.execute(
            """
            SELECT d.id
            FROM ml_decisions d
            WHERE d.session_id = ?
              AND NOT (
                (
                  d.snap_link = 'no_snap_recorded'
                  AND NOT EXISTS (
                    SELECT 1 FROM snaps s WHERE s.ml_snap_id = d.snap_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM ml_outcomes o
                    WHERE o.snap_id = d.snap_id OR o.decision_id = d.id
                  )
                )
                OR EXISTS (SELECT 1 FROM snaps s WHERE s.ml_snap_id = d.snap_id)
                OR EXISTS (
                  SELECT 1 FROM ml_outcomes o
                  WHERE o.snap_id = d.snap_id OR o.decision_id = d.id
                )
              )
            """,
            (SESSION,),
        ).fetchall()
        assert orphans == []
        assert db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0] == 42
        assert db.conn.execute("SELECT COUNT(*) FROM ml_outcomes").fetchone()[0] == 42
        rows = db.conn.execute(
            """
            SELECT d.snap_seq, d.accepted, d.behavior_propensity, d.propensity_method,
                   d.recommended_adjustment_id, d.drive_id, d.quarter, d.game_clock,
                   d.score_for, d.score_against, d.snap_link,
                   o.yards, o.penalty, o.executed_adjustment_id, o.recommended_adjustment_id
                     AS outcome_recommended,
                   s.yards AS snap_yards, s.result
            FROM ml_decisions d
            LEFT JOIN ml_outcomes o ON o.snap_id = d.snap_id
            LEFT JOIN snaps s ON s.ml_snap_id = d.snap_id
            WHERE d.session_id = ?
            ORDER BY d.snap_seq
            """,
            (SESSION,),
        ).fetchall()
        assert len(rows) == 57
        for row in rows:
            assert row["drive_id"]
            assert row["quarter"] in {1, 2, 3, 4}
            assert row["game_clock"]
            assert row["score_for"] is not None
            assert row["score_against"] is not None
            assert row["accepted"] in {"true", "unknown"}
            assert row["propensity_method"] in {"exact", "unknown"}
            assert row["recommended_adjustment_id"] == "HOT"
            if row["snap_seq"] % 2 == 0:
                assert row["behavior_propensity"] == 0.25
            else:
                assert row["behavior_propensity"] is None
            if row["snap_link"] == NO_SNAP_RECORDED:
                assert row["yards"] is None
                assert row["result"] is None
                continue
            assert row["snap_link"] == LINKED
            assert row["yards"] == _expected_yards(row["result"])
            assert row["snap_yards"] == row["yards"]
            assert row["yards"] != 0 or row["result"] == "sack 0"
            assert row["outcome_recommended"] == "HOT"
            if row["snap_seq"] == 11:
                assert row["executed_adjustment_id"] == "SLIDE"
            else:
                assert row["executed_adjustment_id"] is None
            if row["snap_seq"] == 8:
                assert row["penalty"] == "declined"
                assert row["yards"] == 8
    finally:
        db.close()


def test_executed_adjustment_is_not_the_displayed_recommendation(tmp_path: Path) -> None:
    db = _open(tmp_path / "madden27.db")
    try:
        decision_id = db.log_ml_decision(
            {
                "decision_ts": "t",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": f"{SESSION}-0001",
                "snap_seq": 1,
                "mode": "heuristic",
                "final_pick": {
                    "formation": "Gun Bunch",
                    "play": "Mesh",
                    "adjustment_id": "HOT-MAN",
                },
                "macro": {"macro_id": "HOT-MAN"},
            }
        )
        db.log_ml_outcome(
            snap_id=f"{SESSION}-0001",
            game_id=SESSION,
            decision_id=decision_id,
            outcome={"result": "+4"},
            recommended_adjustment_id="HOT-MAN",
            executed_adjustment_id="HOT-MAN",
            adjustment_confirmed=False,
        )
        decision = db.conn.execute(
            "SELECT * FROM ml_decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        outcome = db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id = ?", (f"{SESSION}-0001",)
        ).fetchone()
        assert decision["recommended_adjustment_id"] == "HOT-MAN"
        assert "executed_adjustment_id" not in decision.keys()
        assert outcome["recommended_adjustment_id"] == "HOT-MAN"
        assert outcome["executed_adjustment_id"] is None
        db.log_ml_outcome(
            snap_id=f"{SESSION}-0001",
            game_id=SESSION,
            decision_id=decision_id,
            outcome={"result": "+4"},
            recommended_adjustment_id="HOT-MAN",
            executed_adjustment_id="SLIDE",
            adjustment_confirmed=True,
        )
        outcome = db.conn.execute(
            "SELECT executed_adjustment_id, recommended_adjustment_id FROM ml_outcomes"
        ).fetchone()
        assert outcome["recommended_adjustment_id"] == "HOT-MAN"
        assert outcome["executed_adjustment_id"] == "SLIDE"
    finally:
        db.close()


def test_superseded_call_without_a_snap_is_marked(tmp_path: Path) -> None:
    db = _open(tmp_path / "madden27.db")
    try:
        tracker = LiveDecisionTracker.from_session(SESSION)
        first_snap, _seq, _key, is_new = tracker.seal_call(
            side="offense", formation="Gun Bunch", play="Mesh", situation_raw="1st and 10"
        )
        assert is_new
        first_id = db.log_ml_decision(
            {
                "decision_ts": "t",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": first_snap,
                "snap_seq": 1,
                "mode": "heuristic",
                "accepted": "unknown",
                "propensity_method": "unknown",
                "behavior_propensity": None,
                "propensity_override": False,
            }
        )
        tracker.bind_decision_row(first_id)
        second_snap, seq, _key, is_new = tracker.seal_call(
            side="offense", formation="Gun Trips", play="Flood", situation_raw="2nd and 7"
        )
        assert is_new
        assert seq == 2
        assert tracker.abandoned_decision_row_id == first_id
        second_id = db.log_ml_decision(
            {
                "decision_ts": "t",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": second_snap,
                "snap_seq": 2,
                "mode": "heuristic",
                "accepted": "unknown",
                "propensity_method": "exact",
                "behavior_propensity": 0.5,
                "propensity_override": False,
            }
        )
        tracker.bind_decision_row(second_id)
        sit = Situation(raw="2nd and 7 drive id 4", side="offense")
        sit.score_us = 14
        sit.score_them = 7
        sit.extras["quarter"] = 2
        sit.extras["clock_seconds"] = 90
        sit.extras["drive_id"] = "4"

        class _Call:
            adjustment = {"id": "HOT"}
            macro = None

        apply_seal_logging(
            db, tracker, decision_id=second_id, sit=sit, call=_Call(), is_new=True,
        )
        first = db.conn.execute(
            "SELECT snap_link, orphan_reason FROM ml_decisions WHERE id = ?", (first_id,)
        ).fetchone()
        second = db.conn.execute(
            "SELECT snap_link, drive_id, quarter, game_clock, score_for, score_against, "
            "recommended_adjustment_id FROM ml_decisions WHERE id = ?",
            (second_id,),
        ).fetchone()
        assert first["snap_link"] == NO_SNAP_RECORDED
        assert first["orphan_reason"] == SUPERSEDED
        assert second["drive_id"] == "4"
        assert second["quarter"] == 2
        assert second["game_clock"] == "1:30"
        assert second["score_for"] == 14
        assert second["score_against"] == 7
        assert second["recommended_adjustment_id"] == "HOT"
        assert second["snap_link"] in {None, LINKED}
    finally:
        db.close()


def test_run_config_logs_mode_model_flags_commit_and_playbook(tmp_path: Path) -> None:
    path = tmp_path / "madden27.db"
    db = _open(path)
    try:
        assert db.get_meta("ml_mode") is None
        db.set_meta(
            "active_playbook_json",
            json.dumps(
                {
                    "applied": {
                        "offense": {"name": "PRIMARY META O", "mode": "custom", "rev": 3},
                        "defense": {"name": "PRIMARY META D", "mode": "custom", "rev": 2},
                    }
                }
            ),
        )
        db.set_meta("ml_model_version", "synthetic-model")
        with use_cli_flags(no_vod_prior=True, freeze_vod_book=False):
            record_session_run_config(db, SESSION)
            record_session_run_config(
                db,
                SESSION,
                flags={"no_vod_prior": False},
                mode="shadow",
                model_id="other-model",
            )
        assert db.get_meta("ml_mode") is None
        row = db.conn.execute(
            "SELECT * FROM ml_run_config WHERE session_id = ?", (SESSION,)
        ).fetchone()
        flags = json.loads(row["flags_json"])
        commit, branch = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(), subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        assert row["mode"] == "heuristic"
        assert row["model_id"] == "synthetic-model"
        assert flags["no_vod_prior"] is True
        assert "freeze_vod_book" in flags
        assert row["git_commit"] == commit
        assert row["git_branch"] == branch
        assert row["playbook_version"] == playbook_version_of(db)
        assert "rev3" in row["playbook_version"]
        assert "rev2" in row["playbook_version"]
    finally:
        db.close()


class _SyntheticCall:
    side = "offense"
    formation = "Gun Bunch"
    play = "Mesh"
    macro = None
    adj_or_macro = "base"
    rationale = "synthetic"
    reads = ()
    adjustments = ()

    def format(self) -> str:
        return "Gun Bunch — Mesh"

    def headline(self) -> str:
        return self.format()


def _shadow_hook(db: CoachDB):
    def hook(sit, call, *, game_id, snap_id, snap_seq, session_id, is_new):
        _decision, row_id = inference.evaluate_live_shadow(
            db=db,
            situation=sit,
            call=call,
            opponent_id="cpu",
            game_id=game_id,
            snap_id=snap_id,
            snap_seq=snap_seq,
            session_id=session_id,
            formations={"Gun Bunch": ["Mesh", "Flood"]},
            run=is_new,
        )
        return row_id

    return hook


def _live(db: CoachDB) -> LivePlayController:
    inference.set_mode(db, CoachingMode.SHADOW)
    ctrl = LivePlayController(
        db=db,
        opponent_id="cpu",
        make_call=lambda sit, **_kwargs: _SyntheticCall(),
        parse_situation=parse_situation,
        learn_summary=lambda: "synthetic",
        brand="Madden 27 Franchise",
        play_cmd="x",
        cpu_only=True,
        shadow_evaluate=_shadow_hook(db),
        enable_execution_verify=True,
    )
    ctrl.start()
    return ctrl


def _lines(count: int) -> list[str]:
    return [f"{(i % 4) + 1}&{10 - (i % 9)} my {25 + (i % 40)}" for i in range(count)]


def _decision_audit(db: CoachDB, session_id: str) -> list:
    return db.conn.execute(
        """
        SELECT d.id, d.snap_seq, d.orphan_reason, d.snap_link,
               (SELECT s.id FROM snaps s WHERE s.ml_decision_id = d.id) AS snap_row
        FROM ml_decisions d
        WHERE d.session_id = ? OR d.game_id = ?
        ORDER BY d.snap_seq, d.id
        """,
        (session_id, session_id),
    ).fetchall()


def _play_game(ctrl: LivePlayController, snaps: int, rerequests: int) -> None:
    """Seal ``snaps`` results. Re-request the first ``rerequests`` before the result.

    An idempotent replay of the opening call must not add a second decision.
    """
    lines = _lines(snaps + 1)
    key = f"seal-{ctrl.session_id}"
    opened = ctrl.call_only(lines[0], request_key=key)
    assert opened["ok"] is True
    replay = ctrl.call_only(lines[0], request_key=key)
    assert replay.get("idempotent_replay") is True
    for index in range(snaps):
        if index < rerequests:
            revised = ctrl.call_only(lines[index] + " showing cover 3")
            assert revised["ok"] is True
        logged = ctrl.result_and_call(
            outcome="+6" if index % 2 == 0 else "incomplete",
            sit_raw=lines[index + 1],
            side="offense",
        )
        assert logged["ok"] is True
        assert logged["logged"] is not None


def test_rerequested_calls_have_no_snap_until_marked(tmp_path: Path) -> None:
    """Two synthetic games: 14 of 63 and 15 of 57 decisions have no matching snap.

    A call is logged when it is sealed. The snap is written only when a result
    arrives, and only for the call still pending. Repeating call-only with a
    new situation supersedes the previous decision. The last sealed call has
    no result. Neither row is deleted. Each one stores why it has no snap.
    The default mode stays heuristic and the model budget stays 150 ms.
    """
    assert ML_LATENCY_BUDGET_MS == 150
    path = tmp_path / "madden27.db"
    default = _open(path)
    try:
        assert inference.resolve_mode(default) is CoachingMode.HEURISTIC
    finally:
        default.close()

    db = _open(path)
    try:
        shapes = (
            {"decisions": 63, "snaps": 49, "superseded": 13},
            {"decisions": 57, "snaps": 42, "superseded": 14},
        )
        for shape in shapes:
            ctrl = _live(db)
            _play_game(ctrl, shape["snaps"], shape["superseded"])
            session_id = ctrl.session_id
            before = _decision_audit(db, session_id)
            assert len(before) == shape["decisions"]
            unmatched = [row for row in before if row["snap_row"] is None]
            assert len(unmatched) == shape["superseded"] + 1
            assert sum(row["orphan_reason"] == SUPERSEDED for row in unmatched) == shape["superseded"]
            assert sum(row["orphan_reason"] is None for row in unmatched) == 1
            kept = [row["id"] for row in before]
            ended = ctrl.end_game(result_wl="win", score="1-0")
            assert ended["ok"] is True
            after = _decision_audit(db, session_id)
            assert [row["id"] for row in after] == kept
            unmatched = [row for row in after if row["snap_row"] is None]
            assert len(unmatched) == shape["superseded"] + 1
            assert sum(row["orphan_reason"] == SUPERSEDED for row in unmatched) == shape["superseded"]
            assert sum(row["orphan_reason"] == NO_RESULT for row in unmatched) == 1
            assert all(row["snap_link"] == NO_SNAP_RECORDED for row in unmatched)
            executed = [row for row in after if row["snap_row"] is not None]
            assert len(executed) == shape["snaps"]
            assert all(
                row["snap_link"] == LINKED and row["orphan_reason"] is None for row in executed
            )
            silent = [
                row["id"]
                for row in after
                if row["snap_row"] is None and row["orphan_reason"] not in {SUPERSEDED, NO_RESULT}
            ]
            assert silent == []
        untouched = _open(tmp_path / "untouched.db")
        try:
            assert inference.resolve_mode(untouched) is CoachingMode.HEURISTIC
        finally:
            untouched.close()
        assert ML_LATENCY_BUDGET_MS == 150
    finally:
        db.close()


def test_timeout_call_is_kept_and_an_executed_one_gets_the_snap(tmp_path: Path) -> None:
    """A timeout does not drop the row or skip the 150 ms fallback.

    The executed timeout is the sealed call, so its snap points at that
    decision. A timeout that was replaced by a later call stays in the log
    as superseded. A second log of the same snap id updates that row.
    """
    assert ML_LATENCY_BUDGET_MS == 150
    db = _open(tmp_path / "madden27.db")
    try:
        assert inference.resolve_mode(db) is CoachingMode.HEURISTIC

        def decision(seq: int, status: str) -> int:
            return db.log_ml_decision(
                {
                    "decision_ts": "t",
                    "game_id": SESSION,
                    "session_id": SESSION,
                    "snap_id": f"{SESSION}-{seq:04d}",
                    "snap_seq": seq,
                    "mode": "heuristic",
                    "shadow_status": status,
                    "final_pick": {"formation": "Gun Bunch", "play": "Mesh"},
                }
            )

        timed_out = decision(1, "timeout")
        followed = decision(2, "ok")
        db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="2nd and 7",
            our_call="Gun Bunch — Mesh",
            result="+4",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0002",
            snap_seq=2,
            ml_decision_id=timed_out,
        )
        executed_timeout = decision(3, "timeout")
        db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1st and 10",
            our_call="Gun Bunch — Mesh",
            result="incomplete",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0003",
            snap_seq=3,
        )
        retry = decision(3, "timeout")
        assert retry == executed_timeout
        snap_two = db.conn.execute(
            "SELECT ml_decision_id FROM snaps WHERE ml_snap_id = ?",
            (f"{SESSION}-0002",),
        ).fetchone()
        snap_three = db.conn.execute(
            "SELECT ml_decision_id FROM snaps WHERE ml_snap_id = ?",
            (f"{SESSION}-0003",),
        ).fetchone()
        assert snap_two["ml_decision_id"] == followed
        assert snap_three["ml_decision_id"] == executed_timeout
        before = [
            row[0]
            for row in db.conn.execute("SELECT id FROM ml_decisions ORDER BY id")
        ]
        stats = db.reconcile_call_links(SESSION)
        after = [
            row[0]
            for row in db.conn.execute("SELECT id FROM ml_decisions ORDER BY id")
        ]
        assert before == after
        assert len(after) == 3
        assert stats["orphans"] == 0
        replaced = db.conn.execute(
            "SELECT snap_link, orphan_reason FROM ml_decisions WHERE id = ?",
            (timed_out,),
        ).fetchone()
        kept = db.conn.execute(
            "SELECT snap_link, orphan_reason FROM ml_decisions WHERE id = ?",
            (executed_timeout,),
        ).fetchone()
        assert replaced["snap_link"] == NO_SNAP_RECORDED
        assert replaced["orphan_reason"] == SUPERSEDED
        assert kept["snap_link"] == LINKED
        assert kept["orphan_reason"] is None
        assert ML_LATENCY_BUDGET_MS == 150
        assert inference.resolve_mode(db) is CoachingMode.HEURISTIC
    finally:
        db.close()


def test_update_snap_does_not_restore_a_stale_decision_id(tmp_path: Path) -> None:
    """A later update_snap cannot put a stale decision back on an owned snap.

    The snap id's sealed decision wins, including when the caller passes None.
    A snap that has no sealed owner still stores the id it was given.
    """
    db = _open(tmp_path / "madden27.db")
    try:

        def decision(seq: int, play: str) -> int:
            return db.log_ml_decision(
                {
                    "decision_ts": "t",
                    "game_id": SESSION,
                    "session_id": SESSION,
                    "snap_id": f"{SESSION}-{seq:04d}",
                    "snap_seq": seq,
                    "mode": "heuristic",
                    "final_pick": {"formation": "Gun Bunch", "play": play},
                }
            )

        owner = decision(1, "Mesh")
        stale = decision(2, "Flood")
        row = db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1st and 10",
            our_call="Gun Bunch — Mesh",
            result="+4",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0001",
            snap_seq=1,
            executed_play="Mesh",
            ml_decision_id=stale,
        )
        db.update_snap(row, ml_decision_id=stale)
        linked = db.conn.execute(
            "SELECT ml_decision_id FROM snaps WHERE id = ?", (row,)
        ).fetchone()
        assert linked["ml_decision_id"] == owner
        db.update_snap(row, ml_decision_id=None)
        cleared = db.conn.execute(
            "SELECT ml_decision_id FROM snaps WHERE id = ?", (row,)
        ).fetchone()
        assert cleared["ml_decision_id"] == owner

        loose = db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="2nd and 7",
            our_call="Gun Bunch — Flood",
            result="incomplete",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0099",
            snap_seq=99,
        )
        db.update_snap(loose, ml_decision_id=stale)
        accepted = db.conn.execute(
            "SELECT ml_decision_id FROM snaps WHERE id = ?", (loose,)
        ).fetchone()
        assert accepted["ml_decision_id"] == stale
    finally:
        db.close()


def test_repoint_flags_executed_play_that_differs_from_final_play(
    tmp_path: Path, caplog
) -> None:
    """A repointed snap whose executed play differs from final_play is flagged.

    The comparison uses the sealed owner, not the stale id that was passed.
    An unknown executed play is not a mismatch. A later correction clears the flag.
    """
    caplog.set_level(logging.WARNING, logger="cfb_coach.db")
    db = _open(tmp_path / "madden27.db")
    try:
        owner = db.log_ml_decision(
            {
                "decision_ts": "t",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": f"{SESSION}-0001",
                "snap_seq": 1,
                "mode": "heuristic",
                "final_pick": {"formation": "Gun Bunch", "play": "Mesh"},
            }
        )
        stale = db.log_ml_decision(
            {
                "decision_ts": "t",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": f"{SESSION}-0002",
                "snap_seq": 2,
                "mode": "heuristic",
                "final_pick": {"formation": "Gun Bunch", "play": "Flood"},
            }
        )
        row = db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1st and 10",
            our_call="Gun Bunch — Mesh",
            result="+4",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0001",
            snap_seq=1,
            executed_play="Flood",
            ml_decision_id=stale,
        )
        flagged = db.conn.execute(
            "SELECT ml_decision_id, executed_play, play_mismatch FROM snaps WHERE id = ?",
            (row,),
        ).fetchone()
        assert flagged["ml_decision_id"] == owner
        assert flagged["executed_play"] == "Flood"
        assert flagged["play_mismatch"] == 1
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cfb_coach.db" and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert "play mismatch" in warnings[0]
        assert "executed_play=Flood" in warnings[0]
        assert "final_play=Mesh" in warnings[0]

        db.update_snap(row, executed_play="Mesh")
        matched = db.conn.execute(
            "SELECT play_mismatch FROM snaps WHERE id = ?", (row,)
        ).fetchone()
        assert matched["play_mismatch"] == 0
        assert len(warnings) == 1

        unknown_owner = db.log_ml_decision(
            {
                "decision_ts": "t",
                "game_id": SESSION,
                "session_id": SESSION,
                "snap_id": f"{SESSION}-0003",
                "snap_seq": 3,
                "mode": "heuristic",
                "final_pick": {"formation": "Gun Bunch", "play": "Mesh"},
            }
        )
        unknown = db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="3rd and 4",
            our_call="Gun Bunch — Mesh",
            result="incomplete",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0003",
            snap_seq=3,
            ml_decision_id=unknown_owner,
        )
        blank = db.conn.execute(
            "SELECT ml_decision_id, executed_play, play_mismatch FROM snaps WHERE id = ?",
            (unknown,),
        ).fetchone()
        assert blank["ml_decision_id"] == unknown_owner
        assert blank["executed_play"] is None
        assert blank["play_mismatch"] is None
        still = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cfb_coach.db" and record.levelno == logging.WARNING
        ]
        assert still == warnings
    finally:
        db.close()


class _LiveCall:
    def __init__(self, adjustment_id: str | None):
        self.side = "offense"
        self.formation = "Gun Bunch"
        self.play = "Mesh"
        self.macro = None
        self.adjustment = (
            {"id": adjustment_id, "label": adjustment_id} if adjustment_id else None
        )

    def format(self) -> str:
        return "Gun Bunch — Mesh"


def _live(tmp_path: Path, adjustment_id: str | None):
    db = _open(tmp_path / "madden27.db")
    ctrl = LivePlayController(
        db=db,
        opponent_id="cpu",
        make_call=lambda sit, **_kwargs: _LiveCall(adjustment_id),
        parse_situation=parse_madden_situation,
        learn_summary=lambda: "",
        brand="Madden 27 Franchise",
        cpu_only=True,
        enable_execution_verify=True,
        session_id=SESSION,
    )
    return db, ctrl


def _pair(db: CoachDB):
    snap = db.conn.execute("SELECT * FROM snaps ORDER BY id DESC LIMIT 1").fetchone()
    outcome = db.conn.execute(
        "SELECT * FROM ml_outcomes WHERE snap_id = ?",
        (snap["ml_snap_id"],),
    ).fetchone()
    return snap, outcome


def test_update_snap_cannot_write_executed_adjustment_id(tmp_path: Path) -> None:
    db = _open(tmp_path / "madden27.db")
    try:
        row_id = db.log_snap(
            opponent_id="cpu",
            side="offense",
            situation_raw="1st and 10",
            our_call="Gun Bunch — Mesh",
            result="+4",
            session_id=SESSION,
            ml_snap_id=f"{SESSION}-0001",
            recommended_adjustment_id="HOT",
            executed_adjustment_id="SLIDE",
            adjustment_confirmed=False,
        )
        db.update_snap(
            row_id,
            result="+5",
            executed_adjustment_id="SLIDE",
            accepted="true",
        )
        snap = db.conn.execute("SELECT * FROM snaps WHERE id = ?", (row_id,)).fetchone()
        assert snap["result"] == "+5"
        assert snap["accepted"] == "true"
        assert snap["recommended_adjustment_id"] == "HOT"
        assert snap["executed_adjustment_id"] is None
    finally:
        db.close()


def test_live_loop_confirmed_adjustment_is_logged(tmp_path: Path) -> None:
    db, ctrl = _live(tmp_path, "SLIDE")
    try:
        assert resolve_mode(db) is CoachingMode.HEURISTIC
        assert ML_LATENCY_BUDGET_MS == 150
        ctrl.call_only(
            "1&10 my 25",
            game_clock="12:00",
            quarter=2,
            score_us=7,
            score_them=0,
            score_set=True,
        )
        logged = ctrl.result_and_call(
            outcome="+8",
            sit_raw="2&2 my 33",
            game_clock="11:20",
            quarter=2,
            score_us=7,
            score_them=0,
            score_set=True,
            executed_status="used_recommended",
            applied_recommended_action=True,
            verify_touched=True,
        )
        assert logged["ok"]
        snap, outcome = _pair(db)
        for row in (snap, outcome):
            assert row["accepted"] == "true"
            assert row["recommended_adjustment_id"] == "SLIDE"
            assert row["executed_adjustment_id"] == "SLIDE"
            assert row["drive_id"] is None
            assert row["game_clock"] == "12:00"
        assert resolve_mode(db) is CoachingMode.HEURISTIC
    finally:
        db.close()


def test_live_loop_unconfirmed_adjustment_stays_a_recommendation(tmp_path: Path) -> None:
    db, ctrl = _live(tmp_path, "SLIDE")
    try:
        ctrl.call_only("1&10 my 25", game_clock="9:00", quarter=1)
        logged = ctrl.result_and_call(
            outcome="+4",
            sit_raw="2&6 my 29",
            quarter=1,
            executed_status="unknown",
            applied_recommended_action=False,
        )
        assert logged["ok"]
        snap, outcome = _pair(db)
        for row in (snap, outcome):
            assert row["accepted"] == "unknown"
            assert row["recommended_adjustment_id"] == "SLIDE"
            assert row["executed_adjustment_id"] is None
            assert row["drive_id"] is None
            assert row["game_clock"] == "9:00"
    finally:
        db.close()


def test_live_loop_without_a_recommendation_leaves_execution_empty(tmp_path: Path) -> None:
    db, ctrl = _live(tmp_path, None)
    try:
        ctrl.call_only("1&10 my 40 clock 5:15", quarter=3)
        logged = ctrl.result_and_call(
            outcome="+3",
            sit_raw="2&7 my 43",
            quarter=3,
            executed_status="used_recommended",
            applied_recommended_action=True,
            verify_touched=True,
        )
        assert logged["ok"]
        snap, outcome = _pair(db)
        for row in (snap, outcome):
            assert row["recommended_adjustment_id"] is None
            assert row["executed_adjustment_id"] is None
            assert row["accepted"] == "true"
            assert row["drive_id"] is None
            assert row["game_clock"] == "5:15"
    finally:
        db.close()


def test_live_loop_fills_snap_and_outcome_fields(tmp_path: Path) -> None:
    db, ctrl = _live(tmp_path, "HOT")
    try:
        assert db.get_meta("ml_mode") is None
        ctrl.call_only(
            "1&10 my 25",
            game_clock="8:00",
            quarter=4,
            score_us=14,
            score_them=7,
            score_set=True,
        )
        first = ctrl.result_and_call(
            outcome="td",
            sit_raw="1&10 my 25",
            game_clock="6:30",
            quarter=4,
            score_us=14,
            score_them=7,
            score_set=True,
            executed_status="used_recommended",
            applied_recommended_action=True,
            verify_touched=True,
        )
        second = ctrl.result_and_call(
            outcome="+4",
            sit_raw="2&6 my 40",
            game_clock="6:00",
            quarter=4,
            score_us=20,
            score_them=7,
            score_set=True,
            executed_status="used_recommended",
            applied_recommended_action=True,
            verify_touched=True,
        )
        assert first["ok"] and second["ok"]
        rows = db.conn.execute(
            """
            SELECT s.drive_id, s.game_clock, s.accepted, s.recommended_adjustment_id,
                   s.executed_adjustment_id, s.result,
                   o.drive_id AS outcome_drive, o.game_clock AS outcome_clock,
                   o.accepted AS outcome_accepted,
                   o.recommended_adjustment_id AS outcome_recommended,
                   o.executed_adjustment_id AS outcome_executed
            FROM snaps s
            JOIN ml_outcomes o ON o.snap_id = s.ml_snap_id
            ORDER BY s.id
            """
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["result"] == "td"
        assert rows[1]["result"] == "+4"
        for row in rows:
            assert row["accepted"] == "true"
            assert row["outcome_accepted"] == "true"
            assert row["recommended_adjustment_id"] == "HOT"
            assert row["outcome_recommended"] == "HOT"
            assert row["executed_adjustment_id"] == "HOT"
            assert row["outcome_executed"] == "HOT"
            assert row["game_clock"]
            assert row["outcome_drive"] == row["drive_id"]
            assert row["outcome_clock"] == row["game_clock"]
        assert rows[0]["drive_id"] is None
        assert rows[0]["outcome_drive"] is None
        assert rows[0]["game_clock"] == "8:00"
        assert rows[1]["drive_id"] is None
        assert rows[1]["outcome_drive"] is None
        assert rows[1]["game_clock"] == "6:30"
        assert resolve_mode(db) is CoachingMode.HEURISTIC
        assert ML_LATENCY_BUDGET_MS == 150
    finally:
        db.close()


def test_live_loop_without_an_explicit_drive_stores_null(tmp_path: Path) -> None:
    db, ctrl = _live(tmp_path, "HOT")
    try:
        ctrl.call_only("1&10 my 25", game_clock="12:00", quarter=1)
        missing = ctrl.result_and_call(
            outcome="+4",
            sit_raw="2&6 my 29 drive id 4",
            quarter=1,
            game_clock="11:00",
            executed_status="used_recommended",
            verify_touched=True,
        )
        explicit = ctrl.result_and_call(
            outcome="+2",
            sit_raw="3&4 my 31",
            quarter=1,
            executed_status="unknown",
        )
        assert missing["ok"] and explicit["ok"]
        rows = db.conn.execute(
            """
            SELECT s.drive_id, o.drive_id AS outcome_drive
            FROM snaps s
            JOIN ml_outcomes o ON o.snap_id = s.ml_snap_id
            ORDER BY s.id
            """
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["drive_id"] is None
        assert rows[0]["outcome_drive"] is None
        assert rows[1]["drive_id"] == "4"
        assert rows[1]["outcome_drive"] == "4"
    finally:
        db.close()


def test_plain_submit_without_touching_verify_stores_unknown(tmp_path: Path) -> None:
    db, ctrl = _live(tmp_path, "SLIDE")
    try:
        html = render_live_html(ctrl)
        assert 'value="used_recommended"' in html
        assert 'value="used_recommended" checked' not in html
        assert "let execVerifyTouched = false;" in html
        assert 'const exec = execVerifyTouched && checked ? checked.value : "unknown";' in html
        assert "verify_touched: execVerifyTouched" in html
        ctrl.call_only("1&10 my 25", game_clock="8:00", quarter=2)
        logged = ctrl.result_and_call(
            outcome="+5",
            sit_raw="2&5 my 30",
            quarter=2,
            executed_status="used_recommended",
            applied_recommended_action=False,
            verify_touched=False,
        )
        assert logged["ok"]
        snap, outcome = _pair(db)
        for row in (snap, outcome):
            assert row["accepted"] == "unknown"
            assert row["recommended_adjustment_id"] == "SLIDE"
            assert row["executed_adjustment_id"] is None
    finally:
        db.close()
