"""Synthetic fixtures for the path-explicit data layer.

These tests build old-schema databases in a temp directory. They do not read
or commit a real Madden database.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden.model.cli import cmd_ml_four_user_games
from cfb_coach.madden.model.data_layer import (
    SHADOW_FEATURE_DENYLIST,
    backup_manifest_path,
    migrate_database,
    export_shadow,
)
from cfb_coach.madden.prep import _opponent_study

# Post-snap labels and outcomes that must not appear as shadow features.
EXPLICIT_FEATURE_DENYLIST = frozenset(
    {
        "yards",
        "result",
        "touchdown",
        "sack",
        "interception",
        "fumble",
        "explosive",
        "first_down",
        "stop",
        "incompletion",
        "drive_result",
        "observed_coverage",
        "observed_concept",
        "executed_play",
        "executed",
        "executed_formation",
        "executed_status",
        "executed_verification",
        "executed_macro",
        "success",
        "success_probability",
        "parsed",
        "coverage_seen",
        "concept_seen",
        "action_play",
        "action_formation",
        "label_available",
        "accepted",
        "outcome",
        "yards_gained",
    }
)

ORIGINAL_SNAP_COLUMNS = (
    "id",
    "ts",
    "opponent_id",
    "side",
    "down",
    "distance",
    "yardline",
    "quarter",
    "situation_raw",
    "our_call",
    "formation",
    "play",
    "macro",
    "result",
    "coverage_seen",
    "concept_seen",
    "notes",
    "session_id",
)

# October 8 audit shape: 245 recommendation-only human snaps, four games.
OCTOBER8_GAMES = (
    ("gavin", "gavin-2026-10-08", 65),
    ("james", "james-2026-10-08", 71),
    ("jaxon", "jaxon-2026-10-08", 54),
    ("tiano", "tiano-2026-10-08", 55),
)


def _create_old_schema(path: Path, snaps: list[dict]) -> None:
    """Old coach snapshot: no ML columns and no ml_decisions/ml_outcomes."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
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
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE play_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            play_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            opponent_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE vision_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO snaps (
            ts, opponent_id, side, down, distance, yardline, quarter,
            situation_raw, our_call, formation, play, macro, result,
            coverage_seen, concept_seen, notes, session_id
        ) VALUES (
            :ts, :opponent_id, :side, :down, :distance, :yardline, :quarter,
            :situation_raw, :our_call, :formation, :play, :macro, :result,
            :coverage_seen, :concept_seen, :notes, :session_id
        )
        """,
        snaps,
    )
    conn.commit()
    conn.close()


def _snap(
    *,
    opponent_id: str,
    session_id: str,
    side: str,
    play: str,
    formation: str,
    coverage_seen: str | None = None,
    seq: int = 1,
) -> dict:
    return {
        "ts": f"2026-10-08T12:{seq % 60:02d}:00+00:00",
        "opponent_id": opponent_id,
        "side": side,
        "down": 2,
        "distance": 7,
        "yardline": 45,
        "quarter": 2,
        "situation_raw": "2nd & 7",
        "our_call": f"{formation} — {play}",
        "formation": formation,
        "play": play,
        "macro": None,
        "result": None,
        "coverage_seen": coverage_seen,
        "concept_seen": None,
        "notes": None,
        "session_id": session_id,
    }


def _october8_snaps() -> list[dict]:
    snaps: list[dict] = []
    seq = 0
    for opponent_id, session_id, count in OCTOBER8_GAMES:
        for index in range(count):
            seq += 1
            side = "offense" if index % 2 == 0 else "defense"
            snaps.append(
                _snap(
                    opponent_id=opponent_id,
                    session_id=session_id,
                    side=side,
                    play=f"Rec-{opponent_id}-{index}",
                    formation="Gun Bunch" if side == "offense" else "Nickel Over",
                    coverage_seen="Cover 3" if index % 5 == 0 else None,
                    seq=seq,
                )
            )
    return snaps


def _original_checksum(path: Path) -> str:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    present = [row[1] for row in conn.execute("PRAGMA table_info(snaps)")]
    columns = [name for name in ORIGINAL_SNAP_COLUMNS if name in present]
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM snaps ORDER BY id"
    ).fetchall()
    payload = json.dumps([list(row) for row in rows], separators=(",", ":")).encode()
    conn.close()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_names(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    conn.close()
    return names


def _reader_snapshot(path: Path) -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cmd_ml_four_user_games(
            Namespace(db=str(path), games=4, train=False, out=None)
        )
    assert code == 0
    db = CoachDB.open_read_only(path)
    try:
        prep = {
            opponent: _opponent_study(db, opponent)
            for opponent, _session, _count in OCTOBER8_GAMES
        }
    finally:
        db.close()
    return {"four_user_games": json.loads(buffer.getvalue()), "prep": prep}


def test_old_schema_migration_is_idempotent_and_preserves_row_checksums(tmp_path):
    source = tmp_path / "old.db"
    snaps = [
        _snap(
            opponent_id="gavin",
            session_id="gavin-2026-10-08",
            side="offense",
            play="Mesh",
            formation="Gun Bunch",
            seq=1,
        ),
        _snap(
            opponent_id="james",
            session_id="james-2026-10-08",
            side="defense",
            play="Cover 3 Sky",
            formation="Nickel Over",
            coverage_seen="Cover 2",
            seq=2,
        ),
    ]
    _create_old_schema(source, snaps)
    before = _original_checksum(source)
    assert "ml_decisions" not in _table_names(source)

    migrate_database(source, tmp_path / "first.db")
    after_first = _original_checksum(source)
    migrated_bytes = _file_sha256(source)

    second_backup = tmp_path / "second.db"
    migrate_database(source, second_backup)

    assert _original_checksum(source) == before == after_first
    assert _file_sha256(source) == migrated_bytes
    assert not second_backup.exists()
    assert not backup_manifest_path(second_backup).exists()

    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT play, formation, our_call, executed_play, executed_formation, "
        "executed_status, executed_macro, executed_verification, accepted "
        "FROM snaps ORDER BY id"
    ).fetchall()
    conn.close()
    assert [row["play"] for row in rows] == ["Mesh", "Cover 3 Sky"]
    for row in rows:
        assert row["executed_play"] in (None, "unknown")
        assert row["executed_play"] != row["play"]
        assert row["executed_formation"] != row["formation"]
        assert row["executed_status"] in (None, "unknown")
        assert row["accepted"] is None


def test_opponent_look_check_constraints_accept_empty_and_null(tmp_path):
    source = tmp_path / "checks.db"
    _create_old_schema(
        source,
        [_snap(opponent_id="gavin", session_id="g1", side="offense", play="Mesh", formation="Gun Bunch")],
    )
    migrate_database(source, tmp_path / "checks.backup.db")
    conn = sqlite3.connect(source)

    def insert(**columns: str | None) -> None:
        payload = {
            "ts": "2026-10-08T00:00:00+00:00",
            "opponent_id": "gavin",
            "side": "offense",
            "opp_personnel": None,
            "opp_set": None,
            "opp_look_verification": None,
        }
        payload.update(columns)
        conn.execute(
            """
            INSERT INTO snaps (
                ts, opponent_id, side, opp_personnel, opp_set, opp_look_verification
            ) VALUES (
                :ts, :opponent_id, :side, :opp_personnel, :opp_set, :opp_look_verification
            )
            """,
            payload,
        )

    insert(opp_set="empty", opp_personnel="11", opp_look_verification="unverified")
    insert(opp_set=None, opp_personnel=None, opp_look_verification=None)
    conn.commit()
    empty_id, null_id = conn.execute(
        "SELECT id FROM snaps WHERE opp_set = 'empty' OR id IN ("
        "SELECT id FROM snaps WHERE opp_set IS NULL ORDER BY id DESC LIMIT 1)"
    ).fetchall()
    stored = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT id, opp_set FROM snaps WHERE id IN (?, ?)",
            (empty_id[0], null_id[0]),
        )
    }
    assert "empty" in stored.values()
    assert None in stored.values()

    for column, bad in (
        ("opp_personnel", "not-a-personnel"),
        ("opp_set", "not-a-set"),
        ("opp_look_verification", "guessed"),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            insert(**{column: bad})
        conn.rollback()
    conn.close()


def test_backup_manifest_precedes_mutation_and_failure_aborts(tmp_path):
    source = tmp_path / "live.db"
    _create_old_schema(
        source,
        [_snap(opponent_id="tiano", session_id="tiano-1", side="defense", play="Cover 4", formation="Dime")],
    )
    before = _file_sha256(source)
    backup = tmp_path / "before-mutation.db"
    migrate_database(source, backup)

    manifest = json.loads(backup_manifest_path(backup).read_text(encoding="utf-8"))
    assert manifest["sha256"] == _file_sha256(backup)
    assert manifest["backup"] == backup.name
    backup_conn = sqlite3.connect(backup)
    backup_columns = {row[1] for row in backup_conn.execute("PRAGMA table_info(snaps)")}
    backup_tables = {
        row[0]
        for row in backup_conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    backup_conn.close()
    assert "ml_snap_id" not in backup_columns
    assert "executed_play" not in backup_columns
    assert "opp_set" not in backup_columns
    assert "ml_decisions" not in backup_tables
    assert "ml_outcomes" not in backup_tables
    assert "ml_decisions" in _table_names(source)
    assert _file_sha256(source) != before

    untouched = tmp_path / "untouched.db"
    _create_old_schema(
        untouched,
        [_snap(opponent_id="gavin", session_id="g1", side="offense", play="Mesh", formation="Gun Bunch")],
    )
    untouched_hash = _file_sha256(untouched)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o555)
    try:
        with pytest.raises(OSError):
            migrate_database(untouched, blocked / "backup.db")
    finally:
        blocked.chmod(0o755)
    assert _file_sha256(untouched) == untouched_hash
    assert "ml_decisions" not in _table_names(untouched)
    assert "executed_play" not in {
        row[1] for row in sqlite3.connect(untouched).execute("PRAGMA table_info(snaps)")
    }


def test_shadow_export_features_exclude_post_snap_and_split_by_game(tmp_path):
    assert EXPLICIT_FEATURE_DENYLIST <= SHADOW_FEATURE_DENYLIST
    source = tmp_path / "shadow.db"
    snaps = []
    for game, session in (("gavin", "gavin-2026-10-08"), ("james", "james-2026-10-08")):
        snaps.append(
            _snap(
                opponent_id=game,
                session_id=session,
                side="offense",
                play="Mesh",
                formation="Gun Bunch",
                coverage_seen="Cover 3",
                seq=1,
            )
        )
        snaps.append(
            _snap(
                opponent_id=game,
                session_id=session,
                side="defense",
                play="Cover 3 Sky",
                formation="Nickel Over",
                seq=2,
            )
        )
    _create_old_schema(source, snaps)
    migrate_database(source, tmp_path / "shadow.backup.db")
    conn = sqlite3.connect(source)
    conn.execute(
        """
        UPDATE snaps
        SET executed_status = 'identified',
            executed_verification = 'verified',
            executed_formation = 'Gun Bunch',
            executed_play = 'Mesh',
            result = '+4'
        WHERE opponent_id = 'gavin' AND side = 'offense'
        """
    )
    conn.commit()
    conn.close()

    out = tmp_path / "shadow-out"
    export_shadow(source, out, max_games=4)
    games = sorted(path.name for path in (out / "by_game").iterdir() if path.is_dir())
    assert games == ["gavin-2026-10-08", "james-2026-10-08"]
    recommendation_flags = []
    coverage_flags = []
    for game in games:
        for side in ("offense", "defense"):
            path = out / "by_game" / game / f"{side}.jsonl"
            assert path.is_file()
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert records
            assert {row["side"] for row in records} == {side}
            assert {row["game_id"] for row in records} == {game}
            for row in records:
                assert EXPLICIT_FEATURE_DENYLIST.isdisjoint(row["features"])
                assert "coverage_seen" not in row
                assert "result" not in row
                recommendation_flags.append(row["recommendation_only"])
                coverage_flags.append(row["unverified_coverage_seen"])
    assert True in recommendation_flags
    assert False in recommendation_flags
    assert True in coverage_flags
    assert False in coverage_flags


def test_october8_shape_migration_does_not_promote_recommendations(tmp_path):
    snaps = _october8_snaps()
    assert len(snaps) == 245
    assert sum(count for _opp, _session, count in OCTOBER8_GAMES) == 245
    source = tmp_path / "october8-shape.db"
    _create_old_schema(source, snaps)
    conn = sqlite3.connect(source)
    assert conn.execute("SELECT COUNT(*) FROM play_records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vision_observations").fetchone()[0] == 0
    columns = {row[1] for row in conn.execute("PRAGMA table_info(snaps)")}
    conn.close()
    assert "ml_snap_id" not in columns
    assert "opponent_type" not in columns
    assert not any(name.startswith("executed_") for name in columns)

    before = _original_checksum(source)
    migrate_database(source, tmp_path / "october8.backup.db")
    assert _original_checksum(source) == before

    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM snaps").fetchall()
    assert len(rows) == 245
    assert conn.execute("SELECT COUNT(*) FROM play_records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vision_observations").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ml_decisions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ml_outcomes").fetchone()[0] == 0
    for row in rows:
        for column in (
            "executed_status",
            "executed_formation",
            "executed_play",
            "executed_macro",
            "executed_verification",
        ):
            assert row[column] in (None, "unknown")
            if row[column] is not None:
                assert row[column] not in {row["play"], row["formation"], row["our_call"]}
        assert row["executed_play"] != row["play"]
        assert row["accepted"] is None
    conn.close()


def test_readers_match_pre_migration_on_october8_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_COACH_AI_RESEARCH", "off")
    source = tmp_path / "readers.db"
    _create_old_schema(source, _october8_snaps())
    before = _reader_snapshot(source)
    migrate_database(source, tmp_path / "readers.backup.db")
    after = _reader_snapshot(source)
    assert after == before
    assert before["four_user_games"]["verified_labeled_human_executions"] == 0
    assert before["four_user_games"]["found_human_games"] == 4
