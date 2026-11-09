"""Read-only audit of live Madden database provenance and missing clock signals."""
from __future__ import annotations

from importlib import util
from pathlib import Path
import sqlite3


_TOOL = Path(__file__).resolve().parents[2] / "scripts" / "madden_ml" / "cpu2_audit_database.py"
_spec = util.spec_from_file_location("cpu2_audit_database", _TOOL)
assert _spec and _spec.loader
_module = util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_audit_reads_realistic_schema_without_writing_database(tmp_path):
    path = tmp_path / "madden27.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE game_sessions (
              session_id TEXT PRIMARY KEY, opponent_id TEXT, play_count INT,
              started_ts TEXT, score TEXT, result_wl TEXT
            );
            CREATE TABLE snaps (
              id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, game_clock INT,
              score_for INT, score_against INT, quarter INT,
              executed_verification TEXT, executed_status TEXT,
              executed_formation TEXT, executed_play TEXT,
              coverage_seen TEXT, recommended_adjustment_id TEXT,
              executed_adjustment_id TEXT
            );
            CREATE TABLE ml_decisions (
              id INTEGER PRIMARY KEY, game_id TEXT, policy_source TEXT,
              shadow_status TEXT, decision_json TEXT
            );
            """
        )
        db.executemany("INSERT INTO meta VALUES (?,?)", [
            ("ml_mode", "experimental"), ("ml_offense_ranker", "route_aware"),
            ("ml_football_priors", "1"),
        ])
        db.execute("INSERT INTO game_sessions VALUES (?,?,?,?,?,?)", (
            "cpu-game", "cpu", 1, "2026-11-08T22:00:00Z", "14-14", "win"
        ))
        db.execute(
            "INSERT INTO snaps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                1, "cpu-game", "offense", None, 14, 14, 4,
                "verified", "identified", "Gun Bunch", "Mesh", "Cover 3 Match",
                "off.hot_route", None,
            )
        )
        db.execute("INSERT INTO ml_decisions VALUES (?,?,?,?,?)", (
            1, "cpu-game", "heuristic", "timeout", "{}"
        ))
    original = path.read_bytes()
    report = _module.audit(path, game_id="cpu-game")
    assert path.read_bytes() == original
    assert report["configured_mode"] == "experimental"
    assert report["configured_ranker"] == "route_aware"
    assert report["football_priors"]["stored"] == "1"
    assert report["game"]["offensive_snaps"] == 1
    assert report["game"]["clock_recorded"] == 0
    assert report["game"]["verified_executions"] == 1
    assert report["game"]["policy_source_counts"]["heuristic"] == 1
    assert any("clock" in message for message in report["findings"])
    assert any("14-14" in message for message in report["findings"])
