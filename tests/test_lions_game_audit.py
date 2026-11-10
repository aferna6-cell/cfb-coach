"""Read-only Lions latest-game audit smoke tests.

Fixtures are synthetic; never read or mutate the owner's DB.
"""
from __future__ import annotations

import hashlib
import sqlite3

from scripts import audit_lions_latest_game as auditmod


def seed(path):
    with sqlite3.connect(path) as c:
        c.executescript("""
          CREATE TABLE game_sessions (
              session_id TEXT PRIMARY KEY, opponent_id TEXT,
              started_ts TEXT, ended_ts TEXT, score TEXT, result_wl TEXT
          );
          CREATE TABLE snaps (
              id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, down INTEGER,
              distance INTEGER, formation TEXT, play TEXT, result TEXT,
              yards INTEGER, executed_verification TEXT, executed_status TEXT,
              drive_id TEXT, quarter INTEGER, yardline INTEGER
          );
          CREATE TABLE ml_decisions (
              id INTEGER PRIMARY KEY, game_id TEXT, session_id TEXT, mode TEXT,
              shadow_status TEXT, model_version TEXT, policy_source TEXT,
              latency_ms_model REAL, final_formation TEXT, final_play TEXT,
              heuristic_formation TEXT, heuristic_play TEXT
          );
        """)
        c.execute("INSERT INTO game_sessions VALUES (?,?,?,?,?,?)",
                  ("old", "cpu", "2026-11-01T01:00:00", "2026-11-01T02:00:00", "7-3", "win"))
        c.execute("INSERT INTO game_sessions VALUES (?,?,?,?,?,?)",
                  ("new", "cpu", "2026-11-09T20:00:00", "2026-11-09T22:00:00", "21-14", "win"))
        snaps = [
            (1, "old", "offense", 1, 10, "Gun 5WR Tight", "Mesh", "+7", 7, "verified", "identified", "old-d1", 1, 25),
            (2, "new", "offense", 1, 10, "Pistol Trips", "HB Stretch", "+8", 8, "verified", "identified", "d1", 1, 25),
            (3, "new", "offense", 3, 2, "Gun Trips X Nasty", "Hi Lo Cross", "+6", 6, "verified", "identified", "d1", 1, 42),
            (4, "new", "offense", 1, 10, "Gun 5WR Tight", "Mesh", "td", None, "unknown", "unknown", "d2", 2, 83),
            (5, "new", "defense", 3, 7, "Nickel Over", "Cover 3 Sky", "+17", 17, "verified", "identified", "opp-d1", 1, 45),
            (6, "new", "defense", 1, 10, "4-3 Over", "Tampa 2", "int", None, "verified", "identified", "opp-d1", 2, 80),
        ]
        c.executemany("INSERT INTO snaps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", snaps)
        c.execute("INSERT INTO ml_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (1, "new", "new", "experimental", "ok",
                   "madden-ml.experimental.offense.650.empirical", "model", 38.2,
                   "Pistol Trips", "HB Stretch", "Gun 5WR Tight", "Mesh"))
        c.execute("INSERT INTO ml_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (2, "new", "new", "experimental", "timeout",
                   "madden-ml.experimental.offense.650.empirical", "heuristic", 180.1,
                   "Gun 5WR Tight", "Mesh", "Gun 5WR Tight", "Mesh"))


def test_latest_game_audits_both_sides_without_writing(tmp_path, monkeypatch):
    path = tmp_path / "lions.db"
    seed(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        auditmod, "evaluate_saved_games",
        lambda db, game_id, limit: {"games": [{"game_id": game_id,
                                               "fallback_count": 1,
                                               "third_down": {"verified_conversions": 1}}]}
    )
    report = auditmod.audit(path)
    assert report["game_id"] == "new"
    assert report["finished"] is True
    assert report["session"]["score"] == "21-14"
    assert report["offense"]["logged_snaps"] == 3
    assert report["offense"]["third_down_conversions"] == 1
    assert report["offense"]["outcomes"]["td"] == 1
    assert report["offense"]["distinct_formation_play_pairs"] == 3
    assert report["offense"]["yards_known_plays"] == 2
    assert report["defense"]["logged_snaps"] == 2
    assert report["defense"]["third_down_conversions"] == 1
    assert report["defense"]["gains_or_yards_allowed_15_plus"] == 1
    assert report["defense"]["outcomes"]["int"] == 1
    assert report["offensive_model_decisions"]["experimental_non_ok"] == 1
    assert report["existing_offense_evaluation"]["fallback_count"] == 1
    assert report["history_modified"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_select_specific_older_game_without_mutating_new(tmp_path, monkeypatch):
    path = tmp_path / "lions.db"
    seed(path)
    monkeypatch.setattr(
        auditmod, "evaluate_saved_games", lambda db, game_id, limit: {"games": []}
    )
    report = auditmod.audit(path, "old")
    assert report["game_id"] == "old"
    assert report["offense"]["logged_snaps"] == 1
    assert report["defense"]["logged_snaps"] == 0
    assert report["session"]["score"] == "7-3"
