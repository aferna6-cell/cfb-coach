"""SQLite coach DB — seed on first run; log snaps; track tendencies."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_coach.seed import load_seed


def resolve_db_path_from_env() -> Path | None:
    raw = os.environ.get("CFB_COACH_DB")
    return Path(raw) if raw else None

def default_db_path() -> Path:
    """Prefer CFB_COACH_DB, then ~/.cfb-coach/coach.db, then package data/."""
    env = resolve_db_path_from_env()
    if env is not None:
        env.parent.mkdir(parents=True, exist_ok=True)
        return env
    home = Path.home() / ".cfb-coach" / "coach.db"
    try:
        home.parent.mkdir(parents=True, exist_ok=True)
        if not home.exists():
            home.touch()
        return home
    except OSError:
        fallback = Path("/workspace/cfb-coach/data/coach.db")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


class CoachDB:
    def __init__(self, path: Path | None = None, *, seed: dict[str, Any] | None = None) -> None:
        self.path = path or default_db_path()
        # Optional per-game seed (Madden 27); default = CFB seed.json
        self._seed = seed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()
        self._seed_if_empty()

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS opponents (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                team_now TEXT,
                skill TEXT,
                confidence TEXT,
                profile_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snaps (
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
                FOREIGN KEY (opponent_id) REFERENCES opponents(id)
            );
            CREATE TABLE IF NOT EXISTS tendencies (
                opponent_id TEXT NOT NULL,
                bucket TEXT NOT NULL,
                key TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (opponent_id, bucket, key)
            );
            CREATE TABLE IF NOT EXISTS gameplan_weights (
                opponent_id TEXT NOT NULL,
                side TEXT NOT NULL,
                key TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (opponent_id, side, key)
            );
            CREATE TABLE IF NOT EXISTS macro_weights (
                opponent_id TEXT NOT NULL,
                macro TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                armed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (opponent_id, macro)
            );
            CREATE TABLE IF NOT EXISTS install_sheets (
                opponent_id TEXT PRIMARY KEY,
                sheet_json TEXT NOT NULL,
                updated_ts TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS install_diffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                opponent_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                detail TEXT NOT NULL,
                why TEXT
            );
            CREATE TABLE IF NOT EXISTS vision_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT,
                play_state TEXT,
                formation TEXT,
                shell TEXT,
                pressure TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vision_obs_ts
                ON vision_observations(ts);
            CREATE TABLE IF NOT EXISTS game_sessions (
                session_id TEXT PRIMARY KEY,
                opponent_id TEXT NOT NULL,
                dynasty TEXT,
                started_ts TEXT NOT NULL,
                ended_ts TEXT,
                notes TEXT,
                play_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS play_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                game_id TEXT,
                play_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                opponent_id TEXT NOT NULL,
                side TEXT,
                down INTEGER,
                distance INTEGER,
                yardline INTEGER,
                quarter INTEGER,
                field_zone TEXT,
                formation TEXT,
                shell TEXT,
                pressure TEXT,
                play_family TEXT,
                concept_tags TEXT,
                result_type TEXT,
                yards INTEGER,
                coach_rec TEXT,
                macro TEXT,
                our_call TEXT,
                confidence_json TEXT,
                payload_json TEXT NOT NULL,
                corrected INTEGER NOT NULL DEFAULT 0,
                UNIQUE(session_id, play_id)
            );
            CREATE INDEX IF NOT EXISTS idx_play_records_session
                ON play_records(session_id);
            CREATE TABLE IF NOT EXISTS live_tendency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                signal TEXT,
                message TEXT,
                payload_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_live_tend_session
                ON live_tendency_events(session_id);
            """
        )
        self.conn.commit()
        self._migrate_m2_columns()

    def _seed_if_empty(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'seeded'"
        ).fetchone()
        if row:
            return
        seed = self._seed or load_seed()
        for oid, opp in seed["opponents"].items():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO opponents
                (id, display_name, team_now, skill, confidence, profile_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    oid,
                    opp.get("display_name", oid),
                    opp.get("team_now"),
                    opp.get("skill"),
                    opp.get("confidence"),
                    json.dumps(opp),
                ),
            )
            # Seed soft priors in *_seed buckets — do NOT count as live repeats
            for bucket, keys in _extract_seed_tendencies(opp).items():
                seed_bucket = f"{bucket}_seed"
                for key, weight in keys.items():
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO tendencies
                        (opponent_id, bucket, key, count, success)
                        VALUES (?, ?, ?, ?, 0)
                        """,
                        (oid, seed_bucket, key, weight),
                    )
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('seeded', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('db_path', ?)",
            (str(self.path),),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def list_opponents(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT id, display_name, team_now, skill, confidence "
                "FROM opponents ORDER BY id"
            )
        )

    def get_opponent(self, oid: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT profile_json FROM opponents WHERE id = ?", (oid,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["profile_json"])

    def bump_tendency(
        self,
        opponent_id: str,
        bucket: str,
        key: str,
        *,
        success: bool = False,
        amount: int = 1,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO tendencies (opponent_id, bucket, key, count, success)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(opponent_id, bucket, key) DO UPDATE SET
                count = count + excluded.count,
                success = success + excluded.success
            """,
            (opponent_id, bucket, key, amount, 1 if success else 0),
        )
        self.conn.commit()

    def get_tendencies(
        self, opponent_id: str, bucket: str | None = None
    ) -> list[sqlite3.Row]:
        if bucket:
            return list(
                self.conn.execute(
                    "SELECT * FROM tendencies WHERE opponent_id = ? AND bucket = ? "
                    "ORDER BY count DESC",
                    (opponent_id, bucket),
                )
            )
        return list(
            self.conn.execute(
                "SELECT * FROM tendencies WHERE opponent_id = ? ORDER BY count DESC",
                (opponent_id,),
            )
        )

    def log_snap(
        self,
        *,
        opponent_id: str,
        side: str,
        situation_raw: str,
        our_call: str,
        formation: str | None = None,
        play: str | None = None,
        macro: str | None = None,
        down: int | None = None,
        distance: int | None = None,
        yardline: int | None = None,
        quarter: int | None = None,
        result: str | None = None,
        coverage_seen: str | None = None,
        concept_seen: str | None = None,
        notes: str | None = None,
        session_id: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO snaps (
                ts, opponent_id, side, down, distance, yardline, quarter,
                situation_raw, our_call, formation, play, macro,
                result, coverage_seen, concept_seen, notes, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                opponent_id,
                side,
                down,
                distance,
                yardline,
                quarter,
                situation_raw,
                our_call,
                formation,
                play,
                macro,
                result,
                coverage_seen,
                concept_seen,
                notes,
                session_id,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)


    def count_snaps(self, opponent_id: str, *, side: str | None = None) -> int:
        if side:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM snaps WHERE opponent_id = ? AND side = ?",
                (opponent_id, side),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM snaps WHERE opponent_id = ?",
                (opponent_id,),
            ).fetchone()
        return int(row["n"] if row else 0)

    def get_recent_snaps(
        self,
        opponent_id: str,
        *,
        side: str | None = None,
        since_id: int | None = None,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        clauses = ["opponent_id = ?"]
        params: list[Any] = [opponent_id]
        if side:
            clauses.append("side = ?")
            params.append(side)
        if since_id is not None:
            clauses.append("id > ?")
            params.append(since_id)
        params.append(limit)
        sql = (
            "SELECT * FROM snaps WHERE "
            + " AND ".join(clauses)
            + " ORDER BY id DESC LIMIT ?"
        )
        return list(self.conn.execute(sql, params))

    def bump_gameplan_weight(
        self,
        opponent_id: str,
        side: str,
        key: str,
        delta: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO gameplan_weights (opponent_id, side, key, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(opponent_id, side, key) DO UPDATE SET
                weight = weight + excluded.weight
            """,
            (opponent_id, side, key, delta),
        )
        self.conn.commit()

    def get_gameplan_weights(
        self, opponent_id: str, side: str | None = None
    ) -> list[sqlite3.Row]:
        if side:
            return list(
                self.conn.execute(
                    "SELECT * FROM gameplan_weights WHERE opponent_id = ? AND side = ? "
                    "ORDER BY ABS(weight) DESC",
                    (opponent_id, side),
                )
            )
        return list(
            self.conn.execute(
                "SELECT * FROM gameplan_weights WHERE opponent_id = ? "
                "ORDER BY ABS(weight) DESC",
                (opponent_id,),
            )
        )

    def bump_macro_weight(
        self,
        opponent_id: str,
        macro: str,
        delta: float,
        *,
        armed: bool | None = None,
    ) -> None:
        macro = macro.upper()
        self.conn.execute(
            """
            INSERT INTO macro_weights (opponent_id, macro, weight, armed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(opponent_id, macro) DO UPDATE SET
                weight = weight + excluded.weight,
                armed = COALESCE(?, armed)
            """,
            (
                opponent_id,
                macro,
                delta,
                1 if armed else 0,
                None if armed is None else (1 if armed else 0),
            ),
        )
        self.conn.commit()

    def get_macro_weights(self, opponent_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM macro_weights WHERE opponent_id = ? "
                "ORDER BY ABS(weight) DESC",
                (opponent_id,),
            )
        )





    def log_vision_observation(self, payload: dict[str, Any]) -> int:
        """Persist a GameObservation (or dict) for later game reconstruct."""
        ts = datetime.now(timezone.utc).isoformat()
        source = str(payload.get("source") or "")
        play_state = str(payload.get("play_state") or "")
        formation = str(payload.get("formation") or "")
        shell = str(payload.get("shell") or "")
        pressure = str(payload.get("pressure") or "")
        cur = self.conn.execute(
            """
            INSERT INTO vision_observations
                (ts, source, play_state, formation, shell, pressure, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                source,
                play_state,
                formation,
                shell,
                pressure,
                json.dumps(payload),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_vision_observations(
        self, *, limit: int = 100, since_ts: str | None = None
    ) -> list[sqlite3.Row]:
        if since_ts:
            return list(
                self.conn.execute(
                    "SELECT * FROM vision_observations WHERE ts >= ? "
                    "ORDER BY id DESC LIMIT ?",
                    (since_ts, limit),
                )
            )
        return list(
            self.conn.execute(
                "SELECT * FROM vision_observations ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )


    def get_install_sheet(self, opponent_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT sheet_json FROM install_sheets WHERE opponent_id = ?",
            (opponent_id,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["sheet_json"])

    def save_install_sheet(self, opponent_id: str, sheet: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO install_sheets (opponent_id, sheet_json, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(opponent_id) DO UPDATE SET
                sheet_json = excluded.sheet_json,
                updated_ts = excluded.updated_ts
            """,
            (
                opponent_id,
                json.dumps(sheet),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def log_install_diff(
        self,
        *,
        opponent_id: str,
        action: str,
        target: str,
        detail: str,
        why: str = "",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO install_diffs (ts, opponent_id, action, target, detail, why)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                opponent_id,
                action,
                target,
                detail,
                why,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_install_diffs(
        self, opponent_id: str, *, limit: int = 50
    ) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM install_diffs WHERE opponent_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (opponent_id, limit),
            )
        )



    def _migrate_m2_columns(self) -> None:
        """Additive column guards for older DBs (never destroy data)."""
        # snaps: optional session_id
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(snaps)").fetchall()}
        if "session_id" not in cols:
            try:
                self.conn.execute("ALTER TABLE snaps ADD COLUMN session_id TEXT")
                self.conn.commit()
            except sqlite3.Error:
                pass
        gs_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(game_sessions)").fetchall()}
        if gs_cols:
            if "result_wl" not in gs_cols:
                try:
                    self.conn.execute("ALTER TABLE game_sessions ADD COLUMN result_wl TEXT")
                    self.conn.commit()
                except sqlite3.Error:
                    pass
            if "score" not in gs_cols:
                try:
                    self.conn.execute("ALTER TABLE game_sessions ADD COLUMN score TEXT")
                    self.conn.commit()
                except sqlite3.Error:
                    pass

    def start_game_session(self, sess: Any) -> str:
        d = sess.to_dict() if hasattr(sess, "to_dict") else dict(sess)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO game_sessions
                (session_id, opponent_id, dynasty, started_ts, ended_ts, notes, play_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d["session_id"],
                d["opponent_id"],
                d.get("dynasty"),
                d["started_ts"],
                d.get("ended_ts"),
                d.get("notes") or "",
                int(d.get("play_count") or 0),
            ),
        )
        self.conn.commit()
        return str(d["session_id"])

    def end_game_session(
        self,
        session_id: str,
        *,
        play_count: int | None = None,
        result_wl: str | None = None,
        score: str | None = None,
    ) -> None:
        ended = datetime.now(timezone.utc).isoformat()
        # Ensure optional columns exist (older DBs)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(game_sessions)").fetchall()}
        if "result_wl" not in cols:
            try:
                self.conn.execute("ALTER TABLE game_sessions ADD COLUMN result_wl TEXT")
            except sqlite3.Error:
                pass
        if "score" not in cols:
            try:
                self.conn.execute("ALTER TABLE game_sessions ADD COLUMN score TEXT")
            except sqlite3.Error:
                pass
        sets = ["ended_ts = ?"]
        params: list[Any] = [ended]
        if play_count is not None:
            sets.append("play_count = ?")
            params.append(int(play_count))
        if result_wl is not None:
            sets.append("result_wl = ?")
            params.append(str(result_wl).lower().strip())
        if score is not None:
            sets.append("score = ?")
            params.append(str(score).strip())
        params.append(session_id)
        self.conn.execute(
            f"UPDATE game_sessions SET {', '.join(sets)} WHERE session_id = ?",
            params,
        )
        self.conn.commit()

    def get_session_snaps(self, session_id: str, *, limit: int = 500) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM snaps WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            )
        )

    def log_play_record(self, rec: Any) -> int:
        """Persist PlayRecord (dict or dataclass). Idempotent on session+play_id."""
        d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        ts = datetime.now(timezone.utc).isoformat()
        tags = d.get("concept_tags") or []
        if isinstance(tags, list):
            tags_s = json.dumps(tags)
        else:
            tags_s = str(tags)
        conf = d.get("confidence") or {}
        cur = self.conn.execute(
            """
            INSERT INTO play_records (
                session_id, game_id, play_id, ts, opponent_id, side,
                down, distance, yardline, quarter, field_zone,
                formation, shell, pressure, play_family, concept_tags,
                result_type, yards, coach_rec, macro, our_call,
                confidence_json, payload_json, corrected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, play_id) DO UPDATE SET
                result_type = excluded.result_type,
                yards = excluded.yards,
                play_family = excluded.play_family,
                concept_tags = excluded.concept_tags,
                coach_rec = excluded.coach_rec,
                confidence_json = excluded.confidence_json,
                payload_json = excluded.payload_json,
                corrected = excluded.corrected
            """,
            (
                d.get("session_id") or "",
                d.get("game_id") or d.get("session_id") or "",
                d.get("play_id") or "",
                ts,
                d.get("opponent_id") or "",
                d.get("side"),
                d.get("down"),
                d.get("distance"),
                d.get("yardline"),
                d.get("quarter"),
                d.get("field_zone"),
                d.get("formation"),
                d.get("shell"),
                d.get("pressure"),
                d.get("play_family"),
                tags_s,
                d.get("result_type"),
                d.get("yards"),
                d.get("coach_rec"),
                d.get("macro"),
                d.get("our_call"),
                json.dumps(conf),
                json.dumps(d),
                1 if d.get("corrected") else 0,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_session_plays(
        self, session_id: str, *, limit: int = 200
    ) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM play_records WHERE session_id = ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            )
        )

    def log_live_tendency_event(
        self,
        *,
        session_id: str,
        kind: str,
        signal: str = "",
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO live_tendency_events
                (session_id, ts, kind, signal, message, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                kind,
                signal,
                message,
                json.dumps(payload or {}),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_live_tendency_events(
        self, session_id: str, *, limit: int = 50
    ) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM live_tendency_events WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        )



def _extract_seed_tendencies(opp: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Soft seed weights from profile lists (dynasty evidence > meta later)."""
    out: dict[str, dict[str, int]] = {
        "offense_concept": {},
        "their_coverage": {},
    }
    offense = opp.get("offense") or {}
    # High-confidence concepts get more seed weight
    conf = (opp.get("confidence") or "low").lower()
    base = {"high": 4, "medium": 2, "low": 1, "none": 0}.get(conf, 1)
    for key in (
        "early_down",
        "explosives",
        "intermediate",
        "rpo",
        "goal_line",
        "core",
        "money",
        "stress",
        "runs",
    ):
        for concept in offense.get(key) or []:
            if isinstance(concept, str):
                out["offense_concept"][concept] = (
                    out["offense_concept"].get(concept, 0) + base
                )
    dvs = opp.get("defense_vs_us") or {}
    for cov in dvs.get("coverages") or []:
        out["their_coverage"][cov] = out["their_coverage"].get(cov, 0) + base
    for cov in dvs.get("example_nd") or []:
        # strip situational tags like "Cover 4 Palms 3rd-long"
        out["their_coverage"][cov] = out["their_coverage"].get(cov, 0) + max(
            1, base - 1
        )
    for snap in opp.get("sample_snaps") or []:
        cov = snap.get("cov")
        if cov:
            out["their_coverage"][cov] = out["their_coverage"].get(cov, 0) + 1
    return out


