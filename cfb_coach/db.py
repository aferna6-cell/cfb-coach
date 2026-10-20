"""SQLite coach DB — seed on first run; log snaps; track tendencies."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cfb_coach.seed import load_seed

_UNSET = object()


def resolve_db_path_from_env() -> Path | None:
    raw = os.environ.get("CFB_COACH_DB")
    return Path(raw) if raw else None


def _madden_db_filename() -> str:
    from cfb_coach.games import GAMES, MADDEN27

    return GAMES[MADDEN27].db_filename


def _nonempty_env(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw if raw else None


def madden_db_path() -> Path:
    """Resolve the Madden database path.

    A non-empty ``CFB_COACH_MADDEN_DB`` wins. Otherwise a non-empty
    ``CFB_COACH_DB`` selects ``<its parent>/madden27.db`` and does not call
    ``Path.home()`` or create ``~/.cfb-coach``. Otherwise the path is
    ``~/.cfb-coach/madden27.db``. Empty strings count as unset. Whenever
    ``CFB_COACH_DB`` is set, a ``UserWarning`` names both variables if the
    resolved Madden path is that same file. The warning also fires when
    ``CFB_COACH_MADDEN_DB`` names the sibling ``madden27.db``.
    """
    filename = _madden_db_filename()
    madden_raw = _nonempty_env("CFB_COACH_MADDEN_DB")
    cfb_raw = _nonempty_env("CFB_COACH_DB")
    sibling = Path(cfb_raw).parent / filename if cfb_raw else None

    def _warn_if_same_file(chosen: Path) -> None:
        if not cfb_raw:
            return
        resolved = chosen.resolve()
        same = resolved == Path(cfb_raw).resolve()
        if madden_raw and sibling is not None and resolved == sibling.resolve():
            same = True
        if same:
            warnings.warn(
                "CFB_COACH_MADDEN_DB and CFB_COACH_DB resolve to the same file "
                f"({chosen})",
                UserWarning,
                stacklevel=3,
            )

    if madden_raw:
        chosen = Path(madden_raw)
        _warn_if_same_file(chosen)
        parent = chosen.parent
        if not parent.exists():
            parent.mkdir()
        return chosen
    if sibling is not None:
        _warn_if_same_file(sibling)
        return sibling
    try:
        home = Path.home() / ".cfb-coach" / filename
        home.parent.mkdir(parents=True, exist_ok=True)
        return home
    except OSError:
        fallback = Path("/workspace/cfb-coach/data") / filename
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def default_db_path() -> Path:
    """Prefer CFB_COACH_DB, then ~/.cfb-coach/coach.db, then package data/.

    A non-empty ``CFB_COACH_DB`` is returned as given. This branch does not
    call ``Path.home()`` and does not create ``~/.cfb-coach``. Only the parent
    of the given path is created, and only when that parent is missing. An
    empty value is treated as unset.
    """
    env = resolve_db_path_from_env()
    if env is not None:
        parent = env.parent
        if not parent.exists():
            parent.mkdir()
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
    def __init__(
        self,
        path: Path | None = None,
        *,
        seed: dict[str, Any] | None = None,
        read_only: bool = False,
        migrate: bool | None = None,
    ) -> None:
        self.path = path or default_db_path()
        # Optional per-game seed (Madden 27); default = CFB seed.json
        self._seed = seed
        self.read_only = bool(read_only)
        do_migrate = (not self.read_only) if migrate is None else bool(migrate)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"read-only open requires an existing database: {self.path}")
            # URI mode=ro never creates or writes the source file.
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if do_migrate:
            if self.read_only:
                raise ValueError("cannot migrate a read-only database connection")
            self._migrate()
            self._seed_if_empty()

    def close(self) -> None:
        self.conn.close()

    @classmethod
    def open_read_only(cls, path: Path | str) -> "CoachDB":
        """Open an existing DB without migrations or writes."""
        return cls(Path(path), read_only=True, migrate=False)

    def backup_to(self, dest: Path | str) -> Path:
        """Consistent SQLite backup via the backup API (safe with WAL / readers).

        Does not use a raw filesystem copy of a live database file.
        """
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Backup always opens a writable destination, even when self is read-only.
        out = sqlite3.connect(str(dest_path))
        try:
            self.conn.backup(out)
            out.commit()
        finally:
            out.close()
        return dest_path

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
        self._migrate_ml_tables()

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

    def adjust_tendency(
        self,
        opponent_id: str,
        bucket: str,
        key: str,
        *,
        amount: int,
        success_delta: int = 0,
    ) -> None:
        """Add ``amount`` (may be negative) to an existing live tendency. Used by undo."""
        self.conn.execute(
            """
            UPDATE tendencies
            SET count = MAX(0, count + ?), success = MAX(0, success + ?)
            WHERE opponent_id = ? AND bucket = ? AND key = ?
            """,
            (amount, success_delta, opponent_id, bucket, key),
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
        ml_snap_id: str | None = None,
        snap_seq: int | None = None,
        executed_status: str | None = None,
        executed_formation: str | None = None,
        executed_play: str | None = None,
        executed_macro: str | None = None,
        executed_verification: str | None = None,
        ml_decision_id: int | None = None,
        data_source: str | None = None,
        drive_id: str | None = None,
        game_clock: str | None = None,
        score_for: int | None = None,
        score_against: int | None = None,
        penalty: str | None = None,
        yards: Any = _UNSET,
        accepted: str | None = None,
        recommended_adjustment_id: str | None = None,
        executed_adjustment_id: str | None = None,
        adjustment_confirmed: bool = False,
    ) -> int:
        self._migrate_ml_snap_columns()
        self._migrate_madden_call_outcome()
        # The decision sealed for this snap id owns the row. A stale id from an
        # earlier call must not stay on the snap that was actually run.
        owned = self._decision_id_for_snap(ml_snap_id)
        if owned is not None:
            ml_decision_id = owned
        # Idempotent on ml_snap_id: corrections / retries update the same logical snap.
        if ml_snap_id:
            existing = self.conn.execute(
                "SELECT id FROM snaps WHERE ml_snap_id = ?", (ml_snap_id,)
            ).fetchone()
            if existing:
                row_id = int(existing["id"] if hasattr(existing, "keys") else existing[0])
                fields: dict[str, Any] = {
                    "result": result,
                    "coverage_seen": coverage_seen,
                    "concept_seen": concept_seen,
                    "notes": notes,
                    "executed_status": executed_status,
                    "executed_formation": executed_formation,
                    "executed_play": executed_play,
                    "executed_macro": executed_macro,
                    "executed_verification": executed_verification,
                    "formation": formation,
                    "play": play,
                    "macro": macro,
                }
                # None must not wipe a decision id already stored on this snap.
                if ml_decision_id is not None:
                    fields["ml_decision_id"] = ml_decision_id
                self.update_snap(row_id, **fields)
                self._finish_snap_log(
                    row_id,
                    result=result,
                    ml_snap_id=ml_snap_id,
                    ml_decision_id=ml_decision_id,
                    drive_id=drive_id,
                    game_clock=game_clock,
                    score_for=score_for,
                    score_against=score_against,
                    penalty=penalty,
                    yards=yards,
                    accepted=accepted,
                    recommended_adjustment_id=recommended_adjustment_id,
                    executed_adjustment_id=executed_adjustment_id,
                    adjustment_confirmed=adjustment_confirmed,
                )
                return row_id
        cur = self.conn.execute(
            """
            INSERT INTO snaps (
                ts, opponent_id, side, down, distance, yardline, quarter,
                situation_raw, our_call, formation, play, macro,
                result, coverage_seen, concept_seen, notes, session_id,
                ml_snap_id, snap_seq, executed_status, executed_formation,
                executed_play, executed_macro, executed_verification,
                ml_decision_id, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ml_snap_id,
                snap_seq,
                executed_status or "unknown",
                executed_formation,
                executed_play,
                executed_macro,
                executed_verification or "unknown",
                ml_decision_id,
                data_source or "live",
            ),
        )
        row_id = int(cur.lastrowid)
        self._finish_snap_log(
            row_id,
            result=result,
            ml_snap_id=ml_snap_id,
            ml_decision_id=ml_decision_id,
            drive_id=drive_id,
            game_clock=game_clock,
            score_for=score_for,
            score_against=score_against,
            penalty=penalty,
            yards=yards,
            accepted=accepted,
            recommended_adjustment_id=recommended_adjustment_id,
            executed_adjustment_id=executed_adjustment_id,
            adjustment_confirmed=adjustment_confirmed,
        )
        return row_id

    def _finish_snap_log(
        self,
        row_id: int,
        *,
        result: str | None,
        ml_snap_id: str | None,
        ml_decision_id: int | None,
        drive_id: str | None,
        game_clock: str | None,
        score_for: int | None,
        score_against: int | None,
        penalty: str | None,
        yards: Any,
        accepted: str | None,
        recommended_adjustment_id: str | None,
        executed_adjustment_id: str | None,
        adjustment_confirmed: bool,
    ) -> None:
        from cfb_coach.madden.model.call_outcome_log import (
            coerce_logged_yards,
            executed_adjustment_for_log,
            penalty_for_log,
        )

        values: dict[str, Any] = {
            "yards": coerce_logged_yards(result, None if yards is _UNSET else yards),
            "penalty": penalty_for_log(result, penalty),
            "executed_adjustment_id": executed_adjustment_for_log(
                executed_adjustment_id, confirmed=adjustment_confirmed
            ),
        }
        for key, value in (
            ("drive_id", drive_id),
            ("game_clock", game_clock),
            ("score_for", score_for),
            ("score_against", score_against),
            ("accepted", accepted),
            ("recommended_adjustment_id", recommended_adjustment_id),
        ):
            if value is not None:
                values[key] = value
        self._set_columns("snaps", row_id, values)
        self._link_decision(ml_snap_id, ml_decision_id)
        self.conn.commit()

    def update_snap(self, snap_id: int, **fields: Any) -> None:
        # ``executed_adjustment_id`` stays off this list. Only
        # ``executed_adjustment_for_log(confirmed=True)`` may write it.
        allowed = (
            "result",
            "coverage_seen",
            "concept_seen",
            "notes",
            "executed_status",
            "executed_formation",
            "executed_play",
            "executed_macro",
            "executed_verification",
            "ml_decision_id",
            "ml_snap_id",
            "snap_seq",
            "formation",
            "play",
            "macro",
            "data_source",
            "drive_id",
            "game_clock",
            "score_for",
            "score_against",
            "penalty",
            "yards",
            "accepted",
            "recommended_adjustment_id",
        )
        if "ml_decision_id" in fields:
            # A later correction must not put a stale decision back on a snap
            # whose snap id already has a sealed owner.
            ml_snap_id = fields.get("ml_snap_id")
            if not ml_snap_id and "ml_snap_id" in self._table_columns("snaps"):
                current = self.conn.execute(
                    "SELECT ml_snap_id FROM snaps WHERE id = ?", (snap_id,)
                ).fetchone()
                if current is not None:
                    ml_snap_id = current["ml_snap_id"] if hasattr(current, "keys") else current[0]
            owned = self._decision_id_for_snap(ml_snap_id)
            if owned is not None:
                fields["ml_decision_id"] = owned
        present = self._table_columns("snaps")
        cols = [(k, fields[k]) for k in allowed if k in fields and k in present]
        if not cols:
            return
        sets = ", ".join(f"{k} = ?" for k, _ in cols)
        self.conn.execute(
            f"UPDATE snaps SET {sets} WHERE id = ?",
            [v for _, v in cols] + [snap_id],
        )
        if "ml_decision_id" in fields or "executed_play" in fields:
            self._record_play_mismatch(snap_id)
        self.conn.commit()

    def delete_snap(self, snap_id: int) -> None:
        row = self.conn.execute(
            "SELECT ml_snap_id FROM snaps WHERE id = ?", (snap_id,)
        ).fetchone()
        ml_snap = None
        if row is not None:
            ml_snap = row["ml_snap_id"] if hasattr(row, "keys") else row[0]
        self.conn.execute("DELETE FROM snaps WHERE id = ?", (snap_id,))
        if ml_snap:
            self.void_ml_outcome(str(ml_snap), reason="snap_deleted")
        self.conn.commit()

    def void_ml_outcome(self, snap_id: str, *, reason: str = "voided") -> int:
        """Archive then remove the active outcome for ``snap_id``.

        Preserves an audit trail so corrections/undos do not leave conflicting
        active labels while remaining inspectable.
        """
        if not snap_id:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        removed = 0
        try:
            rows = self.conn.execute(
                "SELECT * FROM ml_outcomes WHERE snap_id = ?", (snap_id,)
            ).fetchall()
        except sqlite3.Error:
            return 0
        for row in rows:
            d = dict(row)
            try:
                self.conn.execute(
                    """
                    INSERT INTO ml_outcome_audit (
                        snap_id, game_id, decision_id, executed_status,
                        executed_formation, executed_play, executed_verification,
                        outcome_json, void_reason, voided_ts, original_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        d.get("snap_id"),
                        d.get("game_id"),
                        d.get("decision_id"),
                        d.get("executed_status"),
                        d.get("executed_formation"),
                        d.get("executed_play"),
                        d.get("executed_verification"),
                        d.get("outcome_json"),
                        reason,
                        now,
                        d.get("id"),
                    ),
                )
            except sqlite3.Error:
                pass
            self.conn.execute("DELETE FROM ml_outcomes WHERE id = ?", (d["id"],))
            removed += 1
        self.conn.commit()
        return removed

    def get_idempotent_response(self, request_key: str) -> dict[str, Any] | None:
        if not request_key:
            return None
        try:
            row = self.conn.execute(
                "SELECT response_json FROM api_idempotency WHERE request_key = ?",
                (request_key,),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        raw = row["response_json"] if hasattr(row, "keys") else row[0]
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["idempotent_replay"] = True
            return payload
        return None

    def store_idempotent_response(
        self,
        request_key: str,
        *,
        endpoint: str,
        response: Mapping[str, Any] | dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        if not request_key:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO api_idempotency
                    (request_key, session_id, endpoint, response_json, created_ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_key,
                    session_id,
                    endpoint,
                    json.dumps(dict(response), default=str),
                    now,
                ),
            )
            self.conn.commit()
        except sqlite3.Error:
            pass

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
            if "recovered" not in gs_cols:
                try:
                    self.conn.execute(
                        "ALTER TABLE game_sessions ADD COLUMN recovered INTEGER NOT NULL DEFAULT 0"
                    )
                    self.conn.commit()
                except sqlite3.Error:
                    pass
            if "owner_pid" not in gs_cols:
                try:
                    self.conn.execute("ALTER TABLE game_sessions ADD COLUMN owner_pid INTEGER")
                    self.conn.commit()
                except sqlite3.Error:
                    pass
            if "heartbeat_ts" not in gs_cols:
                try:
                    self.conn.execute("ALTER TABLE game_sessions ADD COLUMN heartbeat_ts TEXT")
                    self.conn.commit()
                except sqlite3.Error:
                    pass

    def _migrate_ml_tables(self) -> None:
        """Additive Madden ML decision/outcome tables. Safe to run repeatedly."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ml_decisions (
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
            CREATE INDEX IF NOT EXISTS idx_ml_decisions_snap
                ON ml_decisions(snap_id);
            CREATE INDEX IF NOT EXISTS idx_ml_decisions_game
                ON ml_decisions(game_id);
            CREATE TABLE IF NOT EXISTS ml_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snap_id TEXT NOT NULL,
                game_id TEXT,
                decision_id INTEGER,
                executed_status TEXT,
                executed_formation TEXT,
                executed_play TEXT,
                executed_verification TEXT,
                outcome_json TEXT,
                created_ts TEXT NOT NULL,
                FOREIGN KEY (decision_id) REFERENCES ml_decisions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_ml_outcomes_snap
                ON ml_outcomes(snap_id);
            CREATE TABLE IF NOT EXISTS ml_outcome_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snap_id TEXT NOT NULL,
                game_id TEXT,
                decision_id INTEGER,
                executed_status TEXT,
                executed_formation TEXT,
                executed_play TEXT,
                executed_verification TEXT,
                outcome_json TEXT,
                void_reason TEXT,
                voided_ts TEXT NOT NULL,
                original_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS api_idempotency (
                request_key TEXT PRIMARY KEY,
                session_id TEXT,
                endpoint TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_ts TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
        for stmt in (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_decisions_snap_unique "
            "ON ml_decisions(snap_id) WHERE snap_id IS NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_outcomes_snap_unique "
            "ON ml_outcomes(snap_id) WHERE snap_id IS NOT NULL",
        ):
            try:
                self.conn.execute(stmt)
                self.conn.commit()
            except sqlite3.Error:
                pass
        self._migrate_ml_snap_columns()
        # Older DBs may lack audit / idempotency tables added after first migrate.
        for stmt in (
            """
            CREATE TABLE IF NOT EXISTS ml_outcome_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snap_id TEXT NOT NULL,
                game_id TEXT,
                decision_id INTEGER,
                executed_status TEXT,
                executed_formation TEXT,
                executed_play TEXT,
                executed_verification TEXT,
                outcome_json TEXT,
                void_reason TEXT,
                voided_ts TEXT NOT NULL,
                original_id INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS api_idempotency (
                request_key TEXT PRIMARY KEY,
                session_id TEXT,
                endpoint TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_ts TEXT NOT NULL
            )
            """,
        ):
            try:
                self.conn.execute(stmt)
                self.conn.commit()
            except sqlite3.Error:
                pass
        self._migrate_madden_call_outcome()

    def _table_columns(self, table: str) -> set[str]:
        try:
            return {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error:
            return set()

    def _add_columns(self, table: str, additions: dict[str, str]) -> None:
        present = self._table_columns(table)
        if not present:
            return
        for name, typ in additions.items():
            if name in present:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
        self.conn.commit()

    def _set_columns(self, table: str, row_id: int, values: dict[str, Any]) -> None:
        """Update columns that exist. Missing keys and missing columns are skipped."""
        present = self._table_columns(table)
        pairs = [(key, value) for key, value in values.items() if key in present]
        if not pairs:
            return
        assignments = ", ".join(f"{key} = ?" for key, _value in pairs)
        self.conn.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ?",
            [value for _key, value in pairs] + [row_id],
        )

    def _migrate_madden_call_outcome(self) -> None:
        """Additive Madden call/outcome columns. Safe to run more than once.

        Older databases keep every existing row. New columns are NULL until a
        later log writes them. This does not copy a recommendation into an
        executed adjustment, and it does not change coaching defaults.
        """
        from cfb_coach.madden.model.call_outcome_log import (
            DECISION_COLUMNS,
            OUTCOME_COLUMNS,
            SCHEMA_META_KEY,
            SCHEMA_VERSION,
            SNAP_COLUMNS,
        )

        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "ml_decisions" in tables:
            self._add_columns("ml_decisions", DECISION_COLUMNS)
        if "ml_outcomes" in tables:
            self._add_columns("ml_outcomes", OUTCOME_COLUMNS)
        if "snaps" in tables:
            self._add_columns("snaps", SNAP_COLUMNS)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_run_config (
                session_id TEXT PRIMARY KEY,
                recorded_ts TEXT NOT NULL,
                mode TEXT,
                model_id TEXT,
                flags_json TEXT,
                git_commit TEXT,
                git_branch TEXT,
                playbook_version TEXT
            )
            """
        )
        if "meta" in tables or "meta" in {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }:
            self.conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_META_KEY, SCHEMA_VERSION),
            )
        self.conn.commit()

    def _record_play_mismatch(self, snap_row_id: int) -> None:
        """Flag a snap whose executed play differs from the linked decision.

        An unknown executed play is not a mismatch. A real disagreement is
        stored on ``play_mismatch`` and logged. It is never left unmarked.
        """
        if "play_mismatch" not in self._table_columns("snaps"):
            return
        if "final_play" not in self._table_columns("ml_decisions"):
            return
        snap = self.conn.execute(
            "SELECT ml_decision_id, executed_play FROM snaps WHERE id = ?",
            (snap_row_id,),
        ).fetchone()
        if snap is None:
            return
        decision_id = snap["ml_decision_id"] if hasattr(snap, "keys") else snap[0]
        executed = snap["executed_play"] if hasattr(snap, "keys") else snap[1]
        executed_text = "" if executed is None else str(executed).strip()
        flag = None
        final_text = ""
        if decision_id is not None and executed_text:
            decision = self.conn.execute(
                "SELECT final_play FROM ml_decisions WHERE id = ?",
                (int(decision_id),),
            ).fetchone()
            final = None if decision is None else (
                decision["final_play"] if hasattr(decision, "keys") else decision[0]
            )
            final_text = "" if final is None else str(final).strip()
            if final_text:
                flag = 0 if executed_text == final_text else 1
        self.conn.execute(
            "UPDATE snaps SET play_mismatch = ? WHERE id = ?",
            (flag, snap_row_id),
        )
        if flag == 1:
            logging.getLogger(__name__).warning(
                "play mismatch snap_row=%s decision_id=%s executed_play=%s final_play=%s",
                snap_row_id,
                decision_id,
                executed_text,
                final_text,
            )

    def _decision_id_for_snap(self, snap_id: str | None) -> int | None:
        """The decision sealed for ``snap_id``, if one was logged."""
        if not snap_id or "snap_id" not in self._table_columns("ml_decisions"):
            return None
        row = self.conn.execute(
            "SELECT id FROM ml_decisions WHERE snap_id = ? ORDER BY id DESC LIMIT 1",
            (snap_id,),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"] if hasattr(row, "keys") else row[0])

    def _attach_snap_decision(self, snap_id: str | None, decision_id: int | None) -> None:
        """Point the snap for ``snap_id`` at the decision that was sealed for it."""
        if not snap_id or decision_id is None:
            return
        if "ml_decision_id" not in self._table_columns("snaps"):
            return
        self.conn.execute(
            """
            UPDATE snaps
            SET ml_decision_id = ?
            WHERE ml_snap_id = ?
              AND (ml_decision_id IS NULL OR ml_decision_id != ?)
            """,
            (int(decision_id), snap_id, int(decision_id)),
        )
        row = self.conn.execute(
            "SELECT id FROM snaps WHERE ml_snap_id = ?", (snap_id,)
        ).fetchone()
        if row is not None:
            self._record_play_mismatch(int(row["id"] if hasattr(row, "keys") else row[0]))

    def _link_decision(self, snap_id: str | None, decision_id: int | None) -> None:
        columns = self._table_columns("ml_decisions")
        if "snap_link" not in columns:
            return
        if "orphan_reason" in columns:
            assignment = "snap_link = ?, orphan_reason = NULL"
        else:
            assignment = "snap_link = ?"
        if decision_id is not None:
            self._attach_snap_decision(snap_id, int(decision_id))
            self.conn.execute(
                f"UPDATE ml_decisions SET {assignment} WHERE id = ?",
                ("linked", int(decision_id)),
            )
        elif snap_id:
            owner = self._decision_id_for_snap(snap_id)
            if owner is not None:
                self._attach_snap_decision(snap_id, owner)
            self.conn.execute(
                f"UPDATE ml_decisions SET {assignment} WHERE snap_id = ?",
                ("linked", snap_id),
            )
        self.conn.commit()

    def mark_call_no_snap(self, decision_id: int | None, *, reason: str = "no_result") -> bool:
        """Mark one call that still has no snap or outcome, and keep the row.

        ``reason`` is ``superseded`` when a later call replaced this one, or
        ``no_result`` when the result never arrived. A call that already has a
        snap or outcome row is linked instead, and its reason is cleared.
        """
        from cfb_coach.madden.model.call_outcome_log import NO_RESULT, SUPERSEDED

        if decision_id is None:
            return False
        stored_reason = SUPERSEDED if reason == SUPERSEDED else NO_RESULT
        self._migrate_madden_call_outcome()
        row = self.conn.execute(
            "SELECT id, snap_id FROM ml_decisions WHERE id = ?",
            (int(decision_id),),
        ).fetchone()
        if row is None:
            return False
        snap_id = row["snap_id"] if hasattr(row, "keys") else row[1]
        if self._call_has_snap_or_outcome(snap_id, int(decision_id)):
            self._link_decision(snap_id, int(decision_id))
            return False
        columns = self._table_columns("ml_decisions")
        if "orphan_reason" in columns:
            self.conn.execute(
                "UPDATE ml_decisions SET snap_link = ?, orphan_reason = ? WHERE id = ?",
                ("no_snap_recorded", stored_reason, int(decision_id)),
            )
        else:
            self.conn.execute(
                "UPDATE ml_decisions SET snap_link = ? WHERE id = ?",
                ("no_snap_recorded", int(decision_id)),
            )
        self.conn.commit()
        return True

    def _call_has_snap_or_outcome(self, snap_id: str | None, decision_id: int | None) -> bool:
        if snap_id and "ml_snap_id" in self._table_columns("snaps"):
            found = self.conn.execute(
                "SELECT 1 FROM snaps WHERE ml_snap_id = ? LIMIT 1",
                (snap_id,),
            ).fetchone()
            if found is not None:
                return True
        if snap_id and "ml_outcomes" in {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }:
            found = self.conn.execute(
                "SELECT 1 FROM ml_outcomes WHERE snap_id = ? LIMIT 1",
                (snap_id,),
            ).fetchone()
            if found is not None:
                return True
        if decision_id is not None and "decision_id" in self._table_columns("ml_outcomes"):
            found = self.conn.execute(
                "SELECT 1 FROM ml_outcomes WHERE decision_id = ? LIMIT 1",
                (int(decision_id),),
            ).fetchone()
            if found is not None:
                return True
        return False

    def _same_logged_game(self, left: Any, right: Any) -> bool:
        def keys(row: Any) -> set[str]:
            found = set()
            for name in ("session_id", "game_id"):
                value = row[name] if hasattr(row, "keys") else None
                if value:
                    found.add(str(value))
            return found

        return bool(keys(left) & keys(right))

    def _later_sealed_call(self, row: Any, rows: list[Any]) -> bool:
        """True when a later call in the same game was sealed after ``row``."""
        this_id = int(row["id"])
        this_seq = row["snap_seq"]
        for other in rows:
            other_id = int(other["id"])
            if other_id == this_id or not self._same_logged_game(row, other):
                continue
            other_seq = other["snap_seq"]
            if this_seq is not None and other_seq is not None and other_seq != this_seq:
                if int(other_seq) > int(this_seq):
                    return True
                continue
            if other_id > this_id:
                return True
        return False

    def reconcile_call_links(self, session_id: str | None = None) -> dict[str, int]:
        """Link every executed call to its snap, or record why it has none.

        A snap whose ``ml_snap_id`` matches a decision gets that decision's id.
        A call with no snap is ``superseded`` when a later call in the game was
        sealed, and ``no_result`` when the result never arrived. Rows are not
        deleted. Idempotent. A second call leaves the same statuses in place.
        """
        from cfb_coach.madden.model.call_outcome_log import NO_RESULT, SUPERSEDED

        self._migrate_madden_call_outcome()
        sql = "SELECT id, snap_id, snap_seq, session_id, game_id FROM ml_decisions"
        params: list[Any] = []
        if session_id:
            sql += " WHERE session_id = ? OR game_id = ?"
            params = [session_id, session_id]
        rows = list(self.conn.execute(sql, params))
        for row in rows:
            decision_id = int(row["id"])
            snap_id = row["snap_id"]
            if snap_id:
                self._attach_snap_decision(snap_id, decision_id)
        linked = 0
        closed = 0
        has_reason = "orphan_reason" in self._table_columns("ml_decisions")
        for row in rows:
            decision_id = int(row["id"])
            snap_id = row["snap_id"]
            if self._call_has_snap_or_outcome(snap_id, decision_id):
                if has_reason:
                    self.conn.execute(
                        "UPDATE ml_decisions SET snap_link = ?, orphan_reason = NULL WHERE id = ?",
                        ("linked", decision_id),
                    )
                else:
                    self.conn.execute(
                        "UPDATE ml_decisions SET snap_link = ? WHERE id = ?",
                        ("linked", decision_id),
                    )
                linked += 1
            else:
                reason = SUPERSEDED if self._later_sealed_call(row, rows) else NO_RESULT
                if has_reason:
                    self.conn.execute(
                        "UPDATE ml_decisions SET snap_link = ?, orphan_reason = ? WHERE id = ?",
                        ("no_snap_recorded", reason, decision_id),
                    )
                else:
                    self.conn.execute(
                        "UPDATE ml_decisions SET snap_link = ? WHERE id = ?",
                        ("no_snap_recorded", decision_id),
                    )
                closed += 1
        self.conn.commit()
        return {
            "calls": len(rows),
            "linked": linked,
            "no_snap_recorded": closed,
            "orphans": len(self.orphan_call_rows(session_id)),
        }

    def orphan_call_rows(self, session_id: str | None = None) -> list[sqlite3.Row]:
        """Calls with neither a snap/outcome link nor ``no_snap_recorded``."""
        if "snap_link" not in self._table_columns("ml_decisions"):
            sql = "SELECT id, snap_id, session_id, game_id FROM ml_decisions"
            params: list[Any] = []
            if session_id:
                sql += " WHERE session_id = ? OR game_id = ?"
                params = [session_id, session_id]
            return list(self.conn.execute(sql, params))
        columns = self._table_columns("ml_decisions")
        closed = "d.snap_link = 'no_snap_recorded'"
        if "orphan_reason" in columns:
            closed += " AND d.orphan_reason IN ('superseded', 'no_result')"
        sql = (
            "SELECT d.id, d.snap_id, d.snap_link, d.session_id, d.game_id "
            "FROM ml_decisions d WHERE NOT ("
            f"({closed} "
            "AND NOT EXISTS (SELECT 1 FROM snaps s WHERE s.ml_snap_id = d.snap_id) "
            "AND NOT EXISTS (SELECT 1 FROM snaps s WHERE s.ml_decision_id = d.id) "
            "AND NOT EXISTS (SELECT 1 FROM ml_outcomes o WHERE o.snap_id = d.snap_id "
            "OR o.decision_id = d.id)) "
            "OR d.snap_link = 'linked' AND ("
            "EXISTS (SELECT 1 FROM snaps s WHERE s.ml_decision_id = d.id) "
            "OR EXISTS (SELECT 1 FROM snaps s WHERE s.ml_snap_id = d.snap_id "
            "AND (s.ml_decision_id IS NULL OR s.ml_decision_id = d.id)) "
            "OR EXISTS (SELECT 1 FROM ml_outcomes o WHERE o.snap_id = d.snap_id "
            "OR o.decision_id = d.id))"
            ")"
        )
        params = []
        if session_id:
            sql += " AND (d.session_id = ? OR d.game_id = ?)"
            params = [session_id, session_id]
        return list(self.conn.execute(sql, params))

    def stamp_call_context(
        self,
        decision_id: int,
        context: dict[str, Any] | None = None,
        *,
        recommended_adjustment_id: str | None = None,
    ) -> None:
        """Write pre-snap context onto a decision. Does not set an execution."""
        self._migrate_madden_call_outcome()
        values: dict[str, Any] = {}
        for key in ("drive_id", "quarter", "game_clock", "score_for", "score_against"):
            if context and key in context and context[key] is not None:
                values[key] = context[key]
        if recommended_adjustment_id:
            values["recommended_adjustment_id"] = recommended_adjustment_id
        if not values:
            return
        self._set_columns("ml_decisions", int(decision_id), values)
        self.conn.commit()

    def log_ml_run_config(
        self,
        session_id: str,
        *,
        mode: str | None,
        model_id: str | None,
        flags: dict[str, Any] | None,
        git_commit: str | None,
        git_branch: str | None,
        playbook_version: str | None,
    ) -> None:
        """Record the session's run config once. A repeat keeps the first row."""
        self._migrate_madden_call_outcome()
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO ml_run_config (
                session_id, recorded_ts, mode, model_id, flags_json,
                git_commit, git_branch, playbook_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (
                session_id,
                now,
                mode,
                model_id,
                json.dumps(flags or {}, sort_keys=True),
                git_commit,
                git_branch,
                playbook_version,
            ),
        )
        self.conn.commit()

    def _migrate_ml_snap_columns(self) -> None:
        """Additive snap columns for ML identity and execution verification."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(snaps)").fetchall()}
        additions = {
            "ml_snap_id": "TEXT",
            "snap_seq": "INTEGER",
            "executed_status": "TEXT",
            "executed_formation": "TEXT",
            "executed_play": "TEXT",
            "executed_macro": "TEXT",
            "executed_verification": "TEXT",
            "ml_decision_id": "INTEGER",
            "data_source": "TEXT",
        }
        for name, typ in additions.items():
            if name in cols:
                continue
            try:
                self.conn.execute(f"ALTER TABLE snaps ADD COLUMN {name} {typ}")
                self.conn.commit()
            except sqlite3.Error:
                pass
        try:
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_snaps_ml_snap_id "
                "ON snaps(ml_snap_id) WHERE ml_snap_id IS NOT NULL"
            )
            self.conn.commit()
        except sqlite3.Error:
            pass

    def log_ml_decision(self, decision: Any, *, agree: int | None = None) -> int:
        """Persist one CoachingDecision. Idempotent on snap_id when set."""
        from cfb_coach.madden.model.schema import to_dict

        d = to_dict(decision) if hasattr(decision, "__dataclass_fields__") else dict(decision)
        now = datetime.now(timezone.utc).isoformat()
        heuristic = d.get("heuristic_pick") or {}
        final = d.get("final_pick") or {}
        shadow = d.get("shadow_pick") or {}
        snap_id = d.get("snap_id")
        if snap_id:
            existing = self.conn.execute(
                "SELECT id FROM ml_decisions WHERE snap_id = ?", (snap_id,)
            ).fetchone()
            if existing:
                row_id = int(existing["id"] if hasattr(existing, "keys") else existing[0])
                self.conn.execute(
                    """
                    UPDATE ml_decisions SET
                        decision_ts=?, game_id=?, session_id=?, snap_seq=?, mode=?,
                        model_version=?, policy_version=?, policy_source=?,
                        heuristic_formation=?, heuristic_play=?,
                        final_formation=?, final_play=?,
                        shadow_formation=?, shadow_play=?, shadow_status=?, agree=?,
                        candidates_json=?, decision_json=?, latency_ms_model=?
                    WHERE id=?
                    """,
                    (
                        d.get("decision_ts") or now,
                        d.get("game_id"),
                        d.get("session_id"),
                        d.get("snap_seq"),
                        d.get("mode") or "heuristic",
                        d.get("model_version"),
                        d.get("policy_version"),
                        d.get("policy_source"),
                        heuristic.get("formation") if isinstance(heuristic, dict) else None,
                        heuristic.get("play") if isinstance(heuristic, dict) else None,
                        final.get("formation") if isinstance(final, dict) else None,
                        final.get("play") if isinstance(final, dict) else None,
                        shadow.get("formation") if isinstance(shadow, dict) else None,
                        shadow.get("play") if isinstance(shadow, dict) else None,
                        d.get("shadow_status"),
                        agree,
                        json.dumps(d.get("candidates"), default=str) if d.get("candidates") is not None else None,
                        json.dumps(d, default=str),
                        d.get("latency_ms_model"),
                        row_id,
                    ),
                )
                self._finish_decision_log(row_id, d)
                return row_id
        cur = self.conn.execute(
            """
            INSERT INTO ml_decisions (
                decision_ts, game_id, snap_id, session_id, snap_seq, mode,
                model_version, policy_version, policy_source,
                heuristic_formation, heuristic_play,
                final_formation, final_play,
                shadow_formation, shadow_play, shadow_status, agree,
                candidates_json, decision_json, latency_ms_model, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d.get("decision_ts") or now,
                d.get("game_id"),
                snap_id,
                d.get("session_id"),
                d.get("snap_seq"),
                d.get("mode") or "heuristic",
                d.get("model_version"),
                d.get("policy_version"),
                d.get("policy_source"),
                heuristic.get("formation") if isinstance(heuristic, dict) else None,
                heuristic.get("play") if isinstance(heuristic, dict) else None,
                final.get("formation") if isinstance(final, dict) else None,
                final.get("play") if isinstance(final, dict) else None,
                shadow.get("formation") if isinstance(shadow, dict) else None,
                shadow.get("play") if isinstance(shadow, dict) else None,
                d.get("shadow_status"),
                agree,
                json.dumps(d.get("candidates"), default=str) if d.get("candidates") is not None else None,
                json.dumps(d, default=str),
                d.get("latency_ms_model"),
                now,
            ),
        )
        row_id = int(cur.lastrowid)
        self._finish_decision_log(row_id, d)
        return row_id

    def _finish_decision_log(self, row_id: int, decision: dict[str, Any]) -> None:
        from cfb_coach.madden.model.call_outcome_log import decision_log_values

        values = decision_log_values(decision)
        for key in ("drive_id", "quarter", "game_clock", "score_for", "score_against"):
            if key in decision and decision[key] is not None:
                values[key] = decision[key]
        self._set_columns("ml_decisions", row_id, values)
        snap_id = decision.get("snap_id")
        if self._call_has_snap_or_outcome(snap_id, row_id):
            self._attach_snap_decision(snap_id, row_id)
            self._link_decision(snap_id, row_id)
        self.conn.commit()

    def log_ml_outcome(
        self,
        *,
        snap_id: str,
        game_id: str | None = None,
        decision_id: int | None = None,
        executed_status: str = "unknown",
        executed_formation: str | None = None,
        executed_play: str | None = None,
        executed_verification: str = "unknown",
        outcome: Any = None,
        replace: bool = True,
        yards: Any = _UNSET,
        penalty: str | None = None,
        drive_id: str | None = None,
        quarter: int | None = None,
        game_clock: str | None = None,
        score_for: int | None = None,
        score_against: int | None = None,
        accepted: str | None = None,
        recommended_adjustment_id: str | None = None,
        executed_adjustment_id: str | None = None,
        adjustment_confirmed: bool = False,
    ) -> int:
        """Store a verified (or explicitly unknown) execution for one snap.

        Corrections update the same ``snap_id`` row when ``replace`` is true.
        """
        now = datetime.now(timezone.utc).isoformat()
        status = str(executed_status or "unknown").lower()
        if status != "identified":
            executed_formation = None
            executed_play = None
            executed_verification = "unknown"
        payload = None
        if outcome is not None:
            if hasattr(outcome, "__dataclass_fields__"):
                from cfb_coach.madden.model.schema import to_dict

                payload = json.dumps(to_dict(outcome), default=str)
            else:
                payload = json.dumps(outcome, default=str)
        if replace:
            existing = self.conn.execute(
                "SELECT id FROM ml_outcomes WHERE snap_id = ?", (snap_id,)
            ).fetchone()
            if existing:
                row_id = int(existing["id"] if hasattr(existing, "keys") else existing[0])
                self.conn.execute(
                    """
                    UPDATE ml_outcomes SET
                        game_id=?, decision_id=?, executed_status=?,
                        executed_formation=?, executed_play=?, executed_verification=?,
                        outcome_json=?
                    WHERE id=?
                    """,
                    (
                        game_id,
                        decision_id,
                        status,
                        executed_formation,
                        executed_play,
                        executed_verification,
                        payload,
                        row_id,
                    ),
                )
                self._finish_outcome_log(
                    row_id,
                    snap_id=snap_id,
                    decision_id=decision_id,
                    outcome=outcome,
                    yards=yards,
                    penalty=penalty,
                    drive_id=drive_id,
                    quarter=quarter,
                    game_clock=game_clock,
                    score_for=score_for,
                    score_against=score_against,
                    accepted=accepted,
                    recommended_adjustment_id=recommended_adjustment_id,
                    executed_adjustment_id=executed_adjustment_id,
                    adjustment_confirmed=adjustment_confirmed,
                )
                return row_id
        cur = self.conn.execute(
            """
            INSERT INTO ml_outcomes (
                snap_id, game_id, decision_id, executed_status,
                executed_formation, executed_play, executed_verification,
                outcome_json, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snap_id,
                game_id,
                decision_id,
                status,
                executed_formation,
                executed_play,
                executed_verification,
                payload,
                now,
            ),
        )
        row_id = int(cur.lastrowid)
        self._finish_outcome_log(
            row_id,
            snap_id=snap_id,
            decision_id=decision_id,
            outcome=outcome,
            yards=yards,
            penalty=penalty,
            drive_id=drive_id,
            quarter=quarter,
            game_clock=game_clock,
            score_for=score_for,
            score_against=score_against,
            accepted=accepted,
            recommended_adjustment_id=recommended_adjustment_id,
            executed_adjustment_id=executed_adjustment_id,
            adjustment_confirmed=adjustment_confirmed,
        )
        return row_id

    def _finish_outcome_log(
        self,
        row_id: int,
        *,
        snap_id: str,
        decision_id: int | None,
        outcome: Any,
        yards: Any,
        penalty: str | None,
        drive_id: str | None,
        quarter: int | None,
        game_clock: str | None,
        score_for: int | None,
        score_against: int | None,
        accepted: str | None,
        recommended_adjustment_id: str | None,
        executed_adjustment_id: str | None,
        adjustment_confirmed: bool,
    ) -> None:
        from cfb_coach.madden.model.call_outcome_log import (
            coerce_logged_yards,
            executed_adjustment_for_log,
            penalty_for_log,
        )

        result_text = None
        if isinstance(outcome, dict):
            result_text = outcome.get("result")
        elif outcome is not None and hasattr(outcome, "raw"):
            result_text = getattr(outcome, "raw", None)
        result = None if result_text is None else str(result_text)
        explicit = None if yards is _UNSET else yards
        values: dict[str, Any] = {
            "yards": coerce_logged_yards(result, explicit),
            "penalty": penalty_for_log(result, penalty),
            "executed_adjustment_id": executed_adjustment_for_log(
                executed_adjustment_id, confirmed=adjustment_confirmed
            ),
        }
        for key, value in (
            ("drive_id", drive_id),
            ("quarter", quarter),
            ("game_clock", game_clock),
            ("score_for", score_for),
            ("score_against", score_against),
            ("accepted", accepted),
            ("recommended_adjustment_id", recommended_adjustment_id),
        ):
            if value is not None:
                values[key] = value
        self._set_columns("ml_outcomes", row_id, values)
        self._link_decision(snap_id, decision_id)
        if snap_id and "yards" in self._table_columns("snaps"):
            snap = self.conn.execute(
                "SELECT id FROM snaps WHERE ml_snap_id = ?",
                (snap_id,),
            ).fetchone()
            if snap is not None:
                snap_row = int(snap["id"] if hasattr(snap, "keys") else snap[0])
                self._set_columns("snaps", snap_row, values)
        self.conn.commit()

    def list_ml_decisions(self, *, limit: int = 100) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM ml_decisions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )

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

    def record_score_reconcile(self, session_id: str, rec: dict[str, Any]) -> None:
        """Store the snap-log vs session score gap. Additive columns only."""
        self._add_columns(
            "game_sessions",
            {
                "snaplog_score": "TEXT",
                "score_gap_for": "INTEGER",
                "score_gap_against": "INTEGER",
                "score_reconcile": "TEXT",
            },
        )
        self.conn.execute(
            "UPDATE game_sessions SET snaplog_score = ?, score_gap_for = ?, "
            "score_gap_against = ?, score_reconcile = ? WHERE session_id = ?",
            (
                rec.get("snaplog_score"),
                rec.get("gap_for"),
                rec.get("gap_against"),
                rec.get("status") or "unknown",
                session_id,
            ),
        )
        self.conn.commit()

    def end_game_session(
        self,
        session_id: str,
        *,
        play_count: int | None = None,
        result_wl: str | None = None,
        score: str | None = None,
        recovered: bool | None = None,
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
        if "recovered" not in cols:
            try:
                self.conn.execute(
                    "ALTER TABLE game_sessions ADD COLUMN recovered INTEGER NOT NULL DEFAULT 0"
                )
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
        if recovered is not None:
            sets.append("recovered = ?")
            params.append(1 if recovered else 0)
        params.append(session_id)
        self.conn.execute(
            f"UPDATE game_sessions SET {', '.join(sets)} WHERE session_id = ?",
            params,
        )
        self.conn.commit()
        try:
            self.reconcile_call_links(session_id)
        except sqlite3.Error:
            pass

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


