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
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
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
            """
        )
        self.conn.commit()

    def _seed_if_empty(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'seeded'"
        ).fetchone()
        if row:
            return
        seed = load_seed()
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
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO snaps (
                ts, opponent_id, side, down, distance, yardline, quarter,
                situation_raw, our_call, formation, play, macro,
                result, coverage_seen, concept_seen, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


