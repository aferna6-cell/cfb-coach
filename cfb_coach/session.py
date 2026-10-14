"""Game session — start from watch --opponent/--dynasty; id on all plays.

A live game inserts one ``game_sessions`` row at start. Ending the game writes
``ended_ts``, ``result_wl``, ``score``, and ``play_count``. If the process
exits before that, an exit handler closes the open row from the last logged
score. A hard crash skips the handler; the next start recovers every still-open
row, using that same score, and flags it ``recovered``.
"""

from __future__ import annotations

import atexit
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GameSession:
    session_id: str
    opponent_id: str
    dynasty: str = "alabama"
    started_ts: str = field(default_factory=_now_iso)
    ended_ts: str | None = None
    notes: str = ""
    play_count: int = 0

    @property
    def game_id(self) -> str:
        return self.session_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "game_id": self.game_id,
            "opponent_id": self.opponent_id,
            "dynasty": self.dynasty,
            "started_ts": self.started_ts,
            "ended_ts": self.ended_ts,
            "notes": self.notes,
            "play_count": self.play_count,
        }


def start_session(
    opponent_id: str,
    *,
    dynasty: str = "alabama",
    db: Any | None = None,
    notes: str = "",
) -> GameSession:
    """Create a session; optionally persist via CoachDB.start_game_session."""
    sid = uuid.uuid4().hex[:16]
    sess = GameSession(
        session_id=sid,
        opponent_id=(opponent_id or "unknown").lower(),
        dynasty=dynasty or "alabama",
        notes=notes,
    )
    if db is not None:
        try:
            db.start_game_session(sess)
        except Exception:
            pass
    return sess


def end_session(sess: GameSession, db: Any | None = None) -> GameSession:
    sess.ended_ts = _now_iso()
    if db is not None:
        try:
            db.end_game_session(sess.session_id, play_count=sess.play_count)
        except Exception:
            pass
    disarm_session_exit(sess.session_id)
    return sess


_NOTE_SCORE = re.compile(r"score_us=(\d+)\s+score_them=(\d+)")
_ARMED: dict[str, Any] = {}
_ATEXIT_INSTALLED = False


def _columns(db: Any, table: str) -> set[str]:
    try:
        rows = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _session_row(db: Any, session_id: str) -> Any:
    try:
        return db.conn.execute(
            "SELECT * FROM game_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return None


def session_is_open(row: Any) -> bool:
    """True when the row exists and ``ended_ts`` was never written."""
    if row is None:
        return False
    try:
        ended = row["ended_ts"]
    except (KeyError, IndexError, TypeError):
        return False
    return not (ended and str(ended).strip())


def logged_play_count(db: Any, session_id: str) -> int:
    """Plays already stored for this session. Snaps win; play records are the fallback."""
    if db is None or not session_id:
        return 0
    snap_cols = _columns(db, "snaps")
    if "session_id" in snap_cols:
        try:
            row = db.conn.execute(
                "SELECT COUNT(*) FROM snaps WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            count = int(row[0] if row is not None else 0)
        except sqlite3.Error:
            count = 0
        if count:
            return count
    try:
        row = db.conn.execute(
            "SELECT COUNT(*) FROM play_records WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] if row is not None else 0)


def _wl_for(us: int, them: int) -> str:
    if us > them:
        return "win"
    if us < them:
        return "loss"
    return "tie"


def _score_from_row(row: Any, columns: set[str]) -> tuple[str, str] | None:
    us = them = None
    if "score_for" in columns and "score_against" in columns:
        raw_us = row["score_for"]
        raw_them = row["score_against"]
        if raw_us is not None and raw_them is not None:
            try:
                us = int(raw_us)
                them = int(raw_them)
            except (TypeError, ValueError):
                us = them = None
    if us is None and "notes" in columns:
        match = _NOTE_SCORE.search(str(row["notes"] or ""))
        if match:
            us = int(match.group(1))
            them = int(match.group(2))
    if us is None or them is None:
        return None
    return f"{us}-{them}", _wl_for(us, them)


def last_logged_score(db: Any, session_id: str) -> tuple[str | None, str | None]:
    """Latest us-them score stored on a snap, and the win/loss/tie it implies.

    ``score_for`` / ``score_against`` are the logged pair. A snap that only has
    ``score_us=`` / ``score_them=`` in notes still counts. Snaps with no score
    are skipped, so a later blank snap does not erase the last real score.
    """
    if db is None or not session_id:
        return None, None
    columns = _columns(db, "snaps")
    if "session_id" not in columns:
        return None, None
    selected = ["id"]
    if "score_for" in columns and "score_against" in columns:
        selected.extend(("score_for", "score_against"))
    if "notes" in columns:
        selected.append("notes")
    try:
        rows = db.conn.execute(
            f"SELECT {', '.join(selected)} FROM snaps WHERE session_id = ? ORDER BY id DESC",
            (session_id,),
        ).fetchall()
    except sqlite3.Error:
        return None, None
    for row in rows:
        found = _score_from_row(row, set(selected))
        if found is not None:
            return found
    return None, None


def _flag_recovered(db: Any, session_id: str) -> None:
    row = _session_row(db, session_id)
    notes = "" if row is None else str(row["notes"] or "")
    if "recovered" not in notes.split():
        notes = (notes + " recovered").strip()
        db.conn.execute(
            "UPDATE game_sessions SET notes = ? WHERE session_id = ?",
            (notes, session_id),
        )
        db.conn.commit()


def close_open_session(
    db: Any,
    session_id: str,
    *,
    recovered: bool = False,
    result_wl: str | None = None,
    score: str | None = None,
    play_count: int | None = None,
) -> bool:
    """Write the close columns when ``ended_ts`` is still empty.

    Returns True only when this call closed the row. An explicit result and
    score win over the log. Otherwise the last logged score fills both.
    ``recovered`` is set only for the next-start path.
    """
    if db is None or not session_id:
        return False
    row = _session_row(db, session_id)
    if not session_is_open(row):
        disarm_session_exit(session_id)
        return False
    logged_score, inferred_wl = last_logged_score(db, session_id)
    count = logged_play_count(db, session_id) if play_count is None else int(play_count)
    final_wl = inferred_wl if result_wl is None else result_wl
    if score is None or not str(score).strip():
        final_score = logged_score
    else:
        final_score = str(score).strip()
    db.end_game_session(
        session_id,
        play_count=count,
        result_wl=final_wl,
        score=final_score,
        recovered=True if recovered else False,
    )
    if recovered:
        _flag_recovered(db, session_id)
    disarm_session_exit(session_id)
    return True


def recover_unclosed_sessions(db: Any) -> list[str]:
    """Close every open session in ``db`` from its log and flag each recovered."""
    if db is None:
        return []
    try:
        rows = db.conn.execute(
            "SELECT session_id FROM game_sessions "
            "WHERE ended_ts IS NULL OR TRIM(COALESCE(ended_ts, '')) = ''"
        ).fetchall()
    except sqlite3.Error:
        return []
    closed: list[str] = []
    for row in rows:
        session_id = str(row["session_id"])
        try:
            if close_open_session(db, session_id, recovered=True):
                closed.append(session_id)
        except Exception:
            continue
    return closed


def disarm_session_exit(session_id: str) -> None:
    _ARMED.pop(session_id, None)


def close_armed_sessions(session_id: str | None = None) -> None:
    """Exit handler. Closes armed sessions that are still open.

    Pass ``session_id`` to close one game. Omit it to close every armed game,
    which is what runs at process exit.
    """
    if session_id is None:
        items = list(_ARMED.items())
    else:
        db = _ARMED.get(session_id)
        items = [] if db is None else [(session_id, db)]
    for sid, db in items:
        try:
            close_open_session(db, sid, recovered=False)
        except Exception:
            _ARMED.pop(sid, None)


def arm_session_exit(db: Any, session_id: str) -> None:
    """Remember this open session and close it if the process exits."""
    global _ATEXIT_INSTALLED
    if db is None or not session_id:
        return
    _ARMED[session_id] = db
    if not _ATEXIT_INSTALLED:
        atexit.register(close_armed_sessions)
        _ATEXIT_INSTALLED = True
