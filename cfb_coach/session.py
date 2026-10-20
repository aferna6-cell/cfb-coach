"""Game session — start from watch --opponent/--dynasty; id on all plays.

A live game inserts one ``game_sessions`` row at start. Ending the game writes
``ended_ts``, ``result_wl``, ``score``, and ``play_count``. If the process
exits before that, an exit handler closes the open row from the last logged
score. A hard crash skips the handler; the next start recovers a row only when
its owner process is dead or its heartbeat is stale, using that same score, and
flags it ``recovered``. A later close by the owning process replaces that
recovered row.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

# A live owner refreshes faster than this. Older than this, the owner is stale.
HEARTBEAT_INTERVAL_S = 1.0
HEARTBEAT_STALE_S = 5.0
# Exit must not sit on SQLite's default 5s busy wait.
EXIT_BUSY_TIMEOUT_MS = 250
EXIT_BUSY_TIMEOUT_S = EXIT_BUSY_TIMEOUT_MS / 1000
_DEAD_PID = 2**31 - 1


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
_OWNERS: dict[str, "_LiveOwner"] = {}
_GUARD = threading.Lock()
_ATEXIT_INSTALLED = False
_SIGTERM_INSTALLED = False
_PREVIOUS_SIGTERM: Any = signal.SIG_DFL


@dataclass
class _LiveOwner:
    session_id: str
    db_path: str
    lock_file: Any
    stop: threading.Event
    thread: threading.Thread | None = None


def _columns(db: Any, table: str) -> set[str]:
    rows = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _session_row(db: Any, session_id: str) -> Any:
    """Return the row. A database error propagates; it is not an empty row."""
    return db.conn.execute(
        "SELECT * FROM game_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()


def _cell(row: Any, name: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        return row[name]
    except (KeyError, IndexError):
        return default


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
        row = db.conn.execute(
            "SELECT COUNT(*) FROM snaps WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        count = int(row[0] if row is not None else 0)
        if count:
            return count
    try:
        row = db.conn.execute(
            "SELECT COUNT(*) FROM play_records WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return 0
        raise
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
    rows = db.conn.execute(
        f"SELECT {', '.join(selected)} FROM snaps WHERE session_id = ? ORDER BY id DESC",
        (session_id,),
    ).fetchall()
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


def clear_recovered_note(db: Any, session_id: str) -> None:
    """Remove the recovered marker so a real close is not still labeled recovered."""
    if db is None or not session_id:
        return
    row = db.conn.execute(
        "SELECT notes FROM game_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return
    notes = str(row["notes"] or "")
    parts = [part for part in notes.split() if part != "recovered"]
    cleaned = " ".join(parts)
    if cleaned == notes:
        return
    db.conn.execute(
        "UPDATE game_sessions SET notes = ? WHERE session_id = ?",
        (cleaned, session_id),
    )
    db.conn.commit()


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _heartbeat_fresh(value: Any, *, now: datetime | None = None) -> bool:
    if not value:
        return False
    try:
        stamped = datetime.fromisoformat(str(value))
    except ValueError:
        return False
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return (current - stamped).total_seconds() <= HEARTBEAT_STALE_S


def _lock_path(db_path: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id) or "session"
    root = Path(db_path).resolve()
    directory = Path(str(root) + ".session-locks")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe}.lock"


def _lock_held_by_owner(db_path: Path, session_id: str) -> bool | None:
    """True when another process holds the advisory lock. None if we cannot tell."""
    if fcntl is None:
        return None
    path = _lock_path(db_path, session_id)
    if not path.exists():
        return None
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return None
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def owner_is_live(db: Any, row: Any) -> bool:
    """True when this row's owner process is still in the game.

    A free advisory lock means the recorded pid is not the process that armed
    the session, so a reused pid does not keep a dead game looking live.
    A held lock wins over a stale heartbeat: the owner can be alive while its
    heartbeat writes are stalled on a database lock.
    """
    if row is None:
        return False
    pid_raw = _cell(row, "owner_pid")
    try:
        pid = int(pid_raw) if pid_raw is not None else None
    except (TypeError, ValueError):
        pid = None
    if not _pid_alive(pid):
        return False
    db_path = getattr(db, "path", None)
    session_id = str(_cell(row, "session_id") or "")
    held: bool | None = None
    if db_path is not None and session_id:
        held = _lock_held_by_owner(Path(db_path), session_id)
    if held is False:
        return False
    if held is True:
        return True
    return _heartbeat_fresh(_cell(row, "heartbeat_ts"))


def _recovered_row(row: Any) -> bool:
    try:
        if int(_cell(row, "recovered") or 0) == 1:
            return True
    except (TypeError, ValueError):
        pass
    return "recovered" in str(_cell(row, "notes") or "").split()


def _owner_may_supersede(row: Any) -> bool:
    pid_raw = _cell(row, "owner_pid")
    if pid_raw is None:
        return True
    try:
        return int(pid_raw) == os.getpid()
    except (TypeError, ValueError):
        return False


def _should_write_close(row: Any, db: Any, *, recovered: bool) -> bool:
    """Whether this caller may write ended_ts and the result columns."""
    if row is None:
        return False
    if session_is_open(row):
        if recovered and owner_is_live(db, row):
            return False
        return True
    if recovered or not _recovered_row(row):
        return False
    return _owner_may_supersede(row)


def _apply_close(
    db: Any,
    session_id: str,
    *,
    recovered: bool,
    result_wl: str | None,
    score: str | None,
    play_count: int | None,
) -> None:
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
    else:
        clear_recovered_note(db, session_id)


def close_open_session(
    db: Any,
    session_id: str,
    *,
    recovered: bool = False,
    result_wl: str | None = None,
    score: str | None = None,
    play_count: int | None = None,
) -> bool:
    """Write the close columns for an abandoned row, or supersede a recovered one.

    Returns True only when this call wrote the row. An explicit result and
    score win over the log. Otherwise the last logged score fills both.
    ``recovered`` is set only for the next-start path. A database error
    propagates so the caller can leave the row open.
    """
    if db is None or not session_id:
        return False
    row = _session_row(db, session_id)
    if not _should_write_close(row, db, recovered=recovered):
        if row is not None and not session_is_open(row) and not _recovered_row(row):
            disarm_session_exit(session_id)
        return False
    _apply_close(
        db,
        session_id,
        recovered=recovered,
        result_wl=result_wl,
        score=score,
        play_count=play_count,
    )
    if not recovered:
        disarm_session_exit(session_id)
    return True


def recover_unclosed_sessions(db: Any) -> list[str]:
    """Close open sessions whose owner is dead or stale, and flag each recovered.

    A live owner (alive pid and a fresh heartbeat) is left alone. A read or
    lock error skips that row so it stays open for a later attempt.
    """
    if db is None:
        return []
    try:
        rows = db.conn.execute(
            "SELECT * FROM game_sessions "
            "WHERE ended_ts IS NULL OR TRIM(COALESCE(ended_ts, '')) = ''"
        ).fetchall()
    except sqlite3.Error:
        return []
    closed: list[str] = []
    for row in rows:
        session_id = str(row["session_id"])
        try:
            if owner_is_live(db, row):
                continue
            if close_open_session(db, session_id, recovered=True):
                closed.append(session_id)
        except sqlite3.Error:
            continue
    return closed


def _ensure_owner_columns(db: Any) -> None:
    cols = _columns(db, "game_sessions")
    if "owner_pid" not in cols:
        db.conn.execute("ALTER TABLE game_sessions ADD COLUMN owner_pid INTEGER")
    if "heartbeat_ts" not in cols:
        db.conn.execute("ALTER TABLE game_sessions ADD COLUMN heartbeat_ts TEXT")
    db.conn.commit()


def _write_heartbeat(db_path: str, session_id: str) -> None:
    conn = sqlite3.connect(db_path, timeout=EXIT_BUSY_TIMEOUT_S)
    try:
        conn.execute(f"PRAGMA busy_timeout={EXIT_BUSY_TIMEOUT_MS}")
        conn.execute(
            "UPDATE game_sessions SET owner_pid = ?, heartbeat_ts = ? "
            "WHERE session_id = ? AND (ended_ts IS NULL OR TRIM(COALESCE(ended_ts, '')) = '')",
            (os.getpid(), _now_iso(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


def _heartbeat_loop(owner: _LiveOwner) -> None:
    while not owner.stop.wait(HEARTBEAT_INTERVAL_S):
        try:
            _write_heartbeat(owner.db_path, owner.session_id)
        except sqlite3.Error:
            continue


def _stop_owner(session_id: str) -> None:
    with _GUARD:
        owner = _OWNERS.pop(session_id, None)
    if owner is None:
        return
    owner.stop.set()
    thread = owner.thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2)
    handle = owner.lock_file
    if handle is not None:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass


def _claim_owner(db: Any, session_id: str) -> None:
    """Record this process as the live owner and hold the advisory lock."""
    _ensure_owner_columns(db)
    db_path = str(Path(db.path).resolve())
    _stop_owner(session_id)
    lock_handle = None
    if fcntl is not None:
        path = _lock_path(Path(db_path), session_id)
        lock_handle = open(path, "a+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    _write_heartbeat(db_path, session_id)
    owner = _LiveOwner(
        session_id=session_id,
        db_path=db_path,
        lock_file=lock_handle,
        stop=threading.Event(),
    )
    thread = threading.Thread(
        target=_heartbeat_loop, args=(owner,), name=f"session-heartbeat-{session_id}", daemon=True,
    )
    owner.thread = thread
    with _GUARD:
        _OWNERS[session_id] = owner
    thread.start()


def abandon_owner(db: Any, session_id: str) -> None:
    """Drop liveness without closing the row. Simulates an owner that crashed."""
    _stop_owner(session_id)
    _ARMED.pop(session_id, None)
    if db is None:
        return
    _ensure_owner_columns(db)
    db.conn.execute(
        "UPDATE game_sessions SET owner_pid = ?, heartbeat_ts = ? WHERE session_id = ?",
        (_DEAD_PID, "1970-01-01T00:00:00+00:00", session_id),
    )
    db.conn.commit()


def disarm_session_exit(session_id: str) -> None:
    _ARMED.pop(session_id, None)
    _stop_owner(session_id)


class _ConnDB:
    """Fresh connection used by the exit handler. It does not run migrations."""

    def __init__(self, conn: sqlite3.Connection, path: Path) -> None:
        self.conn = conn
        self.path = path

    def end_game_session(
        self,
        session_id: str,
        *,
        play_count: int | None = None,
        result_wl: str | None = None,
        score: str | None = None,
        recovered: bool | None = None,
    ) -> None:
        sets = ["ended_ts = ?"]
        params: list[Any] = [_now_iso()]
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


def _close_on_fresh_connection(
    path: Path,
    session_id: str,
    *,
    recovered: bool,
    result_wl: str | None = None,
    score: str | None = None,
    play_count: int | None = None,
) -> bool:
    """Close using a new connection with a short busy timeout.

    Raises ``sqlite3.Error`` when the database is locked or unreadable.
    That leaves the row unchanged so a later start can recover it.
    """
    conn = sqlite3.connect(str(path), timeout=EXIT_BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"PRAGMA busy_timeout={EXIT_BUSY_TIMEOUT_MS}")
        db = _ConnDB(conn, path)
        row = _session_row(db, session_id)
        if not _should_write_close(row, db, recovered=recovered):
            return False
        _apply_close(
            db,
            session_id,
            recovered=recovered,
            result_wl=result_wl,
            score=score,
            play_count=play_count,
        )
        return True
    finally:
        conn.close()


def close_armed_sessions(session_id: str | None = None) -> None:
    """Exit handler. Closes armed sessions that are still open.

    Uses a fresh connection and a 250 ms busy timeout. A lock or other
    database error leaves the session armed and the row open. Pass
    ``session_id`` to close one game. Omit it to close every armed game,
    which is what runs at process exit and on SIGTERM.
    """
    if session_id is None:
        items = list(_ARMED.items())
    else:
        db = _ARMED.get(session_id)
        items = [] if db is None else [(session_id, db)]
    for sid, db in items:
        path = getattr(db, "path", None)
        if path is None:
            continue
        try:
            _close_on_fresh_connection(Path(path), sid, recovered=False)
        except sqlite3.Error:
            continue
        disarm_session_exit(sid)


def _on_sigterm(signum: int, frame: Any) -> None:
    try:
        close_armed_sessions()
    except Exception:
        pass
    previous = _PREVIOUS_SIGTERM
    if previous in (signal.SIG_DFL, signal.SIG_IGN, None) or not callable(previous):
        raise SystemExit(128 + int(signum))
    if previous is _on_sigterm:
        raise SystemExit(128 + int(signum))
    previous(signum, frame)


def _install_exit_handlers() -> None:
    global _ATEXIT_INSTALLED, _SIGTERM_INSTALLED, _PREVIOUS_SIGTERM
    if not _ATEXIT_INSTALLED:
        atexit.register(close_armed_sessions)
        _ATEXIT_INSTALLED = True
    if not _SIGTERM_INSTALLED:
        try:
            previous = signal.signal(signal.SIGTERM, _on_sigterm)
        except ValueError:
            return
        _PREVIOUS_SIGTERM = previous
        _SIGTERM_INSTALLED = True


def arm_session_exit(db: Any, session_id: str) -> None:
    """Remember this open session, heartbeat it, and close it if the process exits."""
    if db is None or not session_id:
        return
    _claim_owner(db, session_id)
    _ARMED[session_id] = db
    _install_exit_handlers()


_SCORE_PAIR = re.compile(r"^\s*(\d{1,3})\s*[-–:]\s*(\d{1,3})\s*$")


def parse_score_pair(text: Any) -> tuple[int, int] | None:
    """``us-them`` from text like ``31-14``. Anything else is None."""
    match = _SCORE_PAIR.match(str(text or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def score_reconcile(
    session_score: Any, snaplog_score: tuple[int, int] | None,
) -> dict[str, Any]:
    """Gap between the session's final score and what the snap log reached.

    ``snaplog_score`` is the score after the last logged result. Nothing is
    invented: a missing side gives ``status='unknown'`` and null gaps.
    """
    final = parse_score_pair(session_score)
    out: dict[str, Any] = {
        "session_score": None if final is None else f"{final[0]}-{final[1]}",
        "snaplog_score": None if snaplog_score is None else f"{snaplog_score[0]}-{snaplog_score[1]}",
        "gap_for": None,
        "gap_against": None,
        "status": "unknown",
    }
    if final is None or snaplog_score is None:
        return out
    out["gap_for"] = final[0] - int(snaplog_score[0])
    out["gap_against"] = final[1] - int(snaplog_score[1])
    out["status"] = "match" if out["gap_for"] == 0 and out["gap_against"] == 0 else "gap"
    return out
