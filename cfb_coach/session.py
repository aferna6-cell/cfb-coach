"""Game session — start from watch --opponent/--dynasty; id on all plays."""

from __future__ import annotations

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
    return sess
