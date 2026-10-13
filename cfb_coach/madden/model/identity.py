"""Stable game/session and snap identities for Madden ML logging.

``game_id`` is the ``game_sessions.session_id`` (unique per Franchise game).
``snap_id`` is ``{game_id}-{seq:04d}`` and is allocated when a call is sealed,
before the outcome is known. Retries and UI refreshes reuse the same id.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_game_id() -> str:
    """Unique game/session id. Never reuse an opponent id as a game id."""
    return uuid.uuid4().hex[:16]


def snap_id_for(game_id: str, seq: int) -> str:
    if not game_id:
        raise ValueError("game_id is required")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError("snap sequence must be an int >= 1")
    return f"{game_id}-{seq:04d}"


def parse_snap_seq(snap_id: str | None) -> int | None:
    if not snap_id or "-" not in snap_id:
        return None
    tail = snap_id.rsplit("-", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


@dataclass
class LiveDecisionTracker:
    """Per-session allocator that prevents duplicate ML decisions.

    Allocate a snap id when a call is sealed. Reuse it for the outcome.
    A second seal with the same decision key is a no-op for shadow logging.
    """

    game_id: str
    next_seq: int = 1
    pending_snap_id: str | None = None
    pending_decision_key: str | None = None
    pending_ml_decision_row_id: int | None = None
    sealed_keys: set[str] = field(default_factory=set)

    @classmethod
    def from_session(cls, session_id: str, *, next_seq: int = 1) -> "LiveDecisionTracker":
        return cls(game_id=session_id, next_seq=max(1, int(next_seq)))

    def decision_key(
        self,
        *,
        side: str,
        formation: str | None,
        play: str | None,
        situation_raw: str | None,
        seq: int | None = None,
    ) -> str:
        payload = "|".join(
            [
                self.game_id,
                str(seq or self.next_seq),
                (side or "").strip().lower(),
                (formation or "").strip(),
                (play or "").strip(),
                (situation_raw or "").strip(),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def seal_call(
        self,
        *,
        side: str,
        formation: str | None,
        play: str | None,
        situation_raw: str | None,
    ) -> tuple[str, int, str, bool]:
        """Return ``(snap_id, seq, decision_key, is_new)``.

        ``is_new`` is False when this exact pending seal was already recorded
        (HTML retry / double-submit), so shadow must not run again.
        """
        # If we already have a pending unlogged outcome for this exact call, reuse.
        if self.pending_snap_id and self.pending_decision_key:
            key = self.decision_key(
                side=side,
                formation=formation,
                play=play,
                situation_raw=situation_raw,
                seq=parse_snap_seq(self.pending_snap_id),
            )
            if key == self.pending_decision_key:
                return self.pending_snap_id, parse_snap_seq(self.pending_snap_id) or self.next_seq, key, False

        seq = self.next_seq
        snap_id = snap_id_for(self.game_id, seq)
        key = self.decision_key(
            side=side,
            formation=formation,
            play=play,
            situation_raw=situation_raw,
            seq=seq,
        )
        if key in self.sealed_keys and self.pending_snap_id:
            # Duplicate submission after seal — keep pending identity.
            return self.pending_snap_id, parse_snap_seq(self.pending_snap_id) or seq, key, False

        self.pending_snap_id = snap_id
        self.pending_decision_key = key
        self.pending_ml_decision_row_id = None
        self.sealed_keys.add(key)
        self.next_seq = seq + 1
        return snap_id, seq, key, True

    def bind_decision_row(self, row_id: int | None) -> None:
        self.pending_ml_decision_row_id = row_id

    def consume_pending(self) -> tuple[str | None, int | None]:
        """Hand off pending snap id when the outcome is logged."""
        snap_id = self.pending_snap_id
        row_id = self.pending_ml_decision_row_id
        self.pending_snap_id = None
        self.pending_decision_key = None
        self.pending_ml_decision_row_id = None
        return snap_id, row_id

    def peek_pending(self) -> tuple[str | None, int | None]:
        return self.pending_snap_id, self.pending_ml_decision_row_id


def next_seq_from_db(db: Any, game_id: str) -> int:
    """Resume sequence after reconnect using stored ml_decisions / snaps."""
    if db is None or not game_id:
        return 1
    seq = 1
    try:
        rows = db.conn.execute(
            "SELECT snap_id FROM ml_decisions WHERE game_id = ? OR session_id = ?",
            (game_id, game_id),
        ).fetchall()
        for row in rows:
            n = parse_snap_seq(row["snap_id"] if hasattr(row, "keys") else row[0])
            if n is not None:
                seq = max(seq, n + 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = db.conn.execute(
            "SELECT ml_snap_id FROM snaps WHERE session_id = ? AND ml_snap_id IS NOT NULL",
            (game_id,),
        ).fetchall()
        for row in rows:
            n = parse_snap_seq(row["ml_snap_id"] if hasattr(row, "keys") else row[0])
            if n is not None:
                seq = max(seq, n + 1)
    except Exception:  # noqa: BLE001
        pass
    return seq
