"""Play-state machine (Milestone 2).

Primary states:
  UNKNOWN, MENU, BETWEEN_PLAYS, PRE_SNAP, PLAY_ACTIVE, PLAY_ENDING, POST_PLAY

Compat aliases (M1):
  OTHER → UNKNOWN, PLAY_ENDED → PLAY_ENDING (settle → POST_PLAY)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CANONICAL = (
    "UNKNOWN",
    "MENU",
    "BETWEEN_PLAYS",
    "PRE_SNAP",
    "PLAY_ACTIVE",
    "PLAY_ENDING",
    "POST_PLAY",
)

COMPAT_ALIASES = {
    "OTHER": "UNKNOWN",
    "PLAY_ENDED": "PLAY_ENDING",
}

VALID = CANONICAL + ("OTHER", "PLAY_ENDED")

_TRANSITIONS: dict[str, frozenset[str]] = {
    "UNKNOWN": frozenset(CANONICAL),
    "MENU": frozenset({"MENU", "BETWEEN_PLAYS", "UNKNOWN", "PRE_SNAP"}),
    "BETWEEN_PLAYS": frozenset({"BETWEEN_PLAYS", "PRE_SNAP", "MENU", "UNKNOWN"}),
    "PRE_SNAP": frozenset({"PRE_SNAP", "PLAY_ACTIVE", "BETWEEN_PLAYS", "MENU", "UNKNOWN"}),
    "PLAY_ACTIVE": frozenset({"PLAY_ACTIVE", "PLAY_ENDING", "UNKNOWN", "MENU"}),
    "PLAY_ENDING": frozenset(
        {"PLAY_ENDING", "POST_PLAY", "BETWEEN_PLAYS", "PRE_SNAP", "MENU", "UNKNOWN"}
    ),
    "POST_PLAY": frozenset({"POST_PLAY", "BETWEEN_PLAYS", "PRE_SNAP", "MENU", "UNKNOWN"}),
}


def canonicalize(state: str | None) -> str:
    s = (state or "UNKNOWN").upper().strip()
    s = COMPAT_ALIASES.get(s, s)
    if s not in CANONICAL:
        return "UNKNOWN"
    return s


def to_compat(state: str) -> str:
    """Map canonical → M1-facing name (PLAY_ENDED / OTHER)."""
    c = canonicalize(state)
    if c in ("PLAY_ENDING", "POST_PLAY"):
        return "PLAY_ENDED"
    if c == "UNKNOWN":
        return "OTHER"
    return c


@dataclass
class PlayStateTracker:
    """Track play_state with temporal evidence + play-phase timestamps.

    Timestamps: pre_snap_start, snap_time, play_end_time, next_pre_snap_time
    False snaps worse than late — snap needs clearer burst; end can settle.
    """

    state: str = "UNKNOWN"
    changed_at: float | None = None
    history: list[tuple[float, str]] = field(default_factory=list)

    pre_snap_start: float | None = None
    snap_time: float | None = None
    play_end_time: float | None = None
    next_pre_snap_time: float | None = None

    _motion_hist: list[float] = field(default_factory=list)
    _hud_hist: list[bool] = field(default_factory=list)

    def update(
        self,
        *,
        now: float,
        motion: float | None = None,
        hud_visible: bool | None = None,
        menu_like: bool | None = None,
        proposed: str | None = None,
        snap_signal: bool = False,
        end_signal: bool = False,
    ) -> str:
        """Advance state from multi-frame / external snap|end signals."""
        latest_motion: float | None = None
        if motion is not None:
            latest_motion = float(motion)
            self._motion_hist.append(latest_motion)
            if len(self._motion_hist) > 8:
                self._motion_hist = self._motion_hist[-8:]
        if hud_visible is not None:
            self._hud_hist.append(bool(hud_visible))
            if len(self._hud_hist) > 8:
                self._hud_hist = self._hud_hist[-8:]

        recent_m = self._motion_hist[-3:]
        avg_motion = sum(recent_m) / len(recent_m) if recent_m else None
        high_count = sum(1 for m in recent_m if m > 0.45)
        low_count = sum(1 for m in recent_m if m < 0.08)
        # Burst: latest spike OR 2/3 high (temporal, not single weak frame)
        motion_burst = bool(
            (latest_motion is not None and latest_motion > 0.45)
            or high_count >= 2
        )
        # Settle: sustained low (prefer 2 of last 3; allow single clear drop if only 1 sample)
        motion_settle = bool(
            (len(recent_m) >= 2 and low_count >= 2)
            or (len(recent_m) == 1 and recent_m[0] < 0.08)
            or (latest_motion is not None and latest_motion < 0.08 and low_count >= 1 and len(recent_m) <= 2)
        )
        hud_votes = (
            sum(1 for h in self._hud_hist[-3:] if h) >= max(1, min(2, len(self._hud_hist)))
            if self._hud_hist
            else False
        )

        target = proposed
        if target is not None:
            target = canonicalize(target)
        elif menu_like:
            target = "MENU"
        elif snap_signal and self.state in ("PRE_SNAP", "BETWEEN_PLAYS"):
            target = "PLAY_ACTIVE"
        elif end_signal and self.state == "PLAY_ACTIVE":
            target = "PLAY_ENDING"
        elif motion_burst:
            if self.state in ("PRE_SNAP", "BETWEEN_PLAYS"):
                target = "PLAY_ACTIVE"
            elif self.state == "PLAY_ACTIVE":
                target = "PLAY_ACTIVE"
            elif self.state in ("PLAY_ENDING", "POST_PLAY"):
                # late motion during ending — stay ending unless clearly new play
                target = self.state
            else:
                target = "PLAY_ACTIVE"
        elif motion_settle or (avg_motion is not None and avg_motion < 0.08):
            if self.state == "PLAY_ACTIVE":
                target = "PLAY_ENDING"
            elif self.state == "PLAY_ENDING":
                target = "POST_PLAY"
            elif self.state == "POST_PLAY":
                target = "BETWEEN_PLAYS"
            elif hud_votes and self.state in (
                "BETWEEN_PLAYS",
                "UNKNOWN",
                "POST_PLAY",
                "PLAY_ENDING",
            ):
                target = "PRE_SNAP"
            elif hud_votes:
                target = self.state if self.state != "UNKNOWN" else "PRE_SNAP"
            else:
                target = self.state if self.state != "UNKNOWN" else "BETWEEN_PLAYS"
        elif hud_votes:
            if self.state in ("UNKNOWN", "BETWEEN_PLAYS", "POST_PLAY", "PLAY_ENDING"):
                target = "PRE_SNAP"
            else:
                target = self.state
        else:
            target = self.state

        target = canonicalize(target)
        allowed = _TRANSITIONS.get(self.state, frozenset(CANONICAL))
        if target not in allowed:
            if "UNKNOWN" in allowed and target in _TRANSITIONS.get("UNKNOWN", frozenset()):
                self._set(now, "UNKNOWN")
            else:
                return self.state

        if target != self.state:
            self._set(now, target)
        return self.state

    def _set(self, now: float, state: str) -> None:
        prev = self.state
        state = canonicalize(state)
        self.state = state
        self.changed_at = now
        self.history.append((now, state))
        if len(self.history) > 64:
            self.history = self.history[-64:]

        if state == "PRE_SNAP" and prev != "PRE_SNAP":
            if self.snap_time is not None:
                self.next_pre_snap_time = now
            self.pre_snap_start = now
        elif state == "PLAY_ACTIVE" and prev in ("PRE_SNAP", "BETWEEN_PLAYS"):
            self.snap_time = now
        elif state == "PLAY_ENDING" and prev == "PLAY_ACTIVE":
            self.play_end_time = now
        elif state == "POST_PLAY" and self.play_end_time is None:
            self.play_end_time = now

    def force(self, state: str, now: float) -> str:
        self._set(now, canonicalize(state))
        return self.state

    def reset_play_clocks(self) -> None:
        self.pre_snap_start = None
        self.snap_time = None
        self.play_end_time = None
        self.next_pre_snap_time = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "play_state": self.state,
            "play_state_compat": to_compat(self.state),
            "changed_at": self.changed_at,
            "pre_snap_start": self.pre_snap_start,
            "snap_time": self.snap_time,
            "play_end_time": self.play_end_time,
            "next_pre_snap_time": self.next_pre_snap_time,
            "history": list(self.history[-8:]),
        }

    @property
    def compat_state(self) -> str:
        return to_compat(self.state)
