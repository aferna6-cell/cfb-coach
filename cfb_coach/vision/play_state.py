"""Play-state machine: BETWEEN_PLAYS|PRE_SNAP|PLAY_ACTIVE|PLAY_ENDED|MENU|OTHER."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID = (
    "BETWEEN_PLAYS",
    "PRE_SNAP",
    "PLAY_ACTIVE",
    "PLAY_ENDED",
    "MENU",
    "OTHER",
)

# Allowed transitions (from → frozenset of to)
_TRANSITIONS: dict[str, frozenset[str]] = {
    "OTHER": frozenset(VALID),
    "MENU": frozenset({"MENU", "BETWEEN_PLAYS", "OTHER", "PRE_SNAP"}),
    "BETWEEN_PLAYS": frozenset({"BETWEEN_PLAYS", "PRE_SNAP", "MENU", "OTHER"}),
    "PRE_SNAP": frozenset({"PRE_SNAP", "PLAY_ACTIVE", "BETWEEN_PLAYS", "MENU", "OTHER"}),
    "PLAY_ACTIVE": frozenset({"PLAY_ACTIVE", "PLAY_ENDED", "OTHER", "MENU"}),
    "PLAY_ENDED": frozenset({"PLAY_ENDED", "BETWEEN_PLAYS", "PRE_SNAP", "MENU", "OTHER"}),
}


@dataclass
class PlayStateTracker:
    """Track play_state with optional motion/HUD hints and timestamps."""

    state: str = "OTHER"
    changed_at: float | None = None
    history: list[tuple[float, str]] = field(default_factory=list)

    def update(
        self,
        *,
        now: float,
        motion: float | None = None,
        hud_visible: bool | None = None,
        menu_like: bool | None = None,
        proposed: str | None = None,
    ) -> str:
        """Advance state from signals. motion ~0..1 frame diff energy."""
        target = proposed
        if target is None:
            if menu_like:
                target = "MENU"
            elif motion is not None and motion > 0.45:
                if self.state in ("PRE_SNAP", "BETWEEN_PLAYS"):
                    target = "PLAY_ACTIVE"
                elif self.state == "PLAY_ACTIVE":
                    target = "PLAY_ACTIVE"
                else:
                    target = "PLAY_ACTIVE"
            elif motion is not None and motion < 0.08:
                if self.state == "PLAY_ACTIVE":
                    target = "PLAY_ENDED"
                elif self.state == "PLAY_ENDED":
                    target = "BETWEEN_PLAYS"
                elif hud_visible and self.state in ("BETWEEN_PLAYS", "OTHER", "PLAY_ENDED"):
                    target = "PRE_SNAP"
                elif hud_visible:
                    target = self.state if self.state != "OTHER" else "PRE_SNAP"
                else:
                    target = self.state if self.state != "OTHER" else "BETWEEN_PLAYS"
            elif hud_visible:
                target = "PRE_SNAP" if self.state in ("OTHER", "BETWEEN_PLAYS", "PLAY_ENDED") else self.state
            else:
                target = self.state

        target = (target or "OTHER").upper()
        if target not in VALID:
            target = "OTHER"

        allowed = _TRANSITIONS.get(self.state, frozenset(VALID))
        if target not in allowed:
            # Soft allow via OTHER hop
            if "OTHER" in allowed and target in _TRANSITIONS.get("OTHER", frozenset()):
                self._set(now, "OTHER")
            else:
                return self.state

        if target != self.state:
            self._set(now, target)
        return self.state

    def _set(self, now: float, state: str) -> None:
        self.state = state
        self.changed_at = now
        self.history.append((now, state))
        if len(self.history) > 64:
            self.history = self.history[-64:]

    def force(self, state: str, now: float) -> str:
        state = (state or "OTHER").upper()
        if state not in VALID:
            state = "OTHER"
        self._set(now, state)
        return self.state

    def snapshot(self) -> dict[str, Any]:
        return {
            "play_state": self.state,
            "changed_at": self.changed_at,
            "history": list(self.history[-8:]),
        }
