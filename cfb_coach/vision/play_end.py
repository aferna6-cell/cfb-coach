"""Play-end detection — motion drop + HUD update + debounce."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EndSignal:
    ended: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class PlayEndDetector:
    """Detect end of play. Debounced; one end per snap (caller enforces)."""

    cooldown_s: float = 1.2
    motion_drop_thresh: float = 0.10
    settle_frames: int = 2

    _last_end_t: float | None = None
    _motion_hist: list[float] = field(default_factory=list)
    _active: bool = False
    _low_streak: int = 0

    def on_snap(self) -> None:
        self._active = True
        self._low_streak = 0
        self._motion_hist.clear()

    def reset(self) -> None:
        self._active = False
        self._low_streak = 0
        self._motion_hist.clear()

    def update(
        self,
        *,
        now: float,
        motion: float | None = None,
        hud_changed: bool = False,
        play_state: str | None = None,
    ) -> EndSignal:
        reasons: list[str] = []
        if play_state == "PLAY_ACTIVE":
            self._active = True
        if not self._active and play_state not in ("PLAY_ACTIVE", None):
            return EndSignal(False, 0.0, ["not_active"])

        if self._last_end_t is not None and (now - self._last_end_t) < self.cooldown_s:
            return EndSignal(False, 0.0, ["cooldown"])

        score = 0.0
        if motion is not None:
            self._motion_hist.append(float(motion))
            if len(self._motion_hist) > 12:
                self._motion_hist = self._motion_hist[-12:]
            m = float(motion)
            if m < self.motion_drop_thresh:
                self._low_streak += 1
            else:
                self._low_streak = 0
            if self._low_streak >= self.settle_frames:
                score += 0.5
                reasons.append("motion_drop")
            # peak-then-drop pattern
            if len(self._motion_hist) >= 4:
                peak = max(self._motion_hist[:-2])
                if peak > 0.35 and m < self.motion_drop_thresh:
                    score += 0.2
                    if "motion_drop" not in reasons:
                        reasons.append("post_peak_settle")

        if hud_changed:
            score += 0.3
            reasons.append("hud_update")

        ended = score >= 0.5 and (
            "motion_drop" in reasons or ("hud_update" in reasons and self._low_streak >= 1)
        )
        conf = max(0.0, min(1.0, score if ended else score * 0.3))
        if ended:
            self._last_end_t = now
            self._active = False
            self._low_streak = 0
        return EndSignal(ended, conf, reasons)

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "low_streak": self._low_streak,
            "last_end_t": self._last_end_t,
        }
