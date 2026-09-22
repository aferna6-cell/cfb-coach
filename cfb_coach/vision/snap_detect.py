"""Multi-signal snap detection — false snaps worse than late."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class SnapSignal:
    snapped: bool
    confidence: float  # 0..1
    reasons: list[str] = field(default_factory=list)


@dataclass
class SnapDetector:
    """Detect snap via motion burst + multi-centroid accel + optional play-clock.

    Debounce/cooldown prevents double-count. Prefer miss over false positive.
    """

    cooldown_s: float = 2.5
    motion_burst_thresh: float = 0.42
    accel_thresh: float = 0.08
    min_signals: int = 2  # need ≥2 agreeing signals (or very strong single)
    strong_motion: float = 0.62

    _last_snap_t: float | None = None
    _motion_hist: list[float] = field(default_factory=list)
    _centroid_hist: list[tuple[float, float]] = field(default_factory=list)  # (cx, cy) mean
    _clock_hist: list[str | None] = field(default_factory=list)
    _armed: bool = False  # True after PRE_SNAP dwell

    def arm(self, armed: bool = True) -> None:
        self._armed = armed

    def reset(self) -> None:
        self._motion_hist.clear()
        self._centroid_hist.clear()
        self._clock_hist.clear()
        self._armed = False

    def update(
        self,
        *,
        now: float,
        motion: float | None = None,
        centroids: Sequence[tuple[float, float]] | None = None,
        play_clock: str | None = None,
        play_state: str | None = None,
    ) -> SnapSignal:
        reasons: list[str] = []
        score = 0.0
        signals = 0

        if play_state in ("PRE_SNAP", "BETWEEN_PLAYS"):
            self._armed = True
        elif play_state in ("PLAY_ACTIVE", "PLAY_ENDING", "POST_PLAY"):
            # still allow detection only if armed from pre-snap
            pass

        # Cooldown
        if self._last_snap_t is not None and (now - self._last_snap_t) < self.cooldown_s:
            return SnapSignal(False, 0.0, ["cooldown"])

        if not self._armed and play_state not in ("PRE_SNAP", "BETWEEN_PLAYS", None):
            return SnapSignal(False, 0.0, ["not_armed"])

        # --- motion burst ---
        if motion is not None:
            self._motion_hist.append(float(motion))
            if len(self._motion_hist) > 10:
                self._motion_hist = self._motion_hist[-10:]
            m = float(motion)
            prev = self._motion_hist[-2] if len(self._motion_hist) >= 2 else 0.0
            burst = m - prev
            if m >= self.strong_motion or (m >= self.motion_burst_thresh and burst > 0.2):
                signals += 1
                score += 0.45 if m >= self.strong_motion else 0.3
                reasons.append("motion_burst")

        # --- multi-centroid acceleration ---
        if centroids:
            xs = [c[0] for c in centroids]
            ys = [c[1] for c in centroids]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            self._centroid_hist.append((cx, cy))
            if len(self._centroid_hist) > 10:
                self._centroid_hist = self._centroid_hist[-10:]
            if len(self._centroid_hist) >= 2:
                px, py = self._centroid_hist[-2]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                # normalize-ish: assume coords 0..1 or pixels — use relative jump
                if dist > self.accel_thresh:
                    signals += 1
                    score += min(0.35, dist)
                    reasons.append("centroid_accel")

        # --- play-clock change (optional HUD) ---
        if play_clock is not None:
            self._clock_hist.append(play_clock)
            if len(self._clock_hist) > 6:
                self._clock_hist = self._clock_hist[-6:]
            if len(self._clock_hist) >= 2:
                a, b = self._clock_hist[-2], self._clock_hist[-1]
                if a and b and a != b:
                    # clock jumped / reset often at snap
                    signals += 1
                    score += 0.25
                    reasons.append("play_clock_change")

        # Decision: ≥ min_signals OR one very strong motion
        strong_only = (
            motion is not None
            and float(motion) >= self.strong_motion
            and (play_state in ("PRE_SNAP", "BETWEEN_PLAYS", None))
        )
        snapped = False
        if signals >= self.min_signals or strong_only:
            # Prefer requiring arming from pre-snap when state known
            if play_state in ("PRE_SNAP", "BETWEEN_PLAYS", None) or self._armed:
                snapped = True

        conf = max(0.0, min(1.0, score if snapped else score * 0.4))
        if snapped:
            self._last_snap_t = now
            self._armed = False
        return SnapSignal(snapped, conf, reasons)

    def snapshot(self) -> dict[str, Any]:
        return {
            "last_snap_t": self._last_snap_t,
            "armed": self._armed,
            "motion_hist": list(self._motion_hist[-5:]),
        }
