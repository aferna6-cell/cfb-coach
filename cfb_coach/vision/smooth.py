"""Temporal majority / threshold smoother for vision fields."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque


@dataclass
class TemporalSmoother:
    """Require N of last M frames to agree before changing a major field.

    Pure Python — no numpy. Used so flicker does not thrash tips.
    """

    window: int = 5  # M
    threshold: int = 3  # N of M
    fields: tuple[str, ...] = ("formation", "shell", "pressure", "play_state")
    _hist: dict[str, Deque[str]] = field(default_factory=dict)
    _current: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold < 1:
            self.threshold = 1
        if self.window < 1:
            self.window = 1
        if self.threshold > self.window:
            self.threshold = self.window
        for f in self.fields:
            self._hist[f] = deque(maxlen=self.window)

    def update(self, values: dict[str, Any]) -> dict[str, str]:
        """Push new field values; return smoothed dict for tracked fields."""
        out: dict[str, str] = {}
        for f in self.fields:
            raw = values.get(f)
            if raw is None:
                continue
            s = str(raw)
            hist = self._hist[f]
            hist.append(s)
            # Majority among history
            counts = Counter(hist)
            winner, count = counts.most_common(1)[0]
            prev = self._current.get(f)
            if prev is None:
                # bootstrap: adopt winner if enough votes, else adopt latest
                if count >= self.threshold or len(hist) < self.threshold:
                    self._current[f] = winner if count >= min(self.threshold, len(hist)) else s
                    # On short history, prefer latest until window fills enough
                    if len(hist) < self.threshold:
                        self._current[f] = s
                else:
                    self._current[f] = s
            elif winner != prev and count >= self.threshold:
                self._current[f] = winner
            # else keep prev
            out[f] = self._current[f]
        return out

    def reset(self) -> None:
        for f in self.fields:
            self._hist[f].clear()
        self._current.clear()


def majority_vote(values: list[str], *, threshold: int | None = None) -> str | None:
    """Return majority value if count >= threshold (default ceil(n/2)+)."""
    if not values:
        return None
    counts = Counter(values)
    winner, count = counts.most_common(1)[0]
    need = threshold if threshold is not None else (len(values) // 2 + 1)
    if count >= need:
        return winner
    return None
