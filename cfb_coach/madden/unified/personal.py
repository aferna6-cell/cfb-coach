"""Personal execution modelling, kept SEPARATE from play quality.

``PersonalExecution`` learns, per (side, concept), how the owner's actual results differ from the
shared matchup prediction (yards residual, turnover and sack excess). Everything is shrunk hard
(``K_PERSONAL``) and capped, so a few bad plays cannot dominate the strategic model. It only learns
from VERIFIED executions. With no data it is the identity.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

K_PERSONAL = 40.0
YARD_CAP = 0.8
TO_CAP = 0.02


class PersonalExecution:
    def __init__(self) -> None:
        self.stats: dict[tuple[str, str], list[float]] = {}  # (side, concept) -> [n, sum_resid_yards, sum_to_excess, sum_sack_excess]

    def record(self, side: str, concept: str, *, yards_resid: float, turnover_excess: float = 0.0, sack_excess: float = 0.0) -> None:
        r = self.stats.setdefault((side, concept), [0.0, 0.0, 0.0, 0.0])
        r[0] += 1
        r[1] += yards_resid
        r[2] += turnover_excess
        r[3] += sack_excess

    def _w(self, n: float) -> float:
        return n / (n + K_PERSONAL)

    def adjust(self, profile: Any, ey: float, side: str | None) -> tuple[float, float]:
        r = self.stats.get((side or "", getattr(profile, "concept", "")))
        if not r or r[0] <= 0:
            return ey, 0.0
        shift = max(-YARD_CAP, min(YARD_CAP, self._w(r[0]) * r[1] / r[0]))
        return ey + shift, (3.0 / math.sqrt(r[0] + 5.0)) ** 2 * 0.1

    def turnover_delta(self, profile: Any) -> float:
        best = 0.0
        for (side, concept), r in self.stats.items():
            if concept == getattr(profile, "concept", None) and r[0] > 0:
                best = max(best, min(TO_CAP, self._w(r[0]) * r[2] / r[0]))
        return best

    def n_for(self, side: str, concept: str) -> int:
        r = self.stats.get((side, concept))
        return int(r[0]) if r else 0

    def to_dict(self) -> dict[str, Any]:
        return {"version": "unified_personal.v1", "stats": [[list(k), v] for k, v in self.stats.items()]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PersonalExecution":
        p = cls()
        for k, v in d.get("stats", []):
            p.stats[tuple(k)] = list(v)
        return p
