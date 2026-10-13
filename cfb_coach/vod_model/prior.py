"""Tiers, shrinkage, and the VOD-primary / log-nudge score.

VOD weight grows with the tier as retrains add snaps: none and low contribute
nothing, med contributes 0.65, high contributes 1.0. A thin cell stays near
the stratum base rate because the model already stores ``shrunk_success``
(fallback: (successes + 5 * base) / (n + 5), the same prior as learning).
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from cfb_coach.vod_model.adapter import VodCell, VodModel

VOD_QUALITY_WEIGHT_DEFAULT = 1.0
LOG_NUDGE_DEFAULT = 0.25
# Prep call sheet: scale shrunk success so a med/high VOD rate outweighs the
# existing log term (about ±1.1) without touching that term when VOD is silent.
VOD_SHEET_SCALE = 3.0
INFLUENCE = {"none": 0.0, "low": 0.0, "med": 0.65, "high": 1.0}

# Specific families before the shorter tokens they contain.
_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cover_2_man", ("cover2man",)),
    ("cover_0", ("cover0", "coverzero")),
    ("cover_1", ("cover1",)),
    ("tampa_2", ("tampa2",)),
    ("cover_2", ("cover2",)),
    ("cover_3", ("cover3",)),
    ("cover_4", ("cover4", "quarters")),
    ("cover_6", ("cover6",)),
    ("cover_9", ("cover9",)),
    ("man_other", ("man",)),
    ("pressure_other", ("blitz", "pressure")),
)


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def quality_weight() -> float:
    return _float_env("CFB_COACH_VOD_QUALITY_WEIGHT", VOD_QUALITY_WEIGHT_DEFAULT)


def log_nudge() -> float:
    return _float_env("CFB_COACH_VOD_LOG_NUDGE", LOG_NUDGE_DEFAULT)


def norm_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def family_key(hint: str | None) -> str | None:
    token = norm_text(hint)
    if not token:
        return None
    for family, keys in _FAMILY_RULES:
        if any(key in token for key in keys):
            return family
    return None


def cell_tier(cell: VodCell, *, tentative_n: int | None = None) -> str:
    """none / low / med / high. The model's tentative flag is stricter than med."""
    model_tentative = cell.tentative or cell.confidence == "tentative"
    if tentative_n is not None and cell.n < tentative_n:
        model_tentative = True
    if cell.n < 8:
        return "none"
    if model_tentative:
        if cell.n >= 8 and cell.n_vods >= 2:
            return "low"
        return "none"
    if cell.n >= 30 and cell.n_vods >= 5 and cell.lower_bound > cell.baseline + 0.05:
        return "high"
    if cell.n >= 15 and cell.n_vods >= 3 and cell.lower_bound > cell.baseline:
        return "med"
    if cell.n >= 8 and cell.n_vods >= 2:
        return "low"
    return "none"


def influence(tier: str) -> float:
    return INFLUENCE.get(tier, 0.0)


def clamp_norm(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def combined_score(cell: VodCell, own_norm: float, *, tier: str | None = None) -> float:
    """VOD shrunk success, scaled by tier, plus a clamped log nudge."""
    tier = tier or cell_tier(cell)
    return quality_weight() * influence(tier) * cell.shrunk_success + log_nudge() * clamp_norm(own_norm)


def select_cell(
    cells: list[VodCell] | tuple[VodCell, ...],
    own_norm_of: Callable[[VodCell], float],
    *,
    tentative_n: int | None = None,
) -> VodCell | None:
    """Best med/high cell. Logs only reorder a band within one nudge of the leader."""
    actionable = [c for c in cells if cell_tier(c, tentative_n=tentative_n) in ("med", "high")]
    if not actionable:
        return None
    leader = max(actionable, key=lambda c: (c.shrunk_success, c.lower_bound, c.n))
    band = log_nudge()
    pool = [c for c in actionable if leader.shrunk_success - c.shrunk_success <= band + 1e-9]
    return max(
        pool,
        key=lambda c: (
            combined_score(c, own_norm_of(c), tier=cell_tier(c, tentative_n=tentative_n)),
            c.shrunk_success,
            c.n,
        ),
    )


def cells_for_hint(model: VodModel, game: str, opponent_type: str, hint: str | None) -> tuple[VodCell, ...]:
    """Raw look when the hint names one, otherwise the coverage family."""
    if not hint:
        return ()
    token = norm_text(hint)
    raw_names = model.looks(game, opponent_type, "raw")
    for name in raw_names:
        folded = norm_text(name)
        if folded == token or token in folded or folded in token:
            found = model.cells_for(game, opponent_type, "raw", name)
            if found:
                return found
    family = family_key(hint)
    if family:
        found = model.cells_for(game, opponent_type, "family", family)
        if found:
            return found
    return ()


def expected_coverage(db: Any, opponent_id: str) -> str | None:
    """The coverage Aidan has seen most from this opponent (their look, our offense snaps)."""
    if db is None or not opponent_id:
        return None
    try:
        from cfb_coach.scouting import scout_opponent

        report = scout_opponent(db, opponent_id)
    except Exception:  # noqa: BLE001
        return None
    covs = ((report.get("their_defense") or {}).get("overall") or {}).get("coverages") or []
    if not covs:
        return None
    name = covs[0].get("coverage")
    return str(name) if name else None


def sheet_bonus(cell: VodCell, *, tentative_n: int | None = None) -> float:
    tier = cell_tier(cell, tentative_n=tentative_n)
    weight = influence(tier)
    if weight <= 0:
        return 0.0
    return quality_weight() * weight * cell.shrunk_success * VOD_SHEET_SCALE


def beater_lines(
    model: VodModel,
    game: str,
    opponent_type: str,
    *,
    limit_looks: int = 6,
    label_of: Callable[[VodCell], str] | None = None,
) -> list[str]:
    lines: list[str] = []
    names = model.looks(game, opponent_type, "family")[:limit_looks]
    for name in names:
        cells = list(model.cells_for(game, opponent_type, "family", name))
        cells.sort(key=lambda c: (-c.shrunk_success, -c.n))
        bits = []
        for cell in cells[:3]:
            tier = cell_tier(cell, tentative_n=model.tentative_n)
            mark = " tentative" if cell.tentative or tier in ("none", "low") else ""
            label = label_of(cell) if label_of else cell.call
            bits.append(
                f"{label} n={cell.n} shrunk={cell.shrunk_success:.2f} "
                f"lb={cell.lower_bound:.2f} tier={tier}{mark}"
            )
        look_n = model.look_n.get((game, opponent_type, "family", name), 0)
        lines.append(f"  vs {name} (look n={look_n}): " + "; ".join(bits))
    return lines
