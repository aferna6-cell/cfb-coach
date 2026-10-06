"""Madden 27 prep: 8 offense + 8 defense Custom Adjustments per user opponent.

Defense comes from the research DB (meta rank, live scout, tendencies, learned weights).
Offense comes from Aidan's confirmed Custom Adjustments, ranked for this opponent's
coverage looks, and kept only when a play in the trimmed book matches.
CPU games stay offense-only (no macros).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from cfb_coach.macros import FAILED, PROVEN
from cfb_coach.madden.macro_pool import families, pool_ids, pool_macro
from cfb_coach.madden.macros import LEARNED_SUPPRESS, LOADOUT_N, macro_status

W_META_TOP = 0.50  # research DB meta_rank 1 … decays by W_META_STEP per rank
W_META_STEP = 0.03
W_HINT_FAMILY = 0.10  # per live-scout hit in the macro's family (capped)
W_TENDENCY = 0.08  # per logged tell vs this opponent in the macro's family (capped)
W_PERSONA = 0.15  # persona archetype prior
W_PREVIOUS = 0.05  # picked by the last prep / the migrated Active 8
W_STATUS = {PROVEN: 0.15, FAILED: -0.50}
SUPPRESSED = -1.0

_PERSONA = {
    "pressure_heavy": ["cross", "scram"],
    "split_field_zone": ["vert", "scram"],
    "c2_c3_mixer": ["stack", "run", "red_zone"],
    "two_high_money_downs": ["flood", "cross"],
}


def research_hint_families(scout: Any) -> Counter:
    """Concept families named by this prep's LIVE scout (cache / seed fallback never counts)."""
    out: Counter = Counter()
    if scout is None:
        return out
    d = scout.to_dict() if hasattr(scout, "to_dict") else dict(scout)
    if (d.get("mode") or "") != "live":
        return out
    from cfb_coach.madden.data import get_macro

    for s in d.get("suggestions") or []:
        hint = str(s.get("macro_hint") or "").upper()
        fam = (get_macro(hint) or {}).get("concept_family") or {"HEAT": "pressure", "O-PROT": "pressure",
                                                                  "O-MAN": "cross"}.get(hint)
        if fam:
            out[fam] += 1
    return out


def tendency_families(db: Any, opponent_id: str) -> Counter:
    """Families of the concepts this opponent has run against us."""
    out: Counter = Counter()
    if db is None:
        return out
    from cfb_coach.madden.situation import concept_family

    try:
        for r in db.get_tendencies(opponent_id, "offense_concept"):
            fam = concept_family(r["key"])
            if fam:
                out[fam] += int(r["count"] or 0)
    except Exception:  # noqa: BLE001 — never break prep
        pass
    return out


def learned_weights(db: Any, opponent_id: str) -> dict[str, float]:
    if db is None:
        return {}
    try:
        return {str(r["macro"]).upper(): float(r["weight"] or 0.0) for r in db.get_macro_weights(opponent_id)}
    except Exception:  # noqa: BLE001
        return {}


def rank_defense(
    *,
    db: Any = None,
    archetype: str = "",
    hint_families: Counter | None = None,
    tendencies: Counter | None = None,
    weights: dict[str, float] | None = None,
    previous: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every research-DB defense macro, scored and sorted (best first)."""
    persona = _PERSONA.get((archetype or "").lower(), [])
    rows = []
    for i, mid in enumerate(pool_ids("defense")):
        meta = pool_macro(mid) or {}
        fams = families(mid)
        parts: dict[str, float] = {}
        rank = int(meta.get("meta_rank") or 99)
        parts[f"research meta #{rank}"] = round(max(0.0, W_META_TOP - W_META_STEP * (rank - 1)), 3)
        fam_hits = sum((hint_families or Counter())[f] for f in fams)
        if fam_hits:
            parts["live research family"] = min(3, fam_hits) * W_HINT_FAMILY
        tell = sum((tendencies or Counter())[f] for f in fams)
        if tell:
            parts["opponent tendency"] = min(5, tell) * W_TENDENCY
        if set(fams) & set(persona):
            parts[f"persona {archetype}"] = W_PERSONA
        st = macro_status(mid, db)
        if st in W_STATUS:
            parts[f"status {st}"] = W_STATUS[st]
        w = (weights or {}).get(mid)
        if w is not None:
            parts["learned weight"] = SUPPRESSED if w <= LEARNED_SUPPRESS else max(-0.5, min(0.5, w))
        if mid in (previous or []):
            parts["last prep"] = W_PREVIOUS
        score = round(sum(parts.values()) - 0.001 * i, 4)
        rows.append({"id": mid, "side": "defense", "score": score, "parts": parts,
                     "why": "; ".join(f"{k} {v:+.2f}" for k, v in parts.items())})
    rows.sort(key=lambda r: -r["score"])
    return rows


def snap_concept_families(db: Any, opponent_id: str) -> Counter:
    """Concept families on snaps where we were on defense (their offense), before postgame."""
    out: Counter = Counter()
    if db is None:
        return out
    from cfb_coach.madden.situation import concept_family

    try:
        snaps = db.get_recent_snaps(opponent_id, side="defense", limit=40)
    except Exception:  # noqa: BLE001
        return out
    for snap in snaps:
        fam = concept_family(snap["concept_seen"] if hasattr(snap, "keys") else "")
        if fam:
            out[fam] += 1
    return out


def select_loadout(
    opponent_id: str,
    *,
    db: Any = None,
    archetype: str = "",
    scout: Any = None,
    offense_only: bool = False,
    previous: list[str] | None = None,
    previous_offense: list[str] | None = None,
    offense_book: dict[str, list[str]] | None = None,
    n: int = LOADOUT_N,
) -> dict[str, Any]:
    """{"offense": [8 ids], "defense": [8 ids], "ranked": defense rows, "ranked_offense": rows}.

    CPU (``offense_only``) gets neither. Offense ids are Aidan's Custom Adjustments that
    pair with ``offense_book``.
    """
    out: dict[str, Any] = {"offense": [], "defense": [], "ranked": [], "ranked_offense": []}
    if offense_only:
        return out
    from cfb_coach.madden.offense_macros import select_offense

    tends = tendency_families(db, opponent_id) + snap_concept_families(db, opponent_id)
    weights = learned_weights(db, opponent_id)
    rows = rank_defense(db=db, archetype=archetype, hint_families=research_hint_families(scout),
                        tendencies=tends, weights=weights, previous=previous)
    out["ranked"] = rows
    out["defense"] = [r["id"] for r in rows[:n]]
    offense, ranked_o = select_offense(opponent_id, db=db, archetype=archetype, book=offense_book,
                                       previous=previous_offense, weights=weights, n=n)
    out["offense"] = offense
    out["ranked_offense"] = ranked_o
    return out


__all__ = [
    "rank_defense", "research_hint_families", "select_loadout", "snap_concept_families", "tendency_families",
]
