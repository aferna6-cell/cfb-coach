"""Madden 27 prep: pick and rank 10 offense + 10 defense macros per opponent (v1.17).

Each side is ranked over the whole Madden macro pool (Madden research macros + Aidan's CFB
macros, same editor) from:

  * fresh meta research — this prep's live scout hits (``macro_hint`` + its family),
  * the opponent's tendencies — persona archetype + concepts / coverages logged vs them,
  * per-opponent learned weights (``macro_weights`` in madden27.db) + postgame status,
  * the Madden baseline loadout (the old Active 8 default) and last prep's pick (stability),
  * whether Aidan's settings are on file (it is already built in his editor).

Every score part is listed in the card's ``why`` so the ranking is inspectable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from cfb_coach.macros import FAILED, PROVEN
from cfb_coach.madden.macro_pool import families, macro_side, pool_ids, pool_macro
from cfb_coach.madden.macros import LEARNED_SUPPRESS, PER_SIDE, aidan_settings, macro_status

W_BASELINE = 0.30  # in the Madden research baseline loadout (profile default)
W_MADDEN = 0.10  # a Madden-researched macro
W_SETTINGS = 0.20  # Aidan's settings on file (shared with CFB)
W_HINT = 0.25  # named by this prep's live research (per hit, capped)
W_HINT_FAMILY = 0.10  # same family as a research hit (per hit, capped)
W_TENDENCY = 0.08  # per logged tell vs this opponent in the macro's family (capped)
W_PERSONA = 0.15  # persona archetype prior
W_PREVIOUS = 0.05  # picked by the last prep (keeps a migrated Active 8)
W_STATUS = {PROVEN: 0.15, FAILED: -0.50}
W_LAB = 0.40  # lab profile: experimental / ADD delta macros
SUPPRESSED = -1.0

_PERSONA = {
    "pressure_heavy": {"offense": ["pressure"], "defense": ["cross", "scram"]},
    "split_field_zone": {"offense": ["two_high", "two_high_run"], "defense": ["vert", "scram"]},
    "c2_c3_mixer": {"offense": ["cover2", "single_high"], "defense": ["stack", "run"]},
    "two_high_money_downs": {"offense": ["two_high", "two_high_run"], "defense": ["flood", "cross"]},
}
_COV_FAMILY = {"pressure": "pressure", "man": "man", "cover2": "cover2",
               "single_high": "single_high", "two_high": "two_high"}


def research_hints(scout: Any) -> list[str]:
    """Macro ids named by this prep's LIVE scout (cache / seed fallback never counts)."""
    if scout is None:
        return []
    d = scout.to_dict() if hasattr(scout, "to_dict") else dict(scout)
    if (d.get("mode") or "") != "live":
        return []
    return [str(s.get("macro_hint")).upper() for s in d.get("suggestions") or [] if s.get("macro_hint")]


def tendency_families(db: Any, opponent_id: str) -> dict[str, Counter]:
    """Families of what this opponent has shown (their concepts → D; their coverages → O)."""
    out = {"offense": Counter(), "defense": Counter()}
    if db is None:
        return out
    from cfb_coach.madden.playcaller import coverage_class
    from cfb_coach.madden.situation import concept_family

    try:
        for r in db.get_tendencies(opponent_id, "offense_concept"):
            fam = concept_family(r["key"])
            if fam:
                out["defense"][fam] += int(r["count"] or 0)
        for r in db.get_tendencies(opponent_id, "their_coverage"):
            fam = _COV_FAMILY.get(coverage_class(r["key"]) or "")
            if fam:
                out["offense"][fam] += int(r["count"] or 0)
                if fam == "two_high":
                    out["offense"]["two_high_run"] += int(r["count"] or 0)
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


def rank_side(
    side: str,
    *,
    db: Any = None,
    opponent_id: str = "",
    archetype: str = "",
    baseline: list[str] | None = None,
    hints: list[str] | None = None,
    tendencies: Counter | None = None,
    weights: dict[str, float] | None = None,
    previous: list[str] | None = None,
    lab_boost: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every pool macro on ``side``, scored and sorted (best first)."""
    hints = [h for h in hints or [] if pool_macro(h) and macro_side(h) == side]
    hint_fams = Counter(f for h in hints for f in families(h))
    persona = _PERSONA.get((archetype or "").lower(), {}).get(side, [])
    rows = []
    for i, mid in enumerate(pool_ids(side)):
        if mid in (exclude or []):
            continue
        meta = pool_macro(mid) or {}
        fams = families(mid)
        parts: dict[str, float] = {}
        if mid in (baseline or []):
            parts["baseline"] = W_BASELINE
        if meta.get("origin") == "madden":
            parts["madden research macro"] = W_MADDEN
        if aidan_settings(mid):
            parts["your settings on file"] = W_SETTINGS
        n_hit = hints.count(mid)
        if n_hit:
            parts["live research names it"] = min(2, n_hit) * W_HINT
        fam_hits = sum(hint_fams[f] for f in fams) - n_hit
        if fam_hits > 0:
            parts["research family"] = min(3, fam_hits) * W_HINT_FAMILY
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
        if mid in (lab_boost or []):
            parts["lab experiment"] = W_LAB
        score = round(sum(parts.values()) - 0.001 * i, 4)  # pool order breaks ties
        rows.append({"id": mid, "side": side, "score": score, "parts": parts,
                     "why": "; ".join(f"{k} {v:+.2f}" for k, v in parts.items()) or "pool order"})
    rows.sort(key=lambda r: -r["score"])
    return rows


def select_loadout(
    opponent_id: str,
    *,
    db: Any = None,
    archetype: str = "",
    baseline: list[str] | None = None,
    scout: Any = None,
    offense_only: bool = False,
    previous: dict[str, list[str]] | None = None,
    lab_boost: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """{"offense": [10 ids], "defense": [10 ids] (empty for CPU), "ranked": {side: rows}}."""
    hints = research_hints(scout)
    tend = tendency_families(db, opponent_id)
    weights = learned_weights(db, opponent_id)
    out: dict[str, Any] = {"offense": [], "defense": [], "ranked": {}}
    for side in ("offense",) if offense_only else ("offense", "defense"):
        rows = rank_side(side, db=db, opponent_id=opponent_id, archetype=archetype, baseline=baseline,
                         hints=hints, tendencies=tend[side], weights=weights,
                         previous=(previous or {}).get(side), lab_boost=lab_boost, exclude=exclude)
        out["ranked"][side] = rows
        out[side] = [r["id"] for r in rows[:PER_SIDE]]
    return out


__all__ = ["rank_side", "research_hints", "select_loadout", "tendency_families"]
