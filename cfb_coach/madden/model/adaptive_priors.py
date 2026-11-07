"""Three-level learning over the football priors: general football -> Madden 27 (all games) -> this opponent.

Cells are (coverage group, offensive concept). Each cell's lift is a normal posterior in logit units:

    prior      theta ~ N(mu0, s0^2)          mu0 = the structured prior (football_priors), s0 shrinks with the prior's confidence
    Madden     all closed snaps, every opponent, against revealed coverage (post-play labels, used as OUTCOME context only)
    opponent   same, this opponent only, with the Madden posterior as ITS prior (partial pooling; thin opponent data cannot dominate)

An observation is a closed snap whose play carried the concept: z = (y - p0) / (p0 (1 - p0)) is an unbiased estimate of the lift, with precision
p0 (1 - p0) = 0.25, scaled by trust (route-derived 1.0, name-only 0.35), execution (attested 1.0, unattested 0.5) and attribution (a sack or a
penalty says little about the concept: 0.5 / 0). With s0 of 0.13-0.17 a single failed snap moves a cell by about 0.01: a concept is never
abandoned after one play, and 20 clean failures move it by roughly 0.2.

Leakage rules: only CLOSED earlier snaps count; the current snap's revealed coverage is never an input to its own prediction; revealed coverage is
used here as the OUTCOME context for learning, and predictions use the opponent's coverage MIXTURE built from earlier snaps (time-correct).
p0 is the pooled success rate of the same snaps (shrunk toward 0.5): a documented approximation, not a context-specific baseline.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.model import football_priors as fp

VERSION = "adaptive_priors.v1"
OBS_PRECISION = 0.25            # p0 (1 - p0) at p0 = 0.5
Z_CLIP = 2.0

GROUP_SD = 0.07          # how far evidence pooled over a whole coverage GROUP may move every variant in it
VARIANT_SD = 0.06        # extra, variant-specific correction (Robber vs Cover 1, Man vs Zone, Spot vs Match, Quarters vs Palms)
OPPONENT_SD = 0.10       # this opponent's deviation from the Madden-wide picture


@functools.lru_cache(maxsize=4096)
def _concepts_of(formation: str, play: str) -> tuple[tuple[str, float], ...]:
    from cfb_coach.madden.model import offense_structural_model as sm
    try:
        return tuple(sorted(fp.play_concepts(sm.structure_any(formation, play)).items()))
    except Exception:                                       # noqa: BLE001 - play not in the structure bank: no concept evidence
        return ()


@functools.lru_cache(maxsize=1024)
def _keys_of(label: str, typed: str) -> tuple[tuple[str, float], ...]:
    """Revealed coverage text -> {mix key: weight}. Exact typed text wins; a collapsed stored label is a uniform mixture (see defense_semantics)."""
    from cfb_coach.madden.model import defense_semantics as ds
    return tuple(sorted(ds.stored_label_weights(label or None, typed or None).items()))


def _exec_weight(row: Mapping[str, Any]) -> float:
    return 1.0 if str(row.get("executed_verification") or "").lower() in ("verified", "confirmed") else 0.5


def observations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Closed offense snaps -> per-cell evidence. Rows are in time order; ONLY earlier rows may be passed for a given prediction.

    Each observation stores the RESIDUAL e = z - (the structured prior lift for that exact coverage descriptor), so learning corrects the prior of the
    variant that was actually seen instead of replacing it."""
    from cfb_coach.outcome import outcome_success
    closed = [r for r in rows if str(r.get("side") or "offense") == "offense" and str(r.get("result") or "").strip()]
    ys = [bool(outcome_success(r["result"], "offense")) for r in closed]
    p0 = (sum(ys) + 0.5 * 20) / (len(ys) + 20) if closed else 0.5       # shrunk pooled rate
    out = []
    for r, y in zip(closed, ys):
        res = str(r["result"]).lower()
        if "penalty" in res:
            continue                                                    # a flag says nothing about the concept
        attr = 0.5 if ("sack" in res or "fumble" in res or ("int" in res and "incomplete" not in res)) else 1.0
        keys = _keys_of(str(r.get("coverage_seen") or ""), str(r.get("coverage_typed") or ""))
        if not keys:
            continue                                                    # no revealed coverage: nothing to attribute
        f, p = r.get("executed_formation") or r.get("formation"), r.get("executed_play") or r.get("play")
        if r.get("executed_verification") not in ("verified", "confirmed") and r.get("executed_play") and r.get("executed_play") != r.get("play"):
            f, p = r.get("formation"), r.get("play")                    # unattested "used different": fall back to the shown call
        z = max(-Z_CLIP, min(Z_CLIP, ((1.0 if y else 0.0) - p0) / (p0 * (1 - p0))))
        for concept, trust in _concepts_of(str(f), str(p)):
            for key, kw in keys:
                d = fp.descriptor(key)
                g, v = (fp.learning_group(d), fp.variant_key(d)) if d else (None, None)
                if not g or not v:
                    continue
                prior_lift = fp.lift_for(d, concept)[0]
                out.append({"g": ("g", g, concept), "v": ("v", v, concept), "e": z - prior_lift, "trust": trust * kw * attr * _exec_weight(r),
                            "opponent": r.get("opponent_id")})
    return out


def _fit(obs: Sequence[tuple[float, float]], sd: float, prior_mean: float = 0.0) -> float:
    """Posterior mean of a residual: normal prior N(prior_mean, sd^2), observations (e, trust) with precision 0.25 * trust."""
    prec = 1.0 / (sd * sd)
    num = prior_mean * prec
    for e, t in obs:
        w = OBS_PRECISION * t
        prec += w
        num += w * e
    return num / prec


def _madden_residuals(obs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], float]:
    """Group residuals first, then variant residuals on what the group residual did not explain. Every snap is used ONCE at each level."""
    res: dict[tuple[str, str, str], float] = {}
    by_g: dict[tuple, list] = {}
    for o in obs:
        by_g.setdefault(o["g"], []).append(o)
    for k, items in by_g.items():
        res[k] = _fit([(o["e"], o["trust"]) for o in items], GROUP_SD)
    by_v: dict[tuple, list] = {}
    for o in obs:
        by_v.setdefault(o["v"], []).append(o)
    for k, items in by_v.items():
        res[k] = _fit([(o["e"] - res.get(o["g"], 0.0), o["trust"]) for o in items], VARIANT_SD)
    return res


def posterior_cells(rows: Iterable[Mapping[str, Any]], opponent_id: str | None) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    """Residual corrections for ``football_priors.matchup_edge``: keys ("g", group, concept) and ("v", variant, concept).

    For a known opponent the Madden-wide level is estimated from the OTHER opponents only, and this opponent's snaps enter exactly once, at the
    opponent level (leave-one-opponent-out). So no snap is counted in both the Madden-wide and the opponent posterior."""
    obs = observations(rows)
    others = [o for o in obs if not (opponent_id and o["opponent"] == opponent_id)]
    mine = [o for o in obs if opponent_id and o["opponent"] == opponent_id]
    base = _madden_residuals(others if opponent_id else obs)
    learned = dict(base)
    info: dict[str, Any] = {"version": VERSION, "observations": len(obs), "opponent_observations": len(mine), "madden_observations": len(others if opponent_id else obs), "cells_updated": {}}
    by_cell: dict[tuple, list] = {}
    for o in mine:
        by_cell.setdefault(o["g"], []).append(o)
    for g_key, items in by_cell.items():
        # what the Madden-wide residuals (without this opponent) already explain for each of this opponent's snaps
        rem = [(o["e"] - base.get(o["g"], 0.0) - base.get(o["v"], 0.0), o["trust"]) for o in items]
        d_o = _fit(rem, OPPONENT_SD)
        learned[g_key] = base.get(g_key, 0.0) + d_o                       # the opponent residual rides on the group key
        info["cells_updated"][f"{g_key[1]}|{g_key[2]}"] = {"madden_residual": round(base.get(g_key, 0.0), 4), "opponent_residual": round(d_o, 4),
                                                           "n_opponent": round(sum(t for _, t in rem), 2)}
    return learned, info


def learned_cells(db: Any, opponent_id: str, session_id: str | None = None) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Live hook: posterior from closed snaps already in the database (all earlier by construction). Never raises into the live path."""
    try:
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(snaps)")}
        pick = ", ".join(c if c in cols else f"NULL AS {c}" for c in ("opponent_id", "side", "result", "formation", "play", "coverage_seen", "coverage_typed",
                                                                       "executed_formation", "executed_play", "executed_verification"))
        rows = [dict(r) for r in db.conn.execute(f"SELECT {pick} FROM snaps WHERE side = 'offense' AND result IS NOT NULL AND trim(result) != '' ORDER BY id")]
        return posterior_cells(rows, opponent_id)
    except Exception as exc:                                # noqa: BLE001
        return {}, {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------------------------------- cautious exploration
def exploration_choice(rows: Sequence[Mapping[str, Any]], *, situation: Mapping[str, Any], margin: float = 0.015, beta: float = 0.5) -> dict[str, Any]:
    """Among near-tied top candidates, prefer the one with the most uncertain (least tested) estimate. Deterministic UCB; logged with its reason.

    Never explores in costly situations (4th down, goal to go inside the 5, late-game protect-the-lead) or when one option clearly leads."""
    top = rows[0]
    near = [r for r in rows if top["probability"] - r["probability"] <= margin]
    why = None
    if int(situation.get("down") or 0) == 4:
        why = "4th down: not exploring"
    elif situation.get("yardline") is not None and int(situation["yardline"]) >= 95:
        why = "goal line: not exploring"
    elif situation.get("late_lead"):
        why = "protecting a late lead: not exploring"
    elif len(near) < 2:
        why = "one option clearly leads"
    if why:
        return {"explored": False, "reason": why, "chosen": (top["formation"], top["play"]), "candidates": []}
    scored = [(r["probability"] + beta * float(r.get("route_aware_se", 0.0)) * 0.1, r) for r in near]
    scored.sort(key=lambda t: -t[0])
    pick = scored[0][1]
    return {"explored": pick is not top, "reason": "near-tied candidates; preferred the less-tested estimate" if pick is not top else "top candidate also best on uncertainty-adjusted value",
            "chosen": (pick["formation"], pick["play"]), "policy": "deterministic_ucb",
            "candidates": [{"play": f"{r['formation']} / {r['play']}", "p": round(r["probability"], 4), "se": round(float(r.get("route_aware_se", 0.0)), 4),
                            "ucb": round(v, 4)} for v, r in scored[:5]]}
