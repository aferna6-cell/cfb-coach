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
OPPONENT_PRIOR_SD = 0.10        # how far an opponent may drift from the Madden-level posterior before data says so
Z_CLIP = 2.0

# representative descriptor per coverage group, used only to read the structured prior lift of a (group, concept) cell
GROUP_REP = {"man": {"shell": "1", "type": "man", "variants": set(), "press": "standard"},
             "cover_0": {"shell": "0", "type": "man", "variants": set(), "press": "standard"},
             "cover_2_zone": {"shell": "2", "type": "spot_zone", "variants": set(), "press": "standard"},
             "cover_3_spot": {"shell": "3", "type": "spot_zone", "variants": set(), "press": "standard"},
             "cover_3_match": {"shell": "3", "type": "match", "variants": set(), "press": "standard"},
             "cover_4": {"shell": "4", "type": "match", "variants": set(), "press": "standard"},
             "pressure": {"shell": "unk", "type": "unk", "variants": set(), "press": "unspecified"}}


def prior_cell(group: str, concept: str) -> tuple[float, float]:
    """(mu0, s0) of a cell from the structured priors. Cells with no prior start at 0 with a wide sd."""
    lift, conf_w, ids = fp.lift_for(GROUP_REP[group], concept)
    if not ids:
        return 0.0, 0.20
    conf = min(1.0, conf_w / len(ids))
    return lift, max(0.07, 0.20 - 0.12 * conf)


def _post(mu0: float, s0: float, obs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Normal-normal update. obs = [(z, trust)]. Returns (mean, sd)."""
    prec = 1.0 / (s0 * s0)
    num = mu0 * prec
    for z, t in obs:
        w = OBS_PRECISION * t
        prec += w
        num += w * z
    return num / prec, math.sqrt(1.0 / prec)


@functools.lru_cache(maxsize=4096)
def _concepts_of(formation: str, play: str) -> tuple[tuple[str, float], ...]:
    from cfb_coach.madden.model import offense_structural_model as sm
    try:
        return tuple(sorted(fp.play_concepts(sm.structure_any(formation, play)).items()))
    except Exception:                                       # noqa: BLE001 - play not in the structure bank: no concept evidence
        return ()


@functools.lru_cache(maxsize=1024)
def _groups_of(label: str, typed: str) -> tuple[tuple[str, float], ...]:
    """Revealed coverage text -> {group: weight}. Exact typed text wins; a collapsed stored label is a uniform mixture (see defense_semantics)."""
    from cfb_coach.madden.model import defense_semantics as ds
    w = ds.stored_label_weights(label or None, typed or None)
    out: dict[str, float] = {}
    for key, p in w.items():
        d = fp.descriptor(key)
        g = fp.route_group(d) if d else None
        if g:
            out[g] = out.get(g, 0.0) + p
    if "man" in out and not any(k for k in out if k != "man"):
        pass
    return tuple(sorted(out.items()))


def _exec_weight(row: Mapping[str, Any]) -> float:
    return 1.0 if str(row.get("executed_verification") or "").lower() in ("verified", "confirmed") else 0.5


def observations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Closed offense snaps -> per-cell evidence. Rows are in time order; ONLY earlier rows may be passed for a given prediction."""
    from cfb_coach.outcome import outcome_success
    closed = [r for r in rows if str(r.get("side") or "offense") == "offense" and str(r.get("result") or "").strip()]
    ys = [bool(outcome_success(r["result"], "offense")) for r in closed]
    p0 = (sum(ys) + 0.5 * 20) / (len(ys) + 20) if closed else 0.5       # shrunk pooled rate
    out = []
    for r, y in zip(closed, ys):
        res = str(r["result"]).lower()
        if "penalty" in res:
            continue                                                    # a flag says nothing about the concept
        attr = 0.5 if ("sack" in res or "fumble" in res or "int" in res and "incomplete" not in res) else 1.0
        groups = _groups_of(str(r.get("coverage_seen") or ""), str(r.get("coverage_typed") or ""))
        if not groups:
            continue                                                    # no revealed coverage: nothing to attribute
        f, p = r.get("executed_formation") or r.get("formation"), r.get("executed_play") or r.get("play")
        if r.get("executed_verification") not in ("verified", "confirmed") and r.get("executed_play") and r.get("executed_play") != r.get("play"):
            f, p = r.get("formation"), r.get("play")                    # unattested "used different": fall back to the shown call, flagged by weight
        z = max(-Z_CLIP, min(Z_CLIP, ((1.0 if y else 0.0) - p0) / (p0 * (1 - p0))))
        for concept, trust in _concepts_of(str(f), str(p)):
            for g, gw in groups:
                out.append({"cell": (g, concept), "z": z, "trust": trust * gw * attr * _exec_weight(r), "opponent": r.get("opponent_id")})
    return out


def posterior_cells(rows: Iterable[Mapping[str, Any]], opponent_id: str | None) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    obs = observations(rows)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for o in obs:
        by_cell.setdefault(o["cell"], []).append(o)
    learned: dict[tuple[str, str], float] = {}
    info: dict[str, Any] = {"version": VERSION, "observations": len(obs), "cells_updated": {}}
    for cell, items in by_cell.items():
        mu0, s0 = prior_cell(*cell)
        mu_m, sd_m = _post(mu0, s0, [(o["z"], o["trust"]) for o in items])
        mine = [(o["z"], o["trust"]) for o in items if opponent_id and o["opponent"] == opponent_id]
        mu_o, sd_o = _post(mu_m, OPPONENT_PRIOR_SD, mine) if mine else (mu_m, sd_m)
        learned[cell] = mu_o
        info["cells_updated"][f"{cell[0]}|{cell[1]}"] = {"prior": round(mu0, 4), "madden": round(mu_m, 4), "opponent": round(mu_o, 4),
                                                           "n_madden": round(sum(t for _, t in [(o["z"], o["trust"]) for o in items]), 2), "n_opponent": round(sum(t for _, t in mine), 2)}
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
