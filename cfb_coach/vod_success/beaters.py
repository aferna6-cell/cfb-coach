"""Beater tables: call success shrunk toward the look, per stratum.

CPU strata also get a streamer-on-offense view. Human strata do not.
Unknown opponent types are not written here.
"""

from __future__ import annotations

import math
from collections import defaultdict

from cfb_coach.vod_success.features import Snap
from cfb_coach.vod_success.glm import LogisticModel, feature_map

TENTATIVE_N = 15
_Z90 = 1.6448536269514722
VIEW_OFFENSE = "vs_cpu_defense (streamer on offense)"


def prior_strength(cells: list[tuple[int, int]]) -> tuple[float, str]:
    """Method-of-moments prior, clipped to [2, 50]. Default 10 when the sample is thin."""
    big = [(n, s) for n, s in cells if n >= 2]
    if len(big) < 3:
        return 10.0, "default (fewer than 3 cells with n>=2)"
    n_sum = sum(n for n, _ in big)
    s_sum = sum(s for _, s in big)
    mu = s_sum / n_sum if n_sum else 0.5
    if mu <= 0.0 or mu >= 1.0:
        return 50.0, "method-of-moments: no excess variance -> cap 50"
    k = len(big)
    var = sum(n * ((s / n) - mu) ** 2 for n, s in big) / (k - 1)
    inv_n = sum(1.0 / n for n, _ in big) / k
    binomial = mu * (1.0 - mu) * inv_n
    excess = var - binomial
    if excess <= 1e-12:
        return 50.0, "method-of-moments: no excess variance -> cap 50"
    ceiling = mu * (1.0 - mu) - binomial
    if ceiling <= 1e-12:
        return 50.0, "method-of-moments: no excess variance -> cap 50"
    rho = min(0.999, max(1e-6, excess / ceiling))
    raw = (1.0 - rho) / rho
    clipped = min(50.0, max(2.0, raw))
    shown = 50.0 if clipped == 50.0 else (10.0 if clipped == 10.0 else round(clipped, 3))
    return shown, f"method-of-moments (raw {raw:.2f}, clipped to [2,50])"


def build_stratum_beaters(
    known: list[Snap],
    situation_model: LogisticModel,
    *,
    opponent_type: str,
) -> dict:
    base_rate = _rate(sum(1 for snap in known if snap.success), len(known))
    views = {
        "all_clean_pairs": _view([snap for snap in known if snap.clean], situation_model),
    }
    if opponent_type == "cpu":
        offense = [snap for snap in known if snap.clean and snap.side == "offense"]
        views[VIEW_OFFENSE] = _view(offense, situation_model)
    return {
        "stratum_base_rate_all_known": base_rate,
        "views": views,
    }


def _view(snaps: list[Snap], situation_model: LogisticModel) -> dict:
    probs = {
        id(snap): situation_model.predict_row(feature_map(snap, situation_only=True))
        for snap in snaps
    }
    return {
        "n_rows": len(snaps),
        "stratum_clean_success_rate": _rate(sum(1 for snap in snaps if snap.success), len(snaps)),
        "look_family": _block(
            snaps,
            probs,
            key=lambda snap: snap.look_family,
            display=lambda look_id, _members: look_id,
        ),
        "look": _block(
            snaps,
            probs,
            key=lambda snap: snap.look_key or "none",
            display=lambda _look_id, members: _majority(snap.look for snap in members) or _look_id,
        ),
    }


def _block(snaps, probs, *, key, display) -> dict:
    grouped: dict[str, list[Snap]] = defaultdict(list)
    for snap in snaps:
        grouped[key(snap)].append(snap)
    draft: list[tuple[str, list[tuple[str, list[Snap]]]]] = []
    cell_counts: list[tuple[int, int]] = []
    for look_id, members in grouped.items():
        by_call: dict[str, list[Snap]] = defaultdict(list)
        for snap in members:
            by_call[snap.call_key].append(snap)
        pairs = list(by_call.items())
        draft.append((look_id, pairs))
        for _call, group in pairs:
            cell_counts.append((len(group), sum(1 for snap in group if snap.success)))
    strength, note = prior_strength(cell_counts)
    looks = {}
    flat = []
    for look_id, pairs in draft:
        members = grouped[look_id]
        look_success = sum(1 for snap in members if snap.success)
        look_prior = _mean(probs[id(snap)] for snap in members)
        calls = [
            _call_row(group, strength, look_prior, probs)
            for _call, group in pairs
        ]
        calls.sort(key=lambda row: (-row["shrunk_success"], row["call"].casefold()))
        label = display(look_id, members)
        looks[look_id] = {
            "look": label,
            "n": len(members),
            "successes": look_success,
            "raw_success": _rate(look_success, len(members)),
            "look_prior": round(look_prior, 4),
            "calls": calls,
        }
        for call in calls:
            flat.append({"n": call["n"], "look": label, "call": call["call"]})
    ordered = dict(sorted(looks.items(), key=lambda item: (-item[1]["n"], item[0])))
    flat.sort(key=lambda row: (-row["n"], str(row["look"]), row["call"]))
    return {
        "prior_strength_m": strength,
        "prior_strength_note": note,
        "n_cells": len(flat),
        "cells_n_ge_tentative": sum(1 for row in flat if row["n"] >= TENTATIVE_N),
        "largest_cells": flat[:5],
        "looks": ordered,
    }


def _call_row(group: list[Snap], strength: float, look_prior: float, probs) -> dict:
    n = len(group)
    successes = sum(1 for snap in group if snap.success)
    shrunk = (successes + strength * look_prior) / (n + strength) if n + strength else 0.0
    expected = _mean(probs[id(snap)] for snap in group)
    play_conf = [snap.conf_play for snap in group if snap.conf_play is not None]
    success_conf = [snap.conf_success for snap in group if snap.conf_success is not None]
    tentative = n < TENTATIVE_N
    return {
        "call": _majority(snap.call for snap in group) or group[0].call_key,
        "call_key": group[0].call_key,
        "call_class": group[0].call_class,
        "n": n,
        "successes": successes,
        "raw_success": _rate(successes, n),
        "shrunk_success": round(shrunk, 4),
        "lower_bound_90": round(_wilson(successes + strength * look_prior, n + strength), 4),
        "wilson_lb_90_raw": round(_wilson(float(successes), float(n)), 4),
        "situation_expected": round(expected, 4),
        "lift_vs_situation": round(shrunk - expected, 4),
        "n_with_off_adjustment": sum(1 for snap in group if snap.off_adj),
        "n_streamer_offense": sum(1 for snap in group if snap.side == "offense"),
        "n_streamer_defense": sum(1 for snap in group if snap.side == "defense"),
        "n_vods": len({snap.vod for snap in group}),
        "mean_conf_play": round(_mean(play_conf), 4) if play_conf else 0.0,
        "mean_conf_success": round(_mean(success_conf), 4) if success_conf else 0.0,
        "tentative": tentative,
        "confidence": "tentative" if tentative else "supported",
    }


def _majority(values) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        text = (value or "").strip()
        if text:
            counts[text] += 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[0][0]


def _wilson(successes: float, n: float, z: float = _Z90) -> float:
    if n <= 0:
        return 0.0
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    margin = z * math.sqrt(max(0.0, phat * (1.0 - phat) / n + z2 / (4.0 * n * n)))
    return (center - margin) / denom


def _rate(successes: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return round(successes / n, 4)


def _mean(values) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
