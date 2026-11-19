"""The scheme model's input vector. Same function for history rows and for adjusted variants.

Groups (all pre-snap):
  s_*   down & distance, field position, red zone / goal-to-go / backed up
  b_*   opponent beliefs: man / zone / match, shells, pressure, information
  f_*   formation tokens (from the formation NAME) and t_* play type (from the play NAME)
  g_*/r_*  adjusted-design geometry and route-combination features (design.route_features)
  x_*   interactions route shape x belief x situation (learned, not rules)
  k_*   existing knowledge: the existing coordinator's logit (offset) and the football_priors matchup edge
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme.beliefs import Belief

SIT = ("s_down1", "s_down2", "s_down3", "s_down4", "s_dist", "s_logdist", "s_short", "s_medium", "s_long",
       "s_yard", "s_redzone", "s_goal_to_go", "s_inside10", "s_backed", "s_late_down")
BEL = ("b_man", "b_zone", "b_match", "b_one_high", "b_two_high", "b_zero", "b_pressure", "b_info", "b_entropy")
INTER = ("x_man_manbeater", "x_man_crossing", "x_zone_zonebeater", "x_zone_curl", "x_match_crossing",
         "x_press_quick", "x_press_deep", "x_twohigh_deepmid", "x_onehigh_vertical", "x_onehigh_flat",
         "x_short_quick", "x_short_run", "x_long_deep", "x_long_depth", "x_redzone_vertical", "x_goal_quick",
         "x_goal_run", "x_late_depth")
KNOW = ("k_matchup_edge",)
OPP = ("o_cpu",)
NAMES: tuple[str, ...] = (SIT + OPP + BEL + tuple("f_" + t for t in dz.FORMATION_TOKENS)
                          + tuple("t_" + t for t in dz.PLAY_TYPES) + dz.ROUTE_FEATURES + INTER + KNOW)


def _i(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def situation(down: Any, distance: Any, yardline: Any) -> dict[str, float]:
    d, dist, yl = _i(down), _i(distance), _i(yardline)
    f = {n: 0.0 for n in SIT}
    if d in (1, 2, 3, 4):
        f[f"s_down{d}"] = 1.0
        f["s_late_down"] = 1.0 if d >= 3 else 0.0
    if dist is not None:
        dist = max(0, dist)
        f["s_dist"] = min(dist, 30) / 10.0
        f["s_logdist"] = math.log1p(dist)
        f["s_short"] = 1.0 if dist <= 3 else 0.0
        f["s_medium"] = 1.0 if 4 <= dist <= 7 else 0.0
        f["s_long"] = 1.0 if dist >= 8 else 0.0
    if yl is not None:
        f["s_yard"] = max(0, min(100, yl)) / 100.0
        f["s_redzone"] = 1.0 if yl >= 80 else 0.0
        f["s_inside10"] = 1.0 if yl >= 90 else 0.0
        f["s_backed"] = 1.0 if yl <= 10 else 0.0
        if dist is not None:
            f["s_goal_to_go"] = 1.0 if yl + dist >= 100 else 0.0
    return f


def vector(*, down: Any, distance: Any, yardline: Any, belief: Belief, design: dz.Design, opponent_type: Any = None,
           route_f: Mapping[str, float] | None = None) -> tuple[list[float], dict[str, float]]:
    """(vector in NAMES order, the named dict). ``route_f`` may be precomputed for a fixed design."""
    f: dict[str, float] = {}
    f.update(situation(down, distance, yardline))
    f["o_cpu"] = 1.0 if str(opponent_type or "") == "cpu" else 0.0
    fam = belief.families
    f.update({"b_man": fam["man"], "b_zone": fam["zone"], "b_match": fam["match"], "b_one_high": fam["one_high"],
              "b_two_high": fam["two_high"], "b_zero": fam["zero"], "b_pressure": belief.pressure,
              "b_info": belief.info, "b_entropy": belief.entropy})
    f.update(dz.formation_features(design.formation))
    f.update(dz.play_type_features(design.play))
    r = dict(route_f) if route_f is not None else dz.route_features(design)
    f.update(r)
    run = f.get("t_run", 0.0) + f.get("t_option", 0.0)
    cross = r["r_crossing"] + r["r_horizontal"]
    f.update({
        "x_man_manbeater": fam["man"] * r["r_man_beater"], "x_man_crossing": fam["man"] * cross,
        "x_zone_zonebeater": fam["zone"] * r["r_zone_beater"], "x_zone_curl": fam["zone"] * r["r_curl_sit"],
        "x_match_crossing": fam["match"] * cross, "x_press_quick": belief.pressure * r["r_quick"],
        "x_press_deep": belief.pressure * r["r_deep"], "x_twohigh_deepmid": fam["two_high"] * r["r_deep_middle"],
        "x_onehigh_vertical": fam["one_high"] * r["r_vertical"], "x_onehigh_flat": fam["one_high"] * r["r_flat_short"],
        "x_short_quick": f["s_short"] * r["r_quick"], "x_short_run": f["s_short"] * run,
        "x_long_deep": f["s_long"] * r["r_deep"], "x_long_depth": f["s_long"] * r["r_mean_depth"],
        "x_redzone_vertical": f["s_redzone"] * r["r_vertical"], "x_goal_quick": f["s_goal_to_go"] * r["r_quick"],
        "x_goal_run": f["s_goal_to_go"] * run, "x_late_depth": f["s_late_down"] * r["r_mean_depth"],
    })
    edge, _cells = dz.matchup_edge(design, belief.mix, belief.pressure)
    f["k_matchup_edge"] = edge
    return [float(f.get(n, 0.0)) for n in NAMES], f
