"""Hierarchical shrinkage defensive model, independent of the offensive artifact.

Trains only on verified human defensive executions. Opponent-specific rates
are stored separately from the general prior and are capped at live time.
Post-snap looks train historical tendencies only; they are never treated as
the current offensive call.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import FAMILIES, call_family
from cfb_coach.madden.model.defense_intelligence import verified_human_rows
from cfb_coach.madden.model.experimental_model import RateBucket, _bucket_map
from cfb_coach.outcome import parse_outcome


def _add_util(store: dict[str, list[float]], key: str, util: float) -> None:
    slot = store.setdefault(key, [0.0, 0.0])
    slot[0] += float(util)
    slot[1] += 1.0

MODEL_KIND = "hierarchical_shrinkage.v1"
OBJECTIVE = "situational_defense_utility.v1"
ARTIFACT_VERSION = "madden.defense.shrinkage.v1"
PRIOR_STRENGTH = 8.0
OPP_PRIOR_STRENGTH = 12.0
CONTEXT_STRENGTH = 10.0
PLAY_STRENGTH = 14.0
MAX_OPPONENT_DELTA = 0.075

# Higher is better for the defense. A 40-yard touchdown must hurt more than
# a routine first-down gain.
TD_PENALTY = 4.0
EXPLOSIVE_40 = 2.5
EXPLOSIVE_15 = 1.5
CONVERSION_PENALTY = 1.0
YARDS_WEIGHT = 0.025
STOP_REWARD = 0.80
TURNOVER_REWARD = 2.0
SACK_REWARD = 1.20
INCOMPLETE_REWARD = 0.70


def _kind(row: Mapping[str, Any]) -> str:
    payload = row.get("outcome_json")
    if isinstance(payload, Mapping) and payload.get("kind"):
        return str(payload.get("kind"))
    result = row.get("result")
    if result:
        return parse_outcome(str(result)).kind
    success = str(row.get("success") or "").lower()
    if success == "true":
        return "stop"
    if success == "false":
        return "gain"
    return "unknown"


def _yards(row: Mapping[str, Any]) -> int | None:
    value = row.get("yards")
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _distance(row: Mapping[str, Any]) -> int | None:
    try:
        if row.get("distance") in (None, ""):
            return None
        return int(row.get("distance"))
    except (TypeError, ValueError):
        return None


def converted_against(row: Mapping[str, Any]) -> bool | None:
    """Whether the offense converted. Unknown stays unknown."""
    kind = _kind(row)
    if kind in ("td", "convert"):
        return True
    if kind in ("int", "fumble", "incomplete", "sack", "stop"):
        return False
    yards = _yards(row)
    need = _distance(row)
    if yards is None or need is None:
        success = str(row.get("success") or "").lower()
        if success == "true":
            return False
        if success == "false":
            return True
        return None
    return yards >= need


def defensive_utility(row: Mapping[str, Any]) -> float | None:
    """Situational defensive objective. Missing labels return None."""
    kind = _kind(row)
    yards = _yards(row)
    if kind == "unknown" and yards is None and str(row.get("success") or "").lower() not in (
        "true", "false",
    ):
        return None
    util = 0.0
    if kind == "td":
        util -= TD_PENALTY
        if yards is not None and yards >= 40:
            util -= EXPLOSIVE_40
        elif yards is not None and yards >= 15:
            util -= EXPLOSIVE_15
        return round(util, 6)
    if kind in ("int", "fumble"):
        return TURNOVER_REWARD
    if kind == "sack":
        util += SACK_REWARD
    elif kind == "incomplete":
        util += INCOMPLETE_REWARD
    conv = converted_against(row)
    success = str(row.get("success") or "").lower()
    if conv is True:
        util -= CONVERSION_PENALTY
    elif conv is False or success == "true":
        util += STOP_REWARD
    elif success == "false":
        util -= CONVERSION_PENALTY * 0.6
    if yards is not None:
        util -= YARDS_WEIGHT * max(0, yards)
        if yards >= 40:
            util -= EXPLOSIVE_40
        elif yards >= 15:
            util -= EXPLOSIVE_15
    return round(util, 6)


def situation_token(
    down: Any = None,
    distance: Any = None,
    *,
    red_zone: bool = False,
    goal_line: bool = False,
    two_minute: bool = False,
    score_us: Any = None,
    score_them: Any = None,
) -> str:
    from cfb_coach.madden.model.defense_coordinator import _bucket

    base = _bucket(down, distance, red_zone=red_zone, goal_line=goal_line, two_minute=two_minute)
    phase = "score_unknown"
    try:
        if score_us is not None and score_them is not None:
            us, them = int(score_us), int(score_them)
            if us > them:
                phase = "lead"
            elif us < them:
                phase = "trail"
            else:
                phase = "tied"
    except (TypeError, ValueError):
        phase = "score_unknown"
    return f"{base}|{phase}"


def _personnel_token(row: Mapping[str, Any]) -> str:
    personnel = str(row.get("opp_personnel") or "").strip()
    opp_set = str(row.get("opp_set") or "").strip()
    if not personnel and not opp_set:
        return "missing"
    return f"{personnel or 'unk'}|{opp_set or 'unk'}"


def _observed_formation(row: Mapping[str, Any]) -> str:
    """Pre-snap offensive formation if recorded. Post-snap looks are ignored."""
    for key in ("opp_formation", "offensive_formation"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return "unknown"


def fit_shrinkage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit general and opponent-specific utility / stop tables."""
    from cfb_coach.madden.model.defense_coordinator import _bucket

    verified = [
        row for row in verified_human_rows(rows)
        if str(row.get("side") or "") == "defense"
    ]
    family_stop: dict[str, RateBucket] = {}
    family_util: dict[str, list[float]] = {}
    context_stop: dict[str, RateBucket] = {}
    context_util: dict[str, list[float]] = {}
    play_stop: dict[str, RateBucket] = {}
    play_util: dict[str, list[float]] = {}
    personnel_util: dict[str, list[float]] = {}
    opp_form_util: dict[str, list[float]] = {}
    opp_family_util: dict[str, dict[str, list[float]]] = defaultdict(dict)
    by_game: dict[str, int] = defaultdict(int)
    n = 0
    util_n = 0
    explosive_n = 0
    td_n = 0
    for row in verified:
        play = str(row.get("executed_play") or row.get("action_play") or "")
        fam = call_family(play)
        if fam not in FAMILIES:
            continue
        success = str(row.get("success") or "").lower()
        if success not in ("true", "false"):
            continue
        ok = success == "true"
        token = situation_token(
            row.get("down"), row.get("distance"),
            red_zone=bool(row.get("red_zone")),
            goal_line=bool(row.get("goal_line")),
            two_minute=bool(row.get("two_minute")),
            score_us=row.get("score_us") or row.get("score_for"),
            score_them=row.get("score_them") or row.get("score_against"),
        )
        mix = _bucket(
            row.get("down"), row.get("distance"),
            red_zone=bool(row.get("red_zone")),
            goal_line=bool(row.get("goal_line")),
            two_minute=bool(row.get("two_minute")),
        )
        family_stop.setdefault(fam, RateBucket()).add(ok)
        context_stop.setdefault(mix + "|" + fam, RateBucket()).add(ok)
        play_stop.setdefault(play, RateBucket()).add(ok)
        n += 1
        by_game[str(row.get("game_id"))] += 1
        util = defensive_utility(row)
        if util is None:
            continue
        _add_util(family_util, fam, util)
        _add_util(context_util, token + "|" + fam, util)
        _add_util(play_util, play, util)
        _add_util(personnel_util, _personnel_token(row), util)
        _add_util(opp_form_util, f"{_observed_formation(row)}|{fam}", util)
        opponent = str(row.get("opponent_id") or "").strip()
        if opponent:
            _add_util(opp_family_util[opponent], fam, util)
        util_n += 1
        yards = _yards(row)
        if _kind(row) == "td":
            td_n += 1
        if yards is not None and yards >= 15:
            explosive_n += 1

    def _pack(store: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        return {key: [round(float(slot[0]), 6), float(slot[1])] for key, slot in store.items()}

    opponents = {
        opponent: _pack(families) for opponent, families in opp_family_util.items()
    }
    quality = "empirical" if n >= 40 else ("limited" if n >= 10 else "prior_driven")
    return {
        "model_kind": MODEL_KIND,
        "objective": OBJECTIVE,
        "artifact_version": ARTIFACT_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "supervised_defensive_snaps": n,
        "utility_labeled_snaps": util_n,
        "evidence_quality": quality,
        "families": _bucket_map(family_stop),
        "contexts": _bucket_map(context_stop),
        "plays": _bucket_map(play_stop),
        "family_utility": _pack(family_util),
        "context_utility": _pack(context_util),
        "play_utility": _pack(play_util),
        "personnel_utility": _pack(personnel_util),
        "opp_formation_utility": _pack(opp_form_util),
        "opponent_family_utility": opponents,
        "supervised_by_game": dict(sorted(by_game.items())),
        "explosive_allowed": explosive_n,
        "touchdowns_allowed": td_n,
        "prior_strength": PRIOR_STRENGTH,
        "opponent_prior_strength": OPP_PRIOR_STRENGTH,
        "max_opponent_delta": MAX_OPPONENT_DELTA,
        "post_snap_not_used_as_live": True,
        "note": (
            f"Hierarchical shrinkage on {n} verified defensive executions "
            f"({util_n} with a utility label). Evidence quality: {quality}. "
            "Opponent tables are separate from the general prior."
        ),
    }


def _mean(raw: Sequence[float] | None, prior: float, strength: float) -> tuple[float, int]:
    if not raw or len(raw) < 2:
        return prior, 0
    total = float(raw[1])
    if total <= 0:
        return prior, 0
    return (float(raw[0]) + prior * strength) / (total + strength), int(total)


def utility_delta(
    model: Mapping[str, Any] | None,
    play: str,
    sit: Any,
    *,
    opponent_id: str = "",
) -> dict[str, Any]:
    """Pre-snap utility adjustment. Missing tables contribute 0."""
    if not model or not model.get("family_utility"):
        return {"delta": 0.0, "n": 0, "source": "absent"}
    family = call_family(play) or "unknown"
    token = situation_token(
        getattr(sit, "down", None), getattr(sit, "distance", None),
        red_zone=bool(getattr(sit, "red_zone", False)),
        goal_line=bool(getattr(sit, "goal_line", False)),
        two_minute=bool(getattr(sit, "two_minute", False)),
        score_us=getattr(sit, "score_us", None),
        score_them=getattr(sit, "score_them", None),
    )
    fam_u, n_fam = _mean(model.get("family_utility", {}).get(family), 0.0, PRIOR_STRENGTH)
    ctx_u, n_ctx = _mean(
        model.get("context_utility", {}).get(token + "|" + family), fam_u, CONTEXT_STRENGTH,
    )
    play_u, n_play = _mean(model.get("play_utility", {}).get(play), ctx_u, PLAY_STRENGTH)
    personnel = getattr(sit, "opp_personnel", None)
    opp_set = getattr(sit, "opp_set", None)
    pers_key = "missing"
    if personnel or opp_set:
        pers_key = f"{str(personnel or 'unk').strip()}|{str(opp_set or 'unk').strip()}"
    pers_u, n_pers = _mean(
        model.get("personnel_utility", {}).get(pers_key), 0.0, PRIOR_STRENGTH,
    )
    if pers_key == "missing":
        pers_u, n_pers = 0.0, 0
    opp_form = str(getattr(sit, "opp_formation", None) or "").strip() or "unknown"
    form_u, n_form = _mean(
        model.get("opp_formation_utility", {}).get(f"{opp_form}|{family}"), 0.0, PRIOR_STRENGTH,
    )
    if opp_form == "unknown":
        form_u, n_form = 0.0, 0
    general = 0.55 * play_u + 0.25 * ctx_u + 0.12 * fam_u + 0.08 * (pers_u + form_u)
    opp_delta = 0.0
    n_opp = 0
    if opponent_id:
        raw = ((model.get("opponent_family_utility") or {}).get(opponent_id) or {}).get(family)
        opp_mean, n_opp = _mean(raw, general, OPP_PRIOR_STRENGTH)
        opp_delta = max(-MAX_OPPONENT_DELTA, min(MAX_OPPONENT_DELTA, opp_mean - general))
    delta = 0.22 * general + opp_delta
    return {
        "delta": round(delta, 6),
        "n": n_play or n_ctx or n_fam,
        "n_play": n_play,
        "n_context": n_ctx,
        "n_family": n_fam,
        "n_personnel": n_pers,
        "n_opp_formation": n_form,
        "n_opponent": n_opp,
        "family": family,
        "situation": token,
        "used_live_post_snap": False,
        "source": "shrinkage" if (n_play or n_ctx or n_fam) else "prior",
    }
