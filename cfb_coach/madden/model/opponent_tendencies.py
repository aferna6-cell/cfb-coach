"""Learned opponent tendencies for live offense and defense calls.

Roster ratings are not used. A missing table, an unknown opponent, or a
sample of zero leaves every score exactly as the caller computed it.
Small samples shrink toward that baseline. The adjustment is capped.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.schema import OpponentContextKey

TABLE_VERSION = "learned_tendencies.v1"
SHRINK_STRENGTH = 8.0
MIN_BUCKET = 3
OFFENSE_LOGIT_CAP = 0.15
DEFENSE_SUCCESS_CAP = 0.12
_LIGHT = ("dime", "dollar", "quarter")
_COUNTER_SCALE = 0.20
_LIGHT_SCALE = 0.40

_FAMILY_TOKENS = {
    "single_high": ("singlehigh", "cover3", "c3", "sky", "cloud", "buzz"),
    "two_high": ("twohigh", "cover4", "cover6", "quarters", "palms"),
    "cover2": ("cover2", "tampa", "c2"),
    "man": ("man", "cover1", "c1"),
    "pressure": ("pressure", "blitz", "fire"),
}


def opponent_context_key(opponent_id: str | None) -> OpponentContextKey:
    """Identity of the learned-tendency lookup. Not a roster snapshot."""
    text = str(opponent_id or "").strip()
    return OpponentContextKey(
        opponent_id=text or None,
        table_version=TABLE_VERSION,
    )


def coverage_bucket_name(text: str | None) -> str:
    """Coverage family for a logged shell. Cover 2 Man stays its own bucket."""
    raw = (text or "").strip().lower()
    if not raw:
        return "unknown"
    if ("cover 2" in raw or "cover2" in raw) and "man" in raw:
        return "cover2_man"
    if any(token in raw for token in ("blitz", "pressure", "fire")):
        return "pressure"
    from cfb_coach.madden.model.experimental_model import _cov_bucket

    return _cov_bucket(text)


def empty_profile(opponent_id: str | None = None) -> dict[str, Any]:
    return {
        "opponent_id": opponent_id or "",
        "n": 0,
        "shrink": 0.0,
        "mix": {},
        "labels": {},
        "scope": "",
        "play_rates": {},
        "concepts": {},
        "run_share": None,
        "summary": "no learned tendencies",
    }


def _shrink(n: float) -> float:
    if n <= 0:
        return 0.0
    return float(n) / (float(n) + SHRINK_STRENGTH)


def _cap(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def _row_get(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return getattr(row, key, None)


def _shares(counts: Mapping[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items() if value > 0}


def _blend_shares(
    left: Mapping[str, float], n_left: float,
    right: Mapping[str, float], n_right: float,
) -> dict[str, float]:
    if n_left <= 0 and n_right <= 0:
        return {}
    if n_left <= 0:
        return dict(right)
    if n_right <= 0:
        return dict(left)
    keys = set(left) | set(right)
    mixed = {
        key: (n_left * left.get(key, 0.0) + n_right * right.get(key, 0.0)) / (n_left + n_right)
        for key in keys
    }
    return _shares(mixed)


def _dist_class(distance: Any) -> str:
    try:
        dist = int(distance)
    except (TypeError, ValueError):
        return "normal"
    if dist <= 3:
        return "short"
    if dist >= 8:
        return "long"
    return "normal"


def _same_bucket(sit: Any, down: Any, distance: Any) -> bool:
    if sit is None or getattr(sit, "down", None) is None:
        return False
    try:
        return int(down) == int(sit.down) and _dist_class(distance) == _dist_class(getattr(sit, "distance", None))
    except (TypeError, ValueError):
        return False


def _verified(row: Any) -> bool:
    status = str(_row_get(row, "executed_status") or "").lower()
    verification = str(_row_get(row, "executed_verification") or "").lower()
    return status == "identified" or verification in ("verified", "confirmed")


def _snap_rows(db: Any, opponent_id: str) -> list[Any]:
    if db is None or not opponent_id or not hasattr(db, "conn"):
        return []
    try:
        return list(db.conn.execute(
            "SELECT side, down, distance, coverage_seen, concept_seen, formation, play, "
            "result, executed_status, executed_verification FROM snaps WHERE opponent_id = ?",
            (opponent_id,),
        ).fetchall())
    except Exception:  # noqa: BLE001 — a database without those columns has no verified snaps
        return []


def _tendency_rows(db: Any, opponent_id: str) -> list[Any]:
    if db is None or not opponent_id or not hasattr(db, "get_tendencies"):
        return []
    try:
        return list(db.get_tendencies(opponent_id))
    except Exception:  # noqa: BLE001
        return []


def _count_coverage(rows: Sequence[Any], *, bucket: str | None) -> tuple[dict[str, float], dict[str, str], float]:
    counts: dict[str, float] = {}
    labels: dict[str, str] = {}
    label_n: dict[str, float] = {}
    total = 0.0
    for row in rows:
        name = str(_row_get(row, "bucket") or "")
        if name.endswith("_seed"):
            continue
        if bucket is None:
            if name != "their_coverage":
                continue
        elif name != bucket:
            continue
        key = str(_row_get(row, "key") or "")
        cov = coverage_bucket_name(key)
        if cov in ("unknown", "other"):
            continue
        amount = float(_row_get(row, "count") or 0)
        if amount <= 0:
            continue
        counts[cov] = counts.get(cov, 0.0) + amount
        total += amount
        if amount >= label_n.get(cov, 0.0):
            labels[cov] = key
            label_n[cov] = amount
    return counts, labels, total


def _count_snap_coverage(rows: Sequence[Any], sit: Any, *, bucketed: bool) -> tuple[dict[str, float], dict[str, str], float]:
    counts: dict[str, float] = {}
    labels: dict[str, str] = {}
    total = 0.0
    for row in rows:
        if str(_row_get(row, "side") or "offense").startswith("d"):
            continue
        if not _verified(row):
            continue
        seen = str(_row_get(row, "coverage_seen") or "").strip()
        if not seen:
            continue
        if bucketed and not _same_bucket(sit, _row_get(row, "down"), _row_get(row, "distance")):
            continue
        cov = coverage_bucket_name(seen)
        if cov in ("unknown", "other"):
            continue
        counts[cov] = counts.get(cov, 0.0) + 1.0
        total += 1.0
        labels.setdefault(cov, seen)
    return counts, labels, total


def _play_rates(rows: Sequence[Any], opponent_side: str) -> dict[str, list[float]]:
    from cfb_coach.outcome import outcome_success

    rates: dict[str, list[float]] = {}
    for row in rows:
        side = str(_row_get(row, "side") or "offense")
        if opponent_side == "offense" and side.startswith("d"):
            continue
        if opponent_side == "defense" and not side.startswith("d"):
            continue
        if not _verified(row):
            continue
        formation = str(_row_get(row, "formation") or "")
        play = str(_row_get(row, "play") or "")
        if not formation or not play:
            continue
        ok = outcome_success(_row_get(row, "result"), "offense" if not side.startswith("d") else "defense")
        slot = rates.setdefault(f"{formation}|{play}", [0.0, 0.0])
        slot[1] += 1.0
        if ok:
            slot[0] += 1.0
    return rates


def _learned_weights(db: Any, opponent_id: str, side: str) -> Any:
    try:
        from cfb_coach.learning import LearnedWeights

        return LearnedWeights.load(db, opponent_id, side=side)
    except Exception:  # noqa: BLE001 — missing learn_stats is no personal result
        return None


def build_offense_profile(db: Any, opponent_id: str, sit: Any = None) -> dict[str, Any]:
    """Coverage mix and play results for one opponent. Empty when nothing is stored."""
    profile = empty_profile(opponent_id)
    from cfb_coach.tendency import situation_bucket

    specific = f"their_coverage|{situation_bucket(sit)}" if sit is not None else None
    rows = _tendency_rows(db, opponent_id)
    snaps = _snap_rows(db, opponent_id)
    spec_counts, spec_labels, spec_n = _count_coverage(rows, bucket=specific) if specific else ({}, {}, 0.0)
    all_counts, all_labels, all_n = _count_coverage(rows, bucket=None)
    snap_spec, snap_labels, snap_spec_n = _count_snap_coverage(snaps, sit, bucketed=True)
    snap_all, snap_all_labels, snap_all_n = _count_snap_coverage(snaps, sit, bucketed=False)

    if spec_n >= MIN_BUCKET or snap_spec_n >= MIN_BUCKET:
        mix = _blend_shares(_shares(spec_counts), spec_n, _shares(snap_spec), snap_spec_n)
        labels = {**snap_labels, **spec_labels}
        scope = "this down"
        n = max(spec_n, snap_spec_n)
    else:
        mix = _blend_shares(_shares(all_counts), all_n, _shares(snap_all), snap_all_n)
        labels = {**snap_all_labels, **all_labels}
        scope = "all downs"
        n = max(all_n, snap_all_n)
    profile["mix"] = mix
    profile["labels"] = labels
    profile["scope"] = scope if n else ""
    profile["n"] = int(n)
    profile["shrink"] = _shrink(n)
    profile["play_rates"] = _play_rates(snaps, "offense")
    learned = _learned_weights(db, opponent_id, "offense")
    if learned is not None:
        profile["learned_weights"] = learned
        try:
            from cfb_coach.zones import zone_of_situation

            profile["zone"] = zone_of_situation(sit) if sit is not None else "open"
        except Exception:  # noqa: BLE001
            profile["zone"] = "open"
    profile["summary"] = format_offense_summary(profile)
    return profile


def format_offense_summary(profile: Mapping[str, Any] | None) -> str:
    if not profile or int(profile.get("n") or 0) <= 0:
        return "no learned tendencies"
    labels = profile.get("labels") or {}
    mix = profile.get("mix") or {}
    top = sorted(mix.items(), key=lambda item: (-item[1], item[0]))[:3]
    shown = ", ".join(
        f"{labels.get(key, key)} {round(100 * share)}%" for key, share in top
    ) or "no coverage mix"
    scope = profile.get("scope") or "all downs"
    return (
        f"{scope}: {shown} (n={int(profile['n'])}, "
        f"shrink={float(profile.get('shrink') or 0):.2f})"
    )


def _matchup_rate(artifact: Any, play: str, coverage: str, prior: float) -> float:
    from cfb_coach.madden.model.experimental_model import (
        FAMILY_PRIOR_STRENGTH,
        _play_concept,
        _play_family,
        _rate_from,
    )

    concept = _play_concept(play)
    family = _play_family(play)
    keys = [f"{concept}|{coverage}", f"{family}|{coverage}"]
    if coverage == "cover2_man":
        keys.extend([
            f"{concept}|cover2", f"{concept}|cover1",
            f"{family}|cover2", f"{family}|cover1",
        ])
    stores = (
        getattr(artifact, "concept_vs_cov", None) or {},
        getattr(artifact, "play_vs_cov", None) or {},
    )
    for key in keys:
        for store in stores:
            raw = store.get(key)
            if raw:
                return _rate_from(raw, prior, FAMILY_PRIOR_STRENGTH)
    return prior


def offense_tendency_delta(
    artifact: Any,
    formation: str,
    play: str,
    *,
    baseline: float,
    profile: Mapping[str, Any] | None,
) -> float:
    """Capped logit added to one candidate. Zero when the profile has no sample."""
    if not profile or int(profile.get("n") or 0) <= 0:
        return 0.0
    shrink = float(profile.get("shrink") or 0.0)
    delta = 0.0
    mix = profile.get("mix") or {}
    if mix and shrink > 0:
        expected = sum(
            share * _matchup_rate(artifact, play, cov, baseline)
            for cov, share in mix.items()
        )
        delta += shrink * (expected - baseline) * 2.0
    rates = (profile.get("play_rates") or {}).get(f"{formation}|{play}")
    if rates and float(rates[1]) > 0:
        trials = float(rates[1])
        weight = trials / (trials + SHRINK_STRENGTH)
        global_rate = float(getattr(artifact, "global_rate", None) or 0.5)
        shrunk = (float(rates[0]) + global_rate * SHRINK_STRENGTH) / (trials + SHRINK_STRENGTH)
        delta += weight * (shrunk - global_rate) * 1.5
    learned = profile.get("learned_weights")
    if learned is not None:
        try:
            info = learned.learned_score(profile.get("zone") or "open", formation, play)
        except Exception:  # noqa: BLE001
            info = {}
        n_obs = float(info.get("n_obs_zone") or 0)
        if n_obs > 0:
            weight = n_obs / (n_obs + SHRINK_STRENGTH)
            delta += weight * float(info.get("learned") or 0.0)
    return _cap(delta, OFFENSE_LOGIT_CAP)


def apply_offense_probability(probability: float, delta: float) -> float:
    """Return ``probability`` unchanged when ``delta`` is zero."""
    if not delta:
        return probability
    p = min(1.0 - 1e-4, max(1e-4, float(probability)))
    logit = math.log(p / (1.0 - p)) + delta
    return 1.0 / (1.0 + math.exp(-logit))


def _concept_name(text: str) -> str:
    from cfb_coach.madden.situation import concept_family

    family = concept_family(text)
    if family:
        return family
    raw = (text or "").lower()
    if any(token in raw for token in ("pass", "vertical", "mesh", "flood", "curl")):
        return "pass"
    return "other"


def _count_concepts(rows: Sequence[Any], *, bucket: str | None) -> tuple[dict[str, float], float, float]:
    counts: dict[str, float] = {}
    runs = 0.0
    total = 0.0
    for row in rows:
        name = str(_row_get(row, "bucket") or "")
        if name.endswith("_seed"):
            continue
        if bucket is None:
            if name != "offense_concept":
                continue
        elif name != bucket:
            continue
        key = str(_row_get(row, "key") or "")
        amount = float(_row_get(row, "count") or 0)
        if amount <= 0 or not key:
            continue
        family = _concept_name(key)
        counts[family] = counts.get(family, 0.0) + amount
        total += amount
        if family == "run":
            runs += amount
    return counts, runs, total


def _count_snap_concepts(rows: Sequence[Any], sit: Any, *, bucketed: bool) -> tuple[dict[str, float], float, float]:
    counts: dict[str, float] = {}
    runs = 0.0
    total = 0.0
    for row in rows:
        if not str(_row_get(row, "side") or "").startswith("d"):
            continue
        if not _verified(row):
            continue
        seen = str(_row_get(row, "concept_seen") or "").strip()
        if not seen:
            continue
        if bucketed and not _same_bucket(sit, _row_get(row, "down"), _row_get(row, "distance")):
            continue
        family = _concept_name(seen)
        counts[family] = counts.get(family, 0.0) + 1.0
        total += 1.0
        if family == "run":
            runs += 1.0
    return counts, runs, total


def build_defense_profile(db: Any, opponent_id: str, sit: Any = None) -> dict[str, Any]:
    """Run/pass and concept mix from this opponent's learned offense. Empty if none."""
    profile = empty_profile(opponent_id)
    from cfb_coach.tendency import situation_bucket

    specific = f"offense_concept|{situation_bucket(sit)}" if sit is not None else None
    rows = _tendency_rows(db, opponent_id)
    snaps = _snap_rows(db, opponent_id)
    spec_counts, spec_runs, spec_n = _count_concepts(rows, bucket=specific) if specific else ({}, 0.0, 0.0)
    all_counts, all_runs, all_n = _count_concepts(rows, bucket=None)
    snap_spec, snap_runs, snap_spec_n = _count_snap_concepts(snaps, sit, bucketed=True)
    snap_all, snap_all_runs, snap_all_n = _count_snap_concepts(snaps, sit, bucketed=False)
    if spec_n >= MIN_BUCKET or snap_spec_n >= MIN_BUCKET:
        counts = _blend_shares(_shares(spec_counts), spec_n, _shares(snap_spec), snap_spec_n)
        runs = spec_runs + snap_runs
        total = spec_n + snap_spec_n
        n = max(spec_n, snap_spec_n)
        scope = "this down"
    else:
        counts = _blend_shares(_shares(all_counts), all_n, _shares(snap_all), snap_all_n)
        runs = all_runs + snap_all_runs
        total = all_n + snap_all_n
        n = max(all_n, snap_all_n)
        scope = "all downs"
    # Shrinkage uses the larger source, not the sum, so one snap is not counted twice.
    profile["n"] = int(n)
    profile["shrink"] = _shrink(n)
    profile["concepts"] = counts
    profile["scope"] = scope if n else ""
    profile["run_share"] = (runs / total) if total > 0 else None
    profile["summary"] = format_defense_summary(profile)
    return profile


POOLED_SHRINK_SCALE = 0.5
POOLED_SHRINK_CAP = 0.4


def pooled_human_profile(db: Any, sit: Any = None, *, exclude: str | None = None) -> dict[str, Any]:
    """A starting prior for a human opponent with no history: every other human, pooled.

    Weighted by each opponent's evidence. Shrink is halved and capped at 0.4,
    so the opponent's own snaps take over as soon as they are logged.
    """
    from cfb_coach.opponents import is_cpu_opponent

    profile = empty_profile(exclude)
    try:
        ids = sorted({str(r[0]) for r in db.conn.execute(
            "SELECT id FROM opponents UNION SELECT opponent_id FROM tendencies "
            "UNION SELECT opponent_id FROM snaps WHERE side = 'defense'"
        ).fetchall() if r[0]})
    except Exception:  # noqa: BLE001
        return profile
    runs = total = 0.0
    concepts: dict[str, float] = {}
    used = 0
    for oid in ids:
        if oid == exclude or is_cpu_opponent(oid):
            continue
        p = build_defense_profile(db, oid, sit)
        n = float(p.get("n") or 0)
        if n <= 0 or p.get("run_share") is None:
            continue
        used += 1
        total += n
        runs += n * float(p["run_share"])
        for name, share in (p.get("concepts") or {}).items():
            concepts[name] = concepts.get(name, 0.0) + n * float(share)
    if total <= 0:
        return profile
    profile.update(
        n=int(total), run_share=runs / total, concepts=_shares(concepts), scope="pooled human opponents",
        shrink=min(POOLED_SHRINK_CAP, POOLED_SHRINK_SCALE * _shrink(total)), pooled_from=used,
    )
    profile["summary"] = f"league prior from {used} human opponents: " + format_defense_summary(profile)
    return profile


def format_defense_summary(profile: Mapping[str, Any] | None) -> str:
    if not profile or int(profile.get("n") or 0) <= 0 or profile.get("run_share") is None:
        return "no learned tendencies"
    run = float(profile["run_share"])
    scope = profile.get("scope") or "all downs"
    concepts = profile.get("concepts") or {}
    top = sorted(concepts.items(), key=lambda item: (-item[1], item[0]))[:3]
    mix = ", ".join(f"{name} {round(100 * share)}%" for name, share in top)
    return (
        f"run {round(100 * run)}% / pass {round(100 * (1 - run))}% {scope}"
        + (f" ({mix})" if mix else "")
        + f" (n={int(profile['n'])}, shrink={float(profile.get('shrink') or 0):.2f})"
    )


def _is_light_package(row: Mapping[str, Any]) -> bool:
    package = str(row.get("package") or "").lower()
    formation = str(row.get("formation") or "").lower()
    return any(token in package or token in formation for token in _LIGHT)


def _family_answers(row_family: str, counter: str) -> bool:
    fam = (row_family or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    return any(token in fam for token in _FAMILY_TOKENS.get(counter, (counter.replace("_", ""),)))


def defense_success_delta(row: Mapping[str, Any], profile: Mapping[str, Any] | None) -> float:
    """Added to predicted offense success. Positive makes a call less attractive.

    Zero when the opponent has no learned offense sample.
    """
    if not profile or int(profile.get("n") or 0) <= 0 or profile.get("run_share") is None:
        return 0.0
    shrink = float(profile.get("shrink") or 0.0)
    if shrink <= 0:
        return 0.0
    from cfb_coach.madden.defense_select import COUNTERS

    delta = 0.0
    run_share = float(profile["run_share"])
    if _is_light_package(row):
        delta += shrink * max(0.0, run_share - 0.5) * _LIGHT_SCALE
    concepts = profile.get("concepts") or {}
    family = str(row.get("family") or "")
    for their, share in concepts.items():
        for ours, weight in (COUNTERS.get(their) or {}).items():
            if _family_answers(family, ours):
                # A positive counter weight means this family answers the concept,
                # so predicted offense success goes down.
                delta -= shrink * float(share) * float(weight) * _COUNTER_SCALE
    return _cap(delta, DEFENSE_SUCCESS_CAP)
