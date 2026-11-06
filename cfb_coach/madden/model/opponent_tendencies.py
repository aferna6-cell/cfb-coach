"""Learned opponent tendencies for live offense and defense calls.

Roster ratings are not used. A missing table, an unknown opponent, or a
sample of zero leaves every score exactly as the caller computed it.
Small samples shrink toward that baseline. The adjustment is capped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

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


PRIOR_HANDOFF = 12


def with_pooled_prior(db: Any, own: Mapping[str, Any] | None, sit: Any, opponent_id: str) -> dict[str, Any]:
    """A thin human profile leaned toward the league prior, handing off as his snaps pile up.

    His own share weighs n / (n + 12): about a third after 6 logged plays, half after 12.
    """
    own = dict(own or empty_profile(opponent_id))
    n = float(own.get("n") or 0)
    if n >= 4 * PRIOR_HANDOFF:
        return own
    prior = pooled_human_profile(db, sit, exclude=opponent_id)
    if int(prior.get("n") or 0) <= 0 or prior.get("run_share") is None:
        return own
    if n <= 0 or own.get("run_share") is None:
        return prior
    w = n / (n + PRIOR_HANDOFF)
    keys = set(own.get("concepts") or {}) | set(prior.get("concepts") or {})
    concepts = {
        k: w * float((own.get("concepts") or {}).get(k, 0.0)) + (1 - w) * float(prior["concepts"].get(k, 0.0))
        for k in keys
    }
    blended = dict(own)
    blended.update(
        run_share=w * float(own["run_share"]) + (1 - w) * float(prior["run_share"]),
        concepts=_shares(concepts),
        shrink=max(float(own.get("shrink") or 0.0), (1 - w) * float(prior.get("shrink") or 0.0)),
        scope=f"his {int(n)} plays + league prior",
    )
    blended["summary"] = f"{round(100 * w)}% his own, rest league prior: " + format_defense_summary(blended)
    return blended


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


# ---------------------------------------------------------------------------
# Hierarchical run/pass prior: league -> playing style -> individual.
# Not read by the live caller yet; evaluated offline in heldout_eval.


@dataclass(frozen=True)
class RunPassObservation:
    """One snap where the opponent's run/pass was observed.

    Comes from what he was seen doing, so it does not depend on whether our
    own defensive call was verified.
    """

    opponent_id: str
    game_id: str
    down: Any
    distance: Any
    is_pass: int


def _situation_key(down: Any, distance: Any) -> str:
    try:
        d = int(down)
    except (TypeError, ValueError):
        return "unknown"
    return f"{d}_{_dist_class(distance)}"


def _logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RunPassPrior:
    """League situational pass rates and evidence-learned playing styles.

    A style is a cluster of opponents by how much more (or less) they pass than
    the league in the same situations. Styles are only formed when there are at
    least two qualifying opponents per style; otherwise the prior is league-only.
    """

    situation_strength: float = 10.0
    style_strength: float = 10.0
    n_styles: int = 2
    min_style_plays: int = 15
    overall: float = 0.5
    league: dict[str, float] = field(default_factory=dict)
    styles: list[dict[str, Any]] = field(default_factory=list)

    def fit(self, observations: Iterable[RunPassObservation]) -> "RunPassPrior":
        obs = list(observations)
        self.overall = (sum(o.is_pass for o in obs) + 1.0) / (len(obs) + 2.0)
        self.league = self._table(obs, lambda key: self.overall, self.situation_strength)
        self.styles = self._fit_styles(obs)
        return self

    @staticmethod
    def _table(obs: Sequence[RunPassObservation], parent: Any, strength: float) -> dict[str, float]:
        counts: dict[str, list[float]] = {}
        for o in obs:
            c = counts.setdefault(_situation_key(o.down, o.distance), [0.0, 0.0])
            c[0] += o.is_pass
            c[1] += 1
        return {k: (s + strength * parent(k)) / (m + strength) for k, (s, m) in counts.items()}

    def league_rate(self, down: Any, distance: Any) -> float:
        return self.league.get(_situation_key(down, distance), self.overall)

    def style_rate(self, style: Mapping[str, Any], down: Any, distance: Any) -> float:
        key = _situation_key(down, distance)
        if key in style["rates"]:
            return style["rates"][key]
        return _sigmoid(_logit(self.league_rate(down, distance)) + style["offset"])

    def _fit_styles(self, obs: Sequence[RunPassObservation]) -> list[dict[str, Any]]:
        by_opp: dict[str, list[RunPassObservation]] = {}
        for o in obs:
            by_opp.setdefault(o.opponent_id, []).append(o)
        eligible = {k: v for k, v in by_opp.items() if len(v) >= self.min_style_plays}
        if self.n_styles < 2 or len(eligible) < 2 * self.n_styles:
            return []
        offsets = {}
        for k, v in eligible.items():
            expected = sum(self.league_rate(o.down, o.distance) for o in v) / len(v)
            share = (sum(o.is_pass for o in v) + 1.0) / (len(v) + 2.0)
            offsets[k] = _logit(share) - _logit(expected)
        ordered = sorted(offsets.values())
        centres = [ordered[int((i + 0.5) * len(ordered) / self.n_styles)] for i in range(self.n_styles)]
        groups: list[list[str]] = []
        for _ in range(50):
            groups = [[] for _ in centres]
            for k in sorted(offsets):
                groups[min(range(len(centres)), key=lambda i: abs(offsets[k] - centres[i]))].append(k)
            new = [sum(offsets[k] for k in g) / len(g) if g else centres[i] for i, g in enumerate(groups)]
            if new == centres:
                break
            centres = new
        styles = []
        for centre, members in zip(centres, groups):
            if not members:
                continue
            member_obs = [o for k in members for o in eligible[k]]
            rates = self._table(
                member_obs,
                lambda key, c=centre: _sigmoid(_logit(self.league.get(key, self.overall)) + c),
                self.style_strength,
            )
            styles.append({"offset": centre, "weight": len(members) / len(offsets),
                           "opponents": len(members), "plays": len(member_obs), "rates": rates})
        return styles


def estimate_run_pass(
    prior: RunPassPrior,
    history: Iterable[RunPassObservation],
    down: Any,
    distance: Any,
    *,
    use_styles: bool = True,
    use_individual: bool = True,
    handoff: float = PRIOR_HANDOFF,
) -> dict[str, Any]:
    """P(pass) on the next snap for one opponent.

    ``history`` must hold only his plays observed before this snap, each once.
    His plays first update which style he resembles; the individual offset is
    then the shrunk residual against that updated style, so evidence the style
    already explains is not added a second time. With no history the answer is
    the league (or style-average) prior and ``confidence`` is 0.
    """
    hist = list(history)
    n = len(hist)
    styles = prior.styles if use_styles else []
    weights: list[float] = []
    if styles:
        logw = []
        for s in styles:
            ll = math.log(s["weight"])
            for o in hist:
                p = min(1 - 1e-4, max(1e-4, prior.style_rate(s, o.down, o.distance)))
                ll += math.log(p if o.is_pass else 1 - p)
            logw.append(ll)
        top = max(logw)
        raw = [math.exp(x - top) for x in logw]
        weights = [x / sum(raw) for x in raw]

        def base(d: Any, dist: Any) -> float:
            return sum(w * prior.style_rate(s, d, dist) for w, s in zip(weights, styles))

        source = "style"
    else:
        base = prior.league_rate
        source = "league"
    p0 = base(down, distance)
    delta = 0.0
    if use_individual and n:
        expected = sum(base(o.down, o.distance) for o in hist) / n
        shrunk = (sum(o.is_pass for o in hist) + handoff * expected) / (n + handoff)
        delta = _logit(shrunk) - _logit(expected)
        source += "+individual"
    return {
        "p_pass": _sigmoid(_logit(p0) + delta),
        "prior_p_pass": p0,
        "n_own": n,
        "confidence": (n / (n + handoff)) if use_individual else 0.0,
        "style_weights": [round(w, 3) for w in weights],
        "source": source,
    }
