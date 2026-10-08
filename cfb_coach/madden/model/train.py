"""Offline supervised training. Owner: ML Training.

Optional learners stay inside this module so importing the package does not
require them. ``seed`` has no default: a run must pass the seed it wants recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import features as feature_mod
from cfb_coach.madden.model.adapters import situation_to_state
from cfb_coach.madden.model.schema import (
    FEATURE_SCHEMA_VERSION,
    DataHash,
    FeatureVector,
    GateResult,
    MetricRecord,
    ModelRegistryEntry,
    OpponentKind,
    Possession,
    Tri,
    candidate_in_book,
)
from cfb_coach.situation import Situation

_CLIP = 15.0
_MAX_ITER = 25
POLICY_VERSION = "madden-ml.policy.heuristic-context.1"


def train(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    out_dir: str,
) -> ModelRegistryEntry:
    """Fit candidate models and write a registry entry.

    The returned gate stays ``not_run`` or ``insufficient`` until
    :mod:`cfb_coach.madden.model.evaluate` has scored held-out rows. Do not
    fill metrics that were not computed.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")

    from cfb_coach.madden.model.dataset import SUPERVISED_ELIGIBLE, classify_eligibility

    # Default: play-specific supervised training uses verified / trusted rows only.
    supervised = []
    for r in rows:
        row = dict(r)
        elig = row.get("eligibility") or classify_eligibility(row)
        row["eligibility"] = elig
        row["supervised_eligible"] = elig in SUPERVISED_ELIGIBLE
        if row["supervised_eligible"] and row.get("label_available"):
            supervised.append(row)
    # Plumbing fallback: if nothing is verified yet, fit is marked insufficient
    # rather than silently treating recommendations as executions.
    labeled = supervised
    offense = [r for r in labeled if str(r.get("side") or "").startswith("o")]
    defense = [r for r in labeled if str(r.get("side") or "").startswith("d")]

    # Prefer the larger labeled side; fall back to all labeled rows.
    if len(offense) >= len(defense) and offense:
        side = Possession.OFFENSE
        train_rows = offense
        label_key = "success"
    elif defense:
        side = Possession.DEFENSE
        train_rows = defense
        label_key = "stop"
    else:
        side = Possession.UNKNOWN
        train_rows = labeled
        label_key = "success"

    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_hash = hashlib.sha256(
        json.dumps([dict(r) for r in train_rows], sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    if len(train_rows) < 5:
        version = f"madden-ml.baseline.{side.value}.{seed}.insufficient"
        artifact = {
            "model_type": "intercept_prior",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "side": side.value,
            "label_key": label_key,
            "intercept": 0.0,
            "weights": [],
            "feature_names": list(feature_mod.pre_snap_feature_names()),
            "n_train": len(train_rows),
            "seed": seed,
            "note": "insufficient labeled rows for supervised fit",
        }
        art_path = out / "model_artifact.json"
        art_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        now = datetime.now(timezone.utc).isoformat()
        return ModelRegistryEntry(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_version=version,
            code_version="cfb_coach.madden.model.train",
            game_version="madden27",
            data_hashes=(DataHash(label="train_rows", sha256=data_hash),),
            metrics=(
                MetricRecord(name="n_labeled", value=float(len(labeled)), split="all", n=len(labeled)),
                MetricRecord(name="n_train_side", value=float(len(train_rows)), split="train", n=len(train_rows)),
            ),
            gate=GateResult.INSUFFICIENT,
            gate_passed=False,
            seed=seed,
            trained_at=now,
            created_ts=now,
            artifact_path=str(art_path),
            artifact_sha256=hashlib.sha256(art_path.read_bytes()).hexdigest(),
            data_hash=data_hash,
            side=side,
            opponent_type_scope=OpponentKind.UNKNOWN,
        )

    # Grouped split by game_id so the same game/VOD does not leak.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        gid = str(row.get("game_id") or row.get("opponent_id") or row.get("source_path") or "unknown")
        groups.setdefault(gid, []).append(row)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    if len(group_ids) >= 2:
        cut = max(1, int(round(len(group_ids) * 0.7)))
        train_ids = set(group_ids[:cut])
        fit_rows = [r for gid in group_ids if gid in train_ids for r in groups[gid]]
    else:
        # Single group: still fit, but evaluation will mark insufficient holdout.
        train_ids = set(group_ids)
        fit_rows = list(train_rows)

    x_rows: list[list[float]] = []
    y: list[float] = []
    for row in fit_rows:
        vec = _row_vector(row)
        x_rows.append(feature_mod.dense_pair(vec))
        y.append(1.0 if str(row.get(label_key)) == Tri.TRUE.value else 0.0)

    intercept, weights = _fit_l2_logistic(x_rows, y, l2=1.0, seed=seed)
    baseline_rate = sum(y) / len(y) if y else 0.5

    version = f"madden-ml.logit.{side.value}.{seed}"
    artifact = {
        "model_type": "l2_logistic",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "side": side.value,
        "label_key": label_key,
        "intercept": intercept,
        "weights": weights,
        "feature_names": list(feature_mod.pre_snap_feature_names()),
        "dense_dim": len(weights),
        "n_train": len(fit_rows),
        "baseline_rate": baseline_rate,
        "seed": seed,
        "policy_version": POLICY_VERSION,
        "train_groups": sorted(train_ids),
        "all_groups": group_ids,
    }
    art_path = out / "model_artifact.json"
    art_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    return ModelRegistryEntry(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=version,
        code_version="cfb_coach.madden.model.train",
        game_version="madden27",
        data_hashes=(DataHash(label="train_rows", sha256=data_hash),),
        metrics=(
            MetricRecord(name="n_labeled", value=float(len(labeled)), split="all", n=len(labeled)),
            MetricRecord(name="n_fit", value=float(len(fit_rows)), split="train", n=len(fit_rows)),
            MetricRecord(name="baseline_rate", value=float(baseline_rate), split="train", n=len(fit_rows)),
        ),
        gate=GateResult.NOT_RUN,
        gate_passed=False,
        seed=seed,
        trained_at=now,
        created_ts=now,
        artifact_path=str(art_path),
        artifact_sha256=hashlib.sha256(art_path.read_bytes()).hexdigest(),
        data_hash=data_hash,
        side=side,
        opponent_type_scope=OpponentKind.UNKNOWN,
    )


def predict_proba(artifact: Mapping[str, Any], vector: FeatureVector) -> float:
    """Score one feature vector with a trained artifact."""
    model_type = str(artifact.get("model_type") or "")
    dense = feature_mod.dense_pair(vector)
    if model_type == "intercept_prior":
        return 0.5
    intercept = float(artifact.get("intercept") or 0.0)
    weights = list(artifact.get("weights") or [])
    if len(weights) != len(dense):
        # Incompatible artifact — safe neutral score.
        return 0.5
    logit = intercept
    for w, x in zip(weights, dense):
        logit += float(w) * float(x)
    return _sigmoid(logit)


def load_artifact(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model artifact must be a JSON object")
    return payload


def _row_vector(row: Mapping[str, Any]) -> FeatureVector:
    raw = str(row.get("situation_raw") or "").strip()
    down = _int(row.get("down"))
    distance = _int(row.get("distance"))
    yardline = _int(row.get("yardline"))
    if not raw and down is not None and distance is not None:
        raw = f"{down}&{distance}"
        if yardline is not None:
            raw += f" yl {yardline}"
    sit = Situation(
        raw=raw or "unknown",
        side=str(row.get("side") or "offense"),
        down=down,
        distance=distance,
        yardline=yardline,
        coverage_hint=row.get("coverage_hint") if row.get("coverage_hint") else None,
        coverage_source="last" if row.get("coverage_hint") else "none",
        concept_hint=row.get("concept_hint") if row.get("concept_hint") else None,
        concept_source="last" if row.get("concept_hint") else "none",
    )
    opp_type = OpponentKind.UNKNOWN
    oid = str(row.get("opponent_id") or "")
    if oid.startswith("cpu") or "cpu" in oid.lower():
        opp_type = OpponentKind.CPU
    state, obs = situation_to_state(
        sit,
        game_id=None if row.get("game_id") is None else str(row.get("game_id")),
        snap_id=None if row.get("snap_id") is None else str(row.get("snap_id")),
        opponent_id=None if row.get("opponent_id") is None else str(row.get("opponent_id")),
        opponent_type=opp_type,
        quarter=_int(row.get("quarter")),
        madden_version="madden27",
    )
    formation = str(row.get("recommended_formation") or row.get("formation") or "Unknown")
    play = str(row.get("recommended_play") or row.get("play") or "Unknown")
    # Training rows may name plays outside a locked book; build a confirmed
    # candidate against a one-play synthetic book so feature code can run.
    book = {formation: [play]}
    side = Possession.DEFENSE if str(row.get("side") or "").startswith("d") else Possession.OFFENSE
    cand = candidate_in_book(book, side=side, formation=formation, play=play)
    return feature_mod.vector_for(state, obs, cand, None, None, ())


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sigmoid(logit: float) -> float:
    logit = max(-_CLIP, min(_CLIP, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def _fit_l2_logistic(
    x_rows: list[list[float]],
    y: list[float],
    *,
    l2: float,
    seed: int,
) -> tuple[float, list[float]]:
    """Pure-Python L2 logistic regression (no numpy required at import).

    Prefer the existing ``vod_success.glm`` IRLS fit when numpy is available;
    otherwise fall back to a conservative gradient step with a rate prior.
    """
    del seed  # deterministic given data order; seed recorded on the entry
    n = len(x_rows)
    if n == 0:
        return 0.0, []
    p = len(x_rows[0])
    rate = sum(y) / n
    rate = min(1.0 - 1e-3, max(1e-3, rate))
    intercept0 = math.log(rate / (1.0 - rate))

    try:
        import numpy as np

        x = np.asarray(x_rows, dtype=float)
        yy = np.asarray(y, dtype=float)
        # Design matrix with intercept column.
        design = np.concatenate([np.ones((n, 1)), x], axis=1)
        weights = np.zeros(p + 1, dtype=float)
        weights[0] = intercept0
        penalty = np.zeros(p + 1, dtype=float)
        penalty[1:] = float(l2)
        for _ in range(_MAX_ITER):
            eta = np.clip(design @ weights, -_CLIP, _CLIP)
            mu = 1.0 / (1.0 + np.exp(-eta))
            var = np.clip(mu * (1.0 - mu), 1e-4, None)
            z = eta + (yy - mu) / var
            xtw = design.T * var
            a = xtw @ design
            a.flat[:: p + 2] += penalty
            b = xtw @ z
            try:
                updated = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                updated = np.linalg.lstsq(a, b, rcond=None)[0]
            if float(np.max(np.abs(updated - weights))) < 1e-6:
                weights = updated
                break
            weights = updated
        return float(weights[0]), [float(w) for w in weights[1:]]
    except Exception:  # noqa: BLE001 — keep training usable without numpy
        pass

    weights = [0.0] * (p + 1)
    weights[0] = intercept0
    step = 0.05
    for _ in range(_MAX_ITER):
        grad = [0.0] * (p + 1)
        for i in range(n):
            logit = weights[0]
            row = x_rows[i]
            for j, x in enumerate(row):
                logit += weights[j + 1] * x
            mu = _sigmoid(logit)
            err = mu - y[i]
            grad[0] += err
            for j, x in enumerate(row):
                grad[j + 1] += err * x
        grad[0] /= n
        for j in range(p):
            grad[j + 1] = grad[j + 1] / n + l2 * weights[j + 1]
        delta = 0.0
        for j in range(p + 1):
            weights[j] -= step * grad[j]
            delta = max(delta, abs(step * grad[j]))
        if delta < 1e-6:
            break
    return float(weights[0]), [float(w) for w in weights[1:]]
