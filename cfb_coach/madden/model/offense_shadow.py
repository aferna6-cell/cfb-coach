"""Shadow offense model: situation × call shrinkage. Not the live caller.

Verified offense rows only: live human sessions, live CPU sessions, and VOD
labels at v0.34 or newer. Situation features are down, distance bucket, and
field zone. Each is crossed with the call. Interaction weights take a stronger
penalty than situation weights, the same grouped-shrinkage idea as the v0.24
VOD fit, so a thin cell falls back toward the situation-only model.

Nothing here changes the coaching mode or promotes an artifact.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.learning import SUCCESS_NEED
from cfb_coach.madden.model.dataset import (
    ELIGIBILITY_TRUSTED_VOD,
    ELIGIBILITY_VERIFIED_EXECUTION,
    EXCLUDED_VOD_SOURCE,
    SUPERVISED_ELIGIBLE,
    _normalize_row,
)
from cfb_coach.madden.model.experimental_model import _field_bucket
from cfb_coach.outcome import parse_outcome

MODEL_KIND = "offense_situation_call.v1"
MIN_VOD_LABEL = (0, 34)
LAMBDA_SITUATION = 1.0
LAMBDA_INTERACTION = 16.0
SITUATION_GROUPS = ("down", "dist", "zone")
INTERACTION_GROUPS = ("down_x_call", "dist_x_call", "zone_x_call")
SOURCE_LIVE_HUMAN = "live_human"
SOURCE_LIVE_CPU = "live_cpu"
SOURCE_VOD = "vod"
SOURCES = (SOURCE_LIVE_HUMAN, SOURCE_LIVE_CPU, SOURCE_VOD)
LATENCY_BUDGET_MS = 150
_BOOTSTRAP = 400
_MAX_ITER = 12
_CLIP = 15.0

_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)")


def repo_root() -> Path:
    """Repository root derived from this file, never a machine path."""
    return Path(__file__).resolve().parents[3]


def inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root())
    except ValueError:
        return False
    return True


def distance_bucket(distance: Any) -> str:
    try:
        dist = int(distance)
    except (TypeError, ValueError):
        return ""
    if dist <= 3:
        return "short"
    if dist >= 8:
        return "long"
    return "med"


def _down_token(down: Any) -> str:
    try:
        value = int(down)
    except (TypeError, ValueError):
        return ""
    if value in (1, 2, 3, 4):
        return str(value)
    return ""


def _zone_token(yardline: Any) -> str:
    try:
        yl = int(yardline)
    except (TypeError, ValueError):
        return ""
    zone = _field_bucket(yl)
    if zone == "unknown":
        return ""
    return zone


def _call_key(row: Mapping[str, Any]) -> str:
    raw = row.get("action_play") or row.get("executed_play") or row.get("play")
    text = "" if raw is None else str(raw).strip().lower()
    return text


def _version_tuple(text: str) -> tuple[int, int] | None:
    match = _VERSION_RE.search(text or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_vod(row: Mapping[str, Any]) -> bool:
    if row.get("trusted_vod") or row.get("eligibility") == ELIGIBILITY_TRUSTED_VOD:
        return True
    blob = " ".join(
        str(row.get(key) or "")
        for key in ("provenance", "data_source", "source")
    ).lower()
    return "vod" in blob


def _vod_allowed(row: Mapping[str, Any]) -> bool:
    source = str(row.get("vod_id") or row.get("source_id") or "")
    if source == EXCLUDED_VOD_SOURCE:
        return False
    parts = [
        str(row.get("label_version") or ""),
        str(row.get("vod_label_version") or ""),
        str(row.get("provenance") or ""),
    ]
    found = None
    for text in parts:
        found = _version_tuple(text)
        if found is not None:
            break
    if found is None:
        return False
    return found >= MIN_VOD_LABEL


def _opponent_kind(row: Mapping[str, Any]) -> str:
    kind = str(row.get("opponent_type") or "").strip().lower()
    oid = str(row.get("opponent_id") or "").strip().lower()
    if kind in {"cpu", "ai", "computer"} or oid == "cpu" or oid.startswith("cpu"):
        return "cpu"
    if kind in {"human", "h2h", "user"} or oid:
        return "human"
    return ""


def prepare_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one row and keep the fields the splitter and VOD gate need."""
    kept = {
        "label_version": raw.get("label_version") or raw.get("vod_label_version"),
        "player_id": raw.get("player_id") or raw.get("player"),
        "match_id": raw.get("match_id"),
        "vod_id": raw.get("vod_id") or raw.get("source_id"),
    }
    if raw.get("eligibility") in SUPERVISED_ELIGIBLE and raw.get("action_play"):
        item = dict(raw)
    else:
        item = _normalize_row(
            dict(raw),
            provenance=str(raw.get("provenance") or "offense_shadow"),
        )
    for key, value in kept.items():
        if value not in (None, "") and not item.get(key):
            item[key] = value
    return item


def training_source(row: Mapping[str, Any]) -> str | None:
    """Which verified offense source this row belongs to, or None to leave it out."""
    if str(row.get("data_source") or "") == "session_close":
        return None
    side = str(row.get("side") or "")
    if side.startswith("d"):
        return None
    eligible = row.get("eligibility") in SUPERVISED_ELIGIBLE or row.get("supervised_eligible")
    if not eligible:
        return None
    if not _call_key(row):
        return None
    if snap_success(row) is None:
        return None
    if _is_vod(row):
        return SOURCE_VOD if _vod_allowed(row) else None
    if row.get("eligibility") != ELIGIBILITY_VERIFIED_EXECUTION and not row.get("supervised_eligible"):
        return None
    kind = _opponent_kind(row)
    if kind == "cpu":
        return SOURCE_LIVE_CPU
    if kind == "human":
        return SOURCE_LIVE_HUMAN
    return None


def snap_success(row: Mapping[str, Any]) -> bool | None:
    """40% / 60% / 100% of needed yards. A turnover is a failure."""
    result = row.get("result")
    parsed = parse_outcome(None if result is None else str(result))
    if parsed.is_turnover or parsed.kind in {"int", "fumble"}:
        return False
    if parsed.kind == "td":
        return True
    yards = row.get("yards")
    if yards is None:
        yards = parsed.yards
    down = row.get("down")
    distance = row.get("distance")
    try:
        if yards is not None and down is not None and distance is not None:
            need = SUCCESS_NEED.get(int(down))
            if need is None:
                return None
            return float(yards) >= float(need) * float(distance)
    except (TypeError, ValueError):
        return None
    label = row.get("success")
    if isinstance(label, bool):
        return label
    text = str(label or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def select_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Verified offense rows and the count of each source."""
    counts = {name: 0 for name in SOURCES}
    chosen: list[dict[str, Any]] = []
    for raw in rows:
        item = prepare_row(raw)
        source = training_source(item)
        if source is None:
            continue
        item["training_source"] = source
        item["call_key"] = _call_key(item)
        item["success_bit"] = bool(snap_success(item))
        counts[source] += 1
        chosen.append(item)
    return chosen, counts


def _features(row: Mapping[str, Any], *, situation_only: bool) -> dict[str, str]:
    down = _down_token(row.get("down"))
    dist = distance_bucket(row.get("distance"))
    zone = _zone_token(row.get("yardline"))
    feats: dict[str, str] = {}
    if down:
        feats["down"] = down
    if dist:
        feats["dist"] = dist
    if zone:
        feats["zone"] = zone
    if situation_only:
        return feats
    call = str(row.get("call_key") or _call_key(row) or "")
    if not call:
        return feats
    if down:
        feats["down_x_call"] = f"{down}|{call}"
    if dist:
        feats["dist_x_call"] = f"{dist}|{call}"
    if zone:
        feats["zone_x_call"] = f"{zone}|{call}"
    return feats


def _penalty(group: str) -> float:
    if group in INTERACTION_GROUPS:
        return LAMBDA_INTERACTION
    return LAMBDA_SITUATION


def _sigmoid(logit: float) -> float:
    logit = max(-_CLIP, min(_CLIP, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def _solve(matrix: list[list[float]], target: list[float]) -> list[float] | None:
    size = len(target)
    work = [row[:] + [target[index]] for index, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-10:
            return None
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        for col_j in range(col, size + 1):
            work[col][col_j] /= scale
        for row in range(size):
            if row == col:
                continue
            factor = work[row][col]
            if factor == 0.0:
                continue
            for col_j in range(col, size + 1):
                work[row][col_j] -= factor * work[col][col_j]
    return [work[row][size] for row in range(size)]


def _fit(rows: Sequence[Mapping[str, Any]], *, situation_only: bool) -> dict[str, Any]:
    labels = [1.0 if row.get("success_bit") else 0.0 for row in rows]
    feats = [_features(row, situation_only=situation_only) for row in rows]
    n = len(labels)
    if n == 0:
        return {"intercept": 0.0, "coef": {}, "situation_only": situation_only}
    rate = min(1.0 - 1e-3, max(1e-3, sum(labels) / n))
    columns: list[tuple[str, str, float]] = []
    index: dict[tuple[str, str], int] = {}
    for feat in feats:
        for group, level in feat.items():
            key = (group, level)
            if key not in index:
                index[key] = len(columns)
                columns.append((group, level, _penalty(group)))
    width = len(columns) + 1
    weights = [0.0] * width
    weights[0] = math.log(rate / (1.0 - rate))
    for _ in range(_MAX_ITER):
        eta = []
        mu = []
        var = []
        for feat in feats:
            logit = weights[0]
            for group, level in feat.items():
                logit += weights[index[(group, level)] + 1]
            logit = max(-_CLIP, min(_CLIP, logit))
            prob = 1.0 / (1.0 + math.exp(-logit))
            eta.append(logit)
            mu.append(prob)
            var.append(min(0.25, max(1e-4, prob * (1.0 - prob))))
        gram = [[0.0] * width for _ in range(width)]
        rhs = [0.0] * width
        for i, feat in enumerate(feats):
            design = [0.0] * width
            design[0] = 1.0
            for group, level in feat.items():
                design[index[(group, level)] + 1] = 1.0
            weight = var[i]
            zed = eta[i] + (labels[i] - mu[i]) / var[i]
            for a in range(width):
                rhs[a] += design[a] * weight * zed
                for b in range(a, width):
                    gram[a][b] += design[a] * weight * design[b]
        for a in range(width):
            for b in range(a):
                gram[a][b] = gram[b][a]
        for col, (_group, _level, lam) in enumerate(columns):
            gram[col + 1][col + 1] += lam
        updated = _solve(gram, rhs)
        if updated is None:
            break
        if max(abs(updated[j] - weights[j]) for j in range(width)) < 1e-6:
            weights = updated
            break
        weights = updated
    coef: dict[str, dict[str, float]] = {}
    for (group, level, _lam), weight in zip(columns, weights[1:]):
        coef.setdefault(group, {})[level] = float(weight)
    return {
        "intercept": float(weights[0]),
        "coef": coef,
        "situation_only": situation_only,
        "lambda_situation": LAMBDA_SITUATION,
        "lambda_interaction": None if situation_only else LAMBDA_INTERACTION,
    }


def _predict(model: Mapping[str, Any], feats: Mapping[str, str]) -> float:
    logit = float(model.get("intercept") or 0.0)
    coef = model.get("coef") or {}
    for group, level in feats.items():
        logit += float((coef.get(group) or {}).get(level) or 0.0)
    return _sigmoid(logit)


def predict_success(artifact: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    """P(success | situation, call). A call with no cell uses the situation model."""
    model = artifact.get("full_model") or {}
    return _predict(model, _features(row, situation_only=False))


def predict_situation(artifact: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    model = artifact.get("situation_model") or {}
    return _predict(model, _features(row, situation_only=True))


def fit_artifact(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit situation-only and situation × call models. Does not write a file."""
    chosen, counts = select_training_rows(rows)
    calls = sorted({str(row["call_key"]) for row in chosen})
    artifact = {
        "kind": MODEL_KIND,
        "shadow_only": True,
        "auto_promote": False,
        "heuristic_remains_default": True,
        "lambda_situation": LAMBDA_SITUATION,
        "lambda_interaction": LAMBDA_INTERACTION,
        "sources": counts,
        "n_rows": len(chosen),
        "calls": calls,
        "situation_model": _fit(chosen, situation_only=True),
        "full_model": _fit(chosen, situation_only=False),
    }
    return artifact


def default_artifact_path() -> Path:
    from cfb_coach.games import data_dir

    return Path(data_dir()) / "madden_ml_offense" / "offense.json"


def save_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    dest = Path(path)
    if dest.is_dir():
        dest = dest / "offense.json"
    if inside_repo(dest):
        raise ValueError(f"refusing to write an offense artifact inside the repo: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def load_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != MODEL_KIND:
        raise ValueError("not an offense situation×call artifact")
    return payload


def _match_id(row: Mapping[str, Any]) -> str:
    for key in ("match_id", "game_id", "session_id", "vod_id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return "match:unknown"


def _player_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("player_id")
    if value in (None, ""):
        return None
    return f"player:{value}"


def holdout_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Group row indexes so a match, and a player when present, stay together."""
    parent = list(range(len(rows)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    match_owner: dict[str, int] = {}
    player_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        match = _match_id(row)
        if match in match_owner:
            union(index, match_owner[match])
        else:
            match_owner[match] = index
        player = _player_id(row)
        if player is None:
            continue
        if player in player_owner:
            union(index, player_owner[player])
        else:
            player_owner[player] = index
    buckets: dict[int, list[int]] = {}
    for index in range(len(rows)):
        buckets.setdefault(find(index), []).append(index)
    return [buckets[key] for key in sorted(buckets)]


def _log_loss(prob: float, label: bool) -> float:
    p = min(1.0 - 1e-6, max(1e-6, prob))
    y = 1.0 if label else 0.0
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rank_calls(
    artifact: Mapping[str, Any],
    row: Mapping[str, Any],
    calls: Sequence[str],
) -> list[tuple[str, float]]:
    scored = []
    for call in calls:
        probe = dict(row)
        probe["call_key"] = call
        scored.append((call, predict_success(artifact, probe)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def _prior_probs(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    calls: Sequence[str],
) -> tuple[list[float], list[int], list[int]]:
    """Current prior-driven offense model, fit on the train fold only."""
    from cfb_coach.madden.model.experimental_model import predict_success as prior_predict
    from cfb_coach.madden.model.experimental_model import train_experimental

    try:
        prior = train_experimental(train_rows, side="offense")
    except Exception:  # noqa: BLE001 — a missing research file must not hide the eval
        from cfb_coach.madden.model.experimental_model import ExperimentalArtifact

        prior = ExperimentalArtifact(evidence_quality="prior_driven")
    probs: list[float] = []
    top1: list[int] = []
    top3: list[int] = []
    for row in test_rows:
        call = str(row.get("call_key") or "")
        formation = str(row.get("action_formation") or row.get("executed_formation") or "")
        scored = []
        for name in calls:
            pred = prior_predict(
                prior,
                formation=formation,
                play=name,
                down=row.get("down"),
                distance=row.get("distance"),
                yardline=row.get("yardline"),
            )
            scored.append((name, float(pred["probability"])))
        scored.sort(key=lambda item: (-item[1], item[0]))
        lookup = dict(scored)
        probs.append(float(lookup.get(call, scored[0][1] if scored else 0.5)))
        order = [name for name, _prob in scored]
        top1.append(1 if order and order[0] == call else 0)
        top3.append(1 if call in order[:3] else 0)
    return probs, top1, top3


def _metric_block(
    loss: Sequence[float],
    top1: Sequence[int],
    top3: Sequence[int],
    labels: Sequence[bool],
) -> dict[str, Any]:
    return {
        "log_loss": _mean(list(loss)),
        "top1": _mean([float(v) for v in top1]),
        "top3": _mean([float(v) for v in top3]),
        "success_rate": _mean([1.0 if label else 0.0 for label in labels]),
        "n": len(labels),
    }


def _bootstrap_delta(
    baseline: Sequence[float],
    model: Sequence[float],
    *,
    seed: int,
) -> dict[str, Any]:
    """Mean of baseline minus model. Positive means the model has lower loss."""
    n = len(model)
    if n == 0:
        return {"mean": None, "ci95": [None, None], "excludes_zero": False}
    point = [float(baseline[i]) - float(model[i]) for i in range(n)]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(_BOOTSTRAP):
        draw = [point[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    lo = _percentile(samples, 0.025)
    hi = _percentile(samples, 0.975)
    excludes = lo is not None and hi is not None and (lo > 0.0 or hi < 0.0)
    return {
        "mean": sum(point) / n,
        "ci95": [lo, hi],
        "excludes_zero": excludes,
        "beats": bool(lo is not None and lo > 0.0),
    }


def _situation_name(row: Mapping[str, Any]) -> str:
    down = _down_token(row.get("down")) or "?"
    dist = distance_bucket(row.get("distance")) or "?"
    zone = _zone_token(row.get("yardline")) or "?"
    return f"down {down} / {dist} / {zone}"


def evaluate_offense(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hold out by match and by player. Compare artifact, situation-only, and prior."""
    chosen, counts = select_training_rows(rows)
    full = artifact if artifact is not None else fit_artifact(chosen)
    groups = holdout_groups(chosen)
    held_rows: list[dict[str, Any]] = []
    model_loss: list[float] = []
    sit_loss: list[float] = []
    prior_loss: list[float] = []
    model_top1: list[int] = []
    sit_top1: list[int] = []
    prior_top1: list[int] = []
    model_top3: list[int] = []
    sit_top3: list[int] = []
    prior_top3: list[int] = []
    labels: list[bool] = []
    if len(groups) >= 2:
        for held in groups:
            train_idx = [i for i in range(len(chosen)) if i not in set(held)]
            if not train_idx or not held:
                continue
            train_rows = [chosen[i] for i in train_idx]
            test_rows = [chosen[i] for i in held]
            fold = fit_artifact(train_rows)
            calls = list(fold.get("calls") or [])
            for row in test_rows:
                call = str(row["call_key"])
                if call not in calls:
                    calls.append(call)
            calls = sorted(set(calls))
            prior_p, prior_t1, prior_t3 = _prior_probs(train_rows, test_rows, calls)
            for offset, row in enumerate(test_rows):
                label = bool(row["success_bit"])
                ranked = rank_calls(fold, row, calls)
                order = [name for name, _prob in ranked]
                call = str(row["call_key"])
                model_p = predict_success(fold, row)
                sit_p = predict_situation(fold, row)
                sit_ranked = sorted(calls)
                held_rows.append(row)
                labels.append(label)
                model_loss.append(_log_loss(model_p, label))
                sit_loss.append(_log_loss(sit_p, label))
                prior_loss.append(_log_loss(prior_p[offset], label))
                model_top1.append(1 if order and order[0] == call else 0)
                model_top3.append(1 if call in order[:3] else 0)
                sit_top1.append(1 if sit_ranked and sit_ranked[0] == call else 0)
                sit_top3.append(1 if call in sit_ranked[:3] else 0)
                prior_top1.append(prior_t1[offset])
                prior_top3.append(prior_t3[offset])
    metrics = {
        "artifact": _metric_block(model_loss, model_top1, model_top3, labels),
        "situation_only": _metric_block(sit_loss, sit_top1, sit_top3, labels),
        "prior_driven": _metric_block(prior_loss, prior_top1, prior_top3, labels),
    }
    vs_situation = _bootstrap_delta(sit_loss, model_loss, seed=seed)
    vs_prior = _bootstrap_delta(prior_loss, model_loss, seed=seed + 1)
    latency = _latency_ms(full, chosen)
    p95 = latency.get("p95")
    under_budget = bool(p95 is not None and p95 < LATENCY_BUDGET_MS)
    beats_both = bool(vs_situation["beats"] and vs_prior["beats"])
    passed = bool(beats_both and under_budget and labels)
    if not chosen:
        reason = "zero verified offense rows"
    elif len(groups) < 2:
        reason = "holdout needs at least two matches or players"
    elif not beats_both:
        reason = "log-loss does not beat both baselines with a CI above 0"
    elif not under_budget:
        reason = f"p95 latency {p95} ms is not under {LATENCY_BUDGET_MS} ms"
    else:
        reason = "beats both baselines; CI excludes 0; p95 latency under 150 ms"
    breakdown = _breakdowns(
        held_rows, model_loss, sit_loss, prior_loss, model_top1, labels,
    )
    return {
        "kind": "offense_eval.v1",
        "shadow_only": True,
        "auto_promoted": False,
        "heuristic_remains_default": True,
        "sources": counts,
        "n_rows": len(chosen),
        "n_groups": len(groups),
        "holdout": "match_and_player" if any(_player_id(row) for row in chosen) else "match",
        "gate_metric": "log_loss",
        "metrics": metrics,
        "deltas": {
            "vs_situation_only": vs_situation,
            "vs_prior_driven": vs_prior,
        },
        "per_situation": breakdown,
        "latency_ms": latency,
        "promotion_gate": {
            "passed": passed,
            "beats_situation_only": bool(vs_situation["beats"]),
            "beats_prior_driven": bool(vs_prior["beats"]),
            "ci_excludes_zero": bool(
                vs_situation["excludes_zero"] and vs_prior["excludes_zero"] and beats_both
            ),
            "p95_latency_ms": p95,
            "p95_under_150ms": under_budget,
            "auto_promoted": False,
            "reason": reason,
        },
    }


def _breakdowns(
    rows: Sequence[Mapping[str, Any]],
    model_loss: Sequence[float],
    sit_loss: Sequence[float],
    prior_loss: Sequence[float],
    top1: Sequence[int],
    labels: Sequence[bool],
) -> list[dict[str, Any]]:
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault(_situation_name(row), []).append(index)
    out = []
    for name in sorted(buckets):
        idx = buckets[name]
        labs = [labels[i] for i in idx]
        out.append({
            "situation": name,
            "n": len(idx),
            "success_rate": _mean([1.0 if lab else 0.0 for lab in labs]),
            "artifact_log_loss": _mean([model_loss[i] for i in idx]),
            "situation_only_log_loss": _mean([sit_loss[i] for i in idx]),
            "prior_driven_log_loss": _mean([prior_loss[i] for i in idx]),
            "artifact_top1": _mean([float(top1[i]) for i in idx]),
        })
    return out


def _latency_ms(artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = list(artifact.get("calls") or [])
    if not calls and rows:
        calls = sorted({str(row.get("call_key") or "") for row in rows if row.get("call_key")})
    sample = list(rows[:40]) or [{}]
    if not calls:
        return {"n": 0, "p50": None, "p95": None}
    times: list[float] = []
    for row in sample:
        started = time.perf_counter()
        rank_calls(artifact, row, calls)
        times.append((time.perf_counter() - started) * 1000.0)
    return {
        "n": len(times),
        "p50": _percentile(times, 0.50),
        "p95": _percentile(times, 0.95),
        "budget_ms": LATENCY_BUDGET_MS,
    }


def render_eval_md(report: Mapping[str, Any]) -> str:
    gate = report.get("promotion_gate") or {}
    lines = [
        "# Offense evaluation",
        "",
        "Shadow only. The heuristic stays the default. This report does not promote a model.",
        "",
        f"Auto-promoted: {report.get('auto_promoted')}",
        f"Holdout: {report.get('holdout')} ({report.get('n_groups')} groups, {report.get('n_rows')} rows)",
        "",
        "## Promotion gate",
        "",
        f"Passed: **{gate.get('passed')}**",
        "",
        str(gate.get("reason") or ""),
        "",
        "## Metrics",
        "",
        "| model | log-loss | top-1 | top-3 | success rate | n |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    metrics = report.get("metrics") or {}
    for name in ("artifact", "situation_only", "prior_driven"):
        block = metrics.get(name) or {}
        lines.append(
            "| {name} | {ll} | {t1} | {t3} | {sr} | {n} |".format(
                name=name,
                ll=_fmt(block.get("log_loss")),
                t1=_fmt(block.get("top1")),
                t3=_fmt(block.get("top3")),
                sr=_fmt(block.get("success_rate")),
                n=block.get("n"),
            )
        )
    lines.extend(["", "## Log-loss deltas (baseline − artifact)", ""])
    deltas = report.get("deltas") or {}
    for name, block in deltas.items():
        ci = (block or {}).get("ci95") or [None, None]
        lines.append(
            f"- {name}: mean {_fmt((block or {}).get('mean'))}, "
            f"95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}], "
            f"excludes 0: {(block or {}).get('excludes_zero')}"
        )
    latency = report.get("latency_ms") or {}
    lines.extend([
        "",
        "## Latency",
        "",
        f"p50 {_fmt(latency.get('p50'))} ms, p95 {_fmt(latency.get('p95'))} ms "
        f"(budget {LATENCY_BUDGET_MS} ms).",
        "",
        "## Per situation",
        "",
        "| situation | n | success | artifact log-loss | situation-only | prior-driven | top-1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in report.get("per_situation") or []:
        lines.append(
            "| {s} | {n} | {sr} | {a} | {b} | {c} | {t} |".format(
                s=row.get("situation"),
                n=row.get("n"),
                sr=_fmt(row.get("success_rate")),
                a=_fmt(row.get("artifact_log_loss")),
                b=_fmt(row.get("situation_only_log_loss")),
                c=_fmt(row.get("prior_driven_log_loss")),
                t=_fmt(row.get("artifact_top1")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_eval_report(report: Mapping[str, Any], out: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown. ``out`` may be a directory or a .json/.md path."""
    dest = Path(out)
    if dest.suffix == ".json":
        json_path = dest
        md_path = dest.with_suffix(".md")
    elif dest.suffix == ".md":
        md_path = dest
        json_path = dest.with_suffix(".json")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        json_path = dest / "offense_eval.json"
        md_path = dest / "offense_eval.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_eval_md(report), encoding="utf-8")
    return json_path, md_path
