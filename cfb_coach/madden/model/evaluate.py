"""Held-out evaluation and the promotion gate. Owner: ML Training.

Splits must keep whole games, opponents, and VOD sources together. In-sample
fit is not a passing gate.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.schema import (
    GateResult,
    MetricRecord,
    ModelRegistryEntry,
    Tri,
)
from cfb_coach.madden.model.train import _row_vector, load_artifact, predict_proba


def evaluate(
    entry: ModelRegistryEntry,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> ModelRegistryEntry:
    """Score ``entry`` on held-out rows and return a new registry entry.

    Must not mark the gate passed from training accuracy alone.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if not entry.artifact_path:
        return _with_gate(entry, GateResult.FAILED, False, metrics=(
            MetricRecord(name="error", value=None, split="holdout", n=0),
        ), report={"error": "missing artifact_path"})

    try:
        artifact = load_artifact(entry.artifact_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _with_gate(
            entry,
            GateResult.FAILED,
            False,
            metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
            report={"error": str(exc)},
        )

    label_key = str(artifact.get("label_key") or "success")
    side = str(artifact.get("side") or entry.side.value)
    labeled = [
        dict(r)
        for r in rows
        if r.get("label_available")
        and (
            side == "unknown"
            or str(r.get("side") or "").startswith(side[:1])
        )
    ]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in labeled:
        gid = str(row.get("game_id") or row.get("opponent_id") or row.get("source_path") or "unknown")
        groups.setdefault(gid, []).append(row)

    train_groups = set(artifact.get("train_groups") or [])
    all_groups = list(artifact.get("all_groups") or sorted(groups))
    if train_groups:
        holdout_ids = [g for g in all_groups if g not in train_groups]
    else:
        rng = random.Random(seed)
        ids = sorted(groups)
        rng.shuffle(ids)
        if len(ids) < 2:
            holdout_ids = []
        else:
            cut = max(1, int(round(len(ids) * 0.7)))
            holdout_ids = ids[cut:]

    holdout = [r for gid in holdout_ids for r in groups.get(gid, [])]
    # Thresholds chosen before looking at metrics (conservative Franchise gate).
    MIN_HOLDOUT = 30
    MIN_LABELED = 80
    MIN_GAMES = 4
    MIN_IMPROVEMENT = 0.02  # absolute log-loss improvement vs train-only baseline

    from cfb_coach.madden.model.dataset import SUPERVISED_ELIGIBLE, classify_eligibility

    holdout = [
        r for r in holdout
        if (r.get("eligibility") or classify_eligibility(r)) in SUPERVISED_ELIGIBLE
        or r.get("supervised_eligible")
    ]
    supervised_all = [
        r for r in labeled
        if (r.get("eligibility") or classify_eligibility(r)) in SUPERVISED_ELIGIBLE
        or r.get("supervised_eligible")
    ]
    supervised_holdout = [
        r for r in holdout
        if (r.get("eligibility") or classify_eligibility(r)) in SUPERVISED_ELIGIBLE
        or r.get("supervised_eligible")
    ]
    eval_rows = supervised_holdout

    if (
        len(eval_rows) < MIN_HOLDOUT
        or len(supervised_all) < MIN_LABELED
        or len(holdout_ids) < MIN_GAMES
    ):
        report = {
            "n_labeled": len(labeled),
            "n_supervised": len(supervised_all),
            "n_holdout": len(eval_rows),
            "holdout_groups": holdout_ids,
            "min_holdout": MIN_HOLDOUT,
            "min_labeled": MIN_LABELED,
            "min_games": MIN_GAMES,
            "baseline_from": "train_artifact.baseline_rate",
            "reason": "insufficient held-out supervised rows for meaningful validation",
        }
        return _with_gate(
            entry,
            GateResult.INSUFFICIENT,
            False,
            metrics=(
                MetricRecord(name="n_labeled", value=float(len(labeled)), split="all", n=len(labeled)),
                MetricRecord(name="n_supervised", value=float(len(supervised_all)), split="all", n=len(supervised_all)),
                MetricRecord(name="n_holdout", value=float(len(eval_rows)), split="holdout", n=len(eval_rows)),
            ),
            report=report,
        )

    probs: list[float] = []
    labels: list[bool] = []
    for row in eval_rows:
        vec = _row_vector(row)
        probs.append(predict_proba(artifact, vec))
        labels.append(str(row.get(label_key)) == Tri.TRUE.value)

    ll = _log_loss(probs, labels)
    br = _brier(probs, labels)
    ece, bins = _reliability(probs, labels)
    # Baseline from TRAINING data only — never from holdout labels.
    baseline_rate = float(artifact.get("baseline_rate") or 0.5)
    baseline_probs = [baseline_rate] * len(labels)
    baseline_ll = _log_loss(baseline_probs, labels)
    baseline_br = _brier(baseline_probs, labels)

    improved = (baseline_ll - ll) >= MIN_IMPROVEMENT and br <= baseline_br + 1e-9
    # Conservative gate: never pass on recommendation-only / insufficient evidence.
    gate = GateResult.PASSED if improved and len(supervised_holdout) >= MIN_HOLDOUT else GateResult.FAILED
    if not supervised_holdout:
        gate = GateResult.INSUFFICIENT
    gate_passed = gate is GateResult.PASSED

    metrics = (
        MetricRecord(name="log_loss", value=ll, split="holdout", n=len(holdout)),
        MetricRecord(name="brier", value=br, split="holdout", n=len(holdout)),
        MetricRecord(name="ece", value=ece, split="holdout", n=len(holdout)),
        MetricRecord(name="baseline_log_loss", value=baseline_ll, split="holdout", n=len(holdout)),
        MetricRecord(name="baseline_brier", value=baseline_br, split="holdout", n=len(holdout)),
        MetricRecord(name="n_holdout", value=float(len(holdout)), split="holdout", n=len(holdout)),
        MetricRecord(name="label_rate", value=baseline_rate, split="holdout", n=len(holdout)),
    )
    report = {
        "n_labeled": len(labeled),
        "n_holdout": len(holdout),
        "holdout_groups": holdout_ids,
        "log_loss": ll,
        "brier": br,
        "ece": ece,
        "baseline_log_loss": baseline_ll,
        "baseline_brier": baseline_br,
        "improved_vs_baseline": improved,
        "calibration_bins": bins,
        "gate": gate.value,
    }
    return _with_gate(entry, gate, gate_passed, metrics=metrics, report=report)


def _with_gate(
    entry: ModelRegistryEntry,
    gate: GateResult,
    gate_passed: bool | None,
    *,
    metrics: tuple[MetricRecord, ...],
    report: Mapping[str, Any],
) -> ModelRegistryEntry:
    report_path = None
    if entry.artifact_path:
        report_path = str(Path(entry.artifact_path).with_name("eval_report.json"))
        Path(report_path).write_text(
            json.dumps(dict(report), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return ModelRegistryEntry(
        feature_schema_version=entry.feature_schema_version,
        model_version=entry.model_version,
        code_version=entry.code_version,
        game_version=entry.game_version,
        data_hashes=entry.data_hashes,
        metrics=metrics,
        gate=gate,
        gate_passed=gate_passed,
        gate_report_path=report_path,
        seed=entry.seed,
        trained_at=entry.trained_at,
        created_ts=entry.created_ts,
        artifact_path=entry.artifact_path,
        artifact_sha256=entry.artifact_sha256,
        data_hash=entry.data_hash,
        side=entry.side,
        opponent_type_scope=entry.opponent_type_scope,
        title_update=entry.title_update,
    )


def _log_loss(probs: list[float], labels: list[bool]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for prob, label in zip(probs, labels):
        p = min(1.0 - 1e-6, max(1e-6, prob))
        y = 1.0 if label else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(probs)


def _brier(probs: list[float], labels: list[bool]) -> float:
    if not probs:
        return 0.0
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probs, labels)) / len(probs)


def _reliability(
    probs: list[float], labels: list[bool], *, n_bins: int = 10
) -> tuple[float, list[dict[str, Any]]]:
    if not probs:
        return 0.0, []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for prob, label in zip(probs, labels):
        idx = min(n_bins - 1, max(0, int(prob * n_bins)))
        bins[idx].append((prob, label))
    rows: list[dict[str, Any]] = []
    ece = 0.0
    n = len(probs)
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        obs = sum(1.0 for _, lab in bucket if lab) / len(bucket)
        ece += (len(bucket) / n) * abs(obs - mean_pred)
        rows.append(
            {
                "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "n": len(bucket),
                "mean_pred": round(mean_pred, 4),
                "obs_rate": round(obs, 4),
            }
        )
    return ece, rows
