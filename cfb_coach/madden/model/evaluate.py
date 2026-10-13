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
    A passing offline gate does not activate hybrid mode.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if not entry.artifact_path:
        return _with_gate(entry, GateResult.FAILED, False, metrics=(
            MetricRecord(name="error", value=None, split="holdout", n=0),
        ), report={"error": "missing artifact_path"})

    try:
        artifact = load_artifact(entry.artifact_path)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return _with_gate(
            entry,
            GateResult.FAILED,
            False,
            metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
            report={"error": f"invalid_or_corrupted_artifact: {exc}"},
        )

    if not isinstance(artifact, dict) or not artifact.get("model_type"):
        return _with_gate(
            entry,
            GateResult.FAILED,
            False,
            metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
            report={"error": "invalid_or_corrupted_artifact: missing model_type"},
        )

    label_key = str(artifact.get("label_key") or "success")
    side = str(artifact.get("side") or entry.side.value)

    from cfb_coach.madden.model.dataset import SUPERVISED_ELIGIBLE, classify_eligibility

    # Reject promotion attempts that mix recommendation-only into the eval pool.
    mixed_ineligible = 0
    supervised_all: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        elig = row.get("eligibility") or classify_eligibility(row)
        row["eligibility"] = elig
        row["supervised_eligible"] = elig in SUPERVISED_ELIGIBLE
        if row.get("label_available") and not row["supervised_eligible"]:
            mixed_ineligible += 1
        if (
            row.get("label_available")
            and row["supervised_eligible"]
            and (
                side == "unknown"
                or str(row.get("side") or "").startswith(side[:1])
            )
            and row.get("action_play")
            and row.get("action_formation")
        ):
            # Trusted VOD provenance check.
            if elig == "trusted_vod" and not row.get("trusted_vod"):
                continue
            supervised_all.append(row)

    # Group by game only among eligible supervised rows (not recommendation-only).
    groups: dict[str, list[dict[str, Any]]] = {}
    snap_to_group: dict[str, str] = {}
    for row in supervised_all:
        gid = str(row.get("game_id") or row.get("source_path") or "unknown")
        # Never collapse onto opponent_id alone — that merges distinct games.
        if not row.get("game_id") and row.get("opponent_id"):
            gid = f"oppfile:{row.get('opponent_id')}:{row.get('source_path') or 'x'}"
        groups.setdefault(gid, []).append(row)
        sid = str(row.get("snap_id") or "")
        if sid:
            if sid in snap_to_group and snap_to_group[sid] != gid:
                return _with_gate(
                    entry,
                    GateResult.FAILED,
                    False,
                    metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
                    report={
                        "error": "duplicate_snap_identity_across_groups",
                        "snap_id": sid,
                        "groups": [snap_to_group[sid], gid],
                    },
                )
            snap_to_group[sid] = gid

    train_groups = set(artifact.get("train_groups") or [])
    all_groups = list(artifact.get("all_groups") or sorted(groups))
    # Restrict group lists to groups that actually have eligible rows.
    eligible_group_ids = sorted(groups)
    if train_groups:
        holdout_ids = [g for g in eligible_group_ids if g not in train_groups]
        # Detect snap leakage: same snap_id in both train artifact groups and holdout.
        train_snaps = {
            str(r.get("snap_id"))
            for g in train_groups
            for r in groups.get(g, [])
            if r.get("snap_id")
        }
    else:
        rng = random.Random(seed)
        ids = list(eligible_group_ids)
        rng.shuffle(ids)
        if len(ids) < 2:
            holdout_ids = []
            train_snaps = set()
        else:
            cut = max(1, int(round(len(ids) * 0.7)))
            holdout_ids = ids[cut:]
            train_snaps = {
                str(r.get("snap_id"))
                for g in ids[:cut]
                for r in groups.get(g, [])
                if r.get("snap_id")
            }

    holdout = [r for gid in holdout_ids for r in groups.get(gid, [])]
    leaked = [r for r in holdout if r.get("snap_id") and str(r["snap_id"]) in train_snaps]
    if leaked:
        return _with_gate(
            entry,
            GateResult.FAILED,
            False,
            metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
            report={
                "error": "duplicate_snap_identity_spanning_train_holdout",
                "n_leaked": len(leaked),
                "examples": [str(r.get("snap_id")) for r in leaked[:5]],
            },
        )

    # Thresholds chosen before looking at metrics (conservative Franchise gate).
    MIN_HOLDOUT = 30
    MIN_LABELED = 80
    MIN_GAMES = 4
    MIN_IMPROVEMENT = 0.02  # absolute log-loss improvement vs train-only baseline

    eval_rows = holdout
    # Count holdout games using only eligible evaluation groups.
    holdout_game_count = len([g for g in holdout_ids if groups.get(g)])

    cpu_holdout = [r for r in eval_rows if str(r.get("opponent_type") or "").lower() == "cpu"]
    human_holdout = [r for r in eval_rows if str(r.get("opponent_type") or "").lower() == "human"]

    if (
        len(eval_rows) < MIN_HOLDOUT
        or len(supervised_all) < MIN_LABELED
        or holdout_game_count < MIN_GAMES
    ):
        report = {
            "n_labeled_supervised": len(supervised_all),
            "n_holdout": len(eval_rows),
            "holdout_groups": holdout_ids,
            "holdout_game_count": holdout_game_count,
            "min_holdout": MIN_HOLDOUT,
            "min_labeled": MIN_LABELED,
            "min_games": MIN_GAMES,
            "baseline_from": "train_artifact.baseline_rate",
            "mixed_recommendation_only_skipped": mixed_ineligible,
            "cpu_holdout": len(cpu_holdout),
            "human_holdout": len(human_holdout),
            "reason": "insufficient held-out supervised rows for meaningful validation",
            "hybrid_activation": False,
        }
        return _with_gate(
            entry,
            GateResult.INSUFFICIENT,
            False,
            metrics=(
                MetricRecord(
                    name="n_supervised",
                    value=float(len(supervised_all)),
                    split="all",
                    n=len(supervised_all),
                ),
                MetricRecord(
                    name="n_holdout",
                    value=float(len(eval_rows)),
                    split="holdout",
                    n=len(eval_rows),
                ),
            ),
            report=report,
        )

    probs: list[float] = []
    labels: list[bool] = []
    for row in eval_rows:
        try:
            vec = _row_vector(row)
        except ValueError as exc:
            return _with_gate(
                entry,
                GateResult.FAILED,
                False,
                metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
                report={"error": f"action_attribution: {exc}"},
            )
        probs.append(predict_proba(artifact, vec))
        labels.append(str(row.get(label_key)) == Tri.TRUE.value)

    ll = _log_loss(probs, labels)
    br = _brier(probs, labels)
    ece, bins = _reliability(probs, labels)
    # Baseline from TRAINING data only — never from holdout labels.
    # Exact 0.0 is a valid empirical rate; keep it (log-loss clips probabilities).
    if "baseline_rate" not in artifact:
        return _with_gate(
            entry,
            GateResult.FAILED,
            False,
            metrics=(MetricRecord(name="error", value=None, split="holdout", n=0),),
            report={"error": "artifact missing train-only baseline_rate"},
        )
    baseline_rate = float(artifact["baseline_rate"])
    baseline_probs = [baseline_rate] * len(labels)
    baseline_ll = _log_loss(baseline_probs, labels)
    baseline_br = _brier(baseline_probs, labels)

    improved = (baseline_ll - ll) >= MIN_IMPROVEMENT and br <= baseline_br + 1e-9
    gate = GateResult.PASSED if improved and len(eval_rows) >= MIN_HOLDOUT else GateResult.FAILED
    if not eval_rows:
        gate = GateResult.INSUFFICIENT
    gate_passed = gate is GateResult.PASSED

    metrics = (
        MetricRecord(name="log_loss", value=ll, split="holdout", n=len(eval_rows)),
        MetricRecord(name="brier", value=br, split="holdout", n=len(eval_rows)),
        MetricRecord(name="ece", value=ece, split="holdout", n=len(eval_rows)),
        MetricRecord(name="baseline_log_loss", value=baseline_ll, split="holdout", n=len(eval_rows)),
        MetricRecord(name="baseline_brier", value=baseline_br, split="holdout", n=len(eval_rows)),
        MetricRecord(name="n_holdout", value=float(len(eval_rows)), split="holdout", n=len(eval_rows)),
        MetricRecord(name="label_rate", value=baseline_rate, split="train", n=len(eval_rows)),
        MetricRecord(name="cpu_holdout", value=float(len(cpu_holdout)), split="holdout", n=len(cpu_holdout)),
        MetricRecord(name="human_holdout", value=float(len(human_holdout)), split="holdout", n=len(human_holdout)),
    )
    report = {
        "n_supervised": len(supervised_all),
        "n_holdout": len(eval_rows),
        "holdout_groups": holdout_ids,
        "holdout_game_count": holdout_game_count,
        "log_loss": ll,
        "brier": br,
        "ece": ece,
        "baseline_rate": baseline_rate,
        "baseline_log_loss": baseline_ll,
        "baseline_brier": baseline_br,
        "improved_vs_baseline": improved,
        "calibration_bins": bins,
        "mixed_recommendation_only_skipped": mixed_ineligible,
        "cpu_holdout": len(cpu_holdout),
        "human_holdout": len(human_holdout),
        "gate": gate.value,
        "hybrid_activation": False,
        "note": "Offline gate pass does not authorize hybrid live selection.",
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
        # Clip after allowing an exact empirical baseline of 0.0 or 1.0.
        p = min(1.0 - 1e-6, max(1e-6, float(prob)))
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
