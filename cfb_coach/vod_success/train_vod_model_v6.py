#!/usr/bin/env python3
"""VOD play-call success model v6, on the v0.24 shared feature module.

Every feature is built by ``cfb_coach.vod_success.shared_features``, the same
module the scorer and the live path (``inference.vod_candidate_features``)
use, so training and scoring cannot drift apart (gate v2.2 G6). The groups are
the v0.24 recipe's: candidate-by-situation keys (offense ``call_x_*``,
defense ``look_x_*``) built only when the training fold has at least five rows
for that key, dense class/family back-off groups, and no score, off_adj or
def_adj. Held-out rows never contribute to a key count. ``enable_cxs=False``
fits the same groups without the keyed and back-off terms.

This module does not read ``/workspace/vod_data``, does not write coach.db,
and does not change live or hybrid coaching defaults. Rows are tagged
``scored_on=unreviewed``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Small designs. Extra BLAS threads make each Newton fit slower, same as v5.
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import numpy as np

from cfb_coach.madden.model.dataset import EXCLUDED_VOD_SOURCE, vod_source_excluded
from cfb_coach.vod_success import shared_features as sf

SCRIPT_VERSION = "train_vod_model.v6"
CONF_MIN = sf.CONF_MIN
MIN_CXS_ROWS = sf.MIN_KEYED_ROWS
N_BINS = 10
EPS = 1e-6
N_BOOT = 1000
EXCLUDED_VODS = frozenset({EXCLUDED_VOD_SOURCE})
GAMES = ("madden27", "cfb27")
TRAIN_TYPES = ("cpu", "human")
SIDES = ("offense", "defense")
SIT_GROUPS = sf.SITUATION_GROUPS
CXS_GROUPS = sf.KEYED_GROUPS
PERSONNEL_LOOK_KEYS = frozenset({"nickel", "base", "dime", "empty"})
LAMBDA_CXS_GRID = sf.LAMBDA_KEYED_GRID
DEFAULT_LAMBDA_CXS = 256.0
DEFAULT_HP = (2.0, 16.0, 64.0)
DEFAULT_HP_GRID = tuple(
    (float(ls), float(le), float(lx))
    for ls in (1.0, 4.0, 16.0)
    for le in (1.0, 4.0, 16.0, 64.0, 256.0)
    for lx in (16.0, 128.0, 1024.0)
)
VOD_DATA_BAN = "/workspace/vod_data"
PASS_RULE = (
    "v6 is better only when the 95% CI of (log_loss_v6 - log_loss_v5_refit) "
    "is entirely below 0 on both the match holdout and the player holdout"
)
KEYS_NOTE = (
    "A candidate|bucket key survives only when that training scope has at least "
    "5 rows for it. Stratum counts use every row in the stratum. Each fold "
    "count uses that fold's training rows and leaves the held-out match or "
    "player out. Held-out rows never increment a key."
)

# Feature derivation lives in shared_features. These names stay for callers.
PRESSURE = sf.PRESSURE
PICK_AXES = (
    ("down", "down", ("1", "2", "3", "4")),
    ("distance", "dist", ("short", "medium", "long", "xlong")),
    ("zone", "zone", ("own_deep", "own", "opp", "red_zone")),
)
_POOL: dict[str, Any] = {}


norm_key = sf.norm_key


def offense_success(down: Any, distance: Any, yardline: Any, kind: Any, yards: Any) -> bool | None:
    """Fallback only. Same rule as v5: 40/60/100 percent of distance, turnovers fail."""
    if kind in ("", None, "unknown"):
        return None
    if kind in ("td", "convert"):
        return True
    if kind in ("int", "fumble", "sack", "incomplete", "stop", "incomplete_or_no_gain"):
        return False
    gained = yards or 0
    if down in (None, "") or distance in (None, ""):
        return gained > 0
    need = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.0}.get(int(down), 1.0) * float(distance)
    if yardline not in (None, ""):
        need = min(need, float(100 - int(yardline)))
    return gained >= max(need, 0.5)


call_class = sf.call_class
look_family = sf.look_family
dist_bucket = sf.dist_bucket
zone_bucket = sf.zone_bucket
time_bucket = sf.two_minute_flag


def is_personnel_look(value: Any) -> bool:
    """Personnel and package names are never constant-pick look keys."""
    return norm_key(value) in PERSONNEL_LOOK_KEYS


def candidate_of(prepared: Mapping[str, Any]) -> str | None:
    """The ranked candidate: our offense call, or our defensive call (the look)."""
    if prepared.get("side") == "defense":
        return prepared.get("look")
    return prepared.get("call")


def prep(row: Mapping[str, Any]) -> dict[str, Any]:
    """Label row to model inputs. Features come only from shared_features.training_inputs."""
    conf = row.get("conf") or {}
    if "success" in row:
        success = row["success"]
        source = "label"
    else:
        success = offense_success(
            row.get("down"),
            row.get("distance"),
            row.get("yardline"),
            row.get("result_kind"),
            row.get("yards"),
        )
        source = "fallback_rule"
    inputs = sf.training_inputs(row)
    offense_call = row.get("offense_call") or {}
    family_src = inputs["family_src"]
    if family_src == "candidate":
        conf_look = float(conf.get("def_call", 0) or 0)
    elif family_src:
        conf_look = float(conf.get(family_src, 0) or 0)
    else:
        conf_look = 0.0
    prepared = {
        "snap_id": row.get("snap_id"),
        "vod": row["video_id"],
        "game": row.get("game"),
        "otype": row.get("opponent_type") or "unknown",
        "otype_src": row.get("opponent_type_source"),
        "scrimmage": (row.get("side") != "special_teams" and not row.get("is_kickoff")),
        "y": None if success is None else int(bool(success)),
        "success_src": source,
        "conf_success": float(conf.get("success", 0) or 0),
        "clean": bool(row.get("clean_pair")),
        "call_disp": offense_call.get("play") if inputs["call"] else None,
        "conf_play": float(conf.get("play", 0) or 0),
        "conf_look": conf_look,
        "quality_flags": row.get("quality_flags") or [],
    }
    prepared.update(inputs)
    prepared["side"] = row.get("side") or "unknown"
    prepared["look_src"] = family_src
    prepared["look_family_src"] = family_src
    return prepared


def match_id_of(row: Mapping[str, Any]) -> str:
    for key in ("match_id", "game_id", "video_id", "vod_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def player_id_of(row: Mapping[str, Any]) -> str:
    for key in ("player_id", "player", "creator"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def prepare_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """prep() plus the v6 holdout ids. Every prepared row is scored_on=unreviewed."""
    prepared = prep(row)
    prepared["match_id"] = match_id_of(row)
    prepared["player_id"] = player_id_of(row)
    prepared["scored_on"] = "unreviewed"
    return prepared


def row_features(
    prepared: Mapping[str, Any],
    situation_only: bool = False,
    enable_cxs: bool = False,
    allowed_cxs: Mapping[str, set[str]] | None = None,
) -> list[tuple[str, str]]:
    """Active (group, level) indicators, from shared_features.feature_levels."""
    levels = sf.feature_levels(
        prepared,
        situation_only=situation_only,
        enable_keyed=enable_cxs,
        allowed_keys=allowed_cxs if allowed_cxs is not None else {group: set() for group in CXS_GROUPS},
    )
    return list(levels.items())


def group_penalty(group: str, hp: Sequence[float], lambda_cxs: float | None = None) -> float:
    """Shared penalties. Keyed groups use lambda_cxs; back-off groups 2 x lambda_situation."""
    return sf.group_penalty(group, hp, lambda_cxs)


def allowed_cxs_keys(rows: Sequence[Mapping[str, Any]], min_rows: int = MIN_CXS_ROWS) -> dict[str, set[str]]:
    """Keys with at least ``min_rows`` rows. ``rows`` must be the training fold only."""
    return sf.allowed_keys(rows, min_rows)


def cxs_keys_survived(rows: Sequence[Mapping[str, Any]], min_rows: int = MIN_CXS_ROWS) -> dict[str, list[str]]:
    allowed = allowed_cxs_keys(rows, min_rows)
    return {group: sorted(allowed[group]) for group in CXS_GROUPS}


def _key_counts(keys: Mapping[str, Sequence[str]]) -> dict[str, int]:
    return {group: len(keys.get(group, ())) for group in CXS_GROUPS}


class GroupLR:
    """Group-penalised logistic regression. Same Newton steps as v5, plus lambda_cxs."""

    def __init__(
        self,
        hp: Sequence[float],
        situation_only: bool = False,
        enable_cxs: bool = False,
        lambda_cxs: float = DEFAULT_LAMBDA_CXS,
        min_cxs: int = MIN_CXS_ROWS,
    ) -> None:
        self.hp = tuple(float(value) for value in hp)
        self.situation_only = bool(situation_only)
        self.enable_cxs = bool(enable_cxs) and not self.situation_only
        self.lambda_cxs = float(lambda_cxs)
        self.min_cxs = int(min_cxs)
        self.vocab: dict[tuple[str, str], int] = {}
        self.w: np.ndarray | None = None
        self.allowed_cxs = {group: set() for group in CXS_GROUPS}
        self.penalty_by_group: dict[str, float] = {}

    def _features(self, prepared: Mapping[str, Any]) -> list[tuple[str, str]]:
        return row_features(
            prepared,
            situation_only=self.situation_only,
            enable_cxs=self.enable_cxs,
            allowed_cxs=self.allowed_cxs,
        )

    def _design(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.zeros((len(rows), len(self.vocab) + 1), dtype=float)
        matrix[:, 0] = 1.0
        for i, prepared in enumerate(rows):
            for feature in self._features(prepared):
                column = self.vocab.get(feature)
                if column is not None:
                    matrix[i, column + 1] = 1.0
        return matrix

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "GroupLR":
        """Fit on ``rows`` only. Call-by-situation keys are counted on this sequence."""
        if not rows:
            raise ValueError("GroupLR.fit needs at least one training row")
        if self.enable_cxs:
            self.allowed_cxs = allowed_cxs_keys(rows, self.min_cxs)
        else:
            self.allowed_cxs = {group: set() for group in CXS_GROUPS}
        feats = sorted({feature for prepared in rows for feature in self._features(prepared)})
        self.vocab = {feature: index for index, feature in enumerate(feats)}
        self.penalty_by_group = {}
        design = self._design(rows)
        target = np.array([prepared["y"] for prepared in rows], dtype=float)
        penalty = np.zeros(design.shape[1], dtype=float)
        for feature, index in self.vocab.items():
            group = feature[0]
            value = group_penalty(group, self.hp, self.lambda_cxs)
            penalty[index + 1] = value
            self.penalty_by_group[group] = value
        ybar = min(max(float(target.mean()), 0.02), 0.98)
        weights = np.zeros(design.shape[1], dtype=float)
        weights[0] = math.log(ybar / (1.0 - ybar))
        for _ in range(100):
            linear = np.clip(design @ weights, -30.0, 30.0)
            mu = 1.0 / (1.0 + np.exp(-linear))
            grad = design.T @ (mu - target) + penalty * weights
            weight = (mu * (1.0 - mu))[:, None]
            hess = (design * weight).T @ design + np.diag(penalty + 1e-9)
            step = np.linalg.solve(hess, grad)
            weights -= step
            if float(np.max(np.abs(step))) < 1e-8:
                break
        self.w = weights
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fit")
        linear = np.clip(self._design(rows) @ self.w, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-linear))

    def groups(self) -> list[str]:
        return sorted({group for group, _level in self.vocab})


def logloss(y: Sequence[float] | np.ndarray, pred: Sequence[float] | np.ndarray) -> float | None:
    target = np.asarray(y, dtype=float)
    prob = np.asarray(pred, dtype=float)
    if len(target) == 0:
        return None
    prob = np.clip(prob, EPS, 1.0 - EPS)
    return float(-np.mean(target * np.log(prob) + (1.0 - target) * np.log(1.0 - prob)))


def brier(y: Sequence[float] | np.ndarray, pred: Sequence[float] | np.ndarray) -> float | None:
    target = np.asarray(y, dtype=float)
    prob = np.asarray(pred, dtype=float)
    if len(target) == 0:
        return None
    return float(np.mean((prob - target) ** 2))


def ece(y: Sequence[float] | np.ndarray, pred: Sequence[float] | np.ndarray, n_bins: int = N_BINS) -> float | None:
    target = np.asarray(y, dtype=float)
    prob = np.asarray(pred, dtype=float)
    n = len(target)
    if n == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for bin_index in range(n_bins):
        lo, hi = edges[bin_index], edges[bin_index + 1]
        if bin_index < n_bins - 1:
            mask = (prob >= lo) & (prob < hi)
        else:
            mask = (prob >= lo) & (prob <= hi)
        count = int(mask.sum())
        if count:
            total += (count / n) * abs(float(prob[mask].mean()) - float(target[mask].mean()))
    return float(total)


def metric_block(y: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    return {"log_loss": logloss(y, pred), "brier": brier(y, pred), "ece": ece(y, pred)}


def leave_one_group_predictions(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    fit: Callable[[Sequence[Mapping[str, Any]]], GroupLR],
) -> np.ndarray:
    """Predict each row from a model fit on the other groups. The fit sees training rows only."""
    preds = np.zeros(len(rows), dtype=float)
    groups = sorted({row[group_key] for row in rows})
    for held in groups:
        train = [row for row in rows if row[group_key] != held]
        test_index = [i for i, row in enumerate(rows) if row[group_key] == held]
        if not train or not test_index:
            continue
        model = fit(train)
        preds[test_index] = model.predict([rows[i] for i in test_index])
    return preds


def select_hp(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    grid: Sequence[Sequence[float]],
    *,
    situation_only: bool = False,
    min_cxs: int = MIN_CXS_ROWS,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Inner leave-one-group CV on the v5 architecture. Fewer than 3 groups keeps DEFAULT_HP."""
    if len({row[group_key] for row in rows}) < 3:
        return DEFAULT_HP, {"note": "default (fewer than 3 groups for inner CV)", "scores": []}
    target = np.array([row["y"] for row in rows], dtype=float)
    scores = []
    for hp in grid:
        hp_t = (float(hp[0]), float(hp[1]), float(hp[2]))
        preds = leave_one_group_predictions(
            rows,
            group_key,
            lambda train, hp_t=hp_t: GroupLR(
                hp_t, situation_only=situation_only, enable_cxs=False, min_cxs=min_cxs,
            ).fit(train),
        )
        loss = logloss(target, preds)
        scores.append({"hp": list(hp_t), "inner_logloss": None if loss is None else round(loss, 6)})
    best = min(scores, key=lambda item: (item["inner_logloss"], item["hp"]))
    chosen = (float(best["hp"][0]), float(best["hp"][1]), float(best["hp"][2]))
    return chosen, {"note": None, "scores": scores, "chosen": list(chosen)}


def select_lambda_cxs(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    hp: Sequence[float],
    grid: Sequence[float] = LAMBDA_CXS_GRID,
    min_cxs: int = MIN_CXS_ROWS,
) -> tuple[float, dict[str, Any]]:
    """Pick lambda_cxs by inner leave-one-group log loss. The grid is nested inside the outer fold."""
    if len({row[group_key] for row in rows}) < 3:
        return DEFAULT_LAMBDA_CXS, {
            "note": "default (fewer than 3 groups for inner CV)",
            "scores": [],
            "chosen": DEFAULT_LAMBDA_CXS,
        }
    target = np.array([row["y"] for row in rows], dtype=float)
    scores = []
    for lam in grid:
        lam_f = float(lam)
        preds = leave_one_group_predictions(
            rows,
            group_key,
            lambda train, lam_f=lam_f: GroupLR(
                hp, enable_cxs=True, lambda_cxs=lam_f, min_cxs=min_cxs,
            ).fit(train),
        )
        loss = logloss(target, preds)
        scores.append({
            "lambda_cxs": lam_f,
            "inner_logloss": None if loss is None else round(loss, 6),
        })
    best = min(scores, key=lambda item: (item["inner_logloss"], item["lambda_cxs"]))
    return float(best["lambda_cxs"]), {"note": None, "scores": scores, "chosen": float(best["lambda_cxs"])}


def situation_hp_grid(hp_grid: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(sorted({(float(hp[0]), DEFAULT_HP[1], DEFAULT_HP[2]) for hp in hp_grid}))


def constant_pick_check(
    model: Any,
    raw_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """How often the argmax candidate changes with down, distance, or zone inside one look.

    Look and look_family come from prep() (shared_features). Candidates are
    swapped with ``shared_features.swap_candidate``, so a swapped candidate gets
    the same derived class/family the scorer would give it. Personnel tokens
    (nickel, base, dime, empty) are never look keys.
    """
    paired = [(raw, prep(raw)) for raw in raw_rows]
    by_look: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for _raw, prepared in paired:
        look = prepared.get("look")
        if not look:
            continue
        if is_personnel_look(look):
            if look not in skipped:
                skipped.append(look)
            continue
        by_look.setdefault(look, prepared)
    if candidates is None:
        candidate_keys = sorted({candidate_of(prepared) for _raw, prepared in paired if candidate_of(prepared)})
    else:
        candidate_keys = sorted({key for key in (norm_key(call) for call in candidates) if key})
    candidate_keys = [key for key in candidate_keys if not sf.is_kneel(key)]
    looks: list[dict[str, Any]] = []
    if candidate_keys and by_look:
        for look in sorted(by_look):
            template = by_look[look]
            record: dict[str, Any] = {
                "look": look,
                "look_src": template.get("look_src"),
                "look_family": template.get("look_family"),
                "look_family_src": template.get("look_family_src"),
            }
            any_change = False
            for axis_name, field, levels in PICK_AXES:
                picks = []
                for level in levels:
                    variants = []
                    for call in candidate_keys:
                        variant = sf.swap_candidate(template, call)
                        variant[field] = level
                        variants.append(variant)
                    preds = np.asarray(model.predict(variants), dtype=float)
                    picks.append(candidate_keys[int(np.argmax(preds))])
                changed = len(set(picks)) > 1
                record[f"changed_{axis_name}"] = changed
                record[f"picks_{axis_name}"] = picks
                any_change = any_change or changed
            record["changed_any"] = any_change
            looks.append(record)
    return _constant_pick_report(looks, sorted(skipped))


def _constant_pick_report(looks: Sequence[Mapping[str, Any]], skipped: Sequence[str]) -> dict[str, Any]:
    n = len(looks)

    def _count(flag: str) -> int:
        return sum(1 for row in looks if row.get(flag))

    def _rate(flag: str) -> float | None:
        if n == 0:
            return None
        return round(_count(flag) / n, 6)

    return {
        "n_looks": n,
        "skipped_personnel_looks": list(skipped),
        "n_changed_down": _count("changed_down"),
        "n_changed_distance": _count("changed_distance"),
        "n_changed_zone": _count("changed_zone"),
        "n_changed_any": _count("changed_any"),
        "rate_changed_down": _rate("changed_down"),
        "rate_changed_distance": _rate("changed_distance"),
        "rate_changed_zone": _rate("changed_zone"),
        "rate_changed_any": _rate("changed_any"),
        "looks": [dict(row) for row in looks],
    }


def _compact_pick(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in report if key != "looks"}


def _pool_call(item: Any) -> Any:
    return _POOL["fn"](item)


def map_items(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    """Ordered map. workers=1 stays in-process. More workers use forked processes."""
    batch = list(items)
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")
    if n_workers <= 1 or len(batch) <= 1:
        return [fn(item) for item in batch]
    _POOL["fn"] = fn
    try:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(n_workers, len(batch))) as pool:
            return pool.map(_pool_call, batch, chunksize=1)
    finally:
        _POOL.pop("fn", None)


def apply_nice(level: int | None) -> int | None:
    """Set this process's niceness. ``None`` leaves it unchanged. Level must be >= 0."""
    if level is None:
        return None
    nice_level = int(level)
    if nice_level < 0:
        raise ValueError("nice level must be 0 or higher")
    os.setpriority(os.PRIO_PROCESS, 0, nice_level)
    return int(os.getpriority(os.PRIO_PROCESS, 0))


def resolve_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")
    return n_workers


def row_audit_ids(row: Mapping[str, Any]) -> list[str]:
    found = []
    for key in ("audit_game_id", "game_id", "match_id"):
        value = row.get(key)
        if value not in (None, ""):
            found.append(str(value))
    return found


def exclusion_reason(row: Mapping[str, Any], audit_game_ids: Sequence[Any] = ()) -> str | None:
    """F8Hm324SMtg (any source field) and a passed-in audit game-id list."""
    if vod_source_excluded(row) or str(row.get("video_id") or "") in EXCLUDED_VODS:
        return "excluded_vod_F8Hm324SMtg"
    audit = {str(item) for item in audit_game_ids}
    for game_id in row_audit_ids(row):
        if game_id in audit:
            return "audit_game_id"
    return None


def split_excluded(
    rows: Sequence[Mapping[str, Any]],
    audit_game_ids: Sequence[Any] = (),
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    source = list(rows)
    kept: list[Mapping[str, Any]] = []
    reasons: collections.Counter[str] = collections.Counter()
    for row in source:
        reason = exclusion_reason(row, audit_game_ids)
        if reason:
            reasons[reason] += 1
            continue
        kept.append(row)
    summary = {
        "n_input": len(source),
        "n_kept": len(kept),
        "n_excluded": int(sum(reasons.values())),
        "by_reason": dict(reasons),
        "excluded_vods": sorted(EXCLUDED_VODS),
        "audit_game_ids": [str(item) for item in audit_game_ids],
    }
    return kept, summary


def ci_below_zero(ci: Sequence[float] | None) -> bool:
    """True when an ordered 95% CI lies entirely below 0."""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    lo, hi = ci
    if lo is None or hi is None:
        return False
    lo_f, hi_f = float(lo), float(hi)
    if math.isnan(lo_f) or math.isnan(hi_f):
        return False
    return lo_f <= hi_f and hi_f < 0.0


def v6_beats_v5(match_ci: Sequence[float] | None, player_ci: Sequence[float] | None) -> bool:
    """Pass rule: both holdout CIs for (v6 - v5) log loss are below 0."""
    return ci_below_zero(match_ci) and ci_below_zero(player_ci)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _ci(samples: Sequence[float]) -> list[float] | None:
    if not samples:
        return None
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return [round(float(np.percentile(arr, 2.5)), 6), round(float(np.percentile(arr, 97.5)), 6)]


def _public_metrics(block: Mapping[str, float | None], cis: Mapping[str, list[float] | None]) -> dict[str, Any]:
    return {
        "log_loss": _round(block["log_loss"]),
        "brier": _round(block["brier"]),
        "ece": _round(block["ece"]),
        "log_loss_ci95": cis.get("log_loss"),
        "brier_ci95": cis.get("brier"),
        "ece_ci95": cis.get("ece"),
    }


def fit_fold(
    prepped: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    group_key: str,
    hp_grid: Sequence[Sequence[float]],
    lambda_grid: Sequence[float],
    enable_cxs: bool,
    min_cxs_rows: int,
    held: Any,
) -> dict[str, Any]:
    """One outer fold. Min-count keys and both refits see the training rows only."""
    train_index = [i for i, row in enumerate(prepped) if row[group_key] != held]
    test_index = [i for i, row in enumerate(prepped) if row[group_key] == held]
    train = [prepped[i] for i in train_index]
    train_raw = [raw_rows[i] for i in train_index]
    test = [prepped[i] for i in test_index]
    if any(row[group_key] == held for row in train):
        raise RuntimeError("held-out rows leaked into the training fold")
    hp, _hp_detail = select_hp(train, group_key, hp_grid, situation_only=False, min_cxs=min_cxs_rows)
    if enable_cxs:
        lam, lam_detail = select_lambda_cxs(train, group_key, hp, lambda_grid, min_cxs_rows)
    else:
        lam, lam_detail = None, {"scores": [], "note": "cxs disabled"}
    sit_grid = situation_hp_grid(hp_grid)
    hp_sit, _sit_detail = select_hp(train, group_key, sit_grid, situation_only=True, min_cxs=min_cxs_rows)
    v6 = GroupLR(
        hp,
        enable_cxs=enable_cxs,
        lambda_cxs=DEFAULT_LAMBDA_CXS if lam is None else lam,
        min_cxs=min_cxs_rows,
    ).fit(train)
    v5 = GroupLR(hp, enable_cxs=False, min_cxs=min_cxs_rows).fit(train)
    situation = GroupLR(hp_sit, situation_only=True, min_cxs=min_cxs_rows).fit(train)
    if enable_cxs:
        survived = {group: sorted(v6.allowed_cxs[group]) for group in CXS_GROUPS}
        counted = cxs_keys_survived(train, min_cxs_rows)
        if survived != counted:
            raise RuntimeError("fitted call-by-situation keys do not match the training-fold counts")
    else:
        survived = {group: [] for group in CXS_GROUPS}
    candidates = sorted({candidate_of(row) for row in train if candidate_of(row)})
    pick = constant_pick_check(v6, train_raw, candidates=candidates)
    return {
        "held_out": held,
        "n_train": len(train),
        "n_test": len(test),
        "test_index": test_index,
        "hp": [float(value) for value in hp],
        "lambda_cxs": None if lam is None else float(lam),
        "lambda_cxs_inner": list(lam_detail.get("scores") or []),
        "lambda_cxs_note": lam_detail.get("note"),
        "keys_survived": survived,
        "n_keys_survived": _key_counts(survived),
        "v6_groups": v6.groups(),
        "v5_groups": v5.groups(),
        "pred_v6": [float(value) for value in v6.predict(test)],
        "pred_v5": [float(value) for value in v5.predict(test)],
        "pred_sit": [float(value) for value in situation.predict(test)],
        "constant_pick": _compact_pick(pick),
    }


def _bootstrap(
    y: np.ndarray,
    preds: Mapping[str, np.ndarray],
    clusters: Sequence[Any],
    labels: Mapping[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = sorted(set(clusters))
    index = {group: np.flatnonzero(np.asarray(clusters) == group) for group in groups}
    names = list(preds)
    samples = {name: {"log_loss": [], "brier": [], "ece": []} for name in names}
    diffs: list[float] = []
    breakdown_samples: dict[str, dict[Any, dict[str, dict[str, list[float]]]]] = {}
    for axis, lab in labels.items():
        breakdown_samples[axis] = {
            level: {name: {"log_loss": [], "brier": [], "ece": []} for name in names}
            for level in sorted(set(lab.tolist()), key=str)
        }
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), size=len(groups))
        ii = np.concatenate([index[groups[int(k)]] for k in draw])
        losses = {}
        for name in names:
            block = metric_block(y[ii], preds[name][ii])
            for metric, value in block.items():
                if value is not None:
                    samples[name][metric].append(value)
            losses[name] = block["log_loss"]
        if losses.get("v6") is not None and losses.get("v5_refit") is not None:
            diffs.append(float(losses["v6"]) - float(losses["v5_refit"]))
        for axis, lab in labels.items():
            drawn = lab[ii]
            for level, store in breakdown_samples[axis].items():
                mask = drawn == level
                if int(mask.sum()) == 0:
                    continue
                for name in names:
                    block = metric_block(y[ii][mask], preds[name][ii][mask])
                    for metric, value in block.items():
                        if value is not None:
                            store[name][metric].append(value)
    return {"samples": samples, "diffs": diffs, "breakdowns": breakdown_samples}


def evaluate_holdout(
    raw_rows: Sequence[Mapping[str, Any]],
    prepped: Sequence[Mapping[str, Any]],
    group_key: str,
    *,
    enable_cxs: bool,
    hp_grid: Sequence[Sequence[float]],
    lambda_grid: Sequence[float],
    min_cxs_rows: int,
    min_groups: int,
    workers: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    groups = sorted({row[group_key] for row in prepped})
    if len(groups) < min_groups:
        return {
            "status": "not_evaluated",
            "group_key": group_key,
            "n_groups": len(groups),
            "reason": f"need at least {min_groups} groups to hold out by {group_key}, found {len(groups)}",
            "folds": [],
            "n_keys_survived_by_fold": [],
            "passes_against_v5": False,
            "logloss_diff_v6_minus_v5": {"estimate": None, "ci95": None},
        }
    fold_fn = partial(
        fit_fold,
        prepped,
        raw_rows,
        group_key,
        hp_grid,
        lambda_grid,
        enable_cxs,
        min_cxs_rows,
    )
    fitted = map_items(fold_fn, groups, workers)
    pred = {name: np.zeros(len(prepped), dtype=float) for name in ("v6", "v5_refit", "situation_only")}
    for fold in fitted:
        idx = fold["test_index"]
        pred["v6"][idx] = fold["pred_v6"]
        pred["v5_refit"][idx] = fold["pred_v5"]
        pred["situation_only"][idx] = fold["pred_sit"]
    y = np.array([row["y"] for row in prepped], dtype=float)
    point = {name: metric_block(y, values) for name, values in pred.items()}
    labels = {
        "down": np.array([row["down"] for row in prepped]),
        "distance": np.array([row["dist"] for row in prepped]),
        "zone": np.array([row["zone"] for row in prepped]),
    }
    clusters = [row[group_key] for row in prepped]
    boot = _bootstrap(y, pred, clusters, labels, n_boot, seed) if n_boot > 0 else {
        "samples": {name: {"log_loss": [], "brier": [], "ece": []} for name in pred},
        "diffs": [],
        "breakdowns": {},
    }
    metrics = {}
    for name in pred:
        cis = {metric: _ci(boot["samples"][name][metric]) for metric in ("log_loss", "brier", "ece")}
        metrics[name] = _public_metrics(point[name], cis)
    diff_est = None
    if point["v6"]["log_loss"] is not None and point["v5_refit"]["log_loss"] is not None:
        diff_est = _round(float(point["v6"]["log_loss"]) - float(point["v5_refit"]["log_loss"]))
    diff_ci = _ci(boot["diffs"])
    breakdowns: dict[str, dict[str, Any]] = {}
    for axis, lab in labels.items():
        breakdowns[axis] = {}
        for level in sorted(set(lab.tolist()), key=str):
            mask = lab == level
            level_metrics = {}
            for name in pred:
                block = metric_block(y[mask], pred[name][mask])
                store = boot["breakdowns"].get(axis, {}).get(level, {}).get(name, {})
                cis = {metric: _ci(store.get(metric, [])) for metric in ("log_loss", "brier", "ece")}
                level_metrics[name] = _public_metrics(block, cis)
            breakdowns[axis][str(level)] = {"n": int(mask.sum()), **level_metrics}
    public_folds = []
    for fold in fitted:
        public_folds.append({
            "held_out": fold["held_out"],
            "n_train": fold["n_train"],
            "n_test": fold["n_test"],
            "hp": fold["hp"],
            "lambda_cxs": fold["lambda_cxs"],
            "lambda_cxs_inner": fold["lambda_cxs_inner"],
            "lambda_cxs_note": fold["lambda_cxs_note"],
            "keys_survived": fold["keys_survived"],
            "n_keys_survived": fold["n_keys_survived"],
            "v6_groups": fold["v6_groups"],
            "v5_groups": fold["v5_groups"],
            "constant_pick": fold["constant_pick"],
        })
    predictions = []
    for i, row in enumerate(prepped):
        predictions.append({
            "snap_id": row.get("snap_id"),
            "match_id": row["match_id"],
            "player_id": row["player_id"],
            "y": int(row["y"]),
            "down": row["down"],
            "dist": row["dist"],
            "zone": row["zone"],
            "time": row["time"],
            "scored_family": row["look_family"],
            "family_src": row["family_src"],
            "location_missing": bool(row["location_missing"]),
            "v6": _round(pred["v6"][i]),
            "v5_refit": _round(pred["v5_refit"][i]),
            "situation_only": _round(pred["situation_only"][i]),
            "scored_on": "unreviewed",
        })
    return {
        "status": "evaluated",
        "group_key": group_key,
        "n_groups": len(groups),
        "n_boot": n_boot,
        "folds": public_folds,
        "n_keys_survived_by_fold": [
            {"held_out": fold["held_out"], **fold["n_keys_survived"]}
            for fold in public_folds
        ],
        "metrics": metrics,
        "breakdowns": breakdowns,
        "logloss_diff_v6_minus_v5": {"estimate": diff_est, "ci95": diff_ci},
        "passes_against_v5": ci_below_zero(diff_ci),
        "predictions": predictions,
    }


def _final_fit(
    raw_rows: Sequence[Mapping[str, Any]],
    prepped: Sequence[Mapping[str, Any]],
    *,
    enable_cxs: bool,
    hp_grid: Sequence[Sequence[float]],
    lambda_grid: Sequence[float],
    min_cxs_rows: int,
) -> dict[str, Any]:
    hp, _hp_detail = select_hp(prepped, "match_id", hp_grid, situation_only=False, min_cxs=min_cxs_rows)
    if enable_cxs:
        lam, lam_detail = select_lambda_cxs(prepped, "match_id", hp, lambda_grid, min_cxs_rows)
    else:
        lam, lam_detail = None, {"scores": [], "note": "cxs disabled"}
    model = GroupLR(
        hp,
        enable_cxs=enable_cxs,
        lambda_cxs=DEFAULT_LAMBDA_CXS if lam is None else lam,
        min_cxs=min_cxs_rows,
    ).fit(prepped)
    if enable_cxs:
        survived = {group: sorted(model.allowed_cxs[group]) for group in CXS_GROUPS}
    else:
        survived = {group: [] for group in CXS_GROUPS}
    candidates = sorted({candidate_of(row) for row in prepped if candidate_of(row)})
    return {
        "hp": [float(value) for value in hp],
        "lambda_cxs": None if lam is None else float(lam),
        "lambda_cxs_inner": list(lam_detail.get("scores") or []),
        "lambda_cxs_note": lam_detail.get("note"),
        "keys_survived": survived,
        "n_keys_survived": _key_counts(survived),
        "v6_groups": model.groups(),
        "constant_pick": constant_pick_check(model, raw_rows, candidates=candidates),
        "scored_on": "unreviewed",
    }


def evaluate_stratum(
    raw_rows: Sequence[Mapping[str, Any]],
    prepped: Sequence[Mapping[str, Any]],
    *,
    enable_cxs: bool,
    hp_grid: Sequence[Sequence[float]],
    lambda_grid: Sequence[float],
    min_cxs_rows: int,
    min_groups: int,
    workers: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    final = _final_fit(
        raw_rows,
        prepped,
        enable_cxs=enable_cxs,
        hp_grid=hp_grid,
        lambda_grid=lambda_grid,
        min_cxs_rows=min_cxs_rows,
    )
    holdouts = {}
    for name, group_key, fold_seed in (
        ("match", "match_id", seed),
        ("player", "player_id", seed + 1),
    ):
        holdouts[name] = evaluate_holdout(
            raw_rows,
            prepped,
            group_key,
            enable_cxs=enable_cxs,
            hp_grid=hp_grid,
            lambda_grid=lambda_grid,
            min_cxs_rows=min_cxs_rows,
            min_groups=min_groups,
            workers=workers,
            n_boot=n_boot,
            seed=fold_seed,
        )
    match_ci = holdouts["match"]["logloss_diff_v6_minus_v5"]["ci95"]
    player_ci = holdouts["player"]["logloss_diff_v6_minus_v5"]["ci95"]
    return {
        "n_rows": len(prepped),
        "scored_on": "unreviewed",
        "cxs_enabled": bool(enable_cxs),
        "keys_survived_note": KEYS_NOTE,
        "keys_survived": final["keys_survived"],
        "n_keys_survived": final["n_keys_survived"],
        "final": {
            "hp": final["hp"],
            "lambda_cxs": final["lambda_cxs"],
            "lambda_cxs_inner": final["lambda_cxs_inner"],
            "lambda_cxs_note": final["lambda_cxs_note"],
            "v6_groups": final["v6_groups"],
            "scored_on": "unreviewed",
        },
        "final_constant_pick": final["constant_pick"],
        "holdouts": holdouts,
        "pass_rule": PASS_RULE,
        "v6_better": v6_beats_v5(match_ci, player_ci),
    }


def _stratum_key(game: Any, otype: Any, side: Any) -> str:
    return f"{game}/{otype}/{side}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def train(
    rows: Sequence[Mapping[str, Any]],
    *,
    enable_cxs: bool = True,
    audit_game_ids: Sequence[Any] = (),
    workers: int | None = 1,
    nice: int | None = None,
    hp_grid: Sequence[Sequence[float]] | None = None,
    lambda_cxs_grid: Sequence[float] | None = None,
    n_boot: int = N_BOOT,
    min_cxs_rows: int = MIN_CXS_ROWS,
    min_groups: int = 2,
    seed: int = 7,
    defense_book: Any = None,
) -> dict[str, Any]:
    """Fit each Madden/CFB x CPU/human x offense/defense stratum on its own.

    ``defense_book`` (a ``shared_features.DefenseBook`` or playbook blob) is the
    book for the exact defense family lookup. ``None`` keeps the active book
    (the public catalog unless one was set). The report records which book.

    ``enable_cxs=False`` drops the three new groups so the same code refits the
    v5 architecture. ``audit_game_ids`` are dropped along with F8Hm324SMtg.
    """
    previous_book = sf._ACTIVE_BOOK
    if defense_book is not None:
        sf.set_defense_book(defense_book)
    try:
        return _train(
            rows, enable_cxs=enable_cxs, audit_game_ids=audit_game_ids, workers=workers, nice=nice,
            hp_grid=hp_grid, lambda_cxs_grid=lambda_cxs_grid, n_boot=n_boot, min_cxs_rows=min_cxs_rows,
            min_groups=min_groups, seed=seed,
        )
    finally:
        sf.set_defense_book(previous_book)


def _train(
    rows: Sequence[Mapping[str, Any]],
    *,
    enable_cxs: bool,
    audit_game_ids: Sequence[Any],
    workers: int | None,
    nice: int | None,
    hp_grid: Sequence[Sequence[float]] | None,
    lambda_cxs_grid: Sequence[float] | None,
    n_boot: int,
    min_cxs_rows: int,
    min_groups: int,
    seed: int,
) -> dict[str, Any]:
    nice_applied = apply_nice(nice)
    n_workers = resolve_workers(workers)
    grid = tuple(tuple(float(value) for value in hp) for hp in (hp_grid or DEFAULT_HP_GRID))
    lam_grid = tuple(float(value) for value in (lambda_cxs_grid or LAMBDA_CXS_GRID))
    kept, excluded = split_excluded(rows, audit_game_ids)
    raw_by_key: dict[str, list[Mapping[str, Any]]] = {}
    prepped_by_key: dict[str, list[dict[str, Any]]] = {}
    not_trained: dict[str, dict[str, Any]] = {}
    n_unknown_success = 0
    n_kneel_excluded = 0
    defense_book_hits = {"override": 0, "book": 0, "pattern": 0, "other": 0, "none": 0}
    for raw in kept:
        prepared = prepare_row(raw)
        if prepared.get("side") == "defense":
            defense_book_hits[prepared.get("family_path") or "none"] += 1
        if prepared["kneel"]:
            # QB Kneel is a clock rule, never a training target (Q6).
            n_kneel_excluded += 1
            continue
        if prepared["y"] is None or not prepared["scrimmage"]:
            n_unknown_success += 0 if prepared["y"] is not None else 1
            continue
        key = _stratum_key(prepared["game"], prepared["otype"], prepared["side"])
        trainable = (
            prepared["game"] in GAMES
            and prepared["otype"] in TRAIN_TYPES
            and prepared["side"] in SIDES
        )
        if not trainable:
            bucket = not_trained.setdefault(key, {"n_rows": 0, "reason": _skip_reason(prepared)})
            bucket["n_rows"] += 1
            continue
        raw_by_key.setdefault(key, []).append(raw)
        prepped_by_key.setdefault(key, []).append(prepared)
    strata = {}
    for game in GAMES:
        for otype in TRAIN_TYPES:
            for side in SIDES:
                key = _stratum_key(game, otype, side)
                if key not in prepped_by_key:
                    continue
                strata[key] = evaluate_stratum(
                    raw_by_key[key],
                    prepped_by_key[key],
                    enable_cxs=enable_cxs,
                    hp_grid=grid,
                    lambda_grid=lam_grid,
                    min_cxs_rows=min_cxs_rows,
                    min_groups=min_groups,
                    workers=n_workers,
                    n_boot=n_boot,
                    seed=seed,
                )
    report = {
        "script_version": SCRIPT_VERSION,
        "scored_on": "unreviewed",
        "training_report": {"tags": {"scored_on": "unreviewed"}},
        "pass_rule": PASS_RULE,
        "cxs_enabled": bool(enable_cxs),
        "min_cxs_rows": int(min_cxs_rows),
        "lambda_cxs_grid": list(lam_grid),
        "runtime": {"workers": n_workers, "nice": nice_applied},
        "excluded": excluded,
        "not_trained": not_trained,
        "n_unknown_success": n_unknown_success,
        "n_kneel_excluded": n_kneel_excluded,
        "feature_module": {
            "module": "cfb_coach.vod_success.shared_features",
            "version": sf.FEATURE_MODULE_VERSION,
            "recipe_sha256": sf.RECIPE_SHA256,
            "gate_sha256": sf.GATE_SHA256,
            "dropped_groups": {side: list(groups) for side, groups in sf.DROPPED_GROUPS.items()},
            "notes": list(sf.MANIFEST_NOTES),
            "defense_book": sf.active_defense_book().describe(),
            "defense_family_rows": defense_book_hits,
        },
        "strata": strata,
    }
    return _jsonable(report)


def _skip_reason(prepared: Mapping[str, Any]) -> str:
    if prepared["game"] not in GAMES:
        return "game is not madden27 or cfb27"
    if prepared["otype"] not in TRAIN_TYPES:
        return "opponent type is not cpu or human"
    if prepared["side"] not in SIDES:
        return "side is not offense or defense"
    return "not trained"


def assert_not_vod_data(path: str) -> None:
    banned = os.path.realpath(VOD_DATA_BAN)
    resolved = os.path.realpath(path)
    if resolved == banned or resolved.startswith(banned + os.sep):
        raise SystemExit("refusing to read or write /workspace/vod_data")


def parse_hp_grid(text: str | None) -> tuple[tuple[float, float, float], ...]:
    if not text:
        return DEFAULT_HP_GRID
    triples = []
    for part in text.split(";"):
        nums = [float(piece) for piece in part.split(",") if piece.strip() != ""]
        if len(nums) != 3:
            raise SystemExit(f"bad --hp-grid triple: {part}")
        triples.append((nums[0], nums[1], nums[2]))
    if not triples:
        raise SystemExit("bad --hp-grid")
    return tuple(triples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the v6 VOD success model on a JSON row file. "
            "Does not read /workspace/vod_data and does not change live coaching."
        )
    )
    parser.add_argument("--rows", required=True, help="JSON list of snap rows")
    parser.add_argument("--out", help="Write the report JSON to this path")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for outer folds")
    parser.add_argument("--nice", type=int, default=None, help="Process niceness, 0 or higher")
    parser.add_argument("--audit-game-id", action="append", default=[], help="Game id to exclude")
    parser.add_argument("--disable-cxs", action="store_true", help="Fit the v5 feature set")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument(
        "--hp-grid",
        help="v5 penalty grid as ls,le,lx triples separated by semicolons. Defaults to the v5 grid.",
    )
    parser.add_argument(
        "--defense-book",
        help=(
            "JSON defensive book for the exact family lookup (a list of play names, a "
            "{formations: {...}} side, or an active_playbook_json blob). Defaults to the repo catalog."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    assert_not_vod_data(args.rows)
    if args.out:
        assert_not_vod_data(args.out)
    rows = json.loads(Path(args.rows).read_text())
    report = train(
        rows,
        enable_cxs=not args.disable_cxs,
        audit_game_ids=args.audit_game_id,
        workers=args.workers,
        nice=args.nice,
        hp_grid=parse_hp_grid(args.hp_grid),
        n_boot=args.n_boot,
        defense_book=sf.DefenseBook.from_file(args.defense_book) if args.defense_book else None,
    )
    payload = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
