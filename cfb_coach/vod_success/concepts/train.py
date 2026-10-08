"""Train P(success | situation, concept) and write ``concept/c<N>/``.

Strata are ``game x opponent_type``. CPU and human are never pooled. Unknown
opponent rows are counted and left out of the fit (display-only). Coverage is
an optional concept x coverage-family term, shrunk toward the concept's
situation estimate in the beater table. Holdout is leave-one-VOD-out. Penalties
are chosen on the training VODs only.

The coach reads ``concept_beaters.json``. It does not read coefficients to
move a call.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cfb_coach.vod_success.concepts.taxonomy import nkey

SCRIPT_VERSION = "train_concept_model.v1"
GAMES = ("madden27", "cfb27")
TYPES = ("cpu", "human")
MIN_VODS_EVAL = 3
MIN_ROWS_EVAL = 40
TENTATIVE_N = 15
GRID = [(ls, lc, lp) for ls in (1.0, 4.0, 16.0, 64.0) for lc in (4.0, 16.0, 64.0, 256.0) for lp in (64.0, 512.0, 4096.0)]
GRID_X = (4.0, 16.0, 128.0, 1024.0)
DEFAULT_HP = (4.0, 16.0, 512.0)
DEFAULT_LX = 128.0
NON_CONCEPTS = ("no_call", "unmapped_name")
OUTPUTS = ["model_params.json", "metrics.json", "metrics.md", "concept_beaters.json", "manifest.json"]
PREDICTORS = (
    "concept_model", "concept_cov_model", "situation_only_model", "overall_base_rate",
    "situation_base_rate", "call_base_rate", "concept_base_rate",
)
SIT_GROUPS = ("down", "dist", "zone", "margin", "time", "offense_by")
_Z90 = 1.6448536269514722


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def distance_band(distance: int | None) -> str:
    if distance is None:
        return "unk"
    if distance <= 2:
        return "short"
    if distance <= 6:
        return "medium"
    if distance <= 10:
        return "long"
    return "xlong"


def field_zone(yardline: int | None) -> str:
    if yardline is None or yardline < 0 or yardline > 100:
        return "unk"
    if yardline < 20:
        return "own_deep"
    if yardline < 50:
        return "own"
    if yardline < 80:
        return "opp"
    return "red_zone"


def margin_band(left: Any, right: Any) -> str:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return "unk"
    gap = abs(left - right)
    if gap <= 3:
        return "0-3"
    if gap <= 8:
        return "4-8"
    if gap <= 16:
        return "9-16"
    return "17+"


def _clock_secs(clock: Any) -> int | None:
    match = re.match(r"\s*(\d{1,2}):(\d{2})", str(clock or ""))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def time_band(quarter: Any, clock: Any) -> str:
    if quarter in (None, "") or not str(quarter).isdigit():
        return "unk"
    qtr = int(quarter)
    secs = _clock_secs(clock)
    if qtr >= 5:
        return "ot"
    if qtr in (2, 4) and secs is not None and secs <= 120:
        return "late_half"
    return {1: "q1", 2: "q2", 3: "q3", 4: "q4"}.get(qtr, "unk")


def wilson_lb(successes: float, n: float, z: float = _Z90) -> float:
    if n <= 0:
        return 0.0
    phat = successes / n
    z2 = z * z
    centre = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z2 / n))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz)."""
    am = 1.0
    a0 = 1.0
    b0 = 1.0 - (a + b) * x / (a + 1.0)
    if abs(b0) < 1e-30:
        b0 = 1e-30
    d = 1.0 / b0
    c = 1.0
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        b0 = 1.0 + aa * d
        if abs(b0) < 1e-30:
            b0 = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / b0
        am *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        b0 = 1.0 + aa * d
        if abs(b0) < 1e-30:
            b0 = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / b0
        delta = d * c
        am *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return am


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - log_beta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x)
    return 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - log_beta) / b * _betacf(b, a, 1 - x)


def beta_ppf(p: float, a: float, b: float) -> float:
    """Lower tail quantile of Beta(a, b). ``p=0.05`` is the 90% lower bound."""
    if a <= 0 or b <= 0:
        return 0.0
    target = min(1.0 - 1e-12, max(1e-12, p))
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _regularized_beta(a, b, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return 0.0
    total = 0.0
    for i in range(n_bins):
        mask = (p >= i / n_bins) & (p < (i + 1) / n_bins if i < n_bins - 1 else p <= 1)
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def score_block(y: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, Any]:
    block: dict[str, Any] = {"n": int(len(y)), "obs_rate": round(float(y.mean()), 4) if len(y) else None}
    for name, pred in preds.items():
        block[name] = {
            "log_loss": round(logloss(y, pred), 4),
            "brier": round(brier(y, pred), 4),
            "ece": round(ece(y, pred), 4),
        }
    return block


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _success(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "success", "y"}:
        return 1
    if text in {"0", "false", "no", "fail", "n"}:
        return 0
    return None


def _conf(row: dict[str, Any], field: str) -> float:
    blob = row.get("conf")
    if isinstance(blob, dict) and blob.get(field) is not None:
        parsed = _float(blob.get(field))
        return parsed if parsed is not None else 0.0
    if field == "play":
        parsed = _float(row.get("conf_play"))
    else:
        parsed = _float(row.get("conf_coverage") or row.get("conf_look"))
    return 1.0 if parsed is None else parsed


def _opponent(row: dict[str, Any]) -> str:
    raw = nkey(str(row.get("opponent_type") or ""))
    if raw in {"cpu", "ai", "computer"} or raw.startswith("cpu"):
        return "cpu"
    if raw in {"human", "h2h", "user", "person"} or raw.startswith("human"):
        return "human"
    return "unknown"


def _game(row: dict[str, Any]) -> str:
    text = nkey(str(row.get("game") or ""))
    if "cfb" in text or "college" in text:
        return "cfb27"
    if text in {"", "madden", "madden27", "m27"}:
        return "madden27"
    return text or "madden27"


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """One scrimmage snap with a known result, or None when it cannot be used."""
    success = _success(row.get("success"))
    if success is None:
        return None
    side = str(row.get("side") or "unknown")
    if side == "special_teams" or row.get("is_kickoff"):
        return None
    down = _int(row.get("down"))
    if down not in (1, 2, 3, 4):
        return None
    call = ""
    offense = row.get("offense_call")
    if isinstance(offense, dict) and offense.get("play"):
        call = str(offense["play"])
    else:
        call = str(row.get("play") or row.get("call") or row.get("our_call") or "")
    if _conf(row, "play") < 0.5:
        call = ""
    looks: list[tuple[str, str]] = []
    blob = row.get("defense_look")
    if isinstance(blob, dict):
        for field in ("def_call", "coverage_seen", "def_overlay"):
            if blob.get(field):
                looks.append((field, str(blob[field])))
    else:
        if row.get("def_call"):
            looks.append(("def_call", str(row["def_call"])))
        coverage = row.get("coverage_seen") or row.get("coverage") or row.get("look")
        if coverage:
            looks.append(("coverage_seen", str(coverage)))
        if row.get("def_overlay"):
            looks.append(("def_overlay", str(row["def_overlay"])))
    left = row.get("score_left")
    right = row.get("score_right")
    if left is None and right is None:
        left, right = row.get("score_us"), row.get("score_them")
    return {
        "vod": str(row.get("vod") or row.get("video_id") or row.get("game_id") or row.get("session_id") or "unknown"),
        "game": _game(row),
        "otype": _opponent(row),
        "creator": str(row.get("creator") or row.get("streamer") or ""),
        "y": success,
        "side": side,
        "offense_by": {"offense": "streamer", "defense": "opponent"}.get(side, "unknown"),
        "down": str(down),
        "dist": distance_band(_int(row.get("distance"))),
        "zone": field_zone(_int(row.get("yardline"))),
        "margin": margin_band(_float(left) if _float(left) is not None else left, _float(right) if _float(right) is not None else right),
        "time": time_band(row.get("quarter"), row.get("clock")),
        "call": nkey(call) or None,
        "call_disp": call or None,
        "looks": [(field, value) for field, value in looks if _conf(row, field) >= 0.5],
        "clean": bool(call),
    }


def build_rows(rows: list[dict[str, Any]], taxonomy: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    offense = taxonomy.get("offense", {}).get("names") or {}
    defense = taxonomy.get("defense", {}).get("names") or {}
    out = []
    unknown = 0
    for raw in rows:
        snap = normalize_row(raw)
        if snap is None:
            continue
        if snap["game"] not in GAMES:
            continue
        if snap["otype"] not in TYPES:
            unknown += 1
            continue
        key = snap["call"]
        rec = (offense.get(snap["game"]) or {}).get(key) if key else None
        if not key:
            concept, family, pa, play = "no_call", "no_call", False, None
        elif not rec or not rec.get("concept"):
            concept, family, pa, play = "unmapped_name", "unmapped_name", bool(rec and rec.get("pa")), key
        else:
            concept, family, pa = rec["concept"], rec["family"], bool(rec.get("pa"))
            play = key
        cov, pressure, look_src = None, None, None
        for field, value in snap["looks"]:
            mapped = (defense.get(snap["game"]) or {}).get(nkey(value)) or {}
            if cov is None and mapped.get("coverage_family"):
                cov, look_src = mapped["coverage_family"], field
            if pressure is None and mapped.get("pressure"):
                pressure = mapped["pressure"]
        out.append({
            "vod": snap["vod"], "game": snap["game"], "otype": snap["otype"], "creator": snap["creator"],
            "y": snap["y"], "clean": snap["clean"] and cov not in (None, "unknown"),
            "side": snap["side"], "offense_by": snap["offense_by"], "down": snap["down"], "dist": snap["dist"],
            "zone": snap["zone"], "margin": snap["margin"], "time": snap["time"],
            "call": snap["call"], "call_disp": snap["call_disp"],
            "play": play, "play_disp": (rec or {}).get("canonical_name") or snap["call_disp"],
            "concept": concept, "family": family, "pa": pa,
            "cov": cov or "unknown", "pressure": pressure or "unknown", "look_src": look_src,
        })
    return out, unknown


def feats(row: dict[str, Any], kind: str) -> list[tuple[str, str]]:
    features = [(group, str(row[group])) for group in SIT_GROUPS]
    features.append(("down_x_dist", f"{row['down']}|{row['dist']}"))
    if kind == "sit":
        return features
    features.append(("family", row["family"]))
    if row["concept"] not in NON_CONCEPTS:
        features.append(("concept", row["concept"]))
    if row["pa"]:
        features.append(("pa", "yes"))
    if row["play"]:
        features.append(("play", row["play"]))
    if kind == "cov":
        features.append(("cov", row["cov"]))
        features.append(("pressure", row["pressure"]))
        if row["cov"] != "unknown" and row["concept"] not in NON_CONCEPTS:
            features.append(("concept_x_cov", f"{row['concept']}|{row['cov']}"))
    return features


def penalty(group: str, hp: tuple[float, float, float], lx: float) -> float:
    ls, lc, lp = hp
    return {
        "family": lc / 4, "concept": lc, "pa": lc, "play": lp, "down_x_dist": 2 * ls,
        "cov": 2 * ls, "pressure": 2 * ls, "concept_x_cov": lx,
    }.get(group, ls)


class LR:
    def __init__(self, kind: str, hp: tuple[float, float, float], lx: float = DEFAULT_LX) -> None:
        self.kind = kind
        self.hp = hp
        self.lx = lx
        self.vocab: dict[tuple[str, str], int] = {}
        self.w = np.zeros(1)

    def _X(self, rows: list[dict[str, Any]]) -> np.ndarray:
        x = np.zeros((len(rows), len(self.vocab) + 1))
        x[:, 0] = 1.0
        for i, row in enumerate(rows):
            for feature in feats(row, self.kind):
                j = self.vocab.get(feature)
                if j is not None:
                    x[i, j + 1] = 1.0
        return x

    def fit(self, rows: list[dict[str, Any]]) -> LR:
        columns = sorted({feature for row in rows for feature in feats(row, self.kind)})
        self.vocab = {feature: j for j, feature in enumerate(columns)}
        x = self._X(rows)
        y = np.array([row["y"] for row in rows], float)
        pen = np.array([0.0] + [penalty(group, self.hp, self.lx) for group, _level in columns])
        if len(y) == 0:
            self.w = np.zeros(x.shape[1])
            return self
        base = min(max(float(y.mean()), 0.02), 0.98)
        w = np.zeros(x.shape[1])
        w[0] = math.log(base / (1 - base))
        for _ in range(100):
            mu = 1 / (1 + np.exp(-(x @ w)))
            weight = mu * (1 - mu)
            gram = (x * weight[:, None]).T @ x + np.diag(pen + 1e-9)
            grad = x.T @ (mu - y) + pen * w
            try:
                step = np.linalg.solve(gram, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(gram, grad, rcond=None)[0]
            w = w - step
            if float(np.max(np.abs(step))) < 1e-8:
                break
        self.w = w
        return self

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        logits = np.clip(self._X(rows) @ self.w, -30, 30)
        return 1 / (1 + np.exp(-logits))

    def params(self) -> dict[str, Any]:
        inverse = {j: feature for feature, j in self.vocab.items()}
        coef: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for j in range(len(self.vocab)):
            group, level = inverse[j]
            coef[group][level] = round(float(self.w[j + 1]), 6)
        hp = {
            "lambda_situation": self.hp[0],
            "lambda_concept": self.hp[1],
            "lambda_family": self.hp[1] / 4,
            "lambda_play": self.hp[2],
        }
        if self.kind == "cov":
            hp["lambda_concept_x_cov"] = self.lx
        return {
            "kind": self.kind,
            "intercept": round(float(self.w[0]), 6),
            "coef": {group: dict(sorted(levels.items())) for group, levels in sorted(coef.items())},
            "penalties": hp,
        }


def lovo(rows: list[dict[str, Any]], kind: str, hp: tuple[float, float, float], lx: float = DEFAULT_LX) -> np.ndarray:
    vods = sorted({row["vod"] for row in rows})
    out = np.zeros(len(rows))
    for vod in vods:
        train = [row for row in rows if row["vod"] != vod]
        test_i = [i for i, row in enumerate(rows) if row["vod"] == vod]
        if not train or not test_i:
            continue
        out[test_i] = LR(kind, hp, lx).fit(train).predict([rows[i] for i in test_i])
    return out


def select(
    rows: list[dict[str, Any]],
    kind: str,
    *,
    base_hp: tuple[float, float, float] | None = None,
    grid: list[tuple[float, float, float]] | None = None,
    grid_x: tuple[float, ...] | None = None,
) -> tuple[tuple[tuple[float, float, float], float], Any]:
    if len({row["vod"] for row in rows}) < 3:
        return (DEFAULT_HP, DEFAULT_LX), "default (fewer than 3 VODs for inner CV)"
    y = np.array([row["y"] for row in rows], float)
    if kind == "base":
        cands = [(hp, DEFAULT_LX) for hp in (grid or GRID)]
    elif kind == "sit":
        lambs = sorted({hp[0] for hp in grid}) if grid else (1.0, 4.0, 16.0, 64.0)
        cands = [((ls, DEFAULT_HP[1], DEFAULT_HP[2]), DEFAULT_LX) for ls in lambs]
    else:
        cands = [(base_hp or DEFAULT_HP, lx) for lx in (grid_x or GRID_X)]
    scores = [round(logloss(y, lovo(rows, kind, hp, lx)), 6) for hp, lx in cands]
    best = min(range(len(cands)), key=lambda i: scores[i])
    table = [
        {"hp": list(hp), "lambda_concept_x_cov": lx, "inner_logloss": score}
        for score, (hp, lx) in zip(scores, cands)
    ]
    return cands[best], table


class Base:
    def fit(self, rows: list[dict[str, Any]]) -> Base:
        y = [row["y"] for row in rows]
        self.pa = min(max(sum(y) / len(y), 0.01), 0.99) if y else 0.5
        buckets = {name: collections.defaultdict(lambda: [0, 0]) for name in ("down", "sit", "call", "concept")}
        for row in rows:
            for name, key in (
                ("down", row["down"]),
                ("sit", (row["down"], row["dist"])),
                ("call", row["call"]),
                ("concept", row["concept"]),
            ):
                if key is None:
                    continue
                buckets[name][key][0] += row["y"]
                buckets[name][key][1] += 1

        def shrink(successes: float, n: float, prior: float) -> float:
            return (successes + 2 * prior) / (n + 2)

        self.down = {key: shrink(s, n, self.pa) for key, (s, n) in buckets["down"].items()}
        self.sit = {key: shrink(s, n, self.down[key[0]]) for key, (s, n) in buckets["sit"].items()}
        self.call = {key: shrink(s, n, self.pa) for key, (s, n) in buckets["call"].items()}
        self.concept = {key: shrink(s, n, self.pa) for key, (s, n) in buckets["concept"].items()}
        return self

    def predict(self, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        return {
            "overall_base_rate": np.full(len(rows), self.pa),
            "situation_base_rate": np.array([
                self.sit.get((row["down"], row["dist"]), self.down.get(row["down"], self.pa)) for row in rows
            ]),
            "call_base_rate": np.array([
                self.call.get(row["call"], self.pa) if row["call"] else self.pa for row in rows
            ]),
            "concept_base_rate": np.array([self.concept.get(row["concept"], self.pa) for row in rows]),
        }


def eb_m(cells: list[tuple[float, int, float]]) -> tuple[float, str]:
    usable = [(s, n, p0) for s, n, p0 in cells if n >= 2]
    if len(usable) < 3:
        return 10.0, "default 10 (fewer than 3 cells with n>=2)"
    total_n = sum(n for _s, n, _p0 in usable)
    excess = sum(n * ((s / n - p0) ** 2 - p0 * (1 - p0) / n) for s, n, p0 in usable) / total_n
    pbar = sum(n * p0 for _s, n, p0 in usable) / total_n
    if excess <= 1e-9:
        return 50.0, "method of moments: no excess variance -> cap 50"
    raw = pbar * (1 - pbar) / excess - 1
    return float(min(max(raw, 2.0), 50.0)), f"method of moments (raw {raw:.2f}, clipped to [2,50])"


def tier(n: int, n_vods: int, lower: float, base: float) -> str:
    if n < 8:
        return "none"
    if n < TENTATIVE_N:
        return "low" if n_vods >= 2 else "none"
    if n >= 30 and n_vods >= 5 and lower > base + 0.05:
        return "high"
    if n >= 15 and n_vods >= 3 and lower > base:
        return "med"
    return "low" if n_vods >= 2 else "none"


def tier_with_streamer_rule(label: str, n_streamers: int) -> str:
    """Med needs at least 2 streamers. High needs at least 3."""
    if label == "high" and n_streamers < 3:
        label = "med"
    if label == "med" and n_streamers < 2:
        label = "low"
    return label


def _cell(rows: list[dict[str, Any]], prior: float, strength: float, base: float, key: str, name: str | None = None) -> dict[str, Any]:
    successes = sum(row["y"] for row in rows)
    n = len(rows)
    alpha = successes + strength * prior
    beta = n - successes + strength * (1 - prior)
    shrunk = alpha / (alpha + beta)
    lower = beta_ppf(0.05, alpha, beta)
    n_vods = len({row["vod"] for row in rows})
    n_streamers = len({row["creator"] for row in rows})
    top_streamer = collections.Counter(row["creator"] for row in rows).most_common(1)[0]
    top_vod = collections.Counter(row["vod"] for row in rows).most_common(1)[0][1]
    label = tier(n, n_vods, lower, base)
    payload = {
        "key": key,
        "n": n,
        "successes": successes,
        "raw_success": round(successes / n, 4),
        "prior_mean": round(prior, 4),
        "shrunk_success": round(shrunk, 4),
        "lower_bound_90": round(lower, 4),
        "wilson_lb_90_raw": round(wilson_lb(successes, n), 4),
        "baseline": round(base, 4),
        "lift_vs_baseline": round(shrunk - base, 4),
        "n_vods": n_vods,
        "n_streamers": n_streamers,
        "top_streamer": top_streamer[0],
        "top_streamer_share": round(top_streamer[1] / n, 4),
        "top_vod_share": round(top_vod / n, 4),
        "n_streamer_offense": sum(row["offense_by"] == "streamer" for row in rows),
        "n_pa": sum(bool(row["pa"]) for row in rows),
        "tentative": n < TENTATIVE_N,
        "tier": label,
        "tier_with_streamer_rule": tier_with_streamer_rule(label, n_streamers),
    }
    if name:
        payload["name"] = name
    return payload


def _display(names: list[str | None]) -> str | None:
    counts = collections.Counter(name for name in names if name)
    nice = [name for name, _n in counts.most_common() if not name.isupper()]
    if nice:
        return nice[0]
    if counts:
        return counts.most_common(1)[0][0]
    return None


def hierarchy(
    groups: dict[str, tuple[list[dict[str, Any]], float]],
    sit_exp: dict[int, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def expected(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([sit_exp[row["_i"]] for row in rows])) if rows else 0.0

    family_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for bucket, (rows, _base) in groups.items():
        by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            if row["concept"] not in NON_CONCEPTS:
                by_family[row["family"]].append(row)
        family_rows[bucket] = by_family
    m_family, note_family = eb_m([
        (sum(row["y"] for row in members), len(members), expected(members))
        for by_family in family_rows.values() for members in by_family.values()
    ])
    family_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket, by_family in family_rows.items():
        for family, members in by_family.items():
            family_cell[(bucket, family)] = _cell(members, expected(members), m_family, groups[bucket][1], family)
    concept_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for bucket, by_family in family_rows.items():
        for family, members in by_family.items():
            by_concept: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for row in members:
                by_concept[row["concept"]].append(row)
            for concept, members_c in by_concept.items():
                concept_rows[(bucket, family, concept)] = members_c
    m_concept, note_concept = eb_m([
        (sum(row["y"] for row in members), len(members), family_cell[(bucket, family)]["shrunk_success"])
        for (bucket, family, _concept), members in concept_rows.items()
    ])
    concept_cell = {
        key: _cell(members, family_cell[(key[0], key[1])]["shrunk_success"], m_concept, groups[key[0]][1], key[2])
        for key, members in concept_rows.items()
    }
    play_rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for key, members in concept_rows.items():
        by_play: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in members:
            by_play[row["play"]].append(row)
        for play, members_p in by_play.items():
            play_rows[(*key, play)] = members_p
    m_play, note_play = eb_m([
        (sum(row["y"] for row in members), len(members), concept_cell[key[:3]]["shrunk_success"])
        for key, members in play_rows.items()
    ])
    out: dict[str, Any] = {}
    for bucket, (rows, base) in groups.items():
        families = []
        for family in sorted(family_rows[bucket], key=lambda name: -len(family_rows[bucket][name])):
            fam = dict(family_cell[(bucket, family)])
            concepts = []
            for (bkt, fam_name, concept), members in concept_rows.items():
                if bkt != bucket or fam_name != family:
                    continue
                concept_payload = dict(concept_cell[(bucket, family, concept)])
                plays = []
                for (b2, f2, c2, play), members_p in play_rows.items():
                    if (b2, f2, c2) == (bucket, family, concept):
                        plays.append(_cell(
                            members_p, concept_payload["shrunk_success"], m_play, base, play,
                            _display([row["play_disp"] for row in members_p]),
                        ))
                plays.sort(key=lambda item: (-item["shrunk_success"], -item["n"], item["key"]))
                concept_payload["plays"] = plays
                concepts.append(concept_payload)
            concepts.sort(key=lambda item: (-item["shrunk_success"], -item["n"], item["key"]))
            fam["concepts"] = concepts
            families.append(fam)
        out[bucket] = {
            "n": len(rows),
            "successes": sum(row["y"] for row in rows),
            "baseline": round(base, 4),
            "situation_expected": round(expected(rows), 4) if rows else None,
            "n_with_concept": sum(row["concept"] not in NON_CONCEPTS for row in rows),
            "n_no_call": sum(row["concept"] == "no_call" for row in rows),
            "n_unmapped_name": sum(row["concept"] == "unmapped_name" for row in rows),
            "families": families,
        }
    strengths = {
        "family": {"m": round(m_family, 3), "note": note_family},
        "concept": {"m": round(m_concept, 3), "note": note_concept},
        "play": {"m": round(m_play, 3), "note": note_play},
    }
    return out, strengths


def coverage_table(rows: list[dict[str, Any]], base_pred: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    known = [row for row in rows if row["cov"] != "unknown"]
    by_cov: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in known:
        by_cov[row["cov"]].append(row)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cov, members in by_cov.items():
        for row in members:
            if row["concept"] not in NON_CONCEPTS:
                cells.setdefault((cov, row["concept"]), []).append(row)

    def prior_of(members: list[dict[str, Any]]) -> float:
        return float(np.mean([base_pred[row["_i"]] for row in members])) if members else 0.0

    m_x, note_x = eb_m([(sum(row["y"] for row in members), len(members), prior_of(members)) for members in cells.values()])
    plays: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (cov, concept), members in cells.items():
        for row in members:
            plays.setdefault((cov, concept, row["play"]), []).append(row)
    rate = {cov: sum(row["y"] for row in members) / len(members) for cov, members in by_cov.items()}
    concept_cell = {
        key: _cell(members, prior_of(members), m_x, rate[key[0]], key[1]) for key, members in cells.items()
    }
    m_play, note_play = eb_m([
        (sum(row["y"] for row in members), len(members), concept_cell[key[:2]]["shrunk_success"])
        for key, members in plays.items()
    ])
    out = {}
    for cov in sorted(by_cov, key=lambda name: -len(by_cov[name])):
        members = by_cov[cov]
        concepts = []
        for (cov_name, concept), concept_members in cells.items():
            if cov_name != cov:
                continue
            payload = dict(concept_cell[(cov, concept)])
            payload["family"] = concept_members[0]["family"]
            payload["plays"] = sorted(
                [
                    _cell(play_members, payload["shrunk_success"], m_play, rate[cov], play, _display([row["play_disp"] for row in play_members]))
                    for (c2, concept2, play), play_members in plays.items() if (c2, concept2) == (cov, concept)
                ],
                key=lambda item: (-item["shrunk_success"], -item["n"], item["key"]),
            )
            concepts.append(payload)
        concepts.sort(key=lambda item: (-item["shrunk_success"], -item["n"], item["key"]))
        out[cov] = {
            "n": len(members),
            "successes": sum(row["y"] for row in members),
            "baseline": round(rate[cov], 4),
            "pressure_counts": dict(collections.Counter(row["pressure"] for row in members)),
            "concepts": concepts,
        }
    return out, {
        "concept_x_cov": {"m": round(m_x, 3), "note": note_x},
        "play_in_concept_x_cov": {"m": round(m_play, 3), "note": note_play},
    }


def _stamp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stamped = []
    for i, row in enumerate(rows):
        stamped.append(dict(row, _i=i))
    return stamped


def fit_stratum(
    rows: list[dict[str, Any]],
    *,
    grid: list[tuple[float, float, float]] | None,
    grid_x: tuple[float, ...] | None,
    min_vods: int,
    min_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    vods = sorted({row["vod"] for row in rows})
    status: dict[str, Any] = {"n_train_rows": len(rows), "vods": vods}
    if len(vods) >= min_vods and len(rows) >= min_rows:
        y = np.array([row["y"] for row in rows], float)
        preds = {name: np.zeros(len(rows)) for name in PREDICTORS}
        folds = []
        for vod in vods:
            train = [row for row in rows if row["vod"] != vod]
            test_i = [i for i, row in enumerate(rows) if row["vod"] == vod]
            test = [rows[i] for i in test_i]
            (hp, _), _table = select(train, "base", grid=grid)
            concept_p = LR("base", hp).fit(train).predict(test)
            (_hp, lx), _table_c = select(train, "cov", base_hp=hp, grid_x=grid_x)
            cov_p = LR("cov", hp, lx).fit(train).predict(test)
            (hs, _), _table_s = select(train, "sit", grid=grid)
            sit_p = LR("sit", hs).fit(train).predict(test)
            base_p = Base().fit(train).predict(test)
            preds["concept_model"][test_i] = concept_p
            preds["concept_cov_model"][test_i] = cov_p
            preds["situation_only_model"][test_i] = sit_p
            for name, values in base_p.items():
                preds[name][test_i] = values
            yt = y[test_i]
            folds.append({
                "held_out_vod": vod,
                "n_test": len(test_i),
                "n_train": len(train),
                "hp": list(hp),
                "lambda_concept_x_cov": lx,
                "test_obs_rate": round(float(yt.mean()), 4) if len(yt) else None,
                "concept_model_log_loss": round(logloss(yt, concept_p), 4),
                "overall_base_log_loss": round(logloss(yt, base_p["overall_base_rate"]), 4),
            })
        status["status"] = "evaluated"
        status["held_out"] = {"all_known": score_block(y, preds)}
        masks = {
            "rows_with_concept": [row["concept"] not in NON_CONCEPTS for row in rows],
            "rows_with_concept_and_coverage": [row["concept"] not in NON_CONCEPTS and row["cov"] != "unknown" for row in rows],
            "streamer_offense": [row["offense_by"] == "streamer" for row in rows],
            "clean_pairs": [row["clean"] for row in rows],
        }
        for name, mask in masks.items():
            idx = np.array(mask, bool)
            status["held_out"][name] = score_block(y[idx], {key: values[idx] for key, values in preds.items()})
        status["folds"] = folds
    else:
        status["status"] = "not_evaluated"
        status["reason"] = (
            f"too small for leave-one-VOD-out: {len(rows)} known rows, {len(vods)} VODs "
            f"(need >= {min_rows} rows and >= {min_vods} VODs)"
        )
    (hp, _), table_b = select(rows, "base", grid=grid)
    base_model = LR("base", hp).fit(rows)
    (_hp, lx), table_c = select(rows, "cov", base_hp=hp, grid_x=grid_x)
    cov_model = LR("cov", hp, lx).fit(rows)
    (hs, _), table_s = select(rows, "sit", grid=grid)
    sit_model = LR("sit", hs).fit(rows)
    params = {
        "n_train_rows": len(rows),
        "base_rate": round(sum(row["y"] for row in rows) / len(rows), 4) if rows else None,
        "concept_model": base_model.params(),
        "concept_cov_model": cov_model.params(),
        "situation_only_model": sit_model.params(),
        "hp_selection": {"concept_model": table_b, "concept_cov_model": table_c, "situation_only_model": table_s},
    }
    status["final_hp"] = {"concept_model": list(hp), "lambda_concept_x_cov": lx, "situation_only_lambda": hs[0]}
    view_rows = _stamp(rows)
    views = {"all_offense": view_rows}
    if rows and rows[0]["otype"] == "cpu":
        views["streamer_offense_vs_cpu_defense"] = [row for row in view_rows if row["offense_by"] == "streamer"]
    beaters: dict[str, Any] = {}
    for view_name, members in views.items():
        if not members:
            beaters[view_name] = {"n": 0}
            continue
        local = _stamp(members)
        sit_exp_arr = sit_model.predict(local)
        base_arr = base_model.predict(local)
        sit_exp = {row["_i"]: float(sit_exp_arr[i]) for i, row in enumerate(local)}
        rate = lambda group: sum(row["y"] for row in group) / len(group)
        levels = {
            "all": {"all": local},
            "down": collections.defaultdict(list),
            "down_dist": collections.defaultdict(list),
        }
        for row in local:
            levels["down"][f"down={row['down']}"].append(row)
            levels["down_dist"][f"down={row['down']}|dist={row['dist']}"].append(row)
        situation_out = {}
        strengths = {}
        for level, groups in levels.items():
            table, strength = hierarchy({bucket: (group, rate(group)) for bucket, group in groups.items()}, sit_exp)
            situation_out[level] = table
            strengths[level] = strength
        cov_out, cov_strength = coverage_table(local, base_arr)
        beaters[view_name] = {
            "n": len(local),
            "success_rate": round(rate(local), 4),
            "eb_strengths": {"by_situation": strengths, "by_coverage": cov_strength},
            "by_situation": situation_out,
            "by_coverage": cov_out,
        }
    return status, params, {
        "stratum_success_rate": round(sum(row["y"] for row in rows) / len(rows), 4) if rows else None,
        "views": beaters,
    }


def _counts(rows: list[dict[str, Any]], vods: list[str]) -> dict[str, Any]:
    return {
        "known_success_rows": len(rows),
        "rows_with_call": sum(row["call"] is not None for row in rows),
        "rows_with_concept": sum(row["concept"] not in NON_CONCEPTS for row in rows),
        "rows_unmapped_name": sum(row["concept"] == "unmapped_name" for row in rows),
        "rows_no_call": sum(row["concept"] == "no_call" for row in rows),
        "rows_with_coverage": sum(row["cov"] != "unknown" for row in rows),
        "rows_with_concept_and_coverage": sum(row["cov"] != "unknown" and row["concept"] not in NON_CONCEPTS for row in rows),
        "offense_by": dict(collections.Counter(row["offense_by"] for row in rows)),
        "vods": len(vods),
        "streamers": len({row["creator"] for row in rows}),
        "success_rate": round(sum(row["y"] for row in rows) / len(rows), 4) if rows else None,
    }


def _markdown(meta: dict[str, Any], metrics: dict[str, Any], beaters: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "-" if value is None else f"{value:.4f}"

    lines = [
        f"# Concept model {meta['version']} (labels {meta['label_version']}, concepts {meta['concepts_version']})",
        "",
        f"Trained {meta['trained_at']} by `{SCRIPT_VERSION}`.",
        "",
        "Target P(success | situation, concept) per stratum (game x opponent_type, never pooled). "
        "Snaps without a readable call are `no_call`. Calls the taxonomy cannot map are `unmapped_name`.",
        "",
        "## Data",
        "",
        "| stratum | known rows | with concept | unmapped name | no call | with coverage | VODs | streamers | success |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, counts in metrics["counts"].items():
        if name == "unknown":
            continue
        lines.append(
            f"| {name} | {counts['known_success_rows']} | {counts['rows_with_concept']} | {counts['rows_unmapped_name']} | "
            f"{counts['rows_no_call']} | {counts['rows_with_coverage']} | {counts['vods']} | {counts['streamers']} | {fmt(counts['success_rate'])} |"
        )
    unknown = metrics["counts"].get("unknown") or {}
    if unknown:
        lines += ["", f"Unknown opponent_type: {unknown.get('rows', 0)} known-result rows, display-only, not fit.", ""]
    lines += ["", "## Held-out (leave-one-VOD-out) log loss / Brier", ""]
    for name, stratum in metrics["strata"].items():
        lines.append(f"### {name}")
        if stratum.get("status") != "evaluated":
            lines += ["", f"Not evaluated: {stratum.get('reason', stratum.get('status'))}", ""]
            continue
        for subset, block in (stratum.get("held_out") or {}).items():
            lines += [f"**{subset}**: n={block['n']}, observed {fmt(block.get('obs_rate'))}", ""]
            if not block["n"]:
                continue
            lines += ["| predictor | log loss | Brier | ECE |", "|---|---|---|---|"]
            for predictor in PREDICTORS:
                row = block[predictor]
                lines.append(f"| {predictor} | {row['log_loss']:.4f} | {row['brier']:.4f} | {row['ece']:.4f} |")
            lines.append("")
    lines += [
        "## Concept cells at tier low or above",
        "",
        "| stratum | view | bucket | concept | family | n | VODs | streamers | success | lower bound | tier | tier w/ streamer rule |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, stratum in beaters.items():
        for view_name, view in stratum["views"].items():
            for level, table in (view.get("by_situation") or {}).items():
                for bucket, body in table.items():
                    for family in body.get("families") or []:
                        for concept in family.get("concepts") or []:
                            if concept["tier"] == "none":
                                continue
                            lines.append(
                                f"| {name} | {view_name} | {bucket} | {concept['key']} | {family['key']} | {concept['n']} | "
                                f"{concept['n_vods']} | {concept['n_streamers']} | {concept['shrunk_success']:.3f} | "
                                f"{concept['lower_bound_90']:.3f} | {concept['tier']} | {concept['tier_with_streamer_rule']} |"
                            )
    lines += [
        "",
        "## Method notes",
        "",
        "- Situation: down, distance band (short <=2, medium 3-6, long 7-10, xlong 11+), field zone, absolute score-margin band, time band, offense side.",
        "- Shrinkage: play → concept → family → situation expectation. Wilson 90% lower bound is `wilson_lb_90_raw`. "
        "`lower_bound_90` is the posterior 5% quantile.",
        "- Tier: none n<8; low when n>=8 and >=2 VODs but under the med bar; med n>=15, >=3 VODs, lower bound above baseline; "
        "high n>=30, >=5 VODs, lower bound above baseline+0.05. `tier_with_streamer_rule` also needs >=2 streamers for med and >=3 for high.",
        "- Unknown opponent_type is not a stratum.",
        "",
    ]
    return "\n".join(lines)


def _next_version(previous: str | None) -> str:
    digits = re.sub(r"[^0-9]", "", previous or "")
    return f"c{int(digits or '0') + 1}"


def train_and_write(
    rows: list[dict[str, Any]],
    models_dir: str | Path,
    taxonomy: dict[str, Any],
    *,
    label_version: str,
    concepts_path: str | Path | None = None,
    label_paths: list[str | Path] | None = None,
    grid: list[tuple[float, float, float]] | None = None,
    grid_x: tuple[float, ...] | None = None,
    min_vods_eval: int = MIN_VODS_EVAL,
    min_rows_eval: int = MIN_ROWS_EVAL,
    force: bool = False,
    version: str | None = None,
    update_pointer: bool = True,
) -> dict[str, Any]:
    """Fit strata and write the concept model directory. Returns the written metadata."""
    root = Path(models_dir)
    concept_root = root / "concept"
    concept_root.mkdir(parents=True, exist_ok=True)
    concepts_bytes = json.dumps(taxonomy, sort_keys=True).encode("utf-8")
    if concepts_path is not None and Path(concepts_path).is_file():
        concepts_bytes = Path(concepts_path).read_bytes()
    label_bytes = b"".join(Path(path).read_bytes() for path in (label_paths or []) if Path(path).is_file())
    if not label_bytes:
        label_bytes = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    script_sha = sha256_file(Path(__file__).resolve())
    last_path = concept_root / "LAST_TRAINED.json"
    last = {}
    if last_path.is_file():
        try:
            last = json.loads(last_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            last = {}
    concepts_sha = sha256_bytes(concepts_bytes)
    labels_sha = sha256_bytes(label_bytes)
    latest_sha = sha256_bytes(json.dumps({"label_version": label_version, "files": [str(p) for p in (label_paths or [])]}).encode())
    same = (
        last.get("latest_sha256") == latest_sha
        and last.get("labels_data_sha256") == labels_sha
        and last.get("concepts_sha256") == concepts_sha
        and last.get("script_sha256") == script_sha
    )
    if same and not force and not version and (concept_root / str(last.get("version") or "") / "metrics.json").is_file():
        return {"unchanged": True, "version": last.get("version"), "model_dir": str(concept_root / str(last.get("version")))}
    if version:
        chosen = version
    elif last.get("version") and last.get("labels_data_sha256") == labels_sha and last.get("concepts_sha256") == concepts_sha:
        chosen = str(last["version"])
    else:
        chosen = _next_version(str(last.get("version") or ""))
    fitted, unknown = build_rows(rows, taxonomy)
    metrics: dict[str, Any] = {"strata": {}, "counts": {"unknown": {"rows": unknown, "display_only": True}}}
    params: dict[str, Any] = {}
    beaters: dict[str, Any] = {}
    by_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in fitted:
        by_key[f"{row['game']}/{row['otype']}"].append(row)
    for game in GAMES:
        for opponent in TYPES:
            key = f"{game}/{opponent}"
            stratum_rows = by_key.get(key) or []
            if not stratum_rows:
                metrics["strata"][key] = {"n_train_rows": 0, "vods": [], "status": "no data"}
                metrics["counts"][key] = _counts([], [])
                continue
            status, stratum_params, stratum_beaters = fit_stratum(
                stratum_rows, grid=grid, grid_x=grid_x, min_vods=min_vods_eval, min_rows=min_rows_eval,
            )
            metrics["strata"][key] = status
            metrics["counts"][key] = _counts(stratum_rows, status["vods"])
            params[key] = stratum_params
            beaters[key] = stratum_beaters
    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {
        "version": chosen,
        "label_version": label_version,
        "concepts_version": taxonomy.get("version"),
        "trained_at": trained_at,
    }
    metrics.update(meta)
    metrics["script_version"] = SCRIPT_VERSION
    metrics["settings"] = {
        "grid": [list(item) for item in (grid or GRID)],
        "grid_lambda_concept_x_cov": list(grid_x or GRID_X),
        "default_hp": list(DEFAULT_HP),
        "min_vods_eval": min_vods_eval,
        "min_rows_eval": min_rows_eval,
        "tentative_n": TENTATIVE_N,
        "split": "outer leave-one-VOD-out; penalties by inner leave-one-VOD-out on training VODs only",
        "unknown_opponent_type": "display-only; not fit",
    }
    situation_bands = {
        "dist": "short<=2, medium 3-6, long 7-10, xlong 11+",
        "zone": "own_deep <20, own <50, opp <80, red_zone",
        "margin": "abs(score_left-score_right): 0-3, 4-8, 9-16, 17+, unk",
        "time": "q1..q4, late_half (Q2/Q4 <=2:00), ot, unk",
        "offense_by": "streamer (side=offense) | opponent (side=defense) | unknown",
    }
    tmp = concept_root / f".{chosen}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "model_params.json").write_text(json.dumps({
        **meta, "script_version": SCRIPT_VERSION, "situation_bands": situation_bands, "strata": params,
    }, indent=1), encoding="utf-8")
    (tmp / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (tmp / "concept_beaters.json").write_text(json.dumps({
        **meta, "schema": "concept_beaters.v1", "tentative_n": TENTATIVE_N, "strata": beaters,
    }, indent=1), encoding="utf-8")
    (tmp / "metrics.md").write_text(_markdown(meta, metrics, beaters), encoding="utf-8")
    manifest = {
        **meta,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha,
        "labels_dir": str(Path(label_paths[0]).parent) if label_paths else None,
        "latest_json_sha256": latest_sha,
        "labels_data_sha256": labels_sha,
        "label_files": [Path(path).name for path in (label_paths or [])],
        "concepts_path": str(concepts_path) if concepts_path else None,
        "concepts_sha256": concepts_sha,
        "rows_loaded": len(rows),
        "rows_used": len(fitted),
        "unknown_opponent_rows": unknown,
        "outputs": list(OUTPUTS),
        "rerun": "python3 -m cfb_coach.vod_success concepts train --force",
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    dest = concept_root / chosen
    if dest.exists():
        shutil.rmtree(dest)
    os.rename(tmp, dest)
    pointer = {
        "version": chosen,
        "label_version": label_version,
        "concepts_version": taxonomy.get("version"),
        "latest_sha256": latest_sha,
        "labels_data_sha256": labels_sha,
        "concepts_sha256": concepts_sha,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha,
        "model_dir": str(dest),
        "trained_at": trained_at,
    }
    if update_pointer:
        (concept_root / "LAST_TRAINED.json.tmp").write_text(json.dumps(pointer, indent=1), encoding="utf-8")
        os.replace(concept_root / "LAST_TRAINED.json.tmp", last_path)
    return {"unchanged": False, "version": chosen, "model_dir": str(dest), "metrics": metrics, "pointer": pointer}
