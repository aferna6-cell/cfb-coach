"""Offline, held-out evaluation of Madden play-calling models.

Answers two separate questions and never mixes them:

1. *Predictive quality* — on plays we know were actually executed, how well
   does each model forecast the outcome (success, conversion, explosive,
   turnover)? Models are refit per fold on earlier games only and scored on a
   later, unseen game (rolling origin). An optional mode also removes every
   game against the test game's opponent from training.
2. *Live behaviour* — what the logged live system actually did (fallbacks,
   legality, latency, provenance, how often ML and heuristic agreed). These
   numbers are descriptive. Outcomes split by "ML agreed with the heuristic"
   are correlations, not proof that one policy calls better plays: every
   logged decision was deterministic, so there is no randomisation to support
   a counterfactual estimate.

The database is opened read-only. Nothing here trains, saves, or activates a
live model.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import opponent_tendencies as ot
from cfb_coach.outcome import parse_outcome

EVAL_VERSION = "madden-heldout-eval.v1"
DEFAULT_SEED = 20261106
DEFAULT_MIN_TRAIN_GAMES = 3
DEFAULT_BOOTSTRAP = 1000
OFFENSE_TARGETS = ("success", "conversion", "explosive", "turnover")
EXPLOSIVE_YARDS = {"run": 10, "pass": 20, "screen": 20}
EXPLOSIVE_YARDS_UNKNOWN_FAMILY = 15
TURNOVER_KINDS = frozenset({"int", "fumble"})
EPS = 1e-6


# ---------------------------------------------------------------- data


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class EvalData:
    rows: list[dict[str, Any]]
    game_order: dict[str, int]
    decisions: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""


def load_eval_data(db_path: str | Path) -> EvalData:
    """Rows via the production dataset builder, plus game chronology and logs."""
    conn = connect_readonly(db_path)
    try:
        rows = dataset_mod.build_rows(db=SimpleNamespace(conn=conn, path=str(db_path)))
        order = game_chronology(conn)
        try:
            decisions = [dict(r) for r in conn.execute("SELECT * FROM ml_decisions ORDER BY id")]
        except sqlite3.DatabaseError:
            decisions = []
    finally:
        conn.close()
    return EvalData(rows=rows, game_order=order, decisions=decisions, source=str(db_path))


def game_chronology(conn: sqlite3.Connection) -> dict[str, int]:
    """Rank sessions by their first recorded snap (timestamp, then row id)."""
    found = conn.execute(
        "SELECT session_id, MIN(ts) AS first_ts, MIN(id) AS first_id FROM snaps "
        "WHERE session_id IS NOT NULL AND session_id != '' GROUP BY session_id"
    ).fetchall()
    ranked = sorted(found, key=lambda r: (str(r["first_ts"] or ""), int(r["first_id"] or 0)))
    return {str(r["session_id"]): i for i, r in enumerate(ranked)}


# ---------------------------------------------------------------- labels


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _kind(row: Mapping[str, Any]) -> str | None:
    result = row.get("result")
    if result in (None, ""):
        return None
    return parse_outcome(str(result)).kind


def offense_label(row: Mapping[str, Any], target: str) -> int | None:
    """Binary label for one offensive target, or None when not knowable."""
    if target == "success":
        s = str(row.get("success") or "")
        return 1 if s == "true" else 0 if s == "false" else None
    if target == "conversion":
        if _int(row.get("down")) not in (3, 4):
            return None
        conv = em._converted_offense(row)
        return None if conv is None else int(conv)
    if target == "turnover":
        kind = _kind(row)
        return None if kind is None else int(kind in TURNOVER_KINDS)
    if target == "explosive":
        return explosive_label(row.get("yards"), _kind(row), row.get("action_play"))
    raise ValueError(target)


_PASS_CONCEPTS = frozenset({"mesh", "flood", "vert", "smash", "stick", "cross", "pa", "slant", "wheel"})


def run_or_pass(play: str | None) -> str:
    """'run', 'pass', 'screen' or 'unknown' using the family, then concept, classifier."""
    fam = em._play_family(play)
    if fam in ("run", "pass", "screen"):
        return fam
    concept = em._play_concept(play)
    if concept == "run_concept":
        return "run"
    if concept == "screen":
        return "screen"
    if concept in _PASS_CONCEPTS:
        return "pass"
    return "unknown"


def explosive_label(yards: Any, kind: str | None, play: str | None) -> int | None:
    if kind in TURNOVER_KINDS or kind in ("incomplete", "sack"):
        return 0
    y = _int(yards)
    if y is None:
        return None
    threshold = EXPLOSIVE_YARDS.get(run_or_pass(play), EXPLOSIVE_YARDS_UNKNOWN_FAMILY)
    return int(y >= threshold)


def is_verified_offense(row: Mapping[str, Any]) -> bool:
    return (
        row.get("side") == "offense"
        and row.get("eligibility") in dataset_mod.SUPERVISED_ELIGIBLE
        and bool(row.get("action_play"))
    )


def verification_basis(row: Mapping[str, Any]) -> str:
    src = str(row.get("execution_attestation_source") or "")
    return "standing_user_attestation" if src == "standing_user_attestation" else "per_snap_or_legacy"


def opponent_pass_label(row: Mapping[str, Any]) -> int | None:
    """Opponent's observed run(0)/pass(1) on our defensive snaps.

    Uses what the opponent was seen doing, never what we called, so it is
    independent of our own unverified defensive execution.
    """
    if row.get("side") != "defense":
        return None
    fam = run_or_pass(row.get("concept_seen"))
    if fam == "run":
        return 0
    if fam in ("pass", "screen"):
        return 1
    return None


# ---------------------------------------------------------------- models


class Predictor(Protocol):
    name: str
    targets: tuple[str, ...]

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None: ...

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]: ...


LabelFn = Callable[[Mapping[str, Any], str], "int | None"]


def _shrunk(successes: float, n: float, prior: float, strength: float) -> float:
    return (successes + strength * prior) / (n + strength)


class BaseRate:
    """Training-set rate per target (Laplace-smoothed)."""

    def __init__(self, targets: Sequence[str], label_fn: LabelFn, train_filter: Callable[[Mapping[str, Any]], bool]):
        self.name = "base_rate"
        self.targets = tuple(targets)
        self._label = label_fn
        self._keep = train_filter
        self.rates: dict[str, float] = {}

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        for t in self.targets:
            ys = [y for r in train_rows if self._keep(r) for y in [self._label(r, t)] if y is not None]
            self.rates[t] = (sum(ys) + 1.0) / (len(ys) + 2.0)

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        return dict(self.rates)


class ShrunkTable:
    """Nested partial pooling: global -> level-1 key -> level-2 key.

    Each level is a beta-binomial style shrinkage toward its parent, so sparse
    cells fall back smoothly to the broader rate.
    """

    def __init__(
        self,
        name: str,
        targets: Sequence[str],
        label_fn: LabelFn,
        train_filter: Callable[[Mapping[str, Any]], bool],
        keys: Sequence[Callable[[Mapping[str, Any]], str]],
        strength: float = 10.0,
    ):
        self.name = name
        self.targets = tuple(targets)
        self._label = label_fn
        self._keep = train_filter
        self._keys = list(keys)
        self.strength = strength
        self.glob: dict[str, float] = {}
        self.cells: dict[str, list[dict[tuple[str, ...], list[float]]]] = {}

    def _path(self, row: Mapping[str, Any]) -> list[tuple[str, ...]]:
        parts: list[str] = []
        out = []
        for key in self._keys:
            parts.append(key(row))
            out.append(tuple(parts))
        return out

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        for t in self.targets:
            levels: list[dict[tuple[str, ...], list[float]]] = [{} for _ in self._keys]
            total = [0.0, 0.0]
            for r in train_rows:
                if not self._keep(r):
                    continue
                y = self._label(r, t)
                if y is None:
                    continue
                total[0] += y
                total[1] += 1
                for lvl, k in enumerate(self._path(r)):
                    cell = levels[lvl].setdefault(k, [0.0, 0.0])
                    cell[0] += y
                    cell[1] += 1
            self.glob[t] = (total[0] + 1.0) / (total[1] + 2.0)
            self.cells[t] = levels

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        out = {}
        for t in self.targets:
            p = self.glob[t]
            for lvl, k in enumerate(self._path(row)):
                cell = self.cells[t][lvl].get(k)
                if cell:
                    p = _shrunk(cell[0], cell[1], p, self.strength)
            out[t] = p
        return out


def dd_key(row: Mapping[str, Any]) -> str:
    return em._dd_bucket(row.get("down"), row.get("distance"))


def field_key(row: Mapping[str, Any]) -> str:
    return em._field_bucket(_int(row.get("yardline")))


def action_family_key(row: Mapping[str, Any]) -> str:
    return em._play_family(row.get("action_play"))


class ExperimentalSuccess:
    """The live experimental shrinkage model, refit on the fold's training games.

    Trained exactly as production does (verified rows plus discounted context
    rows), then scores the play that was actually executed. Coverage is left
    unknown and the heuristic bonus at 0 because neither is reconstructable
    offline; this is noted in the report.
    """

    name = "experimental_shrinkage"
    targets = ("success",)

    def __init__(self) -> None:
        self.artifact: em.ExperimentalArtifact | None = None

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        self.artifact = em.train_experimental([dict(r) for r in train_rows], side="offense")

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        assert self.artifact is not None
        res = em.predict_success(
            self.artifact,
            formation=str(row.get("action_formation") or ""),
            play=str(row.get("action_play") or ""),
            down=row.get("down"),
            distance=row.get("distance"),
            yardline=_int(row.get("yardline")),
            opponent_id=row.get("opponent_id"),
            opponent_type=row.get("opponent_type"),
        )
        return {"success": float(res["probability"])}


def default_offense_models() -> list[Predictor]:
    keep = is_verified_offense
    return [
        BaseRate(OFFENSE_TARGETS, offense_label, keep),
        ShrunkTable("situation_table", OFFENSE_TARGETS, offense_label, keep, [dd_key, field_key]),
        ShrunkTable("situation_play_family", OFFENSE_TARGETS, offense_label, keep, [dd_key, action_family_key]),
        ExperimentalSuccess(),
    ]


OPPONENT_VARIANTS: dict[str, dict[str, Any] | None] = {
    "coin_flip": None,
    "league_situation": {"use_styles": False, "use_individual": False},
    "individual_only": None,
    "league+individual": {"use_styles": False, "use_individual": True},
    "league+style": {"use_styles": True, "use_individual": False},
    "league+style+individual": {"use_styles": True, "use_individual": True},
}
HANDOFF_SENSITIVITY = (4.0, 30.0)


def run_pass_observations(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[ot.RunPassObservation]]:
    """Observed opponent run/pass per game, in snap order."""
    out: dict[str, list[ot.RunPassObservation]] = {}
    for r in rows:
        y = opponent_pass_label(r)
        if y is None or not r.get("game_id") or not r.get("opponent_id"):
            continue
        out.setdefault(str(r["game_id"]), []).append(ot.RunPassObservation(
            opponent_id=str(r["opponent_id"]), game_id=str(r["game_id"]),
            down=r.get("down"), distance=r.get("distance"), is_pass=int(y),
        ))
    return out


def _own_bucket(n: int) -> str:
    return "0-4" if n < 5 else "5-14" if n < 15 else "15+"


def opponent_sequential_eval(
    rows: Sequence[Mapping[str, Any]],
    game_order: Mapping[str, int],
    *,
    scheme: str = "chronological",
    handoff: float = ot.PRIOR_HANDOFF,
    min_train_obs: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Predict each observed opponent snap from information available before it.

    ``chronological``: the league/style prior is fit on other opponents' games
    played earlier. ``leave_one_opponent_out``: fit on every other opponent's
    games regardless of date (measures structure; not deployable as-is). In both,
    the individual level only sees this opponent's earlier games and the earlier
    snaps of the current game.
    """
    if scheme not in ("chronological", "leave_one_opponent_out"):
        raise ValueError(scheme)
    by_game = {g: o for g, o in run_pass_observations(rows).items() if g in game_order}
    games = sorted(by_game, key=lambda g: game_order[g])
    opp_of = {g: by_game[g][0].opponent_id for g in games}
    preds: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for g in games:
        opp = opp_of[g]
        earlier = [h for h in games if game_order[h] < game_order[g]]
        others = earlier if scheme == "chronological" else games
        train = [o for h in others if opp_of[h] != opp for o in by_game[h]]
        if len(train) < min_train_obs:
            continue
        prior = ot.RunPassPrior().fit(train)
        own_before = [o for h in earlier if opp_of[h] == opp for o in by_game[h]]
        folds.append({"game": g, "train_plays": len(train), "styles": len(prior.styles),
                      "style_sizes": [s["opponents"] for s in prior.styles]})
        test = by_game[g]
        for t, o in enumerate(test):
            history = own_before + test[:t]
            p: dict[str, float] = {}
            for name, kw in OPPONENT_VARIANTS.items():
                if name == "coin_flip":
                    p[name] = 0.5
                elif kw is None:
                    p[name] = (sum(h.is_pass for h in history) + 1.0) / (len(history) + 2.0)
                else:
                    p[name] = ot.estimate_run_pass(prior, history, o.down, o.distance, handoff=handoff, **kw)["p_pass"]
            for k in HANDOFF_SENSITIVITY:
                p[f"league+style+individual@k{k:g}"] = ot.estimate_run_pass(prior, history, o.down, o.distance, handoff=k)["p_pass"]
            preds.append({"target": "opp_pass", "game": g, "y": o.is_pass, "own_n": _own_bucket(len(history)),
                          "n_history": len(history), "down": o.down, "distance": o.distance, "p": p})
    return preds, folds


# ---------------------------------------------------------------- metrics


def binary_metrics(ys: Sequence[int], ps: Sequence[float], bins: int = 10) -> dict[str, Any]:
    n = len(ys)
    if n == 0:
        return {"n": 0}
    clipped = [min(1 - EPS, max(EPS, p)) for p in ps]
    ll = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for y, p in zip(ys, clipped)) / n
    brier = sum((p - y) ** 2 for y, p in zip(ys, ps)) / n
    table = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(ps) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        mean_p = sum(ps[i] for i in idx) / len(idx)
        obs = sum(ys[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(mean_p - obs)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "mean_pred": round(mean_p, 4), "observed": round(obs, 4)})
    return {
        "n": n,
        "positives": int(sum(ys)),
        "observed_rate": sum(ys) / n,
        "mean_pred": sum(ps) / n,
        "log_loss": ll,
        "brier": brier,
        "auc": auc(ys, ps),
        "ece": ece,
        "calibration": table,
    }


def auc(ys: Sequence[int], ps: Sequence[float]) -> float | None:
    pos = sum(ys)
    neg = len(ys) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranks = [0.0] * len(ps)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ps[order[j + 1]] == ps[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, ys) if y == 1)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def wilson(k: int, n: int, z: float = 1.96) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"k": k, "n": n, "rate": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[k]


def _row_ll(y: int, p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def paired_bootstrap(
    preds: Sequence[Mapping[str, Any]], model: str, reference: str, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Game-clustered bootstrap of mean log-loss difference (model - reference).

    Negative means the model beats the reference. Resampling whole games keeps
    plays from the same game together, so the interval is not overconfident.
    """
    by_game: dict[str, list[float]] = {}
    for p in preds:
        if model in p["p"] and reference in p["p"]:
            d = _row_ll(p["y"], p["p"][model]) - _row_ll(p["y"], p["p"][reference])
            by_game.setdefault(p["game"], []).append(d)
    games = sorted(by_game)
    if len(games) < 2:
        return {"games": len(games), "mean_diff": None, "lo": None, "hi": None}
    flat = [d for g in games for d in by_game[g]]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        pick = [games[rng.randrange(len(games))] for _ in games]
        vals = [d for g in pick for d in by_game[g]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return {
        "games": len(games),
        "plays": len(flat),
        "mean_diff": sum(flat) / len(flat),
        "lo": stats[int(0.025 * n_boot)],
        "hi": stats[min(n_boot - 1, int(0.975 * n_boot))],
        "share_model_better": sum(1 for s in stats if s < 0) / n_boot,
    }


# ---------------------------------------------------------------- folds


@dataclass
class Fold:
    test_game: str
    train_games: list[str]
    test_opponent: str | None


def make_folds(
    rows: Sequence[Mapping[str, Any]],
    game_order: Mapping[str, int],
    *,
    is_test_row: Callable[[Mapping[str, Any]], bool],
    mode: str = "chronological",
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> list[Fold]:
    """Rolling-origin folds by whole game.

    ``chronological``: train on every earlier game.
    ``exclude_opponent``: also drop every game against the test opponent, so
    the model has never seen that opponent (cold-start check).
    """
    if mode not in ("chronological", "exclude_opponent"):
        raise ValueError(mode)
    opponent_of: dict[str, str | None] = {}
    test_games: set[str] = set()
    for r in rows:
        g = r.get("game_id")
        if not g or g not in game_order:
            continue
        opponent_of.setdefault(g, r.get("opponent_id"))
        if is_test_row(r):
            test_games.add(g)
    ordered = sorted(opponent_of, key=lambda g: game_order[g])
    folds = []
    for g in ordered:
        if g not in test_games:
            continue
        earlier = [h for h in ordered if game_order[h] < game_order[g]]
        if mode == "exclude_opponent":
            earlier = [h for h in earlier if opponent_of.get(h) != opponent_of.get(g)]
        if sum(1 for h in earlier if h in test_games) < min_train_games:
            continue
        folds.append(Fold(test_game=g, train_games=earlier, test_opponent=opponent_of.get(g)))
    return folds


def run_folds(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Fold],
    model_factory: Callable[[], list[Predictor]],
    *,
    targets: Sequence[str],
    label_fn: LabelFn,
    is_test_row: Callable[[Mapping[str, Any]], bool],
    extra: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``extra`` adds per-row fields (for example situation slices) to each prediction."""
    by_game: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if r.get("game_id"):
            by_game.setdefault(str(r["game_id"]), []).append(r)
    preds: list[dict[str, Any]] = []
    fit_ms: dict[str, list[float]] = {}
    predict_us: dict[str, list[float]] = {}
    for fold in folds:
        train = [r for g in fold.train_games for r in by_game.get(g, [])]
        test = [r for r in by_game.get(fold.test_game, []) if is_test_row(r)]
        models = model_factory()
        for m in models:
            t0 = time.perf_counter()
            m.fit(train)
            fit_ms.setdefault(m.name, []).append((time.perf_counter() - t0) * 1000)
        for r in test:
            outs: dict[str, dict[str, float]] = {}
            for m in models:
                t0 = time.perf_counter()
                outs[m.name] = m.predict(r)
                predict_us.setdefault(m.name, []).append((time.perf_counter() - t0) * 1e6)
            for t in targets:
                y = label_fn(r, t)
                if y is None:
                    continue
                preds.append(
                    {
                        "target": t,
                        "game": fold.test_game,
                        "y": y,
                        "opponent_type": r.get("opponent_type") or "unknown",
                        "basis": verification_basis(r),
                        "p": {name: o[t] for name, o in outs.items() if t in o},
                        **(extra(r) if extra else {}),
                    }
                )
    timing = {
        name: {
            "fit_ms_median": percentile(fit_ms[name], 50),
            "predict_us_p50": percentile(predict_us.get(name, []), 50),
            "predict_us_p95": percentile(predict_us.get(name, []), 95),
        }
        for name in fit_ms
    }
    return preds, timing


def summarize_predictions(
    preds: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    *,
    reference: str = "base_rate",
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    strata: Sequence[str] = ("opponent_type", "basis"),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in targets:
        tp = [p for p in preds if p["target"] == t]
        models = sorted({m for p in tp for m in p["p"]})
        block: dict[str, Any] = {"models": {}, "vs_reference": {}, "strata": {}}
        for m in models:
            sub = [p for p in tp if m in p["p"]]
            block["models"][m] = binary_metrics([p["y"] for p in sub], [p["p"][m] for p in sub])
            if m != reference:
                block["vs_reference"][m] = paired_bootstrap(tp, m, reference, n_boot=n_boot, seed=seed)
        for s in strata:
            for value in sorted({str(p.get(s)) for p in tp}):
                sub = [p for p in tp if str(p.get(s)) == value]
                block["strata"][f"{s}={value}"] = {
                    m: {k: v for k, v in binary_metrics([p["y"] for p in sub if m in p["p"]], [p["p"][m] for p in sub if m in p["p"]]).items() if k != "calibration"}
                    for m in models
                }
        block["test_games"] = len({p["game"] for p in tp})
        out[t] = block
    return out


# ---------------------------------------------------------------- live logs


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _pick_key(formation: Any, play: Any) -> tuple[str, str]:
    return (str(formation or "").strip().lower(), str(play or "").strip().lower())


def live_behavior(decisions: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive reliability and provenance of the logged live decisions."""
    n = len(decisions)
    fell_back = 0
    budget_over = 0
    lat_total: list[float] = []
    lat_model: list[float] = []
    legal = {"in_book": 0, "not_in_book": 0, "unknown": 0}
    agree = 0
    with_propensity = 0
    overlap = 0
    fallback_reasons: dict[str, int] = {}
    counts: dict[str, dict[str, int]] = {"mode": {}, "policy_source": {}, "policy_version": {}, "propensity_method": {}}
    for d in decisions:
        for k in counts:
            v = str(d.get(k) or "none")
            counts[k][v] = counts[k].get(v, 0) + 1
        try:
            dj = json.loads(d.get("decision_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            dj = {}
        if dj.get("fell_back") or str(d.get("policy_source")) == "heuristic":
            fell_back += 1
            reason = str(dj.get("fallback_reason") or d.get("shadow_status") or "unspecified")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        total = dj.get("latency_ms")
        if isinstance(total, (int, float)):
            lat_total.append(float(total))
            if float(total) > float(dj.get("inference_budget_ms") or 150):
                budget_over += 1
        if isinstance(d.get("latency_ms_model"), (int, float)):
            lat_model.append(float(d["latency_ms_model"]))
        final = _pick_key(d.get("final_formation"), d.get("final_play"))
        if final == _pick_key(d.get("heuristic_formation"), d.get("heuristic_play")):
            agree += 1
        try:
            cands = json.loads(d.get("candidates_json") or "null")
        except (TypeError, json.JSONDecodeError):
            cands = None
        match = [c for c in cands or [] if isinstance(c, dict) and _pick_key(c.get("formation"), c.get("play")) == final]
        if not match:
            legal["unknown"] += 1
        elif any(_truthy(c.get("in_applied_book")) for c in match):
            legal["in_book"] += 1
        else:
            legal["not_in_book"] += 1
        bp = d.get("behavior_propensity")
        if bp not in (None, ""):
            with_propensity += 1
            try:
                if 0.0 < float(bp) < 1.0:
                    overlap += 1
            except (TypeError, ValueError):
                pass

    outcome_by_snap = {str(r.get("snap_id")): r for r in rows if is_verified_offense(r)}
    agree_rows: dict[str, list[int]] = {"agreed": [], "disagreed": []}
    exec_matches_final = [0, 0]
    for d in decisions:
        r = outcome_by_snap.get(str(d.get("snap_id")))
        if r is None:
            continue
        y = offense_label(r, "success")
        final = _pick_key(d.get("final_formation"), d.get("final_play"))
        exec_matches_final[1] += 1
        if _pick_key(r.get("action_formation"), r.get("action_play")) == final:
            exec_matches_final[0] += 1
        if y is None:
            continue
        same = final == _pick_key(d.get("heuristic_formation"), d.get("heuristic_play"))
        agree_rows["agreed" if same else "disagreed"].append(y)

    return {
        "decisions": n,
        "counts": counts,
        "fallback": wilson(fell_back, n),
        "fallback_reasons": fallback_reasons,
        "latency_ms_total": {q: percentile(lat_total, q) for q in (50, 95, 99, 100)},
        "latency_ms_model": {q: percentile(lat_model, q) for q in (50, 95, 99, 100)},
        "over_budget": wilson(budget_over, len(lat_total)),
        "final_pick_in_applied_book": legal,
        "ml_heuristic_agreement": wilson(agree, n),
        "behavior_propensity_logged": with_propensity,
        "propensity_strictly_between_0_and_1": overlap,
        "executed_matches_final": wilson(exec_matches_final[0], exec_matches_final[1]),
        "success_when_ml_agreed_with_heuristic": wilson(sum(agree_rows["agreed"]), len(agree_rows["agreed"])),
        "success_when_ml_overrode_heuristic": wilson(sum(agree_rows["disagreed"]), len(agree_rows["disagreed"])),
    }


# ---------------------------------------------------------------- defense


def defense_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Observed results while we were on defense, by opponent type.

    These are what the opponent achieved. They are not credited to our
    defensive call, because our executed defensive play is not verified.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if r.get("side") != "defense" or r.get("result") in (None, ""):
            continue
        groups.setdefault("all", []).append(r)
        groups.setdefault(f"opponent_type={r.get('opponent_type') or 'unknown'}", []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        stops = [1 if r.get("stop") == "true" else 0 for r in rs if r.get("stop") in ("true", "false")]
        kinds = [_kind(r) for r in rs]
        expl = [explosive_label(r.get("yards"), k, r.get("concept_seen")) for r, k in zip(rs, kinds)]
        expl = [e for e in expl if e is not None]
        out[key] = {
            "plays": len(rs),
            "games": len({r.get("game_id") for r in rs}),
            "stop_rate": wilson(sum(stops), len(stops)),
            "explosive_allowed": wilson(sum(expl), len(expl)),
            "td_allowed": wilson(sum(1 for k in kinds if k == "td"), len(kinds)),
            "takeaways": wilson(sum(1 for k in kinds if k in TURNOVER_KINDS), len(kinds)),
        }
    return out


# ---------------------------------------------------------------- driver


def evaluate(
    data: EvalData,
    *,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    offense_models: Callable[[], list[Predictor]] = default_offense_models,
) -> dict[str, Any]:
    rows = data.rows
    verified = [r for r in rows if is_verified_offense(r)]
    _, _, knowledge_version, knowledge_origin = em._load_research_boosts()
    report: dict[str, Any] = {
        "eval_version": EVAL_VERSION,
        "config": {"min_train_games": min_train_games, "bootstrap": n_boot, "seed": seed,
                   "explosive_yards": EXPLOSIVE_YARDS, "explosive_unknown_family": EXPLOSIVE_YARDS_UNKNOWN_FAMILY,
                   "research_knowledge": f"{knowledge_version} ({knowledge_origin or 'none'})"},
        "data": {
            "rows": len(rows),
            "games": len(data.game_order),
            "verified_offense_rows": len(verified),
            "verified_offense_games": len({r.get("game_id") for r in verified}),
            "verified_by_basis": _count(verification_basis(r) for r in verified),
            "verified_by_opponent_type": _count(r.get("opponent_type") or "unknown" for r in verified),
            "defense_rows": sum(1 for r in rows if r.get("side") == "defense"),
            "defense_rows_with_opponent_run_pass": sum(1 for r in rows if opponent_pass_label(r) is not None),
        },
        "offense": {},
        "opponent_run_pass": {},
    }
    for mode in ("chronological", "exclude_opponent"):
        folds = make_folds(rows, data.game_order, is_test_row=is_verified_offense, mode=mode, min_train_games=min_train_games)
        preds, timing = run_folds(rows, folds, offense_models, targets=OFFENSE_TARGETS, label_fn=offense_label, is_test_row=is_verified_offense)
        report["offense"][mode] = {
            "folds": len(folds),
            "timing": timing,
            "metrics": summarize_predictions(preds, OFFENSE_TARGETS, n_boot=n_boot, seed=seed),
        }
    for scheme in ("chronological", "leave_one_opponent_out"):
        opreds, ofolds = opponent_sequential_eval(rows, data.game_order, scheme=scheme)
        report["opponent_run_pass"][scheme] = {
            "folds": ofolds,
            "metrics": summarize_predictions(opreds, ("opp_pass",), reference="league_situation",
                                             n_boot=n_boot, seed=seed, strata=("own_n",)),
            "feature_rows": [
                {"game": p["game"], "y": p["y"], "down": p["down"], "distance": p["distance"],
                 "n_history": p["n_history"], **{f"p_{k}": round(v, 5) for k, v in p["p"].items()}}
                for p in opreds
            ],
        }
    report["live_behavior"] = live_behavior(data.decisions, rows)
    report["defense_outcomes"] = defense_outcomes(rows)
    return report


def _count(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- report


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(w: Mapping[str, Any]) -> str:
    if not w or w.get("rate") is None:
        return "— (n=0)"
    return f"{w['rate']:.1%} [{w['lo']:.1%}, {w['hi']:.1%}] (n={w['n']})"


def render_markdown(report: Mapping[str, Any]) -> str:
    d = report["data"]
    lines = [
        "# Madden offline held-out evaluation",
        "",
        f"Evaluator `{report['eval_version']}`. Config: `{json.dumps(report['config'], sort_keys=True)}`.",
        "",
        "## Data",
        "",
        f"- Rows: {d['rows']} across {d['games']} games.",
        f"- Verified executed offensive plays: {d['verified_offense_rows']} in {d['verified_offense_games']} games "
        f"(basis {d['verified_by_basis']}; opponent type {d['verified_by_opponent_type']}).",
        f"- Defensive plays: {d['defense_rows']}; with the opponent's run/pass observed: {d['defense_rows_with_opponent_run_pass']}.",
        "",
        "## 1. Predicting outcomes of plays actually run (held-out games)",
        "",
        "Lower log loss / Brier is better; AUC 0.5 = no ranking skill. `Δ vs base` is the "
        "game-clustered bootstrap 95% interval of the log-loss difference against the "
        "training base rate (negative = better than base rate).",
    ]
    for mode, block in report["offense"].items():
        lines += ["", f"### Split: {mode} ({block['folds']} test games)"]
        for t, tb in block["metrics"].items():
            lines += ["", f"**{t}** — test games {tb.get('test_games', 0)}", "",
                      "| model | n | observed | mean pred | log loss | Brier | AUC | ECE | Δ vs base [95% CI] |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for m, mm in tb["models"].items():
                if not mm.get("n"):
                    continue
                vr = tb["vs_reference"].get(m)
                delta = "reference" if vr is None else (
                    f"{_f(vr['mean_diff'], 4)} [{_f(vr['lo'], 4)}, {_f(vr['hi'], 4)}]" if vr.get("mean_diff") is not None else "—"
                )
                lines.append(
                    f"| {m} | {mm['n']} | {_f(mm['observed_rate'])} | {_f(mm['mean_pred'])} | {_f(mm['log_loss'], 4)} | "
                    f"{_f(mm['brier'], 4)} | {_f(mm['auc'])} | {_f(mm['ece'])} | {delta} |"
                )
        timing = block["timing"]
        lines += ["", "Timing (fit median ms / predict p95 µs): " + ", ".join(
            f"{m} {_f(v['fit_ms_median'], 1)} / {_f(v['predict_us_p95'], 0)}" for m, v in timing.items())]
    lines += ["", "### Success by stratum (chronological split, log loss / AUC)", ""]
    chron = report["offense"].get("chronological", {}).get("metrics", {}).get("success", {})
    for key, ms in chron.get("strata", {}).items():
        lines.append(f"- {key}: " + "; ".join(
            f"{m} n={v.get('n')} ll={_f(v.get('log_loss'), 4)} auc={_f(v.get('auc'))}" for m, v in ms.items() if v.get("n")))

    lines += [
        "", "## 2. Predicting the opponent's run/pass on our defensive snaps", "",
        "Each snap is predicted from the league/style prior (other opponents only) plus this "
        "opponent's earlier plays. `own n` is how many of his plays had been seen. Primary "
        "hand-off strength k = 12 (the live `PRIOR_HANDOFF`); k = 4 and 30 are sensitivity checks.",
    ]
    for scheme, block in report["opponent_run_pass"].items():
        tb = block["metrics"].get("opp_pass", {})
        styles = [f["styles"] for f in block["folds"]]
        lines += ["", f"### {scheme} — {len(block['folds'])} test games, styles formed per fold {styles}", "",
                  "| model | n | log loss | Brier | AUC | ECE | Δ vs league_situation [95% CI] |",
                  "|---|---|---|---|---|---|---|"]
        for m, v in tb.get("models", {}).items():
            if not v.get("n"):
                continue
            vr = tb["vs_reference"].get(m)
            delta = "reference" if vr is None else (
                f"{_f(vr['mean_diff'], 4)} [{_f(vr['lo'], 4)}, {_f(vr['hi'], 4)}]" if vr.get("mean_diff") is not None else "—")
            lines.append(f"| {m} | {v['n']} | {_f(v['log_loss'], 4)} | {_f(v['brier'], 4)} | {_f(v['auc'])} | "
                         f"{_f(v['ece'])} | {delta} |")
        for key, ms in tb.get("strata", {}).items():
            lines.append(f"- {key}: " + "; ".join(
                f"{m} ll={_f(v.get('log_loss'), 4)}" for m, v in ms.items()
                if v.get("n") and "@k" not in m) + f" (n={next(iter(ms.values())).get('n')})")

    lb = report["live_behavior"]
    lines += [
        "", "## 3. What the live system did (descriptive, not causal)", "",
        f"- Decisions logged: {lb['decisions']}; modes {lb['counts']['mode']}; sources {lb['counts']['policy_source']}.",
        f"- Heuristic fallback: {_ci(lb['fallback'])}; reasons {lb['fallback_reasons']}.",
        f"- Total latency ms p50/p95/p99/max: " + "/".join(_f(v, 1) for v in lb["latency_ms_total"].values())
        + f"; over the 150 ms budget: {_ci(lb['over_budget'])}.",
        f"- Model latency ms p50/p95/p99/max: " + "/".join(_f(v, 1) for v in lb["latency_ms_model"].values()) + ".",
        f"- Final pick found in the applied playbook: {lb['final_pick_in_applied_book']}.",
        f"- ML final pick equalled the heuristic pick: {_ci(lb['ml_heuristic_agreement'])}.",
        f"- Executed play equalled the displayed final pick (verified rows): {_ci(lb['executed_matches_final'])}. "
        "Most of these are verified by the standing attestation, so this is close to true by construction.",
        f"- Propensity logged on {lb['behavior_propensity_logged']} decisions; strictly between 0 and 1 on "
        f"{lb['propensity_strictly_between_0_and_1']}. Without randomised choices, off-policy estimates of "
        "\"what if the heuristic had called it\" are not identifiable.",
        f"- Success when ML agreed with the heuristic: {_ci(lb['success_when_ml_agreed_with_heuristic'])}; "
        f"when ML overrode it: {_ci(lb['success_when_ml_overrode_heuristic'])}. Correlation only — "
        "agreement happens in different situations than overrides.",
        "", "## 4. Defensive results (what opponents achieved; not credited to our call)", "",
        "| group | plays | games | stop rate | explosive allowed | TD allowed | takeaways |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, v in report["defense_outcomes"].items():
        lines.append(f"| {key} | {v['plays']} | {v['games']} | {_ci(v['stop_rate'])} | {_ci(v['explosive_allowed'])} | "
                     f"{_ci(v['td_allowed'])} | {_ci(v['takeaways'])} |")
    lines += [
        "", "## Caveats", "",
        "- Only plays with a verified execution are scored; recommendations are never treated as executed plays.",
        "- `base_rate` is constant within a test game, so its pooled AUC only reflects drift between folds; read it as 0.5.",
        "- In `exclude_opponent`, every CPU game shares one opponent identity, so CPU test games have no eligible "
        "training games and only human games are scored.",
        "- The experimental model is scored with unknown coverage and no heuristic bonus (not reconstructable offline).",
        "- Touchdowns recorded without yardage have unknown explosive labels and are excluded from that target.",
        "- Opponent run/pass comes from the observed concept text; unclassifiable concepts are excluded.",
    ]
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "heldout_eval.json"
    mp = out / "heldout_eval.md"
    jp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    mp.write_text(render_markdown(report), encoding="utf-8")
    return jp, mp
