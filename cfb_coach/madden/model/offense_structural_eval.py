"""Offline, game-level, chronological evaluation of the structure-aware offense.

Variants
  A  experimental_shrinkage          current offensive ML coordinator (heldout_eval.ExperimentalSuccess)
  B  structural_features(+ablations) A's logit as offset + structural residual (predictive, scored on the played play)
  C  adjustment-aware selection      DECISION BEHAVIOR only: hot-route outcomes were never logged, so no success claim
  D  C plus verified macros          reported "not evaluable" unless verified macro effects exist

Reuses ``heldout_eval`` folds, bootstrap and metrics. Reads only; never writes the DB.
No synthetic outcomes are generated here: with too few games the report says so and claims nothing.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_structural_model as sm

EVAL_VERSION = "offense_structural_eval.v1"
MIN_GAMES_FOR_CLAIM = 8


class StructuralSuccess:
    """Predictor wrapper (heldout_eval protocol): structural residual over the experimental offset."""

    targets = ("success",)

    def __init__(self, name: str = "structural_features", *, use_matchup: bool = True, penalty_scale: float = 1.0):
        self.name = name
        self.use_matchup = use_matchup
        self.penalty_scale = penalty_scale
        self.art: sm.StructuralArtifact | None = None

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        self.art = sm.train(train_rows, penalty_scale=self.penalty_scale, use_matchup=self.use_matchup)

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        assert self.art is not None
        s = sm.structure_any(str(row.get("action_formation") or ""), str(row.get("action_play") or ""))
        sc = sm.score(self.art, s, down=row.get("down"), distance=row.get("distance"), yardline=row.get("yardline"),
                      opponent_id=row.get("opponent_id"), opponent_type=row.get("opponent_type"))
        return {"success": sc.probability}


def predictive_models() -> list[he.Predictor]:
    keep = he.is_verified_offense
    return [
        he.BaseRate(("success",), he.offense_label, keep),
        he.ExperimentalSuccess(),
        StructuralSuccess("structural_features"),
        StructuralSuccess("structural_no_matchup", use_matchup=False),
        StructuralSuccess("structural_penalty_x0.3", penalty_scale=0.3),
        StructuralSuccess("structural_penalty_x3", penalty_scale=3.0),
    ]


def _entropy(counts: Mapping[Any, int]) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log(c / n) for c in counts.values() if c) if n else 0.0


def behavior_eval(data: he.EvalData, folds: Sequence[he.Fold], *, max_snaps_per_fold: int = 60) -> dict[str, Any]:
    """What C (and D) would DO on held-out snaps, using pre-snap fields only."""
    by_game: dict[str, list[Mapping[str, Any]]] = {}
    for r in data.rows:
        if r.get("game_id"):
            by_game.setdefault(str(r["game_id"]), []).append(r)
    book = [(p["formation"], p["play"]) for p in oa.load_effective()["plays"]]
    book_set = set(book)
    out = {"snaps": 0, "illegal_or_off_book": 0, "adjusted_research": 0, "adjusted_live": 0,
           "rejected_unconfirmed": 0, "truncated": 0}
    lat: list[float] = []
    picks: Counter = Counter()
    actual: Counter = Counter()
    repeats = total_pairs = 0
    form_counts: Counter = Counter()
    concepts: Counter = Counter()
    for fold in folds:
        train = [r for g in fold.train_games for r in by_game.get(g, [])]
        art = sm.train(train)
        test = [r for r in by_game.get(fold.test_game, []) if he.is_verified_offense(r)][:max_snaps_per_fold]
        prev = None
        for r in test:
            sit = od.Situation.from_mapping({k: r.get(k) for k in ("down", "distance", "yardline", "opponent_id", "opponent_type")})
            live = od.decide(art, sit, book, mode="live")
            res = od.decide(art, sit, book, mode="research")
            lat.append(max(live.latency_ms, res.latency_ms))
            out["snaps"] += 1
            if (live.formation, live.play) not in book_set:
                out["illegal_or_off_book"] += 1
            out["adjusted_live"] += live.adjustment != od.NO_ADJUSTMENT
            out["adjusted_research"] += res.adjustment != od.NO_ADJUSTMENT
            out["rejected_unconfirmed"] += sum(1 for p in res.plans if p.accepted and not p.executable_now) > 0
            out["truncated"] += bool(live.truncated or res.truncated)
            key = (res.formation, res.play)
            picks[key] += 1
            form_counts[res.formation] += 1
            for c in (res.final_structure.hint.get("concepts", []) if res.final_structure else []):
                concepts[c] += 1
            actual[(r.get("action_formation"), r.get("action_play"))] += 1
            if prev is not None:
                total_pairs += 1
                repeats += prev == key
            prev = key
    n = out["snaps"] or 1
    return {
        **out,
        "adjusted_live_rate": out["adjusted_live"] / n, "adjusted_research_rate": out["adjusted_research"] / n,
        "latency_ms_p50": he.percentile(lat, 50), "latency_ms_p95": he.percentile(lat, 95),
        "latency_ms_max": max(lat) if lat else None, "budget_ms": 150,
        "distinct_plays_chosen": len(picks), "play_entropy_nats": _entropy(picks),
        "distinct_formations_chosen": len(form_counts), "concept_hints_chosen": dict(concepts),
        "repeat_rate_consecutive": (repeats / total_pairs) if total_pairs else None,
        "actual_played_distinct_plays": len(actual), "actual_play_entropy_nats": _entropy(actual),
        "variant_D_macros": "not evaluable: no user-verified offensive macro effects exist",
        "note": "C/D are behavior metrics. Adjusted-play success cannot be measured from past games "
                "(no hot-route executions were logged).",
    }


def evaluate(data: he.EvalData, *, min_train_games: int = he.DEFAULT_MIN_TRAIN_GAMES, n_boot: int = he.DEFAULT_BOOTSTRAP,
             seed: int = he.DEFAULT_SEED, run_behavior: bool = True) -> dict[str, Any]:
    rows = data.rows
    verified = [r for r in rows if he.is_verified_offense(r)]
    book = {(p["formation"], p["play"]) for p in oa.load_effective()["plays"]}
    games = len({r.get("game_id") for r in verified})
    report: dict[str, Any] = {
        "eval_version": EVAL_VERSION, "source": data.source,
        "data": {"rows": len(rows), "verified_offense_rows": len(verified), "verified_offense_games": games,
                 "verified_rows_on_installed_93": sum((r.get("action_formation"), r.get("action_play")) in book for r in verified),
                 "by_opponent_type": dict(Counter(r.get("opponent_type") or "unknown" for r in verified))},
        "claims_allowed": games >= MIN_GAMES_FOR_CLAIM,
    }
    if games < MIN_GAMES_FOR_CLAIM:
        report["verdict"] = (f"INSUFFICIENT DATA: {games} verified offense games (< {MIN_GAMES_FOR_CLAIM}). "
                             "Metrics below are plumbing only; no model-superiority claim is made.")
    folds = he.make_folds(rows, data.game_order, is_test_row=he.is_verified_offense, mode="chronological",
                          min_train_games=min_train_games)
    report["folds"] = len(folds)
    if not folds:
        report["metrics"] = {}
        report["verdict"] = report.get("verdict", "") + " No folds could be formed."
        return report
    t0 = time.perf_counter()
    preds, timing = he.run_folds(rows, folds, predictive_models, targets=("success",), label_fn=he.offense_label,
                                 is_test_row=he.is_verified_offense)
    metrics = he.summarize_predictions(preds, ("success",), reference="experimental_shrinkage", n_boot=n_boot, seed=seed)
    # Within-play-set slice: plays with sourced routes are the only ones the structure channel can really use.
    report["metrics"] = metrics
    report["timing"] = timing
    report["eval_seconds"] = round(time.perf_counter() - t0, 1)
    if run_behavior:
        report["behavior"] = behavior_eval(data, folds)
    return report


def render_markdown(rep: Mapping[str, Any]) -> str:
    L = [f"# Offense structural evaluation ({rep['eval_version']})", "",
         f"Source: `{rep.get('source')}`", ""]
    d = rep["data"]
    L += [f"- verified offense rows: {d['verified_offense_rows']} in {d['verified_offense_games']} games "
          f"({d['verified_rows_on_installed_93']} on the installed 93 plays)", f"- folds (chronological, whole games): {rep.get('folds')}"]
    if "verdict" in rep:
        L += ["", f"**{rep['verdict']}**"]
    m = (rep.get("metrics") or {}).get("success")
    if m:
        L += ["", "## Held-out success prediction", "", "| model | n | log loss | brier | auc |", "|---|---|---|---|---|"]
        for name, v in m["models"].items():
            L.append(f"| {name} | {v['n']} | {v['log_loss']:.4f} | {v['brier']:.4f} | {v['auc'] if v['auc'] is None else round(v['auc'], 3)} |")
        L += ["", "Paired bootstrap vs `experimental_shrinkage` (negative log-loss difference = better):", ""]
        for name, v in m["vs_reference"].items():
            L.append(f"- {name}: {v}")
    b = rep.get("behavior")
    if b:
        L += ["", "## Decision behavior (C/D)", ""]
        for k in ("snaps", "illegal_or_off_book", "adjusted_live_rate", "adjusted_research_rate", "latency_ms_p50",
                  "latency_ms_p95", "latency_ms_max", "distinct_plays_chosen", "play_entropy_nats",
                  "actual_played_distinct_plays", "actual_play_entropy_nats", "repeat_rate_consecutive", "truncated"):
            L.append(f"- {k}: {b[k]}")
        L += ["", f"- D: {b['variant_D_macros']}", f"- {b['note']}"]
    return "\n".join(L) + "\n"
