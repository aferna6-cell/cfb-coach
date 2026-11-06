"""Can we predict what a human opponent's offense will do before the snap? Offline, read-only.

Targets, from what the opponent was seen doing on our defensive snaps (never
our own call):
  * run vs pass
  * broad concept family (run / rpo / vert / flood / cross / stack / scram)

Each snap is predicted from ``strategic_context`` pre-snap features and the
earlier snaps of the same game. Two schemes:
  * chronological: trained only on games played earlier (deployable)
  * leave_one_opponent_out: trained on every other opponent (structure check)

Models: coin flip / class frequency, league situational table, the existing
partial-pooling estimator (``opponent_tendencies.estimate_run_pass``), logistic
regression, gradient-boosted trees and a small neural network. Intervals are a
game-clustered bootstrap. Output names games G1..Gn only.

    python scripts/madden_ml/opponent_prediction_eval.py --db AUDIT_COPY.db --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.madden.model import opponent_tendencies as ot  # noqa: E402
from cfb_coach.madden.model.strategic_context import context_from_snap  # noqa: E402
from cfb_coach.madden.situation import concept_family  # noqa: E402
from cfb_coach.opponents import is_cpu_opponent  # noqa: E402

FAMILIES = ("run", "rpo", "vert", "flood", "cross", "stack", "scram")
DROP_FEATURES = ("opp_run_share", "opp_run_share_missing", "opp_own_plays", "opp_pooled_plays", "opp_is_cpu")
MIN_TRAIN_LABELS = 40
N_BOOT = 2000
SEED = 7
EPS = 1e-6


def load_games(db: str) -> list[list[dict[str, Any]]]:
    """Human defensive snaps grouped by game (session), games and snaps in time order."""
    conn = sqlite3.connect(f"file:{Path(db).expanduser().resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM snaps WHERE side = 'defense' AND session_id IS NOT NULL AND session_id != '' "
            "ORDER BY ts, id")]
    finally:
        conn.close()
    games: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if is_cpu_opponent(str(r.get("opponent_id") or "")):
            continue
        games.setdefault(str(r["session_id"]), []).append(r)
    return sorted(games.values(), key=lambda g: (str(g[0]["ts"]), int(g[0]["id"])))


def build_examples(games: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for gi, game in enumerate(games):
        for t, row in enumerate(game):
            y = he.opponent_pass_label(row)
            fam = concept_family(row.get("concept_seen"))
            if y is None and fam is None:
                continue
            ctx = context_from_snap(row, earlier=game[:t])
            feats = {k: v for k, v in ctx.as_features().items() if k not in DROP_FEATURES}
            fam_hist = [concept_family(r.get("concept_seen")) for r in game[:t]]
            fam_hist = [f for f in fam_hist if f in FAMILIES]
            for f in FAMILIES:
                feats[f"game_fam_{f}"] = (fam_hist.count(f) + 0.0) / max(1, len(fam_hist))
            out.append({
                "game": f"G{gi + 1}", "gi": gi, "opp": str(row.get("opponent_id") or ""),
                "y": y, "fam": fam if fam in FAMILIES else None, "x": feats,
                "down": row.get("down"), "distance": row.get("distance"),
                "red_zone": bool(ctx.red_zone.value), "third_down": row.get("down") in (3, 4),
                "obs": ot.RunPassObservation(opponent_id=str(row.get("opponent_id") or ""), game_id=str(gi),
                                             down=row.get("down"), distance=row.get("distance"),
                                             is_pass=int(y)) if y is not None else None,
                "fam_hist": tuple(fam_hist),
            })
    return out


def _xy(examples: Sequence[dict[str, Any]], names: list[str], target: str):
    import numpy as np

    keep = [e for e in examples if e[target] is not None]
    return keep, np.array([[e["x"][n] for n in names] for e in keep], dtype=float), [e[target] for e in keep]


def _sk_models() -> dict[str, Callable[[], Any]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "boosted_trees": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=15, l2_regularization=1.0,
            random_state=SEED),
        "small_nn": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(8,), alpha=1.0, max_iter=3000, random_state=SEED)),
    }


def _train_ids(scheme: str, test_gi: int, n_games: int) -> list[int]:
    if scheme == "chronological":
        return list(range(test_gi))
    return [g for g in range(n_games) if g != test_gi]


def run_pass_eval(examples: list[dict[str, Any]], n_games: int, scheme: str) -> tuple[list[dict], list[dict]]:
    names = sorted(examples[0]["x"])
    preds, folds = [], []
    for gi in range(n_games):
        train_ids = set(_train_ids(scheme, gi, n_games))
        train = [e for e in examples if e["gi"] in train_ids and e["y"] is not None]
        test = [e for e in examples if e["gi"] == gi and e["y"] is not None]
        if len(train) < MIN_TRAIN_LABELS or not test:
            continue
        prior = ot.RunPassPrior().fit([e["obs"] for e in train])
        base = (sum(e["y"] for e in train) + 1.0) / (len(train) + 2.0)
        fitted = {}
        _, X, y = _xy(train, names, "y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, make in _sk_models().items():
                fitted[name] = make().fit(X, y)
        folds.append({"game": f"G{gi + 1}", "train_plays": len(train), "test_plays": len(test),
                      "styles": len(prior.styles)})
        _, Xt, _ = _xy(test, names, "y")
        sk_p = {name: m.predict_proba(Xt)[:, list(m.classes_).index(1)] for name, m in fitted.items()}
        for t, e in enumerate(test):
            history = [h["obs"] for h in test[:t]]
            p = {
                "coin_flip": 0.5,
                "league_rate": base,
                "league_situation": ot.estimate_run_pass(prior, [], e["down"], e["distance"],
                                                         use_styles=False, use_individual=False)["p_pass"],
                "in_game_only": (sum(h.is_pass for h in history) + 1.0) / (len(history) + 2.0),
                "existing_pooling": ot.estimate_run_pass(prior, history, e["down"], e["distance"],
                                                         use_styles=False, use_individual=True)["p_pass"],
                "existing_pooling+style": ot.estimate_run_pass(prior, history, e["down"], e["distance"])["p_pass"],
            }
            for k in (4.0, 30.0):
                p[f"existing_pooling@handoff{k:g}"] = ot.estimate_run_pass(
                    prior, history, e["down"], e["distance"], use_styles=False, handoff=k)["p_pass"]
            for name, arr in sk_p.items():
                p[name] = float(arr[t])
            preds.append({"game": e["game"], "y": e["y"], "p": p, "n_history": len(history),
                          "own_n": "0-4" if len(history) < 5 else "5-14" if len(history) < 15 else "15+",
                          "down": str(e["down"]), "red_zone": e["red_zone"], "third_down": e["third_down"]})
    return preds, folds


def _mc_ll(probs: dict[str, float], y: str) -> float:
    return -math.log(max(EPS, probs.get(y, 0.0)))


def family_eval(examples: list[dict[str, Any]], n_games: int, scheme: str) -> list[dict]:
    names = sorted(examples[0]["x"])
    preds = []
    for gi in range(n_games):
        train_ids = set(_train_ids(scheme, gi, n_games))
        train = [e for e in examples if e["gi"] in train_ids and e["fam"]]
        test = [e for e in examples if e["gi"] == gi and e["fam"]]
        if len(train) < MIN_TRAIN_LABELS or not test:
            continue
        counts = Counter(e["fam"] for e in train)
        league = {f: (counts[f] + 0.5) / (len(train) + 0.5 * len(FAMILIES)) for f in FAMILIES}
        by_down: dict[Any, Counter] = {}
        for e in train:
            by_down.setdefault(e["down"], Counter())[e["fam"]] += 1
        _, X, y = _xy(train, names, "fam")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = {n: make().fit(X, y) for n, make in _sk_models().items() if n != "logistic"}
        _, Xt, _ = _xy(test, names, "fam")
        sk = {}
        for n, m in models.items():
            raw = m.predict_proba(Xt)
            sk[n] = [{f: EPS for f in FAMILIES} | {c: max(EPS, float(row[i])) for i, c in enumerate(m.classes_)}
                     for row in raw]
        for t, e in enumerate(test):
            c = by_down.get(e["down"], Counter())
            nd = sum(c.values())
            situ = {f: (c[f] + 4.0 * league[f]) / (nd + 4.0) for f in FAMILIES}
            hist = e["fam_hist"]
            w = len(hist) / (len(hist) + ot.PRIOR_HANDOFF)
            hc = Counter(hist)
            ind = {f: (hc[f] + 0.5) / (len(hist) + 0.5 * len(FAMILIES)) for f in FAMILIES}
            blend = {f: w * ind[f] + (1 - w) * situ[f] for f in FAMILIES}
            p = {"uniform": {f: 1 / len(FAMILIES) for f in FAMILIES}, "league_rate": league,
                 "league_by_down": situ, "league+in_game_blend": blend}
            for n, rows in sk.items():
                s = sum(rows[t].values())
                p[n] = {f: v / s for f, v in rows[t].items()}
            preds.append({"game": e["game"], "y": e["fam"], "p": p, "n_history": len(hist)})
    return preds


def _boot_diff(preds: list[dict], a: str, b: str, loss: Callable[[dict, str], float]) -> dict[str, Any]:
    by_game: dict[str, list[float]] = {}
    for r in preds:
        by_game.setdefault(r["game"], []).append(loss(r, a) - loss(r, b))
    games = sorted(by_game)
    flat = [d for g in games for d in by_game[g]]
    if len(games) < 2:
        return {"mean_diff": None}
    rng = random.Random(SEED)
    stats = []
    for _ in range(N_BOOT):
        vals = [d for g in (games[rng.randrange(len(games))] for _ in games) for d in by_game[g]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return {"mean_diff": sum(flat) / len(flat), "lo": stats[int(0.025 * N_BOOT)], "hi": stats[int(0.975 * N_BOOT)],
            "games": len(games), "plays": len(flat)}


def _bin_loss(r: dict, m: str) -> float:
    p = min(1 - EPS, max(EPS, r["p"][m]))
    return -(r["y"] * math.log(p) + (1 - r["y"]) * math.log(1 - p))


def _fam_loss(r: dict, m: str) -> float:
    return _mc_ll(r["p"][m], r["y"])


def summarize_binary(preds: list[dict], reference: str) -> dict[str, Any]:
    models = list(preds[0]["p"]) if preds else []
    out: dict[str, Any] = {"plays": len(preds), "games": len({p["game"] for p in preds}), "models": {}}
    for m in models:
        met = he.binary_metrics([p["y"] for p in preds], [p["p"][m] for p in preds])
        acc = sum(int((p["p"][m] >= 0.5) == bool(p["y"])) for p in preds) / len(preds)
        out["models"][m] = {"log_loss": met["log_loss"], "brier": met["brier"], "auc": met["auc"],
                            "ece": met["ece"], "accuracy": acc,
                            "vs_" + reference: _boot_diff(preds, m, reference, _bin_loss)}
    strata: dict[str, Any] = {}
    for key in ("own_n", "down", "red_zone", "third_down"):
        for val in sorted({str(p[key]) for p in preds}):
            sub = [p for p in preds if str(p[key]) == val]
            strata[f"{key}={val}"] = {"n": len(sub), **{
                m: sum(_bin_loss(p, m) for p in sub) / len(sub)
                for m in ("league_situation", "in_game_only", "existing_pooling", "boosted_trees")}}
    out["strata_log_loss"] = strata
    return out


def summarize_family(preds: list[dict], reference: str) -> dict[str, Any]:
    models = list(preds[0]["p"]) if preds else []
    out: dict[str, Any] = {"plays": len(preds), "class_counts": dict(Counter(p["y"] for p in preds)), "models": {}}
    for m in models:
        ll = sum(_fam_loss(p, m) for p in preds) / len(preds)
        acc = sum(int(max(p["p"][m], key=p["p"][m].get) == p["y"]) for p in preds) / len(preds)
        out["models"][m] = {"log_loss": ll, "top1_accuracy": acc,
                            "vs_" + reference: _boot_diff(preds, m, reference, _fam_loss)}
    return out


def latency_probe(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-snap cost of building the context and scoring the trees, CPU."""
    import numpy as np

    names = sorted(examples[0]["x"])
    _, X, y = _xy(examples, names, "y")
    model = _sk_models()["boosted_trees"]().fit(X, y)
    times = []
    for e in examples[:200]:
        t0 = time.perf_counter()
        model.predict_proba(np.array([[e["x"][n] for n in names]]))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"p50_ms": times[len(times) // 2], "p95_ms": times[int(0.95 * len(times))], "max_ms": times[-1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    games = load_games(args.db)
    examples = build_examples(games)
    report: dict[str, Any] = {
        "games": len(games),
        "plays_per_game": [len(g) for g in games],
        "labeled_run_pass": sum(e["y"] is not None for e in examples),
        "labeled_family": sum(e["fam"] is not None for e in examples),
        "distinct_opponents": len({e["opp"] for e in examples}),
        "games_per_opponent": sorted(Counter(e["opp"] for e in {e["opp"] + e["game"]: e for e in examples}.values()).values()),
        "features": sorted(examples[0]["x"]),
    }
    for scheme in ("chronological", "leave_one_opponent_out"):
        rp, folds = run_pass_eval(examples, len(games), scheme)
        report[f"run_pass/{scheme}"] = {"folds": folds, **summarize_binary(rp, "league_situation")}
        report[f"family/{scheme}"] = summarize_family(family_eval(examples, len(games), scheme), "league_by_down")
    report["latency_boosted_trees"] = latency_probe(examples)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "opponent_prediction_eval.json").write_text(json.dumps(report, indent=1, default=str))
    for key in ("run_pass/chronological", "run_pass/leave_one_opponent_out"):
        r = report[key]
        print(f"== {key}: {r['plays']} plays in {r['games']} games")
        for m, v in sorted(r["models"].items(), key=lambda kv: kv[1]["log_loss"]):
            d = v["vs_league_situation"]
            ci = f"[{d['lo']:+.3f},{d['hi']:+.3f}]" if d.get("mean_diff") is not None else ""
            auc = f"{v['auc']:.3f}" if v["auc"] is not None else "n/a"
            print(f"  {m:32s} ll={v['log_loss']:.3f} acc={v['accuracy']:.3f} auc={auc} "
                  f"vs league_situation {d.get('mean_diff') or 0:+.3f} {ci}")
    for key in ("family/chronological", "family/leave_one_opponent_out"):
        r = report[key]
        print(f"== {key}: {r['plays']} plays {r['class_counts']}")
        for m, v in sorted(r["models"].items(), key=lambda kv: kv[1]["log_loss"]):
            d = v["vs_league_by_down"]
            ci = f"[{d['lo']:+.3f},{d['hi']:+.3f}]" if d.get("mean_diff") is not None else ""
            print(f"  {m:24s} ll={v['log_loss']:.3f} top1={v['top1_accuracy']:.3f} "
                  f"vs league_by_down {d.get('mean_diff') or 0:+.3f} {ci}")
    print("latency", report["latency_boosted_trees"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
