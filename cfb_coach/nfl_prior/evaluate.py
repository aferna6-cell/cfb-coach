"""Season-separated comparison of the NFL prior models, with held-out teams.

Splits
- ``season``: train on ``train_seasons``, early-stop on ``val_season``, test on
  ``test_seasons`` (all teams).
- ``team_holdout``: same seasons, but plays by ``n_holdout_teams`` offenses are
  removed from training/validation and are the only test plays.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cfb_coach.nfl_prior import PRIOR_VERSION
from cfb_coach.nfl_prior import data as data_mod
from cfb_coach.nfl_prior.models import MLPModel, TableBaseline, XGBModel

TASKS: dict[str, dict[str, Any]] = {
    "run_pass": {"task": "binary", "target": "is_pass", "features": data_mod.SITUATION_FEATURES, "play_type": False,
                 "question": "Will the offense call a pass (dropback) or a run, given only the situation?"},
    "success": {"task": "binary", "target": "success", "features": data_mod.OUTCOME_FEATURES, "play_type": True,
                "question": "Given the situation and run/pass, will the play gain positive EPA?"},
    "epa": {"task": "regression", "target": "epa", "features": data_mod.OUTCOME_FEATURES, "play_type": True,
            "question": "Given the situation and run/pass, what EPA should we expect?"},
}
FORBIDDEN_LIVE_ROOT = Path("~/.cfb-coach")


def default_config() -> dict[str, Any]:
    return {
        "train_seasons": list(range(2016, 2023)),
        "val_season": 2023,
        "test_seasons": [2024, 2025],
        "season_types": ["REG", "POST"],
        "n_holdout_teams": 8,
        "seed": 20261106,
        "bootstrap": 300,
        "threads": 8,
        "quick": False,
    }


# ---------------------------------------------------------------- metrics


def binary_metrics(y: Any, p: Any, bins: int = 10) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "observed": float(y.mean()),
        "mean_pred": float(p.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
    }
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        gap = abs(p[m].mean() - y[m].mean())
        ece += m.mean() * gap
        table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                      "mean_pred": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4)})
    out["ece"] = float(ece)
    out["calibration"] = table
    return out


def regression_metrics(y: Any, p: Any) -> dict[str, Any]:
    import numpy as np

    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    resid = y - p
    var = float(((y - y.mean()) ** 2).mean())
    return {
        "n": int(len(y)),
        "observed_mean": float(y.mean()),
        "mean_pred": float(p.mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(np.abs(resid).mean()),
        "r2": float(1 - (resid ** 2).mean() / var) if var > 0 else None,
    }


def row_loss(task: str, y: Any, p: Any) -> Any:
    import numpy as np

    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if task == "binary":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (y - p) ** 2


def paired_game_bootstrap(games: Any, diff: Any, n_boot: int, seed: int) -> dict[str, Any]:
    """Mean per-play loss difference with a game-clustered 95% interval."""
    import numpy as np
    import pandas as pd

    g = pd.DataFrame({"g": games, "d": diff}).groupby("g")["d"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(sums), len(sums))
        stats[i] = sums[pick].sum() / counts[pick].sum()
    return {"games": int(len(sums)), "mean_diff": float(sums.sum() / counts.sum()),
            "lo": float(np.quantile(stats, 0.025)), "hi": float(np.quantile(stats, 0.975))}


def slices(df: Any) -> dict[str, Any]:
    return {
        "1st_down": df["down"] == 1,
        "2nd_down": df["down"] == 2,
        "3rd_short_(1-2)": (df["down"] == 3) & (df["ydstogo"] <= 2),
        "3rd_long_(7+)": (df["down"] == 3) & (df["ydstogo"] >= 7),
        "4th_down": df["down"] == 4,
        "red_zone": df["yardline_100"] <= 20,
        "two_minute": df["two_minute"] > 0,
        "trailing_9+": df["score_differential"] <= -9,
        "leading_9+": df["score_differential"] >= 9,
    }


# ---------------------------------------------------------------- splits


def holdout_teams(train: Any, n: int, seed: int) -> list[str]:
    import numpy as np

    teams = sorted(str(t) for t in train["posteam"].dropna().unique())
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(teams, size=min(n, len(teams)), replace=False).tolist())


def make_splits(df: Any, cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    season_ok = df["season_type"].isin(cfg["season_types"])
    train = df[season_ok & df["season"].isin(cfg["train_seasons"])]
    val = df[season_ok & (df["season"] == cfg["val_season"])]
    test = df[season_ok & df["season"].isin(cfg["test_seasons"])]
    held = holdout_teams(train, cfg["n_holdout_teams"], cfg["seed"])
    return {
        "season": {"train": train, "val": val, "test": test, "holdout_teams": []},
        "team_holdout": {
            "train": train[~train["posteam"].isin(held)],
            "val": val[~val["posteam"].isin(held)],
            "test": test[test["posteam"].isin(held)],
            "holdout_teams": held,
        },
    }


# ---------------------------------------------------------------- driver


def build_models(spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[Any]:
    return [
        TableBaseline(spec["task"], spec["target"], use_play_type=spec["play_type"]),
        XGBModel(spec["task"], spec["target"], spec["features"], seed=cfg["seed"], n_jobs=cfg["threads"], quick=cfg["quick"]),
        MLPModel(spec["task"], spec["target"], spec["features"], seed=cfg["seed"], threads=cfg["threads"], quick=cfg["quick"]),
    ]


def latency(model: Any, test: Any, n: int = 300) -> dict[str, float]:
    import numpy as np

    sample = test.iloc[: min(n, len(test))]
    single = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t0 = time.perf_counter()
        model.predict(row)
        single.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    model.predict(test)
    batch_us = (time.perf_counter() - t0) * 1e6 / max(1, len(test))
    return {"single_row_ms_p50": float(np.percentile(single, 50)),
            "single_row_ms_p95": float(np.percentile(single, 95)),
            "batch_us_per_row": float(batch_us)}


def run_task(name: str, spec: Mapping[str, Any], splits: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any],
             model_dir: Path | None, log: Callable[[str], None] = print) -> dict[str, Any]:
    result: dict[str, Any] = {"question": spec["question"], "task": spec["task"], "target": spec["target"],
                              "features": list(spec["features"]), "splits": {}}
    metric_fn = binary_metrics if spec["task"] == "binary" else regression_metrics
    for split_name, sp in splits.items():
        train, val, test = sp["train"], sp["val"], sp["test"]
        block: dict[str, Any] = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
                                 "holdout_teams": sp["holdout_teams"], "models": {}, "slices": {}, "vs_best": {}}
        preds: dict[str, Any] = {}
        for model in build_models(spec, cfg):
            t0 = time.perf_counter()
            model.fit(train, val)
            fit_s = time.perf_counter() - t0
            p = model.predict(test)
            preds[model.name] = p
            m = metric_fn(test[spec["target"]].to_numpy(), p)
            m["train_seconds"] = fit_s
            m.update(latency(model, test))
            if isinstance(model, MLPModel):
                m["epochs"] = model.epochs_run
            if isinstance(model, XGBModel):
                m["trees"] = int(model.model.best_iteration) + 1
            block["models"][model.name] = m
            log(f"[{name}/{split_name}] {model.name}: fit {fit_s:.1f}s")
            if model_dir is not None and split_name == "season":
                model.save(model_dir / name)
        key = "log_loss" if spec["task"] == "binary" else "rmse"
        ranked = sorted(block["models"], key=lambda k: block["models"][k][key])
        best = ranked[0]
        y = test[spec["target"]].to_numpy()
        for other in ranked[1:]:
            diff = row_loss(spec["task"], y, preds[other]) - row_loss(spec["task"], y, preds[best])
            block["vs_best"][other] = paired_game_bootstrap(test["game_id"].to_numpy(), diff, cfg["bootstrap"], cfg["seed"])
        block["winner"] = best
        block["winner_metric"] = key
        block["winner_separated"] = all(v["lo"] > 0 for v in block["vs_best"].values())
        for slice_name, mask in slices(test).items():
            mask = mask.to_numpy()
            if mask.sum() < 50:
                continue
            block["slices"][slice_name] = {
                mname: {k: v for k, v in metric_fn(y[mask], p[mask]).items() if k != "calibration"}
                for mname, p in preds.items()
            }
        result["splits"][split_name] = block
    return result


def run(cfg: Mapping[str, Any], out_dir: str | Path, cache_dir: str | Path = data_mod.DEFAULT_CACHE,
        frame: Any = None, log: Callable[[str], None] = print) -> Path:
    """Run every task; write metadata, metrics, report and models under a fresh run dir."""
    out_root = Path(out_dir).expanduser().resolve()
    live = FORBIDDEN_LIVE_ROOT.expanduser().resolve()
    if out_root == live or live in out_root.parents:
        raise ValueError(f"Refusing to write research artifacts under the live data dir {live}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"run-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    seasons = sorted(set(cfg["train_seasons"]) | {cfg["val_season"]} | set(cfg["test_seasons"]))
    raw = frame if frame is not None else data_mod.load_seasons(seasons, cache_dir)
    df = data_mod.prepare(raw)
    splits = make_splits(df, cfg)
    metadata = {
        "prior_version": PRIOR_VERSION,
        "created_utc": stamp,
        "config": dict(cfg),
        "rows_by_season": {str(k): int(v) for k, v in df.groupby("season").size().items()},
        "data_fingerprint": data_mod.frame_fingerprint(df, ("game_id", "play_id", "is_pass", "epa")),
        "features": {"situation": list(data_mod.SITUATION_FEATURES), "outcome": list(data_mod.OUTCOME_FEATURES)},
        "excluded_post_snap": sorted(data_mod.POST_SNAP_COLUMNS),
        "holdout_teams": splits["team_holdout"]["holdout_teams"],
        "versions": _versions(),
        "promotion": "offline research only; not loaded by any live code",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics = {name: run_task(name, spec, splits, cfg, run_dir / "models", log) for name, spec in TASKS.items()}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(metadata, metrics), encoding="utf-8")
    return run_dir


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "torch", "nflreadpy", "polars"):
        try:
            out[mod] = str(__import__(mod).__version__)
        except Exception:  # noqa: BLE001
            out[mod] = "unavailable"
    return out


# ---------------------------------------------------------------- report


def _f(v: Any, d: int = 4) -> str:
    return "—" if v is None else f"{v:.{d}f}" if isinstance(v, float) else str(v)


def render_report(meta: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    cfg = meta["config"]
    lines = [
        "# NFL prior prototype — model comparison", "",
        f"`{meta['prior_version']}`, run {meta['created_utc']}. Train {cfg['train_seasons'][0]}–{cfg['train_seasons'][-1]}, "
        f"validate {cfg['val_season']}, test {', '.join(map(str, cfg['test_seasons']))}. Seed {cfg['seed']}. "
        f"Held-out offenses: {', '.join(meta['holdout_teams'])}. Data fingerprint `{meta['data_fingerprint']}`.",
        "", f"Rows by season: {meta['rows_by_season']}.",
        "", "Pre-snap features only; formation (shotgun/no-huddle) is excluded. Offline research only.",
        "`Δ vs winner` is how much worse each model is per play (log loss, or squared error for EPA), "
        "with a game-clustered bootstrap 95% interval; an interval above 0 means the winner is clearly better.",
    ]
    for name, res in metrics.items():
        lines += ["", f"## {name}: {res['question']}"]
        for split, block in res["splits"].items():
            lines += ["", f"### {split} split — train {block['n_train']:,}, val {block['n_val']:,}, test {block['n_test']:,}", ""]
            if res["task"] == "binary":
                lines += ["| model | log loss | Brier | AUC | ECE | train s | 1-row p95 ms | Δ log loss vs winner [95% CI] |",
                          "|---|---|---|---|---|---|---|---|"]
            else:
                lines += ["| model | RMSE | MAE | R² | train s | 1-row p95 ms | Δ squared error vs winner [95% CI] |",
                          "|---|---|---|---|---|---|---|"]
            for m, v in block["models"].items():
                vs = block["vs_best"].get(m)
                delta = "winner" if vs is None else f"{_f(vs['mean_diff'])} [{_f(vs['lo'])}, {_f(vs['hi'])}]"
                if res["task"] == "binary":
                    lines.append(f"| {m} | {_f(v['log_loss'])} | {_f(v['brier'])} | {_f(v['auc'], 3)} | {_f(v['ece'], 3)} | "
                                 f"{_f(v['train_seconds'], 1)} | {_f(v['single_row_ms_p95'], 2)} | {delta} |")
                else:
                    lines.append(f"| {m} | {_f(v['rmse'])} | {_f(v['mae'])} | {_f(v['r2'], 3)} | "
                                 f"{_f(v['train_seconds'], 1)} | {_f(v['single_row_ms_p95'], 2)} | {delta} |")
            lines += ["", f"Winner: **{block['winner']}** ({block['winner_metric']}); "
                      + ("clearly separated from every other model." if block["winner_separated"]
                         else "not clearly separated from at least one other model.")]
            if split == "season":
                key = "log_loss" if res["task"] == "binary" else "rmse"
                lines += ["", f"Slices ({key}):", "", "| slice | n | " + " | ".join(block["models"]) + " |",
                          "|---|---|" + "---|" * len(block["models"])]
                for s, ms in block["slices"].items():
                    n = next(iter(ms.values()))["n"]
                    lines.append(f"| {s} | {n:,} | " + " | ".join(_f(ms[m][key]) for m in block["models"]) + " |")
    lines += [
        "", "## Caveats", "",
        "- EPA and success come from nflfastR's expected-points model, which is itself fit on historical seasons.",
        "- EPA already subtracts the expected points of the situation, so situation features can barely "
        "predict it (R² near 0 is expected); the useful signal is the run/pass difference by situation.",
        "- NFL tendencies are a prior for Madden, not ground truth: game mechanics and player behaviour differ.",
        "- Nothing here is connected to live play-calling.",
    ]
    return "\n".join(lines) + "\n"
