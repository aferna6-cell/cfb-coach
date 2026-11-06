"""Score an offline NFL XGBoost prior against the Madden models on held-out Madden games.

Research only; needs xgboost and pandas (``pip install -e .[nfl-research]``).
The Madden database is opened read-only and the NFL models are loaded from a
``python -m cfb_coach.nfl_prior`` run directory. Nothing here is imported by
live code, and results are written outside the live data directory.

Offense (our verified plays, CPU and human games reported separately):
  nfl_xgb_raw         NFL P(EPA > 0 | situation, run/pass) with no Madden data.
  nfl_xgb_calibrated  The same score, logistic-recalibrated on the fold's training games.
  nfl_prior_table     Madden down/distance -> field shrinkage table whose root is the
                      calibrated NFL score instead of one global rate.
  experimental+nfl    Equal-weight logit average of the experimental model and
                      nfl_xgb_calibrated (no weights fit on test games).
Opponent run/pass (our defensive snaps, human only):
  nfl_xgb_raw / nfl_xgb_calibrated next to the existing pooled-prior variants.

Madden snaps carry down, distance, yardline and (usually) quarter only. Clock,
score and timeouts are imputed at neutral values; this is reported.

    python scripts/madden_ml/nfl_prior_heldout_eval.py --db AUDIT_COPY.db \
        --nfl-run ~/.cfb-research/nfl/runs/RUN --out ~/.cfb-research/madden-nfl-eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.nfl_prior.data import OUTCOME_FEATURES, SITUATION_FEATURES  # noqa: E402

EPS = 1e-6
TARGETS = ("success", "conversion")
NFL_KEY = {"success": "_nfl_success", "conversion": "_nfl_success"}
IMPUTED = {
    "score_differential": 0.0, "posteam_timeouts_remaining": 3.0, "defteam_timeouts_remaining": 3.0,
    "two_minute": 0.0, "is_home": "averaged over 0 and 1",
    "half_seconds_remaining / game_seconds_remaining": "middle of the logged quarter",
    "qtr": "2 when not logged",
}


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, float(p)))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _yardline_100(row: Mapping[str, Any]) -> float:
    yl = he._int(row.get("yardline"))
    if yl is None:
        return 70.0
    # Offense logs yards from our own goal; defense logs yards from the goal we defend.
    to_goal = yl if row.get("side") == "defense" else 100 - yl
    return float(min(99, max(1, to_goal)))


def nfl_frame(rows: Sequence[Mapping[str, Any]], *, is_home: float, is_pass: Sequence[float] | None) -> Any:
    import numpy as np
    import pandas as pd

    recs = []
    for r in rows:
        down = he._int(r.get("down")) or 1
        dist = float(min(30, max(0, he._int(r.get("distance")) or 10)))
        q = he._int(r.get("quarter"))
        q = q if q in (1, 2, 3, 4) else 2
        yl100 = _yardline_100(r)
        recs.append({
            **{f"down_{d}": float(down == d) for d in (1, 2, 3, 4)},
            "ydstogo": dist, "log_ydstogo": float(np.log1p(dist)), "yardline_100": yl100,
            "goal_to_go": float(dist >= yl100), "qtr": float(q),
            "half_seconds_remaining": 1350.0 if q in (1, 3) else 450.0,
            "game_seconds_remaining": float((4 - q) * 900 + 450),
            "score_differential": 0.0, "posteam_timeouts_remaining": 3.0,
            "defteam_timeouts_remaining": 3.0, "is_home": is_home, "two_minute": 0.0,
        })
    df = pd.DataFrame.from_records(recs, columns=list(SITUATION_FEATURES))
    if is_pass is not None:
        df["is_pass"] = np.asarray(is_pass, dtype="float32")
    return df.astype("float32")


def load_xgb(path: Path) -> Any:
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def _predict(model: Any, rows: Sequence[Mapping[str, Any]], features: Sequence[str], is_pass: Sequence[float] | None) -> Any:
    out = 0.0
    for home in (0.0, 1.0):
        df = nfl_frame(rows, is_home=home, is_pass=is_pass)
        out = out + 0.5 * model.predict_proba(df[list(features)])[:, 1]
    return out


# The only play information the NFL models take is run vs pass. Concept,
# formation, play action and RPO reads have no NFL counterpart here.
NFL_PLAY_INPUT = {"run": "run", "pass": "pass", "screen": "pass", "unknown": "mixture"}


def nfl_play_input(play: str | None) -> str:
    """Madden play -> NFL model input: 'run', 'pass' or 'mixture' (weighted by NFL P(pass))."""
    return NFL_PLAY_INPUT.get(he.run_or_pass(play), "mixture")


def attach_nfl_scores(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Add ``_nfl_success`` (offense rows) and ``_nfl_pass`` (all rows) in place."""
    import numpy as np

    succ = load_xgb(run_dir / "models" / "success" / "xgboost.json")
    rp = load_xgb(run_dir / "models" / "run_pass" / "xgboost.json")
    p_pass = _predict(rp, rows, SITUATION_FEATURES, None)
    p_run = _predict(succ, rows, OUTCOME_FEATURES, np.zeros(len(rows)))
    p_throw = _predict(succ, rows, OUTCOME_FEATURES, np.ones(len(rows)))
    counts: dict[str, int] = {}
    mapped: dict[str, int] = {}
    for r, pp, pr, pt in zip(rows, p_pass, p_run, p_throw):
        r["_nfl_pass"] = float(pp)
        if r.get("side") != "offense":
            continue
        fam = he.run_or_pass(r.get("action_play"))
        target = nfl_play_input(r.get("action_play"))
        counts[fam] = counts.get(fam, 0) + 1
        mapped[target] = mapped.get(target, 0) + 1
        if target == "run":
            r["_nfl_success"] = float(pr)
        elif target == "pass":
            r["_nfl_success"] = float(pt)
        else:
            r["_nfl_success"] = float(pp * pt + (1 - pp) * pr)
    return {"offense_play_family_counts": counts, "nfl_input_counts": mapped, "mapping": dict(NFL_PLAY_INPUT)}


def situation_slices(row: Mapping[str, Any]) -> dict[str, str]:
    dist = he._int(row.get("distance"))
    return {
        "short_yardage": "yes" if dist is not None and dist <= 2 else "no",
        "red_zone": "yes" if _yardline_100(row) <= 20 and row.get("yardline") is not None else "no",
    }


def fit_platt(xs: Sequence[float], ys: Sequence[int], l2: float = 1.0, iters: int = 50) -> tuple[float, float]:
    """Logistic y ~ a + b * x by Newton steps, ridge-penalised toward a=0, b=1."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        ga, gb = l2 * a, l2 * (b - 1.0)
        haa, hab, hbb = l2, 0.0, l2
        for x, y in zip(xs, ys):
            p = _sigmoid(a + b * x)
            w = p * (1 - p)
            ga += p - y
            gb += (p - y) * x
            haa += w
            hab += w * x
            hbb += w * x * x
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-8:
            break
    return a, b


class NFLRaw:
    name = "nfl_xgb_raw"
    targets = TARGETS

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        pass

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        return {t: float(row[NFL_KEY[t]]) for t in self.targets}


class NFLCalibrated:
    name = "nfl_xgb_calibrated"
    targets = TARGETS

    def __init__(self) -> None:
        self.coef: dict[str, tuple[float, float]] = {}

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        for t in self.targets:
            pairs = [(_logit(r[NFL_KEY[t]]), y) for r in train_rows if he.is_verified_offense(r)
                     for y in [he.offense_label(r, t)] if y is not None]
            self.coef[t] = fit_platt([x for x, _ in pairs], [y for _, y in pairs])

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        return {t: _sigmoid(a + b * _logit(row[NFL_KEY[t]])) for t, (a, b) in self.coef.items()}


class NFLPriorTable:
    """Madden shrinkage table rooted at the calibrated NFL score for each play."""

    name = "nfl_prior_table"
    targets = TARGETS

    def __init__(self, strength: float = 10.0) -> None:
        self.cal = NFLCalibrated()
        self.strength = strength
        self.keys = (he.dd_key, he.field_key)
        self.cells: dict[str, list[dict[tuple[str, ...], list[float]]]] = {}

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        self.cal.fit(train_rows)
        for t in self.targets:
            levels: list[dict[tuple[str, ...], list[float]]] = [{} for _ in self.keys]
            for r in train_rows:
                if not he.is_verified_offense(r):
                    continue
                y = he.offense_label(r, t)
                if y is None:
                    continue
                prior = self.cal.predict(r)[t]
                path: list[str] = []
                for lvl, key in enumerate(self.keys):
                    path.append(key(r))
                    cell = levels[lvl].setdefault(tuple(path), [0.0, 0.0, 0.0])
                    cell[0] += y
                    cell[1] += 1
                    cell[2] += prior
            self.cells[t] = levels

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        base = self.cal.predict(row)
        out = {}
        for t in self.targets:
            p = base[t]
            path: list[str] = []
            for lvl, key in enumerate(self.keys):
                path.append(key(row))
                cell = self.cells[t][lvl].get(tuple(path))
                if cell:
                    # Residual of the cell over what the NFL prior expected there, shrunk.
                    lift = (cell[0] - cell[2]) / (cell[1] + self.strength)
                    p = min(1 - EPS, max(EPS, p + lift))
            out[t] = p
        return out


class ExperimentalPlusNFL:
    name = "experimental+nfl"
    targets = ("success",)

    def __init__(self) -> None:
        self.exp = he.ExperimentalSuccess()
        self.cal = NFLCalibrated()

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        self.exp.fit(train_rows)
        self.cal.fit(train_rows)

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        z = 0.5 * _logit(self.exp.predict(row)["success"]) + 0.5 * _logit(self.cal.predict(row)["success"])
        return {"success": _sigmoid(z)}


PRIOR_STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0)


def blend(p_exp: float, p_nfl: float, base: float, w: float) -> float:
    """Experimental logit plus ``w`` times how far the NFL prior sits from the base rate."""
    return _sigmoid(_logit(p_exp) + w * (_logit(p_nfl) - _logit(base)))


class ExperimentalNFLTuned:
    """Experimental model nudged by the calibrated NFL prior, strength chosen on validation games.

    The latest quarter of the fold's training games (at least one) is the
    validation set; the test game is never seen. Ties go to the weaker prior.
    """

    name = "experimental+nfl_tuned"
    targets = ("success",)

    def __init__(self, game_order: Mapping[str, int], choices: list[float] | None = None) -> None:
        self.game_order = game_order
        self.choices = choices
        self.w = 0.0

    @staticmethod
    def _parts(train: Sequence[Mapping[str, Any]]) -> tuple[Any, NFLCalibrated, float]:
        exp, cal = he.ExperimentalSuccess(), NFLCalibrated()
        exp.fit(train)
        cal.fit(train)
        ys = [y for r in train if he.is_verified_offense(r) for y in [he.offense_label(r, "success")] if y is not None]
        return exp, cal, (sum(ys) + 1) / (len(ys) + 2)

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        games = sorted({str(r["game_id"]) for r in train_rows if r.get("game_id")},
                       key=lambda g: self.game_order.get(g, 0))
        val_games = set(games[-max(1, len(games) // 4):]) if len(games) >= 2 else set()
        inner = [r for r in train_rows if str(r.get("game_id")) not in val_games]
        val = [(r, y) for r in train_rows if str(r.get("game_id")) in val_games and he.is_verified_offense(r)
               for y in [he.offense_label(r, "success")] if y is not None]
        if val and any(he.is_verified_offense(r) for r in inner):
            exp, cal, base = self._parts(inner)
            scored = [(exp.predict(r)["success"], cal.predict(r)["success"], y) for r, y in val]
            best = None
            for w in PRIOR_STRENGTHS:
                ll = sum(he._row_ll(y, blend(pe, pn, base, w)) for pe, pn, y in scored) / len(scored)
                if best is None or ll < best[0] - 1e-12:
                    best = (ll, w)
            self.w = best[1]
        if self.choices is not None:
            self.choices.append(self.w)
        self.exp, self.cal, self.base = self._parts(train_rows)

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        return {"success": blend(self.exp.predict(row)["success"], self.cal.predict(row)["success"], self.base, self.w)}


def offense_models(game_order: Mapping[str, int] | None = None, choices: list[float] | None = None) -> list[Any]:
    keep = he.is_verified_offense
    return [
        he.BaseRate(TARGETS, he.offense_label, keep),
        he.ShrunkTable("situation_table", TARGETS, he.offense_label, keep, [he.dd_key, he.field_key]),
        he.ExperimentalSuccess(),
        NFLRaw(), NFLCalibrated(), NFLPriorTable(), ExperimentalPlusNFL(),
        ExperimentalNFLTuned(game_order or {}, choices),
    ]


def offense_block(rows: list[dict[str, Any]], game_order: Mapping[str, int], *, mode: str,
                  n_boot: int, seed: int) -> dict[str, Any]:
    folds = he.make_folds(rows, game_order, is_test_row=he.is_verified_offense, mode=mode)
    choices: list[float] = []
    preds, timing = he.run_folds(rows, folds, lambda: offense_models(game_order, choices), targets=TARGETS,
                                 label_fn=he.offense_label, is_test_row=he.is_verified_offense,
                                 extra=situation_slices)
    block: dict[str, Any] = {"folds": len(folds), "timing": timing, "subsets": {},
                             "tuned_prior_strength_per_fold": choices}
    for name, keep in (("all", lambda p: True), ("cpu", lambda p: p["opponent_type"] == "cpu"),
                       ("human", lambda p: p["opponent_type"] != "cpu")):
        sub = [p for p in preds if keep(p)]
        if not sub:
            continue
        block["subsets"][name] = {
            "vs_base_rate": he.summarize_predictions(sub, TARGETS, reference="base_rate", n_boot=n_boot,
                                                     seed=seed, strata=("basis", "short_yardage", "red_zone")),
            "vs_experimental": he.summarize_predictions(
                [p for p in sub if p["target"] == "success"], ("success",),
                reference="experimental_shrinkage", n_boot=n_boot, seed=seed, strata=()),
        }
    return block


def opponent_block(rows: list[dict[str, Any]], game_order: Mapping[str, int], *, scheme: str,
                   n_boot: int, seed: int) -> dict[str, Any]:
    preds, folds = he.opponent_sequential_eval(rows, game_order, scheme=scheme)
    labelled: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if he.opponent_pass_label(r) is not None and r.get("game_id") and r.get("opponent_id"):
            labelled.setdefault(str(r["game_id"]), []).append(r)
    scored_games = [f["game"] for f in folds]
    by_game_preds: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        by_game_preds.setdefault(p["game"], []).append(p)
    opp_of = {g: str(labelled[g][0]["opponent_id"]) for g in labelled}
    for g in scored_games:
        earlier = [h for h in labelled if h in game_order and game_order[h] < game_order[g]]
        others = earlier if scheme == "chronological" else [h for h in labelled if h in game_order]
        train = [r for h in others if opp_of[h] != opp_of[g] for r in labelled[h]]
        a, b = fit_platt([_logit(r["_nfl_pass"]) for r in train], [he.opponent_pass_label(r) for r in train])
        for p, r in zip(by_game_preds[g], labelled[g]):
            assert p["y"] == he.opponent_pass_label(r)
            p["p"]["nfl_xgb_raw"] = r["_nfl_pass"]
            p["p"]["nfl_xgb_calibrated"] = _sigmoid(a + b * _logit(r["_nfl_pass"]))
    return {
        "folds": len(folds),
        "metrics": he.summarize_predictions(preds, ("opp_pass",), reference="league_situation",
                                            n_boot=n_boot, seed=seed, strata=("own_n",)),
    }


def _fmt(v: Any, d: int = 3) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def render(report: Mapping[str, Any]) -> str:
    lines = [
        "# NFL XGBoost prior vs Madden models on held-out Madden games",
        "",
        f"Generated {report['generated']} from `{report['nfl_run']}`. Research only; nothing is wired to live calls.",
        "",
        f"Offense play families used to pick the NFL run/pass input: {report['nfl_inputs']['offense_play_family_counts']}.",
        "Imputed NFL features (not logged in Madden): " + "; ".join(f"{k}: {v}" for k, v in report["imputed"].items()) + ".",
        "",
    ]
    lines += [f"NFL input mapping: {report['nfl_inputs'].get('mapping')}; offense rows per input: "
              f"{report['nfl_inputs'].get('nfl_input_counts')}.", ""]
    for mode, block in report["offense"].items():
        lines += [f"## Offense success — {mode} folds ({block['folds']} test games)", "",
                  f"Prior strength chosen on validation games, per fold: {block.get('tuned_prior_strength_per_fold')}.", ""]
        for subset, sb in block["subsets"].items():
            for target in TARGETS:
                m = sb["vs_base_rate"].get(target)
                if not m:
                    continue
                lines += [f"### {subset} games — {target} (plays scored: {next(iter(m['models'].values()))['n']})", "",
                          "| model | log loss | Brier | AUC | ECE | Δ log loss vs base rate [95% CI] | Δ vs experimental [95% CI] |",
                          "|---|---|---|---|---|---|---|"]
                vx = sb["vs_experimental"].get("success", {}).get("vs_reference", {}) if target == "success" else {}
                for name, mm in sorted(m["models"].items(), key=lambda kv: kv[1]["log_loss"]):
                    vb = m["vs_reference"].get(name)
                    vb_s = "reference" if vb is None else f"{_fmt(vb['mean_diff'], 4)} [{_fmt(vb['lo'], 4)}, {_fmt(vb['hi'], 4)}]"
                    ve = vx.get(name)
                    ve_s = ("reference" if name == "experimental_shrinkage" and target == "success" else
                            "—" if ve is None else f"{_fmt(ve['mean_diff'], 4)} [{_fmt(ve['lo'], 4)}, {_fmt(ve['hi'], 4)}]")
                    lines.append(f"| {name} | {_fmt(mm['log_loss'], 4)} | {_fmt(mm['brier'], 4)} | {_fmt(mm.get('auc'))} "
                                 f"| {_fmt(mm.get('ece'))} | {vb_s} | {ve_s} |")
                lines.append("")
                if target == "success" and m.get("strata"):
                    names = ["base_rate", "experimental_shrinkage", "nfl_xgb_calibrated", "experimental+nfl",
                             "experimental+nfl_tuned"]
                    lines += ["Log loss by slice (n):", "", "| slice | " + " | ".join(names) + " |",
                              "|---|" + "---|" * len(names)]
                    for stratum, ms in m["strata"].items():
                        n = next((v.get("n") for v in ms.values() if v.get("n")), 0)
                        if not n:
                            continue
                        lines.append(f"| {stratum} ({n}) | " + " | ".join(_fmt(ms.get(k, {}).get("log_loss"), 4) for k in names) + " |")
                    lines.append("")
    for scheme, block in report["opponent_run_pass"].items():
        m = block["metrics"]["opp_pass"]
        lines += [f"## Opponent run/pass (human, defensive snaps) — {scheme} ({m['test_games']} test games)", "",
                  "| model | log loss | Brier | AUC | Δ vs league_situation [95% CI] |", "|---|---|---|---|---|"]
        for name, mm in sorted(m["models"].items(), key=lambda kv: kv[1]["log_loss"]):
            if "@k" in name:
                continue
            v = m["vs_reference"].get(name)
            v_s = "reference" if v is None else f"{_fmt(v['mean_diff'], 4)} [{_fmt(v['lo'], 4)}, {_fmt(v['hi'], 4)}]"
            lines.append(f"| {name} | {_fmt(mm['log_loss'], 4)} | {_fmt(mm['brier'], 4)} | {_fmt(mm.get('auc'))} | {v_s} |")
        lines.append("")
    lines += ["Negative Δ = lower log loss than the reference. Intervals resample whole games."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--nfl-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=he.DEFAULT_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=he.DEFAULT_SEED)
    args = ap.parse_args(argv)

    from cfb_coach.games import data_dir

    out_root = Path(args.out).expanduser().resolve()
    live = Path(data_dir()).expanduser().resolve()
    if out_root == live or live in out_root.parents:
        raise SystemExit(f"refusing to write inside the live data directory {live}")
    run_dir = Path(args.nfl_run).expanduser().resolve()

    data = he.load_eval_data(args.db)
    rows = [dict(r) for r in data.rows]
    inputs = attach_nfl_scores(rows, run_dir)
    report: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nfl_run": str(run_dir), "nfl_inputs": inputs, "imputed": IMPUTED,
        "config": {"bootstrap": args.bootstrap, "seed": args.seed},
        "offense": {}, "opponent_run_pass": {},
    }
    for mode in ("chronological", "exclude_opponent"):
        report["offense"][mode] = offense_block(rows, data.game_order, mode=mode, n_boot=args.bootstrap, seed=args.seed)
    for scheme in ("chronological", "leave_one_opponent_out"):
        report["opponent_run_pass"][scheme] = opponent_block(rows, data.game_order, scheme=scheme,
                                                             n_boot=args.bootstrap, seed=args.seed)
    out = out_root / f"run-{report['generated'].replace('-', '').replace(':', '')}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(report), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
