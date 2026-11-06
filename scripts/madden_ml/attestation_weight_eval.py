"""Does admitting user-attested human-game plays improve out-of-sample predictions?

Research only; reads the database read-only and writes outside the live data
directory. Nothing is promoted. Every human game is held out in turn
(leave-one-game-out); any tuning happens inside the training games.

Defense (``offense_won`` from the defensive snap, logistic v2 design):
  No human defensive play is independently verified, so a sample weight on
  attested rows alone would only rescale the data against the penalty. The
  weight is applied where attestation matters, to the play-specific part:
  ``logit = situation_only + w * (situation_and_call - situation_only)``.
  ``w = 0`` uses no play identity at all, so it needs no execution evidence.
Offense (experimental shrinkage model, refit per fold on all other games):
  Human attested plays enter training with sample weight ``w``; CPU rows are
  unchanged. ``status_quo`` is today's dataset (standing-toggle plays at 1,
  retrospectively attested plays only as situation context).

Test rows are attested plays of the held-out game scored on the displayed
call, so the test labels themselves rest on the attestation; independently
verified human plays (none today) would be reported separately.

    python scripts/madden_ml/attestation_weight_eval.py --db AUDIT_COPY.db \
        --defense-template DEFENSE_V2_COPY.json --defense-current DEFENSE_V1_COPY.json --out DIR
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

import numpy as np  # noqa: E402

from cfb_coach.madden.model import attestation_audit as aa  # noqa: E402
from cfb_coach.madden.model import defense_logistic_train as dlt  # noqa: E402
from cfb_coach.madden.model import defense_v1  # noqa: E402
from cfb_coach.madden.model import experimental_model as em  # noqa: E402
from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.madden.situation import parse_madden_situation  # noqa: E402

WEIGHTS = (0.0, 0.5, 1.0)
EPS = 1e-6


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


# ---------------------------------------------------------------- defense


def defense_rows(conn, audit: Sequence[aa.AuditRow]) -> tuple[list[dlt.SnapRow], dict[int, aa.AuditRow]]:
    started = {r["session_id"]: str(r["started_ts"] or "") for r in conn.execute("SELECT session_id, started_ts FROM game_sessions")}
    rows, meta = [], {}
    for a in audit:
        if a.side != "defense" or a.tier == aa.TIER_EXCLUDED:
            continue
        rows.append(dlt.SnapRow(
            snap_id=a.snap_id, session_id=a.session_id, opponent=a.opponent_id,
            started_ts=started.get(a.session_id, ""),
            situation=parse_madden_situation(a.situation_raw or "", default_side="defense"),
            formation=str(a.executed_formation), play=str(a.executed_play),
            offense_success=bool(a.offense_success),
            provenance=dlt.PROVENANCE_VERIFIED if a.tier == aa.TIER_VERIFIED else dlt.PROVENANCE_ATTESTED,
            admitted=True,
        ))
        meta[a.snap_id] = a
    rows.sort(key=lambda r: (r.started_ts, r.snap_id))
    return rows, meta


class SituationOnly:
    def __init__(self, features: Sequence[str]):
        self.features = list(features)

    def _x(self, rows):
        return np.array([[defense_v1.situation_feature_vector(r.situation, self.features)[f] for f in self.features]
                         for r in rows])

    def fit(self, rows, lam):
        y = np.array([1.0 if r.offense_success else 0.0 for r in rows])
        self.b0, self.w = dlt._fit_logistic(self._x(rows), y, lam)
        return self

    def logit(self, rows):
        return self.b0 + self._x(rows) @ self.w


def _choose_sit_lambda(rows, features, lambdas=dlt.LAMBDAS) -> float:
    games = list(dict.fromkeys(r.session_id for r in rows))
    best = None
    for lam in sorted(lambdas, reverse=True):
        ll, n = 0.0, 0
        for g in games:
            train = [r for r in rows if r.session_id != g]
            test = [r for r in rows if r.session_id == g]
            z = SituationOnly(features).fit(train, lam).logit(test)
            for zi, r in zip(z, test):
                p = min(1 - EPS, max(EPS, _sigmoid(float(zi))))
                ll -= math.log(p) if r.offense_success else math.log(1 - p)
                n += 1
        score = ll / n
        if best is None or score < best[0] - 1e-12:
            best = (score, lam)
    return best[1]


def defense_eval(rows, meta, template, current) -> list[dict[str, Any]]:
    labels = dlt.label_calls(rows, template)
    features = list(template["features"])
    preds = []
    for g in dict.fromkeys(r.session_id for r in rows):
        train = [r for r in rows if r.session_id != g]
        test = [r for r in rows if r.session_id == g]
        sit = SituationOnly(features).fit(train, _choose_sit_lambda(train, features))
        lam = dlt.choose_lambda(train, labels, template)
        full = defense_v1.validate_artifact(dlt.fit_artifact(train, labels, template, lam))
        z_sit = sit.logit(test)
        rate = sum(r.offense_success for r in train) / len(train)
        for zs, r in zip(z_sit, test):
            z_full = _logit(dlt.predict(full, r, labels))
            a = meta[r.snap_id]
            p = {f"w{int(w * 100)}": _sigmoid(float(zs) + w * (z_full - float(zs))) for w in WEIGHTS}
            p["train_base_rate"] = rate
            p["live_defense_v1_in_sample"] = dlt.predict(current, r, labels)
            preds.append({"target": "offense_won", "game": g, "y": int(r.offense_success), "p": p,
                          "confidence": a.confidence or "verified", "source": a.attestation_source or a.tier})
    return preds


# ---------------------------------------------------------------- offense


def offense_eval(data: he.EvalData, audit: Sequence[aa.AuditRow]) -> list[dict[str, Any]]:
    attested = {(a.ml_snap_id or str(a.snap_id)): a for a in audit if a.side == "offense" and a.tier == aa.TIER_ATTESTED}
    rows = [dict(r) for r in data.rows]
    for r in rows:
        a = attested.get(str(r.get("snap_id")))
        r["_attest"] = a
        if a is not None:
            r["_as_verified"] = {**r, "eligibility": "verified_execution", "supervised_eligible": True,
                                 "executed_formation": a.executed_formation, "executed_play": a.executed_play,
                                 "action_formation": a.executed_formation, "action_play": a.executed_play}
    by_game: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_game.setdefault(str(r.get("game_id")), []).append(r)
    human_games = [g for g in by_game if any(r["_attest"] is not None for r in by_game[g])]
    preds = []
    for g in human_games:
        train_raw = [r for h, rs in by_game.items() if h != g for r in rs]
        policies: dict[str, em.ExperimentalArtifact] = {
            "status_quo": em.train_experimental(train_raw, side="offense"),
        }
        for w in WEIGHTS:
            if w == 0:
                train = [r if r["_attest"] is None else {**r, "eligibility": "outcome_known_execution_uncertain",
                                                         "supervised_eligible": False, "action_play": None}
                         for r in train_raw]
                policies["w0"] = em.train_experimental(train, side="offense")
            else:
                train = [r["_as_verified"] if r["_attest"] is not None else r for r in train_raw]
                policies[f"w{int(w * 100)}"] = em.train_experimental(
                    train, side="offense", row_weight=lambda row, w=w: w if row.get("_attest") is not None else 1.0)
        for r in by_game[g]:
            a = r["_attest"]
            y = he.offense_label(r, "success") if a is not None else None
            if y is None:
                continue
            p = {
                name: float(em.predict_success(
                    art, formation=a.executed_formation, play=a.executed_play, down=r.get("down"),
                    distance=r.get("distance"), yardline=he._int(r.get("yardline")),
                    opponent_id=r.get("opponent_id"), opponent_type=r.get("opponent_type"))["probability"])
                for name, art in policies.items()
            }
            preds.append({"target": "success", "game": g, "y": int(y), "p": p,
                          "confidence": a.confidence, "source": a.attestation_source})
    return preds


def _block(preds: Sequence[Mapping[str, Any]], reference: str, n_boot: int, seed: int) -> dict[str, Any]:
    names = list(preds[0]["p"]) if preds else []
    ys = [p["y"] for p in preds]
    return {
        "models": {n: he.binary_metrics(ys, [p["p"][n] for p in preds]) for n in names},
        "vs_reference": {n: he.paired_bootstrap(preds, n, reference, n_boot=n_boot, seed=seed)
                         for n in names if n != reference},
    }


def summarize(preds: Sequence[Mapping[str, Any]], reference: str, n_boot: int, seed: int) -> dict[str, Any]:
    out = {"all": _block(preds, reference, n_boot, seed), "strata": {}}
    for key in ("confidence", "source"):
        for value in sorted({str(p[key]) for p in preds}):
            sub = [p for p in preds if str(p[key]) == value]
            out["strata"][f"{key}={value}"] = _block(sub, reference, n_boot, seed)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--defense-template", required=True)
    ap.add_argument("--defense-current", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=he.DEFAULT_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=he.DEFAULT_SEED)
    args = ap.parse_args(argv)

    from cfb_coach.games import data_dir

    out_root = Path(args.out).expanduser().resolve()
    live = Path(data_dir()).expanduser().resolve()
    if out_root == live or live in out_root.parents:
        raise SystemExit(f"refusing to write inside the live data directory {live}")
    db = str(Path(args.db).expanduser().resolve())
    conn = aa.connect_readonly(db)
    try:
        audit = aa.audit_rows(conn)
        drows, meta = defense_rows(conn, audit)
    finally:
        conn.close()
    template = defense_v1.validate_artifact(json.loads(Path(args.defense_template).read_text()))
    current = defense_v1.validate_artifact(json.loads(Path(args.defense_current).read_text()))
    dpreds = defense_eval(drows, meta, template, current)
    opreds = offense_eval(he.load_eval_data(db), audit)
    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "independently_verified_human_plays": {s: sum(1 for a in audit if a.side == s and a.tier == aa.TIER_VERIFIED)
                                               for s in ("offense", "defense")},
        "defense": {"games": len({p["game"] for p in dpreds}), **summarize(dpreds, "w0", args.bootstrap, args.seed)},
        "offense_human": {"games": len({p["game"] for p in opreds}),
                          **summarize(opreds, "status_quo", args.bootstrap, args.seed)},
    }
    stamp = report["generated"].replace("-", "").replace(":", "")
    out = out_root / f"run-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(out)
    for side in ("defense", "offense_human"):
        print(f"== {side} ({report[side]['games']} games)")
        blocks = {"all": report[side]["all"], **report[side]["strata"]}
        for label, block in blocks.items():
            print(f" -- {label}")
            for name, m in sorted(block["models"].items(), key=lambda kv: kv[1].get("log_loss", 9)):
                if not m.get("n"):
                    continue
                v = block["vs_reference"].get(name)
                ci = ("reference" if v is None else "n/a" if v.get("mean_diff") is None
                      else f"{v['mean_diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
                a = m["auc"]
                print(f"  {name:<28} n={m['n']} ll={m['log_loss']:.4f} brier={m['brier']:.4f} "
                      f"auc={'-' if a is None else f'{a:.3f}'} ece={m['ece']:.3f}  d={ci}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
