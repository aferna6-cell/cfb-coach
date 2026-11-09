"""Postgame learning: validated snapshot -> challenger -> game-separated chronological comparison.

* Labels are validated first: only snaps with a VERIFIED execution, a revealed opponent action and a
  known outcome become training rows; the rest are counted and excluded with reasons.
* The snapshot is immutable (read-only file + sha256 in the manifest).
* Evaluation is expanding-window by WHOLE GAME: fold k fits only on games < k.
* The champion artifact is only read. The checksum is verified before/after, the output directory
  must differ from the champion's, and there is deliberately NO promotion function in this module.
"""
from __future__ import annotations

import json
import math
import os
import random
import stat
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from cfb_coach.madden.unified.guard import sha256_file
from cfb_coach.madden.unified.opponent import OpponentModel
from cfb_coach.madden.unified.types import DEF, OFF, GameState, Outcome, SnapObservation
from cfb_coach.madden.unified.world_model import DefSig, Profile, WorldModel, records_from_observation

POSTGAME_VERSION = "unified_postgame.v1"
MIN_GAMES_FOR_GATE = 8


# ----------------------------------------------------------------------------- snapshot
def validate_labels(games: Sequence[Sequence[SnapObservation]]) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    excluded: Counter = Counter()
    for order, game in enumerate(games):
        for o in game:
            if not o.closed:
                excluded["snap_not_closed"] += 1
            elif o.outcome is None or not o.outcome.known:
                excluded["no_outcome"] += 1
            elif o.executed is None or o.executed_status not in ("confirmed", "modified"):
                excluded["execution_unverified(recommendation_only)"] += 1
            elif o.opponent_action is None or o.opponent_action.verification == "unverified":
                excluded["opponent_action_unverified"] += 1
            else:
                recs = records_from_observation(o)
                if not recs:
                    excluded["unusable_record"] += 1
                    continue
                st, prof, d, oc, w, side = recs[0]
                rows.append({"game_id": st.game_id, "game_order": order, "seq": st.seq, "state": st.to_dict(),
                             "profile": [prof.play_type, prof.concept, prof.play_id, dict(prof.features)],
                             "defsig": [d.ctype, d.family, d.vkey, d.play_id, dict(d.features)],
                             "outcome": asdict(oc), "weight": w, "our_side": side,
                             "opp_signature": list(o.opponent_action.signature)})
    return rows, excluded


def write_snapshot(rows: Sequence[dict[str, Any]], excluded: Counter, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "snapshot.jsonl"
    if data.exists():
        raise FileExistsError("snapshots are immutable; choose a new directory")
    data.write_text("\n".join(json.dumps(r, sort_keys=True, default=list) for r in rows))
    os.chmod(data, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    manifest = {"version": POSTGAME_VERSION, "rows": len(rows), "games": len({r["game_id"] for r in rows}),
                "excluded": dict(excluded), "sha256": sha256_file(data)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_snapshot(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    man = json.loads((p.parent / "manifest.json").read_text())
    if sha256_file(p) != man["sha256"]:
        raise ValueError("snapshot checksum mismatch: file was modified after it was written")
    return [json.loads(l) for l in p.read_text().splitlines() if l]


def _row_objs(r: Mapping[str, Any]):
    st = GameState.from_dict(r["state"])
    pt, cc, pid, pf = r["profile"]
    ct, fam, vk, did, df = r["defsig"]
    return st, Profile(pt, cc, pid, pf), DefSig(ct, fam, vk, did, df), Outcome(**r["outcome"])


# ----------------------------------------------------------------------------- metrics
def _brier(ps: Sequence[float], ys: Sequence[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(1, len(ys))


def _logloss(ps: Sequence[float], ys: Sequence[float]) -> float:
    e = 1e-4
    return -sum(y * math.log(min(1 - e, max(e, p))) + (1 - y) * math.log(min(1 - e, max(e, 1 - p))) for p, y in zip(ps, ys)) / max(1, len(ys))


def calibration_bins(ps: Sequence[float], ys: Sequence[float], k: int = 5) -> list[dict[str, float]]:
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p, y in zip(ps, ys):
        bins[min(k - 1, int(p * k))].append((p, y))
    return [{"bin": b, "n": len(v), "mean_pred": sum(p for p, _ in v) / len(v), "obs_rate": sum(y for _, y in v) / len(v)} for b, v in sorted(bins.items())]


def eval_world_model(rows: Sequence[dict[str, Any]], make_model: Callable[[Sequence[dict[str, Any]]], WorldModel], *,
                     min_train_games: int = 2) -> dict[str, Any]:
    """Expanding-window, whole-game folds. Returns per-game errors (for game-clustered bootstrap) and pooled metrics."""
    games = sorted({r["game_order"] for r in rows})
    per_game: dict[int, dict[str, list[float]]] = {}
    pooled: dict[str, list[float]] = defaultdict(list)
    for g in games[min_train_games:]:
        train = [r for r in rows if r["game_order"] < g]
        test = [r for r in rows if r["game_order"] == g]
        m = make_model(train)
        stats: dict[str, list[float]] = defaultdict(list)
        for r in test:
            st, p, d, o = _row_objs(r)
            pr = m.predict(st, p, d)
            fd = o.first_down if o.first_down is not None else (o.yards or 0) >= (st.distance or 99)
            stats["abs_yards"].append(abs(pr.e_yards - (o.yards or 0.0)))
            stats["sq_yards"].append((pr.e_yards - (o.yards or 0.0)) ** 2)
            stats["fd_p"].append(pr.p_first_down)
            stats["fd_y"].append(float(bool(fd)))
        per_game[g] = {k: v for k, v in stats.items()}
        for k, v in stats.items():
            pooled[k] += v
    n = len(pooled["abs_yards"])
    out = {"n_test_rows": n, "test_games": len(per_game)}
    if n:
        out.update({"mae_yards": sum(pooled["abs_yards"]) / n, "rmse_yards": math.sqrt(sum(pooled["sq_yards"]) / n),
                    "brier_first_down": _brier(pooled["fd_p"], pooled["fd_y"]), "logloss_first_down": _logloss(pooled["fd_p"], pooled["fd_y"]),
                    "calibration_first_down": calibration_bins(pooled["fd_p"], pooled["fd_y"])})
    out["_per_game_abs"] = {g: sum(v["abs_yards"]) / len(v["abs_yards"]) for g, v in per_game.items() if v["abs_yards"]}
    return out


def fit_on(rows: Sequence[dict[str, Any]]) -> WorldModel:
    m = WorldModel()
    recs = []
    for r in rows:
        st, p, d, o = _row_objs(r)
        recs.append((st, p, d, o, r["weight"], r["our_side"]))
    m.fit_hierarchy(recs)
    return m


def bootstrap_diff(a: Mapping[int, float], b: Mapping[int, float], n: int = 2000, seed: int = 7) -> dict[str, float]:
    """Game-clustered bootstrap of mean(a - b); negative => a (lower error) is better."""
    gs = sorted(set(a) & set(b))
    if len(gs) < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "games": len(gs)}
    rng = random.Random(seed)
    diffs = [a[g] - b[g] for g in gs]
    means = sorted(sum(rng.choice(diffs) for _ in gs) / len(gs) for _ in range(n))
    return {"mean": sum(diffs) / len(diffs), "lo": means[int(0.025 * n)], "hi": means[int(0.975 * n)], "games": len(gs)}


def eval_opponent(games: Sequence[Sequence[SnapObservation]], opp_side: str, *, min_train_games: int = 1) -> dict[str, Any]:
    """Chronological log-loss of the Bayesian belief vs a pooled-frequency baseline (same support, add-one smoothing)."""
    total_b = total_f = 0.0
    n = 0
    freq: Counter = Counter()
    om = OpponentModel(opp_side)
    for gi, game in enumerate(games):
        prev = None
        for o in game:
            sig = tuple(o.opponent_action.signature) if o.opponent_action and o.opponent_action.signature else None
            if sig and gi >= min_train_games:
                b = om.beliefs(o.state).dist
                support = set(b) | set(freq) | {sig}
                pb = max(b.get(sig, 0.0), 1e-4)
                tot = sum(freq.values()) + len(support)
                pf = (freq.get(sig, 0) + 1) / tot
                total_b -= math.log(pb)
                total_f -= math.log(pf)
                n += 1
            if sig:
                freq[sig] += 1
            om.observe(o, prev_obs=prev)
            prev = o
        om.close_game(game[0].state.game_id)
    return {"n": n, "belief_logloss": total_b / n if n else None, "frequency_logloss": total_f / n if n else None,
            "belief_better": bool(n and total_b < total_f)}


# ----------------------------------------------------------------------------- challenger
def train_challenger(rows: Sequence[dict[str, Any]], out_dir: str | Path, *, champion_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir).resolve()
    champ_sha_before = None
    if champion_path:
        cp = Path(champion_path).resolve()
        if out == cp.parent or cp in out.parents or out in cp.parents:
            raise ValueError("challenger output directory must be separate from the champion's location")
        champ_sha_before = sha256_file(cp)
    out.mkdir(parents=True, exist_ok=True)
    challenger = fit_on(rows)
    art = out / "challenger.json"
    art.write_text(json.dumps(challenger.to_dict()))
    cold = eval_world_model(rows, lambda tr: WorldModel())
    chall = eval_world_model(rows, fit_on)
    comp: dict[str, Any] = {"cold_start_prior": cold, "challenger_refit_each_fold": chall}
    gate: dict[str, Any] = {}
    champ = None
    if champion_path:
        champ_model = WorldModel.from_dict(json.loads(Path(champion_path).read_text()))
        champ = eval_world_model(rows, lambda tr, _m=champ_model: _m)
        comp["champion_frozen"] = champ
        gate["vs_champion_mae_diff"] = bootstrap_diff(chall["_per_game_abs"], champ["_per_game_abs"])
    gate["vs_cold_prior_mae_diff"] = bootstrap_diff(chall["_per_game_abs"], cold["_per_game_abs"])
    ref = gate.get("vs_champion_mae_diff") or gate["vs_cold_prior_mae_diff"]
    n_games = len({r["game_id"] for r in rows})
    gate["advisory"] = {"games": n_games, "min_games": MIN_GAMES_FOR_GATE,
                        "statistically_better": bool(ref["games"] >= 2 and ref["hi"] < 0),
                        "enough_games": n_games >= MIN_GAMES_FOR_GATE}
    gate["promotion_recommended"] = bool(gate["advisory"]["statistically_better"] and gate["advisory"]["enough_games"])
    report = {"version": POSTGAME_VERSION, "challenger_artifact": str(art), "challenger_sha256": sha256_file(art),
              "rows": len(rows), "games": n_games, "comparison": {k: {kk: vv for kk, vv in v.items() if kk != "_per_game_abs"} for k, v in comp.items()},
              "gate": gate, "promotion": "NOT PERFORMED - the champion is never modified by this pipeline; promotion needs an explicit human action",
              "champion_modified": False}
    if champion_path:
        report["champion_sha256_before"] = champ_sha_before
        report["champion_sha256_after"] = sha256_file(Path(champion_path))
        report["champion_unchanged"] = champ_sha_before == report["champion_sha256_after"]
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
