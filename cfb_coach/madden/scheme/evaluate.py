"""Chronological, game-grouped evaluation and training of the civil-scheme model. Read-only on the DB.

Leakage controls:
  * folds are whole games in time order (``heldout_eval.make_folds``): train = every EARLIER game only;
  * the existing-model offset for any row is trained on strictly earlier games;
  * beliefs use earlier games + earlier closed snaps of the same game only;
  * the head architecture per target is chosen on the EARLY test games and then only REPORTED on the LATE ones.
All counterfactual numbers (what the restricted coordinator would have picked) are model predictions,
never observed outcomes, and are labeled as such.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.scheme import beliefs as bl
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import coordinator as co
from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme import evidence as evd
from cfb_coach.madden.scheme import learners as ln
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import model as sm
from cfb_coach.madden.scheme import priors as pr

DO_NO_HARM = 0.002          # a head must beat the reference log loss by this much on the selection games
ARCHS = ("logistic", "logistic_strong", "gbt")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _applied_offense_book(conn) -> dict[str, list[str]]:
    row = conn.execute("SELECT value FROM meta WHERE key = 'active_playbook_json'").fetchone()
    if not row:
        return {}
    data = json.loads(row[0])
    rec = (data.get("applied") or data).get("offense") or {}
    return dict(rec.get("formations") or {})


def _game_ids(examples: Sequence[sm.Example]) -> set[str]:
    return {e.game for e in examples}


def prepare(db_path: str | Path, geo: dz.GeometrySource, cat: sc.SchemeCatalog) -> dict[str, Any]:
    data = he.load_eval_data(db_path)
    rows, order = data.rows, data.game_order
    conn = he.connect_readonly(db_path)
    try:
        excluded = evd.excluded_snap_keys(conn)
        vev = [e for e in evd.variant_evidence(conn, cat) if e.usable]
        book = _applied_offense_book(conn)
    finally:
        conn.close()
    by_key = {e.snap_key: e for e in vev}
    designs: dict[int, dz.Design] = {}
    for i, r in enumerate(rows):
        e = by_key.get(str(r.get("snap_id")))
        if e is not None:
            designs[i] = dz.adjusted_design(cat.get(e.variant_id), geo, applied=e.applied)  # type: ignore[arg-type]

    def keep(r: Mapping[str, Any]) -> bool:
        if not he.is_verified_offense(r):
            return False
        k = str(r.get("snap_id"))
        return k not in excluded or k in by_key

    offsets = sm.existing_offsets(rows, order, keep)
    beliefs = bl.annotate_rows(rows, order)
    examples = sm.build_examples(rows, order, geo, offsets=offsets, beliefs=beliefs, keep=keep, designs=designs)
    return {"data": data, "rows": rows, "order": order, "examples": examples, "beliefs": beliefs, "keep": keep,
            "variant_evidence": vev, "excluded": excluded, "book": book}


def _ref_p(model_name: str, e: sm.Example, t: str, base: Mapping[str, float]) -> float:
    if model_name == "existing":
        return e.offset_p if t == "success" else base[t]
    raise ValueError(model_name)


def run(db_path: str | Path, *, geo: dz.GeometrySource | None = None, cat: sc.SchemeCatalog | None = None,
        n_members: int = 3, coordinator_sim: bool = True, log=print) -> dict[str, Any]:
    geo = geo or dz.GeometrySource()
    cat = cat or sc.load()
    sha_before = file_sha256(db_path)
    t_start = time.time()
    P = prepare(db_path, geo, cat)
    rows, order, examples, beliefs = P["rows"], P["order"], P["examples"], P["beliefs"]
    folds = he.make_folds(rows, order, is_test_row=P["keep"], mode="chronological")
    by_game: dict[str, list[sm.Example]] = defaultdict(list)
    for e in examples:
        by_game[e.game].append(e)
    preds: list[dict[str, Any]] = []
    sims: list[dict[str, Any]] = []
    scorer_cache = None
    variants = list(cat.variants)
    stock_ok = {v.variant_id for v in variants if all(lg.stock_has(v.formation, v.base_play))}
    sim_variants = [v for v in variants if v.variant_id in stock_ok]
    book_pairs = [(f, p) for f, ps in P["book"].items() for p in ps]
    for k, fold in enumerate(folds):
        train = [e for g in fold.train_games for e in by_game.get(g, [])]
        test = by_game.get(fold.test_game, [])
        if not train or not test:
            continue
        base = {t: (sum(e.y[t] for e in train if e.y[t] is not None) + 1.0)
                / (sum(1 for e in train if e.y[t] is not None) + 2.0) for t in sm.TARGETS}
        models = {a: sm.fit(train, {t: a for t in sm.TARGETS}, n_members=n_members, with_yards=False) for a in ARCHS}
        for e in test:
            outs = {a: m.predict(e.x, e.offset_p) for a, m in models.items()}
            for t in sm.TARGETS:
                y = e.y[t]
                if y is None:
                    continue
                pp = {"existing": _ref_p("existing", e, t, base)}
                sd = {}
                for a in ARCHS:
                    pp[a] = outs[a][t][2]
                    sd[a] = outs[a][t][1]
                r = e.row
                preds.append({"target": t, "game": fold.test_game, "fold": k, "y": int(y), "p": pp, "sd": sd,
                              "opponent_type": r.get("opponent_type") or "unknown", "basis": he.verification_basis(r),
                              "down": r.get("down"), "distance": r.get("distance"), "yardline": r.get("yardline"),
                              "pair": [r.get("action_formation"), r.get("action_play")], "geometry": e.geometry_status})
        log(f"fold {k + 1}/{len(folds)} test game {fold.test_game}: train {len(train)} test {len(test)}")
    chosen = choose(preds)
    if coordinator_sim:
        for k, fold in enumerate(folds):
            train = [e for g in fold.train_games for e in by_game.get(g, [])]
            test = by_game.get(fold.test_game, [])
            if not train or not test:
                continue
            m = sm.fit(train, chosen, n_members=n_members, with_yards=False)
            tg = set(fold.train_games)
            em_art = em.train_experimental([dict(r) for r in rows if r.get("game_id") in tg], side="offense")
            m.existing = em_art
            scorer = co.Scorer(m, cat, geo)
            recent: list[str] = []                       # this game's earlier picks (repetition penalty)
            for e in test:
                r = e.row
                b = beliefs[e.idx]
                opp_t = r.get("opponent_type") or "human"
                kw = dict(down=r.get("down"), distance=r.get("distance"), yardline=he._int(r.get("yardline")),
                          opponent_id=str(r.get("opponent_id") or ""), opponent_type=opp_t, belief=b)
                ranked = scorer.score(sim_variants, recent_calls=list(recent), **kw)
                stock_rank = scorer.score(sim_variants, recent_calls=list(recent), stock_only=True, **kw)
                recent.insert(0, ranked[0].variant.variant_id)
                top = ranked[0]
                ex_best = None
                if book_pairs:
                    sc_ = [(em.predict_success(em_art, formation=f, play=p, down=r.get("down"), distance=r.get("distance"),
                                               yardline=he._int(r.get("yardline")), opponent_id=r.get("opponent_id"),
                                               opponent_type=opp_t)["probability"], f, p) for f, p in book_pairs]
                    ex_best = max(sc_)
                gates = pr.gates(b.mix, b.families, b.gate_confidence)
                gtop = max(gates.items(), key=lambda kv: kv[1])
                sims.append({
                    "game": fold.test_game, "fold": k, "down": r.get("down"), "distance": r.get("distance"),
                    "yardline": r.get("yardline"), "opponent_type": opp_t,
                    "pick": top.variant.variant_id, "pick_pair": list(top.variant.pair),
                    "pick_p_success_pred": top.p_success, "pick_sd": top.sd,
                    "pick_existing_offset_p": top.parts["existing_model_p"],
                    "unrestricted_best_pred": ex_best[0] if ex_best else None,
                    "unrestricted_best_pair": [ex_best[1], ex_best[2]] if ex_best else None,
                    "stock_only_pick": stock_rank[0].variant.variant_id,
                    "ranking": [s.variant.variant_id for s in ranked],
                    "gate_top": [gtop[0], gtop[1]], "pick_tagged_for_top": top.parts["tagged_beater_for_top"],
                    "observed_run": he.run_or_pass(r.get("action_play")) == "run",
                    "observed_success": e.y["success"],
                })
            log(f"simulated fold {k + 1}/{len(folds)}")
    report = summarize(preds, sims, folds, P, cat)
    report["chosen"] = chosen
    sha_after = file_sha256(db_path)
    report["db_integrity"] = {"path": str(db_path), "sha256_before": sha_before, "sha256_after": sha_after,
                              "unchanged": sha_before == sha_after, "opened": "read-only (mode=ro URI)"}
    report["runtime_s"] = round(time.time() - t_start, 1)
    return report


def choose(preds) -> dict[str, str]:
    """Per target: the head with the lowest log loss on the EARLY (selection) folds, if it beats the existing
    reference by DO_NO_HARM; otherwise the existing model / base rate."""
    folds_used = sorted({p["fold"] for p in preds})
    cut = folds_used[len(folds_used) // 2] if folds_used else 0
    out = {}
    for t in sm.TARGETS:
        te = [p for p in preds if p["target"] == t and p["fold"] < cut]
        if not te:
            out[t] = "existing"
            continue
        ref = _metrics(te, "existing")["log_loss"]
        best, best_ll = "existing", ref - DO_NO_HARM
        for a in ARCHS:
            ll = _metrics(te, a)["log_loss"]
            if ll < best_ll:
                best, best_ll = a, ll
        out[t] = best
    return out


def _metrics(sub: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    m = he.binary_metrics([p["y"] for p in sub], [p["p"][model] for p in sub])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def summarize(preds, sims, folds, P, cat) -> dict[str, Any]:
    folds_used = sorted({p["fold"] for p in preds})
    cut = folds_used[len(folds_used) // 2] if folds_used else 0
    early = [p for p in preds if p["fold"] < cut]
    late = [p for p in preds if p["fold"] >= cut]
    out: dict[str, Any] = {"folds": len(folds_used), "selection_folds": [f for f in folds_used if f < cut],
                           "confirmation_folds": [f for f in folds_used if f >= cut], "targets": {}, "chosen": {}}
    for t in sm.TARGETS:
        tp_all = [p for p in preds if p["target"] == t]
        te = [p for p in early if p["target"] == t]
        tl = [p for p in late if p["target"] == t]
        block: dict[str, Any] = {"all": {}, "selection": {}, "confirmation": {}, "bootstrap_vs_existing": {}}
        for name in ("existing",) + ARCHS:
            block["all"][name] = _metrics(tp_all, name) if tp_all else {"n": 0}
            block["selection"][name] = _metrics(te, name) if te else {"n": 0}
            block["confirmation"][name] = _metrics(tl, name) if tl else {"n": 0}
        for a in ARCHS:
            block["bootstrap_vs_existing"][a] = {
                "all": he.paired_bootstrap(tp_all, a, "existing", n_boot=1000, seed=7),
                "confirmation": he.paired_bootstrap(tl, a, "existing", n_boot=1000, seed=7)}
        ref_ll = block["selection"]["existing"].get("log_loss")
        best, best_ll = "existing", ref_ll
        for a in ARCHS:
            ll = block["selection"][a].get("log_loss")
            if ll is not None and ref_ll is not None and ll < best_ll - (DO_NO_HARM if best == "existing" else 0):
                best, best_ll = a, ll
        out["chosen"][t] = best
        # uncertainty: does the reported sd separate easy from hard rows?
        if best in ARCHS and tp_all:
            s = sorted(tp_all, key=lambda p: p["sd"][best])
            n3 = max(1, len(s) // 3)
            block["uncertainty_terciles"] = [
                {"sd_mean": round(sum(p["sd"][best] for p in part) / len(part), 4),
                 "log_loss": round(_metrics(part, best)["log_loss"], 4), "n": len(part)}
                for part in (s[:n3], s[n3:2 * n3], s[2 * n3:]) if part]
        out["targets"][t] = block
    # conversion = first down on 3rd/4th down
    conv = [p for p in preds if p["target"] == "first_down" and str(p.get("down")) in ("3", "4")]
    out["conversion_3rd_4th"] = {n: _metrics(conv, n) for n in ("existing",) + ARCHS} if conv else {"n": 0}
    # expected vs observed for the 12 catalog base plays (stock design; no variant has outcomes)
    pairs = {(sc_norm(v.formation), sc_norm(v.base_play)) for v in cat.variants}
    chosen_s = out["chosen"]["success"]
    evo = defaultdict(lambda: {"n": 0, "obs": 0, "pred_existing": 0.0, "pred_chosen": 0.0})
    for p in preds:
        if p["target"] != "success":
            continue
        key = (sc_norm(p["pair"][0]), sc_norm(p["pair"][1]))
        if key in pairs:
            d = evo[f"{p['pair'][0]} / {p['pair'][1]}"]
            d["n"] += 1
            d["obs"] += p["y"]
            d["pred_existing"] += p["p"]["existing"]
            d["pred_chosen"] += p["p"][chosen_s]
    out["catalog_base_plays_expected_vs_observed"] = {
        k: {"n": d["n"], "observed_rate": round(d["obs"] / d["n"], 3), "wilson": he.wilson(d["obs"], d["n"]),
            "pred_existing": round(d["pred_existing"] / d["n"], 3), f"pred_{chosen_s}": round(d["pred_chosen"] / d["n"], 3)}
        for k, d in sorted(evo.items())}
    out["coordinator_simulation"] = simulate_summary(sims, cat)
    ex = P["examples"]
    out["data"] = {
        "verified_offense_snaps": len(ex), "games": len({e.game for e in ex}),
        "by_opponent_type": dict(Counter(e.row.get("opponent_type") for e in ex)),
        "by_basis": dict(Counter(he.verification_basis(e.row) for e in ex)),
        "geometry_status": dict(Counter(e.geometry_status for e in ex)),
        "verified_variant_outcomes": len(P["variant_evidence"]),
        "civil_recommendations_excluded_unconfirmed": len(P["excluded"]),
    }
    nn_ok, nn_why = ln.neural_challenger_justified(len(ex), out["data"]["games"])
    out["neural_challenger"] = {"trained": nn_ok, "reason": nn_why}
    return out


def sc_norm(s: Any) -> str:
    from cfb_coach.madden import catalog as mc
    return mc.norm(str(s or ""))


def simulate_summary(sims: Sequence[Mapping[str, Any]], cat: sc.SchemeCatalog) -> dict[str, Any]:
    if not sims:
        return {"n": 0}
    n = len(sims)
    picks = Counter(s["pick"] for s in sims)
    ent = -sum(c / n * math.log(c / n) for c in picks.values())
    stock_changes = sum(1 for s in sims if s["pick"] != s["stock_only_pick"])
    # same-base variants ranking differently: for each base pair with >=2 variants, how often the order flips
    by_pair = defaultdict(list)
    for v in cat.variants:
        by_pair[v.pair].append(v.variant_id)
    flips = {}
    for pair, ids in by_pair.items():
        if len(ids) < 2:
            continue
        first = Counter()
        for s in sims:
            ranked = [r for r in s["ranking"] if r in ids]
            if ranked:
                first[ranked[0]] += 1
        flips[f"{pair[0]} / {pair[1]}"] = dict(first)
    sy = [s for s in sims if s.get("distance") is not None and int(s["distance"]) <= 2]
    gl = [s for s in sims if s.get("yardline") is not None and s.get("distance") is not None
          and int(s["yardline"]) + int(s["distance"]) >= 100]

    def damage(sub):
        if not sub:
            return {"n": 0}
        runs = [s for s in sub if s["observed_run"]]
        passes = [s for s in sub if not s["observed_run"]]
        rs = [s["observed_success"] for s in runs if s["observed_success"] is not None]
        ps = [s["observed_success"] for s in passes if s["observed_success"] is not None]
        ub = [s["unrestricted_best_pred"] for s in sub if s["unrestricted_best_pred"] is not None]
        return {"n": len(sub), "historical_run_share": round(len(runs) / len(sub), 3),
                "observed_run_success": he.wilson(sum(rs), len(rs)), "observed_pass_success": he.wilson(sum(ps), len(ps)),
                "pred_restricted_pick_p_success": round(sum(s["pick_p_success_pred"] for s in sub) / len(sub), 3),
                "pred_unrestricted_best_existing_model_p": round(sum(ub) / len(ub), 3) if ub else None,
                "note": "predicted, not observed; the restricted set has no run plays"}

    gate_hi = [s for s in sims if s["gate_top"][1] >= 0.5]
    gate_lo = [s for s in sims if s["gate_top"][1] < 0.1]
    ub_all = [s["unrestricted_best_pred"] for s in sims if s["unrestricted_best_pred"] is not None]
    return {
        "n_situations": n, "distinct_picks": len(picks), "pick_entropy_nats": round(ent, 3),
        "max_entropy_nats": round(math.log(max(1, len({v.variant_id for v in cat.variants}))), 3),
        "pick_counts": dict(picks.most_common()),
        "adjustments_changed_top_pick_share": round(stock_changes / n, 3),
        "same_base_top_variant_counts": flips,
        "mean_pred_p_success_restricted_pick": round(sum(s["pick_p_success_pred"] for s in sims) / n, 4),
        "mean_existing_model_p_restricted_pick_base": round(sum(s["pick_existing_offset_p"] for s in sims) / n, 4),
        "mean_existing_model_p_unrestricted_best": round(sum(ub_all) / len(ub_all), 4) if ub_all else None,
        "short_yardage_dist_le_2": damage(sy), "goal_to_go": damage(gl),
        "beater_gate_high_n": len(gate_hi),
        "beater_gate_high_pick_tagged_share": round(sum(1 for s in gate_hi if s["pick_tagged_for_top"]) / len(gate_hi), 3) if gate_hi else None,
        "beater_gate_low_n": len(gate_lo),
        "beater_gate_low_pick_tagged_share": round(sum(1 for s in gate_lo if s["pick_tagged_for_top"]) / len(gate_lo), 3) if gate_lo else None,
        "label": "PREDICTED counterfactuals on held-out situations; not observed outcomes",
    }


def train_final(db_path: str | Path, arch: Mapping[str, str], *, geo: dz.GeometrySource | None = None,
                cat: sc.SchemeCatalog | None = None, n_members: int = 5, eval_report: Mapping[str, Any] | None = None) -> sm.SchemeModel:
    geo = geo or dz.GeometrySource()
    cat = cat or sc.load()
    sha_before = file_sha256(db_path)
    P = prepare(db_path, geo, cat)
    model = sm.fit(P["examples"], arch, n_members=n_members)
    model.existing = em.train_experimental([dict(r) for r in P["rows"]], side="offense")
    ex = P["examples"]
    dates = sorted(str(r.get("game_id")) for r in P["rows"])
    sha_after = file_sha256(db_path)
    model.meta = {
        "schema": sm.SCHEMA, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "catalog_id": cat.catalog_id, "catalog_version": cat.version(), "architecture": dict(arch),
        "targets": list(sm.TARGETS), "utility": sm.UTILITY, "variant_transfer": sm.VARIANT_TRANSFER,
        "beater": {"max_logit": pr.BEATER_MAX, "gate": [pr.GATE_LO, pr.GATE_HI], "k": pr.BEATER_K},
        "training_db": {"path": str(db_path), "sha256": sha_before, "unchanged_after": sha_before == sha_after,
                        "opened": "read-only"},
        "training_rows": len(ex), "games": len({e.game for e in ex}),
        "by_opponent_type": dict(Counter(e.row.get("opponent_type") for e in ex)),
        "verified_variant_outcomes": len(P["variant_evidence"]),
        "geometry_source": str(geo.path) if getattr(geo, "path", None) else None,
        "geometry_status_counts": dict(Counter(e.geometry_status for e in ex)),
        "neural_challenger": ln.neural_challenger_justified(len(ex), len({e.game for e in ex}))[1],
        "eval_summary": _eval_digest(eval_report) if eval_report else None,
    }
    return model


def _eval_digest(rep: Mapping[str, Any]) -> dict[str, Any]:
    out = {"chosen": rep.get("chosen"), "folds": rep.get("folds")}
    for t, b in (rep.get("targets") or {}).items():
        out[t] = {n: {k: b["all"][n].get(k) for k in ("n", "log_loss", "brier", "auc", "ece")} for n in b["all"]}
    return out
