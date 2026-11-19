"""Forward-only, whole-game historical validation and shadow replay (read-only)."""
from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import coordinator as co
from cfb_coach.madden.ravens_defense import legality as lg
from cfb_coach.madden.ravens_defense import model as rm
from cfb_coach.madden.ravens_defense import opponent as op
from cfb_coach.madden.ravens_defense import outcome as oc
from cfb_coach.madden.ravens_defense.concepts import CLASSES
from cfb_coach.madden.ravens_defense.confirmations import Confirmations


def _games(rows: Sequence[Any]) -> list[tuple[str, list[Any]]]:
    g: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        g[r.session].append(r)
    return sorted(g.items(), key=lambda kv: (min(r.started_ts for r in kv[1]), kv[0]))


def _ll(p: float, y: bool) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return -math.log(p if y else 1 - p)


def _ci(per_game: Sequence[Mapping[str, Any]], a: str, b: str, *, n_boot: int = 2000, seed: int = 11) -> list[float] | None:
    if len(per_game) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [rng.choice(per_game) for _ in per_game]
        n = sum(x["n"] for x in s)
        vals.append(sum((x[a] - x[b]) * x["n"] for x in s) / n)
    vals.sort()
    return [round(vals[int(0.025 * (n_boot - 1))], 5), round(vals[int(0.975 * (n_boot - 1))], 5)]


def presnap_success_eval(rows: Sequence[Any]) -> dict[str, Any]:
    """Candidate-AGNOSTIC: predict 'offense succeeds on this snap' before the snap.

    challenger  = base success head (situation x concept) integrated over the V2 concept belief
    V2 baseline = Defensive V2's context-only shrunk success rate (its baseline() rule) refit on observed snaps
    global      = earlier-games success rate
    Coverage choice is NOT evaluated here (that needs verified executions)."""
    from cfb_coach.madden.model import defense_v2_pilot as v2
    per_game = []
    games = _games(rows)
    for i, (sess, g) in enumerate(games):
        prior = [r for _, gg in games[:i] for r in gg if r.closed and r.offense_success is not None]
        if len(prior) < 20:
            continue
        base = oc.fit_base(prior)
        hist = op.History(prior)
        glob = sum(r.offense_success for r in prior) / len(prior)
        ctx_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in prior:
            k = v2.context_key(r.down, r.distance)
            ctx_counts[k][0] += int(r.offense_success)
            ctx_counts[k][1] += 1
        tot = {"n": 0, "challenger": 0.0, "v2_context": 0.0, "global": 0.0,
               "b_challenger": 0.0, "b_v2_context": 0.0, "b_global": 0.0}
        for r in sorted(g, key=lambda r: r.order):
            if not r.closed or r.offense_success is None:
                continue
            earlier = [x for x in g if x.order < r.order and x.closed]
            b = op.v2_dirichlet(hist, earlier, r.order)
            ctx = op.context_tokens(r.down, r.distance, r.yards_to_goal)
            p = 0.0
            for k, c in enumerate(CLASSES):
                z = oc.base_logit(base, "success", c, ctx)
                p += b[k] * (oc._sig(z) if z is not None else glob)
            if not math.isfinite(p):
                p = glob
            s, n = ctx_counts.get(v2.context_key(r.down, r.distance), [0, 0])
            pv2 = (s + 16.0 * glob) / (n + 16.0)
            y = bool(r.offense_success)
            tot["n"] += 1
            for name, val in (("challenger", p), ("v2_context", pv2), ("global", glob)):
                tot[name] += _ll(val, y)
                tot["b_" + name] += (val - (1.0 if y else 0.0)) ** 2
        if tot["n"]:
            n = tot["n"]
            per_game.append({"session": sess, "opponent": g[0].opponent, "n": n, "train_snaps": len(prior),
                             **{k: v / n for k, v in tot.items() if k != "n"}})
    N = sum(p["n"] for p in per_game)
    out = {"heldout_games": len(per_game), "heldout_snaps": N, "per_game": per_game,
           "what_is_evaluated": "pre-snap P(offense success) without using our call (candidate-agnostic)"}
    for k in ("challenger", "v2_context", "global"):
        out[k] = {"logloss": round(sum(p[k] * p["n"] for p in per_game) / N, 5) if N else None,
                  "brier": round(sum(p["b_" + k] * p["n"] for p in per_game) / N, 5) if N else None}
    out["selected_success_base"] = "v2_context"
    if N:
        ci = _ci(per_game, "v2_context", "challenger")
        out["challenger_vs_v2_logloss_gain"] = {"point": round(out["v2_context"]["logloss"] - out["challenger"]["logloss"], 5),
                                                "game_cluster_bootstrap_95ci": ci}
        if ci and ci[0] > 0:
            out["selected_success_base"] = "concept_logit"
    out["selection_rule"] = "concept-level base head replaces V2's context rate only if its 95% CI gain is above 0"
    return out


def v2_outcome_eval(conn: Any) -> dict[str, Any]:
    """Run the actual Defensive V2 verified-execution reader and evaluator on each recorded franchise."""
    from cfb_coach.madden.model import defense_v2_pilot as v2
    out = {}
    for fr in v2.available_franchises(conn):
        rows, stats = v2.verified_rows(conn, franchise=fr["franchise"])
        out[fr["franchise"]] = {"games": fr["games"], "training_audit": stats, "validation": v2.evaluate(rows)}
    return out


def candidate_outcome_eval(rows: Sequence[Any]) -> dict[str, Any]:
    verified = [r for r in rows if r.execution == "verified_ravens_execution" and r.offense_success is not None]
    if not verified:
        return {"status": "NOT_EVALUABLE", "verified_ravens_executions": 0,
                "reason": "no verified Nickel 2-4 Wide / Dime 2-3 Odd executions with outcomes exist in any audited DB"}
    games = _games(verified)
    cat = rc.load()
    per_game = []
    for i, (sess, g) in enumerate(games):
        prior = [r for _, gg in games[:i] for r in gg]
        if not prior:
            continue
        allprior = [r for r in rows if r.started_ts < g[0].started_ts and r.closed]
        base = oc.fit_base(allprior)
        model = oc.OutcomeModel(cat, base, oc.fit_candidate_terms(cat, prior, base))
        prior_only = oc.OutcomeModel(cat, base, oc.fit_candidate_terms(cat, [], base))
        hist = op.History(allprior)
        t = {"n": 0, "trained": 0.0, "prior_only": 0.0}
        for r in sorted(g, key=lambda r: r.order):
            earlier = [x for x in g if x.order < r.order]
            b = op.v2_dirichlet(hist, earlier, r.order)
            ctx = op.context_tokens(r.down, r.distance, r.yards_to_goal)
            c = cat.by_id()[r.candidate_id]
            t["trained"] += _ll(model.score(c, b, ctx)["predicted_offense_success"], bool(r.offense_success))
            t["prior_only"] += _ll(prior_only.score(c, b, ctx)["predicted_offense_success"], bool(r.offense_success))
            t["n"] += 1
        per_game.append({"session": sess, "n": t["n"], "trained": t["trained"] / t["n"], "prior_only": t["prior_only"] / t["n"]})
    N = sum(p["n"] for p in per_game)
    return {"status": "EVALUATED", "heldout_games": len(per_game), "heldout_snaps": N, "per_game": per_game,
            "trained_vs_prior_gain_ci": _ci(per_game, "prior_only", "trained")}


def replay(rows: Sequence[Any], *, sessions: Sequence[str] | None = None, conf: Confirmations | None = None,
           assume_setup: bool = True, keep_cards: bool = False) -> dict[str, Any]:
    """Shadow replay of historical defensive situations through the coordinator.

    For every game, the model is refit on strictly earlier games only. Picks are MODEL PREDICTIONS on
    situations where we actually ran other (non-Ravens) calls: they are not observed outcomes."""
    cat = rc.load()
    conf = conf or Confirmations()
    legal, _ = lg.legal_set(cat, installed=None, conf=conf, assume_setup=assume_setup)
    games = _games(rows)
    picks: Counter[str] = Counter()
    rz_picks: Counter[str] = Counter()
    by_ctx: dict[str, Counter] = defaultdict(Counter)
    lat: list[float] = []
    cards = []
    users = Counter()
    n = 0
    for i, (sess, g) in enumerate(games):
        if sessions and sess not in sessions:
            continue
        prior = [r for _, gg in games[:i] for r in gg]
        if not prior:
            continue
        model = rm.train(prior, cat, belief_eval=None, source={"replay_for": sess})
        hist = model.history
        for r in sorted(g, key=lambda r: r.order):
            snap = co.SnapCtx(sess, r.opponent, r.order, r.down, r.distance, r.yards_to_goal, r.shape)
            t0 = time.perf_counter()
            d = co.decide(model, legal=legal, snap=snap, same_game=g, history=hist, conf=conf, cat=cat,
                          assume_setup=assume_setup)
            lat.append((time.perf_counter() - t0) * 1000)
            if d["status"] != "CALL":
                continue
            n += 1
            cid = d["candidate"]["candidate_id"]
            picks[cid] += 1
            users[d["user"]["status"]] += 1
            ctxk = "|".join(d["context"])
            by_ctx[ctxk][cid] += 1
            if r.yards_to_goal is not None and r.yards_to_goal <= 20:
                rz_picks[cid] += 1
            if keep_cards:
                cards.append({"session": sess, "order": r.order, "situation": f"{r.down}&{r.distance} ytg {r.yards_to_goal}",
                              "card": co.card_lines(d), "top3": [(s["candidate_id"], s["defensive_loss"]) for s in d["ranked"][:3]],
                              "belief_top": sorted(d["belief"].items(), key=lambda kv: -kv[1])[:3],
                              "observed_concept_after_snap": r.concept})
    ent = -sum((v / n) * math.log(v / n) for v in picks.values()) if n else 0.0
    return {"calls": n, "pick_distribution": dict(picks.most_common()), "distinct_candidates": len(picks),
            "pick_entropy_nats": round(ent, 3), "max_entropy_nats": round(math.log(len(legal)), 3) if legal else 0,
            "blitz_rate": round(sum(v for k, v in picks.items() if k == "dime_dbfire2") / n, 4) if n else None,
            "zone_it_rate": round(sum(v for k, v in picks.items() if k.endswith("+zone_it")) / n, 4) if n else None,
            "red_zone_20_picks": dict(rz_picks.most_common()),
            "contexts_with_more_than_one_pick": sum(1 for c in by_ctx.values() if len(c) > 1),
            "user_status": dict(users),
            "decision_latency_ms": {"n": len(lat), "median": round(statistics.median(lat), 3) if lat else None,
                                    "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3) if lat else None,
                                    "max": round(max(lat), 3) if lat else None},
            "setup_assumed": assume_setup, "legal_candidates": [c.candidate_id for c in legal],
            "cards": cards if keep_cards else None,
            "warning": "predicted choices on historical situations; not observed outcomes of these calls"}
