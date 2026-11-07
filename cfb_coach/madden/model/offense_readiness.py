"""Data audit, held-out evaluation extras, shadow replay, scenarios A-F and readiness gates for the route-aware offense.

Honesty rules baked in:
* the game database is opened read-only (``mode=ro``) and never written;
* ``source`` on every shadow record says ``recorded_game_replay`` (or ``synthetic_selfcheck``): a replay is NOT a live
  session and a shadow decision is NEVER an outcome or a training label;
* with no real database the model-quality gate is ``NOT EVALUATED`` -- never faked from synthetic data.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_eval as ev
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_time_correct as tc
from cfb_coach.madden.model import historical_defense_baseline as hb

READINESS_VERSION = "offense_readiness.v1"
LATENCY_BUDGET_MS = 150.0
MIN_GAMES_FOR_CLAIM = ev.MIN_GAMES_FOR_CLAIM


# ------------------------------------------------------------------------------------------ 1.1 data audit

def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect("file:" + quote(str(p), safe="/") + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_rows(db_path: str | Path) -> he.EvalData:
    return he.load_eval_data(db_path)


def audit_data(data: he.EvalData, db_path: str | Path | None = None) -> dict[str, Any]:
    """What is actually usable for supervised training, and what is not. Counts only; no raw personal data."""
    rows = data.rows
    off = [r for r in rows if r.get("side") == "offense"]
    book = {(p["formation"], p["play"]) for p in oa.load_effective()["plays"]}
    verified = [r for r in off if he.is_verified_offense(r)]
    labelled = [r for r in verified if he.offense_label(r, "success") is not None]
    cov = [hb.normalize_label(r.get("coverage_seen")) for r in off]
    named = [c for c in cov if c in hb.KNOWN_COVERAGES]
    generic = [c for c in named if c in hb.GENERIC_TO_BUCKET]
    games = {str(r.get("game_id")) for r in off}
    by_game = Counter(str(r.get("game_id")) for r in verified)
    kinds = Counter(str(he._kind(r)) for r in off if r.get("result") not in (None, ""))
    tc_rows = tc.annotate(rows)
    out: dict[str, Any] = {
        "games_total": len({str(r.get("game_id")) for r in rows if r.get("game_id")}),
        "games_with_offense_snaps": len(games),
        "games_with_verified_offense": len({g for g, n in by_game.items() if n}),
        "offense_snaps": len(off),
        "verified_offense_snaps": len(verified),
        "verified_with_success_label": len(labelled),
        "verified_by_opponent_type": dict(Counter(r.get("opponent_type") or "unknown" for r in verified)),
        "games_by_opponent_type": dict(Counter(
            {str(r.get("game_id")): (r.get("opponent_type") or "unknown") for r in off}.values())),
        "verified_basis": dict(Counter(he.verification_basis(r) for r in verified)),
        "distinct_opponents": len({r.get("opponent_id") for r in verified}),
        "verified_on_installed_93": sum((r.get("action_formation"), r.get("action_play")) in book for r in verified),
        "verified_distinct_installed_plays": len({(r.get("action_formation"), r.get("action_play")) for r in verified
                                                  if (r.get("action_formation"), r.get("action_play")) in book}),
        "distinct_formations_played": len({r.get("action_formation") for r in verified}),
        "offense_snaps_with_named_coverage": len(named),
        "offense_snaps_generic_coverage_label": len(generic),
        "offense_snaps_pressure_only": sum(c == hb.PRESSURE_UNKNOWN for c in cov),
        "named_coverage_rate_of_offense_snaps": round(len(named) / len(off), 3) if off else None,
        "results_by_kind": dict(kinds),
        "with_yards": sum(r.get("yards") not in (None, "") for r in off),
        "with_down_distance_yardline": sum(all(r.get(k) not in (None, "") for k in ("down", "distance", "yardline")) for r in off),
        "with_quarter": sum(r.get("quarter") not in (None, "") for r in off),
        "adjustment_records": {
            "rows_with_recommended_macro": sum(bool(r.get("recommended_macro")) for r in off),
            "rows_with_confirmed_adjustment_outcomes": 0,
            "note": "hot-route executions were never logged; adjustment outcomes cannot supervise the model",
        },
        "same_game_coverage_history_available_for": sum(r[tc.KEY]["n_named"] > 0 for r in tc_rows if r.get("side") == "offense"),
        "not_available_in_training_rows": ["score", "game clock", "timeouts", "drive id"],
        "unreliable_or_unusable": {
            "unverified_execution": len(off) - len(verified),
            "verified_but_no_success_label": len(verified) - len(labelled),
            "coverage_label_provenance": "stored labels are manual entries, not independently post-play verified",
            "generic_coverage_labels_kept_generic": len(generic),
        },
        "usable_for_supervised_training": len(labelled),
    }
    if db_path:
        try:
            conn = open_readonly(db_path)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(snaps)")}
            out["snaps_columns_present"] = sorted(c for c in ("session_id", "snap_seq", "ml_snap_id", "quarter", "coverage_seen",
                                                              "concept_seen", "macro") if c in cols)
            conn.close()
        except sqlite3.Error as exc:
            out["snaps_columns_present"] = f"unreadable: {exc}"
    return out


# ------------------------------------------------------------------------------------------ 1.4/1.5 extras

def _slices(row: Mapping[str, Any]) -> dict[str, Any]:
    d, dist, yl = he._int(row.get("down")), he._int(row.get("distance")), he._int(row.get("yardline"))
    return {"short_yardage": bool(dist is not None and dist <= 3), "third_fourth": d in (3, 4),
            "long_yardage": bool(dist is not None and dist >= 8), "red_zone": bool(yl is not None and yl >= 80),
            "own_territory": bool(yl is not None and yl < 40)}


def evaluate_models(data: he.EvalData, *, n_boot: int = 1000, min_train_games: int = 3) -> dict[str, Any]:
    """Held-out (whole-game, chronological) comparison with slices, train-vs-validation gap and coverage sensitivity."""
    rows = tc.annotate(data.rows)
    data = he.EvalData(rows=rows, game_order=data.game_order, decisions=data.decisions, source=data.source)
    rep = ev.evaluate(data, n_boot=n_boot, min_train_games=min_train_games, run_behavior=True)
    folds = he.make_folds(rows, data.game_order, is_test_row=he.is_verified_offense, mode="chronological",
                          min_train_games=min_train_games)
    if not folds:
        return rep
    preds, _ = he.run_folds(rows, folds, ev.predictive_models, targets=("success",), label_fn=he.offense_label,
                            is_test_row=he.is_verified_offense, extra=_slices)
    sl: dict[str, Any] = {}
    for name in ("short_yardage", "third_fourth", "long_yardage", "red_zone", "own_territory"):
        sub = [p for p in preds if p.get(name)]
        sl[name] = {m: {k: v for k, v in he.binary_metrics([p["y"] for p in sub if m in p["p"]],
                                                           [p["p"][m] for p in sub if m in p["p"]]).items() if k != "calibration"}
                    for m in ("base_rate", "experimental_shrinkage", "structural_features")}
        if len(sub) >= 5:
            sl[name]["structural_vs_experimental"] = he.paired_bootstrap(sub, "structural_features", "experimental_shrinkage",
                                                                         n_boot=n_boot, seed=he.DEFAULT_SEED)
    rep["slices"] = sl
    rep["generalization_gap"] = generalization_gap(rows, folds)
    rep["coverage_sensitivity"] = coverage_sensitivity(rows, folds)
    return rep


def generalization_gap(rows: Sequence[Mapping[str, Any]], folds: Sequence[he.Fold]) -> dict[str, Any]:
    by_game: dict[str, list] = {}
    for r in rows:
        by_game.setdefault(str(r.get("game_id")), []).append(r)
    ins, outs = {"experimental": [], "structural": []}, {"experimental": [], "structural": []}

    def ll(y: int, p: float) -> float:
        p = min(1 - 1e-6, max(1e-6, p))
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    for f in folds:
        train = [r for g in f.train_games for r in by_game.get(g, [])]
        test = [r for r in by_game.get(f.test_game, []) if he.is_verified_offense(r)]
        art = sm.train(train)
        for tag, group in (("in", [r for r in train if he.is_verified_offense(r)]), ("out", test)):
            for r in group:
                y = he.offense_label(r, "success")
                if y is None:
                    continue
                s = sm.structure_any(str(r.get("action_formation") or ""), str(r.get("action_play") or ""))
                sc = sm.score(art, s, down=r.get("down"), distance=r.get("distance"), yardline=r.get("yardline"),
                              opponent_id=r.get("opponent_id"), opponent_type=r.get("opponent_type"), cov=r.get(tc.KEY))
                tgt = ins if tag == "in" else outs
                tgt["experimental"].append(ll(y, sc.base_probability))
                tgt["structural"].append(ll(y, sc.probability))
    m = lambda xs: round(statistics.fmean(xs), 4) if xs else None
    return {"train_log_loss": {k: m(v) for k, v in ins.items()}, "heldout_log_loss": {k: m(v) for k, v in outs.items()},
            "gap_structural": (m(outs["structural"]) - m(ins["structural"])) if outs["structural"] else None,
            "gap_experimental": (m(outs["experimental"]) - m(ins["experimental"])) if outs["experimental"] else None,
            "note": "in-sample numbers use the fold's own base fit, so they are optimistic for both; compare the gaps"}


def coverage_sensitivity(rows: Sequence[Mapping[str, Any]], folds: Sequence[he.Fold], *, per_fold: int = 25) -> dict[str, Any]:
    """How much the model's ranking of all installed plays moves when ONLY the coverage belief changes (not quality)."""
    by_game: dict[str, list] = {}
    for r in rows:
        by_game.setdefault(str(r.get("game_id")), []).append(r)
    book = [(p["formation"], p["play"]) for p in oa.load_effective()["plays"]]
    shifts, flips, n = [], 0, 0
    for f in folds[-3:]:
        train = [r for g in f.train_games for r in by_game.get(g, [])]
        art = sm.train(train)
        for r in [x for x in by_game.get(f.test_game, []) if he.is_verified_offense(x)][:per_fold]:
            ps = {}
            for tag, mix in (("man", {"cover_2_man": 1.0}), ("zone", {"cover3": 1.0})):
                ps[tag] = {b: sm.score(art, sm.structure_any(*b), down=r.get("down"), distance=r.get("distance"),
                                       yardline=r.get("yardline"), opponent_id=r.get("opponent_id"),
                                       opponent_type=r.get("opponent_type"), mix=mix).probability for b in book}
            shifts.append(statistics.fmean(abs(ps["man"][b] - ps["zone"][b]) for b in book))
            flips += max(ps["man"], key=ps["man"].get) != max(ps["zone"], key=ps["zone"].get)
            n += 1
    return {"snaps": n, "mean_abs_prob_shift_man_vs_zone": round(statistics.fmean(shifts), 5) if shifts else None,
            "top_pick_changes_rate": round(flips / n, 3) if n else None,
            "note": "sensitivity of the learned ranking to the coverage belief, not evidence the shifts are correct"}


# ------------------------------------------------------------------------------------------ 3.1 shadow replay

def shadow_replay(data: he.EvalData, *, out_path: Path, max_snaps_per_game: int = 80,
                  min_train_games: int = 3) -> dict[str, Any]:
    """Replay RECORDED held-out games snap by snap: both coordinators on the same time-correct state; log, never act."""
    rows = tc.annotate(data.rows)
    folds = he.make_folds(rows, data.game_order, is_test_row=he.is_verified_offense, mode="chronological",
                          min_train_games=min_train_games)
    by_game: dict[str, list] = {}
    for r in rows:
        by_game.setdefault(str(r.get("game_id")), []).append(r)
    book = [(p["formation"], p["play"]) for p in oa.load_effective()["plays"]]
    book_set = set(book)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = agree = illegal = fallbacks = 0
    lat: list[float] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for f in folds:
            train = [r for g in f.train_games for r in by_game.get(g, [])]
            art = sm.train(train)
            base_art = em.ExperimentalArtifact.from_dict(art.base)
            test = [r for r in by_game.get(f.test_game, []) if he.is_verified_offense(r)][:max_snaps_per_game]
            for r in test:
                t0 = time.perf_counter()
                rec: dict[str, Any] = {"schema": "madden27.route_aware.shadow.v1", "source": "recorded_game_replay",
                                       "kind": "shadow_decision_not_an_outcome", "game_id": f.test_game,
                                       "situation": {k: r.get(k) for k in ("down", "distance", "yardline", "quarter")},
                                       "opponent_type": r.get("opponent_type"),
                                       "prior_coverage_counts": r[tc.KEY]["counts"], "prior_named": r[tc.KEY]["n_named"]}
                try:
                    prior, _ = sm.pre_snap_mix(art, r.get("opponent_id"))
                    mix = tc.mix_from(prior, r[tc.KEY])
                    sit = od.Situation(down=r.get("down"), distance=r.get("distance"), yardline=r.get("yardline"),
                                       opponent_id=r.get("opponent_id"), opponent_type=r.get("opponent_type"), mix=mix)
                    dec = od.decide(art, sit, book, mode="live")
                    cur = em.rank_candidates(base_art, book, down=r.get("down"), distance=r.get("distance"),
                                             yardline=r.get("yardline"), opponent_id=r.get("opponent_id"),
                                             opponent_type=r.get("opponent_type"))[0]
                    legal = (dec.formation, dec.play) in book_set
                    plans_bad = [p.label for p in dec.plans if p.accepted and ex.validate_plan(
                        p.actions, ex.PlayContext("offense", dec.formation, dec.play, "pass",
                                                  eligible_receivers=dec.final_structure.skill_slots), require_live=True)]
                    rec.update(estimated_coverage=sorted(((k, round(v, 3)) for k, v in mix.items()), key=lambda kv: -kv[1])[:4],
                               recommended={"formation": dec.formation, "play": dec.play}, adjustment=dec.adjustment,
                               model_probability=round(dec.probability, 4), interval90=[round(x, 3) for x in dec.interval],
                               existing_coordinator={"formation": cur["formation"], "play": cur["play"],
                                                     "probability": round(cur["probability"], 4)},
                               agree=(dec.formation, dec.play) == (cur["formation"], cur["play"]), legal=legal,
                               illegal_adjustments=plans_bad, n_scored=len(dec.ranking), error="")
                    agree += rec["agree"]
                    illegal += (not legal) + len(plans_bad)
                except Exception as exc:  # noqa: BLE001
                    rec.update(error=f"{type(exc).__name__}: {exc}", legal=False)
                    fallbacks += 1
                rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                lat.append(rec["latency_ms"])
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                n += 1
    return {"records": n, "agree_with_existing": agree, "agree_rate": round(agree / n, 3) if n else None,
            "illegal": illegal, "errors_or_fallbacks": fallbacks, "latency_ms_p50": he.percentile(lat, 50),
            "latency_ms_p95": he.percentile(lat, 95), "latency_ms_max": max(lat) if lat else None,
            "log": str(out_path), "source": "recorded_game_replay (NOT a live CPU session)"}


# ------------------------------------------------------------------------------------------ 3.2 scenarios A-F

SITUATIONS = {
    "first_and_ten": dict(down=1, distance=10, yardline=35), "second_and_long": dict(down=2, distance=12, yardline=40),
    "third_and_short": dict(down=3, distance=2, yardline=45), "third_and_long": dict(down=3, distance=11, yardline=40),
    "fourth_and_2": dict(down=4, distance=2, yardline=60), "fourth_and_goal": dict(down=4, distance=1, yardline=99),
    "red_zone": dict(down=2, distance=7, yardline=85), "goal_line": dict(down=1, distance=2, yardline=98),
    "backed_up": dict(down=2, distance=8, yardline=4),
    "two_minute_proxy": dict(down=2, distance=10, yardline=50),
}


def scenario_suite(art: sm.StructuralArtifact, workdir: Path) -> dict[str, Any]:
    """Scenarios A-F against the actual artifact. A/B go through the REAL live controller + existing post-play form."""
    from cfb_coach.db import CoachDB
    from cfb_coach.live_server import LivePlayController
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model import experimental_live as xl
    from cfb_coach.madden.model import inference
    from cfb_coach.madden.model import offense_route_ranker as rr
    from cfb_coach.madden.model.schema import CoachingMode
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation

    book_map = {f: list(p) for f, p in json.loads(oa.SNAPSHOT_PATH.read_text())["sides"]["offense"]["formations"].items()}
    pairs = [(f, p) for f, ps in book_map.items() for p in ps]
    pair_set = set(pairs)
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}

    # ---- A/B through the live controller (temp DB; the user's database is never touched)
    db = CoachDB(workdir / "scenario_madden27.db", seed={"opponents": {"cpu": {
        "display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low", "profile_json": "{}"}}})
    try:
        playbook._save_state(db, {"applied": {"offense": {
            "side": "offense", "mode": "custom", "name": "ML Designed Offense (custom)", "rev": 1, "source_book": "Buccaneers",
            "trimmed": True, "formations": book_map, "audibles": {}, "core": list(book_map), "locked_ts": "scenario"}}, "pending": {}})
        exp_path = workdir / "scenario_offense.json"
        em.save_artifact(em.ExperimentalArtifact.from_dict(art.base), exp_path)
        xl.set_artifact_path(db, exp_path)
        art_path = sm.save_artifact(art, workdir / "scenario_route_aware.json")
        db.set_meta(rr.META_ARTIFACT, str(art_path))
        inference.set_mode(db, CoachingMode.EXPERIMENTAL)
        rr.set_mode(db, "route_aware")

        def ctrl() -> LivePlayController:
            c = LivePlayController(db=db, opponent_id="cpu",
                                   make_call=lambda sit, **kw: make_call(sit, "cpu", db, rng=random.Random(5), **kw),
                                   parse_situation=parse_madden_situation, learn_summary=lambda: "ok",
                                   brand="Madden 27 Franchise", cpu_only=True, enable_execution_verify=True)
            c.start()
            return c

        c = ctrl()
        c.call_only("1&10 my 25")
        trail, spots = [], ["2&5 my 30", "1&10 my 41", "2&7 my 45", "3&3 my 50", "1&10 my 60"]
        info0 = rr.info_of(c.last_call)
        trail.append({"step": 0, "entered": None, "named_n": info0["coverage"]["named_n"],
                      "cover_2_man": dict(info0["coverage"]["top"]).get("cover_2_man", 0.0),
                      "top5": info0["route_aware_top"], "call": [c.last_call.formation, c.last_call.play]})
        for i, spot in enumerate(spots):
            c.result_and_call(outcome="+5", sit_raw=spot, their="cover 2 man")
            inf = rr.info_of(c.last_call)
            trail.append({"step": i + 1, "entered": "cover 2 man", "named_n": inf["coverage"]["named_n"],
                          "cover_2_man": dict(inf["coverage"]["top"]).get("cover_2_man", 0.0), "top5": inf["route_aware_top"],
                          "call": [c.last_call.formation, c.last_call.play], "adjustment": inf.get("adjustment"),
                          "text_has_route_aware_lines": "ROUTE-AWARE:" in c.call_text})
        shares = [t["cover_2_man"] for t in trail]
        tops = [json.dumps(t["top5"]) for t in trail]
        out["A_repeated_cover_2_man"] = {
            "trail": trail, "belief_monotone_increasing": all(b >= a for a, b in zip(shares, shares[1:])) and shares[-1] > shares[0],
            "named_counts_follow_form_entries": [t["named_n"] for t in trail] == list(range(len(trail))),
            "rankings_recomputed_each_snap": len(set(tops)) > 1,
            "recommendation_changed_at_least_once": len({tuple(t["call"]) for t in trail}) > 1,
            "note": "the call is not required to change; probabilities feed the scoring and the ranking is recomputed",
        }
        seq = ["cover 3 match"] * 5
        for i, spot in enumerate(["2&6 my 62", "1&10 my 70", "2&4 my 74", "1&10 my 80", "2&8 my 83"]):
            c.result_and_call(outcome="+4", sit_raw=spot, their=seq[i])
            inf = rr.info_of(c.last_call)
            trail.append({"step": len(trail), "entered": seq[i], "named_n": inf["coverage"]["named_n"],
                          "cover_2_man": dict(inf["coverage"]["top"]).get("cover_2_man", 0.0),
                          "cover_3_match": dict(inf["coverage"]["top"]).get("cover_3_match", 0.0)})
        man_peak = max(t["cover_2_man"] for t in trail[:len(spots) + 1])
        out["B_coverage_change"] = {"cover_2_man_peak": round(man_peak, 3), "cover_2_man_after_change": round(trail[-1]["cover_2_man"], 3),
                                    "cover_3_match_after_change": round(trail[-1].get("cover_3_match", 0.0), 3),
                                    "beliefs_move_away_from_previous_look": trail[-1]["cover_2_man"] < man_peak - 0.03,
                                    "does_not_assume_repeat": trail[-1].get("cover_3_match", 0.0) > 0.1}
        c2 = ctrl()
        c2.call_only("1&10 my 25")
        out["A_new_game_isolated"] = rr.info_of(c2.last_call)["coverage"]["named_n"] == 0

        # ---- D (live path): diversity through the REAL loop incl. the existing anti-repeat policy, both rankers, same states
        def live_run(mode: str, n: int = 40) -> dict[str, Any]:
            rr.set_mode(db, mode)
            cc = ctrl()
            cc.call_only("1&10 my 25")
            picks, repeats, elig = [], 0, set()
            rnd = random.Random(9)
            covs = ["cover 3", "cover 2 man", "cover 4 quarters", "cover 1 hole"]
            for i in range(n):
                call = (cc.last_call.formation, cc.last_call.play)
                picks.append(call)
                repeats += len(picks) > 1 and picks[-1] == picks[-2]
                ml = getattr(cc.last_call, "ml_experimental", {}) or {}
                elig.add(ml.get("n_eligible_candidates"))
                down = rnd.choice([1, 2, 3])
                cc.result_and_call(outcome=rnd.choice(["+3", "+7", "incomplete", "+11"]), sit_raw=f"{down}&{rnd.choice([3, 6, 10])} my {rnd.randint(21, 79)}",
                                   their=rnd.choice(covs))
            return {"snaps": n, "distinct_plays": len(set(picks)), "distinct_formations": len({f for f, _ in picks}),
                    "consecutive_repeat_rate": round(repeats / (n - 1), 3), "eligible_candidates_seen": sorted(x for x in elig if x),
                    "all_installed": all(p in pair_set for p in picks)}

        out["D_live_diversity"] = {"route_aware": live_run("route_aware"), "current_ranker": live_run("experimental"),
                                   "note": "measured, never enforced; the existing anti-repeat selection policy is in both paths"}
    finally:
        db.close()

    # ---- C: situational football (pre-snap inputs only), D: diversity, E: legality, F: latency
    rows_c, lat, picks, illegal = {}, [], Counter(), 0
    for name, s in SITUATIONS.items():
        sit = od.Situation(opponent_id="cpu", opponent_type="cpu", mix={"cover_2_man": 0.4, "cover3": 0.4, "cover4": 0.2}, **s)
        d = od.decide(art, sit, pairs, mode="live")
        lat.append(d.latency_ms)
        legal = (d.formation, d.play) in pair_set
        illegal += not legal
        picks[(d.formation, d.play)] += 1
        rows_c[name] = {"pick": [d.formation, d.play], "p": round(d.probability, 3), "adjustment": d.adjustment, "legal": legal,
                        "n_scored": len(d.ranking), "type": d.final_structure.hint.get("play_type"),
                        "punt_or_kick_recommended": any(w in d.play.lower() for w in ("punt", "field goal", "kick"))}
    out["C_situations"] = {"cases": rows_c, "all_legal": illegal == 0, "all_93_scored": all(v["n_scored"] == 93 for v in rows_c.values()),
                           "no_punt_or_kick": not any(v["punt_or_kick_recommended"] for v in rows_c.values()),
                           "not_modeled": ["score", "game clock", "timeouts"],
                           "note": "punt/field-goal decisions are the user's; the offensive book contains none"}
    allr, spread = [], Counter()
    rnd = random.Random(3)
    for _ in range(120):
        sit = od.Situation(down=rnd.choice([1, 2, 3]), distance=rnd.choice([2, 5, 8, 12]), yardline=rnd.randint(5, 95),
                           opponent_id="cpu", opponent_type="cpu")
        d = od.decide(art, sit, pairs, mode="live", allow_adjustments=False)
        lat.append(d.latency_ms)
        allr.append(d)
        spread[(d.formation, d.play)] += 1
    out["D_diversity_raw_model"] = {"eligible_plays_scored_per_decision": sorted({len(d.ranking) for d in allr}),
                          "distinct_plays_chosen_of_93": len(spread), "distinct_formations_chosen": len({f for f, _ in spread}),
                          "top_play_share": round(max(spread.values()) / sum(spread.values()), 3),
                          "forced_rotation": False,
                          "note": "diversity is measured, never enforced"}
    adj = [od.decide(art, od.Situation(down=3, distance=7, yardline=45, opponent_id="cpu", opponent_type="cpu"), pairs,
                     mode="research", margin=0.0, z=0.0) for _ in range(1)]
    d_live = od.decide(art, od.Situation(down=3, distance=7, yardline=45, opponent_id="cpu", opponent_type="cpu"), pairs, mode="live")
    out["E_adjustment_legality"] = {"live_adjustment": d_live.adjustment, "no_adjustment_is_valid_choice": True,
                                    "unconfirmed_adjustment_never_live": d_live.adjustment == od.NO_ADJUSTMENT or d_live.executable_now,
                                    "research_plans_considered": len(adj[0].plans),
                                    "plans_rejected_reasons": dict(Counter(p.reject_reason.split(":")[0][:60] for p in adj[0].plans if not p.accepted))}
    out["F_latency"] = {"decisions": len(lat), "p50_ms": he.percentile(lat, 50), "p95_ms": he.percentile(lat, 95),
                        "max_ms": max(lat), "budget_ms": LATENCY_BUDGET_MS, "within_budget": max(lat) < LATENCY_BUDGET_MS}
    return out


# ------------------------------------------------------------------------------------------ 3.4 readiness gates

def gates(*, scenarios: Mapping[str, Any], shadow: Mapping[str, Any] | None, evaluation: Mapping[str, Any] | None,
          tests: Mapping[str, Any] | None, real_data: bool) -> dict[str, Any]:
    A, B = scenarios["A_repeated_cover_2_man"], scenarios["B_coverage_change"]
    g: dict[str, Any] = {}

    def gate(name: str, ok: bool | None, detail: str, kind: str) -> None:
        g[name] = {"status": "PASS" if ok else ("NOT EVALUATED" if ok is None else "FAIL"), "detail": detail, "kind": kind}

    gate("coverage_feedback_connected", bool(A["named_counts_follow_form_entries"] and A["belief_monotone_increasing"]
                                              and A["rankings_recomputed_each_snap"]),
         "existing post-play form -> time-correct mix -> structural scoring -> ranking recomputed each snap", "integration")
    gate("no_cross_game_or_future_leakage", bool(scenarios["A_new_game_isolated"]) and B["beliefs_move_away_from_previous_look"],
         "new game starts with 0 coverage entries; beliefs follow only earlier snaps", "integration")
    ill = scenarios["C_situations"]["all_legal"] and scenarios["C_situations"]["all_93_scored"] and scenarios["C_situations"]["no_punt_or_kick"]
    if shadow:
        ill = ill and shadow["illegal"] == 0
    gate("no_invalid_formations_plays_or_adjustments", ill, "every pick is an installed play; every adjustment passed the live registry check", "safety")
    gate("adjustment_gate", scenarios["E_adjustment_legality"]["unconfirmed_adjustment_never_live"],
         "an unconfirmed adjustment is never offered live; NO ADJUSTMENT stays valid", "safety")
    gate("latency_within_150ms", scenarios["F_latency"]["within_budget"] and (shadow is None or (shadow["latency_ms_max"] or 0) < LATENCY_BUDGET_MS),
         f"max {scenarios['F_latency']['max_ms']:.1f} ms over {scenarios['F_latency']['decisions']} decisions", "performance")
    gate("interface_displays_recommendation", bool(all(t.get("text_has_route_aware_lines", True) for t in A["trail"][1:])),
         "route-aware lines present in the live call text on every snap", "integration")
    if shadow:
        gate("shadow_replay_clean", shadow["errors_or_fallbacks"] == 0, f"{shadow['records']} replayed decisions, {shadow['errors_or_fallbacks']} errors", "safety")
    if tests is not None:
        gate("regression_and_integration_tests", tests.get("failed_new", 0) == 0,
             f"{tests.get('passed')} passed / {tests.get('failed')} failed (pre-existing: {tests.get('preexisting_failures')})", "integration")
    if not real_data or not evaluation or not evaluation.get("folds"):
        gate("model_quality_vs_existing_coordinator", None, "NOT EVALUATED: no real game database was available to this run", "quality")
    else:
        m = evaluation["metrics"]["success"]["vs_reference"].get("structural_features")
        games = evaluation["data"]["verified_offense_games"]
        if games < MIN_GAMES_FOR_CLAIM or not m:
            gate("model_quality_vs_existing_coordinator", None, f"INSUFFICIENT DATA: {games} verified games", "quality")
        elif m["hi"] < 0:
            gate("model_quality_vs_existing_coordinator", True, f"log-loss diff {m['mean_diff']:+.4f} CI [{m['lo']:+.4f},{m['hi']:+.4f}] favors route-aware", "quality")
        elif m["lo"] > 0:
            gate("model_quality_vs_existing_coordinator", False, f"log-loss diff {m['mean_diff']:+.4f} CI [{m['lo']:+.4f},{m['hi']:+.4f}] is WORSE", "quality")
        else:
            gate("model_quality_vs_existing_coordinator", None, f"no demonstrated difference: {m['mean_diff']:+.4f} CI [{m['lo']:+.4f},{m['hi']:+.4f}]", "quality")
    hard = [k for k, v in g.items() if v["status"] == "FAIL" and v["kind"] in ("integration", "safety", "performance")]
    quality = g["model_quality_vs_existing_coordinator"]["status"]
    if hard:
        verdict, why = "NO-GO", f"failed gates: {', '.join(hard)}"
    elif quality == "FAIL":
        verdict, why = "NO-GO", "route-aware model is demonstrably worse than the existing coordinator on held-out games"
    elif quality == "PASS" and real_data:
        verdict, why = "GO", "integration/safety gates pass and held-out evidence favors the route-aware model"
    else:
        verdict, why = "CONDITIONAL GO", ("integration/safety gates pass; model quality is " +
                                          ("NOT EVALUATED here" if quality == "NOT EVALUATED" else "not shown to be better") +
                                          ". Use route_aware_preview first; do not make it primary until quality is demonstrated.")
    return {"gates": g, "verdict": verdict, "reason": why}
