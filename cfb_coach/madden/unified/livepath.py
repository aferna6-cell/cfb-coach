"""Benchmark and verify the ACTUAL live path: the same ``cmd_game`` loop the owner types into.

A scripted owner plays one continuous game (both sides, many possession changes) through ``cli.cmd_game`` with the real
books file, the real model artifact file and a real persisted store. Latency is measured between the owner pressing Enter
on a situation and the card being ready for the next prompt (parse + shared-state decision + persistence + render).
Afterwards the store is reloaded cold and every logged fact is checked against what the script did.
"""
from __future__ import annotations

import json
import statistics
import time
import types
from pathlib import Path
from typing import Any, Mapping

from cfb_coach.madden import catalog_full
from cfb_coach.madden.unified import guard
from cfb_coach.madden.unified.cli import cmd_game
from cfb_coach.madden.unified.session import UnifiedSession
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.types import DEF, OFF
from cfb_coach.madden.unified.world_model import WorldModel

SIT_FORMATS = ["{s} {d}&{n} my {y}", "{s} {d}&{n} opp {y}", "{s} {d}rd and {n} on my {y}", "{s}: {d}&{n} at my {y} showing cover 3", "{s} {d}th & {n} opp {y}"]
RESULTS = ["+7", "inc", "-2", "sack -6", "+15", "+3", "+1", "int", "+9"]
OPP_DEF = ["4-3 Over | Cover 3 Sky", "4-3 Over | Cover 1 Hole", "4-3 Over | Cover 6", "Nickel Over | Cover 4 Palms", "cover 2 man"]


def _opp_offense_calls() -> list[str]:
    out = []
    for plays in catalog_full.book_plays("offense").values():
        for f, ps in plays.items():
            out += [f"{f} | {p}" for p in ps[:2]]
        if len(out) >= 8:
            break
    return out[:8]


def build_script(books: Mapping[str, Mapping[str, Any]], n_per_side: int) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the typed input lines and, per snap, what the owner 'did' (ground truth for the post-run audit)."""
    opp_off = _opp_offense_calls()
    lines: list[str] = []
    truth: list[dict[str, Any]] = []
    counters = {OFF: 0, DEF: 0}
    order: list[str] = []
    while min(counters.values()) < n_per_side:
        for side in (OFF, OFF, OFF, DEF, DEF):  # irregular possessions, many switches
            if counters[side] < n_per_side:
                order.append(side)
                counters[side] += 1
    for i, side in enumerate(order):
        tok = "o" if side == OFF else "d"
        down, dist, yl = 1 + i % 4, 2 + (i * 3) % 10, 10 + (i * 7) % 80
        fmt = SIT_FORMATS[i % len(SIT_FORMATS)]
        if "{d}rd" in fmt:
            down = 3
        if "{d}th" in fmt:
            down = 4
        line = fmt.format(s=tok, d=down, n=dist, y=yl)
        mode = ["y", "y", "y", "n", "u", "other"][i % 6]
        res = RESULTS[i % len(RESULTS)]
        opp = OPP_DEF[i % len(OPP_DEF)] if side == OFF else opp_off[i % len(opp_off)]
        lines.append((line, mode, res, opp))  # type: ignore[arg-type]
        truth.append({"i": i, "side": side, "mode": mode, "result": res, "opp": opp})
    return lines, truth  # type: ignore[return-value]


def run_live_benchmark(books: Mapping[str, Mapping[str, Any]], *, model_path: str | None, store_path: str | Path, n_per_side: int = 100,
                       confirmations: Mapping[str, Any] | None = None, game_id: str = "livepath-1") -> dict[str, Any]:
    plan, truth = build_script(books, n_per_side)
    books_file = Path(store_path).with_suffix(".books.json")
    books_file.write_text(json.dumps({s: {"side": s, **b} for s, b in books.items()}))
    conf_file = None
    if confirmations:
        conf_file = Path(store_path).with_suffix(".adj.json")
        conf_file.write_text(json.dumps(confirmations))
    stamps: dict[str, float] = {}
    lat_start: list[float] = []
    lat_finish: list[float] = []
    pending_mode = {"i": -1, "other_play": None}
    inv = {s: books[s]["inventory"] for s in (OFF, DEF)}
    feed = iter(range(10 ** 9))
    steps = iter([])

    def script():
        for i, (line, mode, res, opp) in enumerate(plan):
            yield ("sit", line, i)
            yield ("ex", mode, i)
            yield ("res", res, i)
            yield ("opp", opp, i)
        yield ("sit", "end", -1)

    gen = script()
    cur: dict[str, Any] = {}

    def input_fn(prompt: str = "") -> str:
        now = time.perf_counter()
        if prompt.startswith("ran it?") and "t_sit" in stamps:
            lat_start.append((now - stamps.pop("t_sit")) * 1e3)
        if prompt.startswith("opponent's call"):
            stamps["t_fin"] = now
        if prompt.startswith("sit>") and "t_fin" in stamps:
            lat_finish.append((now - stamps.pop("t_fin")) * 1e3)
        kind, val, i = next(gen)
        if kind == "sit":
            stamps["t_sit"] = time.perf_counter()
            cur["i"] = i
            return val
        if kind == "ex":
            if val == "other":
                side = truth[i]["side"]
                f = sorted(inv[side])[0]
                p = inv[side][f][-1]
                truth[i]["other"] = (f, p)
                return f"{f} | {p}"
            return val
        return val

    out: list[str] = []
    ns = types.SimpleNamespace(opponent="cpu_livepath", opponent_class="cpu", confirmed_books=str(books_file), store=str(store_path), model=model_path,
                               game_id=game_id, resume=False, verified_roles=None, adjustment_confirmations=str(conf_file) if conf_file else None)
    # prove there is no silent fallback to the established caller: make it explode if anything touches it
    from cfb_coach.madden import playcaller
    orig = playcaller.make_call

    def _boom(*a, **k):
        raise AssertionError("legacy playcaller.make_call was invoked from the unified path")

    playcaller.make_call = _boom
    t0 = time.perf_counter()
    try:
        rc = cmd_game(ns, input_fn=input_fn, out=out.append)
    finally:
        playcaller.make_call = orig
    wall = time.perf_counter() - t0
    return {"rc": rc, "wall_s": wall, "latency_start_ms": lat_start, "latency_finish_ms": lat_finish, "truth": truth, "output": out,
            "store_path": str(store_path), "n_per_side": n_per_side, "model_path": model_path, "game_id": game_id}


def pctl(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def audit_run(run: Mapping[str, Any], books: Mapping[str, Mapping[str, Any]], *, confirmations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Cold-reload the store and check every logged fact against the script."""
    store = UnifiedStore(run["store_path"])
    try:
        obs = store.load_observations()
    except ValueError as e:  # an inconsistent execution row is itself a failed audit, not a crash
        return {"ok": False, "snaps": len(run["truth"]), "problems": {"exec_mismatch": [f"store inconsistent: {e}"]}, "rc": run["rc"],
                "latency_start_ms": {"n": 0, "p50": None, "p95": float("inf"), "max": None}, "latency_finish_ms": {"p50": None, "p95": None},
                "timeouts_over_150ms": 0, "fallbacks": 0, "learning": {"matches": False}, "belief_sources_final": {}, "previous_play_learning_visible": False,
                "decision_model_values": [], "artifact_expected": None, "unique_ids": 0}
    rows = store.db.execute("SELECT snap_id, card_json, fell_back, latency_ms, model_version FROM decisions ORDER BY seq").fetchall()
    truth = run["truth"]
    inv = {s: {(f, p) for f, ps in books[s]["inventory"].items() for p in ps} for s in (OFF, DEF)}
    problems: dict[str, list] = {k: [] for k in ("duplicate_ids", "uninstalled", "exec_mismatch", "attribution", "fallback", "illegal_adjustment", "side", "seq")}
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        problems["duplicate_ids"].append("duplicate snap ids")
    n = len(truth)
    if len(rows) != n:
        problems["seq"].append(f"{len(rows)} decisions for {n} scripted snaps")
    art = (Path(run["model_path"]).name + ":" + guard.sha256_file(Path(run["model_path"]))[:10]) if run["model_path"] else "cold_start_prior_v1"
    expected_learn = {OFF: 0, DEF: 0}
    for (sid, cj, fb, lat, mv), o, t in zip(rows, obs, truth):
        card = json.loads(cj)
        if o.state.our_side != t["side"] or card["side"] != t["side"]:
            problems["side"].append(sid)
        if (card["formation"], card["play"]) not in inv[t["side"]]:
            problems["uninstalled"].append(sid)
        if card["decision_model"] != "unified_v1" or card["artifact"] != art or not str(card["model_version"]).startswith("unified_solver"):
            problems["attribution"].append(sid)
        if fb or card["fallback"]["occurred"]:
            problems["fallback"].append(sid)
        for a in (o.recommended.adjustments if o.recommended else ()):
            if not (confirmations and a.action_id in confirmations) or not o.recommended.executable:
                problems["illegal_adjustment"].append(sid)
        rec = (o.recommended.play.formation, o.recommended.play.play)
        if rec != (card["formation"], card["play"]):
            problems["exec_mismatch"].append(f"{sid}: card vs recommended")
        m = t["mode"]
        if m == "y" and not (o.executed_status == "confirmed" and o.executed and (o.executed.play.formation, o.executed.play.play) == rec):
            problems["exec_mismatch"].append(sid)
        elif m == "n" and not (o.executed_status == "not_executed" and o.executed is None):
            problems["exec_mismatch"].append(sid)
        elif m == "u" and not (o.executed_status == "unknown" and o.executed is None):
            problems["exec_mismatch"].append(sid)
        elif m == "other":
            f, p = t["other"]
            same = (f, p) == rec
            if not (o.executed and (o.executed.play.formation, o.executed.play.play) == (f, p) and o.executed_status == ("confirmed" if same else "modified")):
                problems["exec_mismatch"].append(sid)
        if m in ("y", "other") and o.outcome and o.outcome.known and o.opponent_action:
            expected_learn[t["side"]] += 1
    S2 = UnifiedSession(books=books, opponent_id="cpu_livepath", store=store, game_id=run["game_id"] + "-reload", model=WorldModel())
    prov = S2.history_report.get("provenance", {})
    learn_ok = (prov.get("our_offense_executed", 0), prov.get("our_defense_executed", 0)) == (expected_learn[OFF], expected_learn[DEF])
    last_src = {}
    for sid, cj, *_ in rows:
        c = json.loads(cj)
        last_src[c["side"]] = c["opponent_prediction_source"]
    start = run["latency_start_ms"]
    res = {"snaps": n, "unique_ids": len(set(ids)), "problems": {k: v[:5] for k, v in problems.items()},
           "latency_start_ms": {"n": len(start), "p50": statistics.median(start) if start else None, "p95": pctl(start, 0.95), "max": max(start) if start else None},
           "latency_finish_ms": {"p50": statistics.median(run["latency_finish_ms"]) if run["latency_finish_ms"] else None, "p95": pctl(run["latency_finish_ms"], 0.95)},
           "timeouts_over_150ms": sum(1 for x in start if x > 150.0), "fallbacks": sum(r[2] for r in rows),
           "learning": {"expected_world_model_rows": expected_learn, "replayed_rows_by_provenance": prov, "matches": learn_ok},
           "belief_sources_final": last_src, "previous_play_learning_visible": all("game(" in v for v in last_src.values()),
           "decision_model_values": sorted({json.loads(r[1])["decision_model"] for r in rows}), "artifact_expected": art, "rc": run["rc"]}
    res["ok"] = (not any(problems.values()) and learn_ok and res["previous_play_learning_visible"] and res["rc"] == 0 and n == len(obs))
    store.close()
    return res
