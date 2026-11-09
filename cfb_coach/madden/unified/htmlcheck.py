"""Validate the REAL browser integration: start the actual HTTP server (the established page + handler) over the unified
session and drive it over HTTP exactly as the page's JavaScript does, 100+ snaps per side across possession changes.

Request latency is measured at the client from sending the POST to receiving the full JSON response (so it includes
parsing, the unified decision, persistence, state building and serialisation). The store is then reloaded cold and audited,
and the primary-database trap proves nothing opened ``CoachDB``.
"""
from __future__ import annotations

import http.client
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from cfb_coach.madden.unified import livepath as LP
from cfb_coach.madden.unified.html_controller import UnifiedLiveController
from cfb_coach.madden.unified.html_server import start_server
from cfb_coach.madden.unified.session import UnifiedSession
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.types import DEF, OFF
from cfb_coach.madden.unified.world_model import WorldModel

RESULTS = ["gain 7", "incomplete", "loss 2", "gain 15", "gain 3", "gain 1", "INT", "gain 9"]
MODE_TO_HTML = {"y": "used_recommended", "u": "unknown", "other": "used_different"}


class Client:
    def __init__(self, url: str) -> None:
        host, port = url.split("//")[1].rstrip("/").split(":")
        self.host, self.port = host, int(port)

    def req(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> tuple[int, Any, float]:
        c = http.client.HTTPConnection(self.host, self.port, timeout=30)
        t = time.perf_counter()
        c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        ms = (time.perf_counter() - t) * 1e3
        c.close()
        ct = r.getheader("Content-Type") or ""
        return r.status, (json.loads(raw) if "json" in ct else raw.decode()), ms


def _truth_and_requests(books: Mapping[str, Mapping[str, Any]], n_per_side: int) -> list[dict[str, Any]]:
    opp_off = LP._opp_offense_calls()
    order: list[str] = []
    cnt = {OFF: 0, DEF: 0}
    while min(cnt.values()) < n_per_side:
        for side in (OFF, OFF, OFF, DEF, DEF):
            if cnt[side] < n_per_side:
                order.append(side)
                cnt[side] += 1
    plan = []
    for i, side in enumerate(order):
        down, dist, yl = 1 + i % 4, 2 + (i * 3) % 10, 10 + (i * 7) % 40
        mode = ["y", "y", "y", "u", "other", "y"][i % 6]
        plan.append({"i": i, "side": side, "mode": mode, "down": down, "dist": dist, "yl": yl, "yl_side": "my" if i % 2 else "opp",
                     "result": RESULTS[i % len(RESULTS)], "opp": LP.OPP_DEF[i % len(LP.OPP_DEF)] if side == OFF else opp_off[i % len(opp_off)]})
    return plan


def run_http_workflow(books: Mapping[str, Mapping[str, Any]], *, model_path: str | None, store_path: str | Path, n_per_side: int = 100,
                      confirmations: Mapping[str, Any] | None = None, game_id: str = "htmlcheck-1") -> dict[str, Any]:
    from cfb_coach import db as coachdb
    from cfb_coach.madden import playcaller

    # traps: any legacy caller or any CoachDB construction during this run is a hard failure
    orig_call, orig_init = playcaller.make_call, coachdb.CoachDB.__init__
    hits: list[str] = []

    def _boom_call(*a, **k):
        hits.append("playcaller.make_call")
        raise AssertionError("legacy playcaller.make_call invoked from the unified HTML path")

    def _boom_db(self, *a, **k):
        hits.append("CoachDB")
        raise AssertionError("CoachDB opened from the unified HTML path")

    playcaller.make_call, coachdb.CoachDB.__init__ = _boom_call, _boom_db
    store = UnifiedStore(store_path)
    wm = WorldModel.from_dict(json.loads(Path(model_path).read_text())) if model_path else None
    art = (Path(model_path).name + ":" + LP.guard.sha256_file(Path(model_path))[:10]) if model_path else "cold_start_prior_v1"
    session = UnifiedSession(books=books, opponent_id="cpu_htmlcheck", store=store, model=wm, artifact=art, game_id=game_id,
                             opponent_class="cpu", adjustment_confirmations=confirmations)
    ctrl = UnifiedLiveController(session, model_path=model_path)
    server, th, url = start_server(ctrl, port=None)
    cl = Client(url)
    plan = _truth_and_requests(books, n_per_side)
    inv = {s: books[s]["inventory"] for s in (OFF, DEF)}
    lat: list[float] = []
    errors: list[str] = []
    page_ok = False
    try:
        code, html, _ = cl.req("GET", "/")
        page_ok = code == 200 and "unified-panel" in html and "Submit" in html and "btn-submit" in html
        prev = None
        for step in plan:
            body: dict[str, Any] = {"sit": f"{step['down']}&{step['dist']} {step['yl_side']} {step['yl']}", "side": "offense" if step["side"] == OFF else "defense",
                                    "quarter": "1", "outcome": prev["result"] if prev else "", "their": prev["opp"] if prev else "",
                                    "executed_status": MODE_TO_HTML[prev["mode"]] if prev else "unknown"}
            if prev and prev["mode"] == "other":
                side = prev["side"]
                f = sorted(inv[side])[0]
                p = inv[side][f][-1]
                prev["other"] = (f, p)
                body["executed_formation"], body["executed_play"] = f, p
            if prev and prev["mode"] == "y":
                body["applied_recommended_action"] = True
            code, data, ms = cl.req("POST", "/api/result_call", body)
            lat.append(ms)
            if code != 200 or not data.get("ok"):
                errors.append(f"step {step['i']}: {code} {data}")
                break
            st = data["state"]
            if st["pending_recommendation"]["side"] != ("offense" if step["side"] == OFF else "defense"):
                errors.append(f"step {step['i']}: wrong side card")
            prev = step
        # the last pending snap gets its result through the end-of-game path so every scripted snap is logged
        if prev and not errors:
            code, data, ms = cl.req("POST", "/api/result_call", {"sit": "1&10 my 25", "side": "offense", "outcome": prev["result"], "their": prev["opp"],
                                                                  "executed_status": MODE_TO_HTML[prev["mode"]],
                                                                  "executed_formation": (sorted(inv[prev["side"]])[0] if prev["mode"] == "other" else None),
                                                                  "executed_play": (inv[prev["side"]][sorted(inv[prev["side"]])[0]][-1] if prev["mode"] == "other" else None),
                                                                  "applied_recommended_action": prev["mode"] == "y"})
            if prev["mode"] == "other":
                prev["other"] = (sorted(inv[prev["side"]])[0], inv[prev["side"]][sorted(inv[prev["side"]])[0]][-1])
            if code != 200 or not data.get("ok"):
                errors.append(f"final: {data}")
        cl.req("POST", "/api/end_game", {"result_wl": "win", "score": "0-0"})
    finally:
        server.shutdown()
        server.server_close()
        playcaller.make_call, coachdb.CoachDB.__init__ = orig_call, orig_init
    store.close()
    # the extra closing request opened one more (unplayed) offensive snap: it is decided but unlearned, so it is not in `plan`
    return {"rc": 0 if not errors else 1, "latency_start_ms": lat, "latency_finish_ms": [], "truth": plan, "store_path": str(store_path), "model_path": model_path,
            "game_id": game_id, "page_ok": page_ok, "errors": errors, "legacy_hits": hits, "n_per_side": n_per_side}


def audit_http_run(run: Mapping[str, Any], books: Mapping[str, Mapping[str, Any]], *, confirmations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    store = UnifiedStore(run["store_path"])
    n_decisions = store.counts()["decisions"]
    store.close()
    plan = run["truth"]
    # decisions = scripted snaps + 1 trailing unplayed snap; audit the scripted prefix
    core = dict(run)
    a = _audit_prefix(core, books, confirmations, len(plan))
    a["page_ok"] = run["page_ok"]
    a["request_ms"] = {"n": len(run["latency_start_ms"]), "p50": statistics.median(run["latency_start_ms"]) if run["latency_start_ms"] else None,
                       "p95": LP.pctl(run["latency_start_ms"], 0.95), "max": max(run["latency_start_ms"]) if run["latency_start_ms"] else None}
    a["legacy_hits"] = run["legacy_hits"]
    a["errors"] = run["errors"]
    a["extra_unplayed_snaps"] = n_decisions - len(plan)
    a["ok"] = bool(a["ok"] and run["page_ok"] and not run["legacy_hits"] and not run["errors"] and a["extra_unplayed_snaps"] == 1)
    return a


def _audit_prefix(run: Mapping[str, Any], books, confirmations, n: int) -> dict[str, Any]:
    """Reuse the cold-reload audit on the first ``n`` decisions (the trailing unplayed snap is excluded)."""
    import sqlite3
    # audit_run zips decisions/observations with truth; zip truncates to the shorter list, so the trailing snap is ignored.
    return LP.audit_run({**run, "truth": run["truth"]}, books, confirmations=confirmations)
