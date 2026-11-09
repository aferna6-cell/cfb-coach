"""CPU-game preflight: real checks, honest verdict (GO / SHADOW ONLY / NO-GO).

Nothing here is faked: when the owner's confirmed installed book, database or champion artifact is
not present, the corresponding gate is reported as NOT RUN with the exact command to run locally,
and the verdict cannot be GO.
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.unified import guard, knowledge as K
from cfb_coach.madden.unified.coordinator import (LATENCY_BUDGET_MS, MAX_FORMATIONS, InstalledBookError, UnifiedCoordinator,
                                                  inventory_fp)
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, OFF, FootballAction, GameState, OpponentAction, Outcome,
                                            SnapObservation)
from cfb_coach.madden.unified.world_model import WorldModel

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT = REPO / "cfb_coach" / "data" / "madden27" / "installed_playbook_snapshot.json"


def _gate(name: str, ok: bool | None, evidence: Any, *, critical: bool = True, how_to_run: str | None = None) -> dict[str, Any]:
    st = "NOT_RUN" if ok is None else "PASS" if ok else "FAIL"
    g = {"gate": name, "status": st, "critical": critical, "evidence": evidence}
    if how_to_run:
        g["how_to_run"] = how_to_run
    return g


def load_confirmed_books(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def audit_db(path: str | Path) -> dict[str, Any]:
    """Read-only audit of a local coach database (sqlite URI mode=ro; never writes, never migrates)."""
    import sqlite3
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"ok": False, "error": f"{p} does not exist"}
    home = guard.DEFAULT_HOME.resolve()
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        cnt = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("snaps", "game_sessions", "play_records", "ml_decisions", "ml_outcomes") if t in tables}
        dyn = sorted({r[0] for r in con.execute("select distinct dynasty from game_sessions")}) if "game_sessions" in tables else []
        executed = con.execute("select count(*) from ml_outcomes where executed_status in ('identified','confirmed')").fetchone()[0] if "ml_outcomes" in tables else None
    finally:
        con.close()
    return {"ok": True, "path": str(p), "sha256": guard.sha256_file(p), "inside_primary_data_dir": home == p or home in p.parents,
            "tables": sorted(tables), "row_counts": cnt, "dynasties": dyn, "verified_executions": executed}


def p95(xs: Sequence[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def legality_report(co: UnifiedCoordinator) -> dict[str, Any]:
    inv = {(f, p) for f, ps in co.inventory.items() for p in ps}
    cands = co.legal_candidates()
    outside = [a.key for a in cands if (a.play.formation, a.play.play) not in inv]
    no_adj = {a.play.play_id for a in cands if a.no_adjustment}
    plays = {p.play_id for p in co.plays}
    return {"formations": len(co.inventory), "plays": len(plays), "candidates": len(cands),
            "candidates_outside_installed_book": outside[:5], "plays_missing_no_adjustment": sorted(plays - no_adj)[:5],
            "unexecutable_candidates": sum(1 for a in cands if not a.executable)}


def run_cycle(co: UnifiedCoordinator, *, game_id: str, n: int, opponent_class: str = "cpu", seed_defs: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """recommend -> record execution -> record revealed opponent call -> close -> next recommend. Returns evidence."""
    lat, fb, mism, chron = [], 0, 0, []
    defs = list(seed_defs) or [("4-3 Over", "Cover 3 Sky"), ("4-3 Over", "Cover 1 Hole"), ("Nickel Over", "Cover 4 Palms")]
    offs = [("Gun Trips X Nasty", "Mesh Spot")] if co.side == DEF else []
    for i in range(1, n + 1):
        st = GameState(game_id, i, co.side, "cpu_preflight", 1 + (i % 3), 4 + 3 * (i % 4), 20 + (i * 7) % 70, opponent_class=opponent_class)
        card = co.recommend(st)
        lat.append(card.latency_ms)
        fb += int(card.fallback["occurred"])
        rec = co.snaps[st.snap_id].recommended
        if (card.formation, card.play, card.action_key) != (rec.play.formation, rec.play.play, rec.key):
            mism += 1
        if co.snaps[st.snap_id].executed is not None:
            mism += 1  # a recommendation must never silently become an execution
        co.record_execution(st, rec, EXEC_CONFIRMED)
        f, p = defs[i % len(defs)]
        sig = (K.defense_play(f, p) if co.side == OFF else K.offense_play("Gun Trips X Nasty", "Mesh Spot")).signature
        co.record_opponent(st, OpponentAction(DEF if co.side == OFF else OFF, f if co.side == OFF else "Gun Trips X Nasty",
                                              p if co.side == OFF else "Mesh Spot", sig, "typed_exact", 1.0))
        co.close_snap(st, Outcome(float((i * 3) % 11 - 2), False, False, False, False, False, "synthetic"))
        chron.append(co.opp.beliefs(GameState(game_id, i + 1, co.side, "cpu_preflight", 1, 10, 30, opponent_class=opponent_class)).n_effective)
    return {"calls": n, "latency_ms": {"p50": statistics.median(lat), "p95": p95(lat), "max": max(lat)},
            "timeouts": sum(1 for x in lat if x > LATENCY_BUDGET_MS), "fallbacks": fb, "card_action_mismatches": mism,
            "belief_evidence_monotone": all(b2 >= b1 - 1e-9 for b1, b2 in zip(chron, chron[1:]))}


def _parser_gate() -> dict[str, Any]:
    """The local game-state parser against the formats the owner actually types (side, ordinals, yardline phrases)."""
    from cfb_coach.madden.unified.inputs import InputError, parse_snap_line
    cases = [("o 1&10 my 22", OFF, 22), ("d 3&7 opp 40", DEF, 40), ("o 3rd and 7 on my 45", OFF, 45), ("d: 2&8 my 15", DEF, 85),
             ("o 4th & 1 opp 38", OFF, 62), ("their ball 1&10 my 30", DEF, 70), ("o 2&5 my 35 showing cover 3 2 minute", OFF, 35)]
    bad = []
    for text, side, yl in cases:
        try:
            r = parse_snap_line(text, None)
            if (r.side, r.yardline) != (side, yl):
                bad.append((text, r.side, r.yardline))
        except InputError as e:
            bad.append((text, str(e)))
    return _gate("local_game_state_parser", not bad, {"cases": len(cases), "failures": bad})


def preflight(*, db: str | None = None, confirmed_books: str | None = None, model: str | None = None, history_report: str | None = None,
              adjustment_confirmations: str | None = None, calls: int = 100, extra_protected: Sequence[str] = (),
              workdir: str | None = None, html: bool = False) -> dict[str, Any]:
    """GO needs: owner-attested BOTH books, an explicit model artifact consistent with the history evaluation, an audited
    isolated database, AND a clean run of the real continuous live loop (>= ``calls`` snaps per side) on those exact inputs."""
    from cfb_coach.madden.unified import livepath as LP

    gates: list[dict[str, Any]] = []
    prot = guard.protected_paths([*extra_protected, *[x for x in (db, confirmed_books, model, history_report) if x]], REPO)
    before = guard.snapshot(prot)
    snap_inv = json.loads(SNAPSHOT.read_text())["sides"]
    confirmed = load_confirmed_books(confirmed_books)
    work = Path(workdir or tempfile.mkdtemp(prefix="unified_preflight_"))
    work.mkdir(parents=True, exist_ok=True)
    if (work / "live.db").exists():  # never reuse a previous benchmark store (it would be a duplicate game)
        work = Path(tempfile.mkdtemp(prefix="run_", dir=str(work)))
    books = {}
    for side in (OFF, DEF):
        have = side in confirmed
        inv = confirmed[side]["inventory"] if have else snap_inv[side]["formations"]
        fp = inventory_fp(side, inv)
        books[side] = {"inventory": inv, "fingerprint": fp}
        gates.append(_gate(f"{side}_installed_book_confirmed_by_owner", True if have else None,
                           {"source": "owner_attestation" if have else "repo snapshot (NOT this machine's installed book)", "fingerprint": fp, "formations": len(inv)},
                           how_to_run=f"unified confirm-install --proposal <proposal_{side}.json> --attested-by <you> --confirmed-books <file>"))
        gates.append(_gate(f"{side}_book_fingerprint_matches", (not have) or confirmed[side]["fingerprint"] == fp, {"fingerprint": fp}))
        gates.append(_gate(f"{side}_formation_limit", len(inv) <= MAX_FORMATIONS, {"formations": len(inv), "limit": MAX_FORMATIONS}))
    # ---- model artifact: must be a real file, and consistent with the historical evaluation if one was run
    if model and Path(model).exists():
        wm = WorldModel.from_dict(json.loads(Path(model).read_text()))
        sha = guard.sha256_file(Path(model))
        gates.append(_gate("model_artifact_file", True, {"path": model, "sha256": sha, "records": wm.n_records, "provenance": wm.provenance,
                                                         "mode": "learned weights" if wm.n_records else "UNCALIBRATED COLD-START PRIOR (no trained weights)"}))
    else:
        wm = None
        gates.append(_gate("model_artifact_file", None, {"note": "no --model file; the live benchmark ran the in-memory cold-start prior"},
                           how_to_run="unified history --db <test db> --out <dir>   then pass <dir>/cold_start_model.json or candidate_model.json as --model"))
    if history_report and Path(history_report).exists():
        hr = json.loads(Path(history_report).read_text())
        ok = True
        why = {"decision": hr.get("decision"), "usable_rows": hr.get("usable_rows"), "usable_games": hr.get("usable_games")}
        if model:
            rdir = Path(history_report).parent
            cold = rdir / "cold_start_model.json"
            cand = Path(hr["candidate_artifact"]) if hr.get("candidate_artifact") else rdir / "candidate_model.json"
            sha = guard.sha256_file(Path(model))
            cold_sha = guard.sha256_file(cold) if cold.exists() else None
            cand_sha = guard.sha256_file(cand) if cand.exists() else None
            if sha == cand_sha:
                if hr.get("decision") != "CANDIDATE_SUPPORTED":
                    ok, why["error"] = False, "model is the learned candidate but the evaluation did NOT support it"
                else:
                    why["model"] = "learned candidate (evaluation supported it)"
            elif sha == cold_sha:
                why["model"] = "cold-start prior (explicitly labelled)"
            else:
                ok, why["error"] = False, "model file is not an artifact produced by that history run"
        else:
            ok, why["error"] = None, "no --model to compare with the history evaluation"
        gates.append(_gate("history_evaluation_consistent_with_model", ok, why))
    else:
        gates.append(_gate("history_evaluation_consistent_with_model", None, {"note": "historical import/evaluation not run on the owner's database"},
                           how_to_run="unified history --db <isolated test-franchise db> --out <dir>"))
    if db:
        a = audit_db(db)
        gates.append(_gate("owner_database_audit", bool(a.get("ok")) and not a.get("inside_primary_data_dir", False), a,
                           how_to_run="use an ISOLATED test-franchise database, never the primary one"))
    else:
        gates.append(_gate("owner_database_audit", None, {"note": "no owner database was supplied"}, how_to_run="unified preflight --db <isolated test-franchise coach.db> ..."))
    gates.append(_parser_gate())
    # ---- the real continuous live loop, on these exact books + model file, persisted and cold-reloaded
    conf = json.loads(Path(adjustment_confirmations).read_text()) if adjustment_confirmations else {}
    run = LP.run_live_benchmark(books, model_path=model if wm is not None else None, store_path=work / "live.db", n_per_side=calls, confirmations=conf)
    aud = LP.audit_run(run, books, confirmations=conf)
    P = aud["problems"]
    lat = aud["latency_start_ms"]
    scope = "this machine, real cmd_game loop, persisted store" + ("" if confirmed_books else "; repo snapshot books")
    gates += [
        _gate("live_loop_ran_clean", aud["rc"] == 0 and aud["snaps"] == 2 * calls, {"snaps": aud["snaps"], "rc": aud["rc"], "wall_s": round(run["wall_s"], 2)}),
        _gate("live_latency_p95_le_150ms", lat["p95"] <= LATENCY_BUDGET_MS, {**lat, "scope": scope, "finish_ms": aud["latency_finish_ms"]}, critical=False,
              how_to_run="unified preflight ... on the owner's laptop"),
        _gate("live_no_timeouts", aud["timeouts_over_150ms"] == 0, {"timeouts": aud["timeouts_over_150ms"]}, critical=False),
        _gate("live_no_fallback_or_heuristic_substitution", aud["fallbacks"] == 0 and not P["fallback"] and aud["decision_model_values"] == ["unified_v1"],
              {"fallbacks": aud["fallbacks"], "decision_models": aud["decision_model_values"]}),
        _gate("live_no_uninstalled_calls", not P["uninstalled"], {"problems": P["uninstalled"]}),
        _gate("live_no_duplicate_snap_ids", not P["duplicate_ids"] and aud["unique_ids"] == aud["snaps"], {"unique": aud["unique_ids"], "snaps": aud["snaps"]}),
        _gate("live_recommended_vs_executed_logged_exactly", not P["exec_mismatch"], {"mismatches": P["exec_mismatch"]}),
        _gate("live_model_attribution_correct", not P["attribution"], {"artifact": aud["artifact_expected"], "bad": P["attribution"]}),
        _gate("live_only_legal_adjustments", not P["illegal_adjustment"], {"bad": P["illegal_adjustment"]}),
        _gate("live_side_follows_possession", not P["side"] and not P["seq"], {"bad": P["side"] + P["seq"]}),
        _gate("live_previous_play_learning", aud["learning"]["matches"] and aud["previous_play_learning_visible"],
              {**aud["learning"], "belief_sources_final": aud["belief_sources_final"]}),
    ]
    html_gates: list[dict[str, Any]] = []
    if html:
        from cfb_coach.madden.unified import htmlcheck as HC
        hrun = HC.run_http_workflow(books, model_path=model if wm is not None else None, store_path=work / "html.db", n_per_side=calls, confirmations=conf)
        ha = HC.audit_http_run(hrun, books, confirmations=conf)
        hp = ha["problems"]
        html_gates += [
            _gate("html_page_serves_established_interface", ha["page_ok"], {"page_ok": ha["page_ok"]}),
            _gate("html_workflow_clean", ha["ok"] and not ha["errors"], {"snaps": ha["snaps"], "errors": ha["errors"], "problems": hp,
                                                                         "trailing_unplayed_snaps": ha["extra_unplayed_snaps"]}),
            _gate("html_no_legacy_caller_or_primary_db_access", not ha["legacy_hits"], {"hits": ha["legacy_hits"]}),
            _gate("html_recommended_vs_executed_and_learning", not hp["exec_mismatch"] and ha["learning"]["matches"],
                  {"exec_mismatch": hp["exec_mismatch"], **ha["learning"]}),
            _gate("html_request_latency_p95_le_150ms", ha["request_ms"]["p95"] <= LATENCY_BUDGET_MS,
                  {**ha["request_ms"], "scope": "this machine; client send -> full JSON response over real HTTP"}, critical=False,
                  how_to_run="unified html-check ... on the owner's laptop"),
        ]
    else:
        html_gates.append(_gate("html_interface_validated", None, {"note": "browser integration was not exercised in this preflight"},
                                how_to_run="unified html-check --confirmed-books <file> --model <file>   (or preflight --html)"))
    after = guard.snapshot(prot)
    d = guard.diff(before, after)
    gates.append(_gate("protected_assets_unchanged", not (d["changed"] or d["removed"] or d["added"]), {"protected_files_checksummed": len(prot), **d}))
    crit_fail = [g for g in gates if g["critical"] and g["status"] == "FAIL"]
    soft_fail = [g for g in gates if not g["critical"] and g["status"] == "FAIL"]
    not_run = [g for g in gates if g["status"] == "NOT_RUN"]
    if crit_fail:
        verdict, why = "NO-GO", [g["gate"] for g in crit_fail]
    elif soft_fail or not_run:
        verdict, why = "SHADOW ONLY", list(dict.fromkeys(g["gate"] for g in soft_fail + not_run))
    else:
        verdict, why = "GO", []
    h_fail = [g for g in html_gates if g["critical"] and g["status"] == "FAIL"]
    h_soft = [g for g in html_gates if g["status"] in ("FAIL", "NOT_RUN") and g not in h_fail]
    if h_fail or verdict == "NO-GO":
        html_verdict, html_why = "NO-GO", [g["gate"] for g in h_fail] or why
    elif verdict != "GO" or h_soft:
        html_verdict, html_why = "SHADOW ONLY", list(dict.fromkeys(([] if verdict == "GO" else why) + [g["gate"] for g in h_soft]))
    else:
        html_verdict, html_why = "GO", []
    return {"version": "unified_preflight.v3", "verdict": verdict, "html_verdict": html_verdict, "html_blocking": html_why, "html_gates": html_gates,
            "verdict_note": "`verdict` = terminal loop; `html_verdict` = the familiar HTML window (needs the terminal gates AND the html gates)", "blocking": why, "artifact": aud["artifact_expected"], "gates": gates,
            "live": {k: aud[k] for k in ("snaps", "latency_start_ms", "latency_finish_ms", "timeouts_over_150ms", "fallbacks", "learning")},
            "note": "GO would mean ready for an experimental CPU test, not proven strategic superiority. Latency/legality are measured through the real "
                    "continuous loop, never a bare in-process call."}


def render_report(rep: Mapping[str, Any]) -> str:
    L = [f"# CPU preflight: terminal {rep['verdict']} · HTML {rep.get('html_verdict', 'n/a')}", "", f"artifact: `{rep['artifact']}`", f"blocking: {rep['blocking'] or 'none'}", "",
         "| gate | status | critical |", "|---|---|---|"]
    for g in [*rep["gates"], *rep.get("html_gates", [])]:
        L.append(f"| {g['gate']} | {g['status']} | {g['critical']} |")
    L += ["", "## Evidence", "```json", json.dumps(rep["gates"], indent=1, default=str), "```", "", rep["note"]]
    return "\n".join(L)
