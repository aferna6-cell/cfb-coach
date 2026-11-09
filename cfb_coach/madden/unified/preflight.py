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


def preflight(*, db: str | None = None, confirmed_books: str | None = None, model: str | None = None, calls: int = 100,
              extra_protected: Sequence[str] = (), workdir: str | None = None) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    prot = guard.protected_paths(extra_protected, REPO)
    before = guard.snapshot(prot)
    snapshot_inv = json.loads(SNAPSHOT.read_text())["sides"]
    confirmed = load_confirmed_books(confirmed_books)
    artifact = "cold_start_prior_v1"
    wm: WorldModel | None = None
    if model:
        wm = WorldModel.from_dict(json.loads(Path(model).read_text()))
        artifact = f"{Path(model).name}:{guard.sha256_file(Path(model))[:10]}"
        gates.append(_gate("model_artifact_loads", True, {"artifact": artifact, "records": wm.n_records, "calibrated": wm.objective.calibrated}))
    else:
        gates.append(_gate("model_artifact_loads", True, {"artifact": artifact, "note": "cold-start prior, no trained weights, NOT calibrated"}))
    results: dict[str, Any] = {}
    work = Path(workdir or tempfile.mkdtemp(prefix="unified_preflight_"))
    for side in (OFF, DEF):
        book_src = "owner_confirmed" if side in confirmed else "repo_snapshot_UNCONFIRMED_for_this_machine"
        inv = confirmed[side]["inventory"] if side in confirmed else snapshot_inv[side]["formations"]
        fp = inventory_fp(side, inv)
        ok_fp = (side not in confirmed) or confirmed[side]["fingerprint"] == fp
        gates.append(_gate(f"{side}_installed_book_confirmed_by_owner", True if side in confirmed else None,
                           {"source": book_src, "fingerprint": fp, "formations": len(inv)},
                           how_to_run=f"python -m cfb_coach.madden.unified confirm-install --proposal <proposal_{side}.json> --attested-by <you> --confirmed-books <file>"))
        gates.append(_gate(f"{side}_book_fingerprint_matches", ok_fp, {"fingerprint": fp}))
        gates.append(_gate(f"{side}_formation_limit", len(inv) <= MAX_FORMATIONS, {"formations": len(inv), "limit": MAX_FORMATIONS}))
        try:
            store = UnifiedStore(work / f"{side}.db")
            co = UnifiedCoordinator(side, inv, confirmed_fingerprint=fp, model=wm, store=store, artifact=artifact)
            leg = legality_report(co)
            gates.append(_gate(f"{side}_legal_candidates", not leg["candidates_outside_installed_book"] and not leg["plays_missing_no_adjustment"], leg))
            cyc = run_cycle(co, game_id=f"pf-{side}", n=calls)
            results[side] = cyc
            gates.append(_gate(f"{side}_latency_p95_le_150ms_loaded", cyc["latency_ms"]["p95"] <= LATENCY_BUDGET_MS,
                               {**cyc["latency_ms"], "calls": calls, "scope": "this machine, repo catalogs, in-process (NOT the owner's loaded UI/DB)"},
                               critical=False, how_to_run="python -m cfb_coach.madden.unified preflight --confirmed-books <file> --calls 100  (run on the owner's machine)"))
            gates.append(_gate(f"{side}_no_timeouts_or_fallbacks", cyc["timeouts"] == 0 and cyc["fallbacks"] == 0, {"timeouts": cyc["timeouts"], "fallbacks": cyc["fallbacks"]}, critical=False))
            gates.append(_gate(f"{side}_card_matches_selected_action", cyc["card_action_mismatches"] == 0, {"mismatches": cyc["card_action_mismatches"]}))
            gates.append(_gate(f"{side}_chronological_memory_updates", cyc["belief_evidence_monotone"], {"monotone_evidence": cyc["belief_evidence_monotone"]}))
            counts = store.counts()
            gates.append(_gate(f"{side}_execution_logging_distinct", counts["decisions"] == calls and counts["executions"] == calls and counts["opponent_obs"] == calls, counts))
            replay = store.load_observations()
            gates.append(_gate(f"{side}_store_roundtrip", len(replay) == calls and all(o.executed_status == EXEC_CONFIRMED for o in replay), {"observations": len(replay)}))
            store.close()
        except InstalledBookError as e:
            gates.append(_gate(f"{side}_coordinator_builds", False, str(e)))
    after = guard.snapshot(prot)
    d = guard.diff(before, after)
    gates.append(_gate("protected_assets_unchanged", not (d["changed"] or d["removed"] or d["added"]), {"protected_files_checksummed": len(prot), **d}))
    if db:
        a = audit_db(db)
        gates.append(_gate("owner_database_audit", bool(a.get("ok")) and not a.get("inside_primary_data_dir", False), a,
                           how_to_run="use an ISOLATED test-franchise database, never the primary one"))
    else:
        gates.append(_gate("owner_database_audit", None, {"note": "no owner database was supplied"},
                           how_to_run="python -m cfb_coach.madden.unified preflight --db <isolated test-franchise coach.db> ..."))
    crit_fail = [g for g in gates if g["critical"] and g["status"] == "FAIL"]
    soft_fail = [g for g in gates if not g["critical"] and g["status"] == "FAIL"]
    not_run = [g for g in gates if g["status"] == "NOT_RUN"]
    unconfirmed = [g for g in gates if g["gate"].endswith("installed_book_confirmed_by_owner") and g["status"] != "PASS"]
    if crit_fail:
        verdict, why = "NO-GO", [g["gate"] for g in crit_fail]
    elif unconfirmed or soft_fail or not_run:
        verdict, why = "SHADOW ONLY", list(dict.fromkeys(g["gate"] for g in unconfirmed + soft_fail + not_run))
    else:
        verdict, why = "GO", []
    return {"version": "unified_preflight.v1", "verdict": verdict, "blocking": why, "artifact": artifact, "gates": gates,
            "cycles": results, "note": "GO would mean ready for an experimental CPU test, not proven strategic superiority."}


def render_report(rep: Mapping[str, Any]) -> str:
    L = [f"# CPU preflight: {rep['verdict']}", "", f"artifact: `{rep['artifact']}`", f"blocking: {rep['blocking'] or 'none'}", "",
         "| gate | status | critical |", "|---|---|---|"]
    for g in rep["gates"]:
        L.append(f"| {g['gate']} | {g['status']} | {g['critical']} |")
    L += ["", "## Evidence", "```json", json.dumps(rep["gates"], indent=1, default=str), "```", "", rep["note"]]
    return "\n".join(L)
