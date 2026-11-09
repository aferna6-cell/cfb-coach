"""Unified pregame playbook design: up to 15 formations per side, chosen independently.

The same world model and opponent beliefs that drive live calls score every eligible catalog play in
a spread of game situations; formations are chosen by greedy maximum-coverage (monotone submodular, so
greedy is within (1 - 1/e) of the best portfolio) with explicit situational-coverage requirements.
There is NO fixed rotation: the portfolio changes only with the scores. The result is a PROPOSAL.
Nothing is installed or claimed installed; ``confirm_install`` records the owner's attestation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden import catalog_full
from cfb_coach.madden.model.playbook_architect import family_ingame
from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.coordinator import MAX_FORMATIONS, inventory_fp, now
from cfb_coach.madden.unified.opponent import OpponentModel
from cfb_coach.madden.unified.solver import Solver, SolverConfig
from cfb_coach.madden.unified.types import DEF, OFF, FootballAction, GameState, other_side
from cfb_coach.madden.unified.world_model import WorldModel

PREGAME_VERSION = "unified_pregame.v1"
MAX_PLAYS = int(catalog_full.EDITOR_LIMITS["max_plays"])

SITUATIONS: dict[str, tuple[tuple[int, int, int], float]] = {  # name -> ((down, dist, yardline), weight)
    "1st&10 open": ((1, 10, 30), 0.17), "2nd&short": ((2, 3, 45), 0.06), "2nd&medium": ((2, 6, 45), 0.10),
    "2nd&long": ((2, 10, 40), 0.07), "3rd&short": ((3, 2, 50), 0.07), "3rd&medium": ((3, 6, 50), 0.10),
    "3rd&long": ((3, 10, 45), 0.09), "4th&short": ((4, 1, 55), 0.03), "red zone": ((1, 10, 85), 0.09),
    "goal line": ((1, 2, 98), 0.05), "backed up": ((1, 10, 8), 0.04), "two-minute-ish deep": ((2, 14, 40), 0.03),
}
REQUIREMENTS: dict[str, dict[str, Any]] = {
    OFF: {"run": (6, lambda p: p.play_type == "run"), "quick_pass": (8, lambda p: p.play_type in ("pass", "play_action") and bool(p.features.get("quick"))),
          "deep_shot": (3, lambda p: bool(p.features.get("deep"))), "screen": (2, lambda p: p.play_type == "screen" or p.concept == "screen"),
          "play_action": (2, lambda p: p.play_type == "play_action")},
    DEF: {"man": (4, lambda p: p.features.get("coverage_type") == "man"), "zone": (5, lambda p: p.features.get("coverage_type") == "zone"),
          "match": (3, lambda p: p.features.get("coverage_type") == "match"), "pressure": (3, lambda p: bool(p.features.get("blitz"))),
          "deep_shell": (4, lambda p: p.features.get("deep_shell") in (2, 3, 4, 6))},
}
SHORT_YARDAGE = {OFF: lambda c: c["n_run"] >= 3, DEF: lambda c: c["heavy"]}


@dataclass
class Rec:
    formation: str
    book: str
    plays: list[str]
    family: str
    set_name: str | None
    values: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    footballplays: list[Any] = field(default_factory=list)


def candidate_records(side: str) -> list[Rec]:
    out: list[Rec] = []
    for book, rec in catalog_full.architect_books(side).items():
        for fm, r in (rec.get("formations") or {}).items():
            plays = list(dict.fromkeys(r.get("plays") or []))
            if plays:
                out.append(Rec(fm, book, plays, r.get("family", ""), r.get("set")))
    return out


def score_records(side: str, recs: Sequence[Rec], model: WorldModel, opp: OpponentModel, opponent_id: str,
                  opponent_class: str = "cpu") -> None:
    """Fill ``Rec.values[situation]`` = mean of the best 3 play scores in that situation."""
    cfg = SolverConfig(report_top=100000, keep_families=100000, max_signatures=10)
    solver = Solver(model, config=cfg)
    uniq: dict[tuple[str, str], Any] = {}
    for r in recs:
        for p in r.plays:
            if (r.formation, p) not in uniq:
                uniq[(r.formation, p)] = (K.offense_play if side == OFF else K.defense_play)(r.formation, p)
        r.footballplays = [uniq[(r.formation, p)] for p in r.plays]
    acts = [FootballAction(fp) for fp in uniq.values()]
    for sname, ((d, dist, yl), _w) in SITUATIONS.items():
        st = GameState("pregame", 1, side, opponent_id, d, dist, yl, opponent_class=opponent_class)
        dec = solver.decide(st, side, acts, opp.beliefs(st))
        score = {s.action.play.play_id: s.score for s in dec.ranked}
        for r in recs:
            vals = sorted((score[fp.play_id] for fp in r.footballplays if fp.play_id in score), reverse=True)[:3]
            r.values[sname] = sum(vals) / len(vals) if vals else -9.0


def _counts(side: str, r: Rec) -> dict[str, int]:
    c = {k: sum(1 for p in r.footballplays if f(p)) for k, (_, f) in REQUIREMENTS[side].items()}
    c["n_run"] = sum(1 for p in r.footballplays if p.play_type == "run")
    c["heavy"] = int(any(t in r.formation.lower() for t in ("goal line", "46", "4-3", "5-2", "4-4")))
    return c


DEPTH_WEIGHT, DIVERSITY_WEIGHT = 0.35, 0.002


def _coverage(sel: Sequence[Rec]) -> float:
    """Best option per situation + a depth term (2nd-best option) + concept diversity. Monotone submodular-ish."""
    tot = 0.0
    for s, (_, w) in SITUATIONS.items():
        vals = sorted((r.values[s] for r in sel), reverse=True)
        tot += w * ((vals[0] if vals else -9.0) + DEPTH_WEIGHT * (vals[1] if len(vals) > 1 else (vals[0] - 0.05 if vals else -9.0)))
    sigs = {fp.signature for r in sel for fp in r.footballplays}
    return tot + DIVERSITY_WEIGHT * len(sigs)


def _deficits(side: str, sel: Sequence[Rec]) -> dict[str, int]:
    out = {}
    for k, (need, _) in REQUIREMENTS[side].items():
        have = sum(r.counts[k] for r in sel)
        if have < need:
            out[k] = need - have
    if not any(SHORT_YARDAGE[side](r.counts) for r in sel):
        out["short_yardage"] = 1
    return out


def select_portfolio(side: str, recs: Sequence[Rec], *, max_formations: int = MAX_FORMATIONS, max_plays: int = MAX_PLAYS) -> tuple[list[Rec], list[dict[str, Any]]]:
    for r in recs:
        r.counts = _counts(side, r)
    sel: list[Rec] = []
    log: list[dict[str, Any]] = []
    families: set[str] = set()
    taken = set()
    while len(sel) < max_formations:
        base, defi = _coverage(sel), _deficits(side, sel)
        best, best_gain, best_note = None, -1e9, ""
        for r in recs:
            if r.formation in taken or sum(len(x.plays) for x in sel) + len(r.plays) > max_plays:
                continue
            gain = _coverage(sel + [r]) - base
            fix = sum(min(r.counts.get(k, 0), n) for k, n in defi.items() if k != "short_yardage") * 0.004
            if "short_yardage" in defi and SHORT_YARDAGE[side](r.counts):
                fix += 0.05
            tot = gain + fix
            if tot > best_gain:
                best, best_gain, best_note = r, tot, f"coverage +{gain:.4f}, requirement fill +{fix:.4f}"
        if best is None or (best_gain <= 1e-9 and not _deficits(side, sel)):
            break
        sel.append(best)
        taken.add(best.formation)
        log.append({"added": best.formation, "source_book": best.book, "marginal": round(best_gain, 5), "note": best_note})
    return sel, log


def build_proposal(side: str, *, opponent_id: str = "cpu", opponent_class: str = "cpu", model: WorldModel | None = None,
                   opp: OpponentModel | None = None, max_formations: int = MAX_FORMATIONS) -> dict[str, Any]:
    model, opp = model or WorldModel(), opp or OpponentModel(other_side(side))
    recs = candidate_records(side)
    score_records(side, recs, model, opp, opponent_id, opponent_class)
    sel, log = select_portfolio(side, recs, max_formations=max_formations)
    formations = []
    for r in sel:
        best_s = sorted(SITUATIONS, key=lambda s: -r.values[s])[:3]
        unknown = ["editor availability not verified in Madden 27 (owner must confirm in the playbook editor)",
                   "personnel inferred from the formation name" if side == OFF else "defender alignments UNKNOWN unless first-party verified"]
        known_assign = sum(1 for fp in r.footballplays if fp.assignments)
        if known_assign < len(r.footballplays):
            unknown.append(f"route/defender assignment knowledge for {len(r.footballplays) - known_assign}/{len(r.footballplays)} plays is UNKNOWN")
        formations.append({
            "formation": r.formation, "family": r.family, "set": r.set_name, "source_playbook": r.book, "n_plays": len(r.plays),
            "plays": r.plays, "why": f"best-available option in {', '.join(best_s)}; fills {', '.join(k for k, v in r.counts.items() if v and k in REQUIREMENTS[side]) or 'depth'}",
            "situation_values": {k: round(v, 4) for k, v in r.values.items()}, "verification_missing": unknown,
            "install": f"Playbook editor: family '{family_ingame(r.family) if r.family else r.formation}' -> set "
                       f"'{r.set_name or r.formation}' -> source playbook '{r.book}' -> keep ALL {len(r.plays)} plays listed."})
    inv = {f["formation"]: f["plays"] for f in formations}
    deficits = _deficits(side, sel)
    return {"version": PREGAME_VERSION, "side": side, "created": now(), "opponent_id": opponent_id, "status": "PROPOSAL_NOT_INSTALLED",
            "formations": formations, "n_formations": len(formations), "n_plays": sum(len(v) for v in inv.values()),
            "proposed_fingerprint": inventory_fp(side, inv), "confirmed_install": False,
            "coverage_objective": round(_coverage(sel), 5), "unmet_requirements": deficits, "selection_log": log,
            "limits": {"max_formations": max_formations, "max_plays": MAX_PLAYS},
            "model_note": "scored with the cold-start prior + learned tables if provided; no fixed rotation, no schedule",
            "inventory": inv, "sources": {f["formation"]: f["source_playbook"] for f in formations}}


def confirm_install(proposal: Mapping[str, Any], path: str | Path, *, attested_by: str) -> dict[str, Any]:
    """Record the OWNER's attestation that this exact book is physically installed. Never called automatically."""
    rec = {"side": proposal["side"], "fingerprint": proposal["proposed_fingerprint"], "inventory": proposal["inventory"],
           "attested_by": attested_by, "attested_ts": now(), "method": "owner_attestation"}
    if not attested_by.strip():
        raise ValueError("an attesting person is required")
    p = Path(path)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[proposal["side"]] = rec
    p.write_text(json.dumps(cur, indent=2))
    return rec
