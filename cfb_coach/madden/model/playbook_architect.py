"""Pregame playbook architect: up to 15 offensive and 15 defensive formations.

Each side is chosen as a portfolio. The objective rewards the best and the
second-best formation in every situation (coverage plus a counter), concept or
coverage variety, personnel variety, verified outcomes and the matchup, and it
charges every formation an install cost. Fifteen independent top scores are
never the answer by construction: a formation is worth what it adds to the
others.

There is no schedule. Two searches run per side:
  - ideal: the best portfolio from an empty book.
  - transition: local search (add, remove, swap) from the installed book,
    accepting a move only when its gain clears an evidence-scaled threshold.
The transition result is the proposal. With an already-optimal book no move
clears the threshold, so nothing changes.

This module is read-only. It returns proposals; ``playbook_install`` stages
and confirms them.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from cfb_coach.madden import catalog, catalog_full
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.football_situation import SITUATION_NAMES, explain_play
from cfb_coach.madden.model.offense_portfolio import (
    PORTFOLIO_PROBES,
    _roster_bonus,
    _tendency_bonus,
    _zone_for,
    load_verified_roster,
    opponent_defense_profile,
)

MAX_FORMATIONS = catalog_full.REQUESTED_MAX_FORMATIONS
MAX_PLAYS = int(catalog_full.EDITOR_LIMITS["max_plays"])
MAX_FAMILIES = int(catalog_full.EDITOR_LIMITS["max_formations"])
BOOK_NAMES = {"offense": "ML Designed Offense (custom)", "defense": "ML Designed Defense (custom)"}

OFFENSE_WEIGHTS = {
    "normal": 0.24, "short_yardage": 0.11, "third_and_long": 0.12, "red_zone": 0.11, "goal_line": 0.08,
    "backed_up": 0.05, "two_minute": 0.11, "clock_management": 0.07, "explosive": 0.11,
}
DEFENSE_WEIGHTS = {
    "early": 0.30, "short": 0.10, "medium": 0.12, "long": 0.14, "red_zone": 0.12, "goal_line": 0.08,
    "two_minute": 0.14,
}
# The second-best formation in a situation is the counter when the first is
# taken away; the third keeps a human opponent from keying on two looks.
DEPTH_SHARES = {"human": (1.0, 0.3, 0.12), "cpu": (1.0, 0.3, 0.05)}
# Distinct good calls per situation, saturating. A human learns repeated
# calls over 60+ snaps; the CPU mostly does not.
PLAY_DEPTH_WEIGHT = {"human": 0.05, "cpu": 0.02}
PLAY_DEPTH_SCALE = 10.0
GOOD_PLAY_MARGIN = 0.08
GOOD_PACKAGE_FIT = 0.6
FORMATION_COST = 0.004
# Concept variety saturates: the first dozen concept ids matter, the fortieth barely does.
CONCEPT_TOTAL = 0.06
CONCEPT_SCALE = 12.0
PERSONNEL_WEIGHT = 0.006
PACKAGE_WEIGHT = 0.012
COVERAGE_WEIGHT = 0.008
# The success model already learns from the same verified history, so the
# direct evidence term is kept small to avoid counting it twice.
EVIDENCE_WEIGHT = 0.25
FAILED_PLAY_COST = 0.003
BALANCE_PENALTY = 0.06
MISSING_SIDE_PENALTY = 0.1
# Minimum objective gain to change an installed book, by evidence quality.
# Weaker evidence must promise more before Aidan rebuilds a book.
SWITCH_GAIN = {"empirical": 0.004, "mixed": 0.007, "prior_driven": 0.012}
THIN_EVIDENCE_EXTRA = 0.003
POOL_PER_SITUATION = 14
POOL_TOP = 60
MAX_MOVES = 40

# Package fit by defensive situation (football prior, labeled as such).
PACKAGE_FIT: dict[str, dict[str, float]] = {
    "early": {"base": 0.75, "base_heavy": 0.6, "nickel": 0.8, "dime": 0.4, "quarter": 0.2, "goal_line": 0.2, "prevent": 0.0},
    "short": {"base": 0.85, "base_heavy": 0.85, "nickel": 0.5, "dime": 0.15, "quarter": 0.05, "goal_line": 0.7, "prevent": 0.0},
    "medium": {"base": 0.55, "base_heavy": 0.35, "nickel": 0.85, "dime": 0.7, "quarter": 0.4, "goal_line": 0.1, "prevent": 0.0},
    "long": {"base": 0.3, "base_heavy": 0.15, "nickel": 0.7, "dime": 0.9, "quarter": 0.8, "goal_line": 0.0, "prevent": 0.3},
    "red_zone": {"base": 0.7, "base_heavy": 0.6, "nickel": 0.8, "dime": 0.5, "quarter": 0.2, "goal_line": 0.5, "prevent": 0.0},
    "goal_line": {"base": 0.55, "base_heavy": 0.85, "nickel": 0.2, "dime": 0.05, "quarter": 0.0, "goal_line": 1.0, "prevent": 0.0},
    "two_minute": {"base": 0.3, "base_heavy": 0.1, "nickel": 0.7, "dime": 0.85, "quarter": 0.8, "goal_line": 0.0, "prevent": 0.5},
}
RUN_TILT = {"base_heavy": 1.0, "goal_line": 1.0, "base": 0.7, "nickel": 0.0, "dime": -0.8, "quarter": -1.0, "prevent": -1.0}
DEF_PROBES = {
    "early": SimpleNamespace(down=1, distance=10, goal_line=False, red_zone=False, two_minute=False),
    "short": SimpleNamespace(down=3, distance=2, goal_line=False, red_zone=False, two_minute=False),
    "medium": SimpleNamespace(down=3, distance=5, goal_line=False, red_zone=False, two_minute=False),
    "long": SimpleNamespace(down=3, distance=9, goal_line=False, red_zone=False, two_minute=False),
    "red_zone": SimpleNamespace(down=1, distance=10, goal_line=False, red_zone=True, two_minute=False),
    "goal_line": SimpleNamespace(down=1, distance=2, goal_line=True, red_zone=True, two_minute=False),
    "two_minute": SimpleNamespace(down=2, distance=8, goal_line=False, red_zone=False, two_minute=True),
}
PACKAGE_GROUP = {"base": "base", "base_heavy": "base", "nickel": "nickel", "dime": "dime", "quarter": "dime",
                 "goal_line": "goal_line"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def fingerprint(side: str, formations: Mapping[str, Sequence[str]], sources: Mapping[str, str]) -> str:
    """Short code Aidan types to confirm the exact inventory he installed."""
    body = [[f, sources.get(f), list(formations[f])] for f in sorted(formations)]
    h = digest([side, body]).upper()
    return f"{'OFF' if side == 'offense' else 'DEF'}-{h[:4]}-{h[4:8]}"


def family_ingame(family: str) -> str:
    mapping = catalog_full.load_full().get("family_ingame") or {}
    return str(mapping.get(family) or ("Shotgun" if family == "Gun" else family))


# --------------------------------------------------------------------------- candidates


def _offense_values(art: Any, opponent_id: str, cache: dict) -> Callable[[str, list[str]], dict[str, float]]:
    def play_score(form: str, play: str, sit: str) -> float:
        key = (form, play, sit)
        if key not in cache:
            probe = PORTFOLIO_PROBES[sit]
            pred = experimental_model.predict_success(
                art, formation=form, play=play, down=probe.down, distance=probe.distance,
                yardline=probe.yardline, opponent_id=opponent_id, coverage_hint=None,
                coverage_source="none", heuristic_bonus=0.0,
            )
            fit = explain_play(play, probe)
            cache[key] = float(pred["probability"]) + float(fit["selection_delta"]) + float(fit["coordinator_delta"])
        return cache[key]

    def values(form: str, plays: list[str]) -> tuple[dict[str, float], dict[str, list[float]]]:
        out: dict[str, float] = {}
        scores: dict[str, list[float]] = {}
        for sit in SITUATION_NAMES:
            zone = _zone_for(sit)
            best: dict[str, float] = {}
            fitting: list[float] = []
            for play in plays:
                if sit not in ("goal_line", "red_zone") and not catalog.zone_fit(play, zone):
                    continue
                concept = experimental_model._play_concept(play)
                score = play_score(form, play, sit)
                fitting.append(score)
                if score > best.get(concept, -9.0):
                    best[concept] = score
            top = sorted(best.values(), reverse=True)[:4]
            out[sit] = round(sum(top) / len(top), 6) if top else 0.0
            scores[sit] = sorted(fitting, reverse=True)
        deep = [play_score(form, p, "normal") for p in plays if catalog.is_deep(p)]
        out["explosive"] = round(max(deep), 6) if deep else 0.0
        scores["explosive"] = sorted(deep, reverse=True)
        return out, scores

    return values


def offense_candidates(art: Any, opponent_id: str, db: Any, evidence: Mapping[str, Any],
                       records: Sequence[Mapping[str, Any]] | None = None,
                       roster_input: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    from cfb_coach.madden.model.football_knowledge import concepts_in_plays
    from cfb_coach.madden.model.football_situation import roster_evidence

    roster = roster_evidence(dict(roster_input)) if roster_input is not None else load_verified_roster(db)
    opp_def = opponent_defense_profile(db, opponent_id)
    failing = {(r["formation"], r["play"]) for r in evidence.get("repeated_failures") or []}
    values = _offense_values(art, opponent_id, {})
    out = []
    for rec in records if records is not None else catalog_full.formation_records("offense"):
        plays = list(rec["plays"])
        if not plays:
            continue
        form = rec["formation"]
        ev = (evidence.get("formation") or {}).get(form) or {}
        tendency, tendency_why = _tendency_bonus(plays, opp_def)
        roster_delta, roster_why = _roster_bonus(plays, roster)
        failed = sorted(p for p in plays if (form, p) in failing)
        evidence_delta = EVIDENCE_WEIGHT * float(ev.get("delta") or 0.0) - FAILED_PLAY_COST * len(failed)
        shift = evidence_delta + tendency + roster_delta
        base_values, play_scores = values(form, plays)
        out.append({
            **dict(rec),
            "id": f"{form} | {rec['source_book']}",
            "values": {s: round(v + shift, 6) for s, v in base_values.items()},
            "play_scores": play_scores,
            "concepts": sorted(concepts_in_plays(plays)),
            "group": rec.get("personnel"),
            "runs": sum(catalog.is_run(p) for p in plays),
            "evidence": {"delta": round(evidence_delta, 5), "n": int(ev.get("n") or 0),
                         "rate": ev.get("rate"), "failed_plays": failed},
            "matchup": {"delta": round(tendency + roster_delta, 5), "tendency": tendency_why, "roster": roster_why},
        })
    return out


def fill_good_calls(cands: Sequence[dict[str, Any]], reference: Sequence[Mapping[str, Any]]) -> None:
    """Count plays within GOOD_PLAY_MARGIN of the catalog's top plays (99th percentile) per situation."""
    refs: dict[str, float] = {}
    for sit in OFFENSE_WEIGHTS:
        allv = sorted((s for c in reference for s in (c.get("play_scores") or {}).get(sit, [])), reverse=True)
        refs[sit] = allv[len(allv) // 100] if allv else 0.0
    for c in cands:
        c["good_calls"] = {sit: sum(1 for s in (c.get("play_scores") or {}).get(sit, []) if s >= ref - GOOD_PLAY_MARGIN)
                           for sit, ref in refs.items()}


def _eligible(form: str, situation: str) -> bool:
    from cfb_coach.madden.defense_select import eligible_formations

    picked = eligible_formations(DEF_PROBES[situation], {form: [], "__none__": []})
    return form in picked and "__none__" not in picked


def defense_candidates(db: Any, opponent_id: str, evidence: Mapping[str, Any],
                       records: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    from cfb_coach.madden.defense_select import COUNTERS, MIX

    profile = evidence.get("opponent_profile") or {}
    shrink = float(profile.get("shrink") or 0.0)
    run_share = profile.get("run_share")
    concepts = profile.get("concepts") or {}
    forms_ev = evidence.get("formation") or {}
    out = []
    for rec in records if records is not None else catalog_full.formation_records("defense"):
        plays = list(rec["plays"])
        if not plays:
            continue
        form = rec["formation"]
        pkg = rec.get("package") or catalog_full.defense_package(rec.get("family", ""))
        fams = dict((rec.get("profile") or {}).get("coverage_families") or {})
        values: dict[str, float] = {}
        matchup_parts: dict[str, float] = {}
        for sit, mix in MIX.items():
            menu = sum(share * min(1.0, fams.get(fam, 0) / 2.0) for fam, share in mix.items())
            fit = PACKAGE_FIT[sit].get(pkg, 0.4)
            base = 0.45 * fit + 0.45 * menu + (0.10 if _eligible(form, sit) else 0.0)
            adj = 0.0
            if run_share is not None and shrink > 0 and sit in ("early", "short", "red_zone", "goal_line"):
                adj += 0.3 * shrink * (float(run_share) - 0.5) * RUN_TILT.get(pkg, 0.0)
            if shrink > 0:
                for their, share in concepts.items():
                    for ours, w in (COUNTERS.get(their) or {}).items():
                        adj += 0.08 * shrink * float(share) * float(w) * min(1.0, fams.get(ours, 0) / 2.0)
            values[sit] = round(base + adj, 6)
            matchup_parts[sit] = round(adj, 5)
        ev = forms_ev.get(form) or {}
        shift = EVIDENCE_WEIGHT * float(ev.get("delta") or 0.0)
        out.append({
            **dict(rec),
            "id": f"{form} | {rec['source_book']}",
            "package": pkg,
            "values": {s: round(v + shift, 6) for s, v in values.items()},
            "good_calls": {s: (len(plays) if PACKAGE_FIT[s].get(pkg, 0.4) >= GOOD_PACKAGE_FIT else 0) for s in MIX},
            "concepts": sorted(f for f, n in fams.items() if n and f != "unclassified"),
            "group": PACKAGE_GROUP.get(pkg, pkg),
            "evidence": {"delta": round(EVIDENCE_WEIGHT * float(ev.get("delta") or 0.0), 5),
                         "n": int(ev.get("n") or 0), "rate": ev.get("rate"), "failed_plays": []},
            "matchup": {"delta": 0.0, "by_situation": matchup_parts,
                        "tendency": profile.get("summary") or "no learned tendencies"},
        })
    return out


# --------------------------------------------------------------------------- objective


def situation_weights(side: str, evidence: Mapping[str, Any]) -> dict[str, float]:
    """Base weights, raised where verified or situation-level results show a need."""
    weights = dict(OFFENSE_WEIGHTS if side == "offense" else DEFENSE_WEIGHTS)
    if side == "offense":
        prior = float(evidence.get("prior_success") or 0.4)
        for sit, row in (evidence.get("situation") or {}).items():
            if sit in weights and int(row.get("n") or 0) >= 5:
                weights[sit] *= 1.0 + max(-0.3, min(0.5, 2.0 * (prior - float(row["shrunk"]))))
    else:
        base = evidence.get("opponent_success_allowed")
        for sit, row in (evidence.get("situation_allowed") or {}).items():
            if base is not None and sit in weights and int(row.get("n") or 0) >= 5:
                weights[sit] *= 1.0 + max(-0.3, min(0.5, 2.0 * (float(row["rate"]) - float(base))))
    total = sum(weights.values())
    return {k: round(v / total, 5) for k, v in weights.items()}


def utility(members: Sequence[Mapping[str, Any]], side: str, weights: Mapping[str, float],
            opponent_type: str = "human") -> float:
    if not members:
        return 0.0
    shares = DEPTH_SHARES.get(opponent_type, DEPTH_SHARES["human"])
    total = 0.0
    for sit, w in weights.items():
        vals = sorted((float(m["values"].get(sit, 0.0)) for m in members), reverse=True)
        total += w * sum(s * v for s, v in zip(shares, vals))
    depth = PLAY_DEPTH_WEIGHT.get(opponent_type, PLAY_DEPTH_WEIGHT["human"])
    for sit, w in weights.items():
        n = sum(int((m.get("good_calls") or {}).get(sit, 0)) for m in members)
        total += depth * w * (1.0 - math.exp(-n / PLAY_DEPTH_SCALE))
    # Evidence and matchup live inside each formation's situation values, so
    # they count through the best and counter slots, never once per formation.
    total -= FORMATION_COST * len(members)
    total += PERSONNEL_WEIGHT * min(5, len({m.get("group") for m in members}))
    if side == "offense":
        concepts = {c for m in members for c in m.get("concepts") or []}
        total += CONCEPT_TOTAL * (1.0 - math.exp(-len(concepts) / CONCEPT_SCALE))
        plays = sum(len(m["plays"]) for m in members)
        runs = sum(int(m.get("runs") or 0) for m in members)
        share = runs / plays if plays else 0.0
        total -= BALANCE_PENALTY * max(0.0, 0.22 - share, share - 0.55)
        if len(members) >= 3 and not (any(int(m.get("runs") or 0) >= 3 for m in members)
                                      and any(len(m["plays"]) - int(m.get("runs") or 0) >= 4 for m in members)):
            total -= MISSING_SIDE_PENALTY
    else:
        total += PACKAGE_WEIGHT * len({m.get("group") for m in members} & {"base", "nickel", "dime", "goal_line"})
        calls: dict[str, int] = {}
        for m in members:
            for fam, n in ((m.get("profile") or {}).get("coverage_families") or {}).items():
                calls[fam] = calls.get(fam, 0) + int(n)
        from cfb_coach.madden.defense_select import FAMILIES

        total += COVERAGE_WEIGHT * sum(min(1.0, calls.get(f, 0) / 6.0) for f in FAMILIES)
        match = sum(int((m.get("profile") or {}).get("match_calls") or 0) for m in members)
        total += 0.004 * min(1.0, match / 3.0)
    return total


@dataclass(frozen=True)
class Objective:
    side: str
    weights: Mapping[str, float]
    opponent_type: str = "human"

    def __call__(self, members: Sequence[Mapping[str, Any]]) -> float:
        return utility(members, self.side, self.weights, self.opponent_type)


def feasible(members: Sequence[Mapping[str, Any]], cap: int) -> bool:
    names = [m["formation"] for m in members]
    return (len(members) <= cap and len(set(names)) == len(names)
            and sum(len(m["plays"]) for m in members) <= MAX_PLAYS
            and len({m.get("family") for m in members}) <= MAX_FAMILIES)


def _pool(cands: Sequence[dict[str, Any]], weights: Mapping[str, float], keep: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every candidate that could matter: top per situation, top overall, plus the installed book."""
    def mean(c: Mapping[str, Any]) -> float:
        return sum(w * float(c["values"].get(s, 0.0)) for s, w in weights.items())

    by_name: dict[str, list[dict[str, Any]]] = {}
    for c in cands:
        by_name.setdefault(c["formation"], []).append(c)
    reduced: list[dict[str, Any]] = []
    for rows in by_name.values():
        seen: set[tuple[str, ...]] = set()
        for c in sorted(rows, key=lambda r: (-mean(r), r.get("book_kind") != "team", r["source_book"])):
            sig = tuple(c["plays"])
            if sig in seen:
                continue
            seen.add(sig)
            reduced.append(c)
            if len(seen) >= 2:
                break
    chosen: dict[str, dict[str, Any]] = {c["id"]: c for c in keep}
    for sit in weights:
        for c in sorted(reduced, key=lambda r: (-float(r["values"].get(sit, 0.0)), r["id"]))[:POOL_PER_SITUATION]:
            chosen.setdefault(c["id"], c)
    for c in sorted(reduced, key=lambda r: (-mean(r), r["id"]))[:POOL_TOP]:
        chosen.setdefault(c["id"], c)
    for c in reduced:
        if c["evidence"]["n"] >= 5:
            chosen.setdefault(c["id"], c)
    return sorted(chosen.values(), key=lambda r: r["id"])


def _situation_story(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
                     weights: Mapping[str, float]) -> list[str]:
    out = []
    for sit in weights:
        b = max((float(m["values"].get(sit, 0.0)) for m in before), default=0.0)
        a = max((float(m["values"].get(sit, 0.0)) for m in after), default=0.0)
        if abs(a - b) >= 0.005:
            out.append(f"{sit.replace('_', ' ')} {b:.3f}->{a:.3f}")
    return out


def search(pool: Sequence[dict[str, Any]], start: Sequence[dict[str, Any]], obj: Objective, *,
           cap: int, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Best-improvement local search. Each accepted move must gain more than ``threshold``."""
    current = list(start)
    moves: list[dict[str, Any]] = []
    while not feasible(current, cap):
        drop = max(current, key=lambda c: (obj([m for m in current if m is not c]), c["id"]))
        current = [m for m in current if m is not drop]
        moves.append({"type": "remove", "removed": drop["id"], "gain": None,
                      "why": f"needed to respect the editor limits ({cap} formations, {MAX_PLAYS} plays)"})
    ids = {c["id"] for c in current}
    for _ in range(MAX_MOVES):
        base = obj(current)
        best: tuple[float, str, list[dict[str, Any]], dict[str, Any]] | None = None
        options: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        for c in current:
            options.append(([m for m in current if m is not c], {"type": "remove", "removed": c["id"]}))
        outside = [c for c in pool if c["id"] not in ids]
        names = {c["formation"] for c in current}
        for d in outside:
            if len(current) < cap and d["formation"] not in names:
                options.append((current + [d], {"type": "add", "added": d["id"]}))
            for c in current:
                if d["formation"] in names and d["formation"] != c["formation"]:
                    continue
                options.append(([m for m in current if m is not c] + [d],
                                {"type": "swap", "removed": c["id"], "added": d["id"]}))
        for members, move in options:
            if not feasible(members, cap):
                continue
            gain = obj(members) - base
            key = json.dumps(move, sort_keys=True)
            if gain > threshold and (best is None or (gain, key) > (best[0], best[1])):
                best = (gain, key, members, move)
        if best is None:
            break
        gain, _, members, move = best
        move = {**move, "gain": round(gain, 5),
                "situations": _situation_story(current, members, obj.weights)}
        moves.append(move)
        current = members
        ids = {c["id"] for c in current}
    return current, moves


# --------------------------------------------------------------------------- explanations


def _best_situations(c: Mapping[str, Any], members: Sequence[Mapping[str, Any]], weights: Mapping[str, float]) -> list[str]:
    roles = []
    for sit in weights:
        ranked = sorted(members, key=lambda m: -float(m["values"].get(sit, 0.0)))
        if ranked and ranked[0] is c:
            roles.append(f"best for {sit.replace('_', ' ')}")
        elif len(ranked) > 1 and ranked[1] is c:
            roles.append(f"counter for {sit.replace('_', ' ')}")
    return roles


def _why_member(c: Mapping[str, Any], members: Sequence[Mapping[str, Any]], obj: Objective,
                status: str, threshold: float) -> str:
    side = obj.side
    without = [m for m in members if m is not c]
    contribution = obj(members) - obj(without)
    roles = _best_situations(c, members, obj.weights)
    parts = []
    if roles:
        parts.append("; ".join(roles[:4]).capitalize())
    others = {x for m in without for x in m.get("concepts") or []}
    unique = [x for x in c.get("concepts") or [] if x not in others]
    if side == "offense":
        if unique:
            parts.append(f"only formation carrying {', '.join(u.replace('_', ' ') for u in unique[:4])}")
        if c.get("group") not in {m.get("group") for m in without}:
            parts.append(f"only {c.get('group')} personnel look")
    else:
        fams = (c.get("profile") or {}).get("coverage_families") or {}
        parts.append("coverage menu " + ", ".join(f"{k} {v}" for k, v in sorted(fams.items(), key=lambda kv: -kv[1])[:4]))
        if c.get("group") not in {m.get("group") for m in without}:
            parts.append(f"only {c.get('package')} package")
    if not roles and not unique:
        parts.insert(0, "Depth behind the situation leaders")
    ev = c["evidence"]
    if ev["n"]:
        parts.append(f"verified results: {ev['n']} snaps, shrunk success {float(ev['rate'] or 0):.2f}")
    if ev.get("failed_plays"):
        parts.append(f"repeated verified failures on {', '.join(ev['failed_plays'][:3])} (the live model already avoids them)")
    if side == "defense" and any(abs(v) >= 0.005 for v in (c["matchup"].get("by_situation") or {}).values()):
        parts.append(f"opponent tendency adjusts its value ({c['matchup']['tendency']})")
    if side == "offense" and c["matchup"]["delta"]:
        parts.append(f"{c['matchup']['tendency']}; {c['matchup']['roster']}")
    head = {"KEEP": "Keep", "ADD": "Add", "SOURCE": "Keep the formation, switch its play list to this source book"
            }.get(status, status.title())
    if contribution >= 0:
        tail = f"removing it would lower the portfolio score by {contribution:.4f}"
        if status == "ADD":
            tail += f" (switch threshold {threshold:.4f})"
    else:
        tail = (f"removing it would raise the score by only {-contribution:.4f}, under the "
                f"{threshold:.4f} switch threshold, so the installed formation stays")
    return f"{head}: " + ". ".join(p[0].upper() + p[1:] for p in parts if p) + f". {tail[0].upper() + tail[1:]}."


def _why_removed(c: Mapping[str, Any], before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
                 obj: Objective, moves: Sequence[Mapping[str, Any]]) -> str:
    weights = obj.weights
    move = next((m for m in moves if m.get("removed") == c["id"]), None)
    parts = []
    if move and move.get("gain") is None:
        return f"Remove: {move['why']}."
    replaced = move.get("added") if move and move.get("type") == "swap" else None
    if replaced:
        parts.append(f"replaced by {replaced.split(' | ')[0]}")
    covered = []
    for sit in weights:
        val = float(c["values"].get(sit, 0.0))
        best_after = max((float(m["values"].get(sit, 0.0)) for m in after), default=0.0)
        if val > 0 and best_after >= val - 0.005:
            covered.append(sit.replace("_", " "))
    if covered:
        parts.append(f"its situations ({', '.join(covered[:4])}) are covered at least as well by the new book")
    ev = c["evidence"]
    if ev["n"] and float(ev["delta"]) < 0:
        parts.append(f"verified results were below average ({ev['n']} snaps, shrunk success {float(ev['rate'] or 0):.2f})")
    if move and move.get("gain") is not None:
        parts.append(f"the change gains {move['gain']:.4f} on the portfolio score")
    return "Remove: " + ("; ".join(parts) or "its contribution no longer covers its install cost") + "."


# --------------------------------------------------------------------------- current book


def _current_members(side: str, applied: Mapping[str, Any] | None, by_id: Mapping[str, dict[str, Any]],
                     cands: Sequence[dict[str, Any]], build: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Installed formations as catalog candidates (full source play list) plus notes."""
    if not applied:
        return [], []
    sources = dict(applied.get("formation_sources") or {})
    default_source = str(applied.get("source_book") or applied.get("name") or "")
    members, notes = [], []
    for form, installed in (applied.get("formations") or {}).items():
        installed = list(installed or [])
        src = sources.get(form) or default_source
        cand = by_id.get(f"{form} | {src}")
        how = "source book"
        if cand is None:
            options = [c for c in cands if c["formation"] == form and set(installed) <= set(c["plays"])]
            cand = min(options, key=lambda c: (len(c["plays"]), c["source_book"])) if options else None
            how = "closest catalogued source"
        if cand is None:
            rec = {"side": side, "formation": form, "family": form.split()[0], "set": form,
                   "source_book": src or "unknown", "plays": installed, "book_kind": "installed",
                   "verification": {"status": "installed_uncatalogued", "in_game_verified": False}}
            if side == "offense":
                prof = catalog_full.offense_profile(installed)
                rec.update(personnel=catalog_full.offense_personnel(rec["family"], form), profile=prof)
            else:
                prof = catalog_full.defense_profile(installed)
                rec.update(package=catalog_full.defense_package(rec["family"]), profile=prof)
            cand = build([rec])[0]
            how = "not in the catalog; kept with its installed plays"
        missing = [p for p in cand["plays"] if p not in installed]
        notes.append({"formation": form, "source_book": cand["source_book"], "match": how,
                      "installed_plays": len(installed), "source_plays": len(cand["plays"]),
                      "plays_to_restore": missing})
        members.append(cand)
    return members, notes


# --------------------------------------------------------------------------- design


def design_side(side: str, *, opponent_id: str, db: Any, applied: Mapping[str, Any] | None,
                evidence: Mapping[str, Any], art: Any = None, max_formations: int = MAX_FORMATIONS,
                evidence_quality: str = "prior_driven", records: Sequence[Mapping[str, Any]] | None = None,
                opponent_type: str = "human", roster_input: Mapping[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    cap = max(1, min(int(max_formations), MAX_FORMATIONS))
    if side == "offense":
        cands = offense_candidates(art, opponent_id, db, evidence, records, roster_input)
        build = lambda recs: offense_candidates(art, opponent_id, db, evidence, recs, roster_input)  # noqa: E731
        evidence_n = int(evidence.get("labeled_rows") or 0)
    else:
        cands = defense_candidates(db, opponent_id, evidence, records)
        build = lambda recs: defense_candidates(db, opponent_id, evidence, recs)  # noqa: E731
        evidence_n = int(evidence.get("verified_executions") or 0)
    by_id = {c["id"]: c for c in cands}
    weights = situation_weights(side, evidence)
    obj = Objective(side, weights, opponent_type)
    current, current_notes = _current_members(side, applied, by_id, cands, build)
    if side == "offense":
        fill_good_calls(list(cands) + [c for c in current if c["id"] not in by_id], cands)
    pool = _pool(cands, weights, current)
    threshold = SWITCH_GAIN.get(evidence_quality, SWITCH_GAIN["prior_driven"])
    if evidence_n < 30:
        threshold += THIN_EVIDENCE_EXTRA
    ideal, _ = search(pool, [], obj, cap=cap, threshold=1e-9)
    final, moves = search(pool, current, obj, cap=cap, threshold=threshold)
    cur_ids = {c["id"] for c in current}
    cur_names = {c["formation"]: c for c in current}
    final_ids = {c["id"] for c in final}
    final_names = {c["formation"] for c in final}
    status = {c["id"]: ("KEEP" if c["id"] in cur_ids else "SOURCE" if c["formation"] in cur_names else "ADD")
              for c in final}
    order = sorted(final, key=lambda c: ({"KEEP": 0, "SOURCE": 1}.get(status[c["id"]], 2), c["family"], c["formation"]))
    formations = []
    for c in order:
        formations.append({
            "status": status[c["id"]],
            "formation": c["formation"], "family": c["family"], "family_ingame": family_ingame(c["family"]),
            "set": c.get("set"), "source_book": c["source_book"], "book_kind": c.get("book_kind"),
            "plays": list(c["plays"]), "n_plays": len(c["plays"]),
            "personnel": c.get("personnel"), "package": c.get("package"),
            "profile": c.get("profile"), "situations": c.get("situations"),
            "source_url": c.get("source_url"), "verification": c.get("verification"),
            "situation_values": c["values"], "evidence": c["evidence"],
            "why": _why_member(c, final, obj, status[c["id"]], threshold)
            + (f" Was installed with the {cur_names[c['formation']]['source_book']} play list."
               if status[c["id"]] == "SOURCE" else ""),
            "previous_source_book": cur_names[c["formation"]]["source_book"] if status[c["id"]] == "SOURCE" else None,
            "editor_path": {
                "formation": family_ingame(c["family"]), "set": c.get("set") or c["formation"],
                "source_playbook": c["source_book"],
                "note": "Menu labels come from public guides; Madden 27 wording may differ.",
            },
        })
    removed = [{
        "status": "REMOVE", "formation": c["formation"], "family": c["family"], "source_book": c["source_book"],
        "n_plays": len(c["plays"]), "why": _why_removed(c, current, final, obj, moves),
    } for c in current if c["id"] not in final_ids and c["formation"] not in final_names]
    deferred = []
    for c in ideal:
        if c["id"] in final_ids or any(f["formation"] == c["formation"] for f in formations):
            continue
        trial = [m for m in final] + [c]
        if len(trial) <= cap and feasible(trial, cap):
            gain = obj(trial) - obj(final)
            deferred.append({"formation": c["formation"], "source_book": c["source_book"],
                             "gain_if_added": round(gain, 5),
                             "why_not": f"gain {gain:.4f} does not clear the switch threshold {threshold:.4f}"})
        else:
            deferred.append({"formation": c["formation"], "source_book": c["source_book"], "gain_if_added": None,
                             "why_not": "would need a swap whose gain does not clear the switch threshold"})
    plays = sum(len(c["plays"]) for c in final)
    book = {
        "name": BOOK_NAMES[side], "custom": True, "trimmed": False,
        "formations": {c["formation"]: list(c["plays"]) for c in final},
        "formation_sources": {c["formation"]: c["source_book"] for c in final},
    }
    if final:
        counts: dict[str, int] = {}
        for c in final:
            counts[c["source_book"]] = counts.get(c["source_book"], 0) + 1
        book["source_book"] = max(counts, key=lambda k: (counts[k], k))
    return {
        "side": side,
        "book": book,
        "fingerprint": fingerprint(side, book["formations"], book["formation_sources"]),
        "formations": formations,
        "removed": removed,
        "changes": {
            "keep": [f["formation"] for f in formations if f["status"] == "KEEP"],
            "add": [f["formation"] for f in formations if f["status"] == "ADD"],
            "new_source": [f["formation"] for f in formations if f["status"] == "SOURCE"],
            "remove": [r["formation"] for r in removed],
        },
        "current_book": {"name": (applied or {}).get("name"), "formations": list(((applied or {}).get("formations") or {})),
                         "revision": (applied or {}).get("revision"), "matching": current_notes},
        "moves": moves,
        "ideal": [{"formation": c["formation"], "source_book": c["source_book"]} for c in ideal],
        "deferred": deferred,
        "objective": {
            "current": round(obj(current), 5),
            "proposed": round(obj(final), 5),
            "ideal": round(obj(ideal), 5),
            "switch_threshold": round(threshold, 5),
            "opponent_type": opponent_type,
            "situation_weights": weights,
            "evidence_quality": evidence_quality,
            "evidence_rows": evidence_n,
        },
        "limits": {
            "formations": len(final), "max_formations": cap, "plays": plays, "max_plays": MAX_PLAYS,
            "families": len({c["family"] for c in final}), "max_families": MAX_FAMILIES,
            "editor_limits": catalog_full.EDITOR_LIMITS,
        },
        "catalog": {"candidates_scored": len(cands), "search_pool": len(pool),
                    "formations_inspected": len({c["formation"] for c in cands})},
        "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
    }


def no_schedule_statement() -> str:
    return ("Changes come only from the portfolio score: a formation enters when it adds more than the "
            "switch threshold, leaves when the book is better without it, and nothing changes on a timer, "
            "a game count, a snap count, or a win/loss.")


__all__ = [
    "BOOK_NAMES", "DEFENSE_WEIGHTS", "MAX_FORMATIONS", "MAX_PLAYS", "OFFENSE_WEIGHTS", "SWITCH_GAIN", "Objective",
    "defense_candidates", "design_side", "digest", "feasible", "fingerprint", "offense_candidates",
    "search", "situation_weights", "utility",
]
