"""Madden defense picker — the whole locked defensive book, not one formation.

Two steps every snap:
  1. Coverage family (two-high / Cover 2 / single-high / man / pressure) from a
     situational mix, shifted by what THIS opponent runs in this down bucket (your
     logs), what the daily research says answers it, how each family has done vs
     him, and what we just showed (no family three snaps running).
  2. A (formation, call) inside that family from the formations that fit the
     down-and-distance, ranked by learned results vs this opponent, research
     backing, and an exact-repeat penalty.

Calls benched this half (``cfb_coach.ingame``) are never picked; a family that keeps
getting beaten this half is penalized.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from cfb_coach import ingame
from cfb_coach.madden.situation import Situation

FAMILIES = ("two_high", "cover2", "single_high", "man", "pressure")
FAMILY_LABEL = {
    "two_high": "two-high (C4/C6/C9)",
    "cover2": "Cover 2 / Tampa",
    "single_high": "single-high (C3/match)",
    "man": "man (C1/C2 man/bracket)",
    "pressure": "pressure (sim/blitz)",
}
FAMILY_USER_JOB = {
    "two_high": "User the deep middle (rob the post)",
    "cover2": "User the hole between the safeties",
    "single_high": "User the hook/curl (rob crossers)",
    "man": "User the robber (read QB eyes)",
    "pressure": "User QB contain/spy (don't vacate)",
}

_PRESSURE = re.compile(
    r"blitz|\bsim\b|fire|storm|\bdog\b|sting|smoke|loop|thunder|eagle|okie|overload|\btex\b|\bgo\b|zero|cover\s*0",
    re.I,
)
_MAN = re.compile(r"cover\s*1|\bman\b|^1\s|bracket|press|robber|\bhole\b|\block\b|double\s*(wr|te|slot)", re.I)
_TWO_HIGH = re.compile(r"cover\s*[469]|cov\s*[469]|quarters|palms|willie|\bdrop\b", re.I)
_COVER2 = re.compile(r"cover\s*2|cov\s*2|tampa|invert|\btrap\b|\b2\s*invert", re.I)
_SINGLE = re.compile(r"cover\s*3|cov\s*3|match|\bsky\b|cloud|buzz|seam|\b3\b", re.I)

# Situational family mix (shares sum to 1)
MIX = {
    "early": {"two_high": 0.30, "single_high": 0.30, "cover2": 0.15, "man": 0.10, "pressure": 0.15},
    "short": {"single_high": 0.30, "man": 0.25, "pressure": 0.25, "two_high": 0.10, "cover2": 0.10},
    "medium": {"two_high": 0.20, "single_high": 0.20, "cover2": 0.15, "man": 0.20, "pressure": 0.25},
    "long": {"two_high": 0.35, "cover2": 0.25, "single_high": 0.15, "man": 0.10, "pressure": 0.15},
    "red_zone": {"man": 0.30, "single_high": 0.25, "pressure": 0.20, "cover2": 0.15, "two_high": 0.10},
    "goal_line": {"man": 0.40, "pressure": 0.35, "single_high": 0.25},
    "two_minute": {"two_high": 0.35, "cover2": 0.30, "single_high": 0.20, "man": 0.05, "pressure": 0.10},
}

# Their concept family -> how much each of our coverage families answers it
COUNTERS: dict[str, dict[str, float]] = {
    "vert": {"two_high": 0.6, "cover2": 0.2},
    "flood": {"two_high": 0.35, "single_high": 0.3, "man": 0.2},
    "cross": {"man": 0.45, "cover2": 0.35},
    "stack": {"single_high": 0.35, "two_high": 0.25, "man": -0.2},
    "run": {"single_high": 0.4, "pressure": 0.25, "two_high": -0.2},
    "rpo": {"single_high": 0.3, "man": 0.2},
    "scram": {"man": 0.2, "single_high": 0.2, "pressure": -0.3},
}

SCOUT_WEIGHT = 0.9
RESEARCH_FAMILY_BONUS = 0.25
RESEARCH_CALL_BONUS = 0.2
BASELINE_CALL_BONUS = 0.15
FAMILY_REPEAT_PENALTY = 0.35  # per snap of the same family in the last two
CALL_REPEAT_PENALTY = 0.5  # per use of the exact call in the last four
FORMATION_REPEAT_PENALTY = 0.15
LEARNED_WEIGHT = 0.6
CALL_TEMPERATURE = 0.3
TOP_CALLS = 6


def call_family(name: str | None) -> str | None:
    """Coverage family of a Madden defensive call name (book vocabulary)."""
    n = (name or "").strip()
    if not n:
        return None
    if _PRESSURE.search(n):
        return "pressure"
    if _MAN.search(n):
        return "man"
    if re.search(r"tampa", n, re.I):
        return "cover2"
    if _TWO_HIGH.search(n):
        return "two_high"
    if _COVER2.search(n):
        return "cover2"
    if _SINGLE.search(n):
        return "single_high"
    return None


def coverage_seen_family(cov: str | None) -> str | None:
    """Family of a coverage string seen on film (Cover 6, pressure, Cover 3 Sky …)."""
    c = (cov or "").lower()
    if not c:
        return None
    if "pressure" in c or "blitz" in c:
        return "pressure"
    return call_family(cov)


def mix_key(sit: Situation) -> str:
    if sit.goal_line:
        return "goal_line"
    if sit.red_zone:
        return "red_zone"
    if sit.two_minute:
        return "two_minute"
    dist = sit.distance if sit.distance is not None else 10
    if sit.down in (2, 3, 4) and dist <= 3:
        return "short"
    if sit.down in (3, 4) and dist >= 7 or (sit.down == 2 and dist >= 12):
        return "long"
    if sit.down in (3, 4):
        return "medium"
    return "early"


def eligible_formations(sit: Situation, book: dict[str, list[str]]) -> list[str]:
    """Formations that fit the down-and-distance (from the locked book)."""
    forms = list(book)
    key = mix_key(sit)

    def having(*pats: str) -> list[str]:
        return [f for f in forms if any(re.search(p, f, re.I) for p in pats)]

    if key in ("goal_line", "short"):
        pick = having(r"goal\s*line", r"4-3", r"4-4", r"46", r"double\s*mug", r"big")
    elif key == "long":
        pick = having(r"dime", r"dollar", r"quarter", r"nickel")
    elif key == "two_minute":
        pick = having(r"dime", r"nickel")
    else:
        pick = having(r"nickel", r"4-3", r"3-4", r"mug")
    return pick or forms


@dataclass
class DefensePick:
    formation: str
    play: str
    family: str
    rationale: str
    user_job: str
    mix: dict[str, float] = field(default_factory=dict)
    # Exact P(formation, play) from this call's family mix and within-family softmax.
    # Keys are the pairs ``rng.choices`` could return. Values sum to 1.
    call_probs: dict[tuple[str, str], float] = field(default_factory=dict)


def _recent(records: list[ingame.CallRecord], n: int) -> list[ingame.CallRecord]:
    return records[-n:] if records else []


def _learned(lw: Any, zone: str, form: str, play: str) -> float:
    try:
        return float(lw.learned_score(zone, form, play)["learned"])
    except Exception:  # noqa: BLE001
        return 0.0


def _family_history(report: dict[str, Any] | None) -> dict[str, float]:
    """How each of our coverage families has held up vs this opponent (stop rate - 0.5)."""
    out: dict[str, list[float]] = {}
    for c in ((report or {}).get("their_offense") or {}).get("our_calls") or []:
        fam = call_family(c.get("play"))
        if fam:
            out.setdefault(fam, []).extend([float(c["success"])] * int(c["n"]))
    return {f: round(sum(v) / len(v) - 0.5, 3) for f, v in out.items() if len(v) >= 4}


def select_defense(
    sit: Situation,
    opponent_id: str,
    db: Any,
    book: dict[str, list[str]],
    rng: random.Random,
    *,
    baseline: dict[str, Any] | None = None,
    scouting: dict[str, Any] | None = None,
    research_counters: list[dict[str, Any]] | None = None,
    exclude_families: set[str] | None = None,
    lw: Any = None,
) -> DefensePick:
    from cfb_coach.madden.data import user_job_for
    from cfb_coach.scouting import expected_families
    from cfb_coach.zones import zone_of_situation

    key = mix_key(sit)
    from cfb_coach.game_score import shift_defense_mix

    mix = shift_defense_mix(dict(MIX[key]), sit)
    notes: list[str] = []
    forms = eligible_formations(sit, book)
    pool = [(f, p) for f in forms for p in book.get(f, [])]
    by_family: dict[str, list[tuple[str, str]]] = {}
    for f, p in pool:
        fam = call_family(p)
        if fam:
            by_family.setdefault(fam, []).append((f, p))

    session_id, quarter = ingame.situation_context(sit)
    records = ingame.session_records(db, session_id, "defense")
    bench = ingame.bench_report(db, session_id, "defense", quarter, family_of=call_family, records=records)

    adjust = {fam: 0.0 for fam in FAMILIES}
    # Their tendencies in this down bucket (your logs) -> counter families
    fams, scope, n = expected_families(scouting, sit)
    if fams:
        conf = n / (n + 6.0)
        for their, share in fams.items():
            for ours, w in (COUNTERS.get(their) or {}).items():
                adjust[ours] += SCOUT_WEIGHT * conf * share * w
        top = max(fams, key=fams.get)
        notes.append(f"scout {scope}: {top} {round(fams[top] * 100)}% (n={n})")
        # Daily research counters for this opponent's concepts
        for rc in research_counters or []:
            if rc.get("vs_family") in fams:
                for cls in rc.get("families") or []:
                    if cls in adjust:
                        adjust[cls] += RESEARCH_FAMILY_BONUS * fams[rc["vs_family"]]
    for fam, delta in _family_history(scouting).items():
        adjust[fam] += 0.5 * delta
    for fam, pen in bench.family_penalty.items():
        adjust[fam] -= pen
        notes.append(f"{FAMILY_LABEL.get(fam, fam)} beaten this {bench.scope} (-{pen})")
    last2 = [call_family(r.play) for r in _recent(records, 2)]
    for fam in last2:
        if fam in adjust:
            adjust[fam] -= FAMILY_REPEAT_PENALTY
    if len(last2) == 2 and last2[0] and last2[0] == last2[1]:
        adjust[last2[0]] -= 2.0  # never three in a row

    weights = {}
    for fam in FAMILIES:
        if fam not in by_family or fam in (exclude_families or set()):
            continue
        live = [fp for fp in by_family[fam] if not bench.is_benched(*fp)]
        if not live:
            continue
        weights[fam] = mix.get(fam, 0.03) * math.exp(adjust[fam])
    if not weights:  # everything excluded/benched: fall back to any family in the pool
        weights = {fam: mix.get(fam, 0.05) for fam in by_family} or {"single_high": 1.0}
    total = sum(weights.values())
    shares = {f: round(w / total, 3) for f, w in weights.items()}
    fam = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]

    # Step 2: a call inside the family. Scoring does not touch rng, so the two
    # choices below are the same draws as before call_probs existed.
    bl = baseline or {}
    dg = bl.get("defense_gameplan") or {}
    backed = set((dg.get("home_rotation") or {}).keys())
    for st in (dg.get("situational") or {}).values():
        backed.update(st.get("calls") or [])
    named = {c for rc in research_counters or [] for c in rc.get("calls") or []}
    zone = zone_of_situation(sit)
    recent4 = [(r.formation, r.play) for r in _recent(records, 4)]
    last_form = records[-1].formation if records else None

    def _score_family(family: str) -> list[tuple[float, str, str]]:
        cands = [fp for fp in by_family.get(family, pool) if not bench.is_benched(*fp)] or by_family.get(family, pool) or pool
        scored_rows: list[tuple[float, str, str]] = []
        for f, p in cands:
            s = 0.0
            s += LEARNED_WEIGHT * _learned(lw, zone, f, p) if lw is not None else 0.0
            s += BASELINE_CALL_BONUS if p in backed else 0.0
            s += RESEARCH_CALL_BONUS if p in named else 0.0
            s -= CALL_REPEAT_PENALTY * recent4.count((f, p))
            s -= FORMATION_REPEAT_PENALTY if f == last_form else 0.0
            scored_rows.append((s, f, p))
        scored_rows.sort(key=lambda t: -t[0])
        return scored_rows

    scored = _score_family(fam)
    top = scored[:TOP_CALLS]
    probs = [math.exp(s / CALL_TEMPERATURE) for s, _, _ in top]
    _, form, play = rng.choices(top, weights=probs, k=1)[0]
    call_probs: dict[tuple[str, str], float] = {}
    for family, weight in weights.items():
        top_i = _score_family(family)[:TOP_CALLS]
        raw = [math.exp(s / CALL_TEMPERATURE) for s, _, _ in top_i]
        denom = sum(raw)
        if denom <= 0 or total <= 0:
            continue
        share = weight / total
        for (_s, f, p), r in zip(top_i, raw):
            call_probs[(f, p)] = call_probs.get((f, p), 0.0) + share * (r / denom)

    mix_txt = " · ".join(f"{FAMILY_LABEL[f].split(' (')[0]} {round(v * 100)}%" for f, v in sorted(shares.items(), key=lambda kv: -kv[1]))
    rationale = f"D mix ({key.replace('_', ' ')}): {mix_txt} → {FAMILY_LABEL[fam]}"
    if notes:
        rationale += " | " + "; ".join(notes)
    if bench.benched:
        rationale += f" | {bench.note()}"
    job = user_job_for(play)
    if job == "User hook":
        job = FAMILY_USER_JOB.get(fam, job)
    return DefensePick(form, play, fam, rationale, job, shares, call_probs)


__all__ = [
    "COUNTERS",
    "FAMILIES",
    "DefensePick",
    "call_family",
    "coverage_seen_family",
    "eligible_formations",
    "mix_key",
    "select_defense",
]
