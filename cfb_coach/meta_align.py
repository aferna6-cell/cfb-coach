"""Align the game plan to the CURRENT CFB 27 meta (v1.12).

Two inputs:
  * Seed research (``meta_baseline.json`` -> ``meta_research``): cited, dated,
    zone-aware priors restricted to plays in Aidan's formations.
  * Live scout signals (``meta_scout.run_meta_scout``): per-concept counts of
    sources mentioning a concept, in general and in red-zone context. These add a
    small, capped boost on top of the seed priors.

A prior is a *nudge* in the same normalized units as the learned score
(roughly -1..+1). It fades as Aidan's own sample in that zone grows
(``REC_META_PRIOR_OBS`` pseudo-observations), so his data wins once it exists.
Conflicts = places where the meta and his own results disagree.

Offense-only vs the CPU. Doctrine (max 8 macros, Ohio State lab / Alabama
serious) is untouched: lab candidates are only surfaced for Ohio State.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from cfb_coach.zones import GOAL_LINE, OPEN, RED_ZONE, ZONE_LABELS

# --- Constants (documented in README) ---
PRIOR_CAP = 0.4  # |total meta prior| per play/zone
LIVE_PER_SOURCE = 0.03  # live boost per source mentioning a concept
LIVE_CAP = 0.12  # max live boost per play/zone
NAMED_PER_PAIR = 0.08  # v1.13: max boost from a play being named in current sources
LAB_PRIOR = 0.10  # v1.13: seed prior for lab candidates in their zones
CONFLICT_PRIOR_MIN = 0.1  # meta "likes" a play in a zone
CONFLICT_LEARNED_MAX = -0.2  # ... but his zone learned score is this bad
CONFLICT_MIN_SNAPS = 3
SUPPORT_LEARNED_MIN = 0.25  # his data likes it while the meta is cold/negative

# Live concept -> plays in Aidan's formations (no invented plays)
CONCEPT_PLAYS: dict[str, list[tuple[str, str]]] = {
    "inside_zone": [("Gun Bunch X Nasty", "Inside Zone"), ("Singleback Deuce Close", "Inside Zone Split")],
    "duo_power": [("Singleback Deuce Close", "Mtn Duo"), ("Gun Bunch X Nasty", "Counter Y")],
    "dive": [("Singleback Deuce Close", "HB Dive")],
    "mesh": [("Gun Bunch X Nasty", "Mesh Spot"), ("Gun Cluster", "Mesh Post")],
    "whip": [("Gun Bunch X Nasty", "RZ PA X Whip")],
    "spot_flat": [("Gun Bunch X Nasty", "Z Spot GoalLine"), ("Gun Cluster", "Z Spot Shake")],
    "rpo": [("Gun Bunch X Nasty", "Mtn RPO Zone Alert")],
    "play_action": [("Gun Bunch X Nasty", "RZ PA X Whip")],
    "run_first": [("Gun Bunch X Nasty", "Inside Zone"), ("Gun Bunch X Nasty", "HB Base"), ("Singleback Deuce Close", "Mtn Duo")],
    # CPU goal-line wall -> quick throws + single-back dive (community reports)
    "cpu_gl_wall": [("Gun Bunch X Nasty", "Z Spot GoalLine"), ("Singleback Deuce Close", "HB Dive")],
}
# Concepts whose red-zone mentions only count toward rz/gl (not the open field)
RZ_ONLY = {"cpu_gl_wall"}


def load_research() -> dict[str, Any]:
    from cfb_coach.gameplan import load_baseline

    return load_baseline().get("meta_research") or {}


@dataclass
class MetaPriors:
    """Zone-aware prior per (formation, play), seed + live, capped."""

    seed: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    live: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    live_mode: str = "none"
    live_fetched_at: str = ""
    findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    lab_candidates: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(cls, scout: Any | None = None, research: dict[str, Any] | None = None) -> "MetaPriors":
        research = research if research is not None else load_research()
        mp = cls()
        mp.findings = {f.get("id", ""): f for f in research.get("findings") or []}
        mp.lab_candidates = list(research.get("lab_candidates") or [])
        for p in research.get("priors") or []:
            mp.seed[(p["formation"], p["play"])] = p
        if scout is not None:
            sd = scout if isinstance(scout, dict) else scout.to_dict()
            mp.live_mode = sd.get("mode") or ("live" if sd.get("available") else "none")
            mp.live_fetched_at = sd.get("fetched_at") or ""
            if mp.live_mode in ("live", "cache"):
                mp.live = live_boosts(sd.get("concept_signals") or {}, sd.get("rz_signals") or {})
                mp.add_named_boosts(sd.get("named_signals") or {})
        mp.add_lab_priors()
        return mp

    def add_named_boosts(self, named: dict[str, Any]) -> None:
        """v1.13: formations/plays NAMED in this prep's sources (web + YouTube
        transcripts) get a small zone-fit live boost, capped with the rest at LIVE_CAP."""
        try:
            from cfb_coach.cfb_catalog import zone_fit
        except Exception:  # noqa: BLE001
            return
        for key, v in (named.get("pairs") or {}).items():
            if "::" not in key:
                continue
            f, p = key.split("::", 1)
            b = min(LIVE_CAP, NAMED_PER_PAIR * math.tanh(float(v.get("score", 0.0)) / 1.5))
            if b <= 0.0:
                continue
            d = self.live.setdefault((f, p), {OPEN: 0.0, RED_ZONE: 0.0, GOAL_LINE: 0.0})
            for z in (OPEN, RED_ZONE, GOAL_LINE):
                if zone_fit(p, z):
                    d[z] = round(min(LIVE_CAP, d.get(z, 0.0) + b), 3)

    def add_lab_priors(self) -> None:
        """Lab candidates get a small seed prior in their zones so, once they are in
        the (Ohio State) custom book, the caller actually tries them."""
        zmap = {"gl": GOAL_LINE, "rz": RED_ZONE, "open": OPEN}
        for lc in self.lab_candidates:
            key = (lc.get("formation", ""), lc.get("play", ""))
            if not key[0] or key in self.seed:
                continue
            zones = {zmap.get(z, z): LAB_PRIOR for z in (lc.get("zones") or [])}
            self.seed[key] = {"formation": key[0], "play": key[1], "zones": zones, "refs": list(lc.get("refs") or []),
                              "why": f"lab candidate: {lc.get('note', '')}", "lab": True}

    @classmethod
    def load_cached(cls) -> "MetaPriors":
        """Seed + last cached live scout (no network). Used by the live caller."""
        scout = None
        try:
            from cfb_coach.meta_scout import cache_path

            path = cache_path()
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
                raw.setdefault("mode", "cache")
                scout = raw
        except (OSError, ValueError, AttributeError):
            scout = None
        try:
            return cls.build(scout)
        except Exception:  # noqa: BLE001 — never break play calling
            return cls()

    def prior(self, zone: str, formation: str, play: str, coverage: str | None = None) -> dict[str, Any]:
        key = (formation, play)
        seed = self.seed.get(key) or {}
        s = float((seed.get("zones") or {}).get(zone, 0.0))
        if coverage and seed.get("coverage"):
            cov = seed["coverage"]
            low = coverage.lower()
            for c, v in cov.items():
                if c.lower() in low or (c == "pressure" and any(t in low for t in ("blitz", "pressure", "cover 0"))):
                    s += float(v) * 0.5
                    break
        lv = float((self.live.get(key) or {}).get(zone, 0.0))
        total = max(-PRIOR_CAP, min(PRIOR_CAP, s + lv))
        return {"seed": round(s, 3), "live": round(lv, 3), "prior": round(total, 3), "refs": list(seed.get("refs") or []), "why": seed.get("why", "")}

    def cite(self, refs: list[str]) -> list[str]:
        out = []
        for r in refs:
            f = self.findings.get(r)
            if f:
                out.append(f"{f.get('source')} ({f.get('published')})")
        return out


def live_boosts(general: dict[str, int], rz: dict[str, int]) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for concept, plays in CONCEPT_PLAYS.items():
        g = int(general.get(concept, 0) or 0)
        r = int(rz.get(concept, 0) or 0)
        if not g and not r:
            continue
        open_b = 0.0 if concept in RZ_ONLY else min(LIVE_CAP, LIVE_PER_SOURCE * g) * 0.5
        rz_b = min(LIVE_CAP, LIVE_PER_SOURCE * r + (0.0 if concept in RZ_ONLY else 0.01 * g))
        for fp in plays:
            d = out.setdefault(fp, {OPEN: 0.0, RED_ZONE: 0.0, GOAL_LINE: 0.0})
            d[OPEN] = min(LIVE_CAP, d[OPEN] + open_b)
            d[RED_ZONE] = min(LIVE_CAP, d[RED_ZONE] + rz_b)
            d[GOAL_LINE] = min(LIVE_CAP, d[GOAL_LINE] + rz_b)
    return {k: {z: round(v, 3) for z, v in d.items()} for k, d in out.items()}


# ---------------------------------------------------------------------------
# Combined score (shared by the live caller and the prep plan)
# ---------------------------------------------------------------------------


def combined_score(
    lw: Any,
    priors: MetaPriors | None,
    zone: str,
    formation: str,
    play: str,
    *,
    coverage: str | None = None,
    coverage_source: str = "none",
) -> dict[str, Any]:
    from cfb_coach.learning import REC_META_PRIOR_OBS, REC_UNTESTED_PENALTY, _k_play, zone_key

    parts = lw.learned_score(zone, formation, play, coverage=coverage, coverage_source=coverage_source)
    n_obs = parts.get("n_obs_zone", 0.0)
    pr = priors.prior(zone, formation, play, coverage if coverage_source in ("live", "last") else None) if priors else {"prior": 0.0, "refs": [], "why": "", "seed": 0.0, "live": 0.0}
    fade = REC_META_PRIOR_OBS / (REC_META_PRIOR_OBS + n_obs)
    meta_term = pr["prior"] * fade
    pk = _k_play(formation, play)
    tested = lw.n(pk if zone == OPEN else zone_key(zone, pk)) > 0 or lw.n(pk) > 0
    untested = 0.0 if tested else REC_UNTESTED_PENALTY
    total = parts["learned"] + meta_term + untested
    return {
        "formation": formation,
        "play": play,
        "zone": zone,
        "learned": round(parts["learned"], 3),
        "meta": round(meta_term, 3),
        "meta_raw": pr["prior"],
        "meta_refs": pr["refs"],
        "meta_why": pr["why"],
        "untested": untested,
        "n_zone": int(parts.get("n_zone", 0)),
        "total": round(total, 3),
    }


def softmax_probs(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    t = max(1e-6, temperature)
    m = max(scores)
    ex = [math.exp((s - m) / t) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]


# ---------------------------------------------------------------------------
# Prep: alignment table + conflicts
# ---------------------------------------------------------------------------


def candidate_plays(zone: str) -> list[tuple[str, str]]:
    from cfb_coach.playcaller import zone_candidates

    return zone_candidates(zone)


def alignment(lw: Any, priors: MetaPriors, *, dynasty: str = "ohio_state") -> dict[str, Any]:
    """Per-zone ranked plan + conflicts between the meta and Aidan's own data."""
    from cfb_coach.dynasty import allow_experimental, normalize_dynasty

    zones_out: dict[str, list[dict[str, Any]]] = {}
    conflicts: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for zone in (OPEN, RED_ZONE, GOAL_LINE):
        rows = [combined_score(lw, priors, zone, f, p) for f, p in candidate_plays(zone)]
        rows.sort(key=lambda r: -r["total"])
        zones_out[zone] = rows
        for r in rows:
            if r["meta_raw"] >= CONFLICT_PRIOR_MIN and r["learned"] <= CONFLICT_LEARNED_MAX and r["n_zone"] >= CONFLICT_MIN_SNAPS:
                conflicts.append(
                    {
                        **r,
                        "kind": "meta_likes_you_fail",
                        "message": (
                            f"{ZONE_LABELS.get(zone, zone)}: the meta likes {r['formation']} — {r['play']} "
                            f"(prior {r['meta_raw']:+.2f}), but it keeps failing for you "
                            f"(learned {r['learned']:+.2f} over {r['n_zone']} snaps). Your data wins; call it only as a changeup."
                        ),
                        "sources": priors.cite(r["meta_refs"]),
                    }
                )
            elif r["meta_raw"] <= 0.0 and r["learned"] >= SUPPORT_LEARNED_MIN and r["n_zone"] >= CONFLICT_MIN_SNAPS:
                support.append(
                    {
                        **r,
                        "kind": "you_win_meta_cold",
                        "message": (
                            f"{ZONE_LABELS.get(zone, zone)}: {r['formation']} — {r['play']} works for you "
                            f"(learned {r['learned']:+.2f} over {r['n_zone']} snaps) though the meta gives it no boost."
                        ),
                    }
                )
    labs = []
    if allow_experimental(normalize_dynasty(dynasty)):
        for c in priors.lab_candidates:
            labs.append({**c, "sources": priors.cite(c.get("refs") or [])})
    return {"zones": zones_out, "conflicts": conflicts, "support": support, "lab_candidates": labs,
            "live_mode": priors.live_mode, "live_fetched_at": priors.live_fetched_at}


def zone_plan(al: dict[str, Any], *, top: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Top calls per zone with the same call shares the live caller would use."""
    from cfb_coach.learning import REC_EXPLORE, REC_TEMPERATURE

    out: dict[str, list[dict[str, Any]]] = {}
    for zone, rows in (al.get("zones") or {}).items():
        probs = softmax_probs([r["total"] for r in rows], REC_TEMPERATURE)
        u = 1.0 / max(1, len(rows))
        shares = [dict(r, p=round((1 - REC_EXPLORE) * p + REC_EXPLORE * u, 3)) for r, p in zip(rows, probs)]
        shares.sort(key=lambda r: -r["p"])
        out[zone] = shares[:top]
    return out


def build_prep_alignment(db: Any, opponent_id: str, scout: Any | None, *, dynasty: str) -> dict[str, Any]:
    """Everything the prep page needs for the learned-zone + meta alignment panels."""
    from cfb_coach.learning import (
        META_BACKUP_KEY,
        META_REBUILT_AT_KEY,
        META_RULES_KEY,
        RULES_VERSION,
        LearnedWeights,
        constants_table,
        zone_leaderboard,
    )

    lw = LearnedWeights.load(db, opponent_id) if db is not None else LearnedWeights.empty()
    priors = MetaPriors.build(scout)
    al = alignment(lw, priors, dynasty=dynasty)
    info: dict[str, Any] = {"rules_version": RULES_VERSION}
    if db is not None:
        info.update(
            {
                "db_rules_version": db.get_meta(META_RULES_KEY) or "",
                "rebuilt_at": db.get_meta(META_REBUILT_AT_KEY) or "",
                "backup_path": db.get_meta(META_BACKUP_KEY) or "",
            }
        )
    leaders = {z: zone_leaderboard(lw, z, top=4) for z in ("general", RED_ZONE, GOAL_LINE)}
    return {
        "zone_plan": zone_plan(al),
        "conflicts": al["conflicts"],
        "support": al["support"],
        "lab_candidates": al["lab_candidates"],
        "leaders": leaders,
        "learning": info,
        "constants": constants_table(),
        "meta_live_mode": al["live_mode"],
        "meta_live_fetched_at": al["live_fetched_at"],
        "research_updated": (load_research() or {}).get("updated", ""),
    }


__all__ = ["MetaPriors", "alignment", "build_prep_alignment", "combined_score", "live_boosts", "softmax_probs", "zone_plan"]
