"""SHADOW defensive decision: rank the legal (installed) defensive calls by the offensive lift the football priors say they allow against what this
opponent has been running. Never applied to the live call; writes a separate shadow record only.

Inputs known before the snap: the opponent's earlier offensive play NAMES (what the owner typed on defense: ``concept_typed``/``concept_seen``).
Concept evidence from a name counts at NAME_TRUST; with little evidence the distribution shrinks toward a uniform spread over core concepts, so a
thin sample cannot dominate. Calls the catalogue cannot describe (no coverage descriptor) abstain and are listed, never scored.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import football_knowledge as fk
from cfb_coach.madden.model import football_priors as fp

VERSION = "defense_shadow.v1"
CORE = ("mesh", "flood_sail", "smash", "four_verts", "slant_flat", "stick", "screen", "play_action", "inside_zone", "draw")
SHRINK_K = 6.0          # pseudo-snaps of the uniform prior


def offense_distribution(play_names: Sequence[str]) -> dict[str, float]:
    counts: Counter = Counter()
    for n in play_names:
        cid = fk.profile_for_play(n).get("concept_id")
        if cid:
            counts[cid] += fp.NAME_TRUST            # a name is not a diagram
    n = sum(counts.values())
    dist = {c: (counts.get(c, 0.0) + SHRINK_K / len(CORE) * (1 if c in CORE else 0)) / (n + SHRINK_K) for c in set(CORE) | set(counts)}
    tot = sum(dist.values())
    return {c: v / tot for c, v in dist.items()}


def shadow_decision(legal_calls: Sequence[tuple[str, str]], opponent_play_names: Sequence[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    dist = offense_distribution(opponent_play_names)
    rows, abstained = [], []
    for formation, play in legal_calls:
        res = ds.resolve(play, formation)
        d = fp.descriptor(res.key) if res.key else None
        if not d:
            abstained.append(f"{formation} / {play}")
            continue
        e = fp.defense_call_edge(d, dist)
        if not e["cells"]:
            abstained.append(f"{formation} / {play}")      # no prior speaks to this call vs these concepts: unknown, not 'neutral and therefore best'
            continue
        rows.append({"formation": formation, "play": play, "dk_key": res.key, "offense_edge_allowed": round(e["offense_edge_allowed"], 4),
                     "vulnerable_to": [c for c in e["cells"] if c["lift_allowed"] > 0.04][:3], "stops": [c for c in e["cells"] if c["lift_allowed"] < -0.04][:3]})
    rows.sort(key=lambda r: (r["offense_edge_allowed"], r["formation"], r["play"]))
    return {"schema": "madden27.defense_shadow.v1", "version": VERSION, "applied": False, "kind": "shadow_decision_not_an_outcome",
            "ts": datetime.now(timezone.utc).isoformat(), "opponent_concepts": {k: round(v, 3) for k, v in sorted(dist.items(), key=lambda kv: -kv[1])[:5]},
            "evidence_snaps": len(opponent_play_names), "top": rows[:5], "ranked": len(rows), "abstained_calls": len(abstained),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "note": "football-level hypotheses; exact defender assignments for these calls are unverified; hidden opponent adjustments are not modelled"}


def write_shadow(rec: Mapping[str, Any], path: Path | None = None) -> Path:
    from cfb_coach.games import data_dir
    p = Path(path) if path else Path(data_dir()) / "madden_ml_experimental" / "defense_shadow.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return p
