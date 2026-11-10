"""Expert tags: weak situation prior + a STRONG, belief-gated coverage-beater term.

Coverage beater (Aidan, 2026-11-10: "if we are seeing a certain type of coverage over and over the model
should go to its beater"): each variant's "beats Cover X / Man" tags (catalog + Civil.GG sheet) add
    beater = BEATER_MAX * sum_T gate_T * tagged_T,   gate_T = info * clip((share_T - GATE_LO) / (GATE_HI - GATE_LO), 0, 1)
to the success logit, where share_T is the pre-snap belief that the next coverage is in tag group T and
info is ``Belief.gate_confidence`` (0 at one observation, 1 at four or more effective observations:
same-game repetitions + recency-weighted history). One coverage seen
over and over -> share_T and info high -> tagged beaters get up to +BEATER_MAX logit and normally rise to
the top. Mixed or thin belief -> gate ~0 -> the tag barely matters. It is a term inside the model score
(the down/distance, field, sack/turnover heads and the legal filter still apply), not an if-then rule.
Untagged variants (all Gun 5WR Tight) get the AVERAGE beater term of the tagged variants: unknown, not negative.


Bounded logit bonus, never a filter or a forced pick, and it fades as verified variant evidence arrives:
    bonus = decay * (SIT_W * situation_match + COV_W * coverage_match + CORE_W * core_go_to),  |bonus| <= CAP
    decay = K / (K + verified observations of this variant)
Untagged coverage (all Gun 5WR Tight variants) is imputed with the average match of the tagged
variants at the same belief, so missing tags are unknown, never negative.
Guide situation labels are qualitative; the repo's existing down-and-distance buckets
(``experimental_model._dd_bucket``) are reused rather than inventing thresholds. "Bomb Plays" is not
mapped to any situation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.scheme.catalog import SchemeVariant, repo_root

SIT_W, COV_W, CORE_W, CAP, K = 0.04, 0.06, 0.03, 0.12, 10.0
BEATER_MAX = 0.9          # logit; P(success) ~0.45 -> ~0.67 at full gate
GATE_LO, GATE_HI = 0.35, 0.75
BEATER_K = 40.0           # beater term fades only slowly as verified variant evidence accumulates
TAG_GROUPS = ("Cover 0", "Cover 1", "Cover 2", "Cover 3", "Cover 4", "Man Coverage")
SHEET_RELPATH = Path("datasets") / "madden27" / "civil_bucs_offense_sheet_expert_tags.json"


@lru_cache(maxsize=4)
def sheet_also_beats(path: str | None = None) -> dict[str, tuple[str, ...]]:
    p = Path(path) if path else repo_root() / SHEET_RELPATH
    if not p.is_file():
        return {}
    data = json.loads(p.read_text())
    return {e["catalog_variant_id"]: tuple(e.get("also_beats") or ()) for e in data.get("also_beats") or []
            if e.get("catalog_variant_id")}


def coverage_tags(v: SchemeVariant) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tuple(v.expert_coverage_tags) + sheet_also_beats().get(v.variant_id, ())))


def _sit_match(tags: Sequence[str], down: Any, distance: Any, yardline: Any) -> float:
    dd = em._dd_bucket(down, distance)
    try:
        yl = int(yardline) if yardline not in (None, "") else None
    except (TypeError, ValueError):
        yl = None
    try:
        d, dist = int(down), int(distance)
    except (TypeError, ValueError):
        d = dist = None
    hit = False
    for t in tags:
        if t == "1st And 10" and d == 1 and dist == 10:
            hit = True
        elif t == "Short Yardage" and dd.endswith("_short"):
            hit = True
        elif t == "Medium Yardage" and dd.endswith("_med"):
            hit = True
        elif t == "Long Yardage" and dd.endswith("_long"):
            hit = True
        elif t == "Inside Own 10" and yl is not None and yl <= 10:
            hit = True
        elif t == "Redzone 10 20" and yl is not None and 80 <= yl <= 90:
            hit = True
    return 1.0 if hit else 0.0


def _tag_share(tag: str, mix: Mapping[str, float], families: Mapping[str, float]) -> float:
    if tag == "Man Coverage":
        return families.get("man", 0.0)
    pref = {"Cover 0": ("cover_0",), "Cover 1": ("cover_1",), "Cover 2": ("cover_2", "tampa_2"),
            "Cover 3": ("cover_3",), "Cover 4": ("cover_4",)}.get(tag)
    if not pref:
        return 0.0
    return sum(p for k, p in mix.items() if k.startswith(pref))


def tag_shares(mix: Mapping[str, float], families: Mapping[str, float]) -> dict[str, float]:
    return {t: _tag_share(t, mix, families) for t in TAG_GROUPS}


def gates(mix: Mapping[str, float], families: Mapping[str, float], info: float) -> dict[str, float]:
    sh = tag_shares(mix, families)
    return {t: max(0.0, min(1.0, info)) * max(0.0, min(1.0, (v - GATE_LO) / (GATE_HI - GATE_LO))) for t, v in sh.items()}


def coverage_match(tags: Sequence[str], mix: Mapping[str, float], families: Mapping[str, float]) -> float | None:
    if not tags:
        return None
    return min(1.0, sum(_tag_share(t, mix, families) for t in tags))


def bonuses(variants: Sequence[SchemeVariant], *, down: Any, distance: Any, yardline: Any,
            mix: Mapping[str, float], families: Mapping[str, float], info: float = 0.0,
            verified_counts: Mapping[str, float] | None = None) -> dict[str, dict[str, Any]]:
    covs = {v.variant_id: coverage_match(coverage_tags(v), mix, families) for v in variants}
    known = [c for c in covs.values() if c is not None]
    impute = sum(known) / len(known) if known else 0.0
    g = gates(mix, families, info)
    raw_beat = {}
    for v in variants:
        tags = coverage_tags(v)
        raw_beat[v.variant_id] = (BEATER_MAX * min(1.0, sum(g.get(t, 0.0) for t in tags))) if tags else None
    kb = [b for b in raw_beat.values() if b is not None]
    beat_impute = sum(kb) / len(kb) if kb else 0.0
    out = {}
    for v in variants:
        n = float((verified_counts or {}).get(v.variant_id, 0.0))
        decay = K / (K + n)
        sit = _sit_match(v.expert_situation_tags, down, distance, yardline)
        cov = covs[v.variant_id]
        cov_used = impute if cov is None else cov
        core = 1.0 if v.role == "expert_core_go_to" else 0.0
        b = max(-CAP, min(CAP, decay * (SIT_W * sit + COV_W * cov_used + CORE_W * core)))
        rb = raw_beat[v.variant_id]
        beat = (beat_impute if rb is None else rb) * (BEATER_K / (BEATER_K + n))
        top = max(g.items(), key=lambda kv: kv[1]) if g else ("", 0.0)
        out[v.variant_id] = {"bonus": b, "beater": beat, "situation_match": sit, "coverage_match": cov_used,
                             "coverage_tags_known": cov is not None, "core": core, "decay": decay,
                             "beats": list(coverage_tags(v)), "gate_top": [top[0], round(top[1], 3)],
                             "tagged_beater_for_top": bool(top[1] > 0 and top[0] in coverage_tags(v))}
    return out
