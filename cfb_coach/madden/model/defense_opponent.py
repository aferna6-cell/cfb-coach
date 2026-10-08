"""Opponent offensive tendency model for the Stage 3 defense shadow advisor.

Uses only pre-snap / previously observed information. Evidence counts and
uncertainty are explicit: a single snap never establishes a strong tendency.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from cfb_coach.madden.defense_select import COUNTERS
from cfb_coach.madden.situation import concept_family
from cfb_coach.opponents import is_cpu_opponent

# Concept families we track for opponent offense.
OFFENSE_FAMILIES = (
    "cross",
    "flood",
    "vert",
    "screen",
    "run",
    "inside_zone",
    "outside_zone",
    "power",
    "rpo",
    "scram",
)

# Soft Dirichlet prior strength — shrink toward uniform until n is large.
TENDENCY_PRIOR = 4.0
# Minimum observations before a family rate is "established".
ESTABLISHED_N = 4
# Soft weight when concept_source is "last" (previous snap), not live confirmation.
LAST_SNAP_CONCEPT_WEIGHT = 0.35


def refine_run_family(concept: str | None) -> str | None:
    """Map a concept string to a finer run/RPO/scram family when possible."""
    c = (concept or "").lower()
    if not c:
        return None
    if "screen" in c:
        return "screen"
    if "scram" in c or "qb run" in c or "qb draw" in c:
        return "scram"
    if "rpo" in c or "bubble" in c:
        return "rpo"
    if "power" in c or "iso" in c or "duo" in c:
        return "power"
    if "stretch" in c or "outside zone" in c or "toss" in c or "oz" in c:
        return "outside_zone"
    if "inside zone" in c or "hb zone" in c or "dive" in c or "mid zone" in c:
        return "inside_zone"
    fam = concept_family(concept)
    if fam == "run":
        return "run"
    return fam


@dataclass
class FamilyEvidence:
    """Counted observations with an explicit uncertainty score in ``[0, 1]``."""

    count: int = 0
    success_vs_us: int = 0  # their offensive success when we were on D

    @property
    def uncertainty(self) -> float:
        # 1.0 with zero evidence → ~0.2 at ESTABLISHED_N*2
        return 1.0 / (1.0 + self.count / float(ESTABLISHED_N))

    def rate(self, prior: float = 0.0, strength: float = TENDENCY_PRIOR) -> float:
        return (self.count + prior * strength) / (self.count + strength)


@dataclass
class OpponentOffenseModel:
    """Shrinkage tendency model for one opponent's offense vs us."""

    opponent_id: str
    opponent_kind: str = "unknown"  # cpu | human | unknown
    n_defense_snaps: int = 0
    n_games: int = 0
    families: dict[str, FamilyEvidence] = field(default_factory=dict)
    situational: dict[str, dict[str, FamilyEvidence]] = field(default_factory=dict)
    formation_hints: Counter = field(default_factory=Counter)
    repeated_concepts: Counter = field(default_factory=Counter)
    evidence_quality: str = "prior_driven"
    note: str = ""

    def family_share(self, fam: str, *, bucket: str | None = None) -> tuple[float, float, int]:
        """Return ``(share, uncertainty, count)`` for a family.

        When ``bucket`` is set and has enough evidence, use the situational
        table; otherwise fall back to overall with higher uncertainty.
        """
        if bucket and bucket in self.situational:
            table = self.situational[bucket]
            total = sum(e.count for e in table.values())
            if total >= ESTABLISHED_N:
                ev = table.get(fam) or FamilyEvidence()
                share = ev.count / total if total else 0.0
                return share, ev.uncertainty, ev.count
        ev = self.families.get(fam) or FamilyEvidence()
        total = sum(e.count for e in self.families.values()) or 1
        # Shrink toward uniform over tracked families.
        prior = 1.0 / max(1, len(OFFENSE_FAMILIES))
        share = (ev.count + prior * TENDENCY_PRIOR) / (total + TENDENCY_PRIOR)
        return share, ev.uncertainty, ev.count

    def expected_counters(self, *, bucket: str | None = None) -> dict[str, float]:
        """Coverage-family weights answering expected offense concepts."""
        out: dict[str, float] = {}
        for fam in OFFENSE_FAMILIES:
            share, unc, n = self.family_share(fam, bucket=bucket)
            if n < 1 and share < 0.08:
                continue
            # Soften influence by uncertainty; one-off never dominates.
            weight = share * (1.0 - 0.7 * unc)
            base = fam if fam in COUNTERS else (
                "run" if fam in ("inside_zone", "outside_zone", "power") else fam
            )
            for cov, w in (COUNTERS.get(base) or {}).items():
                out[cov] = out.get(cov, 0.0) + weight * w
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "opponent_kind": self.opponent_kind,
            "n_defense_snaps": self.n_defense_snaps,
            "n_games": self.n_games,
            "evidence_quality": self.evidence_quality,
            "families": {
                k: {"count": v.count, "success_vs_us": v.success_vs_us, "uncertainty": round(v.uncertainty, 3)}
                for k, v in self.families.items()
            },
            "repeated_concepts": dict(self.repeated_concepts.most_common(12)),
            "formation_hints": dict(self.formation_hints.most_common(8)),
            "note": self.note,
        }


def _down_bucket(down: Any, distance: Any, yardline: Any) -> str:
    try:
        from cfb_coach.scouting import down_bucket

        return down_bucket(down, distance, yardline)
    except Exception:  # noqa: BLE001
        return "early"


def build_opponent_offense_model(
    db: Any,
    opponent_id: str,
    *,
    limit: int = 400,
) -> OpponentOffenseModel:
    """Fit tendency buckets from logged defense-side snaps vs this opponent."""
    kind = "cpu" if is_cpu_opponent(opponent_id) else "human"
    model = OpponentOffenseModel(opponent_id=opponent_id, opponent_kind=kind)
    if db is None or not opponent_id:
        model.note = "no db — prior-driven only"
        return model

    try:
        snaps = list(db.get_recent_snaps(opponent_id, side="defense", limit=limit))
    except Exception:  # noqa: BLE001
        snaps = []
    if not snaps:
        # Tendencies table may still have concept counts without snap detail.
        try:
            from cfb_coach.madden.macro_select import tendency_families

            for fam, n in tendency_families(db, opponent_id).items():
                refined = refine_run_family(fam) or fam
                model.families.setdefault(refined, FamilyEvidence()).count += int(n)
            model.n_defense_snaps = sum(e.count for e in model.families.values())
        except Exception:  # noqa: BLE001
            pass
        model.evidence_quality = (
            "empirical_light" if model.n_defense_snaps >= ESTABLISHED_N else "prior_driven"
        )
        model.note = "tendency table only" if model.n_defense_snaps else "no defense snaps"
        return model

    games = {getattr(s, "session_id", None) or (s["session_id"] if hasattr(s, "keys") else None) for s in snaps}
    model.n_games = len({g for g in games if g})
    model.n_defense_snaps = len(snaps)

    from cfb_coach.outcome import outcome_success

    for s in snaps:
        def _g(key: str, default: Any = None) -> Any:
            if hasattr(s, "keys"):
                return s[key] if key in s.keys() else default
            return getattr(s, key, default)

        concept = (_g("concept_seen") or "").strip()
        fam = refine_run_family(concept) or concept_family(concept)
        if not fam:
            continue
        bucket = _down_bucket(_g("down"), _g("distance"), _g("yardline"))
        ev = model.families.setdefault(fam, FamilyEvidence())
        ev.count += 1
        if outcome_success(_g("result"), "offense"):
            ev.success_vs_us += 1
        sev = model.situational.setdefault(bucket, {}).setdefault(fam, FamilyEvidence())
        sev.count += 1
        if concept:
            model.repeated_concepts[concept] += 1
        form = (_g("their_formation") or _g("offense_formation") or "").strip()
        if form:
            model.formation_hints[form] += 1

    if model.n_defense_snaps >= 20 and any(e.count >= ESTABLISHED_N for e in model.families.values()):
        model.evidence_quality = "empirical"
    elif model.n_defense_snaps >= ESTABLISHED_N:
        model.evidence_quality = "empirical_light"
    else:
        model.evidence_quality = "prior_driven"
        model.note = "few defense snaps — heavy shrinkage"
    return model


def situation_concept_prior(
    sit: Any,
    model: OpponentOffenseModel,
) -> dict[str, float]:
    """Pre-snap concept weights from live/last hints + tendency model.

    Live pre-snap observation is confirmation-weight 1.0. A previous-snap
    concept is soft evidence only (``LAST_SNAP_CONCEPT_WEIGHT``).
    """
    weights: dict[str, float] = {}
    # Baseline from model.
    bucket = _down_bucket(
        getattr(sit, "down", None),
        getattr(sit, "distance", None),
        getattr(sit, "yardline", None),
    )
    for fam in OFFENSE_FAMILIES:
        share, unc, n = model.family_share(fam, bucket=bucket)
        if n or share > 0.05:
            weights[fam] = share * (1.0 - 0.5 * unc)

    hint = getattr(sit, "concept_hint", None)
    src = (getattr(sit, "concept_source", None) or "").lower()
    fam = refine_run_family(hint) or concept_family(hint)
    if fam:
        w = 1.0 if src == "live" else (LAST_SNAP_CONCEPT_WEIGHT if src == "last" else 0.0)
        if w > 0:
            weights[fam] = weights.get(fam, 0.0) + w
    # Normalize lightly.
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def is_repeated_concept(model: OpponentOffenseModel, concept: str | None, *, min_n: int = 2) -> bool:
    """True only when the same concept was seen enough times (not a one-off)."""
    if not concept:
        return False
    return int(model.repeated_concepts.get(concept, 0)) >= min_n
