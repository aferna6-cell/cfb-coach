"""Low-data hierarchical / shrinkage model for the experimental pilot.

Not a deep learner. Combines:
- Football family priors (pass/run/rpo/screen × coverage families)
- Down/distance and field-zone effects
- CPU vs human shrinkage
- Opponent-specific rates with strong prior toward league mean
- Research meta priors (discounted)

Predictions use pre-snap information only. Historical post-snap coverage may
train matchup tables but is never assumed known at decision time unless the
situation carries a live coverage hint.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model.features import (
    _coverage_family_flags,
    _formation_family_flags,
    _PASS_RE,
    _RUN_RE,
    _RPO_RE,
    _SCREEN_RE,
)

MODEL_KIND = "hierarchical_shrinkage.v1"
PRIOR_STRENGTH = 8.0  # pseudo-counts toward global rate
OPP_PRIOR_STRENGTH = 12.0
FAMILY_PRIOR_STRENGTH = 6.0


def _play_family(play: str | None) -> str:
    raw = play or ""
    if _SCREEN_RE.search(raw):
        return "screen"
    if _RPO_RE.search(raw):
        return "rpo"
    if _RUN_RE.search(raw) and not _PASS_RE.search(raw):
        return "run"
    if _PASS_RE.search(raw):
        return "pass"
    return "unknown"


def _cov_bucket(text: str | None) -> str:
    flags = _coverage_family_flags(text)
    for name, val in flags.items():
        if val >= 1.0 and name != "cov_fam_unknown":
            return name.replace("cov_fam_", "")
    return "unknown"


def _field_bucket(yardline: int | None) -> str:
    if yardline is None:
        return "unknown"
    yl = int(yardline)
    if yl >= 80:
        return "red_zone"
    if yl < 40:
        return "own"
    return "mid"


def _dd_bucket(down: Any, distance: Any) -> str:
    try:
        d = int(down) if down is not None else None
        dist = int(distance) if distance is not None else None
    except (TypeError, ValueError):
        return "unknown"
    if d is None or dist is None:
        return "unknown"
    if dist <= 3:
        return f"{d}_short"
    if dist >= 8:
        return f"{d}_long"
    return f"{d}_med"


@dataclass
class RateBucket:
    successes: float = 0.0
    trials: float = 0.0

    def add(self, success: bool, weight: float = 1.0) -> None:
        self.trials += weight
        if success:
            self.successes += weight

    def mean(self, prior: float, strength: float) -> float:
        return (self.successes + prior * strength) / (self.trials + strength)


@dataclass
class ExperimentalArtifact:
    kind: str = MODEL_KIND
    trained_at: str = ""
    side: str = "offense"
    global_rate: float = 0.5
    n_supervised: int = 0
    n_discounted_priors: int = 0
    knowledge_version: str = ""
    data_version: str = ""
    model_version: str = ""
    play_family: dict[str, list[float]] = field(default_factory=dict)
    play_vs_cov: dict[str, list[float]] = field(default_factory=dict)
    down_distance: dict[str, list[float]] = field(default_factory=dict)
    field_zone: dict[str, list[float]] = field(default_factory=dict)
    cpu_human: dict[str, list[float]] = field(default_factory=dict)
    opponent: dict[str, list[float]] = field(default_factory=dict)
    research_family_boost: dict[str, float] = field(default_factory=dict)
    evidence_quality: str = "prior_driven"  # or "empirical_light" / "empirical"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "trained_at": self.trained_at,
            "side": self.side,
            "global_rate": self.global_rate,
            "n_supervised": self.n_supervised,
            "n_discounted_priors": self.n_discounted_priors,
            "knowledge_version": self.knowledge_version,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "play_family": self.play_family,
            "play_vs_cov": self.play_vs_cov,
            "down_distance": self.down_distance,
            "field_zone": self.field_zone,
            "cpu_human": self.cpu_human,
            "opponent": self.opponent,
            "research_family_boost": self.research_family_boost,
            "evidence_quality": self.evidence_quality,
            "note": self.note,
            "baseline_rate": self.global_rate,
            "model_type": self.kind,
            "feature_schema_version": "madden-ml.features.2",
            "train_groups": [],
            "all_groups": [],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentalArtifact":
        return cls(
            kind=str(raw.get("kind") or MODEL_KIND),
            trained_at=str(raw.get("trained_at") or ""),
            side=str(raw.get("side") or "offense"),
            global_rate=float(raw.get("global_rate") or 0.5),
            n_supervised=int(raw.get("n_supervised") or 0),
            n_discounted_priors=int(raw.get("n_discounted_priors") or 0),
            knowledge_version=str(raw.get("knowledge_version") or ""),
            data_version=str(raw.get("data_version") or ""),
            model_version=str(raw.get("model_version") or ""),
            play_family={k: list(v) for k, v in (raw.get("play_family") or {}).items()},
            play_vs_cov={k: list(v) for k, v in (raw.get("play_vs_cov") or {}).items()},
            down_distance={k: list(v) for k, v in (raw.get("down_distance") or {}).items()},
            field_zone={
                k: list(v)
                for k, v in (raw.get("field_zone") or raw.get("field") or {}).items()
            },
            cpu_human={k: list(v) for k, v in (raw.get("cpu_human") or {}).items()},
            opponent={k: list(v) for k, v in (raw.get("opponent") or {}).items()},
            research_family_boost={
                k: float(v) for k, v in (raw.get("research_family_boost") or {}).items()
            },
            evidence_quality=str(raw.get("evidence_quality") or "prior_driven"),
            note=str(raw.get("note") or ""),
        )


def _bucket_map(store: dict[str, RateBucket]) -> dict[str, list[float]]:
    return {k: [b.successes, b.trials] for k, b in store.items()}


def _load_research_boosts() -> tuple[dict[str, float], str]:
    boosts: dict[str, float] = {}
    version = "none"
    try:
        from cfb_coach import ai_research

        doc = ai_research.load_research("madden27")
        if not doc:
            return boosts, version
        version = str(doc.get("researched_at") or doc.get("patch", {}).get("version") or "loaded")
        # Light family priors from meta_offense strings — not fabricated CA settings.
        for line in doc.get("meta_offense") or []:
            fam = _play_family(str(line))
            boosts[fam] = boosts.get(fam, 0.0) + 0.03
        for finding in doc.get("findings") or []:
            if finding.get("side") != "offense":
                continue
            conf = str(finding.get("confidence") or "low")
            w = {"high": 0.04, "medium": 0.02, "low": 0.01}.get(conf, 0.01)
            fam = _play_family(str(finding.get("claim") or ""))
            boosts[fam] = boosts.get(fam, 0.0) + w
    except Exception:  # noqa: BLE001
        return boosts, version
    return boosts, version


def train_experimental(
    rows: Sequence[Mapping[str, Any]],
    *,
    side: str = "offense",
    seed: int = 7,
) -> ExperimentalArtifact:
    """Fit shrinkage buckets from verified rows (+ optional discounted priors)."""
    del seed  # deterministic given row order
    supervised = dataset_mod.supervised_rows(rows)
    supervised = [
        r for r in supervised if str(r.get("side") or "").startswith(side[:1])
    ]
    from cfb_coach.madden.model.historical import discounted_prior_rows

    discounted = [
        r
        for r in discounted_prior_rows(rows)
        if str(r.get("side") or "").startswith(side[:1])
    ]

    play_fam: dict[str, RateBucket] = {}
    play_vs: dict[str, RateBucket] = {}
    dd: dict[str, RateBucket] = {}
    field: dict[str, RateBucket] = {}
    ctx: dict[str, RateBucket] = {}
    opp: dict[str, RateBucket] = {}

    def ingest(row: Mapping[str, Any], weight: float) -> None:
        success = str(row.get("success")) == "true"
        action = row.get("action_play") or row.get("executed_play") or row.get("recommended_play")
        fam = _play_family(str(action) if action else None)
        # Matchup table may use historical post-snap coverage for training only.
        cov = _cov_bucket(row.get("coverage_seen") or row.get("coverage_hint"))
        play_fam.setdefault(fam, RateBucket()).add(success, weight)
        play_vs.setdefault(f"{fam}|{cov}", RateBucket()).add(success, weight)
        dd.setdefault(_dd_bucket(row.get("down"), row.get("distance")), RateBucket()).add(
            success, weight
        )
        field.setdefault(_field_bucket(row.get("yardline")), RateBucket()).add(success, weight)
        ot = str(row.get("opponent_type") or "unknown").lower()
        ctx.setdefault(ot if ot in ("cpu", "human") else "unknown", RateBucket()).add(
            success, weight
        )
        oid = str(row.get("opponent_id") or "unknown")
        opp.setdefault(oid, RateBucket()).add(success, weight)

    for row in supervised:
        ingest(row, 1.0)
    for row in discounted:
        ingest(row, float(row.get("prior_discount") or 0.25))

    n_sup = len(supervised)
    total_s = sum(1 for r in supervised if str(r.get("success")) == "true")
    global_rate = (total_s / n_sup) if n_sup else 0.5

    boosts, knowledge_version = _load_research_boosts()
    if n_sup >= 40:
        quality = "empirical"
    elif n_sup >= 10:
        quality = "empirical_light"
    else:
        quality = "prior_driven"

    version = f"madden-ml.experimental.{side}.{n_sup}.{quality}"
    note = (
        f"Hierarchical shrinkage on {n_sup} supervised + {len(discounted)} discounted "
        f"uncertain-execution rows. Evidence quality: {quality}."
    )
    if quality == "prior_driven":
        note += " Insufficient verified executions — live selections will be labeled prior-driven."

    return ExperimentalArtifact(
        trained_at=datetime.now(timezone.utc).isoformat(),
        side=side,
        global_rate=global_rate,
        n_supervised=n_sup,
        n_discounted_priors=len(discounted),
        knowledge_version=knowledge_version,
        data_version=f"supervised={n_sup}|discounted={len(discounted)}",
        model_version=version,
        play_family=_bucket_map(play_fam),
        play_vs_cov=_bucket_map(play_vs),
        down_distance=_bucket_map(dd),
        field_zone=_bucket_map(field),
        cpu_human=_bucket_map(ctx),
        opponent=_bucket_map(opp),
        research_family_boost=boosts,
        evidence_quality=quality,
        note=note,
    )


def _rate_from(raw: list[float] | None, prior: float, strength: float) -> float:
    if not raw or len(raw) < 2:
        return prior
    succ, trials = float(raw[0]), float(raw[1])
    return (succ + prior * strength) / (trials + strength)


def predict_success(
    artifact: ExperimentalArtifact | Mapping[str, Any],
    *,
    formation: str,
    play: str,
    down: Any = None,
    distance: Any = None,
    yardline: Any = None,
    coverage_hint: str | None = None,
    opponent_id: str | None = None,
    opponent_type: str | None = None,
) -> dict[str, Any]:
    """Pre-snap success probability for one candidate."""
    art = (
        artifact
        if isinstance(artifact, ExperimentalArtifact)
        else ExperimentalArtifact.from_dict(artifact)
    )
    fam = _play_family(play)
    # Only use coverage when it is a live/pre-snap hint — caller responsibility.
    cov = _cov_bucket(coverage_hint) if coverage_hint else "unknown"
    g = art.global_rate
    p_fam = _rate_from(art.play_family.get(fam), g, FAMILY_PRIOR_STRENGTH)
    p_match = _rate_from(art.play_vs_cov.get(f"{fam}|{cov}"), p_fam, FAMILY_PRIOR_STRENGTH)
    p_dd = _rate_from(art.down_distance.get(_dd_bucket(down, distance)), g, PRIOR_STRENGTH)
    yl = None
    if isinstance(yardline, int):
        yl = yardline
    elif isinstance(yardline, str) and yardline.isdigit():
        yl = int(yardline)
    p_field = _rate_from(art.field_zone.get(_field_bucket(yl)), g, PRIOR_STRENGTH)
    ot = (opponent_type or "unknown").lower()
    p_ctx = _rate_from(art.cpu_human.get(ot), g, PRIOR_STRENGTH)
    p_opp = _rate_from(
        art.opponent.get(opponent_id or ""),
        p_ctx,
        OPP_PRIOR_STRENGTH,
    )
    # Logit average of shrunk components (stable for small n).
    parts = [p_match, p_dd, p_field, p_opp]
    logit = 0.0
    for p in parts:
        p = min(1 - 1e-4, max(1e-4, p))
        logit += math.log(p / (1 - p))
    logit /= len(parts)
    boost = float(art.research_family_boost.get(fam) or 0.0)
    logit += boost
    prob = 1.0 / (1.0 + math.exp(-logit))
    return {
        "probability": prob,
        "play_family": fam,
        "coverage_bucket": cov,
        "evidence_quality": art.evidence_quality,
        "components": {
            "family_matchup": p_match,
            "down_distance": p_dd,
            "field": p_field,
            "opponent": p_opp,
            "research_boost": boost,
        },
    }


def save_artifact(artifact: ExperimentalArtifact, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_artifact(path: str | Path) -> ExperimentalArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experimental artifact must be a JSON object")
    return ExperimentalArtifact.from_dict(raw)


def rank_candidates(
    artifact: ExperimentalArtifact | Mapping[str, Any],
    candidates: Sequence[tuple[str, str]],
    *,
    down: Any = None,
    distance: Any = None,
    yardline: Any = None,
    coverage_hint: str | None = None,
    opponent_id: str | None = None,
    opponent_type: str | None = None,
    heuristic: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Rank (formation, play) candidates. Near-ties prefer the heuristic pick."""
    scored: list[dict[str, Any]] = []
    for form, play in candidates:
        pred = predict_success(
            artifact,
            formation=form,
            play=play,
            down=down,
            distance=distance,
            yardline=yardline,
            coverage_hint=coverage_hint,
            opponent_id=opponent_id,
            opponent_type=opponent_type,
        )
        scored.append(
            {
                "formation": form,
                "play": play,
                "probability": pred["probability"],
                "play_family": pred["play_family"],
                "evidence_quality": pred["evidence_quality"],
                "components": pred["components"],
            }
        )
    scored.sort(key=lambda r: (-r["probability"], r["formation"], r["play"]))
    if heuristic and scored:
        top = scored[0]["probability"]
        # Constrained near-tie: if heuristic is within 0.03 of best, keep heuristic.
        for row in scored:
            if (row["formation"], row["play"]) == heuristic and (top - row["probability"]) <= 0.03:
                scored.remove(row)
                scored.insert(0, row)
                row["near_tie_kept_heuristic"] = True
                break
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
    return scored
