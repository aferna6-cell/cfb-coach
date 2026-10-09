"""Low-data hierarchical / shrinkage model for the experimental pilot.

Not a deep learner. Combines:
- Football family + concept/subfamily priors (mesh/flood/vert/…, not one generic pass)
- Play-family versus coverage-family interaction (train-time post-snap only)
- Down/distance and field-zone effects
- CPU vs human shrinkage + opponent rates
- Soft researched concept priors (capped; never precise fabricated matchups)
- Optional heuristic situational bonuses blended with shrinkage

Predictions use pre-snap information only. A previous-snap coverage hint is soft
evidence that the opponent *might* repeat a look — not confirmation. Historical
post-snap coverage may train matchup tables but is never assumed known before
the snap unless a live pre-snap observation is present.
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
    _PASS_RE,
    _RUN_RE,
    _RPO_RE,
    _SCREEN_RE,
)

MODEL_KIND = "hierarchical_shrinkage.v1"
PRIOR_STRENGTH = 8.0
OPP_PRIOR_STRENGTH = 12.0
FAMILY_PRIOR_STRENGTH = 6.0
CONCEPT_PRIOR_STRENGTH = 5.0
CONTEXT_INTERACTION_STRENGTH = 10.0  # strong shrinkage; small per-play-context sample sizes
# Soft prior when coverage_source is "last" (previous snap) — not confirmation.
LAST_SNAP_COV_WEIGHT = 0.35
# Cap free-text research influence so we never invent precise matchup odds.
MAX_RESEARCH_BOOST = 0.08
HEURISTIC_BLEND = 0.15  # weight on situational heuristic bonus in logit space

_CONCEPT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmesh\b", re.I), "mesh"),
    (re.compile(r"flood|sail|\bbench\b", re.I), "flood"),
    (re.compile(r"four\s*vert|4\s*vert|\bverts?\b|\bseams?\b|vertical", re.I), "vert"),
    (re.compile(r"\bsmash\b", re.I), "smash"),
    (re.compile(r"\bstick\b", re.I), "stick"),
    (re.compile(r"cross(?:er)?s?|\bdrags?\b|\bhi[\s-]*lo\b", re.I), "cross"),
    (re.compile(r"\bpa\b|play[\s-]*action|boot", re.I), "pa"),
    (re.compile(r"\bslants?\b", re.I), "slant"),
    (re.compile(r"\bwheel\b", re.I), "wheel"),
    (re.compile(r"\bscreen\b", re.I), "screen"),
    (re.compile(r"\brpo\b|bubble", re.I), "rpo"),
    (re.compile(r"zone|dive|power|stretch|toss|iso|duo", re.I), "run_concept"),
)


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


def _play_concept(play: str | None) -> str:
    """Finer concept/subfamily than pass/run — used when evidence supports it."""
    raw = play or ""
    for pat, name in _CONCEPT_PATTERNS:
        if pat.search(raw):
            return name
    fam = _play_family(raw)
    return fam if fam != "unknown" else "unknown"


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
    knowledge_origin: str = ""
    data_version: str = ""
    model_version: str = ""
    play_family: dict[str, list[float]] = field(default_factory=dict)
    play_concept: dict[str, list[float]] = field(default_factory=dict)
    play_vs_cov: dict[str, list[float]] = field(default_factory=dict)
    concept_vs_cov: dict[str, list[float]] = field(default_factory=dict)
    down_distance: dict[str, list[float]] = field(default_factory=dict)
    concept_down_distance: dict[str, list[float]] = field(default_factory=dict)
    family_down_distance: dict[str, list[float]] = field(default_factory=dict)
    field_zone: dict[str, list[float]] = field(default_factory=dict)
    cpu_human: dict[str, list[float]] = field(default_factory=dict)
    opponent: dict[str, list[float]] = field(default_factory=dict)
    research_concept_boost: dict[str, float] = field(default_factory=dict)
    research_family_boost: dict[str, float] = field(default_factory=dict)
    evidence_quality: str = "prior_driven"
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
            "knowledge_origin": self.knowledge_origin,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "play_family": self.play_family,
            "play_concept": self.play_concept,
            "play_vs_cov": self.play_vs_cov,
            "concept_vs_cov": self.concept_vs_cov,
            "down_distance": self.down_distance,
            "concept_down_distance": self.concept_down_distance,
            "family_down_distance": self.family_down_distance,
            "field_zone": self.field_zone,
            "cpu_human": self.cpu_human,
            "opponent": self.opponent,
            "research_concept_boost": self.research_concept_boost,
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
            knowledge_origin=str(raw.get("knowledge_origin") or ""),
            data_version=str(raw.get("data_version") or ""),
            model_version=str(raw.get("model_version") or ""),
            play_family={k: list(v) for k, v in (raw.get("play_family") or {}).items()},
            play_concept={k: list(v) for k, v in (raw.get("play_concept") or {}).items()},
            play_vs_cov={k: list(v) for k, v in (raw.get("play_vs_cov") or {}).items()},
            concept_vs_cov={k: list(v) for k, v in (raw.get("concept_vs_cov") or {}).items()},
            down_distance={k: list(v) for k, v in (raw.get("down_distance") or {}).items()},
            concept_down_distance={
                k: list(v) for k, v in (raw.get("concept_down_distance") or {}).items()
            },
            family_down_distance={
                k: list(v) for k, v in (raw.get("family_down_distance") or {}).items()
            },
            field_zone={
                k: list(v)
                for k, v in (raw.get("field_zone") or raw.get("field") or {}).items()
            },
            cpu_human={k: list(v) for k, v in (raw.get("cpu_human") or {}).items()},
            opponent={k: list(v) for k, v in (raw.get("opponent") or {}).items()},
            research_concept_boost={
                k: float(v) for k, v in (raw.get("research_concept_boost") or {}).items()
            },
            research_family_boost={
                k: float(v) for k, v in (raw.get("research_family_boost") or {}).items()
            },
            evidence_quality=str(raw.get("evidence_quality") or "prior_driven"),
            note=str(raw.get("note") or ""),
        )


def _bucket_map(store: dict[str, RateBucket]) -> dict[str, list[float]]:
    return {k: [b.successes, b.trials] for k, b in store.items()}


def _cap_boost(value: float) -> float:
    return max(-MAX_RESEARCH_BOOST, min(MAX_RESEARCH_BOOST, float(value)))


def _load_research_boosts() -> tuple[dict[str, float], dict[str, float], str, str]:
    """Load soft concept/family priors from validated Madden AI research.

    ``ai_research.load_research`` returns ``(document, origin)`` — never treat
    the tuple as a dict. Boosts are capped and concept-attributed so free-text
    mentions do not become fabricated precise matchup probabilities.
    """
    family_boosts: dict[str, float] = {}
    concept_boosts: dict[str, float] = {}
    version = "none"
    origin = ""
    try:
        from cfb_coach import ai_research

        loaded = ai_research.load_research("madden27", offline=True)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            return family_boosts, concept_boosts, version, origin
        doc, origin = loaded
        if not isinstance(doc, dict):
            return family_boosts, concept_boosts, version, origin
        version = str(
            doc.get("researched_at")
            or (doc.get("patch") or {}).get("version")
            or "loaded"
        )
        origin = str(origin or "")
        # meta_offense lines: soft concept hints only.
        for line in doc.get("meta_offense") or []:
            text = str(line)
            concept = _play_concept(text)
            fam = _play_family(text)
            if concept != "unknown":
                concept_boosts[concept] = _cap_boost(concept_boosts.get(concept, 0.0) + 0.02)
            if fam != "unknown":
                family_boosts[fam] = _cap_boost(family_boosts.get(fam, 0.0) + 0.015)
        for finding in doc.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            if finding.get("side") not in ("offense", "general"):
                continue
            conf = str(finding.get("confidence") or "low")
            w = {"high": 0.025, "medium": 0.015, "low": 0.008}.get(conf, 0.008)
            claim = str(finding.get("claim") or "")
            concept = _play_concept(claim)
            fam = _play_family(claim)
            # Attribute only when a recognizable concept/family appears — skip vague prose.
            if concept not in ("unknown", "pass"):
                concept_boosts[concept] = _cap_boost(concept_boosts.get(concept, 0.0) + w)
            elif fam in ("run", "rpo", "screen"):
                family_boosts[fam] = _cap_boost(family_boosts.get(fam, 0.0) + w)
        for suggestion in doc.get("suggestions") or []:
            if not isinstance(suggestion, dict) or suggestion.get("side") != "offense":
                continue
            tip = str(suggestion.get("tip") or "")
            concept = _play_concept(tip)
            if concept not in ("unknown", "pass"):
                concept_boosts[concept] = _cap_boost(concept_boosts.get(concept, 0.0) + 0.01)
    except Exception:  # noqa: BLE001
        return family_boosts, concept_boosts, version, origin
    return family_boosts, concept_boosts, version, origin


def train_experimental(
    rows: Sequence[Mapping[str, Any]],
    *,
    side: str = "offense",
    seed: int = 7,
) -> ExperimentalArtifact:
    """Fit shrinkage buckets from verified rows (+ optional discounted priors)."""
    del seed
    supervised = dataset_mod.supervised_rows(rows)
    supervised = [r for r in supervised if str(r.get("side") or "").startswith(side[:1])]
    from cfb_coach.madden.model.historical import discounted_prior_rows

    discounted = [
        r
        for r in discounted_prior_rows(rows)
        if str(r.get("side") or "").startswith(side[:1])
    ]

    play_fam: dict[str, RateBucket] = {}
    play_concept: dict[str, RateBucket] = {}
    play_vs: dict[str, RateBucket] = {}
    concept_vs: dict[str, RateBucket] = {}
    dd: dict[str, RateBucket] = {}
    concept_dd: dict[str, RateBucket] = {}
    family_dd: dict[str, RateBucket] = {}
    field_z: dict[str, RateBucket] = {}
    ctx: dict[str, RateBucket] = {}
    opp: dict[str, RateBucket] = {}

    def ingest(row: Mapping[str, Any], weight: float) -> None:
        success = str(row.get("success")) == "true"
        action = row.get("action_play") or row.get("executed_play") or row.get("recommended_play")
        fam = _play_family(str(action) if action else None)
        concept = _play_concept(str(action) if action else None)
        # Matchup tables may use historical post-snap coverage for training only.
        cov = _cov_bucket(row.get("coverage_seen") or row.get("coverage_hint"))
        play_fam.setdefault(fam, RateBucket()).add(success, weight)
        play_concept.setdefault(concept, RateBucket()).add(success, weight)
        play_vs.setdefault(f"{fam}|{cov}", RateBucket()).add(success, weight)
        concept_vs.setdefault(f"{concept}|{cov}", RateBucket()).add(success, weight)
        dd_key = _dd_bucket(row.get("down"), row.get("distance"))
        dd.setdefault(dd_key, RateBucket()).add(success, weight)
        if dd_key != "unknown":
            concept_dd.setdefault(f"{concept}|{dd_key}", RateBucket()).add(success, weight)
            family_dd.setdefault(f"{fam}|{dd_key}", RateBucket()).add(success, weight)
        yl = row.get("yardline")
        try:
            yl_i = int(yl) if yl is not None else None
        except (TypeError, ValueError):
            yl_i = None
        field_z.setdefault(_field_bucket(yl_i), RateBucket()).add(success, weight)
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

    family_boosts, concept_boosts, knowledge_version, knowledge_origin = _load_research_boosts()
    if n_sup >= 40:
        quality = "empirical"
    elif n_sup >= 10:
        quality = "empirical_light"
    else:
        quality = "prior_driven"

    version = f"madden-ml.experimental.{side}.{n_sup}.{quality}"
    note = (
        f"Hierarchical shrinkage on {n_sup} supervised + {len(discounted)} discounted "
        f"uncertain-execution rows. Evidence quality: {quality}. "
        f"Knowledge: {knowledge_version} (origin={knowledge_origin or 'none'})."
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
        knowledge_origin=knowledge_origin,
        data_version=f"supervised={n_sup}|discounted={len(discounted)}",
        model_version=version,
        play_family=_bucket_map(play_fam),
        play_concept=_bucket_map(play_concept),
        play_vs_cov=_bucket_map(play_vs),
        concept_vs_cov=_bucket_map(concept_vs),
        down_distance=_bucket_map(dd),
        concept_down_distance=_bucket_map(concept_dd),
        family_down_distance=_bucket_map(family_dd),
        field_zone=_bucket_map(field_z),
        cpu_human=_bucket_map(ctx),
        opponent=_bucket_map(opp),
        research_concept_boost=concept_boosts,
        research_family_boost=family_boosts,
        evidence_quality=quality,
        note=note,
    )


def _rate_from(raw: list[float] | None, prior: float, strength: float) -> float:
    if not raw or len(raw) < 2:
        return prior
    succ, trials = float(raw[0]), float(raw[1])
    return (succ + prior * strength) / (trials + strength)


def _uncertainty(raw: list[float] | None) -> float:
    """Higher when trials are few (1.0 = no data)."""
    if not raw or len(raw) < 2:
        return 1.0
    trials = float(raw[1])
    return 1.0 / (1.0 + trials)


def predict_success(
    artifact: ExperimentalArtifact | Mapping[str, Any],
    *,
    formation: str,
    play: str,
    down: Any = None,
    distance: Any = None,
    yardline: Any = None,
    coverage_hint: str | None = None,
    coverage_source: str | None = None,
    opponent_id: str | None = None,
    opponent_type: str | None = None,
    heuristic_bonus: float = 0.0,
) -> dict[str, Any]:
    """Pre-snap success probability for one candidate.

    ``coverage_source``:
    - ``live``: treat hint as observed pre-snap look
    - ``last``: soft prior that opponent *might* repeat — never confirmation
    - anything else: ignore hint for matchup (unknown coverage)
    """
    del formation  # formation family reserved for future shrinkage layers
    art = (
        artifact
        if isinstance(artifact, ExperimentalArtifact)
        else ExperimentalArtifact.from_dict(artifact)
    )
    fam = _play_family(play)
    concept = _play_concept(play)
    src = (coverage_source or "").strip().lower()
    cov = "unknown"
    cov_weight = 0.0
    if coverage_hint and src == "live":
        cov = _cov_bucket(coverage_hint)
        cov_weight = 1.0
    elif coverage_hint and src == "last":
        cov = _cov_bucket(coverage_hint)
        cov_weight = LAST_SNAP_COV_WEIGHT

    g = art.global_rate
    p_fam = _rate_from(art.play_family.get(fam), g, FAMILY_PRIOR_STRENGTH)
    p_concept = _rate_from(art.play_concept.get(concept), p_fam, CONCEPT_PRIOR_STRENGTH)
    p_match_full = _rate_from(
        art.concept_vs_cov.get(f"{concept}|{cov}") or art.play_vs_cov.get(f"{fam}|{cov}"),
        p_concept,
        FAMILY_PRIOR_STRENGTH,
    )
    # Blend unknown-coverage baseline with soft/full matchup by cov_weight.
    p_match = (1.0 - cov_weight) * p_concept + cov_weight * p_match_full
    dd_key = _dd_bucket(down, distance)
    p_dd = _rate_from(art.down_distance.get(dd_key), g, PRIOR_STRENGTH)
    # A global down/distance factor is identical for every candidate and
    # cannot alter play rankings. Learn the *interaction* of play concept
    # and situation, heavily shrunk to the normal concept prior. Older
    # artifacts without the new buckets remain backward-compatible.
    interaction = (
        art.concept_down_distance.get(f"{concept}|{dd_key}")
        or art.family_down_distance.get(f"{fam}|{dd_key}")
    ) if dd_key != "unknown" else None
    p_situational = (
        _rate_from(interaction, p_match, CONTEXT_INTERACTION_STRENGTH)
        if interaction is not None else p_match
    )
    p_match = 0.55 * p_match + 0.45 * p_situational
    yl = None
    if isinstance(yardline, int):
        yl = yardline
    elif isinstance(yardline, str) and yardline.isdigit():
        yl = int(yardline)
    p_field = _rate_from(art.field_zone.get(_field_bucket(yl)), g, PRIOR_STRENGTH)
    ot = (opponent_type or "unknown").lower()
    p_ctx = _rate_from(art.cpu_human.get(ot), g, PRIOR_STRENGTH)
    p_opp = _rate_from(art.opponent.get(opponent_id or ""), p_ctx, OPP_PRIOR_STRENGTH)

    parts = [p_match, p_dd, p_field, p_opp]
    logit = 0.0
    for p in parts:
        p = min(1 - 1e-4, max(1e-4, p))
        logit += math.log(p / (1 - p))
    logit /= len(parts)

    boost = float(art.research_concept_boost.get(concept) or 0.0)
    if abs(boost) < 1e-9:
        boost = float(art.research_family_boost.get(fam) or 0.0)
    boost = _cap_boost(boost)
    logit += boost
    # Blend existing heuristic situational score (already football-constrained).
    if heuristic_bonus:
        logit += HEURISTIC_BLEND * float(heuristic_bonus)

    # Shrink toward baseline when evidence is thin.
    unc = max(
        _uncertainty(art.play_concept.get(concept)),
        _uncertainty(art.down_distance.get(_dd_bucket(down, distance))),
    )
    if art.evidence_quality == "prior_driven":
        unc = max(unc, 0.7)
    baseline_logit = math.log(max(1e-4, min(1 - 1e-4, g)) / (1 - max(1e-4, min(1 - 1e-4, g))))
    logit = (1.0 - 0.5 * unc) * logit + (0.5 * unc) * baseline_logit

    prob = 1.0 / (1.0 + math.exp(-logit))
    return {
        "probability": prob,
        "play_family": fam,
        "play_concept": concept,
        "coverage_bucket": cov if cov_weight > 0 else "unknown",
        "coverage_weight": cov_weight,
        "uncertainty": round(unc, 3),
        "evidence_quality": art.evidence_quality,
        "components": {
            "concept_matchup": p_match,
            "down_distance": p_dd,
            "concept_situation": p_situational,
            "situation_bucket": dd_key,
            "situation_interaction_trials": float(interaction[1]) if interaction else 0.0,
            "field": p_field,
            "opponent": p_opp,
            "research_boost": boost,
            "heuristic_bonus": float(heuristic_bonus or 0.0),
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
    coverage_source: str | None = None,
    opponent_id: str | None = None,
    opponent_type: str | None = None,
    heuristic: tuple[str, str] | None = None,
    heuristic_bonuses: Mapping[tuple[str, str], float] | None = None,
    near_tie_margin: float = 0.03,
) -> list[dict[str, Any]]:
    """Rank football-eligible (formation, play) candidates.

    Near-ties prefer the heuristic pick. Sort is by probability then concept
    diversity — never alphabetical first play-family as a hidden preference.
    """
    scored: list[dict[str, Any]] = []
    bonuses = heuristic_bonuses or {}
    for form, play in candidates:
        pred = predict_success(
            artifact,
            formation=form,
            play=play,
            down=down,
            distance=distance,
            yardline=yardline,
            coverage_hint=coverage_hint,
            coverage_source=coverage_source,
            opponent_id=opponent_id,
            opponent_type=opponent_type,
            heuristic_bonus=float(bonuses.get((form, play), 0.0)),
        )
        scored.append(
            {
                "formation": form,
                "play": play,
                "probability": pred["probability"],
                "play_family": pred["play_family"],
                "play_concept": pred["play_concept"],
                "uncertainty": pred["uncertainty"],
                "evidence_quality": pred["evidence_quality"],
                "components": pred["components"],
                "coverage_weight": pred["coverage_weight"],
            }
        )
    # Primary: probability. Tie-break: lower uncertainty, then stable formation/play.
    scored.sort(
        key=lambda r: (
            -r["probability"],
            r.get("uncertainty", 1.0),
            r["formation"],
            r["play"],
        )
    )
    if heuristic and scored:
        top = scored[0]["probability"]
        for row in scored:
            if (row["formation"], row["play"]) == heuristic and (
                top - row["probability"]
            ) <= near_tie_margin:
                scored.remove(row)
                scored.insert(0, row)
                row["near_tie_kept_heuristic"] = True
                break
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
    return scored
