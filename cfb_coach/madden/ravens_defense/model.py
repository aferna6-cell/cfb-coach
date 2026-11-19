"""Versioned Ravens defensive coordinator artifacts: train, save, load. Never promotes or enables."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import opponent as op
from cfb_coach.madden.ravens_defense import outcome as oc
from cfb_coach.madden.ravens_defense.concepts import CLASSES

SCHEMA = "madden27.ravens_defense.coordinator.v1"
MODEL_NAME = "Lions Ravens two-formation defensive ML coordinator"


def models_root() -> Path:
    return Path.home() / ".cfb-coach" / "models" / "ravens_defense"


class CoordinatorModel:
    def __init__(self, meta: Mapping[str, Any], cat: rc.RavensCatalog, outcome: oc.OutcomeModel,
                 opponent: Mapping[str, Any]):
        self.meta, self.cat, self.outcome, self.opponent = dict(meta), cat, outcome, dict(opponent)
        self.history = op.History.from_json(opponent["history"])

    @property
    def belief_method(self) -> str:
        return str(self.opponent.get("selected_live_belief") or "v2_dirichlet")

    def to_json(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "meta": self.meta, "outcome": self.outcome.to_json(), "opponent": self.opponent}


def train(snaps: Sequence[Any], cat: rc.RavensCatalog, *, belief_eval: Mapping[str, Any] | None,
          source: Mapping[str, Any], success_eval: Mapping[str, Any] | None = None) -> CoordinatorModel:
    observed = [s for s in snaps if s.closed]
    verified = [s for s in snaps if s.execution == "verified_ravens_execution"]
    base = oc.fit_base(observed)
    base["success_base_mode"] = (success_eval or {}).get("selected_success_base", "v2_context")
    terms = oc.fit_candidate_terms(cat, verified, base)
    opp = op.fit([s for s in snaps if s.concept is not None])
    opp["selected_live_belief"] = (belief_eval or {}).get("selected_live_belief", "v2_dirichlet")
    cutoff = max((s.started_ts for s in snaps), default=None)
    meta = {
        "model_name": MODEL_NAME, "schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architecture": {"outcome": oc.ARCH, "opponent_belief": opp["selected_live_belief"],
                         "success_base": base["success_base_mode"],
                         "opponent_challengers": list(op.METHODS),
                         "decision": "argmin expected defensive loss over legal complete candidates, "
                                     "integrated over the pre-snap opponent concept belief"},
        "training_config": {"outcome_prior": oc.PRIOR, "utility": oc.UTILITY,
                            "optional_macro_cost": oc.OPTIONAL_MACRO_COST,
                            "min_verified_for_trained": oc.MIN_VERIFIED_FOR_TRAINED,
                            "min_games_for_trained": oc.MIN_GAMES_FOR_TRAINED,
                            "opponent_l2_grid": list(op.L2_GRID), "opponent_selected_l2": opp["l2"],
                            "opponent_residual_l2": opp.get("residual_l2"),
                            "same_game_decay": op.SAME_GAME_DECAY},
        "verified_defensive_execution_examples": terms["n"],
        "verified_execution_games": terms["games"],
        "outcome_model_status": terms["status"],
        "observed_opponent_plays_for_beliefs": opp["n_train"],
        "observed_snaps_for_base_rates": {h: v.get("n") for h, v in base.items() if isinstance(v, dict)},
        "training_cutoff_started_ts": cutoff,
        "trained_sessions": sorted({s.session for s in snaps}),
        "dataset": {"catalog": cat.version(), "sha256": cat.sha256, "path": str(rc.DATASET.relative_to(rc.REPO))},
        "feature_set": {"outcome_candidate_terms": oc.param_names(cat), "outcome_base": ["bias"] +
                        [f"concept:{c}" for c in CLASSES] + [f"ctx:{t}" for t in op.CTX_TOKENS],
                        "opponent_belief": list(op.FEATURE_NAMES), "concept_classes": list(CLASSES)},
        "source": dict(source),
    }
    return CoordinatorModel(meta, cat, oc.OutcomeModel(cat, base, terms), opp)


def save(model: CoordinatorModel, out_dir: str | Path, extra: Mapping[str, Any]) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    body = json.dumps(model.to_json(), sort_keys=True, indent=1)
    (out / "model.json").write_text(body, encoding="utf-8")
    sha = hashlib.sha256(body.encode()).hexdigest()
    manifest = {**model.meta, "model_sha256": sha, **extra}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8")
    return out


_CACHE: dict[tuple[str, int], CoordinatorModel] = {}


def load(art_dir: str | Path) -> CoordinatorModel:
    p = Path(art_dir).expanduser() / "model.json"
    key = (str(p.resolve()), p.stat().st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema") != SCHEMA:
        raise ValueError("not a Ravens defensive coordinator artifact")
    cat = rc.load()
    if d["meta"]["dataset"]["sha256"] != cat.sha256:
        raise ValueError("artifact was trained on a different Ravens scheme dataset version")
    m = CoordinatorModel(d["meta"], cat, oc.OutcomeModel.from_json(cat, d["outcome"]), d["opponent"])
    _CACHE.clear()
    _CACHE[key] = m
    return m


def manifest(art_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(art_dir).expanduser() / "manifest.json").read_text(encoding="utf-8"))
