"""Offline-only scalar score cache and sanitized proposal provenance.

Nothing here imports a ranker. Live callers must not import this module. Portfolio
scoring remains owned by offense_portfolio; cached values are explicitly scoped
by their scoring purpose as well as the full pregame provenance key.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

PROVENANCE_FIELDS = (
    "catalog_hash", "n_indexed", "n_installable", "artifact_hash",
    "opponent_history_hash", "rights_feed_digest",
)
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / f"cfb-coach-offense-scores-{os.getuid()}"
SCHEMA = "madden27.offense_score_cache.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_digest(art: Any) -> str:
    return digest({
        "model_version": art.model_version,
        "knowledge_version": art.knowledge_version,
        "evidence_quality": art.evidence_quality,
        "n_supervised": art.n_supervised,
        "loader": getattr(art, "_pregame_artifact_loader", "experimental"),
    })


def opponent_history_digest(coverage_mix: Mapping | None, mix_source: str, unknown_mass: float) -> str:
    return digest({"coverage_mix": dict(coverage_mix or {}),
                   "mix_source": mix_source, "unknown_mass": unknown_mass})


def catalog_provenance(catalogue: Mapping, catalog_index: Any = None) -> dict[str, Any]:
    """Hash only rights-relevant metadata; never art, URLs or structure payloads."""
    if catalog_index is not None:
        entries = catalog_index.entries
        catalog_hash = catalog_index.catalog_hash
        n_indexed = catalog_index.n_indexed
        n_installable = sum(bool(e.installable) for e in entries)
        permitted = sum(bool(e.model_use_permitted) for e in entries)
        tiers = dict(Counter(e.feature_tier for e in entries))
    else:
        # The explicit catalogue hook does not claim to be the Ace universe.
        pairs = {(f, p) for forms in catalogue.values() for f, plays in forms.items() for p in plays}
        catalog_hash = digest(catalogue)
        n_indexed = n_installable = len(pairs)
        permitted = 0  # No provisional feed was supplied by this hook.
        tiers = {"name_only": n_indexed}
    rights = {"catalog_hash": catalog_hash, "n_indexed": n_indexed,
              "model_use_permitted": permitted, "feature_tier": tiers}
    return {"catalog_hash": catalog_hash, "n_indexed": n_indexed,
            "n_installable": n_installable, "rights_feed_digest": digest(rights)}


def cache_key(*, catalog_hash: str, artifact_hash: str, opponent_history_hash: str,
              priors_on: bool, rights_feed_digest: str) -> str:
    return digest(dict(catalog_hash=catalog_hash, artifact_hash=artifact_hash,
                       opponent_history_hash=opponent_history_hash, priors_on=priors_on,
                       rights_feed_digest=rights_feed_digest))


class ScoreCache:
    """Explicit offline lookup/record API. Persist scalar aggregates only.

    Use a purpose-specific score id: different probes/structures must never share
    an entry. Atomic replacement prevents partial JSON after an interrupted write.
    """

    def __init__(self, key: str, directory: str | Path | None = None):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Expected a SHA-256 cache key")
        self.key = key
        self.path = Path(directory) / f"{key}.json" if directory is not None else DEFAULT_CACHE_DIR / f"{key}.json"
        self.scores: dict[str, float] = {}

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
            if raw.get("schema") == SCHEMA and raw.get("key") == self.key:
                self.scores = {k: float(v) for k, v in raw["scores"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self.scores = {}

    def lookup(self, score_id: Any) -> float | None:
        return self.scores.get(digest(score_id))

    def record(self, score_id: Any, score: float) -> float:
        self.scores[digest(score_id)] = float(score)
        return float(score)

    def get_or_compute(self, score_id: Any, compute: Callable[[], float]) -> float:
        score = self.lookup(score_id)
        return self.record(score_id, compute()) if score is None else score

    def store(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as out:
                name = out.name
                json.dump({"schema": SCHEMA, "key": self.key, "scores": self.scores}, out, allow_nan=False)
            os.replace(name, self.path)
        finally:
            if name is not None:
                Path(name).unlink(missing_ok=True)
