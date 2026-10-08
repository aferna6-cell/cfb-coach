"""Load a concept model from ``<models_dir>/concept/``.

``LAST_TRAINED.json`` names a ``c<N>`` directory. The ``model_dir`` field inside
that file is ignored: it is a path on the machine that trained the model.
A missing, unreadable, or corrupt folder returns None and never raises.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfb_coach.vod_model.loader import models_dir

_VERSION = re.compile(r"c\d+\Z")
SCHEMA = "concept_beaters.v1"


@dataclass(frozen=True)
class ConceptModel:
    """Parsed ``concept_beaters.json``. Decisions read these cells, not coefficients."""

    version: str
    label_version: str
    concepts_version: str
    tentative_n: int
    trained_at: str
    strata: dict[str, Any]

    def stratum(self, game: str, opponent_type: str) -> dict[str, Any] | None:
        found = self.strata.get(f"{game}/{opponent_type}")
        return found if isinstance(found, dict) else None

    def view(self, game: str, opponent_type: str, name: str = "all_offense") -> dict[str, Any] | None:
        stratum = self.stratum(game, opponent_type)
        if not stratum:
            return None
        views = stratum.get("views") or {}
        found = views.get(name)
        return found if isinstance(found, dict) else None


def load_concept_model(root: Path | None = None) -> ConceptModel | None:
    """Newest concept model, or None. Does not read the pair-model directory."""
    try:
        base = root if root is not None else models_dir()
        if base is None:
            return None
        concept_root = Path(base) / "concept"
        pointer = concept_root / "LAST_TRAINED.json"
        if not pointer.is_file():
            return None
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        version = str(payload.get("version") or "")
        if not _VERSION.fullmatch(version):
            return None
        beaters = concept_root / version / "concept_beaters.json"
        data = json.loads(beaters.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return None
        strata = data.get("strata")
        if not isinstance(strata, dict):
            return None
        return ConceptModel(
            version=str(data.get("version") or version),
            label_version=str(data.get("label_version") or payload.get("label_version") or ""),
            concepts_version=str(data.get("concepts_version") or payload.get("concepts_version") or ""),
            tentative_n=int(data.get("tentative_n") or 15),
            trained_at=str(data.get("trained_at") or payload.get("trained_at") or ""),
            strata=strata,
        )
    except Exception:  # noqa: BLE001 — missing or corrupt output is a no-op
        return None
