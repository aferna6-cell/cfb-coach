"""Load packaged seed.json (source of truth)."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any


def _seed_path() -> Path:
    """Prefer packaged data; fall back to relative path for editable runs."""
    try:
        ref = resources.files("cfb_coach").joinpath("data/seed.json")
        with resources.as_file(ref) as p:
            return Path(p)
    except Exception:
        return Path(__file__).resolve().parent / "data" / "seed.json"


@lru_cache(maxsize=1)
def load_seed() -> dict[str, Any]:
    path = _seed_path()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def seed_bytes() -> bytes:
    return _seed_path().read_bytes()
