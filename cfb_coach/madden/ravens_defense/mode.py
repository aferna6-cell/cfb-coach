"""Defense mode selection: the previous defensive caller (default) or ravens_defense_ml.

* DB meta ``ravens_defense_mode_json`` = {"enabled": bool, "artifact": dir, "model_sha256": ..., "previous": {...}}
  is written only by ``ml ravens-defense enable`` / ``disable`` (each after a DB backup).
* Env ``CFB_COACH_DEFENSE_MODE`` overrides for one process: ``ravens_defense_ml`` (``play --ravens-defense``)
  or ``previous`` (``play --previous-defense``).
* Default (nothing set) = the previous defensive caller, unchanged. Offense mode is independent.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ENV_MODE = "CFB_COACH_DEFENSE_MODE"
ENV_ARTIFACT = "CFB_COACH_RAVENS_DEFENSE_MODEL"
META_KEY = "ravens_defense_mode_json"
RAVENS = "ravens_defense_ml"
PREVIOUS = "previous"


def load_state(db: Any) -> dict[str, Any]:
    if db is None:
        return {}
    try:
        raw = db.get_meta(META_KEY)
    except Exception:  # noqa: BLE001 — missing meta table = not enabled
        return {}
    try:
        return dict(json.loads(raw)) if raw else {}
    except (ValueError, TypeError):
        return {}


def defense_mode(db: Any) -> str:
    env = (os.environ.get(ENV_MODE) or "").strip().lower()
    if env in (RAVENS, PREVIOUS):
        return env
    return RAVENS if load_state(db).get("enabled") else PREVIOUS


def artifact_dir(db: Any) -> Path | None:
    env = (os.environ.get(ENV_ARTIFACT) or "").strip()
    if env:
        return Path(env)
    st = load_state(db)
    if st.get("artifact"):
        return Path(st["artifact"])
    from cfb_coach.madden.ravens_defense.model import models_root
    cur = models_root() / "latest"
    return Path(cur.read_text().strip()) if cur.is_file() else None


def set_state(db: Any, *, enabled: bool, artifact: str | None, model_sha256: str | None) -> dict[str, Any]:
    st = load_state(db)
    prev = {k: st.get(k) for k in ("enabled", "artifact", "model_sha256")} if st else None
    st.update({"enabled": bool(enabled), "changed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "previous": prev})
    if artifact is not None:
        st["artifact"] = artifact
    if model_sha256 is not None:
        st["model_sha256"] = model_sha256
    db.set_meta(META_KEY, json.dumps(st, sort_keys=True))
    return st
