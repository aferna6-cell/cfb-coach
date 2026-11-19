"""Offense mode selection: the existing unrestricted coordinator (default) or civil_scheme_only.

* DB meta ``civil_scheme_mode_json`` = {"enabled": bool, "artifact": dir, "model_sha256": ...}
  is written only by ``ml civil-scheme enable`` / ``disable`` (after a DB backup).
* Env ``CFB_COACH_OFFENSE_MODE`` overrides for one process: ``civil_scheme_only`` (``play --civil-scheme-only``)
  or ``unrestricted`` (``play --unrestricted-offense``).
* Default (nothing set) = the existing unrestricted coordinator, unchanged.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ENV_MODE = "CFB_COACH_OFFENSE_MODE"
ENV_ARTIFACT = "CFB_COACH_CIVIL_SCHEME_MODEL"
META_KEY = "civil_scheme_mode_json"
CIVIL = "civil_scheme_only"
UNRESTRICTED = "unrestricted"


def models_root() -> Path:
    return Path.home() / ".cfb-coach" / "models" / "civil_scheme"


def load_state(db: Any) -> dict[str, Any]:
    if db is None:
        return {}
    try:
        raw = db.get_meta(META_KEY)
    except Exception:  # noqa: BLE001 — missing meta table = not enabled
        return {}
    if not raw:
        return {}
    try:
        return dict(json.loads(raw))
    except (ValueError, TypeError):
        return {}


def offense_mode(db: Any) -> str:
    env = (os.environ.get(ENV_MODE) or "").strip().lower()
    if env in (CIVIL, UNRESTRICTED):
        return env
    return CIVIL if load_state(db).get("enabled") else UNRESTRICTED


def artifact_dir(db: Any) -> Path | None:
    env = (os.environ.get(ENV_ARTIFACT) or "").strip()
    if env:
        return Path(env)
    st = load_state(db)
    if st.get("artifact"):
        return Path(st["artifact"])
    # Without `enable`, a session flag (--civil-scheme-only) uses the newest trained artifact.
    cur = models_root() / "latest"
    if cur.is_file():
        return Path(cur.read_text().strip())
    return None


def set_state(db: Any, *, enabled: bool, artifact: str | None, model_sha256: str | None) -> dict[str, Any]:
    st = load_state(db)
    st.update({"enabled": bool(enabled), "changed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if artifact is not None:
        st["artifact"] = artifact
    if model_sha256 is not None:
        st["model_sha256"] = model_sha256
    db.set_meta(META_KEY, json.dumps(st, sort_keys=True))
    return st
