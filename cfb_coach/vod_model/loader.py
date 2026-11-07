"""Find the newest trained model under a configured directory.

``LAST_TRAINED.json`` may name a ``label_version``. That subdirectory is used
when it exists. The ``model_dir`` field inside the file is ignored — it is a
path on the machine that trained the model, not a path this coach may read.
If that version directory is missing, the highest ``vX.Y`` directory wins.
A missing, empty, or corrupt directory returns None.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from cfb_coach.vod_model.adapter import VodModel, load_version_dir

_VERSION = re.compile(r"v\d+(?:\.\d+)*\Z")


def models_dir() -> Path | None:
    env = (os.environ.get("CFB_COACH_VOD_MODELS") or "").strip()
    if env:
        return Path(env)
    try:
        from cfb_coach.madden.franchise import load_config

        configured = str(load_config().get("vod_models_dir") or "").strip()
    except Exception:  # noqa: BLE001 — config must not break a call
        configured = ""
    return Path(configured) if configured else None


def version_dir(root: Path) -> Path | None:
    """Prefer LAST_TRAINED.json's label_version, else the highest vX.Y folder."""
    if not root.is_dir():
        return None
    last = root / "LAST_TRAINED.json"
    if last.is_file():
        try:
            payload = json.loads(last.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        label = str(payload.get("label_version") or "") if isinstance(payload, dict) else ""
        if label:
            chosen = root / label
            if chosen.is_dir() and (chosen / "beaters.json").is_file():
                return chosen
            # label points nowhere — fall through to whatever versions exist
    found = [
        p for p in root.iterdir()
        if p.is_dir() and _VERSION.fullmatch(p.name) and (p / "beaters.json").is_file()
    ]
    if not found:
        return None

    def _key(path: Path) -> tuple[int, ...]:
        return tuple(int(part) for part in path.name[1:].split("."))

    return max(found, key=_key)


def load_model(root: Path | None = None) -> VodModel | None:
    """Newest usable model, or None. Never raises into prep or live."""
    try:
        base = root if root is not None else models_dir()
        if base is None:
            return None
        chosen = version_dir(base)
        if chosen is None:
            return None
        return load_version_dir(chosen)
    except Exception:  # noqa: BLE001 — missing/corrupt model is a no-op
        return None
