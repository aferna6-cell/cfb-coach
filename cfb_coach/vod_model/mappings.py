"""Play mappings that sit next to the trained VOD models.

``<models_dir>/play_mappings/LATEST.json`` points at ``v<N>.json`` in that same
folder. Each record maps a raw VOD call for one game to a playbook, formation,
and play. Only a resolved record with confidence high or med and exactly one
formation is used. Low-confidence, ambiguous, unresolved, and missing records
are ignored.

A missing, unreadable, or corrupt file is a no-op and never raises. Nothing
here hardcodes a machine path. When ``models_dir`` is omitted, the directory is
``CFB_COACH_VOD_MODELS`` or the Madden config, the same one the model loader uses.

``vod_model/live.py`` does not import this module. Prep and book edits do.
A later decision pipeline should call the two functions below and nothing else.

Public API::

    load_mappings(models_dir: Path | None = None) -> MappingIndex | None
    lookup(index: MappingIndex | None, game: str, raw_name: str) -> MappedPlay | None
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfb_coach.vod_model.loader import models_dir as configured_models_dir

_NORM = re.compile(r"[^a-z0-9]")
_CONFIDENCE = {"high": 2, "med": 1, "medium": 1}
_POINTER_KEYS = ("file", "path", "mappings")


def norm_name(value: str | None) -> str:
    """Case and punctuation folding, the same key the Madden catalog uses."""
    return _NORM.sub("", (value or "").lower())


@dataclass(frozen=True)
class MappedPlay:
    game: str
    raw_name: str
    play_name: str
    playbook: str
    playbook_name: str
    formation: str
    plays: tuple[str, ...]
    confidence: str
    catalog_book: str

    def display(self, raw: str) -> str:
        book = self.playbook_name or self.catalog_book or self.playbook or "mapping"
        return f"{raw} → {book} / {self.formation} / {self.play_name}"


@dataclass(frozen=True)
class MappingIndex:
    version_label: str
    by_key: dict[tuple[str, str], MappedPlay]


def lookup(index: MappingIndex | None, game: str, raw_name: str) -> MappedPlay | None:
    """Usable mapping for this game and raw call, or None."""
    if index is None or not (raw_name or "").strip():
        return None
    return index.by_key.get((_game_key(game), norm_name(raw_name)))


def match_pair(
    forms: dict[str, Any],
    play: str,
    formation: str | None = None,
) -> tuple[str, str] | None:
    """A formation+play already in ``forms``.

    When ``formation`` is in the book and contains the play, that pair wins.
    Otherwise the first formation that has the play.
    """
    key = norm_name(play)
    if not key:
        return None
    wanted = norm_name(formation) if formation else ""

    def _hits(only: str) -> tuple[str, str] | None:
        for fname, plays in forms.items():
            if not isinstance(plays, list):
                continue
            if only and norm_name(str(fname)) != only:
                continue
            for item in plays:
                if isinstance(item, str) and norm_name(item) == key:
                    return str(fname), item
        return None

    if wanted:
        found = _hits(wanted)
        if found:
            return found
    return _hits("")


def load_mappings(models_dir: Path | None = None) -> MappingIndex | None:
    """Usable mappings from ``models_dir``, or None. Never raises.

    Omit ``models_dir`` to use ``CFB_COACH_VOD_MODELS`` or the Madden config.
    """
    try:
        base = models_dir if models_dir is not None else configured_models_dir()
        if base is None:
            return None
        return _load_folder(Path(base) / "play_mappings")
    except Exception:  # noqa: BLE001 — missing or corrupt mappings are a no-op
        return None


def _game_key(game: str | None) -> str:
    return (game or "").strip().lower()


def _version_label(version: str) -> str:
    text = (version or "").strip()
    if not text:
        text = "unversioned"
    if text.lower().startswith("play_mappings"):
        return text
    if not text.startswith("v"):
        text = "v" + text
    return f"play_mappings {text}"


def _inside(folder: Path, raw: str) -> Path | None:
    text = (raw or "").strip().replace("\\", "/")
    if text.startswith("play_mappings/"):
        text = text[len("play_mappings/"):]
    if not text:
        return None
    if not text.endswith(".json"):
        text = text + ".json"
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    path = (folder / candidate).resolve()
    try:
        path.relative_to(folder.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _pointer(folder: Path, latest: dict[str, Any]) -> Path | None:
    raw = ""
    for key in _POINTER_KEYS:
        value = latest.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            break
    if raw:
        return _inside(folder, raw)
    version = latest.get("version")
    if isinstance(version, str) and version.strip():
        return _inside(folder, version.strip())
    return None


def _one_formation(rec: dict[str, Any]) -> str | None:
    formation = rec.get("formation")
    if not isinstance(formation, str) or not formation.strip():
        return None
    formation = formation.strip()
    candidates = rec.get("formation_candidates")
    if isinstance(candidates, list):
        names = [item.strip() for item in candidates if isinstance(item, str) and item.strip()]
        unique = {norm_name(name) for name in names}
        if len(unique) > 1:
            return None
        if len(unique) == 1 and norm_name(formation) not in unique:
            return None
    return formation


def _confidence(rec: dict[str, Any]) -> str | None:
    text = str(rec.get("confidence") or "").strip().lower()
    if text == "medium":
        text = "med"
    if text not in {"high", "med"}:
        return None
    return text


def _resolved(rec: dict[str, Any]) -> bool:
    status = str(rec.get("status") or "").strip().lower()
    if not status.startswith("resolved"):
        return False
    if "ambiguous" in status or "unresolved" in status:
        return False
    return True


def _play_list(rec: dict[str, Any], formation: str) -> tuple[str, ...]:
    block = rec.get("formation_plays")
    if not isinstance(block, dict):
        return ()
    plays = block.get(formation)
    if not isinstance(plays, list):
        wanted = norm_name(formation)
        for key, value in block.items():
            if isinstance(key, str) and norm_name(key) == wanted and isinstance(value, list):
                plays = value
                break
    if not isinstance(plays, list):
        return ()
    return tuple(item.strip() for item in plays if isinstance(item, str) and item.strip())


def _parse_record(rec: dict[str, Any]) -> MappedPlay | None:
    if not _resolved(rec):
        return None
    confidence = _confidence(rec)
    formation = _one_formation(rec)
    play_name = rec.get("play_name")
    raw_name = rec.get("raw_name")
    game = rec.get("game")
    if confidence is None or formation is None:
        return None
    if not isinstance(play_name, str) or not play_name.strip():
        return None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None
    if not isinstance(game, str) or not game.strip():
        return None
    catalog_book = rec.get("playbook_in_coach_catalog")
    return MappedPlay(
        game=_game_key(game),
        raw_name=raw_name.strip(),
        play_name=play_name.strip(),
        playbook=str(rec.get("playbook") or "").strip(),
        playbook_name=str(rec.get("playbook_name") or "").strip(),
        formation=formation,
        plays=_play_list(rec, formation),
        confidence=confidence,
        catalog_book=catalog_book.strip() if isinstance(catalog_book, str) else "",
    )


def _spellings(rec: dict[str, Any], raw_name: str) -> set[str]:
    keys = {norm_name(raw_name)}
    spellings = rec.get("raw_spellings")
    if isinstance(spellings, dict):
        values = spellings.keys()
    elif isinstance(spellings, list):
        values = spellings
    else:
        values = ()
    for spelling in values:
        if isinstance(spelling, str) and spelling.strip():
            keys.add(norm_name(spelling))
    keys.discard("")
    return keys


def _load_folder(folder: Path) -> MappingIndex | None:
    latest_path = folder / "LATEST.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(latest, dict):
        return None
    target = _pointer(folder, latest)
    if target is None:
        return None
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        records = payload.get("mappings")
        version = str(payload.get("version") or latest.get("version") or "")
    else:
        return None
    if not isinstance(records, list):
        return None
    by_key: dict[tuple[str, str], MappedPlay] = {}
    rank: dict[tuple[str, str], int] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        parsed = _parse_record(rec)
        if parsed is None:
            continue
        score = _CONFIDENCE.get(parsed.confidence, 0)
        for key in _spellings(rec, parsed.raw_name):
            idx = (parsed.game, key)
            if score >= rank.get(idx, -1):
                by_key[idx] = parsed
                rank[idx] = score
    return MappingIndex(version_label=_version_label(version), by_key=by_key)
