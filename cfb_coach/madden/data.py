"""Madden 27 data loaders + shared-persona opponents.

Opponents are NOT a Madden-only cast: they are derived from the CFB seed's
persona roster (same ids / display names / archetypes). Only persona-level
traits carry over — never CFB playbook or concept names.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from cfb_coach.seed import load_seed as load_cfb_seed

META_VERSION = "madden27-2026-09"

_CONF_STEP_DOWN = {"high": "medium", "medium": "low", "low": "low", "none": "none"}


def _data_path(name: str) -> Path:
    try:
        ref = resources.files("cfb_coach").joinpath(f"data/madden27/{name}")
        with resources.as_file(ref) as p:
            return Path(p)
    except Exception:
        return Path(__file__).resolve().parent.parent / "data" / "madden27" / name


@lru_cache(maxsize=None)
def _load_json(name: str) -> dict[str, Any]:
    with _data_path(name).open(encoding="utf-8") as f:
        return json.load(f)


def load_meta_baseline() -> dict[str, Any]:
    return copy.deepcopy(_load_json("meta_baseline.json"))


def load_macro_catalog() -> dict[str, Any]:
    return copy.deepcopy(_load_json("macro_catalog.json"))


def get_macro(name: str | None) -> dict[str, Any] | None:
    macros = _load_json("macro_catalog.json").get("macros") or {}
    m = macros.get((name or "").strip().upper())
    return copy.deepcopy(m) if m else None


def validation_notes() -> str:
    return _data_path("VALIDATION_NOTES.txt").read_text(encoding="utf-8")


def persona_profile(oid: str, cfb_opp: dict[str, Any]) -> dict[str, Any]:
    """Madden profile for one shared persona (archetype + traits only)."""
    dvs = cfb_opp.get("defense_vs_us") or {}
    offense = cfb_opp.get("offense") or {}
    persona_conf = (cfb_opp.get("confidence") or "none").lower()
    is_cpu = oid == "cpu"
    traits: dict[str, Any] = {}
    if offense.get("escape"):
        traits["escape"] = offense["escape"]
    if dvs.get("man_blitz"):
        traits["man_blitz"] = dvs["man_blitz"]
    return {
        "display_name": "CPU Franchise" if is_cpu else cfb_opp.get("display_name", oid),
        # NFL team label is optional and set later; personas stay shared
        "team_now": "CPU" if is_cpu else None,
        "nfl_team": None,
        "skill": cfb_opp.get("skill"),
        "archetype": dvs.get("archetype") or "unknown",
        "persona_confidence": persona_conf,
        # No Madden film yet — one step below the CFB persona confidence
        "confidence": _CONF_STEP_DOWN.get(persona_conf, "low"),
        "traits": traits,
        "defense_vs_us": {"archetype": dvs.get("archetype") or "unknown"},
        "persona_source": "shared CFB persona (archetype carry-over)",
    }


@lru_cache(maxsize=1)
def _load_seed_cached() -> dict[str, Any]:
    seed = copy.deepcopy(_load_json("seed.json"))
    cfb = load_cfb_seed()
    seed["opponents"] = {
        oid: persona_profile(oid, opp) for oid, opp in cfb["opponents"].items()
    }
    return seed


def load_seed() -> dict[str, Any]:
    """Madden seed with `opponents` derived from the shared CFB persona roster."""
    return copy.deepcopy(_load_seed_cached())


def all_reads() -> dict[str, str]:
    return dict(_load_json("seed.json").get("reads") or {})


def reads_for(play: str) -> str:
    return all_reads().get(play, "Primary → Checkdown")


def user_job_for(call: str) -> str:
    return (_load_json("seed.json").get("user_jobs") or {}).get(call, "User hook")


def archetype_lean(archetype: str | None) -> dict[str, str]:
    leans = _load_json("seed.json").get("archetype_leans") or {}
    return dict(leans.get((archetype or "unknown").lower()) or leans.get("unknown") or {})


def cited_plays() -> set[str]:
    pb = _load_json("seed.json")["playbooks"]
    out: set[str] = set()
    for meta in (pb.get("offense_formations") or {}).values():
        out.update(meta.get("cited") or [])
    for meta in (pb.get("defense_packages") or {}).values():
        out.update(meta.get("cited") or [])
    return out
