"""The only module that knows the trained-model JSON shape (labels v0.7).

A later training PR can keep emitting this shape. Decisions read beaters,
not the logistic coefficients in ``model_params.json``. Those coefficients
are accepted so a corrupt params file still fails the load, and they never
move a call on their own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfb_coach.learning import PRIOR_OBS

# Offense calls versus a defensive look. The CPU-defense view is not a prior.
_CALL_VIEW = "all_clean_pairs"


@dataclass(frozen=True)
class VodCell:
    game: str
    opponent_type: str
    look: str
    look_grain: str
    call: str
    call_key: str
    call_class: str
    n: int
    n_vods: int
    successes: int
    raw_success: float
    shrunk_success: float
    lower_bound: float
    baseline: float
    lift: float
    tentative: bool
    confidence: str
    model_version: str


@dataclass(frozen=True)
class VodModel:
    version: str
    tentative_n: int
    trained_at: str
    script_version: str
    # (game, opponent_type, grain, look) -> cells
    cells: dict[tuple[str, str, str, str], tuple[VodCell, ...]]
    look_n: dict[tuple[str, str, str, str], int]
    base_rate: dict[tuple[str, str], float]

    def looks(self, game: str, opponent_type: str, grain: str = "family") -> list[str]:
        prefix = (game, opponent_type, grain)
        names = [key[3] for key in self.look_n if key[:3] == prefix]
        names.sort(key=lambda name: -self.look_n[(game, opponent_type, grain, name)])
        return names

    def cells_for(self, game: str, opponent_type: str, grain: str, look: str) -> tuple[VodCell, ...]:
        return self.cells.get((game, opponent_type, grain, look), ())


def _shrink(successes: float, n: float, baseline: float) -> float:
    return (successes + PRIOR_OBS * baseline) / (n + PRIOR_OBS) if n >= 0 else baseline


def _cell(
    raw: dict[str, Any],
    *,
    game: str,
    opponent_type: str,
    look: str,
    grain: str,
    baseline: float,
    version: str,
    tentative_n: int,
) -> VodCell:
    n = int(raw.get("n") or 0)
    successes = int(raw.get("successes") or 0)
    base = float(baseline)
    shrunk = raw.get("shrunk_success")
    shrunk_f = float(shrunk) if shrunk is not None else _shrink(successes, n, base)
    lb = raw.get("lower_bound_90")
    lb_f = float(lb) if lb is not None else shrunk_f
    raw_success = float(raw.get("raw_success") if raw.get("raw_success") is not None else (successes / n if n else 0.0))
    lift = raw.get("lift_vs_situation")
    lift_f = float(lift) if lift is not None else shrunk_f - base
    tentative = bool(raw.get("tentative")) or n < tentative_n
    confidence = str(raw.get("confidence") or ("tentative" if tentative else "ok"))
    if confidence == "tentative":
        tentative = True
    return VodCell(
        game=game,
        opponent_type=opponent_type,
        look=look,
        look_grain=grain,
        call=str(raw.get("call") or ""),
        call_key=str(raw.get("call_key") or raw.get("call") or ""),
        call_class=str(raw.get("call_class") or ""),
        n=n,
        n_vods=int(raw.get("n_vods") or 0),
        successes=successes,
        raw_success=raw_success,
        shrunk_success=shrunk_f,
        lower_bound=lb_f,
        baseline=base,
        lift=lift_f,
        tentative=tentative,
        confidence=confidence,
        model_version=version,
    )


def parse_beaters(payload: dict[str, Any], *, trained_at: str = "", script_version: str = "") -> VodModel:
    version = str(payload.get("label_version") or "")
    if not version:
        raise ValueError("beaters missing label_version")
    tentative_n = int(payload.get("tentative_n") or 15)
    cells: dict[tuple[str, str, str, str], tuple[VodCell, ...]] = {}
    look_n: dict[tuple[str, str, str, str], int] = {}
    base_rate: dict[tuple[str, str], float] = {}
    for key, stratum in (payload.get("strata") or {}).items():
        if not isinstance(key, str) or "/" not in key or not isinstance(stratum, dict):
            continue
        game, opponent_type = key.split("/", 1)
        if opponent_type not in ("cpu", "human"):
            continue
        base = float(stratum.get("stratum_base_rate_all_known") or 0.5)
        base_rate[(game, opponent_type)] = base
        views = stratum.get("views") or {}
        view = views.get(_CALL_VIEW) or {}
        if not isinstance(view, dict):
            continue
        if view.get("stratum_clean_success_rate") is not None:
            base_rate[(game, opponent_type)] = float(view["stratum_clean_success_rate"])
        for grain_key, grain in (("look_family", "family"), ("look", "raw")):
            block = view.get(grain_key) or {}
            if not isinstance(block, dict):
                continue
            for look_name, look_block in (block.get("looks") or {}).items():
                if not isinstance(look_block, dict):
                    continue
                parsed = tuple(
                    _cell(
                        c,
                        game=game,
                        opponent_type=opponent_type,
                        look=str(look_name),
                        grain=grain,
                        baseline=base_rate[(game, opponent_type)],
                        version=version,
                        tentative_n=tentative_n,
                    )
                    for c in (look_block.get("calls") or [])
                    if isinstance(c, dict) and (c.get("call") or c.get("call_key"))
                )
                idx = (game, opponent_type, grain, str(look_name))
                cells[idx] = parsed
                look_n[idx] = int(look_block.get("n") or 0)
    if not cells:
        raise ValueError("beaters had no cpu/human rows")
    return VodModel(
        version=version,
        tentative_n=tentative_n,
        trained_at=trained_at,
        script_version=script_version,
        cells=cells,
        look_n=look_n,
        base_rate=base_rate,
    )


def load_version_dir(path: Path) -> VodModel:
    """Read one ``vX.Y`` directory. Raises if beaters or model_params are unusable."""
    beaters_path = path / "beaters.json"
    params_path = path / "model_params.json"
    beaters = json.loads(beaters_path.read_text(encoding="utf-8"))
    if not isinstance(beaters, dict):
        raise ValueError("beaters.json is not an object")
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise ValueError("model_params.json is not an object")
    trained_at = ""
    script_version = ""
    manifest = path / "manifest.json"
    if manifest.is_file():
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("manifest.json is corrupt") from exc
        if isinstance(meta, dict):
            trained_at = str(meta.get("trained_at") or "")
            script_version = str(meta.get("script_version") or "")
    return parse_beaters(beaters, trained_at=trained_at, script_version=script_version)
