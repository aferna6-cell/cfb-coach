"""Live eval harness for one Madden situation.

Runs the same path ``call`` / ``play`` use (parse, bind the opponent look,
``make_call``, then shadow or hybrid scoring) and the heuristic caller on
the same input. Databases are temporary ``CFB_COACH_MADDEN_DB`` and
``CFB_COACH_DB`` files. This module does not read or write ``HOME``.

Heuristic is the default. Shadow scores without changing the shown call.
Hybrid may use a model only when ``rank_live`` accepts it under the
``scored_on='reviewed'`` guard; otherwise the heuristic call stays and the
refusal is recorded.

``clock`` is the model clock. Leave it unset for a realtime clock. Tests
install a ``ManualClock`` and replace :func:`score_model` so a step past
the 150 ms cutoff falls back without waiting on the wall clock.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cfb_coach.madden.model.inference import (
    ML_LATENCY_BUDGET_MS,
    BudgetCut,
    LiveClock,
    _CUTOFF_MARGIN_MS,
)
from cfb_coach.madden.model.schema import CoachingMode, MLStatus

OPPONENT_ID = "gavin"
_PLACEHOLDER_ADJUSTMENTS = frozenset({"", "no adj", "none", "no adjustment"})
_MADDEN_STRATUM = re.compile(r"^madden27/(?:cpu|human)(?:/[^/]+)?$")
_OFFLINE_GATE = "offline_eval"

# Installed by tests that need a fake model clock. Production calls leave it unset.
clock: LiveClock | None = None


def _active_clock() -> LiveClock:
    return clock if clock is not None else LiveClock()


def _mode(mode: CoachingMode | str | None) -> CoachingMode:
    if isinstance(mode, CoachingMode):
        return mode
    text = str(mode or "heuristic").strip().lower()
    try:
        parsed = CoachingMode(text)
    except ValueError:
        return CoachingMode.HEURISTIC
    if parsed in (CoachingMode.HEURISTIC, CoachingMode.SHADOW, CoachingMode.HYBRID):
        return parsed
    return CoachingMode.HEURISTIC


def _side(side: str | None) -> str:
    text = str(side or "offense").strip().lower()
    if text.startswith("d"):
        return "defense"
    return "offense"


def _status_value(status: Any) -> str | None:
    if status is None or status is MLStatus.OK:
        return None
    if isinstance(status, MLStatus):
        return None if status is MLStatus.OK else status.value
    text = str(status).strip().lower()
    if text in {"", "ok", "none"}:
        return None
    return text


def _split_reads(raw: str | None) -> tuple[str | None, str | None]:
    text = (raw or "").strip()
    if "→" not in text:
        return (text or None), None
    parts = [part.strip() for part in text.split("→")]
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], None
    return None, None


def _macro_state(call: Any):
    from cfb_coach.madden.model.schema import MacroState, adapt_macro_state

    explicit = getattr(call, "macro_state", None)
    if explicit is None:
        info = getattr(call, "macro_info", None)
        if isinstance(info, dict):
            explicit = info.get("state", info.get("macro_state"))
    if explicit is not None:
        try:
            return adapt_macro_state(explicit)
        except (TypeError, ValueError):
            return MacroState.UNKNOWN
    if getattr(call, "macro", None):
        # The live caller only attaches a macro it has armed for this snap.
        return MacroState.ARMED
    return None


def _adjustment_id(call: Any) -> str | None:
    macro = getattr(call, "macro", None)
    if macro and str(macro).strip():
        return str(macro).strip()
    adjustment = getattr(call, "adjustment", None)
    if isinstance(adjustment, dict):
        for key in ("id", "name", "label"):
            value = adjustment.get(key)
            if value and str(value).strip():
                return str(value).strip()
    label = str(getattr(call, "adj_or_macro", "") or "").strip()
    if label.lower() in _PLACEHOLDER_ADJUSTMENTS:
        return None
    return label or None


def _exact_line(call: Any, macro_state: Any) -> str:
    from cfb_coach.madden.data import reads_for, user_job_for
    from cfb_coach.madden.model.inference import exact_defense_line, exact_offense_line

    label = str(getattr(call, "adj_or_macro", "") or "").strip()
    adjustment = None if label.lower() in _PLACEHOLDER_ADJUSTMENTS else label or None
    macro_name = getattr(call, "macro", None) or None
    if getattr(call, "side", "offense") == "defense":
        return exact_defense_line(
            call.formation,
            call.play,
            adjustment,
            getattr(call, "read_or_user", None) or user_job_for(call.play),
            macro_name=macro_name,
            macro_state=macro_state,
        )
    read1, read2 = _split_reads(getattr(call, "read_or_user", None) or reads_for(call.play))
    return exact_offense_line(
        call.formation,
        call.play,
        adjustment,
        read1,
        read2,
        macro_name=macro_name,
        macro_state=macro_state,
    )


def _call_record(call: Any, *, mode: str) -> dict[str, Any]:
    state = _macro_state(call)
    return {
        "formation": getattr(call, "formation", None),
        "play": getattr(call, "play", None),
        "adjustment_id": _adjustment_id(call),
        "macro_state": state.value if state is not None else None,
        "line": _exact_line(call, state),
        "mode": mode,
    }


def _in_installed_book(call: Any, book: dict[str, list[str]]) -> bool:
    formation = getattr(call, "formation", None)
    play = getattr(call, "play", None)
    plays = book.get(formation) if formation else None
    return plays is not None and play in list(plays)


def _legal(call: Any, book: dict[str, list[str]]) -> bool:
    """In the installed book, and the macro is armed-or-later or absent.

    A call with no macro and no adjustment is the no-adjustment case. A macro
    below armed, or any other adjustment without an armed-or-later macro, is
    not legal. The book is the applied ``active_playbook_json`` for this side.
    """
    from cfb_coach.madden.model.schema import is_live_selectable

    if not _in_installed_book(call, book):
        return False
    if _adjustment_id(call) is None:
        return True
    return is_live_selectable(_macro_state(call))


def prepare_session(db: Any) -> None:
    """Lock stock books into ``active_playbook_json`` on this temporary database.

    Tests wrap this to register a model before the live path resolves one.
    """
    from cfb_coach.madden.playbook import STOCK, lock_books, make_record

    plans = {}
    for side in ("offense", "defense"):
        plans[side] = {
            "record": make_record(side, STOCK, None),
            "change": "none",
            "untouched": False,
        }
    lock_books(db, plans, applied=True)


def _installed_books(db: Any) -> dict[str, dict[str, list[str]]]:
    from cfb_coach.madden.playbook import eligible, load_books

    return eligible(load_books(db))


def _parse(sit: Any, side: str, opp_look: Any) -> Any:
    """Same situation binding ``cmd_call`` / ``cmd_play`` use."""
    from cfb_coach.game_score import LiveContext, absorb_and_stamp
    from cfb_coach.madden.model.inference import bind_opp_look
    from cfb_coach.madden.situation import Situation, parse_madden_situation

    if isinstance(sit, Situation):
        parsed = copy.deepcopy(sit)
        raw = parsed.raw or ""
    else:
        raw = str(sit)
        parsed = parse_madden_situation(raw, default_side=side)
    parsed.side = side
    # An explicit blank is NULL and does not fall through to a token in ``raw``.
    # ``empty`` stays the empty-backfield token.
    bind_opp_look(parsed, look=opp_look, raw=raw)
    absorb_and_stamp(parsed, LiveContext())
    return parsed


def _make_call(sit: Any, db: Any, active: Any, seed: int) -> Any:
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.cli import _session_live_macros

    return playcaller.make_call(
        sit,
        OPPONENT_ID,
        db,
        rng=random.Random(int(seed)),
        active_macros=active,
        live_macros=_session_live_macros(argparse.Namespace(no_macros=False)),
    )


def _model_report(db: Any) -> tuple[str, str | None]:
    from cfb_coach.madden.model.inference import (
        _load_entry,
        read_scored_on,
        resolve_model_version,
        resolve_registry_dir,
    )
    from cfb_coach.madden.model.train import load_artifact

    version = resolve_model_version(db)
    artifact = None
    try:
        entry = _load_entry(resolve_registry_dir(db), version)
    except Exception:  # noqa: BLE001 — a missing registry is an unknown report
        entry = None
    if entry is not None and entry.artifact_path:
        try:
            artifact = load_artifact(entry.artifact_path)
        except Exception:  # noqa: BLE001
            artifact = None
        else:
            version = entry.model_version or version
    return read_scored_on(artifact), version


def _rank_like_shadow(db: Any, sit: Any, call: Any, *, mode: CoachingMode, seed: int) -> Any:
    """Build candidates the way ``shadow_after_call`` does, then ``rank_live``."""
    from dataclasses import replace

    from cfb_coach.madden.model.inference import (
        _SITUATION_ADAPTER,
        rank_live,
        resolve_model_version,
        resolve_registry_dir,
    )
    from cfb_coach.madden.model.schema import OpponentKind, Possession, candidate_in_book

    side = getattr(call, "side", "offense") or "offense"
    poss = Possession.DEFENSE if str(side).startswith("d") else Possession.OFFENSE
    formations = _installed_books(db).get(side) or {}
    game_id = f"eval-{int(seed)}"
    snap_id = f"eval-{int(seed)}-{side}"
    state, obs = _SITUATION_ADAPTER(
        sit,
        game_id=game_id,
        snap_id=snap_id,
        session_id=game_id,
        opponent_id=OPPONENT_ID,
        opponent_type=OpponentKind.HUMAN,
        madden_version="madden27",
    )
    if state.possession is Possession.UNKNOWN:
        state = replace(state, possession=poss)
    candidates = []
    for formation, plays in formations.items():
        for play in plays:
            try:
                candidates.append(
                    candidate_in_book(formations, side=poss, formation=formation, play=play)
                )
            except ValueError:
                continue
    heuristic = None
    if getattr(call, "formation", None) and getattr(call, "play", None):
        try:
            heuristic = candidate_in_book(
                formations,
                side=poss,
                formation=call.formation,
                play=call.play,
            )
        except ValueError:
            heuristic = None
    return rank_live(
        state,
        obs,
        candidates,
        None,
        None,
        (),
        mode=mode,
        model_version=resolve_model_version(db),
        registry_dir=resolve_registry_dir(db),
        heuristic_pick=heuristic,
        session_id=game_id,
        snap_seq=int(seed),
    )


def score_model(db: Any, sit: Any, call: Any, *, mode: CoachingMode, seed: int) -> Any:
    """Model portion of the live path. Tests replace this to force a cutoff.

    Shadow uses :func:`evaluate_live_shadow`, the entry ``call`` / ``play``
    share. Hybrid uses :func:`rank_live` with that same candidate list so the
    ``scored_on`` guard stays the one in inference.
    """
    if mode is CoachingMode.HEURISTIC:
        return None
    if mode is CoachingMode.SHADOW:
        from cfb_coach.madden.model.inference import evaluate_live_shadow

        side = getattr(call, "side", "offense") or "offense"
        game_id = f"eval-{int(seed)}"
        decision, _row_id = evaluate_live_shadow(
            db=db,
            situation=sit,
            call=call,
            opponent_id=OPPONENT_ID,
            game_id=game_id,
            snap_id=f"eval-{int(seed)}-{side}",
            snap_seq=int(seed),
            session_id=game_id,
            formations=_installed_books(db).get(side) or {},
            run=True,
        )
        return decision
    if mode is CoachingMode.HYBRID:
        return _rank_like_shadow(db, sit, call, mode=CoachingMode.HYBRID, seed=seed)
    return None


def _call_model(
    db: Any, sit: Any, call: Any, mode: CoachingMode, seed: int,
) -> tuple[Any, float, bool, MLStatus | None]:
    """Time ``score_model`` on the live cutoff clock.

    A manual clock that steps past the cutoff raises :class:`BudgetCut`.
    The model result is dropped and the caller keeps the heuristic call.
    The recorded model time stays inside ``ml.latency_budget_ms``.
    """
    budget = float(ML_LATENCY_BUDGET_MS)
    tick = _active_clock()
    if not tick.realtime():
        wait_ms = max(0.0, budget - _CUTOFF_MARGIN_MS)
        tick.arm(wait_ms)
        started = tick.now_ms()
        try:
            decision = score_model(db, sit, call, mode=mode, seed=seed)
        except BudgetCut:
            elapsed = min(wait_ms, max(0.0, tick.now_ms() - started))
            return None, elapsed, True, MLStatus.TIMEOUT
        except Exception:  # noqa: BLE001 — a model error is a fallback
            elapsed = min(wait_ms, max(0.0, tick.now_ms() - started))
            return None, elapsed, False, MLStatus.EXCEPTION
        elapsed = max(0.0, tick.now_ms() - started)
        if tick.cut or elapsed > wait_ms:
            return None, min(elapsed, wait_ms), True, MLStatus.TIMEOUT
        return decision, elapsed, False, None

    started = time.perf_counter()
    try:
        decision = score_model(db, sit, call, mode=mode, seed=seed)
    except BudgetCut:
        elapsed = (time.perf_counter() - started) * 1000.0
        return None, elapsed, True, MLStatus.TIMEOUT
    except Exception:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        return None, elapsed, False, MLStatus.EXCEPTION
    elapsed = (time.perf_counter() - started) * 1000.0
    model_ms = elapsed
    if decision is not None and getattr(decision, "latency_ms_model", None) is not None:
        model_ms = float(decision.latency_ms_model)
    if model_ms > budget:
        return None, model_ms, True, MLStatus.TIMEOUT
    return decision, model_ms, False, None


@contextmanager
def _temp_env() -> Iterator[tuple[Path, Path, Path]]:
    """Point the two coach databases at a temp dir. Does not change ``HOME``."""
    folder = tempfile.TemporaryDirectory(prefix="madden-eval-harness-")
    root = Path(folder.name)
    # Distinct files, and not the sibling name ``madden27.db``, so the live
    # path does not warn that the two database variables name one file.
    madden = root / "madden" / "live.db"
    coach = root / "coach" / "coach.db"
    madden.parent.mkdir()
    coach.parent.mkdir()
    keys = ("CFB_COACH_DB", "CFB_COACH_MADDEN_DB")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["CFB_COACH_DB"] = str(coach)
    os.environ["CFB_COACH_MADDEN_DB"] = str(madden)
    try:
        yield root, madden, coach
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        folder.cleanup()


def _open_temp_dbs(madden: Path, coach: Path) -> Any:
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.cli import open_db
    from cfb_coach.madden.data import load_seed

    # The coach file is the temp CFB_COACH_DB. Live rows go to the Madden file.
    held = CoachDB(coach, seed=load_seed())
    held.close()
    return open_db(str(madden))


def _evaluate(db: Any, sit: Any, *, mode: CoachingMode, side: str, seed: int) -> dict[str, Any]:
    from cfb_coach.madden.cli import _active_after_prep, _maybe_shadow
    from cfb_coach.madden.franchise import get_session_profile
    from cfb_coach.madden.model.inference import set_mode

    active = _active_after_prep(db, OPPONENT_ID, get_session_profile(db))
    books = _installed_books(db)
    bound_look = (getattr(sit, "extras", None) or {}).get("opp_look")
    raw = getattr(sit, "raw", "") or ""

    set_mode(db, CoachingMode.HEURISTIC)
    old_call = _make_call(copy.deepcopy(sit), db, active, seed)

    set_mode(db, mode)
    started = time.perf_counter()
    new_sit = copy.deepcopy(sit)
    new_call = _make_call(new_sit, db, active, seed)
    if mode is CoachingMode.HEURISTIC:
        _maybe_shadow(db, new_sit, new_call, OPPONENT_ID)
        decision, model_ms, over_budget, forced = None, 0.0, False, None
    else:
        decision, model_ms, over_budget, forced = _call_model(db, new_sit, new_call, mode, seed)
    whole_ms = (time.perf_counter() - started) * 1000.0

    shown = new_call
    fallback = _status_value(forced)
    downgrade = getattr(decision, "mode_downgrade_reason", None) if decision is not None else None
    propensity = getattr(decision, "behavior_propensity", None) if decision is not None else None
    if decision is not None and fallback is None:
        if getattr(decision, "fell_back", False):
            fallback = _status_value(getattr(decision, "fallback_reason", None)) or MLStatus.UNKNOWN.value
        else:
            fallback = _status_value(getattr(decision, "shadow_status", None))
    if over_budget:
        fallback = MLStatus.TIMEOUT.value
        shown = new_call
        decision_for_propensity = None
    else:
        decision_for_propensity = decision

    scored_on, artifact_version = _model_report(db)
    model_version = None
    if decision_for_propensity is not None:
        model_version = getattr(decision_for_propensity, "model_version", None) or artifact_version
        if propensity is None:
            propensity = getattr(decision_for_propensity, "behavior_propensity", None)
    else:
        model_version = artifact_version
        propensity = None

    side_book = books.get(getattr(shown, "side", side)) or {}
    return {
        "seed": int(seed),
        "side": side,
        "situation": raw,
        "opp_look": bound_look,
        "new": _call_record(shown, mode=mode.value),
        "old": _call_record(old_call, mode=CoachingMode.HEURISTIC.value),
        "fallback_reason": fallback,
        "model_version": model_version,
        "scored_on": scored_on,
        "latency_ms": {"whole_call": whole_ms, "model": float(model_ms)},
        "budget_ms": ML_LATENCY_BUDGET_MS,
        "over_budget": bool(over_budget),
        "legal": _legal(shown, side_book),
        "propensity": propensity,
        "mode_downgrade_reason": downgrade,
    }


def run_situation(
    sit: Any,
    *,
    mode: CoachingMode | str,
    opp_look: Any,
    side: str,
    seed: int,
) -> dict[str, Any]:
    """Run one situation through the live path and the heuristic caller.

    ``opp_look`` blank (``None``, ``""``, or whitespace) is stored as SQL
    ``NULL`` / JSON ``null``. The token ``empty`` stays ``empty``.
    """
    requested = _mode(mode)
    used_side = _side(side)
    with _temp_env() as (_root, madden, coach):
        db = _open_temp_dbs(madden, coach)
        try:
            prepare_session(db)
            parsed = _parse(sit, used_side, opp_look)
            return _evaluate(db, parsed, mode=requested, side=used_side, seed=int(seed))
        finally:
            db.close()


def _blank_to_none(value: Any) -> Any:
    """Blank personnel or look text is missing. Other values stay as given."""
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _typed(value: Any) -> dict[str, Any]:
    return {"value": value, "type": type(value).__name__}


def _clock_text(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        seconds = value
        return f"{seconds // 60}:{seconds % 60:02d}"
    text = str(value).strip()
    return text or None


def _situation_text(sit: Any) -> str:
    """Text the live parser understands. Mapping columns stay out of this string."""
    from cfb_coach.situation import Situation

    if isinstance(sit, Situation):
        return sit.raw or ""
    if isinstance(sit, str):
        return sit
    if isinstance(sit, Mapping):
        named = sit.get("situation", sit.get("sit", sit.get("raw")))
        if named not in (None, ""):
            return str(named)
        parts: list[str] = []
        down = sit.get("down")
        distance = sit.get("distance")
        if down not in (None, "") and distance not in (None, ""):
            parts.append(f"{down}&{distance}")
        yardline = sit.get("yardline")
        if yardline not in (None, ""):
            parts.append(f"own {yardline}")
        return " ".join(parts) or "1&10"
    return str(sit)


def _field(source: Mapping[str, Any] | None, key: str, fallback: Any) -> Any:
    if source is not None and key in source:
        return source[key]
    return fallback


# Columns ``snap_from_row`` already reads. Pressure is not a column: ``_pressure``
# derives it from the look text, which is passed as ``look`` / ``coverage_seen``.
_SNAP_FLAG_COLUMNS = (
    "off_adjustment",
    "off_adj",
    "def_adjustment",
    "def_adj",
    "adjustment",
    "audible",
)


def _snap_flag_values(
    source: Mapping[str, Any] | None,
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy trainer flag columns through unchanged. Do not interpret them."""
    found: dict[str, Any] = {}
    for key in _SNAP_FLAG_COLUMNS:
        if source is not None and key in source:
            found[key] = source[key]
        elif key in extras:
            found[key] = extras[key]
    return found


def _offline_context(
    sit: Any,
    parsed: Any,
    *,
    side_arg: Any,
    used_side: str,
    opp_look: Any,
    bound_look: str | None,
) -> dict[str, Any]:
    """Training row plus the raw inputs, before blank and numeric coercion.

    Down and distance are handed to ``snap_from_row`` unchanged, so a CSV
    string ``\"1\"`` is coerced by the trainer's own ``_int``. Blank personnel
    and look strings become ``None`` on the training row only.

    ``off_adj``, ``def_adj``, and ``pressure`` are not rebuilt here.
    ``snap_from_row`` sets the adjustment flags from ``off_adj``,
    ``off_adjustment``, ``def_adj``, ``def_adjustment``, ``adjustment``, and
    ``audible``, and sets pressure from the look text. Those columns are
    copied onto the training row when the caller supplied them. Which side
    an ``adjustment`` belongs to stays the trainer's ``_side_flag`` rule.
    """
    from cfb_coach.situation import Situation

    source = sit if isinstance(sit, Mapping) and not isinstance(sit, Situation) else None
    extras = getattr(parsed, "extras", None) or {}
    down = _field(source, "down", getattr(parsed, "down", None))
    distance = _field(source, "distance", getattr(parsed, "distance", None))
    yardline = _field(source, "yardline", getattr(parsed, "yardline", None))
    score_us = _field(source, "score_us", getattr(parsed, "score_us", None))
    score_them = _field(source, "score_them", getattr(parsed, "score_them", None))
    quarter = _field(source, "quarter", extras.get("quarter"))
    clock = _field(source, "clock", extras.get("clock_seconds"))
    personnel_raw = _field(source, "personnel", extras.get("personnel"))
    look_raw = opp_look
    if source is not None and opp_look is None:
        if "opp_look" in source:
            look_raw = source["opp_look"]
        elif "look" in source:
            look_raw = source["look"]
    situation = sit if isinstance(sit, str) else (getattr(parsed, "raw", None) or _situation_text(sit))
    if look_raw is opp_look:
        look = bound_look
    else:
        look = _blank_to_none(look_raw)
        if isinstance(look, str):
            stripped = look.strip()
            look = stripped if stripped else None
    personnel = _blank_to_none(personnel_raw)
    text = getattr(parsed, "raw", None) or _situation_text(sit)
    clock_text = _clock_text(clock)
    if quarter not in (None, "") or clock_text:
        suffix = []
        if quarter not in (None, ""):
            suffix.append(f"q{quarter}")
        if clock_text:
            suffix.append(clock_text)
        text = f"{text} {' '.join(suffix)}".strip()
    row: dict[str, Any] = {
        "game": "madden27",
        "opponent_type": "human",
        "side": used_side,
        "down": down,
        "distance": distance,
        "yardline": yardline,
        "score_us": score_us,
        "score_them": score_them,
        "situation_raw": text,
        "success": None,
    }
    if quarter not in (None, ""):
        row["quarter"] = quarter
    if clock_text:
        row["clock"] = clock_text
    touched = _field(source, "yardline_touched", extras.get("yardline_touched"))
    if isinstance(touched, bool):
        row["yardline_touched"] = touched
    if look is not None:
        row["look"] = look
        row["coverage_seen"] = look
    flags = _snap_flag_values(source, extras)
    for key, value in flags.items():
        if _blank_to_none(value) is None:
            continue
        row[key] = value
    if source is not None and "zone" in source and _blank_to_none(source.get("zone")) is not None:
        row["zone"] = source["zone"]
    if source is not None and "dist" in source and _blank_to_none(source.get("dist")) is not None:
        row["dist"] = source["dist"]
    raw = {
        "situation": _typed(situation),
        "down": _typed(down),
        "distance": _typed(distance),
        "yardline": _typed(yardline),
        "score_us": _typed(score_us),
        "score_them": _typed(score_them),
        "quarter": _typed(quarter),
        "clock": _typed(clock),
        "look": _typed(look_raw),
        "personnel": _typed(personnel_raw),
        "side": _typed(side_arg),
    }
    for key, value in flags.items():
        raw[key] = _typed(value)
    coerced = {"look": look, "personnel": personnel}
    return {"row": row, "raw": raw, "coerced": coerced}


def _live_feature_kwargs(row: Mapping[str, Any]) -> dict[str, Any]:
    """The situation fields the live path supplies. No def_call, no coverage_seen."""
    return {
        "down": row.get("down"),
        "distance": row.get("distance"),
        "yardline": row.get("yardline"),
        "yardline_touched": row.get("yardline_touched"),
        "quarter": row.get("quarter"),
        "clock": row.get("clock"),
    }


def _candidate_feature_map(row: dict[str, Any], *, formation: str, play: str, side: str) -> dict[str, str]:
    """Offense candidate. Same function inference.py uses live (shared feature module)."""
    from cfb_coach.madden.model.inference import vod_candidate_features

    del formation  # not a model input
    return vod_candidate_features(
        side, play, opp_look=row.get("look"), **_live_feature_kwargs(row),
    )


def _defense_feature_map(row: dict[str, Any], *, formation: str, play: str, side: str) -> dict[str, str]:
    """Defensive candidate as the look. The offense call is not a defense input.

    Defense models estimate P(offense succeeds | our defensive call, situation).
    The family, look and pressure come from our own candidate play; the typed
    opponent look and any def_call/coverage_seen column are never read.
    """
    from cfb_coach.madden.model.inference import vod_candidate_features

    del formation  # not a model input
    return vod_candidate_features(side, play, **_live_feature_kwargs(row))


def _candidate_logit(
    models: Sequence[dict[str, Any]],
    feats: Mapping[str, str],
) -> tuple[float, dict[str, Any]]:
    """Intercept plus one weight per training group.

    The key is the one ``feature_map`` built (``call_x_family`` is
    ``{call}|{family}``, ``down_x_dist`` is ``{down}|{dist}``). A group the
    builder omitted, or a key absent from ``coef``, contributes 0.
    """
    from cfb_coach.vod_success.shared_features import ALL_GROUPS

    intercepts = [float(model.get("intercept") or 0.0) for model in models]
    intercept = sum(intercepts) / float(len(intercepts))
    groups: dict[str, Any] = {}
    total = intercept
    for group in ALL_GROUPS:
        key = feats.get(group)
        weights: list[float] = []
        matched = False
        for model in models:
            coef = model.get("coef") if isinstance(model.get("coef"), dict) else {}
            table = coef.get(group) if isinstance(coef, dict) and isinstance(coef.get(group), dict) else {}
            if key is not None and key in table:
                matched = True
                weights.append(float(table[key]))
            else:
                weights.append(0.0)
        weight = sum(weights) / float(len(weights))
        groups[group] = {"key": key, "matched": matched, "weight": weight}
        total += weight
    return total, {"intercept": intercept, "groups": groups}


def _stratum_model(entry: Any, name: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"refused stratum {name}")
    model = entry.get("model")
    if isinstance(model, dict) and ("intercept" in model or "coef" in model):
        return model
    if "intercept" in entry or "coef" in entry:
        return entry
    raise ValueError(f"refused stratum {name}")


def _load_offline_files(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    folder = Path(model_dir)
    params_path = folder / "model_params.json"
    manifest_path = folder / "manifest.json"
    if not params_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"offline model dir needs model_params.json and manifest.json: {folder}"
        )
    params = json.loads(params_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(params, dict) or not isinstance(manifest, dict):
        raise ValueError(f"offline model files must be JSON objects: {folder}")
    return params, manifest


def _available_strata(params: dict[str, Any]) -> dict[str, Any]:
    raw = params.get("strata")
    if not isinstance(raw, dict):
        raise ValueError("model_params.json has no strata object")
    return raw


def _allowed_stratum(name: str) -> bool:
    return _MADDEN_STRATUM.fullmatch(name) is not None


def _expand_strata(strata: str | Sequence[str], available: dict[str, Any]) -> list[str]:
    """Keep only madden27/cpu and madden27/human strata. Anything else is refused."""
    if isinstance(strata, str):
        requested: list[str] = [strata]
    elif isinstance(strata, Sequence):
        requested = [str(item) for item in strata]
    else:
        raise ValueError(f"refused stratum {strata!r}")
    if not requested:
        raise ValueError("missing stratum")
    chosen: list[str] = []
    seen: set[str] = set()
    names = set(available)
    for item in requested:
        text = item.strip()
        if not text:
            raise ValueError("refused stratum ")
        if any(mark in text for mark in "*?["):
            matches = sorted(name for name in names if fnmatch.fnmatchcase(name, text))
            if not matches:
                raise ValueError(f"missing stratum {text}")
            for name in matches:
                if not _allowed_stratum(name):
                    raise ValueError(f"refused stratum {name}")
                if name not in seen:
                    seen.add(name)
                    chosen.append(name)
            continue
        if not _allowed_stratum(text):
            raise ValueError(f"refused stratum {text}")
        if text not in names:
            raise ValueError(f"missing stratum {text}")
        if text not in seen:
            seen.add(text)
            chosen.append(text)
    return chosen


def _offline_report(manifest: dict[str, Any]) -> tuple[str | None, str]:
    from cfb_coach.madden.model.inference import read_scored_on

    version = manifest.get("model_version")
    if version in (None, ""):
        version = manifest.get("label_version")
    model_version = None if version in (None, "") else str(version)
    direct = manifest.get("scored_on")
    if direct in ("reviewed", "unreviewed"):
        return model_version, direct
    return model_version, read_scored_on(manifest)


def _rank_defense_book(
    models: Sequence[dict[str, Any]],
    book: dict[str, list[str]],
    *,
    side: str,
    context: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Lowest predicted offense success. The offense path does not use this."""
    ranked: list[tuple[float, str, str, dict[str, Any]]] = []
    for formation, plays in book.items():
        for play in plays:
            feats = _defense_feature_map(
                context["row"], formation=str(formation), play=str(play), side=side,
            )
            score, detail = _candidate_logit(models, feats)
            ranked.append((score, str(formation), str(play), detail))
    if not ranked:
        raise ValueError("installed book has no legal candidates")
    _score, formation, play, detail = min(ranked, key=lambda item: (item[0], item[1], item[2]))
    features = {
        "intercept": detail["intercept"],
        "groups": detail["groups"],
        "raw": context["raw"],
        "coerced": context["coerced"],
        "call_matches": [],
        "ranking": "min_offense_success",
    }
    return formation, play, features


def _rank_installed_book(
    models: Sequence[dict[str, Any]],
    book: dict[str, list[str]],
    *,
    side: str,
    context: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Top legal candidate. Out-of-book coefficient names are never candidates."""
    if side == "defense":
        return _rank_defense_book(models, book, side=side, context=context)
    ranked: list[tuple[float, str, str, dict[str, Any], bool]] = []
    call_matches: list[str] = []
    for formation, plays in book.items():
        for play in plays:
            feats = _candidate_feature_map(
                context["row"], formation=str(formation), play=str(play), side=side,
            )
            score, detail = _candidate_logit(models, feats)
            matched_call = bool(detail["groups"]["call"]["matched"])
            if matched_call:
                call_matches.append(str(play))
            ranked.append((score, str(formation), str(play), detail, matched_call))
    if not ranked:
        raise ValueError("installed book has no legal candidates")
    _score, formation, play, detail, _matched = min(
        ranked, key=lambda item: (-item[0], item[1], item[2]),
    )
    features = {
        "intercept": detail["intercept"],
        "groups": detail["groups"],
        "raw": context["raw"],
        "coerced": context["coerced"],
        "call_matches": call_matches,
    }
    return formation, play, features


def _offline_call(side: str, formation: str, play: str) -> Any:
    from cfb_coach.madden.data import reads_for, user_job_for
    from cfb_coach.madden.playcaller import MaddenCall

    if side == "defense":
        read = user_job_for(play)
    else:
        read = reads_for(play)
    return MaddenCall(side, formation, play, "No adj", read or "")


def score_offline_model(
    sit: Any,
    *,
    side: str,
    model_dir: str | Path,
    strata: str | Sequence[str],
    opp_look: Any = None,
    seed: int = 0,
    offline_ok: bool = False,
) -> dict[str, Any]:
    """Score one situation from a model directory. Eval only.

    The default ``offline_ok=False`` raises before any database or mode
    write. This function does not call ``set_mode``, ``rank_live``, or the
    live shadow/hybrid path, so an unreviewed manifest still returns a pick.
    ``gate_bypassed`` records that the scored_on hybrid gate was not applied.
    Feature keys come from ``cfb_coach.vod_success.features.feature_map``.
    ``features`` lists each group's key, whether that key hit a weight, and
    the weight, plus the raw situation inputs and their Python types.
    """
    if offline_ok is not True:
        raise RuntimeError("offline model scoring requires offline_ok=True")
    started = time.perf_counter()
    params, manifest = _load_offline_files(Path(model_dir))
    available = _available_strata(params)
    used = _expand_strata(strata, available)
    models = [_stratum_model(available[name], name) for name in used]
    model_version, scored_on = _offline_report(manifest)
    used_side = _side(side)
    with _temp_env() as (_root, madden, coach):
        db = _open_temp_dbs(madden, coach)
        try:
            from cfb_coach.madden.cli import _active_after_prep
            from cfb_coach.madden.franchise import get_session_profile
            from cfb_coach.situation import Situation

            prepare_session(db)
            parsed_input: Any = sit if isinstance(sit, Situation) else _situation_text(sit)
            parsed = _parse(parsed_input, used_side, opp_look)
            active = _active_after_prep(db, OPPONENT_ID, get_session_profile(db))
            heuristic = _make_call(copy.deepcopy(parsed), db, active, int(seed))
            book = (_installed_books(db).get(used_side) or {})
            bound_look = (getattr(parsed, "extras", None) or {}).get("opp_look")
            context = _offline_context(
                sit, parsed, side_arg=side, used_side=used_side, opp_look=opp_look, bound_look=bound_look,
            )
            model_started = time.perf_counter()
            formation, play, features = _rank_installed_book(
                models, book, side=used_side, context=context,
            )
            model_ms = (time.perf_counter() - model_started) * 1000.0
            shown = _offline_call(used_side, formation, play)
            whole_ms = (time.perf_counter() - started) * 1000.0
            record = _call_record(shown, mode="offline")
            return {
                **record,
                "model_version": model_version,
                "scored_on": scored_on,
                "strata_used": list(used),
                "latency_ms": {"whole_call": whole_ms, "model": float(model_ms)},
                "budget_ms": ML_LATENCY_BUDGET_MS,
                "over_budget": float(model_ms) > float(ML_LATENCY_BUDGET_MS),
                "legal": _legal(shown, book),
                "gate_bypassed": _OFFLINE_GATE,
                "heuristic": _call_record(heuristic, mode="heuristic"),
                "seed": int(seed),
                "side": used_side,
                "situation": getattr(parsed, "raw", "") or "",
                "opp_look": bound_look,
                "features": features,
            }
        finally:
            db.close()


def _percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def _load_situations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("situations", [])
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"situations file must be a non-empty list: {path}")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, str):
            rows.append({"situation": item})
        elif isinstance(item, dict) and (item.get("situation") or item.get("sit")):
            rows.append(dict(item))
        else:
            raise SystemExit(f"unreadable situation entry: {item!r}")
    return rows


def _decision_signature(row: dict[str, Any]) -> str:
    kept = {
        "new": row["new"],
        "old": row["old"],
        "opp_look": row["opp_look"],
        "legal": row["legal"],
        "fallback_reason": row["fallback_reason"],
        "scored_on": row["scored_on"],
    }
    return json.dumps(kept, sort_keys=True, default=str)


def run_cli(situations: list[dict[str, Any]], *, passes: int, out: Path, defaults: dict[str, Any]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    whole: list[float] = []
    model: list[float] = []
    fallbacks: Counter[str] = Counter()
    reproduced = True
    lines: list[dict[str, Any]] = []
    for spec in situations:
        raw = spec.get("situation", spec.get("sit"))
        mode = spec.get("mode", defaults["mode"])
        opp_look = spec["opp_look"] if "opp_look" in spec else defaults["opp_look"]
        side = spec.get("side", defaults["side"])
        seed = int(spec.get("seed", defaults["seed"]))
        first: dict[str, Any] | None = None
        signature: str | None = None
        situation_matched = True
        for _index in range(passes):
            row = run_situation(raw, mode=mode, opp_look=opp_look, side=side, seed=seed)
            whole.append(float(row["latency_ms"]["whole_call"]))
            model.append(float(row["latency_ms"]["model"]))
            if row["fallback_reason"]:
                fallbacks[str(row["fallback_reason"])] += 1
            current = _decision_signature(row)
            if signature is None:
                signature = current
                first = row
            elif current != signature:
                situation_matched = False
                reproduced = False
        assert first is not None
        first = dict(first)
        first["reproduced"] = situation_matched
        lines.append(first)

    over = [ms for ms in model if ms > ML_LATENCY_BUDGET_MS]
    fail_reasons: list[str] = []
    if over:
        fail_reasons.append(
            f"{len(over)} model call(s) exceeded {ML_LATENCY_BUDGET_MS} ms (max {max(model):.3f})"
        )
    if not reproduced:
        fail_reasons.append("the same seed did not reproduce the decision")
    summary = {
        "budget_ms": ML_LATENCY_BUDGET_MS,
        "situations": len(situations),
        "passes": passes,
        "samples": len(situations) * passes,
        "whole_call_ms": {
            "p50": _percentile(whole, 50),
            "p95": _percentile(whole, 95),
            "max": max(whole) if whole else 0.0,
        },
        "model_ms": {
            "p50": _percentile(model, 50),
            "p95": _percentile(model, 95),
            "max": max(model) if model else 0.0,
        },
        "fallback_counts": {name: fallbacks[name] for name in sorted(fallbacks)},
        "reproduced": reproduced,
        "pass": not fail_reasons,
        "fail_reasons": fail_reasons,
    }
    jsonl = out / "situations.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Madden live eval harness")
    parser.add_argument("--situations", required=True, help="JSON list of situations")
    parser.add_argument("--passes", type=int, default=50, help="repeats per situation (default 50)")
    parser.add_argument("--out", required=True, help="directory for JSONL and the latency summary")
    parser.add_argument("--mode", default="heuristic", choices=("heuristic", "shadow", "hybrid"))
    parser.add_argument("--side", default="offense")
    parser.add_argument("--opp-look", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.passes < 1:
        print("--passes must be >= 1", file=sys.stderr)
        return 2
    try:
        situations = _load_situations(Path(args.situations))
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    return run_cli(
        situations,
        passes=args.passes,
        out=Path(args.out),
        defaults={
            "mode": args.mode,
            "side": args.side,
            "opp_look": args.opp_look,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
