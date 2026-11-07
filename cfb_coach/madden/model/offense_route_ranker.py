"""Selectable route-aware ranker inside the EXISTING live offense path (``apply_experimental_offense``).

Modes (``ml_offense_ranker`` DB meta, or env ``CFB_COACH_OFFENSE_RANKER`` for one session; default unchanged):

* ``experimental``          current ranker. This module does nothing.
* ``route_aware_preview``   the live call is still made by the current ranker. The route-aware model scores the
                            same game state in parallel; its pick, coverage beliefs and reads are shown as a
                            PREVIEW and written to a separate shadow JSONL. No effect on the call.
* ``route_aware``           the route-aware model ranks EVERY installed play (no situational shortlist). The
                            rest of the live path (anti-repeat selection, verified macro policy, rebuild, logging,
                            150 ms budget, fallback) is unchanged. On a technical failure the current ranker's
                            result is used and the reason is logged: never a silent rule-based override.

Coverage fed to the model is the existing post-play form's ``coverage_seen`` from EARLIER closed snaps of this
game and opponent, blended into the historical prior (same formula as training). Nothing here writes the Madden
database; shadow records go to a separate file and are never training labels.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import football_priors as fpri
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_live_coverage as lc
from cfb_coach.madden.model import offense_presnap_advice as advice
from cfb_coach.madden.model import offense_structural_model as sm

META_MODE = "ml_offense_ranker"
META_ARTIFACT = "ml_route_aware_artifact"
ENV_MODE = "CFB_COACH_OFFENSE_RANKER"
ENV_ARTIFACT = "CFB_COACH_ROUTE_AWARE_ARTIFACT"
ENV_CONFIRMATIONS = "CFB_COACH_ROUTE_AWARE_CONFIRMATIONS"
MODES = ("experimental", "route_aware_preview", "route_aware")
SHADOW_SCHEMA = "madden27.route_aware.shadow.v1"
_lock = threading.Lock()
_cache: dict[str, tuple[float, sm.StructuralArtifact]] = {}


class RouteAwareUnavailable(RuntimeError):
    """Artifact missing/stale/slow. The caller falls back to the current ranker and logs this reason."""


def resolve_mode(db: Any = None) -> str:
    raw = os.environ.get(ENV_MODE)
    if not raw and db is not None:
        try:
            raw = db.get_meta(META_MODE)
        except Exception:  # noqa: BLE001
            raw = None
    raw = (raw or "experimental").strip().lower()
    return raw if raw in MODES else "experimental"


def set_mode(db: Any, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    db.set_meta(META_MODE, mode)


ENV_PRIORS = "CFB_COACH_FOOTBALL_PRIORS"
META_PRIORS = "ml_football_priors"


def football_priors_on(db: Any = None) -> bool:
    """Opt-in (default OFF so the running CPU experiment is unchanged): env CFB_COACH_FOOTBALL_PRIORS=1 or db meta ml_football_priors=1."""
    raw = os.environ.get(ENV_PRIORS)
    if raw is None and db is not None:
        try:
            raw = db.get_meta(META_PRIORS)
        except Exception:  # noqa: BLE001
            raw = None
    return str(raw or "").strip().lower() in ("1", "true", "on", "yes")


def default_artifact_path() -> Path:
    from cfb_coach.games import data_dir

    return Path(data_dir()) / "madden_ml_experimental" / "offense_route_aware.json"


def resolve_artifact_path(db: Any = None) -> Path | None:
    for raw in (os.environ.get(ENV_ARTIFACT), _meta(db, META_ARTIFACT)):
        if raw and Path(raw).expanduser().is_file():
            return Path(raw).expanduser()
    p = default_artifact_path()
    return p if p.is_file() else None


def shadow_log_path() -> Path:
    from cfb_coach.games import data_dir

    return Path(data_dir()) / "madden_ml_experimental" / "route_aware_shadow.jsonl"


def _meta(db: Any, key: str) -> str | None:
    if db is None:
        return None
    try:
        return db.get_meta(key)
    except Exception:  # noqa: BLE001
        return None


def load_artifact(path: Path) -> sm.StructuralArtifact:
    mtime = path.stat().st_mtime
    with _lock:
        hit = _cache.get(str(path))
        if hit and hit[0] == mtime:
            return hit[1]
    art = sm.load_artifact(path)
    errs = sm.validate_artifact(art)
    if errs:
        raise RouteAwareUnavailable("artifact invalid: " + "; ".join(errs[:3]))
    with _lock:
        _cache[str(path)] = (mtime, art)
    return art


@dataclass
class RouteContext:
    mode: str
    art: sm.StructuralArtifact | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)       # route-aware ranking of every scored play
    base_top: list[dict[str, Any]] = field(default_factory=list)   # current ranker's top rows (same state)
    coverage: dict[str, Any] = field(default_factory=dict)
    mix: dict[str, float] = field(default_factory=dict)
    applied: bool = False
    priors: bool = False
    learned: dict[tuple[str, str], float] | None = None
    adaptive: dict[str, Any] | None = None
    exploration: dict[str, Any] | None = None
    error: str = ""
    n_scored: int = 0
    latency_ms: float = 0.0


def _coverage(db: Any, art: sm.StructuralArtifact, opponent_id: str, session_id: str | None,
              snap_seq: int | None) -> tuple[dict[str, float], dict[str, Any]]:
    prior, prior_src = sm.pre_snap_mix(art, opponent_id)
    if db is None or not session_id:
        return lc.blend_counts({}, prior), {"basis": "historical_prior_only", "prior_source": prior_src,
                                            "history_n": 0, "named_n": 0, "pressure_n": 0, "unknown_n": 0}
    ctx = lc.coverage_for_next_snap(db.conn, session_id=session_id, opponent_id=opponent_id,
                                    next_snap_seq=snap_seq if snap_seq and snap_seq > 0 else None, prior_mix=prior,
                                    knowledge=art.uses_defense_knowledge)
    out = ctx.as_log()
    out["prior_source"] = prior_src
    return ctx.mix, out


def apply(mode: str, ranked: list[dict[str, Any]], pairs: list[tuple[str, str]],
          all_pairs: Sequence[tuple[str, str]], *, sit: Any, opponent_id: str, db: Any,
          session_id: str | None, snap_seq: int | None, rank_kwargs: Mapping[str, Any],
          started: float, budget_ms: float) -> tuple[list[dict[str, Any]], list[tuple[str, str]], RouteContext]:
    """Return (ranked, pairs, ctx). ``route_aware_preview`` and any failure return the inputs unchanged."""
    ctx = RouteContext(mode=mode, base_top=[_brief(r) for r in ranked[:5]])
    t0 = time.perf_counter()
    try:
        path = resolve_artifact_path(db)
        if path is None:
            raise RouteAwareUnavailable("no route-aware artifact installed")
        art = load_artifact(path)
        ctx.art = art
        mix, cov = _coverage(db, art, opponent_id, session_id, snap_seq)
        ctx.mix, ctx.coverage = mix, cov
        scored_pairs = list(all_pairs)      # EVERY installed play is scored; no situational shortlist
        ctx.priors = football_priors_on(db)
        if ctx.priors:
            from cfb_coach.madden.model import adaptive_priors as ap
            ctx.learned, ctx.adaptive = ap.learned_cells(db, opponent_id, session_id)
        template = em.rank_candidates(em.ExperimentalArtifact.from_dict(art.base), scored_pairs,
                                      **{k: v for k, v in rank_kwargs.items() if k != "heuristic"})
        rows = []
        for r in template:
            s = sm.structure_any(r["formation"], r["play"])
            sc = sm.score(art, s, down=getattr(sit, "down", None), distance=getattr(sit, "distance", None),
                          yardline=getattr(sit, "yardline", None), opponent_id=opponent_id,
                          opponent_type=rank_kwargs.get("opponent_type"), mix=mix, base_p=float(r["probability"]))
            row = dict(r)
            row["components"] = {**dict(r.get("components") or {}), "route_aware_residual_logit": sc.residual_logit}
            row["base_probability"] = float(r["probability"])
            row["probability"] = sc.probability
            if ctx.priors:                                   # structured football prior (+ learned cells): changes the SCORE, hence the ranking
                e = fpri.matchup_edge(s, mix, learned=ctx.learned)
                row["football_prior_edge"] = e["edge"]
                row["probability"] = sm._sigmoid(sm._logit(sc.probability) + e["edge"])
                row["components"]["football_prior_edge_logit"] = e["edge"]
            row["route_aware_se"] = sc.logit_se
            row["route_aware_known_routes"] = s.n_known
            rows.append(row)
            if (time.perf_counter() - started) * 1000.0 > budget_ms:
                raise RouteAwareUnavailable("latency budget exhausted while scoring")
        rows.sort(key=lambda x: (-x["probability"], x.get("uncertainty", 1.0), x["formation"], x["play"]))
        if ctx.priors and os.environ.get("CFB_COACH_EXPLORE") == "1":     # optional, cautious, logged; off by default
            from cfb_coach.madden.model import adaptive_priors as ap
            ctx.exploration = ap.exploration_choice(rows, situation={"down": getattr(sit, "down", None), "yardline": getattr(sit, "yardline", None)})
            if ctx.exploration["explored"]:
                f, p_ = ctx.exploration["chosen"]
                pick = next(r for r in rows if (r["formation"], r["play"]) == (f, p_))
                rows.remove(pick)
                rows.insert(0, pick)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        ctx.rows, ctx.n_scored = rows, len(rows)
        if mode == "route_aware":
            ctx.applied = True
            ctx.latency_ms = (time.perf_counter() - t0) * 1000.0
            return rows, scored_pairs, ctx
    except Exception as exc:  # noqa: BLE001 - a technical failure: keep the current ranker, say why
        ctx.error = f"{type(exc).__name__}: {exc}"
    ctx.latency_ms = (time.perf_counter() - t0) * 1000.0
    return ranked, pairs, ctx


def _brief(r: Mapping[str, Any]) -> dict[str, Any]:
    return {"formation": r["formation"], "play": r["play"], "p": round(float(r["probability"]), 4)}


def _confirmations() -> dict[str, Any] | None:
    p = os.environ.get(ENV_CONFIRMATIONS)
    if p and Path(p).expanduser().is_file():
        return json.loads(Path(p).expanduser().read_text(encoding="utf-8"))
    return None


def finalize(ctx: RouteContext, call: Any, *, sit: Any, opponent_id: str, game_id: str | None,
             snap_seq: int | None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Display payload + shadow record for the FINAL call. Always JSON-serializable; never raises."""
    info: dict[str, Any] = {"mode": ctx.mode, "applied": ctx.applied, "fallback_reason": ctx.error,
                            "n_scored": ctx.n_scored, "latency_ms": round(ctx.latency_ms, 2)}
    try:
        top = [_brief(r) for r in ctx.rows[:5]]
        info.update(
            route_aware_top=top, current_ranker_top=ctx.base_top,
            agree=bool(top and ctx.base_top and (top[0]["formation"], top[0]["play"]) ==
                       (ctx.base_top[0]["formation"], ctx.base_top[0]["play"])),
            coverage={"top": sorted(((k, round(v, 3)) for k, v in ctx.mix.items()), key=lambda kv: -kv[1])[:4],
                      **{k: v for k, v in ctx.coverage.items() if k not in ("mix", "recent_named")},
                      "recent_named": list(ctx.coverage.get("recent_named") or [])},
            artifact_digest=sm.artifact_digest(ctx.art) if ctx.art else "")
        if ctx.art is not None and getattr(call, "formation", None) and getattr(call, "play", None):
            cand = (call.formation, call.play)
            conf = _confirmations()
            situation = od.Situation(down=getattr(sit, "down", None), distance=getattr(sit, "distance", None),
                                     yardline=getattr(sit, "yardline", None), opponent_id=opponent_id,
                                     opponent_type="cpu" if str(opponent_id).lower().startswith("cpu") else "human",
                                     mix=ctx.mix)
            dec = od.decide(ctx.art, situation, [cand], confirmations=conf,
                            mode="live", allow_adjustments=True, priors=ctx.priors, learned_cells=ctx.learned)
            card = advice.build_card(dec, ctx.mix)
            if ctx.exploration:
                info["exploration"] = ctx.exploration
            if ctx.priors and dec.matchup:
                info["football_priors"] = {"edge": round(dec.matchup["edge"], 4), "cells": dec.matchup["cells"], "adaptive": ctx.adaptive}
            info["card"] = card
            info["adjustment"] = dec.adjustment
            info["call_probability"] = round(dec.probability, 4)
        info.update(extra or {})
        _write_shadow({"schema": SHADOW_SCHEMA, "ts": datetime.now(timezone.utc).isoformat(), "game_id": game_id,
                       "snap_seq": snap_seq, "opponent_id": opponent_id,
                       "situation": {k: getattr(sit, k, None) for k in ("down", "distance", "yardline")},
                       "final_call": [getattr(call, "formation", None), getattr(call, "play", None)],
                       "kind": "shadow_decision_not_an_outcome", **info})
    except Exception as exc:  # noqa: BLE001
        info["display_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _write_shadow(rec: Mapping[str, Any]) -> None:
    try:
        p = shadow_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    except OSError:
        pass          # a log failure must never change the call


def format_lines(info: Mapping[str, Any] | None) -> list[str]:
    """Human text for the existing live window / terminal. Empty when the ranker is not in use."""
    if not info:
        return []
    tag = "ROUTE-AWARE" if info.get("applied") else "ROUTE-AWARE PREVIEW (not the call)"
    lines = []
    if info.get("fallback_reason"):
        return [f"  {tag}: unavailable, current ranker used ({info['fallback_reason']})"]
    top = info.get("route_aware_top") or []
    if top:
        lines.append(f"  {tag}: {top[0]['formation']}/{top[0]['play']} (p={top[0]['p']:.3f})"
                     + ("  = same as current ranker" if info.get("agree") else
                        f"  vs current ranker {info['current_ranker_top'][0]['formation']}/{info['current_ranker_top'][0]['play']}"
                        if info.get("current_ranker_top") else ""))
    cov = info.get("coverage") or {}
    if cov.get("top"):
        n = cov.get("named_n", 0)
        lines.append("  expected coverage: " + ", ".join(f"{k} {v:.0%}" for k, v in cov["top"][:3])
                     + f"  [{n} coverage entr{'y' if n == 1 else 'ies'} this game]")
    card = info.get("card") or {}
    if card:
        lines.append(f"  adjustment: {card.get('ADJUSTMENT')}")
        lines.append(f"  primary read: {card.get('PRIMARY READ')}")
        lines.append(f"  secondary read: {card.get('SECONDARY READ')}")
        if card.get("CHECKDOWN") and not str(card["CHECKDOWN"]).startswith("UNKNOWN"):
            lines.append(f"  checkdown: {card.get('CHECKDOWN')}")
        lines.append(f"  matchup: {card.get('WHY')}")
    return lines


def info_of(call: Any) -> dict[str, Any] | None:
    """Route-aware display payload attached to a call (experimental path or heuristic-mode preview)."""
    info = getattr(call, "ml_experimental", None)
    if isinstance(info, dict) and isinstance(info.get("route_aware"), dict):
        return info["route_aware"]
    prev = getattr(call, "route_aware_preview", None)
    return prev if isinstance(prev, dict) else None


def preview_for_call(call: Any, *, sit: Any, opponent_id: str, db: Any, book: Mapping[str, Sequence[str]],
                     session_id: str | None, snap_seq: int | None, budget_ms: float = 100.0) -> None:
    """Heuristic/other mode with ``route_aware_preview`` selected: score the same state, attach lines, change NOTHING.

    Never raises and never touches the call's formation/play/macro. CPU opponents only.
    """
    started = time.perf_counter()
    try:
        from cfb_coach.madden.model.offense_inventory import pairs_in_inventory
        from cfb_coach.opponents import is_cpu_opponent

        if not is_cpu_opponent(opponent_id) or str(getattr(call, "side", "offense")).startswith("d"):
            return
        all_pairs = list(pairs_in_inventory(dict(book)))
        base = [{"formation": call.formation, "play": call.play, "probability": float("nan")}]
        rank_kwargs = dict(down=getattr(sit, "down", None), distance=getattr(sit, "distance", None),
                           yardline=getattr(sit, "yardline", None), opponent_id=opponent_id, opponent_type="cpu",
                           heuristic=None, heuristic_bonuses={}, near_tie_margin=0.0)
        _, _, ctx = apply("route_aware_preview", base, [], all_pairs, sit=sit, opponent_id=opponent_id, db=db,
                          session_id=session_id, snap_seq=snap_seq, rank_kwargs=rank_kwargs, started=started,
                          budget_ms=budget_ms)
        ctx.base_top = [{"formation": call.formation, "play": call.play, "p": None}]
        info = finalize(ctx, call, sit=sit, opponent_id=opponent_id, game_id=session_id, snap_seq=snap_seq,
                        extra={"current_ranker": "heuristic"})
        call.route_aware_preview = info
    except Exception:  # noqa: BLE001 - a preview can never break live coaching
        pass
