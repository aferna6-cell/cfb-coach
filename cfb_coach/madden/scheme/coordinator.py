"""civil_scheme_only offensive coordinator: picks ONLY from the 26 catalog variants, every time.

Order per snap (never calls the heuristic ``_pick_offense`` or the experimental offense path):
  1. legal filter    ``legality.legal_set``: catalog + Bucs stock + applied book + owner-verified
  2. belief          opponent coverage / pressure belief (earlier games + this game's closed snaps, recency weighted)
  3. score           existing-model offset + learned head + regularized adjustment transfer + expert/beater terms
                     + per-variant learned residual from verified variant outcomes - repetition penalty
  4. pick            highest utility; re-validated by ``legality.revalidate_call`` before display
  5. nothing legal   NO_LEGAL_SCHEME_PLAY with the reasons; no fallback to any other play
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.scheme import beliefs as bl
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme import evidence as ev
from cfb_coach.madden.scheme import features as ft
from cfb_coach.madden.scheme import learners as ln
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import mode as md
from cfb_coach.madden.scheme import model as sm
from cfb_coach.madden.scheme import priors as pr

REPEAT_W = 0.25            # logit penalty per recent same-game call of the same variant
REPEAT_DECAY = 0.6         # per later call
RESIDUAL_K = 10.0
RESIDUAL_CAP = 0.6
LATENCY_TARGET_MS = 150.0


@dataclass
class Scored:
    variant: sc.SchemeVariant
    p_success: float
    sd: float
    p_first_down: float
    p_explosive: float
    p_negative: float
    exp_yards: float | None
    utility: float
    parts: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"variant_id": self.variant.variant_id, "p_success": round(self.p_success, 4),
                "sd_logit": round(self.sd, 4), "p_first_down": round(self.p_first_down, 4),
                "p_explosive": round(self.p_explosive, 4), "p_negative": round(self.p_negative, 4),
                "exp_yards": None if self.exp_yards is None else round(self.exp_yards, 2),
                "utility": round(self.utility, 4), "parts": self.parts}


class Scorer:
    """Loaded once per artifact; per-snap scoring is pure arithmetic."""

    def __init__(self, model: sm.SchemeModel, cat: sc.SchemeCatalog, geo: dz.GeometrySource):
        self.model, self.cat, self.geo = model, cat, geo
        self.adjusted = {v.variant_id: dz.adjusted_design(v, geo) for v in cat.variants}
        self.stock = {v.pair: dz.stock_design(v.formation, v.base_play, geo) for v in cat.variants}
        self.rf_adj = {k: dz.route_features(d) for k, d in self.adjusted.items()}
        self.rf_stock = {k: dz.route_features(d) for k, d in self.stock.items()}

    def offset_p(self, v: sc.SchemeVariant, *, down, distance, yardline, opponent_id, opponent_type) -> float:
        if self.model.existing is None:
            raise RuntimeError("artifact has no existing-model offset (existing_offense.json): retrain")
        return float(em.predict_success(self.model.existing, formation=v.formation, play=v.base_play, down=down,
                                        distance=distance, yardline=yardline, opponent_id=opponent_id,
                                        opponent_type=opponent_type)["probability"])

    def score(self, variants: Sequence[sc.SchemeVariant], *, down, distance, yardline, opponent_id: str,
              opponent_type: str, belief: bl.Belief, recent_calls: Sequence[str] = (),
              variant_stats: Mapping[str, tuple[float, float]] | None = None,
              offset_override: Mapping[str, float] | None = None, stock_only: bool = False) -> list[Scored]:
        """``stock_only`` scores every variant as its unadjusted base play (evaluation diagnostics only)."""
        variant_stats = variant_stats or {}
        counts = {k: n for k, (_s, n) in variant_stats.items()}
        expert = pr.bonuses(variants, down=down, distance=distance, yardline=yardline, mix=belief.mix,
                            families=belief.families, info=belief.gate_confidence, verified_counts=counts)
        out: list[Scored] = []
        base_cache: dict[tuple[str, str], float] = {}
        for v in variants:
            if offset_override and v.variant_id in offset_override:
                op = offset_override[v.variant_id]
            else:
                if v.pair not in base_cache:
                    base_cache[v.pair] = self.offset_p(v, down=down, distance=distance, yardline=yardline,
                                                       opponent_id=opponent_id, opponent_type=opponent_type)
                op = base_cache[v.pair]
            xs, _ = ft.vector(down=down, distance=distance, yardline=yardline, belief=belief,
                              design=self.stock[v.pair], opponent_type=opponent_type, route_f=self.rf_stock[v.pair])
            xa, _ = ft.vector(down=down, distance=distance, yardline=yardline, belief=belief,
                              design=self.adjusted[v.variant_id], opponent_type=opponent_type,
                              route_f=self.rf_adj[v.variant_id])
            if stock_only:
                xa = xs
            ps, pa = self.model.predict(xs, op), self.model.predict(xa, op)
            d = self.adjusted[v.variant_id]
            extra_var = (len(d.unrepresented) * sm.UNREPRESENTED_SD) ** 2
            if d.geometry_status in ("name_only", "unknown"):
                extra_var += sm.NAME_ONLY_SD ** 2
            oos = self.model.out_of_support(xa)
            zs: dict[str, tuple[float, float]] = {}
            for t in sm.TARGETS:
                z_s, sd_s, _ = ps[t]
                z_a, sd_a, _ = pa[t]
                delta = z_a - z_s
                z = z_s + sm.VARIANT_TRANSFER * delta
                sd = math.sqrt(sd_s ** 2 + (sm.VARIANT_TRANSFER * delta) ** 2 + extra_var + 0.02 * len(oos))
                zs[t] = (z, sd)
            e = expert[v.variant_id]
            reps = sum(REPEAT_DECAY ** i for i, rid in enumerate(recent_calls) if rid == v.variant_id)
            z_succ = zs["success"][0] + e["bonus"] + e["beater"] - REPEAT_W * reps
            # per-play learning: verified outcomes of THIS variant pull its estimate (shrunk toward the model)
            k, n = variant_stats.get(v.variant_id, (0.0, 0.0))
            resid = 0.0
            if n > 0:
                p0 = ln.sigmoid(z_succ)
                resid = max(-RESIDUAL_CAP, min(RESIDUAL_CAP, (k - n * p0) / (n + RESIDUAL_K) / max(0.1, p0 * (1 - p0))))
            z_succ += resid
            p_s = ln.sigmoid(z_succ)
            p_fd = ln.sigmoid(zs["first_down"][0])
            p_ex = ln.sigmoid(zs["explosive"][0])
            p_neg = ln.sigmoid(zs["negative"][0])
            util = sm.UTILITY["success"] * p_s + sm.UTILITY["explosive"] * p_ex + sm.UTILITY["negative"] * p_neg
            yards = self.model.exp_yards(xa, op)
            out.append(Scored(v, p_s, zs["success"][1], p_fd, p_ex, p_neg, yards, util, {
                "existing_model_p": round(op, 4), "model_success_stock": round(ps["success"][2], 4),
                "adjustment_logit_delta": round(sm.VARIANT_TRANSFER * (pa["success"][0] - ps["success"][0]), 4),
                "expert_prior": round(e["bonus"], 4), "beater": round(e["beater"], 4),
                "beats": e["beats"], "gate_top": e["gate_top"], "tagged_beater_for_top": e["tagged_beater_for_top"],
                "coverage_tags_known": e["coverage_tags_known"], "repeat_penalty": round(REPEAT_W * reps, 4),
                "verified_variant_outcomes": n, "variant_residual": round(resid, 4),
                "geometry": d.geometry_status, "unrepresented_adjustments": [i + 1 for i in d.unrepresented],
                "out_of_support": oos[:6]}))
        out.sort(key=lambda s: (-s.utility, s.variant.variant_id))
        return out


_SCORER: dict[tuple[str, str], Scorer] = {}


def load_scorer(art_dir: str | Path, cat: sc.SchemeCatalog | None = None) -> Scorer:
    art = Path(art_dir)
    meta = art / "metadata.json"
    key = (str(art.resolve()), str(meta.stat().st_mtime_ns) if meta.is_file() else "")
    hit = _SCORER.get(key)
    if hit is not None:
        return hit
    model = sm.load(art)
    cat = cat or sc.load_cached()
    want = (model.meta or {}).get("catalog_version")
    if want and want != cat.version():
        raise ValueError(f"model was trained for catalog {want}, the repo catalog is {cat.version()}: retrain")
    s = Scorer(model, cat, dz.GeometrySource())
    _SCORER.clear()
    _SCORER[key] = s
    return s


def variant_stats(conn: Any, cat: sc.SchemeCatalog) -> dict[str, tuple[float, float]]:
    """variant -> (successes, n) from verified, fully executed variant outcomes (all games incl. this one)."""
    out: dict[str, list[float]] = {}
    evs = [e for e in ev.variant_evidence(conn, cat) if e.usable and e.complete]
    if not evs:
        return {}
    from cfb_coach.outcome import parse_outcome
    for e in evs:
        r = conn.execute("SELECT down, distance, yards, result FROM snaps WHERE id = ?", (e.snap_id,)).fetchone()
        kind = parse_outcome(str(r[3])).kind if r[3] else None
        y = r[2]
        lab = dataset_mod._success_from_yards(side="offense", down=r[0], distance=r[1],
                                              yards=int(y) if y not in (None, "") else None, kind=kind)
        if lab in ("true", "false"):
            s = out.setdefault(e.variant_id, [0.0, 0.0])
            s[0] += 1.0 if lab == "true" else 0.0
            s[1] += 1.0
    return {k: (v[0], v[1]) for k, v in out.items()}


def recent_calls(conn: Any, session_id: str | None, limit: int = 6) -> list[str]:
    if not session_id or not ev._has_table(conn, "civil_scheme_decisions"):
        return []
    return [r[0] for r in conn.execute(
        "SELECT variant_id FROM civil_scheme_decisions WHERE session_id = ? AND status = 'CALL' ORDER BY id DESC LIMIT ?",
        (session_id, limit)).fetchall()]


def _belief_for(conn, sit, opponent_id: str, opponent_type: str) -> bl.Belief:
    extras = sit.extras if isinstance(getattr(sit, "extras", None), dict) else {}
    look = sit.coverage_hint if getattr(sit, "coverage_source", "none") == "live" else None
    # A "last" hint is the previous snap's coverage, already a closed snap: not counted twice.
    return bl.live_belief(conn, session_id=extras.get("session_id"), opponent_id=opponent_id,
                          opponent_type=opponent_type, presnap_look=look)


def no_legal_call(reasons: Sequence[str], *, explanation: str, verdicts: Mapping[str, Any] | None = None,
                  shadow: bool = False):
    from cfb_coach.madden.playcaller import MaddenCall
    call = MaddenCall(side="offense", formation="", play=lg.NO_LEGAL, adj_or_macro="",
                      read_or_user="", rationale=f"{lg.NO_LEGAL}: {explanation}")
    call.scheme_variant = {"status": lg.NO_LEGAL, "explanation": explanation, "reasons": list(reasons),
                           "verdicts": {k: v.as_dict() if hasattr(v, "as_dict") else v for k, v in (verdicts or {}).items()},
                           "shadow": shadow}
    call.call_label = "CIVIL_SCHEME_ONLY"
    return call


def _summarize_block(verdicts: Mapping[str, lg.Verdict]) -> list[str]:
    from collections import Counter
    c = Counter(r.split(":")[0].split(" (")[0] for v in verdicts.values() for r in v.reasons[:1])
    return [f"{n} variant(s): {r}" for r, n in c.most_common()]


def explain(top: Scored, belief: bl.Belief, runner_up: Scored | None) -> str:
    p = top.parts
    bits = [f"P(success) {top.p_success:.0%} (±{top.sd:.2f} logit)", f"existing model {p['existing_model_p']:.0%} for the base play"]
    if abs(p["adjustment_logit_delta"]) >= 0.01:
        bits.append(f"adjustments {p['adjustment_logit_delta']:+.2f} logit (transferred from route shapes, unverified)")
    if p["beater"] >= 0.05:
        g = p["gate_top"]
        tag = "tagged beater" if p["tagged_beater_for_top"] else "untagged (neutral share)"
        bits.append(f"belief leans {g[0]} (gate {g[1]:.2f}): {tag} {p['beater']:+.2f}")
    if p["repeat_penalty"]:
        bits.append(f"repeat penalty -{p['repeat_penalty']:.2f}")
    if p["verified_variant_outcomes"]:
        bits.append(f"{int(p['verified_variant_outcomes'])} verified outcomes of this variant ({p['variant_residual']:+.2f})")
    top_cov = ", ".join(f"{k} {v:.0%}" for k, v in sorted(belief.mix.items(), key=lambda kv: -kv[1])[:3]) or "no coverage history"
    bits.append(f"coverage belief: {top_cov}; pressure {belief.pressure:.0%}")
    if runner_up is not None:
        bits.append(f"next: {runner_up.variant.variant_name} ({runner_up.utility - top.utility:+.3f})")
    return "; ".join(bits)


def build_call(top: Scored, ranked: Sequence[Scored], belief: bl.Belief, *, book: Mapping[str, Any] | None,
               shadow: bool, ledger: Mapping[str, Any] | None = None):
    from cfb_coach.madden.playcaller import MaddenCall
    v = top.variant
    alias = ((ledger or {}).get(v.variant_id) or {}).get("book_play")
    f, q = lg.book_names(book, v.formation, alias or v.base_play)
    formation, play = (f or v.formation), (q or alias or v.base_play)
    adjs = [f"{a.index + 1}) {a.target}: {a.action}" for a in v.adjustments]
    warnings = []
    if not v.name_verified:
        warnings.append("variant NAME unverified in the source guide")
    if alias:
        warnings.append(f"guide base play '{v.base_play}' is called '{alias}' in your book (owner-confirmed name)")
    if not v.verified_in_user_game:
        warnings.append("catalog: not verified in your game" + (" — owner-verified locally" if not shadow else ""))
    if shadow:
        warnings.append("SHADOW: owner verification skipped; do not treat as a live call")
    if not v.coverage_tags_known:
        warnings.append("no coverage tags for this formation (unknown, not negative)")
    if top.parts.get("unrepresented_adjustments"):
        warnings.append("adjustments not represented on provisional geometry: "
                        + ", ".join(f"#{i}" for i in top.parts["unrepresented_adjustments"]))
    warnings.append("route geometry is a provisional AceMadden-derived prior, not a verified assignment")
    why = explain(top, belief, ranked[1] if len(ranked) > 1 else None)
    call = MaddenCall(side="offense", formation=formation, play=play,
                      adj_or_macro=f"{v.variant_name} [{v.variant_id}]",
                      read_or_user="Read: none provided (no verified read progression)",
                      rationale=f"CIVIL SCHEME ONLY — {why}")
    call.adjustment = {"id": f"{ev.ADJ_PREFIX}{v.variant_id}", "label": f"{v.variant_name} [{v.variant_id}]",
                       "buttons": "no button map — set by hand: " + "; ".join(adjs), "why": why}
    call.scheme_variant = {
        "status": "CALL", "variant_id": v.variant_id, "variant_name": v.variant_name, "formation": v.formation,
        "base_play": v.base_play, "book_play": alias or None,
        "adjustments": [[a.target, a.action] for a in v.adjustments],
        "p_success": round(top.p_success, 4), "uncertainty_logit_sd": round(top.sd, 4),
        "p_first_down": round(top.p_first_down, 4), "p_explosive": round(top.p_explosive, 4),
        "p_negative": round(top.p_negative, 4), "exp_yards": top.exp_yards, "utility": round(top.utility, 4),
        "warnings": warnings, "explanation": why, "shadow": shadow, "parts": top.parts,
        "ranking": [s.as_dict() for s in ranked[:8]], "belief": belief.as_log(),
    }
    call.call_label = "CIVIL_SCHEME_ONLY"
    return call


def select(scorer: Scorer, cat: sc.SchemeCatalog, conn: Any, *, db: Any, sit: Any, opponent_id: str,
           book: Mapping[str, Any] | None, shadow: bool = False, belief: bl.Belief | None = None):
    from cfb_coach.opponents import is_cpu_opponent
    legal, verdicts = lg.legal_set(cat, db=db, book=book, require_verified=not shadow)
    if not legal:
        return no_legal_call(_summarize_block(verdicts), verdicts=verdicts, shadow=shadow,
                             explanation="no catalog variant passes the legal filter: " + " | ".join(_summarize_block(verdicts))), None
    opp_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
    belief = belief or _belief_for(conn, sit, opponent_id, opp_type)
    session_id = (sit.extras or {}).get("session_id") if isinstance(getattr(sit, "extras", None), dict) else None
    ranked = scorer.score(legal, down=sit.down, distance=sit.distance, yardline=sit.yardline,
                          opponent_id=opponent_id, opponent_type=opp_type, belief=belief,
                          recent_calls=recent_calls(conn, session_id), variant_stats=variant_stats(conn, cat))
    call = build_call(ranked[0], ranked, belief, book=book, shadow=shadow, ledger=lg.load_verified(db))
    return call, ranked


def call_civil(sit: Any, opponent_id: str, db: Any, book: Mapping[str, Any] | None, *, shadow: bool = False,
               log: bool = True):
    """Entry point from ``playcaller.make_call``. Never raises into the caller; never falls back to another play."""
    t0 = time.perf_counter()
    try:
        cat = sc.load_cached()
        art = md.artifact_dir(db)
        if art is None or not (Path(art) / "model.json").is_file():
            return no_legal_call(["no trained civil-scheme model"], explanation="civil_scheme_only is on but no trained "
                                 "model artifact was found (run `ml civil-scheme train`)", shadow=shadow)
        scorer = load_scorer(art, cat)
        want = md.load_state(db).get("model_sha256")
        if want and want != (scorer.model.meta or {}).get("model_sha256"):
            return no_legal_call(["model mismatch"], explanation="the loaded model is not the one `enable` recorded",
                                 shadow=shadow)
        applied = None
        if db is not None:
            from cfb_coach.madden import playbook
            applied = playbook.load_books(db).get("offense")
        call, ranked = select(scorer, cat, db.conn, db=db, sit=sit, opponent_id=opponent_id, book=applied, shadow=shadow)
        lg.revalidate_call(db, call, cat)
    except Exception as exc:  # noqa: BLE001 — fail closed with the reason, never a hidden fallback
        call, ranked = no_legal_call([f"error: {exc.__class__.__name__}"], explanation=f"civil coordinator error: {exc}",
                                     shadow=shadow), None
    ms = (time.perf_counter() - t0) * 1000.0
    call.scheme_variant["latency_ms"] = round(ms, 2)
    call.scheme_variant["latency_target_ms"] = LATENCY_TARGET_MS
    if log and db is not None and not getattr(db, "read_only", False):
        try:
            sv = call.scheme_variant
            ev.log_decision(db.conn, {
                "session_id": (sit.extras or {}).get("session_id") if isinstance(getattr(sit, "extras", None), dict) else None,
                "opponent_id": opponent_id, "mode": "shadow" if shadow else md.CIVIL, "status": sv["status"],
                "variant_id": sv.get("variant_id"), "formation": sv.get("formation"), "base_play": sv.get("base_play"),
                "adjustments": sv.get("adjustments"), "down": sit.down, "distance": sit.distance, "yardline": sit.yardline,
                "situation_raw": getattr(sit, "raw", None), "belief": sv.get("belief"),
                "prediction": {k: sv.get(k) for k in ("p_success", "uncertainty_logit_sd", "p_first_down", "p_explosive",
                                                       "p_negative", "exp_yards", "utility")},
                "ranking": [[r["variant_id"], r["utility"]] for r in sv.get("ranking") or []],
                "model_version": (md.load_state(db) or {}).get("model_sha256"), "catalog_version": sc.load_cached().version(),
                "explanation": sv.get("explanation"),
            })
        except Exception:  # noqa: BLE001 — logging must not block the call
            pass
    return call


def format_lines(call: Any) -> list[str]:
    sv = getattr(call, "scheme_variant", None) or {}
    if sv.get("status") == lg.NO_LEGAL:
        return [f"{lg.NO_LEGAL}", f"  why: {sv.get('explanation')}"]
    lines = [f"FORMATION: {sv['formation']}", f"BASE PLAY: {sv['base_play']}",
             f"VARIATION: {sv['variant_name']}  [{sv['variant_id']}]", "ADJUSTMENTS (all, in order):"]
    lines += [f"  {i + 1}) {t}: {a}" for i, (t, a) in enumerate(sv["adjustments"])]
    lines.append(f"P(success) {sv['p_success']:.0%}  uncertainty ±{sv['uncertainty_logit_sd']:.2f} logit  "
                 f"1st down {sv['p_first_down']:.0%}  explosive {sv['p_explosive']:.0%}  sack/TO/loss {sv['p_negative']:.0%}"
                 + (f"  exp yards {sv['exp_yards']:.1f}" if sv.get("exp_yards") is not None else ""))
    lines.append("WARNINGS: " + " | ".join(sv["warnings"]))
    lines.append("WHY: " + sv["explanation"])
    lines.append("READS: none shown (no verified read progression)")
    return lines
