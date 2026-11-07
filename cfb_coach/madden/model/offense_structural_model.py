"""Structure-aware offensive success model (research; not wired into the live caller).

The existing experimental coordinator (hierarchical shrinkage) stays the base
learner: its logit is the *offset*. This module learns a heavily regularized
residual on structural play features (route geometry, concept flags, name-hint
channel, formation, coverage-matchup priors). Properties:

* hierarchical sharing  - structure is shared across plays; unseen plays score
  from their features, and the offset carries all play-identity effects;
* regularization        - group-wise L2 (name hints strongest), residual shrunk
  by alpha = n / (n + N0), unpenalized bias only;
* uncertainty           - Laplace approximation x' H^-1 x on the logit scale;
* honesty               - labels only 'success' from verified supervised rows;
  coverage mix comes from EARLIER snaps only (never this snap's coverage).

Pure stdlib (sparse Newton), so it can run inside the 150 ms live budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_art_features as af
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import offense_time_correct as tc
from cfb_coach.madden.model import historical_defense_baseline as hb

MODEL_KIND = "offense_structural.v1"
ARTIFACT_SCHEMA = "madden27.offense_structural.artifact.v1"
N0 = 30.0                      # residual shrink: alpha = n / (n + N0)
STD_FLOOR = 0.15
LAMBDA = {"form": 20.0, "hint": 30.0, "route": 12.0, "concept": 15.0, "match": 8.0, "art": 24.0, "sit": 20.0}
# Only artifacts trained with the opt-in defense-knowledge block carry this group. It is a weak, partly-hypothesis prior, so it gets
# the same heavy shrinkage as the name-hint group. Kept out of LAMBDA so default artifacts stay byte-identical.
LAMBDA_DK = {"dk": 30.0}
EXTRAPOLATION_SHRINK = 0.5      # modified (never-seen) plays get half the modeled structural gain
MAX_EXTRAPOLATED_LOGIT = 0.30   # and never more than this, whatever the weights say
MIN_ROWS_FOR_RESIDUAL = 25     # below this the residual is switched off (offset only)


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def structure_any(formation: str, play: str) -> pf.PlayStructure:
    """Installed play -> full structure; any other play -> hint-only structure (no routes, no formation)."""
    s = pf.structure_lookup(formation, play)
    if s is not None:
        return s
    return pf.PlayStructure(formation=formation, play=play, routes=(), skill_slots=(), personnel_tier="unknown",
                            hint=oa.name_hint(play))


def _sparse(vec: Sequence[float]) -> list[tuple[int, float]]:
    return [(i, v) for i, v in enumerate(vec) if v != 0.0]


def _invert(a: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse of a symmetric positive-definite matrix (penalized Hessian)."""
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            raise ValueError("singular Hessian")
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                rc = m[c]
                m[r] = [x - f * y for x, y in zip(m[r], rc)]
    return [row[n:] for row in m]


@dataclass
class StructuralArtifact:
    kind: str = MODEL_KIND
    schema: str = ARTIFACT_SCHEMA
    names: list[str] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)       # index 0 = bias
    cov: list[list[float]] = field(default_factory=list)     # (d+1)x(d+1) inverse penalized Hessian
    support_max: list[float] = field(default_factory=list)   # per-feature max |value| seen in training
    selection: dict[str, Any] = field(default_factory=dict)  # nested model-selection record (inner held-out games)
    penalty_scale_chosen: float = 1.0
    n_train: int = 0
    n_positive: int = 0
    alpha: float = 0.0
    residual_active: bool = False
    penalties: dict[str, float] = field(default_factory=dict)
    global_mix: dict[str, float] = field(default_factory=dict)
    opponent_mix: dict[str, dict[str, float]] = field(default_factory=dict)
    assignment_digest: str = ""
    art_digest: str = ""
    defense_knowledge_digest: str = ""   # set only by defense-knowledge artifacts (feature_version == pf.FEATURE_VERSION_DK)
    feature_version: str = pf.FEATURE_VERSION
    trained_games: int = 0
    notes: str = ""
    base: dict[str, Any] = field(default_factory=dict)       # ExperimentalArtifact.to_dict()

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        if not d.get("defense_knowledge_digest"):
            d.pop("defense_knowledge_digest", None)       # default artifacts serialize exactly as before
        return d

    @property
    def uses_defense_knowledge(self) -> bool:
        return self.feature_version == pf.FEATURE_VERSION_DK

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StructuralArtifact":
        if raw.get("schema") != ARTIFACT_SCHEMA or raw.get("kind") != MODEL_KIND:
            raise ValueError("not an offense_structural artifact")
        a = cls(**{k: raw[k] for k in cls().__dict__ if k in raw})
        return a


def save_artifact(a: StructuralArtifact, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(a.to_dict(), sort_keys=True), encoding="utf-8")
    return p


def load_artifact(path: str | Path) -> StructuralArtifact:
    return StructuralArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_artifact(a: StructuralArtifact) -> list[str]:
    """Every reason this artifact must not be used (empty = usable)."""
    errs = []
    dk = a.feature_version == pf.FEATURE_VERSION_DK
    names = list(pf.feature_names(defense_knowledge=dk))
    if a.names != names:
        errs.append("feature names differ from the current feature schema (retrain)")
    if a.feature_version not in (pf.FEATURE_VERSION, pf.FEATURE_VERSION_DK):
        errs.append("feature version mismatch")
    if dk:
        try:
            if a.defense_knowledge_digest != ds.data_digest():
                errs.append("defense knowledge catalog changed or missing; retrain")
        except ValueError as exc:
            errs.append(str(exc))
    elif a.defense_knowledge_digest:
        errs.append("defense knowledge digest on a non-knowledge artifact")
    if a.art_digest != af.data_digest():
        errs.append("route-art feature bank changed or missing; retrain")
    d = len(a.names)
    if len(a.weights) != d + 1 or len(a.mean) != d or len(a.scale) != d:
        errs.append("weight/scale dimensions inconsistent")
    if a.cov and (len(a.cov) != d + 1 or any(len(r) != d + 1 for r in a.cov)):
        errs.append("covariance dimension inconsistent")
    if not all(math.isfinite(w) for w in a.weights):
        errs.append("non-finite weights")
    try:
        if a.assignment_digest != oa.database_digest(oa.load_effective()):
            errs.append("assignment database changed since training (retrain)")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"assignment database unreadable: {exc}")
    if not a.base:
        errs.append("missing base coordinator artifact")
    return errs


# ----------------------------------------------------------------- training


def _mix_from_rows(rows: Sequence[Mapping[str, Any]], knowledge: bool = False) -> dict[str, float]:
    """Historical coverage shares from CLOSED offense snaps (labels exactly as specific as entered).

    ``knowledge=True`` keys the shares by defense-knowledge label (``dk:*``), including untyped pressure."""
    counts: dict[str, float] = {}
    for r in rows:
        if str(r.get("side") or "offense") != "offense" or r.get("result") in (None, ""):
            continue
        if knowledge:
            for k, w in ds.stored_label_weights(r.get("coverage_seen"), r.get("coverage_typed")).items():
                counts[k] = counts.get(k, 0.0) + w
            continue
        lab = hb.normalize_label(r.get("coverage_seen"))
        if lab in hb.KNOWN_COVERAGES:
            b = hb.GENERIC_TO_BUCKET.get(lab, lab)
            counts[b] = counts.get(b, 0.0) + 1.0
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()} if tot else {}


def pre_snap_mix(a: StructuralArtifact, opponent_id: str | None) -> tuple[dict[str, float], str]:
    mix = a.opponent_mix.get(str(opponent_id or ""))
    if mix:
        return mix, "opponent_history"
    if a.global_mix:
        return a.global_mix, "global_history"
    return {}, "none"


AUTO_SCALES = (1.0, 4.0, 16.0)          # candidate L2 multipliers tried by nested selection
MIN_GAMES_FOR_SELECTION = 6              # fewer verified games than this -> residual stays OFF (base coordinator only)
SELECTION_MARGIN = 0.002                 # a residual must beat the base by this much held-out log loss to be switched on


def train(rows: Sequence[Mapping[str, Any]], *, penalty_scale: float | None = None,
          use_matchup: bool = True, max_iter: int = 12, defense_knowledge: bool = False) -> StructuralArtifact:
    """Fit on verified supervised offense rows. ``rows`` must be EARLIER games only.

    ``penalty_scale=None`` (default) runs nested, game-level model selection: rolling-origin over the LAST games of
    the training set picks the L2 multiplier, and the residual is switched on only if it beats the base coordinator
    on those inner held-out games ("do no harm"). A number fixes the multiplier (sensitivity runs, tests).

    ``defense_knowledge=True`` trains the opt-in variant: the same model plus the ``dk_*`` structural-defense block, with
    opponent coverage resolved against the defense knowledge catalog. The default (False) is unchanged.
    """
    key = _rows_key(rows, penalty_scale, use_matchup, max_iter, defense_knowledge)
    hit = _TRAIN_CACHE.get(key)
    if hit is not None:
        return copy.deepcopy(hit)
    art = _train_uncached(rows, penalty_scale=penalty_scale, use_matchup=use_matchup, max_iter=max_iter,
                          defense_knowledge=defense_knowledge)
    if len(_TRAIN_CACHE) >= 48:
        _TRAIN_CACHE.pop(next(iter(_TRAIN_CACHE)))
    _TRAIN_CACHE[key] = copy.deepcopy(art)
    return art


_TRAIN_CACHE: dict[str, "StructuralArtifact"] = {}


def _rows_key(rows: Sequence[Mapping[str, Any]], *parts: Any) -> str:
    """Content digest so identical training data is fitted once per process (evaluation re-fits the same folds)."""
    h = hashlib.sha256(repr(parts).encode())
    for r in rows:
        h.update(repr((r.get("game_id"), r.get("side"), r.get("action_formation"), r.get("action_play"), r.get("down"),
                       r.get("distance"), r.get("yardline"), r.get("result"), r.get("coverage_seen"),
                       r.get("eligibility"), r.get("opponent_id"))).encode())
    return h.hexdigest()


def _train_uncached(rows: Sequence[Mapping[str, Any]], *, penalty_scale: float | None, use_matchup: bool,
                    max_iter: int, defense_knowledge: bool = False) -> StructuralArtifact:
    rows = tc.annotate(rows, knowledge=defense_knowledge)
    dk = defense_knowledge
    if penalty_scale is not None:
        return _train_core(rows, penalty_scale=penalty_scale, use_matchup=use_matchup, max_iter=max_iter, defense_knowledge=dk)
    from cfb_coach.madden.model.heldout_eval import is_verified_offense, offense_label

    games: list[str] = []
    for r in rows:
        g = str(r.get("game_id"))
        if is_verified_offense(r) and offense_label(r, "success") is not None and g not in games:
            games.append(g)
    sel: dict[str, Any] = {"games": len(games), "scales": {}}
    if len(games) < MIN_GAMES_FOR_SELECTION:
        art = _train_core(rows, penalty_scale=AUTO_SCALES[1], use_matchup=use_matchup, max_iter=max_iter, defense_knowledge=dk)
        art.residual_active = False
        sel["decision"] = f"fewer than {MIN_GAMES_FOR_SELECTION} verified games: residual OFF, existing coordinator only"
        art.selection, art.notes = sel, sel["decision"]
        return art
    n_inner = max(2, min(4, len(games) // 3))
    by_game: dict[str, list] = {}
    for r in rows:
        by_game.setdefault(str(r.get("game_id")), []).append(r)
    ll_base: list[float] = []
    ll_scale: dict[float, list[float]] = {sc: [] for sc in AUTO_SCALES}

    def _ll(y: int, p: float) -> float:
        p = min(1 - 1e-6, max(1e-6, p))
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    for test_g in games[-n_inner:]:
        earlier = games[:games.index(test_g)]
        train_rows = [r for g in earlier for r in by_game[g]]
        test_rows = [r for r in by_game[test_g] if is_verified_offense(r) and offense_label(r, "success") is not None]
        for sc in AUTO_SCALES:
            a = _train_core(train_rows, penalty_scale=sc, use_matchup=use_matchup, max_iter=max_iter, defense_knowledge=dk)
            if not a.residual_active:
                continue
            for r in test_rows:
                y = offense_label(r, "success")
                st = structure_any(str(r.get("action_formation") or ""), str(r.get("action_play") or ""))
                scr = score(a, st, down=r.get("down"), distance=r.get("distance"), yardline=r.get("yardline"),
                            opponent_id=r.get("opponent_id"), opponent_type=r.get("opponent_type"), cov=r.get(tc.KEY))
                ll_scale[sc].append(_ll(y, scr.probability))
                if sc == AUTO_SCALES[0]:
                    ll_base.append(_ll(y, scr.base_probability))
    base_ll = statistics_mean(ll_base)
    best, best_ll = None, None
    for sc in AUTO_SCALES:
        m = statistics_mean(ll_scale[sc])
        sel["scales"][str(sc)] = None if m is None else round(m, 5)
        if m is not None and (best_ll is None or m < best_ll):
            best, best_ll = sc, m
    sel["inner_base_log_loss"] = None if base_ll is None else round(base_ll, 5)
    sel["inner_games"] = n_inner
    use = best is not None and base_ll is not None and best_ll < base_ll - SELECTION_MARGIN
    art = _train_core(rows, penalty_scale=best or AUTO_SCALES[1], use_matchup=use_matchup, max_iter=max_iter, defense_knowledge=dk)
    if not use:
        art.residual_active = False
        sel["decision"] = "residual OFF: no penalty beat the existing coordinator on inner held-out games"
    else:
        sel["decision"] = f"residual ON: penalty x{best} beat the existing coordinator on inner held-out games"
    art.selection, art.penalty_scale_chosen = sel, float(best or AUTO_SCALES[1])
    art.notes = (art.notes + " | " if art.notes else "") + sel["decision"]
    return art


def statistics_mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _train_core(rows: Sequence[Mapping[str, Any]], *, penalty_scale: float = 1.0,
                use_matchup: bool = True, max_iter: int = 12, defense_knowledge: bool = False) -> StructuralArtifact:
    from cfb_coach.madden.model import dataset as dataset_mod
    from cfb_coach.madden.model.heldout_eval import is_verified_offense, offense_label

    dk = defense_knowledge
    rows = tc.annotate(rows, knowledge=dk)         # same-game earlier coverage only, per row
    priors = tc.prior_as_of(rows, knowledge=dk)     # historical prior from EARLIER games only, per row
    base = em.train_experimental([dict(r) for r in rows], side="offense")
    keep = [i for i, r in enumerate(rows) if is_verified_offense(r) and offense_label(r, "success") is not None]
    sup = [rows[i] for i in keep]
    sup_prior = [priors[i] for i in keep]
    art = StructuralArtifact(names=list(pf.feature_names(defense_knowledge=dk)), base=base.to_dict(),
                             assignment_digest=oa.database_digest(oa.load_effective()),
                             art_digest=af.data_digest(),
                             penalties={k: v * penalty_scale for k, v in {**LAMBDA, **(LAMBDA_DK if dk else {})}.items()})
    if dk:
        art.feature_version, art.defense_knowledge_digest = pf.FEATURE_VERSION_DK, ds.data_digest()
    art.global_mix = _mix_from_rows(rows, dk)
    by_opp: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_opp.setdefault(str(r.get("opponent_id") or ""), []).append(r)
    art.opponent_mix = {k: m for k, v in by_opp.items() if k and len(v) >= 8 for m in [_mix_from_rows(v, dk)] if m}
    art.trained_games = len({r.get("game_id") for r in sup})
    n = len(sup)
    art.n_train = n
    d = len(art.names)
    art.mean, art.scale = [0.0] * d, [1.0] * d
    art.weights = [0.0] * (d + 1)
    if n < MIN_ROWS_FOR_RESIDUAL:
        art.notes = f"only {n} verified rows; residual disabled (base coordinator only)"
        return art
    # Cross-fitted offsets: each game's base-coordinator logit comes from a base fit on the OTHER games, so the
    # residual learns from out-of-sample base errors instead of an in-sample base that already memorized them.
    game_ids = sorted({str(r.get("game_id")) for r in sup})
    fold_base: dict[str, em.ExperimentalArtifact] = {}
    if len(game_ids) >= 2:
        for g in game_ids:
            fold_base[g] = em.train_experimental([dict(r) for r in rows if str(r.get("game_id")) != g], side="offense")
    X, y, off = [], [], []
    for r, prior in zip(sup, sup_prior):
        s = structure_any(str(r.get("action_formation") or ""), str(r.get("action_play") or ""))
        mix = tc.mix_from(prior, r.get(tc.KEY), knowledge=dk)
        dist, _ = pf.scheme_distribution(mix)
        vec = pf.full_vector(s, dist, {"down": r.get("down"), "distance": r.get("distance"), "yardline": r.get("yardline")},
                             names=art.names)
        if not use_matchup:
            vec = [v if not nm.startswith(("m_", "dk_")) else 0.0 for nm, v in zip(art.names, vec)]
        X.append(vec)
        y.append(offense_label(r, "success"))
        pred = em.predict_success(fold_base.get(str(r.get("game_id")), base), formation=s.formation, play=s.play,
                                  down=r.get("down"), distance=r.get("distance"), yardline=_to_int(r.get("yardline")),
                                  opponent_id=r.get("opponent_id"), opponent_type=r.get("opponent_type"))
        off.append(_logit(float(pred["probability"])))
    art.n_positive = int(sum(y))
    cols = list(zip(*X))
    art.support_max = [max(abs(v) for v in c) for c in cols]
    # Scale only (no centering) so rows stay sparse; the unpenalized bias absorbs the level.
    art.mean = [0.0] * d
    art.scale = [max(STD_FLOOR, math.sqrt(sum(v * v for v in c) / n - (sum(c) / n) ** 2)) for c in cols]
    Z = [[(v - m) / s for v, m, s in zip(row, art.mean, art.scale)] for row in X]
    lam = [0.0] + [art.penalties[pf.group_of(nm)] for nm in art.names]
    w = [0.0] * (d + 1)
    sp = [_sparse([1.0] + z) for z in Z]
    H_inv = None
    for _ in range(max_iter):
        g = [lam[i] * w[i] for i in range(d + 1)]
        H = [[0.0] * (d + 1) for _ in range(d + 1)]
        for row, yi, oi in zip(sp, y, off):
            eta = oi + sum(w[i] * v for i, v in row)
            p = _sigmoid(eta)
            wt = p * (1 - p)
            for i, v in row:
                g[i] += (p - yi) * v
                Hi = H[i]
                for j, u in row:
                    Hi[j] += wt * v * u
        for i in range(d + 1):
            H[i][i] += lam[i] if lam[i] else 1e-6
        H_inv = _invert(H)
        step = [sum(H_inv[i][j] * g[j] for j in range(d + 1)) for i in range(d + 1)]
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(s) for s in step) < 1e-5:
            break
    art.weights = w
    art.cov = [[round(v, 8) for v in row] for row in H_inv] if H_inv else []
    art.alpha = n / (n + N0)
    art.residual_active = True
    art.notes = f"residual fit on {n} verified rows from {art.trained_games} earlier games"
    art.weights = [round(v, 8) for v in art.weights]
    _ = dataset_mod  # keep import local: dataset defines eligibility used by is_verified_offense
    return art


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Score:
    probability: float
    base_probability: float
    residual_logit: float
    logit_se: float
    mix_source: str

    @property
    def interval(self) -> tuple[float, float]:
        lg = _logit(self.probability)
        return _sigmoid(lg - 1.64 * self.logit_se), _sigmoid(lg + 1.64 * self.logit_se)


def _z(a: StructuralArtifact, vec: Sequence[float]) -> list[tuple[int, float]]:
    return _sparse([1.0] + [(v - m) / s for v, m, s in zip(vec, a.mean, a.scale)])


def residual_and_var(a: StructuralArtifact, vec: Sequence[float]) -> tuple[float, float]:
    if not a.residual_active:
        return 0.0, 0.0
    row = _z(a, vec)
    r = sum(a.weights[i] * v for i, v in row)
    var = sum(v * a.cov[i][j] * u for i, v in row for j, u in row) if a.cov else 0.0
    return a.alpha * r, (a.alpha ** 2) * max(0.0, var)


def base_probability(a: StructuralArtifact, s: pf.PlayStructure, *, down: Any = None, distance: Any = None,
                     yardline: Any = None, opponent_id: str | None = None, opponent_type: str | None = None) -> float:
    pred = em.predict_success(a.base, formation=s.formation, play=s.play, down=down, distance=distance,
                              yardline=_to_int(yardline), opponent_id=opponent_id, opponent_type=opponent_type)
    return float(pred["probability"])


def score(a: StructuralArtifact, s: pf.PlayStructure, *, down: Any = None, distance: Any = None, yardline: Any = None,
          opponent_id: str | None = None, opponent_type: str | None = None,
          mix: Mapping[str, float] | None = None, base_p: float | None = None,
          cov: Mapping[str, Any] | None = None) -> Score:
    """``cov`` = this row's ``presnap_cov`` (earlier same-game coverage counts); ignored if ``mix`` is given."""
    if mix is None:
        prior, src = pre_snap_mix(a, opponent_id)
        if cov and a.uses_defense_knowledge and "dk_counts" not in cov:
            raise ValueError("row was annotated without defense knowledge: call annotate(rows, knowledge=True)")
        if cov and (cov.get("dk_counts") if a.uses_defense_knowledge else cov.get("counts")):
            mix, src = tc.mix_from(prior, cov, knowledge=a.uses_defense_knowledge), src + "+same_game"
        else:
            mix = prior
    else:
        src = "caller_supplied"
    dist, _ = pf.scheme_distribution(mix)
    bp = base_p if base_p is not None else base_probability(a, s, down=down, distance=distance, yardline=yardline,
                                                              opponent_id=opponent_id, opponent_type=opponent_type)
    res, var = residual_and_var(a, pf.full_vector(s, dist, {"down": down, "distance": distance, "yardline": yardline},
                                                  names=a.names))
    p = _sigmoid(_logit(bp) + res)
    return Score(p, bp, res, math.sqrt(var), src)


def score_delta(a: StructuralArtifact, stock: pf.PlayStructure, mod: pf.PlayStructure, *,
                mix: Mapping[str, float] | None, opponent_id: str | None = None) -> tuple[float, float]:
    """(logit gain, its standard error) for stock -> modified structure with the play-level offset held fixed.

    Extrapolation: training never saw hot-routed plays, so this is a structural
    inference, not a measured effect. Callers must demand a margin AND a lower bound.
    """
    if not a.residual_active:
        return 0.0, 0.0
    if mix is None:
        mix, _ = pre_snap_mix(a, opponent_id)
    dist, _ = pf.scheme_distribution(mix)
    xa, xb = pf.full_vector(stock, dist, names=a.names), pf.full_vector(mod, dist, names=a.names)
    za, zb = dict(_z(a, xa)), dict(_z(a, xb))
    idx = set(za) | set(zb)
    dz = {i: zb.get(i, 0.0) - za.get(i, 0.0) for i in idx}
    k = a.alpha * EXTRAPOLATION_SHRINK
    gain = k * sum(a.weights[i] * v for i, v in dz.items())
    gain = max(-MAX_EXTRAPOLATED_LOGIT, min(MAX_EXTRAPOLATED_LOGIT, gain))
    var = (k ** 2) * sum(v * a.cov[i][j] * u for i, v in dz.items() for j, u in dz.items())
    return gain, math.sqrt(max(0.0, var))


def in_support(a: StructuralArtifact, s: pf.PlayStructure, *, mix: Mapping[str, float] | None,
               opponent_id: str | None = None, tol: float = 1e-9) -> tuple[bool, list[str]]:
    """Is every feature of ``s`` within the range training actually saw? Hot-routed plays often are not."""
    if not a.support_max:
        return False, ["artifact has no recorded training support"]
    if mix is None:
        mix, _ = pre_snap_mix(a, opponent_id)
    dist, _ = pf.scheme_distribution(mix)
    vec = pf.full_vector(s, dist, names=a.names)
    bad = [n for n, v, m in zip(a.names, vec, a.support_max) if abs(v) > m + tol]
    return (not bad), bad


def artifact_digest(a: StructuralArtifact) -> str:
    return hashlib.sha256(json.dumps(a.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


class DeltaScorer:
    """Stock-vs-modified scoring with the stock vector, mix and distribution computed once per play."""

    def __init__(self, a: StructuralArtifact, stock: pf.PlayStructure, *, mix: Mapping[str, float] | None,
                 opponent_id: str | None = None, sit: Mapping[str, Any] | None = None):
        self.a = a
        if mix is None:
            mix, _ = pre_snap_mix(a, opponent_id)
        self.dist, _ = pf.scheme_distribution(mix)
        self.sit = sit
        self.za = dict(_z(a, pf.full_vector(stock, self.dist, sit, names=a.names)))
        self.k = a.alpha * EXTRAPOLATION_SHRINK

    def score(self, mod: pf.PlayStructure) -> tuple[float, float, list[str]]:
        """(shrunk+capped logit gain, its SE, features outside training support)."""
        a = self.a
        if not a.residual_active:
            return 0.0, 0.0, ["residual inactive"]
        vec = pf.full_vector(mod, self.dist, self.sit, names=a.names)
        outside = [n for n, v, m in zip(a.names, vec, a.support_max) if abs(v) > m + 1e-9] if a.support_max \
            else ["no recorded training support"]
        zb = dict(_z(a, vec))
        dz = {i: zb.get(i, 0.0) - self.za.get(i, 0.0) for i in set(zb) | set(self.za)}
        gain = max(-MAX_EXTRAPOLATED_LOGIT, min(MAX_EXTRAPOLATED_LOGIT,
                                               self.k * sum(a.weights[i] * v for i, v in dz.items())))
        var = (self.k ** 2) * sum(v * a.cov[i][j] * u for i, v in dz.items() for j, u in dz.items())
        return gain, math.sqrt(max(0.0, var)), outside
