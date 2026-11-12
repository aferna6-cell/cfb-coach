"""Train / load / score the civil-scheme model.

Architecture (chosen by chronological held-out evaluation, see ``evaluate.py``):

  logit P(target) = offset + head(features)

* ``offset`` for success is the EXISTING coordinator's model (``experimental_model`` hierarchical
  shrinkage, trained exactly as production trains it). Existing knowledge is kept: the new head only
  learns what the existing model misses, and an all-zero head reproduces the existing model.
* ``head`` is one of ``existing`` (no head), ``logistic`` / ``logistic_strong`` (L2 offset logistic,
  lambda 25 / 400) or ``gbt`` (offset
  gradient-boosted trees, game-bootstrap ensemble), picked per target on the EARLY held-out games with a
  do-no-harm margin, then confirmed on the LATE held-out games.
* Training rows are verified offensive snaps only. Their designs are the STOCK geometry of the play that
  was run (provisional Ace geometry, low-confidence, never a verified assignment).
* A variant is scored as  head(stock design of its base play) + VARIANT_TRANSFER * [head(adjusted design) -
  head(stock design)]  because no verified variant outcome exists yet: the adjustment effect is a
  regularized transfer from base plays / route shapes, and its size is added to the uncertainty.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.scheme import beliefs as bl
from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme import features as ft
from cfb_coach.madden.scheme import learners as ln

SCHEMA = "madden27.civil_scheme_model.v1"
TARGETS = ("success", "first_down", "explosive", "negative")
UTILITY = {"success": 1.0, "explosive": 0.10, "negative": -0.20}   # fixed weights; no situation rules
VARIANT_TRANSFER = 0.5
UNREPRESENTED_SD = 0.15        # added logit sd per adjustment that could not be represented on geometry
DIRECTIONAL_STEM_SD = 0.10     # added logit sd per stock-route stem applied directionally (exact depth unknown)
NAME_ONLY_SD = 0.25            # added logit sd when the base play has no geometry at all
AUX_NAMES = ft.NAMES + ("k_existing_logit",)
NEGATIVE_KINDS = frozenset({"int", "fumble", "sack"})


def label(row: Mapping[str, Any], target: str) -> int | None:
    if target == "success":
        return he.offense_label(row, "success")
    if target == "explosive":
        return he.offense_label(row, "explosive")
    kind = he._kind(row)
    if target == "first_down":
        if kind is None:
            return None
        c = em._converted_offense(row)
        return None if c is None else int(c)
    if target == "negative":
        if kind is None:
            return None
        if kind in NEGATIVE_KINDS:
            return 1
        y = he._int(row.get("yards"))
        return int(y is not None and y < 0)
    if target == "conversion":
        return he.offense_label(row, "conversion")
    raise ValueError(target)


def yards_of(row: Mapping[str, Any]) -> float | None:
    kind = he._kind(row)
    if kind in ("incomplete", "int"):
        return 0.0
    y = he._int(row.get("yards"))
    return None if y is None else float(max(-15, min(60, y)))


# ------------------------------------------------------------------ existing model offsets (prequential)

def existing_offsets(rows: Sequence[Mapping[str, Any]], game_order: Mapping[str, int],
                     keep: Callable[[Mapping[str, Any]], bool] = he.is_verified_offense,
                     shared: Callable[[Mapping[str, Any]], bool] | None = None) -> dict[int, float]:
    """row index -> existing-model P(success) trained ONLY on games strictly earlier than the row's game.

    The same value is the existing model's held-out prediction (``ExperimentalSuccess`` in heldout_eval).
    ``shared``: rows allowed to train the SHARED base-play model. civil_scheme_only passes "not a civil variant
    snap", so a variant's outcomes feed only its own per-variant residual, never its sibling variants (t338u)."""
    by_game: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_game.setdefault(str(r.get("game_id") or ""), []).append(i)
    games = sorted((g for g in by_game if g in game_order), key=lambda g: game_order[g])
    out: dict[int, float] = {}
    for gi, g in enumerate(games):
        targets = [i for i in by_game[g] if keep(rows[i])]
        if not targets:
            continue
        train = [dict(rows[i]) for h in games[:gi] for i in by_game[h] if shared is None or shared(rows[i])]
        art = em.train_experimental(train, side="offense")
        for i in targets:
            r = rows[i]
            out[i] = float(em.predict_success(art, formation=str(r.get("action_formation") or ""),
                                              play=str(r.get("action_play") or ""), down=r.get("down"),
                                              distance=r.get("distance"), yardline=he._int(r.get("yardline")),
                                              opponent_id=r.get("opponent_id"),
                                              opponent_type=r.get("opponent_type"))["probability"])
    return out


@dataclass
class Example:
    idx: int
    game: str
    x: list[float]
    offset_p: float
    y: dict[str, int | None]
    yards: float | None
    row: Mapping[str, Any]
    geometry_status: str


def build_examples(rows: Sequence[Mapping[str, Any]], game_order: Mapping[str, int], geo: dz.GeometrySource, *,
                   offsets: Mapping[int, float] | None = None, beliefs: Sequence[bl.Belief] | None = None,
                   keep: Callable[[Mapping[str, Any]], bool] = he.is_verified_offense,
                   designs: Mapping[int, dz.Design] | None = None) -> list[Example]:
    """``designs`` overrides the stock design for specific rows (verified variant executions)."""
    beliefs = beliefs if beliefs is not None else bl.annotate_rows(rows, game_order)
    offsets = offsets if offsets is not None else existing_offsets(rows, game_order, keep)
    cache: dict[tuple[str, str], tuple[dz.Design, dict[str, float]]] = {}
    out: list[Example] = []
    for i, r in enumerate(rows):
        if not keep(r) or i not in offsets:
            continue
        if designs and i in designs:
            d = designs[i]
            rf = dz.route_features(d)
        else:
            key = (str(r.get("action_formation") or ""), str(r.get("action_play") or ""))
            if key not in cache:
                d0 = dz.stock_design(key[0], key[1], geo)
                cache[key] = (d0, dz.route_features(d0))
            d, rf = cache[key]
        x, _ = ft.vector(down=r.get("down"), distance=r.get("distance"), yardline=r.get("yardline"),
                         belief=beliefs[i], design=d, opponent_type=r.get("opponent_type"), route_f=rf)
        out.append(Example(i, str(r.get("game_id")), x, offsets[i], {t: label(r, t) for t in TARGETS},
                           yards_of(r), r, d.geometry_status))
    return out


# ------------------------------------------------------------------ model

HEAD_KINDS = ("existing", "logistic", "logistic_strong", "gbt")
GBT_PARAMS = {"n_trees": 120, "learning_rate": 0.05, "max_depth": 2, "min_leaf": 25, "l2": 5.0,
              "subsample": 0.8, "colsample": 0.7}
LOGISTIC_L2 = 25.0
LOGISTIC_STRONG_L2 = 400.0


def _aux_x(x: Sequence[float], offset_p: float) -> list[float]:
    return list(x) + [ln.logit(offset_p)]


@dataclass
class Head:
    target: str
    kind: str
    base_logit: float                 # used when the target has no existing-model offset
    model: Any = None

    def offset(self, offset_p: float) -> float:
        return ln.logit(offset_p) if self.target == "success" else self.base_logit

    def xin(self, x: Sequence[float], offset_p: float) -> list[float]:
        return list(x) if self.target == "success" else _aux_x(x, offset_p)

    def margin(self, x: Sequence[float], offset_p: float) -> tuple[float, float]:
        if self.kind == "existing" or self.model is None:
            return 0.0, 0.0
        return self.model.margin(self.xin(x, offset_p))

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "kind": self.kind, "base_logit": self.base_logit,
                "model": self.model.to_dict() if self.model is not None else None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Head":
        m = d.get("model")
        model = None
        if m:
            model = (ln.OffsetLogistic.from_dict(m) if m["kind"] == "offset_logistic"
                     else ln.GBTEnsemble.from_dict(m))
        return cls(d["target"], d["kind"], float(d["base_logit"]), model)


def fit_head(examples: Sequence[Example], target: str, kind: str, *, n_members: int = 5) -> Head:
    ex = [e for e in examples if e.y.get(target) is not None]
    ys = [int(e.y[target]) for e in ex]   # type: ignore[arg-type]
    rate = (sum(ys) + 1.0) / (len(ys) + 2.0) if ys else 0.5
    h = Head(target, kind, ln.logit(rate))
    if kind == "existing" and target != "success":
        # "existing" for an auxiliary target = the training base rate (the existing coordinator has no such head)
        return h
    if kind == "existing" or len(ex) < 40 or len(set(ys)) < 2:
        h.kind = "existing"
        return h
    X = [h.xin(e.x, e.offset_p) for e in ex]
    off = [h.offset(e.offset_p) for e in ex]
    if kind == "logistic":
        h.model = ln.OffsetLogistic(l2=LOGISTIC_L2).fit(X, ys, off)
    elif kind == "logistic_strong":
        h.model = ln.OffsetLogistic(l2=LOGISTIC_STRONG_L2).fit(X, ys, off)
    elif kind == "gbt":
        h.model = ln.GBTEnsemble.fit(X, ys, off, [e.game for e in ex], n_members=n_members, **GBT_PARAMS)
    else:
        raise ValueError(kind)
    return h


@dataclass
class SchemeModel:
    heads: dict[str, Head]
    yards: dict[str, Any] | None
    support_max: list[float]
    support_min: list[float]
    meta: dict[str, Any] = field(default_factory=dict)
    existing: Any = None               # em.ExperimentalArtifact trained on all history (live offset)

    def predict(self, x: Sequence[float], offset_p: float) -> dict[str, tuple[float, float, float]]:
        """target -> (logit, logit sd, p)."""
        out = {}
        for t, h in self.heads.items():
            m, sd = h.margin(x, offset_p)
            z = h.offset(offset_p) + m
            out[t] = (z, sd, ln.sigmoid(z))
        return out

    def exp_yards(self, x: Sequence[float], offset_p: float) -> float | None:
        if not self.yards:
            return None
        return ln.ridge_predict(self.yards, _aux_x(x, offset_p))

    def out_of_support(self, x: Sequence[float]) -> list[str]:
        return [n for n, v, lo, hi in zip(ft.NAMES, x, self.support_min, self.support_max)
                if v > hi + 1e-9 or v < lo - 1e-9]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "feature_names": list(ft.NAMES), "heads": {t: h.to_dict() for t, h in self.heads.items()},
                "yards": self.yards, "support_max": self.support_max, "support_min": self.support_min,
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SchemeModel":
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} artifact")
        if list(d["feature_names"]) != list(ft.NAMES):
            raise ValueError("feature layout changed since this artifact was trained: retrain")
        return cls({t: Head.from_dict(h) for t, h in d["heads"].items()}, d.get("yards"),
                   list(d["support_max"]), list(d["support_min"]), dict(d.get("meta") or {}))


def fit(examples: Sequence[Example], arch: Mapping[str, str], *, n_members: int = 5,
        with_yards: bool = True) -> SchemeModel:
    heads = {t: fit_head(examples, t, arch.get(t, "existing"), n_members=n_members) for t in TARGETS}
    yards = None
    if with_yards:
        ye = [e for e in examples if e.yards is not None]
        if len(ye) >= 40:
            yards = ln.ridge([_aux_x(e.x, e.offset_p) for e in ye], [e.yards for e in ye])  # type: ignore[misc]
    cols = list(zip(*[e.x for e in examples])) if examples else [()] * len(ft.NAMES)
    return SchemeModel(heads, yards, [max(c) if c else 0.0 for c in cols], [min(c) if c else 0.0 for c in cols])


def utility(pred: Mapping[str, tuple[float, float, float]]) -> float:
    return sum(w * pred[t][2] for t, w in UTILITY.items() if t in pred)


# ------------------------------------------------------------------ artifacts

def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def save(model: SchemeModel, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = model.to_dict()
    (out / "model.json").write_text(json.dumps(body, sort_keys=True))
    if model.existing is not None:
        em.save_artifact(model.existing, out / "existing_offense.json")
    meta = dict(model.meta)
    meta["model_sha256"] = digest(body)
    meta["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return out


def load(art_dir: str | Path) -> SchemeModel:
    d = Path(art_dir)
    body = json.loads((d / "model.json").read_text())
    m = SchemeModel.from_dict(body)
    meta_p = d / "metadata.json"
    if meta_p.is_file():
        m.meta = json.loads(meta_p.read_text())
        want = m.meta.get("model_sha256")
        if want and want != digest(body):
            raise ValueError(f"{d}/model.json does not match its metadata sha256: refusing to load")
    ex = d / "existing_offense.json"
    if ex.is_file():
        m.existing = em.load_artifact(ex)
    return m
