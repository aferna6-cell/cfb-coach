"""Champion/challenger held-out comparison of three scorers on WHOLE games in chronological order:

  A name_only            the existing structural score (the champion; uses the play, the situation and the time-correct coverage mixture)
  B structured_prior     A + football_priors edge under the same mixture
  C structured_learned   B with the cells replaced by their adaptive posterior, updated only from EARLIER closed snaps

Rows are scored from state known before the snap (the coverage mixture is time-correct); a row's own revealed coverage reaches the learner only
after it is scored. Promotion needs the game-clustered interval to exclude zero in the challenger's favour, plus >= MIN_GAMES test games.
Results on synthetic rows are plumbing/behaviour checks, never evidence about Madden."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import adaptive_priors as ap
from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_time_correct as tc

MIN_GAMES = 8


def _ll(y: int, p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def evaluate(rows: Sequence[Mapping[str, Any]], game_order: Mapping[str, int], *, min_train_games: int = 3, n_boot: int = 400, seed: int = 11,
             penalty_scale: float = 1.0, max_snaps_per_game: int = 60) -> dict[str, Any]:
    ann = tc.annotate([dict(r) for r in rows])
    folds = he.make_folds(ann, dict(game_order), is_test_row=he.is_verified_offense, mode="chronological", min_train_games=min_train_games)
    by_game: dict[str, list] = {}
    for r in ann:
        by_game.setdefault(str(r.get("game_id")), []).append(r)
    preds: list[dict[str, Any]] = []
    for f in folds:
        train = [r for g in f.train_games for r in by_game.get(g, [])]
        art = sm.train(train, penalty_scale=penalty_scale)
        test = [r for r in by_game.get(f.test_game, []) if he.is_verified_offense(r)][:max_snaps_per_game]
        seen_test: list[Mapping[str, Any]] = []
        for r in test:
            prior, _ = sm.pre_snap_mix(art, r.get("opponent_id"))
            mix = tc.mix_from(prior, r[tc.KEY])
            s = sm.structure_any(r.get("executed_formation") or r.get("formation"), r.get("executed_play") or r.get("play"))
            kw = dict(down=r.get("down"), distance=r.get("distance"), yardline=r.get("yardline"), opponent_id=r.get("opponent_id"),
                      opponent_type=r.get("opponent_type"), mix=mix)
            a = sm.score(art, s, **kw).probability
            lg = sm._logit(a)
            b = sm._sigmoid(lg + fp.matchup_edge(s, mix)["edge"])
            learned, _ = ap.posterior_cells([*train, *seen_test], r.get("opponent_id"))
            c = sm._sigmoid(lg + fp.matchup_edge(s, mix, learned=learned)["edge"])
            preds.append({"game": f.test_game, "y": int(bool(he.offense_label(r, "success"))), "p": {"name_only": a, "structured_prior": b, "structured_learned": c}})
            seen_test.append(r)                         # AFTER scoring: this snap's revealed coverage may now teach the learner
    out: dict[str, Any] = {"folds": len(folds), "rows": len(preds), "games": len({p["game"] for p in preds})}
    for m in ("name_only", "structured_prior", "structured_learned"):
        out[m] = round(sum(_ll(p["y"], p["p"][m]) for p in preds) / max(1, len(preds)), 5)
    for m, ref in (("structured_prior", "name_only"), ("structured_learned", "name_only"), ("structured_learned", "structured_prior")):
        out[f"{m}_vs_{ref}"] = he.paired_bootstrap(preds, m, ref, n_boot=n_boot, seed=seed)
    out["gate"] = gate(out)
    return out


def gate(rep: Mapping[str, Any]) -> dict[str, Any]:
    """Challenger = structured_learned vs champion = name_only. Promote only if the whole-game interval is entirely below zero."""
    c = rep.get("structured_learned_vs_name_only") or {}
    ok = rep["games"] >= MIN_GAMES and c.get("hi") is not None and c["hi"] < 0
    return {"result": "PROMOTE_CANDIDATE" if ok else "KEEP_CHAMPION",
            "reason": ("interval excludes zero in the challenger's favour" if ok else
                       f"games={rep['games']} (need {MIN_GAMES}); interval {c.get('lo')}..{c.get('hi')} does not clear zero"),
            "note": "promotion also requires readiness, latency and legality gates (cpu_experiment.py) and an explicit owner opt-in"}
