"""Evaluation of the defensive-knowledge features: recognition audit, controlled self-checks, held-out comparison, latency.

Three kinds of evidence, kept apart on purpose:

1. ``recognition_audit``    deterministic: how many installed/external names the old recognizer vs the catalog resolve. This
                            measures NAME recognition, not football value and not winning.
2. synthetic worlds         PLUMBING ONLY. A planted pressure x route-shape effect proves the pipeline can learn a signal that only the
                            new features can see; a null world and a label-permutation control prove it does not invent one. They say
                            nothing about real Madden games.
3. ``evaluate``             real held-out comparison on a COPY of the game database (whole games, chronological, paired game-clustered
                            bootstrap). The verdict is NO-GO unless the knowledge model beats the identical model without it with an
                            interval that excludes zero, on enough games. With no database it is reported as NOT EVALUATED.
"""
from __future__ import annotations

import random
import time
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import defense_catalog as dc
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import historical_defense_baseline as hb
from cfb_coach.madden.model import offense_art_features as af
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_structural_eval as ses
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_time_correct as tc

EVAL_VERSION = "defense_knowledge_eval.v1"
MIN_GAMES_FOR_CLAIM = 8
GO_MARGIN = 0.0            # the knowledge model's mean log-loss difference must be below this AND its 95% interval's upper bound too


# ------------------------------------------------------------------------------------------------ 1. recognition audit

def recognition_audit() -> dict[str, Any]:
    """Old exact-name recognizer vs the catalog resolver over the 123 installed entries and the 27 external variants."""
    cat = ds.catalog()
    out: dict[str, Any] = {"installed": {}, "external": {}, "by_formation": {}, "unresolved": [], "legacy_disagreements": []}
    for scope, status in (("installed", "installed"), ("external", "external_identified")):
        es = [e for e in cat["entries"] if e["status"] == status]
        legacy_named = [e for e in es if hb.normalize_label(e["raw_name"]) in hb.KNOWN_COVERAGES]
        res = [ds.resolve(e["raw_name"], e["raw_formation"]) for e in es]
        out[scope] = {
            "entries": len(es), "old_recognizer_named_coverage": len(legacy_named),
            "catalog_resolved": sum(r.key is not None for r in res),
            "resolution_types": {k: sum(r.resolution == k for r in res) for k in sorted({r.resolution for r in res})},
            "with_pressure_axis": sum(e["pressure"]["concept"]["value"] != "standard" for e in es),
            "with_known_deep_shell": sum(e["coverage"]["deep_shell"]["value"] != "unknown" for e in es),
            "with_ea_documented_mechanic": sum(e["verification"]["level"] in ("mechanic_documented", "role_partially_documented")
                                               and any(a["status"] == "ea_documented" for a in
                                                       (e["coverage"]["deep_shell"], e["coverage"]["type"], e["pressure"]["concept"]))
                                               or any(v["meaning_status"] == "ea_documented" for v in e["coverage"]["variants"])
                                               for e in es),
        }
        for e, r in zip(es, res):
            if r.key is None:
                out["unresolved"].append(e["id"])
            lab = hb.normalize_label(e["raw_name"])
            if lab in hb.KNOWN_COVERAGES and hb.GENERIC_TO_BUCKET.get(lab, lab) != r.legacy_bucket:
                out["legacy_disagreements"].append((e["id"], hb.GENERIC_TO_BUCKET.get(lab, lab), r.legacy_bucket))
    for e in (x for x in cat["entries"] if x["status"] == "installed"):
        b = out["by_formation"].setdefault(e["raw_formation"], {"entries": 0, "old": 0, "new": 0})
        b["entries"] += 1
        b["old"] += hb.normalize_label(e["raw_name"]) in hb.KNOWN_COVERAGES
        b["new"] += ds.resolve(e["raw_name"], e["raw_formation"]).key is not None
    out["form_channel"] = ds.channel_audit()
    out["verification"] = {"catalog_entries": cat["summary"]["entries"], "levels": cat["summary"]["verification_levels"],
                           "ledger_known_slots": cat["summary"]["ledger_known_slots"],
                           "ledger_verified_slots": cat["summary"]["ledger_verified_slots"],
                           "note": "Catalog coverage is NOT responsibility verification: no entry has a verified defender role."}
    return out


# ------------------------------------------------------------------------------------------------ 2. synthetic worlds

# Personas differ in how much pressure they show, so earlier same-game entries carry information about later snaps (as a real
# opponent's tendencies do). The legacy recognizer names only the coverage labels it knows (Cover 3 Sky, Cover 4 Quarters, Cover 2 Man,
# Cover 1 Hole, Cover 6) and drops the rest, so it cannot see how blitz-heavy a game is.
PERSONAS_DK = {
    "blitz_heavy": {"Fire Zone 3": 0.30, "FS Zone Blitz": 0.20, "Zero Blitz": 0.20, "Nickel Sim 2": 0.15, "Cover 3 Sky": 0.15},
    "coverage_heavy": {"Cover 3 Cloud": 0.30, "Cover 3 Sky": 0.25, "Cover 4 Quarters": 0.20, "Cover 2 Man": 0.15, "Fire Zone 3": 0.10},
    "mixed": {"Mid Blitz 0": 0.20, "Field Sim 3": 0.20, "Cover 1 Hole": 0.20, "Cover 6": 0.20, "Cover 3 Cloud": 0.20},
}
PRESSURE_LABELS = {"Fire Zone 3", "FS Zone Blitz", "Zero Blitz", "Mid Blitz 0", "Nickel Sim 2", "Field Sim 3"}


def planted_prob(formation: str, play: str, label: str, *, strength: float, null: bool) -> float:
    """P(success): pressure labels REWARD shallow route geometry, everything else mildly punishes it. ``null`` removes the effect."""
    if null:
        return 0.5
    shallow = af.art_features(formation, play)["art_shallow_share"]
    p = 0.5 + (strength if label in PRESSURE_LABELS else -0.5 * strength) * (shallow - 0.62)
    return min(0.9, max(0.1, p))


def make_pressure_world_rows(*, n_games: int = 24, snaps: int = 45, seed: int = 7, strength: float = 0.8, null: bool = False,
                             opponent: str = "cpu", through_form: bool = True) -> list[dict[str, Any]]:
    """SYNTHETIC. Each game has a persistent persona, so earlier entries are informative about later snaps (as with a real CPU).

    The outcome depends on the call the opponent REALLY ran. ``through_form=True`` (default) stores what the existing post-play form
    would store for that typed call (``defense_semantics.form_store``: coarse, sometimes nothing), i.e. today's capture. ``False`` stores
    the exact call name: the counterfactual "capture were lossless", used only to measure how much that would be worth."""
    rnd = random.Random(seed)
    book = [(p["formation"], p["play"]) for p in oa.load_database()["plays"] if af.art_features(p["formation"], p["play"])["art_present"]]
    rows: list[dict[str, Any]] = []
    for g in range(n_games):
        persona = list(PERSONAS_DK)[g % len(PERSONAS_DK)]
        labels, weights = list(PERSONAS_DK[persona]), list(PERSONAS_DK[persona].values())
        for i in range(snaps):
            f, pl = rnd.choice(book)
            cov = rnd.choices(labels, weights)[0]
            down, distance = rnd.choice([1, 2, 3]), rnd.choice([3, 6, 10])
            ok = rnd.random() < planted_prob(f, pl, cov, strength=strength, null=null)
            raw = {"session_id": f"g{g:02d}", "ml_snap_id": f"g{g:02d}-{i:04d}", "opponent_id": opponent, "opponent_type": "cpu",
                   "side": "offense", "down": down, "distance": distance, "yardline": rnd.randint(8, 92), "quarter": 1 + (i * 4) // snaps,
                   "result": f"+{distance + 3}" if ok else "incomplete", "formation": f, "play": pl, "executed_status": "identified",
                   "executed_verification": "verified", "executed_formation": f, "executed_play": pl,
                   "coverage_seen": ds.form_store(cov) if through_form else cov}
            rows.append(dataset_mod._normalize_row(raw, provenance="synthetic_planted_pressure_signal"))
    return rows


def permute_coverage_labels(rows: Sequence[Mapping[str, Any]], seed: int = 3, mode: str = "global") -> list[dict[str, Any]]:
    """Control: shuffle ``coverage_seen`` among rows so a label no longer belongs to the snap whose outcome it explains.

    ``global`` (the valid null control) shuffles across ALL rows: every marginal is kept, every label -> outcome link is destroyed.
    ``within_game`` shuffles inside each game only. It keeps each game's mix of labels, so a model that learns an opponent's TENDENCY
    (how much pressure a game shows) still gains: a diagnostic of what is being learned, NOT a null control.
    """
    if mode not in ("global", "within_game"):
        raise ValueError(mode)
    rnd = random.Random(seed)
    out = [dict(r) for r in rows]
    groups: list[list[int]] = [list(range(len(out)))] if mode == "global" else []
    if mode == "within_game":
        by_game: dict[str, list[int]] = {}
        for i, r in enumerate(out):
            by_game.setdefault(str(r.get("game_id")), []).append(i)
        groups = list(by_game.values())
    for idxs in groups:
        labs = [out[i].get("coverage_seen") for i in idxs]
        rnd.shuffle(labs)
        for i, lab in zip(idxs, labs):
            out[i]["coverage_seen"] = lab
    return out


# ------------------------------------------------------------------------------------------------ 3. held-out comparison

class StructuralDK(ses.StructuralSuccess):
    """Same model as ``StructuralSuccess`` plus the opt-in defense-knowledge feature block."""

    def fit(self, train_rows: Sequence[Mapping[str, Any]]) -> None:
        self.art = sm.train(train_rows, penalty_scale=self.penalty_scale, use_matchup=self.use_matchup, defense_knowledge=True)


def comparison_models() -> list[he.Predictor]:
    return [he.BaseRate(("success",), he.offense_label, he.is_verified_offense), he.ExperimentalSuccess(),
            ses.StructuralSuccess("structural_features"), StructuralDK("structural_defense_knowledge")]


def evaluate_rows(rows: Sequence[Mapping[str, Any]], game_order: Mapping[str, int] | None = None, *, source: str = "",
                  min_train_games: int = he.DEFAULT_MIN_TRAIN_GAMES, n_boot: int = 300, seed: int = he.DEFAULT_SEED) -> dict[str, Any]:
    """Chronological whole-game held-out comparison of the knowledge model vs the identical model without it."""
    rows = tc.annotate(rows, knowledge=True)
    order = dict(game_order) if game_order else _order(rows)
    verified = [r for r in rows if he.is_verified_offense(r)]
    games = len({r.get("game_id") for r in verified})
    rep: dict[str, Any] = {"eval_version": EVAL_VERSION, "source": source, "verified_offense_rows": len(verified),
                           "verified_offense_games": games, "claims_allowed": games >= MIN_GAMES_FOR_CLAIM,
                           "label_resolution": _label_resolution(rows)}
    folds = he.make_folds(rows, order, is_test_row=he.is_verified_offense, mode="chronological", min_train_games=min_train_games)
    rep["folds"] = len(folds)
    if not folds:
        rep["verdict"] = "NOT EVALUATED: no held-out folds could be formed"
        return rep
    t0 = time.perf_counter()
    preds, timing = he.run_folds(rows, folds, comparison_models, targets=("success",), label_fn=he.offense_label,
                                 is_test_row=he.is_verified_offense)
    metrics = he.summarize_predictions(preds, ("success",), reference="structural_features", n_boot=n_boot, seed=seed)
    rep["metrics"], rep["timing"], rep["eval_seconds"] = metrics, timing, round(time.perf_counter() - t0, 1)
    cmp = metrics["success"]["vs_reference"].get("structural_defense_knowledge", {})
    rep["knowledge_vs_baseline"] = cmp
    rep["verdict"] = verdict(cmp, games)
    return rep


def verdict(cmp: Mapping[str, Any], games: int) -> str:
    if not cmp or cmp.get("mean_diff") is None:
        return "NOT EVALUATED: too few held-out games for a paired comparison"
    if games < MIN_GAMES_FOR_CLAIM:
        return (f"NO-GO (insufficient data): {games} verified games < {MIN_GAMES_FOR_CLAIM}; mean log-loss difference "
                f"{cmp['mean_diff']:+.4f} is plumbing evidence only")
    if cmp["mean_diff"] < GO_MARGIN and cmp["hi"] < GO_MARGIN:
        return (f"GO for a CPU shadow/overlay experiment only: knowledge model better by {-cmp['mean_diff']:.4f} nats "
                f"(95% interval {cmp['lo']:+.4f}..{cmp['hi']:+.4f}). Not a primary-caller promotion.")
    return (f"NO-GO: no demonstrated improvement (mean {cmp['mean_diff']:+.4f}, 95% interval {cmp['lo']:+.4f}..{cmp['hi']:+.4f}); "
            "keep the existing model")


def _order(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    order: dict[str, int] = {}
    for r in rows:
        order.setdefault(str(r.get("game_id")), len(order))
    return order


def _label_resolution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How the STORED labels in these rows are read. A stored label may already be a lossy rewrite made by the post-play form, so this
    reports collapsed (many possible calls) separately from exact (a label the form cannot produce, i.e. lossless capture)."""
    labs = [r.get("coverage_seen") for r in rows if str(r.get("side") or "offense") == "offense" and r.get("result") not in (None, "")]
    labs = [x for x in labs if x not in (None, "")]
    cats = {"untyped_pressure": 0, "collapsed_mixture": 0, "exact_lossless": 0, "unresolved": 0}
    for x in labs:
        w = ds.stored_label_weights(x)
        if not w:
            cats["unresolved"] += 1
        elif set(w) == {ds.GENERIC_PRESSURE_KEY}:
            cats["untyped_pressure"] += 1
        elif len(w) == 1:
            cats["exact_lossless"] += 1
        else:
            cats["collapsed_mixture"] += 1
    return {"labels": len(labs), "old_named_coverage": sum(hb.normalize_label(x) in hb.KNOWN_COVERAGES for x in labs),
            "old_pressure_only": sum(hb.normalize_label(x) == hb.PRESSURE_UNKNOWN for x in labs), "knowledge_reading": cats,
            "note": "collapsed_mixture labels keep family structure but not the variant: the post-play form rewrites many calls to one label"}


# ------------------------------------------------------------------------------------------------ latency

def latency_report(art: sm.StructuralArtifact, book: Sequence[tuple[str, str]], mix: Mapping[str, float], *, repeats: int = 5) -> dict[str, Any]:
    """Wall time to score EVERY play of ``book`` against one opponent mix (the unit of work in a live ranking)."""
    times = []
    structures = [sm.structure_any(f, p) for f, p in book]
    for _ in range(repeats):
        t0 = time.perf_counter()
        dist = None
        for s in structures:
            sm.score(art, s, down=3, distance=7, yardline=45, opponent_id="cpu", opponent_type="cpu", mix=mix)
        times.append((time.perf_counter() - t0) * 1000)
    return {"plays": len(structures), "ms_p50": he.percentile(times, 50), "ms_max": max(times), "budget_ms": 150,
            "within_budget": max(times) <= 150, "uses_defense_knowledge": art.uses_defense_knowledge}
