"""Time-correct pre-snap coverage for TRAINING and EVALUATION rows.

Row i of a game may use only coverage entered after EARLIER, closed offensive snaps of the SAME game and
opponent (the form's "coverage seen" box), plus a historical prior built from EARLIER games. Row i's own
coverage and result, and anything later, never enter its features. The blend formula is the live one
(``offense_live_coverage.blend_counts``), so train-time and play-time semantics are identical.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import historical_defense_baseline as hb
from cfb_coach.madden.model import offense_live_coverage as lc

KEY = "presnap_cov"          # {"counts": {...}, "n_named": int, "n_pressure": int, "n_unknown": int, "n_prev": int}


def _closed(row: Mapping[str, Any]) -> bool:
    return row.get("result") not in (None, "") and str(row.get("result")).strip() != ""


def annotate(rows: Sequence[Mapping[str, Any]], *, knowledge: bool = False) -> list[dict[str, Any]]:
    """Copy rows, adding ``presnap_cov`` computed strictly from earlier same-game/opponent offense snaps.

    ``rows`` must be in chronological (snaps.id) order, as ``dataset.build_rows`` produces. Idempotent.

    ``knowledge=True`` additionally records ``dk_counts`` (earlier closed labels resolved against the defense knowledge catalog,
    keyed ``dk:*``) next to the legacy ``counts``. Rows already annotated WITHOUT ``dk_counts`` are recomputed, never reused.
    """
    out: list[dict[str, Any]] = []
    hist: dict[tuple[str, str], deque] = {}
    raw_hist: dict[tuple[str, str], deque] = {}
    for r in rows:
        r = dict(r)
        if KEY in r and (not knowledge or "dk_counts" in r[KEY]):
            out.append(r)
            continue
        key = (str(r.get("game_id") or r.get("session_id") or ""), str(r.get("opponent_id") or ""))
        dq = hist.setdefault(key, deque(maxlen=lc.MAX_PRIOR_SNAPS))
        rdq = raw_hist.setdefault(key, deque(maxlen=lc.MAX_PRIOR_SNAPS))
        labels = list(dq)
        counts, n_named, n_press, n_unk = lc.count_labels(labels)
        r[KEY] = {"counts": dict(counts), "n_named": n_named, "n_pressure": n_press, "n_unknown": n_unk,
                  "n_prev": len(labels)}
        if knowledge:
            r[KEY]["dk_counts"] = ds.family_counts(rdq)
        out.append(r)
        # Add THIS row's coverage only after its own features were fixed, and only if it is an offense snap
        # that was closed (a result was entered). Defense-side rows describe our defense, not their coverage.
        if str(r.get("side") or "") == "offense" and _closed(r):
            dq.appendleft(hb.normalize_label(r.get("coverage_seen")))
            rdq.appendleft(r.get("coverage_seen"))
    return out


def mix_from(prior_mix: Mapping[str, float] | None, cov: Mapping[str, Any] | None, *, knowledge: bool = False) -> dict[str, float]:
    """Pre-snap coverage distribution for a row: historical prior blended with same-game observations.

    ``knowledge=True`` blends the ``dk:*`` counts into a ``dk:*``-keyed prior (see ``annotate(knowledge=True)``).
    """
    if knowledge:
        if cov is not None and "dk_counts" not in cov:
            raise ValueError("row was annotated without defense knowledge: call annotate(rows, knowledge=True)")
        return lc.blend_counts(Counter((cov or {}).get("dk_counts") or {}), prior_mix)
    return lc.blend_counts(Counter((cov or {}).get("counts") or {}), prior_mix)


def prior_as_of(rows: Sequence[Mapping[str, Any]], *, knowledge: bool = False) -> list[dict[str, float]]:
    """Per-row historical prior from EARLIER GAMES ONLY (same opponent when it has >= 8 earlier named snaps).

    Rows are grouped by game in order of first appearance. Used so a training row's prior never contains its
    own game (or later games), mirroring inference where the prior comes from completed earlier games.
    """
    order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        g = str(r.get("game_id") or r.get("session_id") or "")
        if g not in seen:
            seen.add(g)
            order.append(g)
    per_game: dict[str, dict[str, Counter]] = {}
    for r in rows:
        if str(r.get("side") or "") != "offense" or not _closed(r):
            continue
        if knowledge:
            weights = ds.stored_label_weights(r.get("coverage_seen"))
            if not weights:
                continue
            g = str(r.get("game_id") or r.get("session_id") or "")
            d = per_game.setdefault(g, {})
            for bucket, w in weights.items():
                d.setdefault(str(r.get("opponent_id") or ""), Counter())[bucket] += w
            continue
        else:
            lab = hb.normalize_label(r.get("coverage_seen"))
            if lab not in hb.KNOWN_COVERAGES:
                continue
            bucket = hb.GENERIC_TO_BUCKET.get(lab, lab)
        g = str(r.get("game_id") or r.get("session_id") or "")
        d = per_game.setdefault(g, {})
        d.setdefault(str(r.get("opponent_id") or ""), Counter())[bucket] += 1
    priors: dict[tuple[str, str], dict[str, float]] = {}
    cum_all: Counter = Counter()
    cum_opp: dict[str, Counter] = {}
    for g in order:
        opps = per_game.get(g, {})
        snap_opp_before = {o: Counter(c) for o, c in cum_opp.items()}
        glob_before = Counter(cum_all)
        for o in {str(r.get("opponent_id") or "") for r in rows if str(r.get("game_id") or r.get("session_id") or "") == g}:
            c = snap_opp_before.get(o)
            use = c if c and sum(c.values()) >= 8 else glob_before
            tot = sum(use.values())
            priors[(g, o)] = {k: v / tot for k, v in use.items()} if tot else {}
        for o, c in opps.items():
            cum_opp.setdefault(o, Counter()).update(c)
            cum_all.update(c)
    return [priors.get((str(r.get("game_id") or r.get("session_id") or ""), str(r.get("opponent_id") or "")), {})
            for r in rows]
