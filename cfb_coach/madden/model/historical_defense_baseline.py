"""Research-only Madden NFL 27 opponent defense learning baseline.

Uses *our offense* snaps to study defenses opponents showed us. A stored
``coverage_seen`` is a historical label, NOT assumed to be a confirmed Madden
post-play screen. Labels and outcomes are never included in current pre-snap
features. Games are held out as whole sessions in chronological backtests.

The model learns coverage probabilities; this is not a hardcoded play caller.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

VERSION = "madden27.historical_defense_baseline.v1"
SOURCE = "legacy_coverage_seen_not_postplay_verified"
# Keep labels exactly as specific as their source permits. Do not infer Cover 0
# from 'pressure', or Cover 2 spot from the ambiguous 'Cover 2' label.
ALIASES = {
    "cover 1": "cover_1_unspecified",
    "cover 1 hole": "cover_1_hole",
    "cover 1 robber": "cover_1_robber",
    "cover 1 contain": "cover_1_contain",
    "cover 2": "cover_2_unspecified",
    "cover 2 man": "cover_2_man",
    "cover 2 invert": "cover_2_invert",
    "cover 2 spot": "cover_2_spot",
    "tampa 2": "tampa_2",
    "cover 3 sky": "cover_3_sky",
    "cover 3": "cover_3_unspecified",
    "cover 3 match": "cover_3_match",
    "cover 4": "cover_4_unspecified",
    "cover 4 quarters": "cover_4_quarters",
    "cover 4 palms": "cover_4_palms",
    "cover 4 drop": "cover_4_drop",
    "cover 6": "cover_6",
    "cover 9": "cover_9",
    "cover 0": "cover_0",
    "cover 0 blitz": "cover_0",
    "trap 2": "trap_2",
}
KNOWN_COVERAGES = tuple(sorted(set(ALIASES.values())))
PRESSURE_UNKNOWN = "pressure_unspecified"
UNRESOLVED = "unknown"


def normalize_label(label: Any) -> str:
    s = " ".join(str(label or "").lower().replace("-", " ").replace("_", " ").split())
    if not s:
        return UNRESOLVED
    if s in ALIASES:
        return ALIASES[s]
    if s in ("pressure", "blitz", "heavy pressure", "pass rush"):
        return PRESSURE_UNKNOWN
    return UNRESOLVED


def read_offense_snap_rows(db_path: str | Path) -> list[dict[str, Any]]:
    """Open a snapshot read-only and select our offense (opponent defense).

    Does not touch a live DB or write to it. All timestamps and game references
    stay local to the caller. Aggregated reports never publish raw opponents.
    """
    p = Path(db_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    with sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        required = {"side", "opponent_id", "session_id", "ts", "coverage_seen",
                    "formation", "play", "down", "distance", "result", "executed_verification"}
        existing = {r[1] for r in db.execute("PRAGMA table_info(snaps)")}
        if not required <= existing:
            raise ValueError("unsupported Madden snap DB schema: missing " + repr(sorted(required-existing)))
        rows = db.execute("""SELECT id, session_id, ts, opponent_id, side, down, distance,
                            yardline, quarter, formation, play, executed_formation,
                            executed_play, executed_status, executed_verification,
                            coverage_seen, result, yards, data_source
                            FROM snaps WHERE side = 'offense' ORDER BY ts, id""").fetchall()
    return [_clean(dict(r)) for r in rows]


def _clean(r: dict[str, Any]) -> dict[str, Any]:
    r["coverage"] = normalize_label(r.get("coverage_seen"))
    r["label_basis"] = SOURCE if r["coverage"] != UNRESOLVED else "no_usable_coverage"
    r["execution_verified"] = (r.get("executed_verification") == "verified"
                               and r.get("executed_status") == "identified")
    r["actual_formation"] = (r.get("executed_formation") or r.get("formation") or "").strip()
    r["actual_play"] = (r.get("executed_play") or r.get("play") or "").strip()
    return r


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def distance_band(distance: Any) -> str:
    d = _int_or_none(distance)
    if d is None or d < 0:
        return "unknown"
    return "short" if d <= 3 else "medium" if d <= 7 else "long"


def down_distance_context(down: Any, distance: Any) -> str:
    d = _int_or_none(down)
    return f"{d if d in (1,2,3,4) else 'unknown'}:{distance_band(distance)}"


def group_sessions(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Whole session grouping in start-time order (no within-game data leakage)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        session = str(r.get("session_id") or "")
        if not session:
            continue  # cannot construct leak-safe games from unknown session ids
        groups[session].append(dict(r))
    return [g for _, g in sorted(groups.items(),key=lambda kv: (min(str(r.get("ts") or "") for r in kv[1]), kv[0]))]


def _valid_history(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("coverage") in KNOWN_COVERAGES]


def _dist(counts: Mapping[str, int], prior: Mapping[str, float], strength: float,
          classes: Sequence[str]) -> dict[str, float]:
    n = sum(counts.get(c, 0) for c in classes)
    return {c: (counts.get(c, 0) + strength * prior[c]) / (n + strength) for c in classes}


def predict_coverage(history: Sequence[Mapping[str, Any]], situation: Mapping[str, Any], *,
                     mode: str = "contextual", min_label_count: int = 1) -> dict[str, Any]:
    """Predict distribution over *named coverage labels* given PRE-snap context.

    mode='global' ignores situation/opponent. mode='contextual' uses opponent
    and down-distance, with strong hierarchical shrinkage. Never accesses the
    snap's current label/result. This model has no guaranteed calibration.
    """
    # Closed input schema: rejects any accidental post-snap fields, including
    # newly added ones not on a finite deny list.
    allowed = {"opponent_id", "down", "distance", "yardline", "quarter",
               "observed_shell", "observed_front", "observed_pressure", "box_count"}
    extra = set(situation) - allowed
    if extra:
        raise ValueError(f"unknown or post-snap fields forbidden: {sorted(extra)}")
    if mode not in {"global", "contextual", "uniform"}:
        raise ValueError(mode)
    valid = _valid_history(history)
    # Include all supported Madden coverages, even not-yet-seen ones. Zero-
    # frequency classes receive prior mass; never assert they're impossible.
    classes = KNOWN_COVERAGES
    u = {c: 1.0 / len(classes) for c in classes}
    counts = Counter(r["coverage"] for r in valid)
    if mode == "uniform" or len(valid) < min_label_count:
        p = dict(u)
    else:
        glob = _dist(counts, u, 10.0, classes)
        if mode == "global":
            p = glob
        else:
            opp = str(situation.get("opponent_id") or "")
            context = down_distance_context(situation.get("down"), situation.get("distance"))
            global_situation = Counter(r["coverage"] for r in valid if down_distance_context(r.get("down"),r.get("distance"))==context)
            opp_all = Counter(r["coverage"] for r in valid if str(r.get("opponent_id") or "") == opp)
            opp_situation = Counter(r["coverage"] for r in valid if str(r.get("opponent_id") or "") == opp and down_distance_context(r.get("down"),r.get("distance")) == context)
            g = _dist(global_situation, glob, 18.0, classes)
            o = _dist(opp_all, glob, 20.0, classes)
            combined = {c: 0.65*g[c]+0.35*o[c] for c in classes}
            p = _dist(opp_situation,combined,25.0,classes)
    return {"schema": VERSION, "probabilities": p,
            "top": sorted(p,key=lambda c:(-p[c],c))[:5],
            "n_history_labeled": len(valid),
            "n_history_all": len(history),
            "basis": "stored historical labels, provenance not yet independently verified",
            "opponent_specific": mode == "contextual"}


def _brier(probs: Mapping[str, float], actual: str) -> float:
    return sum((p - float(c == actual)) ** 2 for c, p in probs.items())


def evaluate_games(rows: Sequence[Mapping[str, Any]], *, min_train_games: int = 5,
                   require_verified_execution: bool = False) -> dict[str, Any]:
    """Whole-game chronological validation against historical labels, not confirmed truth."""
    groups = group_sessions(rows)
    history: list[dict[str, Any]] = []
    score: dict[str, list[tuple[float,float,int]]] = {k:[] for k in ("uniform","global","contextual")}
    n_tests = 0
    n_sessions = 0
    for i,g in enumerate(groups):
        if i >= min_train_games:
            valid = [r for r in _valid_history(g)
                     if not require_verified_execution or r.get("execution_verified")]
            if valid:
                n_sessions += 1
            for row in valid:
                sit = {"opponent_id":row.get("opponent_id"),"down":row.get("down"),"distance":row.get("distance")}
                for mode in score:
                    p = predict_coverage(history,sit,mode=mode)["probabilities"]
                    y=row["coverage"]
                    score[mode].append((_brier(p,y),-math.log(max(p[y],1e-12)),int(max(p,key=p.get)==y)))
                n_tests += 1
        history.extend(g)
    def avg(xs: Sequence[tuple[float,float,int]]) -> dict[str,float|int|None]:
        return {"n":len(xs),"brier":round(sum(t[0] for t in xs)/len(xs),5) if xs else None,
                "log_loss":round(sum(t[1] for t in xs)/len(xs),5) if xs else None,
                "top1_accuracy":round(sum(t[2] for t in xs)/len(xs),5) if xs else None}
    return {"schema":VERSION,"definition":"Prediction of historical coverage_seen labels; NOT Madden-confirmed ground truth",
            "require_verified_own_execution":require_verified_execution,
            "games_total":len(groups),"games_held_out":n_sessions,"snap_labels_tested":n_tests,
            "min_train_games":min_train_games,"results":{k:avg(v) for k,v in score.items()},
            "no_same_game_leakage":True}


def success_label(row: Mapping[str, Any]) -> int | None:
    """Conventional success rate: 40%/60%/100% of line-to-gain by down.

    TD is success; INT/fumble/incomplete/sack treated as unsuccessful even when
    the raw text lacks yards. This is an analytic proxy, not an EPA model.
    """
    result = str(row.get("result") or "").strip().lower()
    if result in {"td","touchdown","convert","conversion"}:
        return 1
    if result in {"incomplete","int","interception","fumble","sack","stop"}:
        return 0
    if not result:
        return None
    m=re.fullmatch(r"[+\-]?\d+",result)
    yards = int(result) if m else _int_or_none(row.get("yards"))
    down = _int_or_none(row.get("down"))
    distance = _int_or_none(row.get("distance"))
    if yards is None or down not in (1,2,3,4) or distance is None or distance <= 0:
        return None
    return int(yards >= distance * (0.4 if down == 1 else 0.6 if down == 2 else 1.0))


def matchup_summary(rows: Sequence[Mapping[str, Any]], *, min_cell: int = 5) -> dict[str, Any]:
    """Describe, do not causally rank, historic play vs coverage results.

    Matchups only use VERIFIED EXECUTED own calls, plus observed (not proven
    screen-confirmed) coverage labels; cells require enough examples to display.
    """
    cells: dict[tuple[str,str,str],list[int]] = defaultdict(list)
    formations: dict[tuple[str,str],list[int]] = defaultdict(list)
    excluded = Counter()
    by_scheme: dict[str,list[int]] = defaultdict(list)
    for r in rows:
        if not r.get("execution_verified"):
            excluded["own_execution_unverified"]+=1;continue
        c=r.get("coverage")
        if c not in KNOWN_COVERAGES:
            excluded["coverage_unspecified_or_pressure"]+=1;continue
        succ=success_label(r)
        if succ is None:
            excluded["result_missing_or_invalid"]+=1;continue
        play=r.get("actual_play") or ""
        form=r.get("actual_formation") or ""
        if not play or not form:
            excluded["play_not_identified"]+=1;continue
        cells[(form,play,c)].append(succ)
        formations[(form,c)].append(succ)
        by_scheme[c].append(succ)
    def summary(vals:Sequence[int]) -> dict[str,Any]:
        n=len(vals);wins=sum(vals)
        # Empirical Bayes: Beta(2,2). Avoid reporting these as causal effects.
        p=(wins+2)/(n+4)
        se=math.sqrt(p*(1-p)/(n+5))
        return {"n":n,"observed_success_rate":round(wins/n,3),
                "shrunk_rate":round(p,3),"approx_interval_95":[round(max(0.,p-1.96*se),3),round(min(1.,p+1.96*se),3)]}
    visible=[{"formation":f,"play":p,"coverage":c,**summary(v)} for (f,p,c),v in cells.items() if len(v)>=min_cell]
    visible.sort(key=lambda x:(-x["n"],x["formation"],x["play"]))
    visible_formations=[{"formation":f,"coverage":c,**summary(v)} for (f,c),v in formations.items() if len(v)>=min_cell]
    visible_formations.sort(key=lambda x:(-x["n"],x["formation"],x["coverage"]))
    return {"schema":VERSION,"label_provenance":SOURCE,
            "own_execution_verified_only":True,
            "metric":"standard 40/60/100 down-distance success (observational)",
            "usable_snaps":sum(map(len,by_scheme.values())),
            "coverage_counts":{k:summary(v) for k,v in sorted(by_scheme.items())},
            "matchup_cells_at_min_n":visible,
            "formation_cells_at_min_n":visible_formations,
            "min_cell":min_cell,"excluded":dict(excluded),
            "warning":"Small observational samples, opponent skill, play selection, down-distance and game context confound matchup rates; not proof of counters."}


def audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    offense=len(rows)
    labeled=sum(r["coverage"]!=UNRESOLVED for r in rows)
    exact=sum(r["coverage"] in KNOWN_COVERAGES for r in rows)
    pressure=sum(r["coverage"]==PRESSURE_UNKNOWN for r in rows)
    confirmed_calls=sum(r["execution_verified"] for r in rows)
    named_and_executed=sum(r["execution_verified"] and r["coverage"] in KNOWN_COVERAGES for r in rows)
    counts=Counter(r["coverage"] for r in rows)
    return {"schema":VERSION,"offense_snaps":offense,"coverage_or_pressure_labels":labeled,
            "named_coverage_labels":exact,"pressure_only_labels":pressure,
            "own_executions_verified":confirmed_calls,
            "named_coverage_with_own_execution_verified":named_and_executed,
            "coverage_label_screen_verified":0,
            "coverage_screen_verified_note":"Source DB does not store screenshot proof of a named opponent call; 0 is verified-in-DB, not proof no labels were truly observed.",
            "label_counts":dict(sorted(counts.items())),
            "sessions":len(group_sessions(rows)),
            "notes":"No raw opponent IDs, plays, individual outcomes or timestamps in this aggregate."}



# Convert historical exact-scheme probabilities to the existing offense model's
# input contract. Generic names remain generic; labels without provenance are
# only historical estimates. Never misclassify 'pressure' as an exact coverage.
GENERIC_TO_BUCKET = {
    "cover_1_unspecified": "cover1",
    "cover_2_unspecified": "cover2",
    "cover_3_unspecified": "cover3",
    "cover_3_sky": "cover3",  # rotation info, but no proven match check
    "cover_4_unspecified": "cover4",
}


def offense_model_mix(prediction: Mapping[str, Any]) -> dict[str, float]:
    """Optional pre-snap input to offense_play_features.scheme_distribution.

    This doesn't activate the live coach; only the caller can opt into the
    experimental prediction after validating the held-out metrics.
    """
    probs = prediction.get("probabilities", {})
    if not isinstance(probs, Mapping):
        raise ValueError("probabilities mapping required")
    result: dict[str, float] = defaultdict(float)
    for key, raw in probs.items():
        if key not in KNOWN_COVERAGES:
            raise ValueError("unsupported coverage ID " + repr(key))
        p = float(raw)
        if not math.isfinite(p) or p < 0:
            raise ValueError("invalid coverage probability")
        result[GENERIC_TO_BUCKET.get(key, key)] += p
    total = sum(result.values())
    if total <= 0:
        raise ValueError("prediction has zero probability mass")
    return {k: v / total for k, v in sorted(result.items())}


def explain_scheme(label_or_id: str, *, rules: Mapping[str,Any]|None=None) -> dict[str,Any]:
    """Return conditional football knowledge, never a universal QB route order."""
    sid = label_or_id if label_or_id in KNOWN_COVERAGES else normalize_label(label_or_id)
    if sid == PRESSURE_UNKNOWN:
        return {"scheme":"pressure_unspecified", "mechanics":"Defensive pressure reported; actual coverage not identified.",
                "conditional_read":"Use the post-snap screen to identify actual coverage before labeling it.",
                "exact_player_duties_known":False}
    book = rules if rules is not None else load_rules()
    if sid not in book["schemes"]:
        return {"scheme":"unknown", "mechanics":"Coverage unknown; do not infer assignments.",
                "exact_player_duties_known":False}
    return {"scheme":sid, **book["schemes"][sid]}

def load_rules(path: str|Path|None=None) -> dict[str,Any]:
    p=Path(path) if path else Path(__file__).resolve().parents[2]/"data"/"madden27"/"coverage_attack_knowledge.json"
    return json.loads(p.read_text(encoding="utf-8"))


def report(db_path:str|Path, *, rules_path:str|Path|None=None) -> dict[str,Any]:
    rows=read_offense_snap_rows(db_path)
    rules=load_rules(rules_path)
    # Link exact normalized IDs to the coverage rulebook; unknowns and pressure
    # can stay unresolved. Never infer exact coverage off a generic pressure tag.
    covered={r["coverage"] for r in rows if r["coverage"] in KNOWN_COVERAGES}
    return {"schema":VERSION,"audit":audit(rows),"backtest":evaluate_games(rows),
            "backtest_verified_own_calls":evaluate_games(rows,require_verified_execution=True),
            "matchups":matchup_summary(rows),
            "knowledge_status":{"rulebook_schemes":len(rules["schemes"]),
                               "historically_named_schemes":sorted(covered),
                               "unseen_rulebook_schemes":sorted(set(rules["schemes"])-covered),
                               "source":rules["source_url"]}}
