"""Train the v0.7 play-call model and write the box output files.

Outputs land in ``<models_dir>/<label version>/``:

* ``model_params.json``
* ``metrics.json``
* ``beaters.json``
* ``metrics.md``
* ``manifest.json``

``<models_dir>/LAST_TRAINED.json`` points at that directory. CPU and human
are separate strata. Unknown opponent rows are counted and then left out of
the fit.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from cfb_coach.vod_success.beaters import TENTATIVE_N, build_stratum_beaters
from cfb_coach.vod_success.features import ALL_GROUPS, SITUATION_GROUPS, Snap, snaps_from_rows
from cfb_coach.vod_success.glm import brier, fit_model, log_loss, reliability
from cfb_coach.vod_success.load import load_many

SCRIPT_VERSION = "train_vod_model.v2"
OUTPUTS = [
    "model_params.json",
    "metrics.json",
    "beaters.json",
    "metrics.md",
    "manifest.json",
]
SITUATION_LAMBDAS = (1.0, 4.0, 16.0)
ENTITY_LAMBDAS = (1.0, 4.0, 16.0, 64.0, 256.0)
INTERACTION_LAMBDAS = (16.0, 128.0, 1024.0)
DEFAULT_HP = (2.0, 16.0, 64.0)
MIN_VODS_EVAL = 3
MIN_ROWS_EVAL = 40
N_BINS = 10
CONF_FLOOR = 0.6

Hp = tuple[float, float, float]


def default_grid() -> list[Hp]:
    grid: list[Hp] = []
    for situation in SITUATION_LAMBDAS:
        for entity in ENTITY_LAMBDAS:
            for interaction in INTERACTION_LAMBDAS:
                grid.append((float(situation), float(entity), float(interaction)))
    return grid


def version_dirname(label_version: str) -> str:
    text = label_version.strip()
    if not text.startswith("v"):
        text = "v" + text
    return text


def train_and_write(
    rows: list[dict[str, Any]],
    models_dir: str | Path,
    *,
    label_version: str,
    input_paths: list[str | Path] | None = None,
    labels_dir: str | Path | None = None,
    latest_json_path: str | Path | None = None,
    snaps_notes_path: str | Path | None = None,
    grid: list[Hp] | None = None,
    min_vods_eval: int = MIN_VODS_EVAL,
    min_rows_eval: int = MIN_ROWS_EVAL,
) -> dict[str, Any]:
    """Fit every stratum and write the five output files plus LAST_TRAINED."""
    snaps = snaps_from_rows(rows)
    grid = list(grid) if grid is not None else default_grid()
    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    by_stratum: dict[str, list[Snap]] = defaultdict(list)
    for snap in snaps:
        by_stratum[snap.stratum].append(snap)

    order = sorted(by_stratum, key=_stratum_sort)
    fitted: dict[str, dict[str, Any]] = {}
    for name in order:
        kind = name.split("/", 1)[-1]
        if kind not in {"cpu", "human"}:
            continue
        known = [snap for snap in by_stratum[name] if snap.known]
        if not known:
            continue
        fitted[name] = _fit_stratum(
            known,
            grid=grid,
            min_vods=min_vods_eval,
            min_rows=min_rows_eval,
        )

    version = version_dirname(label_version)
    root = Path(models_dir)
    dest = root / version
    dest.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    script_sha = _sha256_file(script_path)
    labels_meta = _labels_meta(
        snaps,
        input_paths or [],
        labels_dir=Path(labels_dir) if labels_dir else None,
        latest_json_path=Path(latest_json_path) if latest_json_path else None,
        snaps_notes_path=Path(snaps_notes_path) if snaps_notes_path else None,
        label_version=version,
        trained_at=trained_at,
    )

    params = {
        "label_version": version,
        "script_version": SCRIPT_VERSION,
        "strata": {name: _params_stratum(fitted[name]) for name in fitted},
    }
    metrics = _metrics_doc(
        snaps,
        by_stratum,
        fitted,
        grid=grid,
        min_vods=min_vods_eval,
        min_rows=min_rows_eval,
        label_version=version,
        trained_at=trained_at,
    )
    beaters = {
        "label_version": version,
        "tentative_n": TENTATIVE_N,
        "strata": {
            name: build_stratum_beaters(
                fitted[name]["known"],
                fitted[name]["situation_model"],
                opponent_type=name.split("/", 1)[1],
            )
            for name in fitted
        },
    }
    manifest = _manifest(
        version=version,
        trained_at=trained_at,
        script_path=script_path,
        script_sha=script_sha,
        labels_meta=labels_meta,
        rows_loaded=len(snaps),
        models_dir=dest,
    )
    _write_json(dest / "model_params.json", params)
    _write_json(dest / "metrics.json", metrics)
    _write_json(dest / "beaters.json", beaters)
    (dest / "metrics.md").write_text(_metrics_md(metrics), encoding="utf-8")
    _write_json(dest / "manifest.json", manifest)
    last = {
        "label_version": version,
        "latest_sha256": labels_meta["latest_sha256"],
        "labels_data_sha256": labels_meta["labels_data_sha256"],
        "input_signature": labels_meta["input_signature"],
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha,
        "model_dir": str(dest),
        "trained_at": trained_at,
    }
    _write_json(root / "LAST_TRAINED.json", last)
    return {"model_dir": str(dest), "last": last, "metrics": metrics, "beaters": beaters, "params": params}


def _fit_stratum(
    known: list[Snap],
    *,
    grid: list[Hp],
    min_vods: int,
    min_rows: int,
) -> dict[str, Any]:
    vods = sorted({snap.vod for snap in known})
    by_vod: dict[str, list[Snap]] = defaultdict(list)
    for snap in known:
        by_vod[snap.vod].append(snap)
    use_default = len(vods) < min_vods
    if use_default:
        hp: Hp = DEFAULT_HP
        hp_selection: list | str = f"default (fewer than {min_vods} VODs for inner CV)"
        situation_hp = DEFAULT_HP
    else:
        scored = [(candidate, _lovo_loss(by_vod, vods, candidate, situation_only=False)) for candidate in grid]
        hp = min(scored, key=lambda item: item[1])[0]
        hp_selection = [
            {"hp": [a, b, c], "inner_logloss": round(loss, 6)}
            for (a, b, c), loss in scored
        ]
        sit_scored = [
            (candidate, _lovo_loss(by_vod, vods, candidate, situation_only=True))
            for candidate in grid
        ]
        situation_hp = min(sit_scored, key=lambda item: item[1])[0]
    full = fit_model(known, hp, situation_only=False)
    situation = fit_model(known, situation_hp, situation_only=True)
    evaluated = len(vods) >= min_vods and len(known) >= min_rows
    result: dict[str, Any] = {
        "known": known,
        "vods": vods,
        "hp": hp,
        "hp_selection": hp_selection,
        "full": full,
        "situation_model": situation,
        "situation_hp": situation_hp,
        "evaluated": evaluated,
    }
    if not evaluated:
        result["reason"] = _too_small(len(known), len(vods), min_vods, min_rows)
        return result
    fold_rows = []
    held_probs: dict[str, list[tuple[Snap, float]]] = {
        "model": [],
        "situation": [],
        "overall": [],
        "situation_base": [],
        "call_base": [],
    }
    for held in vods:
        train_vods = [vod for vod in vods if vod != held]
        train_rows = [snap for vod in train_vods for snap in by_vod[vod]]
        test_rows = by_vod[held]
        if not train_rows or not test_rows:
            continue
        fold_hp = _select_hp(by_vod, train_vods, grid)
        fold_model = fit_model(train_rows, fold_hp, situation_only=False)
        fold_sit = fit_model(train_rows, fold_hp, situation_only=True)
        base = _positive_rate(train_rows)
        call_rates = _group_rate(train_rows, lambda snap: snap.call_key or "none", base)
        sit_rates = _group_rate(train_rows, _situation_key, base)
        model_probs = []
        for snap in test_rows:
            model_p = fold_model.predict_row(_full_feats(snap))
            sit_p = fold_sit.predict_row(_sit_feats(snap))
            model_probs.append(model_p)
            held_probs["model"].append((snap, model_p))
            held_probs["situation"].append((snap, sit_p))
            held_probs["overall"].append((snap, base))
            held_probs["situation_base"].append((snap, sit_rates(_situation_key(snap))))
            held_probs["call_base"].append((snap, call_rates(snap.call_key or "none")))
        fold_rows.append(
            {
                "held_out_vod": held,
                "n_test": len(test_rows),
                "n_train": len(train_rows),
                "hp": [float(v) for v in fold_hp],
                "test_obs_rate": round(_positive_rate(test_rows), 4),
                "model_log_loss": round(log_loss(model_probs, [bool(s.success) for s in test_rows]), 4),
                "overall_base_log_loss": round(
                    log_loss([base] * len(test_rows), [bool(s.success) for s in test_rows]),
                    4,
                ),
            }
        )
    result["folds"] = fold_rows
    result["held_probs"] = held_probs
    return result


def _select_hp(by_vod, vods: list[str], grid: list[Hp]) -> Hp:
    if len(vods) < 2:
        return DEFAULT_HP
    scored = [(candidate, _lovo_loss(by_vod, vods, candidate, situation_only=False)) for candidate in grid]
    return min(scored, key=lambda item: item[1])[0]


def _lovo_loss(by_vod, vods: list[str], hp: Hp, *, situation_only: bool) -> float:
    probs: list[float] = []
    labels: list[bool] = []
    # Situation-only loss depends only on the situation penalty. Callers still
    # pass the full triple so the recorded grid stays aligned with the box.
    for held in vods:
        train_rows = [snap for vod in vods if vod != held for snap in by_vod[vod]]
        test_rows = by_vod[held]
        if not train_rows or not test_rows:
            continue
        model = fit_model(train_rows, hp, situation_only=situation_only)
        for snap in test_rows:
            feats = _sit_feats(snap) if situation_only else _full_feats(snap)
            probs.append(model.predict_row(feats))
            labels.append(bool(snap.success))
    return log_loss(probs, labels)


def _params_stratum(fitted: dict[str, Any]) -> dict[str, Any]:
    known: list[Snap] = fitted["known"]
    return {
        "n_train_rows": len(known),
        "base_rate": round(_positive_rate(known), 4),
        "hp_selection": fitted["hp_selection"],
        "model": fitted["full"].as_params(ALL_GROUPS),
        "situation_only_model": fitted["situation_model"].as_params(SITUATION_GROUPS),
    }


def _metrics_doc(
    snaps: list[Snap],
    by_stratum: dict[str, list[Snap]],
    fitted: dict[str, dict[str, Any]],
    *,
    grid: list[Hp],
    min_vods: int,
    min_rows: int,
    label_version: str,
    trained_at: str,
) -> dict[str, Any]:
    counts = {}
    for name in sorted(by_stratum, key=_stratum_sort):
        group = by_stratum[name]
        known = [snap for snap in group if snap.known]
        clean = [snap for snap in group if snap.clean]
        clean_known = [snap for snap in known if snap.clean]
        with_call = [snap for snap in known if snap.call_key]
        with_look = [snap for snap in known if snap.look_family != "none"]
        sources: dict[str, int] = defaultdict(int)
        for snap in group:
            sources[snap.type_source] += 1
        side_counts: dict[str, int] = defaultdict(int)
        for snap in known:
            side_counts[snap.side] += 1
        vods = {snap.vod for snap in group}
        vods_known = {snap.vod for snap in known}
        positives = sum(1 for snap in known if snap.success)
        counts[name] = {
            "rows": len(group),
            "scrimmage_rows": sum(1 for snap in group if snap.scrimmage),
            "known_success_rows": len(known),
            "success_rate_known": round(positives / len(known), 4) if known else 0.0,
            "clean_pairs": len(clean),
            "clean_pairs_known_success": len(clean_known),
            "rows_with_call_known_success": len(with_call),
            "rows_with_look_known_success": len(with_look),
            "vods": len(vods),
            "vods_with_known_success": len(vods_known),
            "known_success_conf_ge_0.6": sum(
                1 for snap in known if snap.conf_success is not None and snap.conf_success >= CONF_FLOOR
            ),
            "opponent_type_sources": dict(sources),
            "streamer_side_known_success": {
                side: side_counts[side]
                for side in ("offense", "defense", "unknown")
                if side_counts.get(side)
            },
            "trained": name in fitted,
        }
    strata = {}
    for name, fit in fitted.items():
        entry: dict[str, Any] = {
            "n_train_rows": len(fit["known"]),
            "vods": list(fit["vods"]),
        }
        if fit["evaluated"]:
            entry["status"] = "evaluated"
            entry["held_out"] = _held_out(fit)
            entry["folds"] = fit["folds"]
        else:
            entry["status"] = "not_evaluated"
            entry["reason"] = fit["reason"]
        entry["final_hp"] = [float(v) for v in fit["hp"]]
        strata[name] = entry
    success_source: dict[str, int] = defaultdict(int)
    for snap in snaps:
        success_source[snap.success_source] += 1
    unknown = [snap for snap in snaps if snap.opponent_type == "unknown"]
    return {
        "label_version": label_version,
        "script_version": SCRIPT_VERSION,
        "counts": counts,
        "strata": strata,
        "settings": {
            "min_vods_eval": min_vods,
            "min_rows_eval": min_rows,
            "grid": [[a, b, c] for a, b, c in grid],
            "default_hp": [float(v) for v in DEFAULT_HP],
            "conf_min_names": 0.5,
            "n_bins": N_BINS,
            "split": (
                "outer leave-one-VOD-out; hyperparameters chosen by inner "
                "leave-one-VOD-out on the training VODs only"
            ),
        },
        "success_source": dict(success_source),
        "excluded_unknown_opponent": {
            "rows": len(unknown),
            "known_success": sum(1 for snap in unknown if snap.known),
            "clean_pairs": sum(1 for snap in unknown if snap.clean),
            "vods": sorted({snap.vod for snap in unknown}),
        },
        "trained_at": trained_at,
    }


def _held_out(fit: dict[str, Any]) -> dict[str, Any]:
    packs = {
        "all_known": lambda snap: True,
        "clean_pairs": lambda snap: snap.clean,
        "rows_with_call": lambda snap: bool(snap.call_key),
        "conf_success_ge_0.6": (
            lambda snap: snap.conf_success is not None and snap.conf_success >= CONF_FLOOR
        ),
    }
    out = {}
    for slice_name, keep in packs.items():
        chosen_ids = {id(snap) for snap, _prob in fit["held_probs"]["model"] if keep(snap)}
        chosen = [snap for snap, _prob in fit["held_probs"]["model"] if id(snap) in chosen_ids]
        positives = sum(1 for snap in chosen if snap.success)
        block: dict[str, Any] = {
            "n": len(chosen),
            "positives": positives,
            "obs_rate": round(positives / len(chosen), 4) if chosen else 0.0,
        }
        for label, key in (
            ("model", "model"),
            ("situation_only_model", "situation"),
            ("overall_base_rate", "overall"),
            ("situation_base_rate", "situation_base"),
            ("call_base_rate", "call_base"),
        ):
            pairs = [(snap, prob) for snap, prob in fit["held_probs"][key] if id(snap) in chosen_ids]
            block[label] = _score_block(pairs)
        out[slice_name] = block
    return out


def _score_block(pairs: list[tuple[Snap, float]]) -> dict[str, Any]:
    probs = [prob for _snap, prob in pairs]
    labels = [bool(snap.success) for snap, _prob in pairs]
    ece, bins = reliability(probs, labels, n_bins=N_BINS)
    return {
        "log_loss": round(log_loss(probs, labels), 4),
        "brier": round(brier(probs, labels), 4),
        "ece": round(ece, 4),
        "reliability": bins,
    }


def _manifest(
    *,
    version: str,
    trained_at: str,
    script_path: Path,
    script_sha: str,
    labels_meta: dict[str, Any],
    rows_loaded: int,
    models_dir: Path,
) -> dict[str, Any]:
    return {
        "label_version": version,
        "trained_at": trained_at,
        "script": str(script_path),
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha,
        "latest_json": labels_meta["latest_json"],
        "latest_json_sha256": labels_meta["latest_sha256"],
        "labels_dir": labels_meta["labels_dir"],
        "label_files": labels_meta["label_files"],
        "labels_data_sha256": labels_meta["labels_data_sha256"],
        "label_side_files_sha256": labels_meta["label_side_files_sha256"],
        "snaps_notes_sha256": labels_meta["snaps_notes_sha256"],
        "rows_loaded": rows_loaded,
        "outputs": list(OUTPUTS),
        "rerun": f"python3 -m cfb_coach.vod_success train --models-dir {models_dir.parent} --label-version {version}",
    }


def _labels_meta(
    snaps: list[Snap],
    input_paths: list[str | Path],
    *,
    labels_dir: Path | None,
    latest_json_path: Path | None,
    snaps_notes_path: Path | None,
    label_version: str,
    trained_at: str,
) -> dict[str, Any]:
    files = _input_files(input_paths)
    data_hash = _sha256_bytes(_concat(files)) if files else _sha256_bytes(_notes_bytes(snaps))
    signature = data_hash[:12]
    latest_payload: dict[str, Any]
    if latest_json_path and latest_json_path.is_file():
        latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))
        latest_bytes = latest_json_path.read_bytes()
    else:
        latest_payload = {
            "latest": label_version,
            "path": str(labels_dir) if labels_dir else "",
            "schema_version": "vod_labels.v0",
            "vods": len({snap.vod for snap in snaps}),
            "rows": len(snaps),
            "clean_pairs": sum(1 for snap in snaps if snap.clean),
            "script_version": SCRIPT_VERSION,
            "input_signature": signature,
            "created_at": trained_at,
            "note": "Each label build is a full re-run into a new v0.<N>; old dirs are never overwritten.",
        }
        latest_bytes = json.dumps(latest_payload, sort_keys=True).encode("utf-8")
    side = {}
    if labels_dir and labels_dir.is_dir():
        for name in ("manifest.json", "summary.json", "accuracy.json"):
            path = labels_dir / name
            if path.is_file():
                side[name] = _sha256_file(path)
    if snaps_notes_path and snaps_notes_path.is_file():
        notes_hash = _sha256_file(snaps_notes_path)
    else:
        notes_hash = _sha256_bytes(_notes_bytes(snaps))
    label_files = [path.name for path in files] if files else []
    return {
        "latest_json": latest_payload,
        "latest_sha256": hashlib.sha256(latest_bytes).hexdigest(),
        "labels_dir": str(labels_dir) if labels_dir else "",
        "label_files": label_files,
        "labels_data_sha256": data_hash,
        "label_side_files_sha256": side,
        "snaps_notes_sha256": notes_hash,
        "input_signature": signature,
    }


def _metrics_md(metrics: dict[str, Any]) -> str:
    lines = [
        f"# VOD success model {metrics['label_version']}",
        "",
        f"Trained {metrics['trained_at']} with `{metrics['script_version']}`.",
        "",
        "| Stratum | Rows | Known success | Rate | Trained | Eval |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, counts in metrics["counts"].items():
        stratum = metrics["strata"].get(name) or {}
        lines.append(
            f"| {name} | {counts['rows']} | {counts['known_success_rows']} | "
            f"{counts['success_rate_known']:.0%} | {counts['trained']} | {stratum.get('status', 'not trained')} |"
        )
    lines.append("")
    lines.append("Unknown opponent rows are counted and excluded from the fit.")
    lines.append("")
    return "\n".join(lines)


def _too_small(n_rows: int, n_vods: int, min_vods: int, min_rows: int) -> str:
    if n_vods < min_vods:
        return (
            f"too small for a held-out evaluation: only {n_vods} VOD(s) with known-success rows "
            f"(need >= {min_vods} for leave-one-VOD-out) [{n_rows} known rows, {n_vods} VODs]"
        )
    return (
        f"too small for a held-out evaluation: only {n_rows} known-success rows "
        f"(need >= {min_rows}) [{n_rows} known rows, {n_vods} VODs]"
    )


def _positive_rate(snaps: list[Snap]) -> float:
    if not snaps:
        return 0.0
    return sum(1.0 for snap in snaps if snap.success) / len(snaps)


def _group_rate(snaps: list[Snap], key, fallback: float):
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for snap in snaps:
        bucket = totals[key(snap)]
        bucket[0] += 1
        if snap.success:
            bucket[1] += 1

    def rate(name: str) -> float:
        n, s = totals.get(name, (0, 0))
        if n <= 0:
            return fallback
        return s / n

    return rate


def _situation_key(snap: Snap) -> str:
    return f"{snap.down or 'unk'}|{snap.dist or 'unk'}|{snap.zone or 'unk'}"


def _full_feats(snap: Snap):
    from cfb_coach.vod_success.features import feature_map

    return feature_map(snap, situation_only=False)


def _sit_feats(snap: Snap):
    from cfb_coach.vod_success.features import feature_map

    return feature_map(snap, situation_only=True)


def _stratum_sort(name: str) -> tuple:
    game, _, kind = name.partition("/")
    game_order = {"madden27": 0, "cfb27": 1}.get(game, 2)
    kind_order = {"cpu": 0, "human": 1, "unknown": 2}.get(kind, 3)
    return (game_order, kind_order, name)


def _input_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                sorted(
                    p for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".gz"}
                    and p.name != "manifest.jsonl"
                )
            )
        elif path.is_file():
            files.append(path)
    return files


def _concat(files: list[Path]) -> bytes:
    chunks = []
    for path in files:
        chunks.append(path.name.encode("utf-8") + b"\0" + path.read_bytes())
    return b"".join(chunks)


def _notes_bytes(snaps: list[Snap]) -> bytes:
    return "\n".join(sorted({snap.vod for snap in snaps})).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_rows(paths: list[str | Path], manifest: str | Path | None = None) -> list[dict[str, Any]]:
    return load_many(paths, manifest=manifest)
