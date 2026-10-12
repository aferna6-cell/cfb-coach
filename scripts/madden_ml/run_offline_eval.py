"""Run offline decision-quality scoring and the harness latency pass.

Numbers written under the output directory are copied from
``eval_scoring.evaluate`` and ``eval_harness.run_cli``. This script does not
fill in a count that those runs did not measure.

The scored mode is heuristic. Hybrid is probed once on a fresh temp database.
It is not part of the scored rows unless that probe shows a model was eligible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.madden_ml import eval_harness, eval_scoring  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _shas() -> dict[str, str]:
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git("rev-parse", "HEAD"),
        "base_coordinators": "60cb88eb936503d0334616681f6de43c263dfb3e",
        "pr55_head": "004a541117f50e59fe1c0e2d3dc0f9f7173fb43a",
        "pr56_head": "e8d566075b7bbdb5428071764a29be50cfa4af85",
        "merge_pr55": "2cf32ddffcbe4d600dbb1ce61a9e60cd5f693a1d",
        "merge_pr56": "a77f3b7c1938413d9575f3604f8b8c2f2e794d2f",
    }


def _book_record(db_path: Path) -> dict[str, Any]:
    book = eval_scoring.load_installed_book(db_path)
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.playbook import META_KEY

    db = CoachDB(db_path)
    try:
        raw = json.loads(db.get_meta(META_KEY) or "{}")
    finally:
        db.close()
    applied = {}
    for side, record in (raw.get("applied") or {}).items():
        applied[side] = {
            "mode": record.get("mode"),
            "name": record.get("name"),
            "source_book": record.get("source_book"),
            "formations": list((record.get("formations") or {}).keys()),
        }
    return {"active_playbook_json": applied, "plays_by_formation": book}


def _hybrid_probe(sample: dict[str, Any]) -> dict[str, Any]:
    row = eval_harness.run_situation(
        sample["raw"],
        mode="hybrid",
        opp_look=None,
        side=sample["side"],
        seed=int(sample["seed"]),
    )
    eligible = (
        row.get("fallback_reason") is None
        and row.get("scored_on") == "reviewed"
        and bool(row.get("model_version"))
    )
    return {
        "situation": row.get("situation"),
        "side": row.get("side"),
        "seed": row.get("seed"),
        "requested_mode": "hybrid",
        "new_mode": (row.get("new") or {}).get("mode"),
        "fallback_reason": row.get("fallback_reason"),
        "scored_on": row.get("scored_on"),
        "model_version": row.get("model_version"),
        "mode_downgrade_reason": row.get("mode_downgrade_reason"),
        "new_formation": (row.get("new") or {}).get("formation"),
        "new_play": (row.get("new") or {}).get("play"),
        "old_formation": (row.get("old") or {}).get("formation"),
        "old_play": (row.get("old") or {}).get("play"),
        "formation_and_play_match_old": (
            (row.get("new") or {}).get("formation") == (row.get("old") or {}).get("formation")
            and (row.get("new") or {}).get("play") == (row.get("old") or {}).get("play")
        ),
        "model_eligible": eligible,
        "counted_as_hybrid": eligible,
        "included_in_scored_rows": False,
        "rule": (
            "Hybrid counts only when this fresh-database probe has scored_on "
            "reviewed, a model version, and no fallback. Otherwise the scored "
            "grid stays on the requested mode."
        ),
    }


def _signature_differs(row: dict[str, Any]) -> bool:
    new = row["new"]
    old = row["old"]
    return (
        new.get("formation") != old.get("formation")
        or new.get("play") != old.get("play")
        or new.get("adjustment_id") != old.get("adjustment_id")
    )


def _call_differs(row: dict[str, Any]) -> bool:
    """Formation or play differs. Display text is not part of the comparison."""
    new = row["new"]
    old = row["old"]
    return new.get("formation") != old.get("formation") or new.get("play") != old.get("play")


def _pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "side": row["side"],
        "source": row["source"],
        "tags": list(row["tags"]),
        "opp_look_sent": row["opp_look_sent"],
        "opp_look_bound": row["opp_look"],
        "new_formation": row["new"].get("formation"),
        "new_play": row["new"].get("play"),
        "new_adjustment_id": row["new"].get("adjustment_id"),
        "new_mode": row["new"].get("mode"),
        "old_formation": row["old"].get("formation"),
        "old_play": row["old"].get("play"),
        "old_adjustment_id": row["old"].get("adjustment_id"),
        "old_mode": row["old"].get("mode"),
        "formation_agree": row["formation_agree"],
        "play_agree": row["play_agree"],
        "adjustment_differs": row["adjustment_differs"],
    }


def _look_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sent_null = sent_empty = bound_null = bound_empty = crossed = 0
    for row in rows:
        sent = row["opp_look_sent"]
        personnel = set_name = None
        if isinstance(sent, dict):
            personnel = sent.get("opp_personnel")
            set_name = sent.get("opp_set")
        is_null = personnel is None and set_name is None
        is_empty = personnel == "empty" and set_name == "empty"
        if is_null:
            sent_null += 1
            if row["opp_look"] == "empty":
                crossed += 1
        if is_empty:
            sent_empty += 1
            if row["opp_look"] is None:
                crossed += 1
        if row["opp_look"] is None:
            bound_null += 1
        if row["opp_look"] == "empty":
            bound_empty += 1
    return {
        "sent_null_pairs": sent_null,
        "sent_empty_pairs": sent_empty,
        "bound_null": bound_null,
        "bound_empty": bound_empty,
        "null_and_empty_crossed": crossed,
    }


def _differences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    call_rows = [row for row in rows if _call_differs(row)]
    signature_rows = [row for row in rows if _signature_differs(row)]
    adjustment_only = [
        row
        for row in rows
        if row["adjustment_differs"] and not _call_differs(row)
    ]
    return {
        "definition": (
            "A real call difference is a scored row whose new formation or new "
            "play is not the old caller formation or play. Adjustment-only rows "
            "are counted separately. Display lines are not compared."
        ),
        "real_call_differences": len(call_rows),
        "signature_differences": len(signature_rows),
        "adjustment_only_differences": len(adjustment_only),
        "seed_mismatches": sum(1 for row in rows if not row["seed_match"]),
        "paired_examples": [_pair(row) for row in call_rows],
        "adjustment_only_examples": [_pair(row) for row in adjustment_only],
    }


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    scored_on: dict[str, int] = {}
    for row in rows:
        mode = row["new"].get("mode")
        key = "null" if mode is None else str(mode)
        counts[key] = counts.get(key, 0) + 1
        tag = row.get("scored_on")
        label = "null" if tag is None else str(tag)
        scored_on[label] = scored_on.get(label, 0) + 1
    return {"new_mode": counts, "scored_on": scored_on}


def _situations_for_latency(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    specs = []
    for row in rows:
        specs.append(
            {
                "situation": row["raw"],
                "side": row["side"],
                "mode": mode,
                "seed": int(row["seed"]),
                "opp_look": eval_scoring._look_token(row["opp_look"]),
                "row_id": row["row_id"],
            }
        )
    return specs


def _run_latency(specs: list[dict[str, Any]], passes: int, out: Path) -> int:
    original = eval_harness.run_situation
    started = time.perf_counter()

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        wrapped.n += 1  # type: ignore[attr-defined]
        if wrapped.n == 1 or wrapped.n % 250 == 0:  # type: ignore[attr-defined]
            elapsed = time.perf_counter() - started
            print(
                f"latency calls {wrapped.n}/{len(specs) * passes} elapsed_s {elapsed:.1f}",
                flush=True,
            )
        return original(*args, **kwargs)

    wrapped.n = 0  # type: ignore[attr-defined]
    eval_harness.run_situation = wrapped
    try:
        return eval_harness.run_cli(
            specs,
            passes=passes,
            out=out,
            defaults={"mode": "heuristic", "side": "offense", "opp_look": None, "seed": 0},
        )
    finally:
        eval_harness.run_situation = original


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "/").replace("\n", " ")


def _append_markdown(path: Path, report: dict[str, Any]) -> None:
    run = report["run"]
    diff = run["call_differences"]
    latency = run.get("latency_pass")
    lines = [
        "",
        "## Measured run",
        "",
        "These lines are the run that produced this file. Counts above are the scorer's. Nothing here is an effectiveness claim.",
        "",
        f"- branch: `{run['shas']['branch']}`",
        f"- code head at the start of the run: `{run['shas']['head']}`",
        f"- base `madden-ml/coordinators`: `{run['shas']['base_coordinators']}`",
        f"- PR 55 head: `{run['shas']['pr55_head']}`",
        f"- PR 56 head: `{run['shas']['pr56_head']}`",
        f"- scored mode: `{run['scored_mode']}`",
        f"- new-mode counts: `{json.dumps(run['mode_counts']['new_mode'], sort_keys=True)}`",
        f"- scored_on counts: `{json.dumps(run['mode_counts']['scored_on'], sort_keys=True)}`",
        "",
        "### Hybrid probe",
        "",
        "One fresh-database call, not included in the decision rows.",
        "",
        "```json",
        json.dumps(run["hybrid_probe"], indent=2, sort_keys=True),
        "```",
        "",
        "### Situation and look check",
        "",
        "```json",
        json.dumps(run["look_check"], indent=2, sort_keys=True),
        "```",
        "",
        "### Installed book used for the hard-fail check",
        "",
        "```json",
        json.dumps(run["book"], indent=2, sort_keys=True),
        "```",
        "",
        "### Real call differences",
        "",
        diff["definition"],
        "",
        f"- real call differences (formation or play): {diff['real_call_differences']}",
        f"- signature differences (formation, play, or adjustment): {diff['signature_differences']}",
        f"- adjustment-only differences: {diff['adjustment_only_differences']}",
        f"- same-seed mismatches: {diff['seed_mismatches']}",
        "",
    ]
    examples = diff["paired_examples"]
    if not examples:
        lines.append("No paired examples: no scored row had a different formation or play.")
        lines.append("")
    else:
        lines.append("| row | side | source | look sent | look bound | new formation | new play | old formation | old play |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in examples:
            cells = [
                row["row_id"],
                row["side"],
                row["source"],
                json.dumps(row["opp_look_sent"], sort_keys=True),
                json.dumps(row["opp_look_bound"], sort_keys=True),
                row["new_formation"],
                row["new_play"],
                row["old_formation"],
                row["old_play"],
            ]
            lines.append("| " + " | ".join(_md_cell(value) for value in cells) + " |")
        lines.append("")
    lines.extend(
        [
            "### Adapter",
            "",
            run["adapter_resolution"],
            "",
            "### Latency pass",
            "",
        ]
    )
    if latency is None:
        lines.append("The 50-pass latency run has not been attached.")
    else:
        lines.append(
            "This is `eval_harness.run_cli` over every scored decision row, "
            f"{latency['passes']} passes each, same raw line, side, look token, and seed. "
            "The latency section above is the scorer's first call per row, not this pass."
        )
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(latency, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        whole = latency["whole_call_ms"]
        model = latency["model_ms"]
        lines.append(
            f"- whole p50/p95/max: {whole['p50']} / {whole['p95']} / {whole['max']}"
        )
        lines.append(
            f"- model p50/p95/max: {model['p50']} / {model['p95']} / {model['max']}"
        )
        lines.append(
            f"- fallback counts: {json.dumps(latency['fallback_counts'], sort_keys=True)}"
        )
        lines.append(
            f"- 150 ms budget (harness pass flag, model time): {'pass' if latency['pass'] else 'fail'}"
        )
        if latency.get("fail_reasons"):
            lines.append(f"- fail reasons: {json.dumps(latency['fail_reasons'])}")
    lines.append("")
    path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")


def _scoring(out: Path, db_path: Path, mode: str) -> dict[str, Any]:
    identities, note = eval_scoring.load_situation_identities()
    book = eval_scoring.load_installed_book(db_path)
    calls = {"n": 0}
    started = time.perf_counter()

    def runner(sit: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1 or calls["n"] % 50 == 0:
            print(
                f"score calls {calls['n']} elapsed_s {time.perf_counter() - started:.1f}",
                flush=True,
            )
        return eval_scoring.adapt_run_situation(sit, **kwargs)

    report = eval_scoring.evaluate(
        identities,
        runner,
        book,
        mode=mode,
        inventory_note={**note, "fixture_counts": eval_scoring.fixture_counts(identities)},
    )
    rows = report["decisions"]
    expanded = eval_scoring.expand_decision_rows(identities)
    report["run"] = {
        "disclaimer": eval_scoring.DISCLAIMER,
        "shas": _shas(),
        "scored_mode": mode,
        "score_calls": calls["n"],
        "decision_rows": len(rows),
        "expanded_rows": len(expanded),
        "mode_counts": _mode_counts(rows),
        "look_check": _look_check(rows),
        "call_differences": _differences(rows),
        "book": _book_record(db_path),
        "hybrid_probe": _hybrid_probe(expanded[0]),
        "adapter_resolution": (
            "Git merges of PR 55 and PR 56 had no conflicts. "
            "eval_scoring.evaluate passes the situation dict into adapt_run_situation. "
            "eval_harness.run_situation parses a situation line, and its look normalizer "
            "stringifies a mapping, so NULL and empty would not stay distinct. "
            "On this branch only, adapt_run_situation passes the row's raw line and one "
            "look token: both fields missing is NULL, the empty/empty pair is empty, "
            "and any other pair passes the set token when it is present. "
            "The requested personnel/set pair remains on opp_look_sent."
        ),
        "latency_pass": None,
        "home": os.environ.get("HOME"),
        "cfb_coach_db": os.environ.get("CFB_COACH_DB"),
        "cfb_coach_madden_db": os.environ.get("CFB_COACH_MADDEN_DB"),
    }
    eval_scoring.write_report(report, out)
    _append_markdown(out / "eval_scoring.md", report)
    # write_report rewrote the json before the markdown append. Write json again
    # so the run record matches the markdown.
    (out / "eval_scoring.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote scoring report calls={calls['n']}", flush=True)
    return {"expanded": expanded, "report": report}


def _attach_latency(out: Path, summary: dict[str, Any]) -> None:
    json_path = out / "eval_scoring.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["run"]["latency_pass"] = summary
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Rebuild markdown from the scorer, then append the measured-run section once.
    md_body = eval_scoring.render_markdown(report)
    md_path = out / "eval_scoring.md"
    md_path.write_text(md_body, encoding="utf-8")
    _append_markdown(md_path, report)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Offline coordinator eval")
    parser.add_argument("--out", type=Path, default=_ROOT / "reports" / "offline_eval")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--mode", default="heuristic")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args(argv)
    if args.passes < 50 and not args.skip_latency:
        print("--passes must be >= 50 for this eval", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    if args.skip_scoring:
        report = json.loads((args.out / "eval_scoring.json").read_text(encoding="utf-8"))
        identities, _note = eval_scoring.load_situation_identities()
        expanded = eval_scoring.expand_decision_rows(identities)
    else:
        scored = _scoring(args.out, args.db, args.mode)
        expanded = scored["expanded"]
        report = scored["report"]
    if args.skip_latency:
        return 0
    specs = _situations_for_latency(expanded, args.mode)
    latency_dir = args.out / "latency_pass"
    print(f"latency situations {len(specs)} passes {args.passes}", flush=True)
    code = _run_latency(specs, args.passes, latency_dir)
    summary = json.loads((latency_dir / "summary.json").read_text(encoding="utf-8"))
    summary["cli_exit"] = code
    _attach_latency(args.out, summary)
    print(
        f"latency pass={summary['pass']} samples={summary['samples']} exit={code}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
