#!/usr/bin/env python3
"""Read-only history -> no-private-snap aggregate defense research report.

Example:
 python scripts/madden_ml/eval_historical_defense_baseline.py \
   --db /path/to/madden27.backup.db --output-dir /tmp/cfb-coverage-baseline
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from cfb_coach.madden.model import historical_defense_baseline as hb


def write_report(db:Path,output_dir:Path)->dict:
    report=hb.report(db)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"historical_defense_baseline_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    # Summary exposes play names but no usernames, session ids, timestamps or raw outcomes.
    m=report["matchups"]
    rows=m["matchup_cells_at_min_n"]
    cols=("formation","play","coverage","n","observed_success_rate","shrunk_rate","approx_interval_95")
    with (output_dir/"historical_defense_matchups.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in cols})
    s=report["audit"];e=report["backtest"];ev=report["backtest_verified_own_calls"];x=report["matchups"]
    lines=["# Madden 27 — historical opponent coverage baseline", "",
           "**Research only.** This report uses a read-only database backup. Nothing runs in the live coach.","",
           "## Data provenance", "",
           f"- Our offense-side snaps: **{s['offense_snaps']}** (opponents were on defense).",
           f"- Named or pressure-only historical labels: **{s['coverage_or_pressure_labels']}**.",
           f"- Named coverages: **{s['named_coverage_labels']}**; pressure-only: **{s['pressure_only_labels']}**.",
           f"- Our verified executed offensive snaps: **{s['own_executions_verified']}**.",
           f"- Named coverage + our verified execution: **{s['named_coverage_with_own_execution_verified']}**.",
           "- Opponent call labels are stored in `coverage_seen` and are **not independently screen-verified** by this DB.",
           "- We **do not** interpret `pressure` as Cover 0.","",
           "## Chronological whole-game coverage prediction test", "",
           f"Evaluated {e['snap_labels_tested']} historical named labels across {e['games_held_out']} held-out sessions, with first {e['min_train_games']} sessions used for training.",
           "The target is agreement with stored labels, **not proven Madden post-play ground truth**.", "",
           "| Baseline | Brier ↓ | Log loss ↓ | Top-1 ↑ |", "|---|---:|---:|---:|"]
    for mode,res in e['results'].items():
        lines.append(f"| {mode} | {res['brier']} | {res['log_loss']} | {res['top1_accuracy']} |")
    lines += ["", "### Stricter slice: our executed play verified", "",
             f"{ev['snap_labels_tested']} labels evaluated in {ev['games_held_out']} held-out sessions; opponent coverage source still unverified.",
             "", "| Baseline | Brier ↓ | Log loss ↓ | Top-1 ↑ |", "|---|---:|---:|---:|"]
    for mode,res in ev['results'].items():
        lines.append(f"| {mode} | {res['brier']} | {res['log_loss']} | {res['top1_accuracy']} |")
    lines += ["", "## Matchup associations", "",
             f"{x['usable_snaps']} verified executed offensive snaps support at least one named coverage + evaluable outcome.",
             f"Found **{len(rows)}** named play/coverage pairs and **{len(x['formation_cells_at_min_n'])}** formation/coverage pairs with at least {x['min_cell']} examples.",
             "This is observational, using standard 40/60/100 success; it is not a randomized counter-play experiment.",
             "Small cells should not be used as evidence of a guaranteed play-beats-coverage relation.","",
             "## Next pilot requirements", "",
             "1. Capture Madden's displayed **post-play opponent defensive call** with screenshot/frame locator and confidence. Store it separate from pre-snap predictions.",
             "2. Replay the scorer before the snap using only prior games, plus current pre-snap alignment observations, with the revealed call hidden.",
             "3. Evaluate the structural offense model (with the 91 route-art feature records) on held-out outcomes; do not train/evaluate on the same game.",
             "4. For QB read advice, require player-identified routes; visual geometry alone cannot truthfully label 'post first, drag second' for a specific play.",
             "5. Before CPU pilot, confirm no regression of 150ms latency, legal play selection, and no unapproved macro/action.",""]
    (output_dir/"HISTORICAL_DEFENSE_BASELINE.md").write_text("\n".join(lines),encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--db",required=True,type=Path,help="Path to a read-only snapshot, not the active DB")
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args(argv)
    r=write_report(a.db,a.output_dir)
    print(json.dumps({"audit":r["audit"],"backtest":r["backtest"],"shown_matchup_cells":len(r["matchups"]["matchup_cells_at_min_n"])},indent=2))

if __name__=="__main__":main()
