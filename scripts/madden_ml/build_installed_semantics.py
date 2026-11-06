"""Build a standalone, research-only Madden 27 concept/coverage database.

The 216 play identities come from the user's recorded installed-book snapshot.
Official Madden27 coverage semantics are curated; other public listings are
used for cross-checking catalog membership, not unauthorized play-art scraping.

Usage: python scripts/madden_ml/build_installed_semantics.py \
    --output /tmp/madden27-installed-semantics.sqlite
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

from cfb_coach.madden.model.coverage_intelligence import (
    EA, MATCH_CHECKS, MUT, SCHEMES, classify_defensive_play,
)
from cfb_coach.madden.model.installed_playbook_catalog import validate_snapshot
from cfb_coach.madden.model.play_knowledge import play_key

SCHEMA = "madden27.semantic_plays.v1"


def classify_offense(name: str) -> list[str]:
    """Lexical concept hints ONLY; no claim any specific WR is on a route."""
    n = name.lower()
    hints: set[str] = set()
    terms = {
        "mesh":"mesh", "slant":"slants", "stick":"stick", "spot":"spot",
        "flood":"flood", "smash":"smash", "vertical":"verticals",
        "curl":"curl", "hitch":"hitch", "cross":"crossing", "drag":"drag",
        "dig":"dig", "post":"post", "wheel":"wheel", "bench":"bench",
        "screen":"screen", "rpo":"rpo", "inside zone":"inside_zone",
        "read option":"read_option", "counter":"counter", "stretch":"stretch",
        "sweep":"sweep", "draw":"draw", "power":"power",
        "pa ":"play_action", "pa boot":"play_action", "mtn":"motion_name",
        "quick":"quick_name", "salem":"salem_name", "levels":"levels",
    }
    for term,tag in terms.items():
        if term in n: hints.add(tag)
    return sorted(hints)


def build(snapshot_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing research database: {output}")
    if (Path.home()/".cfb-coach").resolve() in output.resolve().parents:
        raise ValueError("Cannot write research tables into live game configuration")
    snapshot=json.loads(snapshot_path.read_text(encoding="utf-8"))
    validate_snapshot(snapshot)
    output.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(output)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript("""
        CREATE TABLE plays (
            play_key TEXT PRIMARY KEY, side TEXT NOT NULL,
            book TEXT NOT NULL, formation TEXT NOT NULL, play TEXT NOT NULL,
            knowledge_tier TEXT NOT NULL, scheme_id TEXT,
            scheme_family TEXT, concept_hints_json TEXT NOT NULL,
            variant_tags_json TEXT NOT NULL,
            verified_player_duties INTEGER NOT NULL DEFAULT 0,
            UNIQUE(side,book,formation,play));
        CREATE TABLE scheme_definitions (
            scheme_id TEXT PRIMARY KEY, family TEXT NOT NULL,
            deep_structure TEXT NOT NULL, description TEXT NOT NULL,
            bonus_defender TEXT, quarter_side TEXT, half_side TEXT,
            rule_json TEXT NOT NULL, sources_json TEXT NOT NULL,
            explanation_scope TEXT NOT NULL);
        CREATE TABLE match_checks (
            check_id TEXT PRIMARY KEY, explanation TEXT NOT NULL,
            source_url TEXT NOT NULL, verification TEXT NOT NULL);
        CREATE INDEX idx_scheme ON plays(scheme_id);
        """)
        for sid, scheme in SCHEMES.items():
            con.execute("INSERT INTO scheme_definitions VALUES(?,?,?,?,?,?,?,?,?,?)",(
                sid, scheme.family, scheme.deep_structure, scheme.philosophy,
                scheme.bonus_defender, scheme.quarter_side, scheme.half_side,
                json.dumps(scheme.conditional_rules),json.dumps(scheme.sources),scheme.scope))
        for k,v in MATCH_CHECKS.items():
            con.execute("INSERT INTO match_checks VALUES (?,?,?,?)",(
                k,v,EA,"official_mechanic_general_not_selected_play"))
        byside={"offense":0,"defense":0}
        families={}
        review=[]
        for side in ("offense","defense"):
            section=snapshot["sides"][side]
            for formation, names in section["formations"].items():
                book=section["formation_sources"][formation]
                for name in names:
                    pkey=play_key(side,book,formation,name)
                    coverage=classify_defensive_play(name) if side=="defense" else None
                    concepts=classify_offense(name) if side=="offense" else []
                    key=coverage.scheme_id if coverage else ""
                    fam=coverage.family if coverage else ""
                    ver=list(coverage.named_variations) if coverage else []
                    status="name_inferred_scheme_not_player_verified" if coverage else "name_inferred_concept_not_player_verified"
                    con.execute("INSERT INTO plays VALUES(?,?,?,?,?,?,?,?,?,?,0)",(
                        pkey,side,book,formation,name,status,key or None,fam or None,
                        json.dumps(concepts),json.dumps(ver)))
                    byside[side]+=1
                    if coverage:families[key]=families.get(key,0)+1
                    review.append({"side":side,"book":book,"formation":formation,"play":name,
                                   "scheme":key,"family":fam,"concept_hints":";".join(concepts),
                                   "named_variants":";".join(ver),"status":status,
                                   "verified_11_player_jobs":0})
        con.commit()
        if byside!={"offense":93,"defense":123}:
            raise AssertionError(f"incorrect installed-book count: {byside}")
        out_csv=output.with_suffix(".review.csv")
        with out_csv.open("w",newline="",encoding="utf-8") as f:
            wr=csv.DictWriter(f,fieldnames=list(review[0]));wr.writeheader();wr.writerows(review)
        # Offline features: explicit football priors, never verified player jobs.
        # A model training pipeline must keep these separate from game outcomes.
        features_path=output.with_suffix(".research_features.jsonl")
        with features_path.open("w", encoding="utf-8") as f:
            for row in con.execute("SELECT play_key,side,book,formation,play,scheme_id,scheme_family,"
                                   "concept_hints_json,variant_tags_json,knowledge_tier FROM plays ORDER BY side,formation,play"):
                key,side_,book,formation,play,scheme_id,family,concepts,variants,tier=row
                data={"play_key":key,"side":side_,"book":book,"formation":formation,"play":play,
                      "scheme_id":scheme_id,"scheme_family":family,
                      "concept_hints":json.loads(concepts),"variant_hints":json.loads(variants),
                      "evidence":tier,"interpretation":"football_prior_only_not_observed_player_assignments"}
                f.write(json.dumps(data,sort_keys=True)+"\n")
        report={"schema":SCHEMA,"catalog_rows":byside,"coverage_classes":families,
                "match_check_variants":len(MATCH_CHECKS),
                "complete_player_assigned_plays":0,
                "scope":"concept and scheme hypotheses only; 0 new verified player responsibilities",
                "source_scope":"original coverage principles summarized from official EA and public educational descriptions"}
        output.with_suffix(".report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        return report
    except Exception:
        con.close()
        output.unlink(missing_ok=True)
        raise
    finally:
        try:con.close()
        except Exception:pass


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[2]
    parser.add_argument("--snapshot",type=Path,default=root/"cfb_coach/data/madden27/installed_playbook_snapshot.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.snapshot,args.output),indent=2))


if __name__=="__main__":main()