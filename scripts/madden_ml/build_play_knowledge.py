"""Create a SEPARATE research play-knowledge DB from Madden 27 catalog names.

Usage: python scripts/madden_ml/build_play_knowledge.py --output /tmp/play-knowledge.db
Never writes the live ~/.cfb-coach database or changes the current playcaller.
"""
import argparse
import json
from pathlib import Path

from cfb_coach.madden.model.play_knowledge import PlayKnowledgeDB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=Path(__file__).resolve().parents[2] / "cfb_coach/data/madden27/playbooks_full.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    live_dir = (Path.home() / ".cfb-coach").resolve()
    if output == live_dir or live_dir in output.parents:
        parser.error("research DB must be outside the live CFB Coach data directory")
    if output.exists():
        parser.error("output exists; refusing to overwrite")
    if not args.catalog.is_file():
        parser.error(f"catalog does not exist: {args.catalog}")
    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with PlayKnowledgeDB(output) as db:
        counts = db.import_catalog(payload)
        print(json.dumps({"imported_play_entries": counts, "coverage": db.coverage(),
                          "note": "Names imported; player assignments remain UNKNOWN until annotated."}, indent=2))


if __name__ == "__main__":
    main()