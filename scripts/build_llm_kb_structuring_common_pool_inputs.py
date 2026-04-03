from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-bundle inputs for structured species knowledge.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.input).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "class_id": row.get("taxon_id", ""),
                    "common_name": row.get("preferred_common_name", ""),
                    "scientific_name": row.get("species_name", ""),
                    "wikipedia_url": row.get("wikipedia_url", ""),
                    "source_bundle": {
                        "summary_text": "",
                        "section_texts": {
                            "appearance": "",
                            "habitat": "",
                            "diet": "",
                            "behavior": "",
                            "distribution": ""
                        }
                    }
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
