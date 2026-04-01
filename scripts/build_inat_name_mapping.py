#!/usr/bin/env python3
"""Build a model-friendly taxon name mapping from the iNaturalist manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List


CLASS_GENERIC_LABEL = {
    "Actinopterygii": "fish",
    "Amphibia": "amphibian",
    "Aves": "bird",
    "Mammalia": "mammal",
    "Reptilia": "reptile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build common-name mappings for detector/VLM prompts.")
    parser.add_argument("--manifest", default="inat_download_manifest_396.jsonl")
    parser.add_argument("--output-csv", default="data/iNaturalist/inat_name_mapping_396.csv")
    parser.add_argument("--output-jsonl", default="data/iNaturalist/inat_name_mapping_396.jsonl")
    return parser.parse_args()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_label(text: str) -> str:
    text = strip_accents(text).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_aliases(common_name_model: str, generic_label: str) -> List[str]:
    aliases = [common_name_model]
    if generic_label not in aliases:
        aliases.append(generic_label)
    return aliases


def build_record(row: Dict) -> Dict:
    common_name_original = row["common_name"].strip()
    common_name_model = normalize_label(common_name_original)
    scientific_name = row["scientific_name"].strip()
    generic_label = CLASS_GENERIC_LABEL.get(row["taxonomy_class"], "animal")
    aliases = build_aliases(common_name_model, generic_label)

    return {
        "taxon_id": row["taxon_id"],
        "scientific_name": scientific_name,
        "common_name_original": common_name_original,
        "common_name_model": common_name_model,
        "taxonomy_class": row["taxonomy_class"],
        "generic_label": generic_label,
        "grounding_dino_prompt": f"{common_name_model}. {generic_label}.",
        "qwen_label": common_name_model,
        "yolo_open_vocab_label": common_name_model,
        "aliases": "|".join(aliases),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / args.manifest
    output_csv = project_root / args.output_csv
    output_jsonl = project_root / args.output_jsonl
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8") if line.strip()]
    records = [build_record(row) for row in rows]

    fieldnames = [
        "taxon_id",
        "scientific_name",
        "common_name_original",
        "common_name_model",
        "taxonomy_class",
        "generic_label",
        "grounding_dino_prompt",
        "qwen_label",
        "yolo_open_vocab_label",
        "aliases",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_csv}")
    print(f"Wrote {len(records)} records to {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
