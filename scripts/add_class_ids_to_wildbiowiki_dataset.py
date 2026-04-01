#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add class_id fields to the WildBioWiki-QA detection dataset from the QA mapping file."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/WildBioWiki-QA",
        help="Root directory of the finalized WildBioWiki-QA dataset.",
    )
    parser.add_argument(
        "--qa-file",
        default="data/WildBioWiki-QA/manual_review_usable396_qa_final.jsonl",
        help="QA file containing class_id to (taxonomy_class, common_name) mapping.",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_class_id_map(qa_file: Path) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for row in load_jsonl(qa_file):
        key = (row["taxonomy_class"], row["common_name"])
        class_id = str(row["class_id"])
        if key in mapping and mapping[key] != class_id:
            raise ValueError("Conflicting class_id mapping for %s" % (key,))
        mapping[key] = class_id
    return mapping


def add_class_ids_to_jsonl(path: Path, mapping: Dict[Tuple[str, str], str]) -> None:
    rows = []
    for row in load_jsonl(path):
        key = (row["taxonomy_class"], row["common_name"])
        if key not in mapping:
            raise KeyError("Missing class_id mapping for %s in %s" % (key, path))
        row["class_id"] = mapping[key]
        rows.append(row)
    write_jsonl(path, rows)


def main() -> int:
    args = parse_args()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    qa_file = (PROJECT_ROOT / args.qa_file).resolve()
    mapping = build_class_id_map(qa_file)

    manifests_dir = dataset_root / "manifests"
    for path in sorted(manifests_dir.glob("*.jsonl")):
        add_class_ids_to_jsonl(path, mapping)

    annotations_dir = dataset_root / "annotations"
    for path in sorted(annotations_dir.rglob("*.jsonl")):
        add_class_ids_to_jsonl(path, mapping)

    summaries_dir = dataset_root / "summaries"
    for path in sorted(summaries_dir.glob("*.summary.json")):
        data = json.loads(path.read_text())
        for species in data.get("species", []):
            key = (data["taxonomy_class"], species["common_name"])
            if key not in mapping:
                raise KeyError("Missing class_id mapping for %s in %s" % (key, path))
            species["class_id"] = mapping[key]
        write_json(path, data)

    dataset_summary_path = dataset_root / "dataset_summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    class_id_index = []
    for class_summary in dataset_summary.get("class_summaries", []):
        for species in class_summary.get("species", []):
            key = (class_summary["taxonomy_class"], species["common_name"])
            if key not in mapping:
                raise KeyError("Missing class_id mapping for %s in %s" % (key, dataset_summary_path))
            species["class_id"] = mapping[key]
            class_id_index.append(
                {
                    "class_id": mapping[key],
                    "taxonomy_class": class_summary["taxonomy_class"],
                    "scientific_name": species["scientific_name"],
                    "common_name": species["common_name"],
                    "taxon_id": species["taxon_id"],
                }
            )
    dataset_summary["class_id_index"] = sorted(
        class_id_index,
        key=lambda row: (row["taxonomy_class"], row["scientific_name"]),
    )
    write_json(dataset_summary_path, dataset_summary)

    class_id_index_path = dataset_root / "class_id_index.jsonl"
    write_jsonl(class_id_index_path, dataset_summary["class_id_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
