#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a portable download manifest for WildBioWiki-QA-medium images."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/WildBioWiki-QA-medium",
        help="Dataset root containing manifests/*.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/WildBioWiki-QA-medium/downloads/image_download_manifest.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/WildBioWiki-QA-medium/downloads/image_download_manifest.csv",
        help="Output CSV manifest path.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_download_url(url: str) -> str:
    if url.startswith("https://static.inaturalist.org/"):
        return url.replace("https://static.inaturalist.org", "https://inaturalist-open-data.s3.amazonaws.com", 1)
    return url


def build_row(record: dict) -> dict:
    original_url = record.get("medium_image_url")
    if not original_url:
        original_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{record['image_id']}/medium.jpg"
    download_url = normalize_download_url(original_url)
    return {
        "split": record["split"],
        "taxonomy_class": record["taxonomy_class"],
        "class_id": str(record["class_id"]),
        "taxon_id": str(record["taxon_id"]),
        "scientific_name": record["scientific_name"],
        "common_name": record["common_name"],
        "observation_id": str(record["observation_id"]),
        "image_id": str(record["image_id"]),
        "filename": Path(record["image_path"]).name,
        "relative_image_path": record["image_path"],
        "image_variant": record.get("image_variant", "medium"),
        "image_width": record["image_width"],
        "image_height": record["image_height"],
        "source_image_width": record["source_image_width"],
        "source_image_height": record["source_image_height"],
        "download_url": download_url,
        "original_medium_url": original_url,
        "image_variant_source": record.get("image_variant_source", ""),
    }


def main() -> int:
    args = parse_args()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    output_jsonl = (PROJECT_ROOT / args.output_jsonl).resolve()
    output_csv = (PROJECT_ROOT / args.output_csv).resolve()

    rows: List[dict] = []
    for split in SPLITS:
        manifest_path = dataset_root / "manifests" / f"{split}.jsonl"
        for record in load_jsonl(manifest_path):
            rows.append(build_row(record))

    write_jsonl(output_jsonl, rows)
    write_csv(
        output_csv,
        rows,
        fieldnames=[
            "split",
            "taxonomy_class",
            "class_id",
            "taxon_id",
            "scientific_name",
            "common_name",
            "observation_id",
            "image_id",
            "filename",
            "relative_image_path",
            "image_variant",
            "image_width",
            "image_height",
            "source_image_width",
            "source_image_height",
            "download_url",
            "original_medium_url",
            "image_variant_source",
        ],
    )
    print(f"Wrote {len(rows)} rows to {output_jsonl}")
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
