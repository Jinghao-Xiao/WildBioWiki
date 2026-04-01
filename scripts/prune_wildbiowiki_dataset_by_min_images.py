#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune WildBioWiki-QA species whose total kept image count is below a threshold."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/WildBioWiki-QA",
        help="Root directory of the WildBioWiki-QA dataset.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=300,
        help="Minimum total images required to keep a species.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def species_slug_from_image_path(image_path: str) -> str:
    parts = Path(image_path).parts
    # images/{split}/{class}/{species_slug}/{filename}
    return parts[3]


def determine_removed_species(dataset_summary: dict, min_images: int) -> List[dict]:
    removed = []
    for class_summary in dataset_summary["class_summaries"]:
        for species in class_summary["species"]:
            if species["total_images"] < min_images:
                removed.append(
                    {
                        "taxonomy_class": class_summary["taxonomy_class"],
                        "species_slug": species["species_slug"],
                        "scientific_name": species["scientific_name"],
                        "common_name": species["common_name"],
                        "class_id": species["class_id"],
                        "taxon_id": species["taxon_id"],
                        "total_images": species["total_images"],
                    }
                )
    return removed


def remove_species_files(dataset_root: Path, removed_species: List[dict]) -> None:
    for species in removed_species:
        taxonomy_class = species["taxonomy_class"]
        species_slug = species["species_slug"]

        for split in SPLITS:
            image_dir = dataset_root / "images" / split / taxonomy_class / species_slug
            if image_dir.exists():
                shutil.rmtree(image_dir)

            annotation_path = dataset_root / "annotations" / split / taxonomy_class / ("%s.jsonl" % species_slug)
            if annotation_path.exists():
                annotation_path.unlink()


def filter_manifests(dataset_root: Path, removed_class_ids: Set[str]) -> None:
    manifests_dir = dataset_root / "manifests"
    for split in SPLITS:
        manifest_path = manifests_dir / ("%s.jsonl" % split)
        rows = [row for row in load_jsonl(manifest_path) if str(row["class_id"]) not in removed_class_ids]
        write_jsonl(manifest_path, rows)
        write_json(
            manifests_dir / ("%s.summary.json" % split),
            {
                "split": split,
                "image_count": len(rows),
                "class_count": len({row["taxonomy_class"] for row in rows}),
                "species_count": len({(row["taxonomy_class"], row["scientific_name"]) for row in rows}),
            },
        )


def update_class_summaries(dataset_root: Path, removed_class_ids: Set[str]) -> List[dict]:
    summaries_dir = dataset_root / "summaries"
    updated_class_summaries = []
    for summary_path in sorted(summaries_dir.glob("*.summary.json")):
        data = load_json(summary_path)
        kept_species = [species for species in data["species"] if str(species["class_id"]) not in removed_class_ids]

        split_images = {split: 0 for split in SPLITS}
        split_species = {split: 0 for split in SPLITS}
        for species in kept_species:
            for split in SPLITS:
                count = species["splits"].get(split, 0)
                split_images[split] += count
                if count > 0:
                    split_species[split] += 1

        data["species"] = kept_species
        data["species_count"] = len(kept_species)
        for split in SPLITS:
            data["splits"][split]["images"] = split_images[split]
            data["splits"][split]["species"] = split_species[split]

        write_json(summary_path, data)
        updated_class_summaries.append(data)
    return updated_class_summaries


def update_qa_files(dataset_root: Path, removed_class_ids: Set[str]) -> None:
    qa_dir = dataset_root / "qa"
    qa_path = qa_dir / "species_qa_by_class_id.jsonl"
    qa_rows = [row for row in load_jsonl(qa_path) if str(row["class_id"]) not in removed_class_ids]
    write_jsonl(qa_path, qa_rows)

    summary_path = qa_dir / "species_qa_summary.json"
    summary = load_json(summary_path)
    summary["row_count"] = len(qa_rows)
    summary["species_count"] = len(qa_rows)
    summary["class_count"] = len({row["taxonomy_class"] for row in qa_rows})
    write_json(summary_path, summary)


def update_class_id_index(dataset_root: Path, removed_class_ids: Set[str]) -> List[dict]:
    index_path = dataset_root / "class_id_index.jsonl"
    rows = [row for row in load_jsonl(index_path) if str(row["class_id"]) not in removed_class_ids]
    write_jsonl(index_path, rows)
    return rows


def update_dataset_summary(
    dataset_root: Path,
    updated_class_summaries: List[dict],
    class_id_index_rows: List[dict],
    removed_species: List[dict],
    min_images: int,
) -> None:
    summary_path = dataset_root / "dataset_summary.json"
    data = load_json(summary_path)
    data["class_summaries"] = updated_class_summaries
    data["class_id_index"] = class_id_index_rows
    data["pruning"] = {
        "applied": True,
        "min_images": min_images,
        "removed_species_count": len(removed_species),
        "removed_species": removed_species,
    }
    write_json(summary_path, data)


def write_prune_report(dataset_root: Path, removed_species: List[dict], min_images: int) -> None:
    report = {
        "min_images": min_images,
        "removed_species_count": len(removed_species),
        "removed_species": removed_species,
    }
    write_json(dataset_root / "pruning_report.json", report)


def main() -> int:
    args = parse_args()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    dataset_summary = load_json(dataset_root / "dataset_summary.json")

    removed_species = determine_removed_species(dataset_summary, args.min_images)
    removed_class_ids = {str(species["class_id"]) for species in removed_species}

    remove_species_files(dataset_root, removed_species)
    filter_manifests(dataset_root, removed_class_ids)
    updated_class_summaries = update_class_summaries(dataset_root, removed_class_ids)
    update_qa_files(dataset_root, removed_class_ids)
    class_id_index_rows = update_class_id_index(dataset_root, removed_class_ids)
    update_dataset_summary(dataset_root, updated_class_summaries, class_id_index_rows, removed_species, args.min_images)
    write_prune_report(dataset_root, removed_species, args.min_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
