#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = [
    "Actinopterygii",
    "Aves",
    "Mammalia",
    "Amphibia",
    "Reptilia",
]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the finalized WildBioWiki-QA dataset from filtered mapped Grounding DINO outputs."
    )
    parser.add_argument(
        "--input-root",
        default="data/iNaturalist/grounding_dino_boxes_filtered_mapped",
        help="Root containing filtered mapped per-class JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/WildBioWiki-QA",
        help="Output root for the finalized dataset.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Taxonomy classes to include.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio within each species.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio within each species.",
    )
    parser.add_argument(
        "--seed",
        default="wildbiowiki-qa-v1",
        help="Deterministic seed string used for per-species shuffling.",
    )
    return parser.parse_args()


def stable_key(seed: str, taxonomy_class: str, scientific_name: str, record: dict) -> str:
    raw = "|".join(
        [
            seed,
            taxonomy_class,
            scientific_name,
            str(record["observation_id"]),
            str(record["image_id"]),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compute_split_counts(total: int, train_ratio: float, val_ratio: float) -> Dict[str, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }


def assign_splits(
    records: List[dict],
    seed: str,
    taxonomy_class: str,
    scientific_name: str,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[dict]]:
    ordered = sorted(
        records,
        key=lambda record: stable_key(seed, taxonomy_class, scientific_name, record),
    )
    counts = compute_split_counts(len(ordered), train_ratio, val_ratio)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def ensure_clean_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def make_relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    rel_target = Path(target).resolve().relative_to(PROJECT_ROOT.resolve())
    relative_from_link = Path(
        Path(
            Path(
                *([ ".." ] * len(link_path.relative_to(PROJECT_ROOT).parents[:-1]))
            )
        )
    )
    # Simpler and more reliable: use os.path.relpath semantics via pathlib string conversion.
    relative_target_str = str(Path(target).resolve().relative_to(link_path.parent.resolve()))
    link_path.symlink_to(relative_target_str)


def build_record_for_dataset(record: dict, split: str, dataset_image_rel: str) -> dict:
    return {
        "split": split,
        "taxonomy_class": record["taxonomy_class"],
        "taxon_id": record["taxon_id"],
        "scientific_name": record["scientific_name"],
        "common_name": record["common_name"],
        "observation_id": record["observation_id"],
        "image_id": record["image_id"],
        "image_path": dataset_image_rel,
        "bbox_xyxy": record["bbox_xyxy"],
        "score": record["score"],
    }


def main() -> int:
    args = parse_args()
    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    ensure_clean_output(output_root)

    global_split_rows = {split: [] for split in SPLITS}
    class_rollup = []

    for class_name in args.classes:
        input_class_dir = input_root / class_name
        if not input_class_dir.exists():
            raise FileNotFoundError("Missing class directory: %s" % input_class_dir)

        class_species_jsonls = sorted(
            [
                path for path in input_class_dir.glob("*.jsonl")
                if not path.name.endswith(".rejected.jsonl")
                and not path.name.endswith(".delete_manifest.jsonl")
            ]
        )

        class_summary = {
            "taxonomy_class": class_name,
            "species_count": len(class_species_jsonls),
            "splits": {split: {"images": 0, "species": 0} for split in SPLITS},
            "species": [],
        }

        split_has_species = {split: set() for split in SPLITS}

        for species_jsonl in class_species_jsonls:
            species_slug = species_jsonl.stem
            records = load_jsonl(species_jsonl)
            if not records:
                continue

            scientific_name = records[0]["scientific_name"]
            taxonomy_class = records[0]["taxonomy_class"]
            split_map = assign_splits(
                records=records,
                seed=args.seed,
                taxonomy_class=taxonomy_class,
                scientific_name=scientific_name,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )

            species_summary = {
                "species_slug": species_slug,
                "scientific_name": scientific_name,
                "common_name": records[0]["common_name"],
                "taxon_id": records[0]["taxon_id"],
                "total_images": len(records),
                "splits": {},
            }

            for split, split_records in split_map.items():
                annotations_rows = []
                for record in split_records:
                    source_image_path = (PROJECT_ROOT / record["image_path"]).resolve()
                    if not source_image_path.exists():
                        raise FileNotFoundError("Source image missing for kept record: %s" % source_image_path)

                    split_image_dir = output_root / "images" / split / class_name / species_slug
                    split_image_dir.mkdir(parents=True, exist_ok=True)
                    link_path = split_image_dir / source_image_path.name
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(source_image_path)

                    dataset_image_rel = str(link_path.relative_to(output_root))
                    dataset_record = build_record_for_dataset(record, split, dataset_image_rel)
                    annotations_rows.append(dataset_record)
                    global_split_rows[split].append(dataset_record)

                annotations_path = output_root / "annotations" / split / class_name / ("%s.jsonl" % species_slug)
                write_jsonl(annotations_path, annotations_rows)

                species_summary["splits"][split] = len(annotations_rows)
                class_summary["splits"][split]["images"] += len(annotations_rows)
                if annotations_rows:
                    split_has_species[split].add(species_slug)

            class_summary["species"].append(species_summary)

        for split in SPLITS:
            class_summary["splits"][split]["species"] = len(split_has_species[split])

        class_summary_path = output_root / "summaries" / ("%s.summary.json" % class_name)
        write_json(class_summary_path, class_summary)
        class_rollup.append(class_summary)

    for split in SPLITS:
        split_rows = sorted(
            global_split_rows[split],
            key=lambda row: (row["taxonomy_class"], row["scientific_name"], row["image_id"]),
        )
        write_jsonl(output_root / "manifests" / ("%s.jsonl" % split), split_rows)
        write_json(
            output_root / "manifests" / ("%s.summary.json" % split),
            {
                "split": split,
                "image_count": len(split_rows),
                "class_count": len({row["taxonomy_class"] for row in split_rows}),
                "species_count": len({(row["taxonomy_class"], row["scientific_name"]) for row in split_rows}),
            },
        )

    dataset_summary = {
        "dataset_name": "WildBioWiki-QA",
        "classes": args.classes,
        "seed": args.seed,
        "split_rule": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
            "within_species": True,
        },
        "class_summaries": class_rollup,
    }
    write_json(output_root / "dataset_summary.json", dataset_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
