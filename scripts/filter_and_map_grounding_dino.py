#!/usr/bin/env python3

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Grounding DINO results, map kept boxes back to scientific names, and optionally delete rejected images."
    )
    parser.add_argument(
        "--input-root",
        default="data/iNaturalist/grounding_dino_boxes",
        help="Root directory containing per-class Grounding DINO outputs.",
    )
    parser.add_argument(
        "--output-root",
        default="data/iNaturalist/grounding_dino_boxes_filtered_mapped",
        help="Root directory for filtered mapped outputs.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Taxonomy classes to process.",
    )
    parser.add_argument(
        "--delete-rejected-images",
        action="store_true",
        help="Delete source images rejected by the filtering rule.",
    )
    return parser.parse_args()


def slugify_scientific_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_species_records(records: List[dict], class_name: str, species_slug: str, jsonl_path: Path) -> None:
    if not records:
        return

    scientific_names = {record.get("scientific_name") for record in records}
    taxonomy_classes = {record.get("taxonomy_class") for record in records}
    if len(scientific_names) != 1:
        raise ValueError("Multiple scientific names found in %s" % jsonl_path)
    if taxonomy_classes != {class_name}:
        raise ValueError("Unexpected taxonomy class in %s" % jsonl_path)

    scientific_name = list(scientific_names)[0]
    expected_slug = slugify_scientific_name(scientific_name)
    if expected_slug != species_slug:
        raise ValueError(
            "Species slug mismatch for %s: file slug=%s scientific_name=%s expected_slug=%s"
            % (jsonl_path, species_slug, scientific_name, expected_slug)
        )

    expected_fragment = "/%s/%s/images/" % (class_name, species_slug)
    for record in records:
        image_path = record.get("image_path", "")
        if expected_fragment not in image_path:
            raise ValueError("Image path mismatch in %s: %s" % (jsonl_path, image_path))


def select_kept_detection(record: dict) -> Tuple[dict, str]:
    if record.get("status") != "ok":
        return None, "non_ok_status"

    detections = record.get("detections", [])
    num_boxes = len(detections)
    if num_boxes == 0:
        return None, "zero_boxes"
    if num_boxes == 1:
        return detections[0], "single_box"
    if num_boxes == 2:
        best = detections[0]
        if detections[1].get("score", 0.0) > detections[0].get("score", 0.0):
            best = detections[1]
        return best, "best_of_two"
    return None, "more_than_two_boxes"


def build_kept_record(record: dict, selected_detection: dict, selection_reason: str) -> dict:
    return {
        "taxon_id": record["taxon_id"],
        "scientific_name": record["scientific_name"],
        "common_name": record["common_name"],
        "taxonomy_class": record["taxonomy_class"],
        "observation_id": record["observation_id"],
        "image_id": record["image_id"],
        "image_path": record["image_path"],
        "bbox_xyxy": selected_detection["bbox_xyxy"],
        "score": selected_detection["score"],
        "selection_reason": selection_reason,
        "num_boxes_original": record.get("num_boxes", len(record.get("detections", []))),
        "prompt_used": record.get("prompt_used"),
    }


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    class_rollup = []

    for class_name in args.classes:
        input_class_dir = input_root / class_name
        taxonomy_summary_path = input_class_dir / ("%s.summary.json" % class_name)
        if not taxonomy_summary_path.exists():
            raise FileNotFoundError("Class summary not found, class not finished: %s" % taxonomy_summary_path)

        output_class_dir = output_root / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        class_delete_manifest = []
        class_species_summaries = []
        kept_total = 0
        rejected_total = 0
        deleted_total = 0

        species_jsonls = sorted(input_class_dir.glob("*.jsonl"))
        for species_jsonl in species_jsonls:
            species_slug = species_jsonl.stem
            records = load_jsonl(species_jsonl)
            validate_species_records(records, class_name, species_slug, species_jsonl)

            kept_rows = []
            rejected_rows = []
            images_to_delete = []

            for record in records:
                selected_detection, selection_reason = select_kept_detection(record)
                if selected_detection is None:
                    rejected = {
                        "taxon_id": record.get("taxon_id"),
                        "scientific_name": record.get("scientific_name"),
                        "common_name": record.get("common_name"),
                        "taxonomy_class": record.get("taxonomy_class"),
                        "observation_id": record.get("observation_id"),
                        "image_id": record.get("image_id"),
                        "image_path": record.get("image_path"),
                        "reject_reason": selection_reason,
                        "num_boxes_original": record.get("num_boxes", len(record.get("detections", []))),
                    }
                    rejected_rows.append(rejected)
                    class_delete_manifest.append(rejected)
                    images_to_delete.append((PROJECT_ROOT / record["image_path"]).resolve())
                    continue

                kept_rows.append(build_kept_record(record, selected_detection, selection_reason))

            kept_path = output_class_dir / ("%s.jsonl" % species_slug)
            rejected_path = output_class_dir / ("%s.rejected.jsonl" % species_slug)
            species_summary_path = output_class_dir / ("%s.summary.json" % species_slug)

            write_jsonl(kept_path, kept_rows)
            write_jsonl(rejected_path, rejected_rows)

            deleted_for_species = 0
            missing_for_species = 0
            if args.delete_rejected_images:
                for image_path in images_to_delete:
                    if image_path.exists():
                        image_path.unlink()
                        deleted_for_species += 1
                    else:
                        missing_for_species += 1

            summary = {
                "taxonomy_class": class_name,
                "species_slug": species_slug,
                "scientific_name": records[0]["scientific_name"] if records else None,
                "common_name": records[0]["common_name"] if records else None,
                "input_records": len(records),
                "kept_records": len(kept_rows),
                "rejected_records": len(rejected_rows),
                "deleted_rejected_images": deleted_for_species,
                "missing_rejected_images": missing_for_species,
                "kept_jsonl": str(kept_path.relative_to(PROJECT_ROOT)),
                "rejected_jsonl": str(rejected_path.relative_to(PROJECT_ROOT)),
            }
            write_json(species_summary_path, summary)

            class_species_summaries.append(summary)
            kept_total += len(kept_rows)
            rejected_total += len(rejected_rows)
            deleted_total += deleted_for_species

        delete_manifest_path = output_class_dir / ("%s.delete_manifest.jsonl" % class_name)
        write_jsonl(delete_manifest_path, class_delete_manifest)

        class_summary = {
            "taxonomy_class": class_name,
            "species_count": len(class_species_summaries),
            "kept_total": kept_total,
            "rejected_total": rejected_total,
            "deleted_rejected_images": deleted_total,
            "delete_manifest": str(delete_manifest_path.relative_to(PROJECT_ROOT)),
            "species": class_species_summaries,
        }
        class_summary_path = output_class_dir / ("%s.summary.json" % class_name)
        write_json(class_summary_path, class_summary)
        class_rollup.append(class_summary)

    rollup_path = output_root / "filtered_mapped_rollup.json"
    write_json(rollup_path, {"classes": class_rollup})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
