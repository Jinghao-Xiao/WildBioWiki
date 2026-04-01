#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample completed Grounding DINO species results and render box visualizations."
    )
    parser.add_argument(
        "--result-root",
        default="data/iNaturalist/grounding_dino_boxes/Actinopterygii",
        help="Directory containing per-species JSONL and summary files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/iNaturalist/grounding_dino_boxes/Actinopterygii_visualization_sample_30x2",
        help="Directory for sampled visualizations.",
    )
    parser.add_argument(
        "--species-count",
        type=int,
        default=30,
        help="Number of species to sample.",
    )
    parser.add_argument(
        "--images-per-species",
        type=int,
        default=2,
        help="Number of images to sample per species.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260325,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_completed_species(result_root: Path) -> List[Path]:
    summary_files = []
    for path in sorted(result_root.glob("*.summary.json")):
        if path.name.endswith("Actinopterygii.summary.json"):
            continue
        summary_files.append(path)
    return summary_files


def load_candidate_rows(jsonl_path: Path) -> List[dict]:
    candidates: List[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("num_boxes", 0) > 0:
                candidates.append(row)
    return candidates


def draw_boxes(record: dict, image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for idx, detection in enumerate(record["detections"], start=1):
            x0, y0, x1, y1 = detection["bbox_xyxy"]
            color = (255, 64 + (idx * 17) % 160, 64 + (idx * 29) % 160)
            for offset in range(3):
                draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

            label = f"{detection['raw_label']} {detection['score']:.2f}"
            if hasattr(draw, "textbbox"):
                text_box = draw.textbbox((x0, y0), label, font=font)
            else:
                text_w, text_h = draw.textsize(label, font=font)
                text_box = (x0, y0, x0 + text_w, y0 + text_h)
            draw.rectangle(text_box, fill=color)
            draw.text((x0, y0), label, fill="black", font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)


def main() -> int:
    args = parse_args()
    result_root = (PROJECT_ROOT / args.result_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    summary_files = load_completed_species(result_root)
    if len(summary_files) < args.species_count:
        raise ValueError(
            f"Only found {len(summary_files)} completed species, need {args.species_count}."
        )

    sampled_summaries = rng.sample(summary_files, args.species_count)
    manifest_rows: List[Dict[str, str]] = []
    summary_rows = []

    for summary_path in sampled_summaries:
        species_slug = summary_path.stem.replace(".summary", "")
        jsonl_path = result_root / f"{species_slug}.jsonl"
        if not jsonl_path.exists():
            continue

        candidates = load_candidate_rows(jsonl_path)
        if len(candidates) < args.images_per_species:
            raise ValueError(f"{species_slug} has only {len(candidates)} drawable images.")

        chosen_records = rng.sample(candidates, args.images_per_species)
        species_dir = output_root / species_slug
        species_dir.mkdir(parents=True, exist_ok=True)

        per_species = {
            "species_slug": species_slug,
            "scientific_name": chosen_records[0]["scientific_name"],
            "common_name": chosen_records[0]["common_name"],
            "jsonl_path": str(jsonl_path.relative_to(PROJECT_ROOT)),
            "visualizations": [],
        }

        for record in chosen_records:
            image_rel_path = record["image_path"]
            source_path = (PROJECT_ROOT / image_rel_path).resolve()
            output_name = f"{Path(image_rel_path).stem}_boxed.jpg"
            output_path = species_dir / output_name
            draw_boxes(record, source_path, output_path)

            entry = {
                "species_slug": species_slug,
                "scientific_name": record["scientific_name"],
                "common_name": record["common_name"],
                "image_id": record["image_id"],
                "observation_id": record["observation_id"],
                "image_path": image_rel_path,
                "visualization_path": str(output_path.relative_to(PROJECT_ROOT)),
                "prompt_used": record["prompt_used"],
                "num_boxes": record["num_boxes"],
            }
            per_species["visualizations"].append(entry)
            manifest_rows.append(entry)

        summary_rows.append(per_species)

    manifest_path = output_root / "sample_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_root / "sample_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
