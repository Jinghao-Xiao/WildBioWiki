#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from PIL import Image, ImageFile

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True


LOGGER = logging.getLogger("run_grounding_dino_taxonomy")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromptRecord:
    taxon_id: str
    scientific_name: str
    taxonomy_class: str
    generic_label: str
    common_name_original: str
    detector_name_specific: str
    detector_name_broad: str
    prompt_primary: str
    prompt_recommended: str
    prompt_fallback: str
    prompt_generic: str
    qwen_label_recommended: str
    prompt_strategy: str


def slugify_scientific_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO over iNaturalist taxonomy folders and save one JSONL per species."
    )
    parser.add_argument(
        "--image-root",
        default="data/iNaturalist/inaturalist_images_396",
        help="Root directory containing downloaded iNaturalist images.",
    )
    parser.add_argument(
        "--prompt-csv",
        default="data/iNaturalist/inat_detector_prompts_396.csv",
        help="CSV containing prompt mapping records.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/grounding-dino-base",
        help="Local Grounding DINO model directory.",
    )
    parser.add_argument(
        "--output-root",
        default="data/iNaturalist/grounding_dino_boxes",
        help="Output root for JSONL results and summaries.",
    )
    parser.add_argument(
        "--taxonomy-class",
        required=True,
        help="Taxonomy class to process, e.g. Actinopterygii.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device string, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text threshold.",
    )
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Optional scientific names to restrict processing.",
    )
    parser.add_argument(
        "--limit-species",
        type=int,
        default=None,
        help="Optional cap on number of species for smoke tests.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional cap on number of images per species for smoke tests.",
    )
    parser.add_argument(
        "--disable-fp16",
        action="store_true",
        help="Disable autocast fp16 inference on CUDA.",
    )
    parser.add_argument(
        "--overwrite-errors",
        action="store_true",
        help="Re-run images previously recorded with status=error.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_prompt_records(prompt_csv: Path, taxonomy_class: str) -> Dict[str, PromptRecord]:
    records: Dict[str, PromptRecord] = {}
    with prompt_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["taxonomy_class"] != taxonomy_class:
                continue
            record = PromptRecord(
                taxon_id=row["taxon_id"],
                scientific_name=row["scientific_name"],
                taxonomy_class=row["taxonomy_class"],
                generic_label=row["generic_label"],
                common_name_original=row["common_name_original"],
                detector_name_specific=row["detector_name_specific"],
                detector_name_broad=row["detector_name_broad"],
                prompt_primary=row["grounding_dino_prompt_primary"],
                prompt_recommended=row["grounding_dino_prompt_recommended"],
                prompt_fallback=row["grounding_dino_prompt_fallback"],
                prompt_generic=row["grounding_dino_prompt_generic"],
                qwen_label_recommended=row["qwen_label_recommended"],
                prompt_strategy=row["prompt_strategy"],
            )
            records[slugify_scientific_name(record.scientific_name)] = record
    return records


def iter_species_dirs(image_root: Path, taxonomy_class: str) -> List[Path]:
    class_dir = image_root / taxonomy_class
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Taxonomy directory not found: {class_dir}")
    return sorted(path for path in class_dir.iterdir() if path.is_dir())


def load_metadata_rows(metadata_path: Path) -> List[dict]:
    rows: List[dict] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_existing_results(result_path: Path, overwrite_errors: bool) -> Set[str]:
    processed: Set[str] = set()
    if not result_path.exists():
        return processed

    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            status = record.get("status", "ok")
            if status == "ok" or not overwrite_errors:
                processed.add(record["image_path"])
    return processed


def build_detection_entry(box, score, label_text: str, scientific_name: str) -> dict:
    box_xyxy = [round(float(value), 2) for value in box.tolist()]
    return {
        "scientific_name": scientific_name,
        "raw_label": label_text,
        "score": round(float(score), 6),
        "bbox_xyxy": box_xyxy,
    }


def load_model(model_dir: Path, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return processor, model


def run_inference(
    image_path: Path,
    prompt: str,
    processor,
    model,
    device: torch.device,
    use_fp16: bool,
    box_threshold: float,
    text_threshold: float,
) -> List[dict]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_fp16 and device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            outputs = model(**inputs)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*The key `labels` is will return integer ids.*",
                category=FutureWarning,
            )
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

    if "text_labels" in results:
        label_texts = results["text_labels"]
    else:
        label_texts = results["labels"]

    return [
        build_detection_entry(box, score, label_text, "")
        for box, score, label_text in zip(results["boxes"], results["scores"], label_texts)
    ]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_species_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def maybe_limit(rows: List[dict], limit_images: Optional[int]) -> Iterable[dict]:
    if limit_images is None:
        return rows
    return rows[:limit_images]


def main() -> int:
    args = parse_args()
    setup_logging()

    image_root = (PROJECT_ROOT / args.image_root).resolve()
    prompt_csv = (PROJECT_ROOT / args.prompt_csv).resolve()
    model_dir = (PROJECT_ROOT / args.model_dir).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()

    requested_species = set(args.species or [])
    prompt_records = load_prompt_records(prompt_csv, args.taxonomy_class)
    species_dirs = iter_species_dirs(image_root, args.taxonomy_class)
    if requested_species:
        species_dirs = [
            path
            for path in species_dirs
            if prompt_records.get(path.name) and prompt_records[path.name].scientific_name in requested_species
        ]
    if args.limit_species is not None:
        species_dirs = species_dirs[: args.limit_species]

    if not species_dirs:
        LOGGER.error("No species directories selected for taxonomy class %s", args.taxonomy_class)
        return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {args.device}, but CUDA is not available.")

    processor, model = load_model(model_dir, device)
    use_fp16 = device.type == "cuda" and not args.disable_fp16

    class_output_dir = output_root / args.taxonomy_class
    class_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Processing %s species for %s on %s (fp16=%s)",
        len(species_dirs),
        args.taxonomy_class,
        args.device,
        use_fp16,
    )

    taxonomy_summary: List[dict] = []

    for index, species_dir in enumerate(species_dirs, start=1):
        prompt_record = prompt_records.get(species_dir.name)
        if prompt_record is None:
            LOGGER.warning("Skipping %s because no prompt record was found", species_dir.name)
            continue

        metadata_path = species_dir / "metadata.jsonl"
        if not metadata_path.exists():
            LOGGER.warning("Skipping %s because metadata.jsonl is missing", species_dir)
            continue

        result_path = class_output_dir / f"{species_dir.name}.jsonl"
        summary_path = class_output_dir / f"{species_dir.name}.summary.json"
        processed_images = load_existing_results(result_path, overwrite_errors=args.overwrite_errors)
        metadata_rows = load_metadata_rows(metadata_path)

        total_images = len(metadata_rows if args.limit_images is None else metadata_rows[: args.limit_images])
        processed_now = 0
        skipped_existing = 0
        errors = 0

        LOGGER.info(
            "[%s/%s] %s -> %s images, prompt=%r",
            index,
            len(species_dirs),
            prompt_record.scientific_name,
            total_images,
            prompt_record.prompt_recommended,
        )

        for row in maybe_limit(metadata_rows, args.limit_images):
            image_rel_path = row["local_path"]
            if image_rel_path in processed_images:
                skipped_existing += 1
                continue

            image_path = (PROJECT_ROOT / image_rel_path).resolve()
            base_record = {
                "taxon_id": row["taxon_id"],
                "scientific_name": row["scientific_name"],
                "common_name": row["common_name"],
                "taxonomy_class": prompt_record.taxonomy_class,
                "observation_id": row["observation_id"],
                "image_id": row["image_id"],
                "image_path": image_rel_path,
                "prompt_used": prompt_record.prompt_recommended,
                "prompt_strategy": prompt_record.prompt_strategy,
                "detector_name_specific": prompt_record.detector_name_specific,
                "detector_name_broad": prompt_record.detector_name_broad,
                "qwen_label_recommended": prompt_record.qwen_label_recommended,
            }
            try:
                detections = run_inference(
                    image_path=image_path,
                    prompt=prompt_record.prompt_recommended,
                    processor=processor,
                    model=model,
                    device=device,
                    use_fp16=use_fp16,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
                for detection in detections:
                    detection["scientific_name"] = row["scientific_name"]
                record = {
                    **base_record,
                    "status": "ok",
                    "num_boxes": len(detections),
                    "detections": detections,
                }
                append_jsonl(result_path, record)
                processed_now += 1
            except Exception as exc:  # noqa: BLE001
                record = {
                    **base_record,
                    "status": "error",
                    "num_boxes": 0,
                    "detections": [],
                    "error": str(exc),
                }
                append_jsonl(result_path, record)
                errors += 1

        summary = {
            "taxon_id": prompt_record.taxon_id,
            "scientific_name": prompt_record.scientific_name,
            "common_name": prompt_record.common_name_original,
            "taxonomy_class": prompt_record.taxonomy_class,
            "prompt_used": prompt_record.prompt_recommended,
            "prompt_strategy": prompt_record.prompt_strategy,
            "total_images_considered": total_images,
            "processed_now": processed_now,
            "skipped_existing": skipped_existing,
            "errors": errors,
            "result_jsonl": str(result_path.relative_to(PROJECT_ROOT)),
        }
        write_species_summary(summary_path, summary)
        taxonomy_summary.append(summary)
        LOGGER.info(
            "[%s/%s] done %s: processed_now=%s skipped_existing=%s errors=%s",
            index,
            len(species_dirs),
            prompt_record.scientific_name,
            processed_now,
            skipped_existing,
            errors,
        )

    taxonomy_summary_path = class_output_dir / f"{args.taxonomy_class}.summary.json"
    write_species_summary(
        taxonomy_summary_path,
        {
            "taxonomy_class": args.taxonomy_class,
            "species_count": len(taxonomy_summary),
            "results": taxonomy_summary,
        },
    )
    LOGGER.info("Wrote taxonomy summary to %s", taxonomy_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
