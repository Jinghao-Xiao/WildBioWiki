#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all Grounding DINO detections for one species JSONL into a dedicated folder."
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Per-species Grounding DINO result JSONL path, relative to project root or absolute.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for rendered images.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_records(jsonl_path: Path) -> List[dict]:
    records = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_text_box(draw: ImageDraw.ImageDraw, x0: float, y0: float, label: str, font) -> Tuple[float, float, float, float]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox((x0, y0), label, font=font)
    text_w, text_h = draw.textsize(label, font=font)
    return (x0, y0, x0 + text_w, y0 + text_h)


def draw_record(record: dict, output_dir: Path) -> dict:
    image_path = resolve_path(record["image_path"])
    output_name = f"{Path(record['image_path']).stem}_boxed.jpg"
    output_path = output_dir / output_name

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        detections = record.get("detections", [])
        for idx, detection in enumerate(detections, start=1):
            x0, y0, x1, y1 = detection["bbox_xyxy"]
            color = (255, 64 + (idx * 17) % 160, 64 + (idx * 29) % 160)
            for offset in range(3):
                draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

            label = f"{detection['raw_label']} {detection['score']:.2f}"
            text_box = get_text_box(draw, x0, y0, label, font)
            draw.rectangle(text_box, fill=color)
            draw.text((x0, y0), label, fill="black", font=font)

        if not detections:
            label = "NO BOXES"
            text_box = get_text_box(draw, 12, 12, label, font)
            draw.rectangle(text_box, fill=(255, 220, 80))
            draw.text((12, 12), label, fill="black", font=font)

        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)

    return {
        "scientific_name": record.get("scientific_name"),
        "common_name": record.get("common_name"),
        "observation_id": record.get("observation_id"),
        "image_id": record.get("image_id"),
        "image_path": record.get("image_path"),
        "visualization_path": str(output_path.relative_to(PROJECT_ROOT)),
        "prompt_used": record.get("prompt_used"),
        "status": record.get("status"),
        "num_boxes": record.get("num_boxes", 0),
    }


def main() -> int:
    args = parse_args()
    jsonl_path = resolve_path(args.jsonl)
    output_dir = resolve_path(args.output_dir)

    records = load_records(jsonl_path)
    manifest_path = output_dir / "render_manifest.jsonl"
    summary_path = output_dir / "render_summary.json"

    rendered = []
    for record in records:
        rendered.append(draw_record(record, output_dir))

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rendered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "jsonl_path": str(jsonl_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "image_count": len(rendered),
        "ok_count": sum(1 for row in rendered if row["status"] == "ok"),
        "error_count": sum(1 for row in rendered if row["status"] == "error"),
        "with_boxes_count": sum(1 for row in rendered if row["num_boxes"] > 0),
        "without_boxes_count": sum(1 for row in rendered if row["num_boxes"] == 0),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
