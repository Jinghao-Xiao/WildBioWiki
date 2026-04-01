#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate caption and QA answers with SAIL-VL."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qa-jsonl", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--caption-max-new-tokens", type=int, default=512)
    parser.add_argument("--qa-max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection-mode",
        default="first_n",
        choices=("first_n", "first_unique_class"),
    )
    parser.add_argument(
        "--caption-system-prompt",
        default="You are a wildlife image description assistant. Describe only what is directly visible in the image.",
    )
    parser.add_argument(
        "--caption-user-prompt",
        default="Identify the species of the animal visible in this image.\nPlease generate one image-level caption for this wildlife image.",
    )
    parser.add_argument(
        "--qa-system-prompt",
        default="You are a wildlife visual question answering assistant. Answer the question concisely and directly.",
    )
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-num", type=int, default=10)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_records(records: List[dict], mode: str, limit: int) -> List[dict]:
    if mode == "first_n":
        if limit <= 0:
            return records
        return records[:limit]
    selected: List[dict] = []
    seen = set()
    for row in records:
        key = row["taxonomy_class"]
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios,
    width: int,
    height: int,
    image_size: int,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 10,
    image_size: int = 448,
    use_thumbnail: bool = True,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: Path, input_size: int = 448, max_num: int = 10) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def generate_text(
    model,
    tokenizer,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    generation_config: dict,
    dtype: torch.dtype,
    input_size: int,
    max_num: int,
) -> str:
    pixel_values = load_image(image_path, input_size=input_size, max_num=max_num)
    pixel_values = pixel_values.to(dtype=dtype, device=model.device)
    original_system_message = model.system_message
    try:
        model.system_message = system_prompt
        response = model.chat(tokenizer, pixel_values, user_prompt, generation_config)
    finally:
        model.system_message = original_system_message
    return response.strip()


def main() -> int:
    args = parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    qa_path = (PROJECT_ROOT / args.qa_jsonl).resolve()
    model_dir = (PROJECT_ROOT / args.model_dir).resolve()
    output_path = (PROJECT_ROOT / args.output_jsonl).resolve()

    records = load_jsonl(manifest_path)
    selected = select_records(records, args.selection_mode, args.limit)

    qa_map: Dict[str, dict] = {}
    for row in load_jsonl(qa_path):
        qa_map[str(row["class_id"])] = row

    dtype = dtype_from_name(args.dtype)
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path.write_text("", encoding="utf-8")

    with output_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(selected, start=1):
            image_path = (
                PROJECT_ROOT / "data/WildBioWiki-QA-medium" / record["image_path"]
            ).resolve()
            qa_row = qa_map[str(record["class_id"])]

            caption = generate_text(
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                system_prompt=args.caption_system_prompt,
                user_prompt=args.caption_user_prompt,
                generation_config={
                    "max_new_tokens": args.caption_max_new_tokens,
                    "do_sample": args.temperature > 0,
                    **({"temperature": args.temperature} if args.temperature > 0 else {}),
                },
                dtype=dtype,
                input_size=args.image_size,
                max_num=args.max_num,
            )

            qa_results = []
            for qid in sorted(qa_row["qa_pairs"], key=lambda x: int(x[1:])):
                qa_item = qa_row["qa_pairs"][qid]
                predicted = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    image_path=image_path,
                    system_prompt=args.qa_system_prompt,
                    user_prompt=qa_item["question"],
                    generation_config={
                        "max_new_tokens": args.qa_max_new_tokens,
                        "do_sample": args.temperature > 0,
                        **({"temperature": args.temperature} if args.temperature > 0 else {}),
                    },
                    dtype=dtype,
                    input_size=args.image_size,
                    max_num=args.max_num,
                )
                qa_results.append(
                    {
                        "qid": qid,
                        "question": qa_item["question"],
                        "predicted_answer": predicted,
                        "reference_answer": qa_item["answer"],
                    }
                )

            row = {
                "split": record["split"],
                "taxonomy_class": record["taxonomy_class"],
                "class_id": record["class_id"],
                "taxon_id": record["taxon_id"],
                "scientific_name": record["scientific_name"],
                "common_name": record["common_name"],
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "caption_result": {
                    "system_prompt": args.caption_system_prompt,
                    "user_prompt": args.caption_user_prompt,
                    "answer": caption,
                },
                "qa_results": qa_results,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(selected)}] {record['taxonomy_class']} {record['scientific_name']} image_id={record['image_id']}",
                flush=True,
            )
            print(f"caption: {caption}", flush=True)
            print(f"qa_answer_count: {len(qa_results)}", flush=True)

    print(output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
