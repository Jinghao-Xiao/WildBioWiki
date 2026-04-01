#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = "You are a wildlife image description assistant. Describe only what is directly visible in the image."
DEFAULT_USER_PROMPT = "Species name: {species_name}\nPlease generate one image-level caption for this wildlife image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate image-level captions for WildBioWiki-QA-medium with Qwen3-VL-8B-Instruct."
    )
    parser.add_argument(
        "--manifest",
        default="data/WildBioWiki-QA-medium/manifests/val.jsonl",
        help="Dataset manifest to caption.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/Qwen3-VL-8B-Instruct",
        help="Local Qwen3-VL model directory.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/WildBioWiki-QA-medium/captions/qwen3_vl_8b_instruct/val.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt.",
    )
    parser.add_argument(
        "--user-prompt-template",
        default=DEFAULT_USER_PROMPT,
        help="User prompt template. Must contain {species_name}.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Model dtype.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum new tokens for generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap for smoke testing. Zero means no cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file instead of resuming.",
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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_prompt(system_prompt: str, user_prompt: str, image: Image.Image) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def load_completed_ids(output_jsonl: Path) -> set:
    if not output_jsonl.exists():
        return set()
    completed = set()
    for row in load_jsonl(output_jsonl):
        completed.add(str(row["image_id"]))
    return completed


def generate_caption(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        messages = build_prompt(system_prompt, user_prompt, image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        generated = model.generate(**inputs, **generation_kwargs)
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return caption.strip()


def build_output_row(record: dict, caption: str, system_prompt: str, user_prompt: str, model_dir: str) -> dict:
    return {
        "split": record["split"],
        "taxonomy_class": record["taxonomy_class"],
        "taxon_id": record["taxon_id"],
        "scientific_name": record["scientific_name"],
        "common_name": record["common_name"],
        "class_id": record["class_id"],
        "observation_id": record["observation_id"],
        "image_id": record["image_id"],
        "image_path": record["image_path"],
        "caption": caption,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model_dir": model_dir,
    }


def main() -> int:
    args = parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    model_dir = (PROJECT_ROOT / args.model_dir).resolve()
    output_jsonl = (PROJECT_ROOT / args.output_jsonl).resolve()

    records = load_jsonl(manifest_path)
    if args.limit > 0:
        records = records[: args.limit]

    if args.overwrite:
        truncate_output(output_jsonl)
    completed = load_completed_ids(output_jsonl)

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype_from_name(args.dtype),
        device_map=None,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    processed = 0
    skipped = 0
    for record in records:
        image_id = str(record["image_id"])
        if image_id in completed:
            skipped += 1
            continue
        image_path = (PROJECT_ROOT / "data/WildBioWiki-QA-medium" / record["image_path"]).resolve()
        user_prompt = args.user_prompt_template.format(species_name=record["scientific_name"])
        caption = generate_caption(
            model=model,
            processor=processor,
            image_path=image_path,
            system_prompt=args.system_prompt,
            user_prompt=user_prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_row = build_output_row(
            record=record,
            caption=caption,
            system_prompt=args.system_prompt,
            user_prompt=user_prompt,
            model_dir=str(model_dir.relative_to(PROJECT_ROOT)),
        )
        append_jsonl(output_jsonl, output_row)
        processed += 1
        print(
            f"{record['split']} {record['scientific_name']} image_id={record['image_id']} processed={processed} skipped={skipped}",
            flush=True,
        )
        print(f"caption: {caption}", flush=True)

    print(
        json.dumps(
            {
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                "output_jsonl": str(output_jsonl.relative_to(PROJECT_ROOT)),
                "processed_now": processed,
                "skipped_existing": skipped,
                "total_considered": len(records),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
