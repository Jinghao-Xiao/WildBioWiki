#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate caption and QA answers with Qwen3-VL."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qa-jsonl", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--caption-max-new-tokens", type=int, default=512)
    parser.add_argument("--qa-max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection-mode",
        default="first_unique_class",
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


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add(str(row["image_id"]))
    return completed


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


def build_messages(system_prompt: str, user_prompt: str, image: Image.Image) -> List[dict]:
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


def generate_text(
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
        messages = build_messages(system_prompt, user_prompt, image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


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
        qa_map[row["class_id"]] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path.write_text("", encoding="utf-8")
    completed_ids = load_completed_ids(output_path)

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype_from_name(args.dtype),
        device_map=None,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    with output_path.open("a", encoding="utf-8") as handle:
        processed = 0
        skipped = 0
        for index, record in enumerate(selected, start=1):
            image_id = str(record["image_id"])
            if image_id in completed_ids:
                skipped += 1
                continue
            image_path = (PROJECT_ROOT / "data/WildBioWiki-QA-medium" / record["image_path"]).resolve()
            qa_row = qa_map[record["class_id"]]
            caption = generate_text(
                model=model,
                processor=processor,
                image_path=image_path,
                system_prompt=args.caption_system_prompt,
                user_prompt=args.caption_user_prompt,
                device=args.device,
                max_new_tokens=args.caption_max_new_tokens,
                temperature=args.temperature,
            )
            qa_results = []
            for qid in sorted(qa_row["qa_pairs"], key=lambda x: int(x[1:])):
                qa_item = qa_row["qa_pairs"][qid]
                predicted = generate_text(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    system_prompt=args.qa_system_prompt,
                    user_prompt=qa_item["question"],
                    device=args.device,
                    max_new_tokens=args.qa_max_new_tokens,
                    temperature=args.temperature,
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
            processed += 1
            print(
                f"[{index}/{len(selected)}] {record['taxonomy_class']} {record['scientific_name']} image_id={record['image_id']}",
                flush=True,
            )
            print(f"caption: {caption}", flush=True)
            print(f"qa_answer_count: {len(qa_results)}", flush=True)
        print(
            json.dumps(
                {
                    "output_jsonl": str(output_path.relative_to(PROJECT_ROOT)),
                    "processed_now": processed,
                    "skipped_existing": skipped,
                    "total_considered": len(selected),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
