from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = "You are a wildlife image captioning assistant. Describe only what is directly visible in the image."

USER_PROMPT_TEMPLATE = """Species name: {species_name}

Write one image-level caption for this wildlife image.

Requirements:
- The caption must include the scientific name of the species.
- The scientific name should appear naturally in the sentence.
- Do not add background knowledge not visible in the image.
- Describe only the visible animal and the immediately visible surroundings.
- Output one concise English caption.
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare image-level caption generation inputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(Path(args.manifest)):
        rows.append(
            {
                "split": row["split"],
                "taxonomy_class": row["taxonomy_class"],
                "class_id": row["class_id"],
                "scientific_name": row["scientific_name"],
                "relative_image_path": row["relative_image_path"],
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT_TEMPLATE.format(species_name=row["scientific_name"])
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
