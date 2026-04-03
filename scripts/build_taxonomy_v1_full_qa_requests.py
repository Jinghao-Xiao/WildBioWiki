from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = """You are generating outside-knowledge wildlife VQA data.
Use only the provided species knowledge record.
Do not use external knowledge.
Do not infer new facts from the image itself.
Assume the image contains an instance of the target species, but all answers must come from the species knowledge record.
Return valid JSON only."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_user_prompt(entry: dict) -> str:
    return f"""Generate exactly 8 QA pairs for a species-level wildlife QA benchmark.

Use these fixed slots:
Q1 scientific_name_lookup
Q2 habitat_lookup
Q3 diet_lookup
Q4 distribution_lookup
Q5 habitat_diet_composition
Q6 behavior_distribution_composition
Q7 extra_grounded_synthesis
Q8 contrastive_reasoning

The question must not reveal the species name.
Use only the provided species knowledge record.
Preserve supporting_evidence and evidence_fields.

Species knowledge record:
{json.dumps(entry, ensure_ascii=True)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA request payloads for taxonomy-v1 full generation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    entries = load_jsonl(Path(args.input))
    rows = []
    for entry in entries:
        rows.append(
            {
                "class_id": entry["class_id"],
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_user_prompt(entry),
                "expected_qa_count": 8,
                "generation_mode": "qa_taxonomy_v1_full"
            }
        )
    write_jsonl(Path(args.output), rows)


if __name__ == "__main__":
    main()
