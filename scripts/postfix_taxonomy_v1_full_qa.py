from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "Q1": ("knowledge_lookup", "scientific_name_lookup", "basic", False),
    "Q2": ("knowledge_lookup", "habitat_lookup", "basic", False),
    "Q3": ("knowledge_lookup", "diet_lookup", "basic", False),
    "Q4": ("knowledge_lookup", "distribution_lookup", "basic", False),
    "Q5": ("attribute_composition", "habitat_diet_composition", "compositional", False),
    "Q6": ("attribute_composition", "behavior_distribution_composition", "compositional", False),
    "Q7": ("core_extra_reasoning", "extra_grounded_synthesis", "reasoning", True),
    "Q8": ("negative_contrastive_reasoning", "contrastive_reasoning", "reasoning", False)
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def fix_row(row: dict) -> dict:
    qa_pairs = row["qa_pairs"]
    for qid, (qtype, reasoning, difficulty, uses_extra) in EXPECTED.items():
        qa_pairs[qid]["type"] = qtype
        qa_pairs[qid]["reasoning_type"] = reasoning
        qa_pairs[qid]["difficulty"] = difficulty
        qa_pairs[qid]["uses_extra"] = uses_extra
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply lightweight post-fixes to taxonomy-v1 QA outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [fix_row(row) for row in load_jsonl(Path(args.input))]
    write_jsonl(Path(args.output), rows)


if __name__ == "__main__":
    main()
