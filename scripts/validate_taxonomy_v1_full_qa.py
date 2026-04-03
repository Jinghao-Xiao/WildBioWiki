from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TYPES = {
    "Q1": ("knowledge_lookup", "scientific_name_lookup", "basic"),
    "Q2": ("knowledge_lookup", "habitat_lookup", "basic"),
    "Q3": ("knowledge_lookup", "diet_lookup", "basic"),
    "Q4": ("knowledge_lookup", "distribution_lookup", "basic"),
    "Q5": ("attribute_composition", "habitat_diet_composition", "compositional"),
    "Q6": ("attribute_composition", "behavior_distribution_composition", "compositional"),
    "Q7": ("core_extra_reasoning", "extra_grounded_synthesis", "reasoning"),
    "Q8": ("negative_contrastive_reasoning", "contrastive_reasoning", "reasoning")
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_row(row: dict) -> list[dict]:
    issues = []
    for qid, (expected_type, expected_reasoning, expected_difficulty) in EXPECTED_TYPES.items():
        qa = row["qa_pairs"][qid]
        if qa["type"] != expected_type:
            issues.append({"qid": qid, "issue": "wrong_type", "detail": qa["type"]})
        if qa["reasoning_type"] != expected_reasoning:
            issues.append({"qid": qid, "issue": "wrong_reasoning_type", "detail": qa["reasoning_type"]})
        if qa["difficulty"] != expected_difficulty:
            issues.append({"qid": qid, "issue": "wrong_difficulty", "detail": qa["difficulty"]})
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate taxonomy-v1 QA outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    report_rows = []
    for row in rows:
        issues = validate_row(row)
        report_rows.append({"class_id": row["class_id"], "issue_count": len(issues), "issues": issues})

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"row_count": len(rows), "results": report_rows}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
