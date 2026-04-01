#!/usr/bin/env python3

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dataset_root = PROJECT_ROOT / "data" / "WildBioWiki-QA"
    old_path = dataset_root / "manual_review_usable396_qa_final.jsonl"
    qa_dir = dataset_root / "qa"
    new_path = qa_dir / "species_qa_by_class_id.jsonl"
    summary_path = qa_dir / "species_qa_summary.json"

    qa_dir.mkdir(parents=True, exist_ok=True)

    if not old_path.exists() and not new_path.exists():
        raise FileNotFoundError("Neither old nor canonical QA file exists.")

    source_path = old_path if old_path.exists() else new_path
    rows = []
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rows = sorted(rows, key=lambda row: (row["taxonomy_class"], row["common_name"], str(row["class_id"])))
    with new_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_component": "species_qa",
        "canonical_file": str(new_path.relative_to(PROJECT_ROOT)),
        "join_key": "class_id",
        "row_count": len(rows),
        "class_count": len({row["taxonomy_class"] for row in rows}),
        "species_count": len(rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if old_path.exists() and old_path.resolve() != new_path.resolve():
        old_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
