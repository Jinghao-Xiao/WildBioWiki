from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate caption and QA predictions with the released LLM-J protocol.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL in benchmark format.")
    parser.add_argument("--judge-output", required=True, help="Judge output JSONL following schemas/llm_judge_output_schema.json")
    parser.add_argument("--summary", required=True, help="Output summary JSON path.")
    args = parser.parse_args()

    preds = read_jsonl(Path(args.predictions))
    judge_rows = read_jsonl(Path(args.judge_output))
    judge_map = {row["image_path"]: row for row in judge_rows}

    caption_scores = []
    qa_scores = []
    qa_acc = []
    group_acc = {"Basic": [], "Compositional": [], "Reasoning": []}
    groups = {
        "Basic": {"Q1", "Q2", "Q3", "Q4"},
        "Compositional": {"Q5", "Q6"},
        "Reasoning": {"Q7", "Q8"}
    }

    for row in preds:
        judgment = judge_map.get(row["image_path"])
        if not judgment:
            continue
        caption_scores.append(judgment["caption_judge"]["score"])
        for qa in judgment["qa_judges"]:
            qa_scores.append(qa["score"])
            verdict = qa["verdict"] == "Correct"
            qa_acc.append(int(verdict))
            for group_name, qids in groups.items():
                if qa["qid"] in qids:
                    group_acc[group_name].append(int(verdict))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "judged_images": len(judge_map),
        "caption_llm_j": mean(caption_scores),
        "qa_llm_j": mean(qa_scores),
        "qa_acc": mean(qa_acc),
        "basic_acc": mean(group_acc["Basic"]),
        "compositional_acc": mean(group_acc["Compositional"]),
        "reasoning_acc": mean(group_acc["Reasoning"])
    }

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
