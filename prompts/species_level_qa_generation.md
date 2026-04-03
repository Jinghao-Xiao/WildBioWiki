# Species-Level QA Generation Prompt

## System Prompt

You are generating outside-knowledge wildlife VQA data.
Use only the provided species knowledge record.
Do not use external knowledge.
Do not infer new facts from the image itself.
Assume the image contains an instance of the target species, but all answers must be grounded in the provided species knowledge record.
Return valid JSON only.

## User Prompt Template

Generate exactly 8 QA pairs for a species-level wildlife QA benchmark.

Requirements:

- Use exactly Q1 through Q8.
- Questions must not reveal the common name or scientific name.
- Questions should refer to "the animal in the image" or "the species shown in the image".
- Answers must be grounded in the species knowledge record only.
- Preserve `supporting_evidence` and `evidence_fields`.

Question slots:

1. scientific name lookup
2. habitat lookup
3. diet lookup
4. distribution lookup
5. habitat-diet composition
6. behavior-distribution composition
7. extra-knowledge synthesis
8. contrastive reasoning

Reporting groups:

- Basic = Q1-Q4
- Compositional = Q5-Q6
- Reasoning = Q7-Q8

The released output schema is provided in `schemas/species_level_qa_output_schema.json`.
