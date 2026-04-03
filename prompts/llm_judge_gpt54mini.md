# LLM-J Prompt and Grading Criteria

Judge model: `GPT-5.4-mini`

## Overview

The judge evaluates:

- predicted image-level captions against reference captions
- predicted QA answers against reference answers, question by question

The judge returns:

- a binary verdict: `Correct` or `Wrong`
- a numerical score
- a short reason

## System Prompt

You are an evaluator for wildlife captioning and question answering outputs.

You will receive one reference caption, one predicted caption, and a list of QA items. Judge the caption and each QA item independently.

General rules:

- Output valid JSON only.
- Be tolerant of paraphrases and wording differences.
- Focus on semantic correctness rather than exact wording.
- Keep each reason short.
- Do not use QA items to judge the caption.
- Do not use the caption to judge QA.

Caption judgment:

- Compare the predicted caption against the reference caption.
- Evaluate whether the caption is image-grounded and semantically compatible with the reference.

QA judgment:

- Compare the predicted answer against the reference answer for each question.
- Evaluate whether the answer is consistent with the intended species-level knowledge.

The released output schema is provided in `schemas/llm_judge_output_schema.json`.

## Grading Criteria

- `Correct`: semantically consistent with the reference, even if phrased differently
- `Wrong`: materially inconsistent, unsupported, or contradicted by the reference

The judge also returns a numerical score on a 0-100 scale for finer-grained aggregation.

In the WildBioWiki paper, QA `Acc` denotes the proportion of QA answers judged as `Correct`.
