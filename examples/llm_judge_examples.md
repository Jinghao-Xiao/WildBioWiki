# LLM-J Examples

## Caption Example

Reference caption:

`Vulpes vulpes stands on a grassy woodland edge.`

Predicted caption:

`Vulpes vulpes stands near grass at the edge of a wooded area.`

Illustrative judge output:

```json
{
  "caption_judge": {
    "verdict": "Correct",
    "score": 91,
    "reason": "Semantically aligned with the reference."
  },
  "qa_judges": []
}
```

## QA Example

Question:

`What type of habitats does the animal in the image commonly occupy?`

Reference answer:

`It is found in forests, grasslands, and urban edges.`

Predicted answer:

`It commonly occupies forests and grasslands, including habitats near human settlements.`

Illustrative judge output:

```json
{
  "qid": "Q2",
  "verdict": "Correct",
  "score": 88,
  "reason": "Broadly consistent with the reference."
}
```
