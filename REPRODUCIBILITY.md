# Reproducibility Materials

This document maps the WildBioWiki paper claims about released construction and evaluation materials to files in this repository.

## Released Prompt Materials

- `prompts/species_level_qa_generation.md`
- `prompts/image_level_caption_generation.md`
- `prompts/llm_judge_gpt54mini.md`

## Released Configs

- `configs/llm_judge_gpt54mini.json`

## Released Schemas

- `schemas/species_level_qa_output_schema.json`
- `schemas/llm_judge_output_schema.json`

## Released Examples

- `examples/llm_judge_examples.md`
- `examples/species_level_qa_example.json`

## Construction and Evaluation Code Map

### Species Filtering and Candidate Pool Construction

- `scripts/build_common_species_pool.py`

### Knowledge Structuring Inputs

- `scripts/build_llm_kb_structuring_common_pool_inputs.py`

### Visual Filtering and Bounding-Box Mapping

- `scripts/build_inat_name_mapping.py`
- `scripts/build_inat_detector_prompts.py`
- `scripts/run_grounding_dino_taxonomy.py`
- `scripts/filter_and_map_grounding_dino.py`

### Species-Level QA Generation

- `scripts/build_taxonomy_v1_full_qa_requests.py`

### QA Post-Processing and Validation

- `scripts/postfix_taxonomy_v1_full_qa.py`
- `scripts/validate_taxonomy_v1_full_qa.py`

### Image-Level Caption Generation

- `scripts/run_caption_responses_openai.py`

### Final Evaluation

- `scripts/evaluate_caption_qa_llmj_openai.py`

## Release Note

The repository is organized as a release-oriented code package rather than a full dump of every intermediate local experiment directory. The files listed above are the paper-aligned construction and evaluation materials intended to support method inspection and reproduction.
