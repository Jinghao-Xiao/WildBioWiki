# WildBioWiki

WildBioWiki is a knowledge-grounded biodiversity benchmark for two coordinated tasks:

- species-level question answering
- image-level captioning

It is built from iNaturalist wildlife imagery, structured species knowledge, retained supporting evidence, and image-level silver captions.

[Website](https://wildbiowiki.vercel.app) | [Dataset Release](https://drive.google.com/drive/folders/1et2e-RDGYtzjO80vihIxtfFQ0DTb-Q71?usp=share_link) | [Reproducibility](./REPRODUCIBILITY.md) | [Dataset Card](./DATASET_CARD.md)

![WildBioWiki sample](https://wildbiowiki.vercel.app/web-assets/sample.jpg)

## At a Glance

| Item | Value |
| --- | --- |
| Species | `392` |
| Images | `264,563` |
| Image-question QA instances | `2,116,504` |
| Vertebrate classes | `Actinopterygii`, `Amphibia`, `Aves`, `Mammalia`, `Reptilia` |
| Candidate pool | `1,000` species total, balanced as `200` species per class |
| Image splits | train `211,492`, validation `26,283`, test `26,788` |
| QA design | 8 fixed question slots per species |
| QA groups | `Basic`, `Compositional`, `Reasoning` |
| Caption design | image-level silver captions generated from image plus scientific name |

## Public Resources

- Website: [wildbiowiki.vercel.app](https://wildbiowiki.vercel.app)
- GitHub: [Jinghao-Xiao/WildBioWiki](https://github.com/Jinghao-Xiao/WildBioWiki)
- Dataset release: [Google Drive folder](https://drive.google.com/drive/folders/1et2e-RDGYtzjO80vihIxtfFQ0DTb-Q71?usp=share_link)

## What This Repository Covers

This code release is intended to cover the construction and evaluation workflow used in the WildBioWiki paper:

- candidate species pool construction
- knowledge retrieval and structuring
- image filtering and manifest preparation
- species-level QA construction and validation
- image-level caption generation
- benchmark evaluation and LLM-based semantic judging

The paired dataset release is staged separately under the Google Drive package `WildBioWiki-QA-medium_release`.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`scripts/`](./scripts) | construction, validation, generation, and evaluation scripts |
| [`prompts/`](./prompts) | released prompt materials for species-level QA, captioning, and LLM judging |
| [`configs/`](./configs) | released judge configuration files |
| [`schemas/`](./schemas) | released output schemas |
| [`examples/`](./examples) | worked prompt and judging examples |
| [`assets/iNaturalist/`](./assets/iNaturalist) | detector prompt mapping and iNaturalist name-mapping assets |
| [`DATASET_CARD.md`](./DATASET_CARD.md) | benchmark summary for code users |
| [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) | pointer map for paper-aligned construction and evaluation materials |
| [`MODEL_PATCHES.md`](./MODEL_PATCHES.md) | notes on local model compatibility edits used during experimentation |

## Prompt and Schema Index

### Prompt Templates

- [`prompts/species_level_qa_generation.md`](./prompts/species_level_qa_generation.md)
- [`prompts/image_level_caption_generation.md`](./prompts/image_level_caption_generation.md)
- [`prompts/llm_judge_gpt54mini.md`](./prompts/llm_judge_gpt54mini.md)

### Configs

- [`configs/llm_judge_gpt54mini.json`](./configs/llm_judge_gpt54mini.json)

### Schemas

- [`schemas/species_level_qa_output_schema.json`](./schemas/species_level_qa_output_schema.json)
- [`schemas/llm_judge_output_schema.json`](./schemas/llm_judge_output_schema.json)

### Examples

- [`examples/species_level_qa_example.json`](./examples/species_level_qa_example.json)
- [`examples/llm_judge_examples.md`](./examples/llm_judge_examples.md)

## Key Released Files

- [`environment.yml`](./environment.yml): conda environment definition
- [`requirements.txt`](./requirements.txt): minimal pip installation path
- [`inat_download_manifest_396.jsonl`](./inat_download_manifest_396.jsonl): class-level image download manifest
- [`CITATION.cff`](./CITATION.cff): citation metadata
- [`LICENSE`](./LICENSE): Apache License 2.0 for the code release package

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate wildbiowiki-vqa
```

Pip:

```bash
python -m pip install -r requirements.txt
```

## Notes

- This code release and the dataset release are separate.
- The dataset contains mixed-source imagery. Image reuse is governed by source-specific attribution and license requirements; see the dataset package for details.
- The current release package does not redistribute third-party model weights.
- Some local runs required compatibility edits to third-party model code; see [`MODEL_PATCHES.md`](./MODEL_PATCHES.md).
- Detailed prompt materials, grading criteria, configs, schemas, judge details, and examples released for the paper are organized under [`prompts/`](./prompts), [`configs/`](./configs), [`schemas/`](./schemas), [`examples/`](./examples), and [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

## Citation

Please cite the WildBioWiki paper and dataset release when using this code or the associated benchmark assets. See [`CITATION.cff`](./CITATION.cff).
