# WildBioWiki

WildBioWiki is a knowledge-grounded biodiversity benchmark with two coordinated tasks:

- species-level question answering (QA)
- image-level captioning

The current release covers:

- 392 species
- 264,563 images
- 3,136 species-level QA pairs
- five vertebrate classes: `Actinopterygii`, `Amphibia`, `Aves`, `Mammalia`, and `Reptilia`

This code release packages the benchmark construction and evaluation workflow used for the WildBioWiki paper. The benchmark is built from iNaturalist wildlife imagery, structured species knowledge, evidence-preserving QA construction, and image-level silver captions.

## Public Resources

- website: `https://wildbiowiki.vercel.app/`
- GitHub: `https://github.com/Jinghao-Xiao/WildBioWiki`
- dataset release: `https://drive.google.com/drive/folders/1et2e-RDGYtzjO80vihIxtfFQ0DTb-Q71?usp=share_link`

## Repository Scope

This release is intended to cover:

- species pool construction
- knowledge retrieval and structuring
- image filtering and manifest preparation
- species-level QA construction and validation
- image-level caption generation
- benchmark evaluation and LLM-judge scoring

The paired dataset release is staged separately under the Google Drive package:

- `WildBioWiki-QA-medium_release`

## Benchmark Summary

- candidate pool: 1,000 species total, balanced as 200 species per vertebrate class
- final retained set: 392 species
- image split counts:
  - train: 211,492
  - validation: 26,283
  - test: 26,788
- QA design:
  - 8 fixed question slots per species
  - grouped as `Basic`, `Compositional`, and `Reasoning`
- captions:
  - image-level silver captions
  - generated from image plus scientific name
  - visible content only

## Included Files

- `environment.yml`: conda environment definition
- `requirements.txt`: minimal pip installation path
- `inat_download_manifest_396.jsonl`: class-level image download manifest
- `DATASET_CARD.md`: benchmark summary for code users
- `MODEL_PATCHES.md`: notes on local model compatibility edits used during experimentation
- `CITATION.cff`: citation metadata
- `LICENSE`: code license for this release package
- `REPRODUCIBILITY.md`: paper-aligned pointer map for construction and evaluation materials
- `prompts/`: released prompt materials for captioning, QA, and LLM-judge evaluation
- `configs/`: released judge configuration files
- `schemas/`: released output schemas
- `examples/`: released worked examples for prompt and judge outputs

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

## Important Notes

- This code release and the dataset release are separate.
- The dataset contains mixed-source imagery. Image reuse is governed by source-specific attribution and license requirements; see the dataset package for details.
- The current release package does not redistribute third-party model weights.
- Some local runs required compatibility edits to third-party model code; see `MODEL_PATCHES.md`.
- Detailed prompt materials, grading criteria, configs, schemas, judge details, and examples released for the paper are organized under `prompts/`, `configs/`, `schemas/`, `examples/`, and `REPRODUCIBILITY.md`.

## Citation

Please cite the WildBioWiki paper and dataset release when using this code or the associated benchmark assets. See `CITATION.cff`.
