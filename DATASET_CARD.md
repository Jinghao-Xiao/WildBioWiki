# WildBioWiki Dataset Card

## Summary

WildBioWiki is a knowledge-grounded biodiversity benchmark pairing wildlife images with structured species knowledge, species-level QA, and image-level captions.

The benchmark is designed to test both:

- visual grounding from input wildlife images
- species-grounded use of external biological knowledge

## Tasks

### Task A: Image-Level Captioning

- one concise English caption per image
- silver reference captions
- generated under an image-plus-scientific-name construction protocol
- intended to describe visible content only

### Task B: Species-Level QA

- 8 fixed QA slots per retained species
- question families cover:
  - scientific name lookup
  - habitat lookup
  - diet lookup
  - distribution lookup
  - habitat-diet composition
  - behavior-distribution composition
  - extra-knowledge synthesis
  - contrastive reasoning

Grouped reporting categories:

- `Basic` = Q1-Q4
- `Compositional` = Q5-Q6
- `Reasoning` = Q7-Q8

## Scale

- 392 species
- 264,563 images
- 2,116,504 image-question QA instances
- five vertebrate classes

## Data Sources

- iNaturalist wildlife imagery
- Wikipedia as the main text source
- Wikidata as the main structured source
- GBIF, EOL, and IUCN as auxiliary sources used in later knowledge enrichment

## Key Design Choices

- candidate pool balanced across five vertebrate classes
- image filtering with open-vocabulary detection and bounding-box retention
- knowledge base split into `core` fields and `extra` blocks
- retained `supporting_evidence` and `evidence_fields` for QA provenance
- benchmark-level packaging for captioning and QA evaluation

## Release Pairing

This code package is paired with the dataset release package:

- `WildBioWiki-QA-medium_release`

See the dataset package for imagery, manifests, attribution notes, and release-level data files.
