#!/usr/bin/env python3
"""Build broader, detector-oriented prompts for Grounding DINO / VLM use."""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CLASS_GENERIC_LABEL = {
    "Actinopterygii": "fish",
    "Amphibia": "amphibian",
    "Aves": "bird",
    "Mammalia": "mammal",
    "Reptilia": "reptile",
}


HEAD_PHRASES = {
    "Actinopterygii": [
        "sergeant major",
        "surgeon fish",
        "surgeonfish",
        "butterfly fish",
        "butterflyfish",
        "parrot fish",
        "parrotfish",
        "box fish",
        "boxfish",
        "puffer fish",
        "pufferfish",
        "hawk fish",
        "hawkfish",
        "sun fish",
        "sunfish",
        "cat fish",
        "catfish",
        "porcupine fish",
        "porcupinefish",
        "damselfish",
        "stickleback",
        "sculpin",
        "wrasse",
        "snapper",
        "salmon",
        "sucker",
        "tilapia",
        "toadfish",
        "trevally",
        "perch",
        "trout",
        "carp",
        "darter",
        "gizzard shad",
        "shad",
        "ray",
        "goby",
        "gudgeon",
        "bass",
        "eel",
        "shark",
        "minnow",
        "mola",
        "molly",
        "pike",
        "chub",
        "idol",
        "crappie",
    ],
    "Amphibia": [
        "tree frog",
        "cricket frog",
        "chorus frog",
        "painted frog",
        "burrowing frog",
        "bullfrog",
        "salamander",
        "spadefoot",
        "froglet",
        "peeper",
        "toad",
        "frog",
        "newt",
    ],
    "Aves": [
        "sea eagle",
        "wood pewee",
        "woodpecker",
        "hummingbird",
        "kingfisher",
        "blackbird",
        "goldfinch",
        "cormorant",
        "mockingbird",
        "wagtail",
        "sparrow",
        "heron",
        "goose",
        "robin",
        "pigeon",
        "egret",
        "cardinal",
        "starling",
        "dove",
        "vulture",
        "junco",
        "warbler",
        "eagle",
        "osprey",
        "swallow",
        "turkey",
        "swan",
        "mallard",
        "owl",
        "pelican",
        "duck",
        "gull",
        "tern",
        "coot",
        "grebe",
        "ibis",
        "stork",
        "falcon",
        "hawk",
        "kite",
        "parakeet",
        "parrot",
    ],
    "Mammalia": [
        "sea lion",
        "kangaroo rat",
        "ground squirrel",
        "tree squirrel",
        "cottontail",
        "chipmunk",
        "opossum",
        "sea otter",
        "river otter",
        "marmoset",
        "monkey",
        "wallaby",
        "kangaroo",
        "porcupine",
        "raccoon",
        "beaver",
        "rabbit",
        "bobcat",
        "coyote",
        "coati",
        "elephant",
        "squirrel",
        "deer",
        "fox",
        "hare",
        "sheep",
        "dog",
        "rat",
        "bat",
        "seal",
        "otter",
        "cat",
        "bear",
        "moose",
        "bison",
        "boar",
        "pig",
        "elk",
        "wolf",
        "mole",
        "muskrat",
    ],
    "Reptilia": [
        "sea turtle",
        "house gecko",
        "water dragon",
        "spiny tailed iguana",
        "whiptail",
        "rattlesnake",
        "watersnake",
        "ratsnake",
        "copperhead",
        "cottonmouth",
        "crocodile",
        "alligator",
        "tortoise",
        "gecko",
        "iguana",
        "lizard",
        "turtle",
        "snake",
        "anole",
        "dragon",
        "boa",
        "python",
        "skink",
    ],
}


POSSESSIVE_PATTERN = re.compile(r"\b([a-z]+)'s\b")
MULTISPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector-friendly prompt tables.")
    parser.add_argument("--manifest", default="inat_download_manifest_396.jsonl")
    parser.add_argument("--output-csv", default="data/iNaturalist/inat_detector_prompts_396.csv")
    parser.add_argument("--output-jsonl", default="data/iNaturalist/inat_detector_prompts_396.jsonl")
    return parser.parse_args()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text).lower().strip()
    text = POSSESSIVE_PATTERN.sub("", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_phrase(text: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", text.strip().lower())


def unique_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def find_head_phrase(clean_name: str, taxonomy_class: str, generic_label: str) -> str:
    candidates = [normalize_phrase(x) for x in HEAD_PHRASES.get(taxonomy_class, [])]
    tokens = clean_name.split()

    for phrase in sorted(candidates, key=lambda x: len(x.split()), reverse=True):
        phrase_tokens = phrase.split()
        size = len(phrase_tokens)
        if tokens[-size:] == phrase_tokens:
            return phrase

    for phrase in sorted(candidates, key=lambda x: len(x.split()), reverse=True):
        if phrase in clean_name:
            return phrase

    if tokens:
        return tokens[-1]
    return generic_label


def build_record(row: Dict) -> Dict:
    generic_label = CLASS_GENERIC_LABEL[row["taxonomy_class"]]
    common_name_original = row["common_name"].strip()
    detector_name_specific = normalize_text(common_name_original)
    detector_name_broad = find_head_phrase(detector_name_specific, row["taxonomy_class"], generic_label)

    prompt_primary = f"{detector_name_specific}. {generic_label}."
    prompt_fallback = f"{detector_name_broad}. {generic_label}."
    prompt_generic = f"{generic_label}."

    recommended = prompt_primary
    strategy = "specific"
    if detector_name_broad != detector_name_specific:
        recommended = prompt_fallback
        strategy = "broad_head"

    aliases = unique_keep_order([detector_name_specific, detector_name_broad, generic_label])

    return {
        "taxon_id": row["taxon_id"],
        "scientific_name": row["scientific_name"],
        "common_name_original": common_name_original,
        "taxonomy_class": row["taxonomy_class"],
        "generic_label": generic_label,
        "detector_name_specific": detector_name_specific,
        "detector_name_broad": detector_name_broad,
        "grounding_dino_prompt_primary": prompt_primary,
        "grounding_dino_prompt_recommended": recommended,
        "grounding_dino_prompt_fallback": prompt_fallback,
        "grounding_dino_prompt_generic": prompt_generic,
        "qwen_label_recommended": detector_name_broad if detector_name_broad != detector_name_specific else detector_name_specific,
        "prompt_strategy": strategy,
        "aliases": "|".join(aliases),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / args.manifest
    output_csv = project_root / args.output_csv
    output_jsonl = project_root / args.output_jsonl
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8") if line.strip()]
    records = [build_record(row) for row in rows]

    fieldnames = [
        "taxon_id",
        "scientific_name",
        "common_name_original",
        "taxonomy_class",
        "generic_label",
        "detector_name_specific",
        "detector_name_broad",
        "grounding_dino_prompt_primary",
        "grounding_dino_prompt_recommended",
        "grounding_dino_prompt_fallback",
        "grounding_dino_prompt_generic",
        "qwen_label_recommended",
        "prompt_strategy",
        "aliases",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_csv}")
    print(f"Wrote {len(records)} records to {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
