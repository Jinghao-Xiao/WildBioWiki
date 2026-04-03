from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_USER_AGENT = "WildBioWiki/1.0 (species pool builder)"
INAT_TAXA_SEARCH_URL = "https://api.inaturalist.org/v1/taxa"
INAT_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"

CLASS_CONFIG = [
    {"class_english": "Aves", "taxon_id": "3"},
    {"class_english": "Mammalia", "taxon_id": "40151"},
    {"class_english": "Amphibia", "taxon_id": "20978"},
    {"class_english": "Reptilia", "taxon_id": "26036"},
    {"class_english": "Actinopterygii", "taxon_id": "47178"},
]


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_species_page(class_taxon_id: str, page: int, per_page: int) -> list[dict]:
    params = {
        "taxon_id": class_taxon_id,
        "rank": "species",
        "is_active": "true",
        "per_page": str(per_page),
        "page": str(page),
        "order": "desc",
        "order_by": "observations_count",
    }
    url = f"{INAT_TAXA_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    payload = fetch_json(url)
    return payload.get("results", [])


def get_research_photo_count(taxon_id: str) -> int:
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": "1",
    }
    url = f"{INAT_OBSERVATIONS_URL}?{urllib.parse.urlencode(params)}"
    payload = fetch_json(url)
    return int(payload.get("total_results", 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced common-species pool from iNaturalist.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-per-class", type=int, default=200)
    parser.add_argument("--min-total-observations", type=int, default=1000)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for class_meta in CLASS_CONFIG:
        collected = 0
        page = 1
        while collected < args.top_per_class:
            candidates = get_species_page(class_meta["taxon_id"], page, min(200, args.top_per_class))
            if not candidates:
                break
            for candidate in candidates:
                total_obs = int(candidate.get("observations_count", 0) or 0)
                if total_obs < args.min_total_observations:
                    break
                taxon_id = str(candidate.get("id", "") or "")
                rows.append(
                    {
                        "class_english": class_meta["class_english"],
                        "preferred_common_name": str(candidate.get("preferred_common_name", "") or ""),
                        "species_name": str(candidate.get("name", "") or ""),
                        "taxon_id": taxon_id,
                        "inat_total_observations": total_obs,
                        "inat_research_photo_observations": get_research_photo_count(taxon_id),
                        "wikipedia_url": str(candidate.get("wikipedia_url", "") or ""),
                    }
                )
                collected += 1
                if collected >= args.top_per_class:
                    break
                time.sleep(args.delay_seconds)
            page += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
