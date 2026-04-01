#!/usr/bin/env python3
"""Download iNaturalist images for taxa listed in a manifest.

The downloader is designed for long-running, resumable jobs:
- It never modifies the input manifest.
- It keeps per-species metadata and summary files.
- It can resume from existing metadata without redownloading completed items.
- It records image/API failures without aborting the whole run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests


API_URL = "https://api.inaturalist.org/v1/observations"
PHOTO_SIZE_PATTERN = re.compile(r"/(square|small|medium|large|original)\.")
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download iNaturalist images from a JSONL manifest.")
    parser.add_argument("--manifest", default="inat_download_manifest_396.jsonl")
    parser.add_argument("--output-root", default="data/iNaturalist/inaturalist_images_396")
    parser.add_argument("--target-per-species", type=int, default=800)
    parser.add_argument("--api-per-page", type=int, default=200)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--species",
        action="append",
        default=[],
        help="Repeatable filter. Matches taxon_id, scientific_name, common_name, or download_slug.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing metadata.")
    parser.add_argument(
        "--overwrite-missing-metadata",
        action="store_true",
        help="Redownload files even if the target image file already exists but has no metadata entry.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def slugify_scientific_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def selector_set(values: Sequence[str]) -> set:
    return {value.strip().lower() for value in values if value and value.strip()}


def filter_species(rows: Sequence[Dict], selectors: Sequence[str]) -> List[Dict]:
    wanted = selector_set(selectors)
    if not wanted:
        return list(rows)

    filtered = []
    for row in rows:
        candidates = {
            str(row.get("taxon_id", "")).strip().lower(),
            str(row.get("scientific_name", "")).strip().lower(),
            str(row.get("common_name", "")).strip().lower(),
            str(row.get("download_slug", "")).strip().lower(),
        }
        if wanted & candidates:
            filtered.append(row)
    return filtered


def build_species_paths(project_root: Path, output_root: Path, manifest_row: Dict) -> Dict[str, Path]:
    class_dir = output_root / manifest_row["taxonomy_class"]
    species_dir = class_dir / slugify_scientific_name(manifest_row["scientific_name"])
    images_dir = species_dir / "images"
    return {
        "species_dir": species_dir,
        "images_dir": images_dir,
        "metadata": species_dir / "metadata.jsonl",
        "summary": species_dir / "species_summary.json",
        "failures": output_root / "download_failures.jsonl",
        "log_dir": output_root / "logs",
        "global_summary": output_root / "download_summary.csv",
        "project_root": project_root,
    }


def split_location(location: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not location or "," not in location:
        return None, None
    try:
        lat_str, lon_str = location.split(",", 1)
        return float(lat_str), float(lon_str)
    except ValueError:
        return None, None


def build_candidate_urls(photo_url: Optional[str]) -> List[str]:
    if not photo_url:
        return []

    candidates = []
    for size in ("original", "large", "medium"):
        candidate = PHOTO_SIZE_PATTERN.sub("/%s." % size, photo_url, count=1)
        if candidate not in candidates:
            candidates.append(candidate)
    if photo_url not in candidates:
        candidates.append(photo_url)
    return candidates


def extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return suffix
    return ".jpg"


def build_metadata_record(manifest_row: Dict, observation: Dict, photo: Dict, local_path: Path, project_root: Path, image_url: str) -> Dict:
    latitude, longitude = split_location(observation.get("location"))
    photo_license = photo.get("license_code")
    observation_license = observation.get("license_code")
    return {
        "image_id": photo.get("id"),
        "observation_id": observation.get("id"),
        "taxon_id": manifest_row.get("taxon_id"),
        "common_name": manifest_row.get("common_name") or observation.get("taxon", {}).get("preferred_common_name"),
        "scientific_name": manifest_row.get("scientific_name") or observation.get("taxon", {}).get("name"),
        "image_url": image_url,
        "local_path": str(local_path.relative_to(project_root)),
        "license": photo_license or observation_license,
        "observed_on": observation.get("observed_on"),
        "latitude": latitude,
        "longitude": longitude,
        "quality_grade": observation.get("quality_grade"),
    }


def parse_existing_metadata(metadata_path: Path, project_root: Path) -> Tuple[Dict[int, Dict], set, set]:
    metadata_by_observation: Dict[int, Dict] = {}
    valid_observation_ids = set()
    valid_image_ids = set()

    if not metadata_path.exists():
        return metadata_by_observation, valid_observation_ids, valid_image_ids

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            observation_id = record.get("observation_id")
            image_id = record.get("image_id")
            local_path = record.get("local_path")
            if observation_id is None or image_id is None or not local_path:
                continue

            absolute_path = project_root / local_path
            if not absolute_path.exists():
                continue

            metadata_by_observation[int(observation_id)] = record
            valid_observation_ids.add(int(observation_id))
            valid_image_ids.add(int(image_id))

    return metadata_by_observation, valid_observation_ids, valid_image_ids


def make_api_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "wildlife_vqa_inat_downloader/1.0"})
    return session


def get_thread_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = make_api_session()
        THREAD_LOCAL.session = session
    return session


def api_get_json(session: requests.Session, params: Dict, timeout: int, max_retries: int) -> Dict:
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(API_URL, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError("retryable status %s" % response.status_code)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise RuntimeError("API request failed after %s attempts: %s" % (attempt, exc)) from exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


def download_image(candidate: Dict, timeout: int, max_retries: int, overwrite_existing_file: bool) -> Dict:
    destination = Path(candidate["destination"])
    if destination.exists() and not overwrite_existing_file:
        return {
            "ok": True,
            "candidate": candidate,
            "used_url": candidate["candidate_urls"][0],
            "reused_existing_file": True,
        }

    ensure_dir(destination.parent)
    errors = []
    session = get_thread_session()

    for url in candidate["candidate_urls"]:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(url, stream=True, timeout=timeout)
                if response.status_code == 200:
                    tmp_path = destination.with_suffix(destination.suffix + ".part")
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    tmp_path.replace(destination)
                    return {
                        "ok": True,
                        "candidate": candidate,
                        "used_url": url,
                        "reused_existing_file": False,
                    }

                if response.status_code == 404:
                    errors.append("%s -> 404" % url)
                    break

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError("retryable status %s for %s" % (response.status_code, url))

                errors.append("%s -> %s" % (url, response.status_code))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= max_retries:
                    errors.append("%s -> %s" % (url, exc))
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    return {
        "ok": False,
        "candidate": candidate,
        "errors": errors,
    }


def write_summary_csv(path: Path, summaries: Sequence[Dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "taxon_id",
        "common_name",
        "scientific_name",
        "downloaded_image_count",
        "successful_observation_count",
        "failure_count",
        "output_dir",
        "status",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})
    tmp_path.replace(path)


def collect_existing_summaries(output_root: Path) -> Dict[str, Dict]:
    summaries: Dict[str, Dict] = {}
    if not output_root.exists():
        return summaries

    for summary_path in output_root.rglob("species_summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        taxon_id = str(summary.get("taxon_id", "")).strip()
        if taxon_id:
            summaries[taxon_id] = summary
    return summaries


def build_failure_record(manifest_row: Dict, candidate: Optional[Dict], error_message: str) -> Dict:
    record = {
        "taxon_id": manifest_row.get("taxon_id"),
        "common_name": manifest_row.get("common_name"),
        "scientific_name": manifest_row.get("scientific_name"),
        "error": error_message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if candidate:
        record.update(
            {
                "observation_id": candidate.get("observation_id"),
                "image_id": candidate.get("image_id"),
                "destination": candidate.get("destination"),
            }
        )
    return record


def process_species(
    manifest_row: Dict,
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path,
) -> Dict:
    paths = build_species_paths(project_root, output_root, manifest_row)
    ensure_dir(paths["images_dir"])
    ensure_dir(paths["log_dir"])

    target_count = int(args.target_per_species)
    metadata_by_observation, completed_observation_ids, completed_image_ids = ({}, set(), set())
    if args.resume:
        metadata_by_observation, completed_observation_ids, completed_image_ids = parse_existing_metadata(
            paths["metadata"], project_root
        )

    downloaded_count = len(metadata_by_observation)
    failure_count = 0
    api_session = make_api_session()
    page = 1
    status = "completed"

    logging.info(
        "Processing taxon %s (%s), resume=%s, existing=%s, target=%s",
        manifest_row["taxon_id"],
        manifest_row["scientific_name"],
        bool(args.resume),
        downloaded_count,
        target_count,
    )

    try:
        while downloaded_count < target_count:
            params = {
                "taxon_id": manifest_row["taxon_id"],
                "quality_grade": "research",
                "photos": "true",
                "per_page": args.api_per_page,
                "page": page,
                "order_by": "created_at",
                "order": "desc",
            }
            payload = api_get_json(api_session, params, args.request_timeout, args.max_retries)
            results = payload.get("results", [])
            if not results:
                break

            futures = []
            with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
                for observation in results:
                    if downloaded_count + len(futures) >= target_count:
                        break

                    observation_id = observation.get("id")
                    if observation_id is None:
                        continue
                    observation_id = int(observation_id)
                    if observation_id in completed_observation_ids:
                        continue

                    observation_photos = observation.get("observation_photos") or []
                    if not observation_photos:
                        continue

                    first_photo = observation_photos[0].get("photo") or {}
                    image_id = first_photo.get("id")
                    photo_url = first_photo.get("url")
                    if image_id is None or not photo_url:
                        continue

                    image_id = int(image_id)
                    if image_id in completed_image_ids:
                        continue

                    candidate_urls = build_candidate_urls(photo_url)
                    if not candidate_urls:
                        continue

                    extension = extension_from_url(candidate_urls[0])
                    image_filename = "%s_%s%s" % (observation_id, image_id, extension)
                    absolute_destination = paths["images_dir"] / image_filename

                    candidate = {
                        "observation_id": observation_id,
                        "image_id": image_id,
                        "observation": observation,
                        "photo": first_photo,
                        "candidate_urls": candidate_urls,
                        "destination": str(absolute_destination),
                    }

                    file_exists_without_metadata = absolute_destination.exists() and observation_id not in completed_observation_ids
                    overwrite_existing_file = args.overwrite_missing_metadata and file_exists_without_metadata
                    future = executor.submit(
                        download_image,
                        candidate,
                        args.request_timeout,
                        args.max_retries,
                        overwrite_existing_file,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    result = future.result()
                    candidate = result["candidate"]
                    if result["ok"]:
                        absolute_destination = Path(candidate["destination"])
                        metadata_record = build_metadata_record(
                            manifest_row,
                            candidate["observation"],
                            candidate["photo"],
                            absolute_destination,
                            project_root,
                            result["used_url"],
                        )
                        metadata_by_observation[candidate["observation_id"]] = metadata_record
                        completed_observation_ids.add(candidate["observation_id"])
                        completed_image_ids.add(candidate["image_id"])
                        append_jsonl(paths["metadata"], metadata_record)
                        downloaded_count = len(metadata_by_observation)
                    else:
                        failure_count += 1
                        failure_record = build_failure_record(
                            manifest_row,
                            candidate,
                            "; ".join(result.get("errors", [])) or "download failed",
                        )
                        append_jsonl(paths["failures"], failure_record)

            logging.info(
                "Taxon %s page %s complete: downloaded=%s failures=%s",
                manifest_row["taxon_id"],
                page,
                downloaded_count,
                failure_count,
            )

            page += 1

    except Exception as exc:  # noqa: BLE001
        status = "failed"
        failure_count += 1
        append_jsonl(paths["failures"], build_failure_record(manifest_row, None, "species error: %s" % exc))
        logging.exception("Species %s failed", manifest_row["taxon_id"])

    if status != "failed" and downloaded_count < target_count:
        status = "completed_below_target"

    compact_records = [metadata_by_observation[key] for key in sorted(metadata_by_observation)]
    atomic_write_jsonl(paths["metadata"], compact_records)

    summary = {
        "taxon_id": manifest_row["taxon_id"],
        "common_name": manifest_row["common_name"],
        "scientific_name": manifest_row["scientific_name"],
        "target_image_count": target_count,
        "downloaded_image_count": downloaded_count,
        "successful_observation_count": downloaded_count,
        "failure_count": failure_count,
        "output_dir": str(paths["species_dir"].relative_to(project_root)),
        "status": status,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(paths["summary"], summary)
    return summary


def main() -> int:
    args = parse_args()
    setup_logging()

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (project_root / args.manifest).resolve()
    output_root = (project_root / args.output_root).resolve()
    ensure_dir(output_root)

    manifest_rows = load_manifest(manifest_path)
    selected_rows = filter_species(manifest_rows, args.species)
    if not selected_rows:
        raise SystemExit("No species matched the provided filters.")

    logging.info("Loaded %s manifest rows, selected %s rows", len(manifest_rows), len(selected_rows))

    current_summaries: Dict[str, Dict] = collect_existing_summaries(output_root)
    for row in selected_rows:
        summary = process_species(row, args, project_root, output_root)
        current_summaries[str(summary["taxon_id"])] = summary
        write_summary_csv(output_root / "download_summary.csv", list(current_summaries.values()))

    logging.info("Finished processing %s species", len(selected_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
