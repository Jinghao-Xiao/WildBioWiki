#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
SIZE_TOKEN_PATTERN = re.compile(r"/(square|small|medium|large|original)\.([A-Za-z0-9]+)(\?.*)?$")
MEDIUM_LONG_EDGE = 500

_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a WildBioWiki-QA-medium derivative dataset using iNaturalist medium images."
    )
    parser.add_argument(
        "--input-root",
        default="data/WildBioWiki-QA",
        help="Existing finalized WildBioWiki-QA dataset root.",
    )
    parser.add_argument(
        "--metadata-root",
        default="data/iNaturalist/inaturalist_images_396",
        help="Root containing per-species metadata.jsonl files from the iNaturalist download stage.",
    )
    parser.add_argument(
        "--output-root",
        default="data/WildBioWiki-QA-medium",
        help="Output root for the medium-sized derivative dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Concurrent image download workers per species annotation file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for downloading each medium image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before rebuilding.",
    )
    parser.add_argument(
        "--max-images-per-species",
        type=int,
        default=0,
        help="Optional smoke-test cap. Zero means process all images.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def derive_medium_url(image_url: str) -> str:
    if "/medium." in image_url:
        return image_url
    return SIZE_TOKEN_PATTERN.sub(r"/medium.\2\3", image_url)


def build_medium_url_candidates(image_url: str) -> List[str]:
    candidates = []
    medium_url = derive_medium_url(image_url)
    candidates.append(medium_url)
    if "static.inaturalist.org" in medium_url:
        candidates.append(medium_url.replace("https://static.inaturalist.org", "https://inaturalist-open-data.s3.amazonaws.com"))
    elif "inaturalist-open-data.s3.amazonaws.com" in medium_url:
        candidates.append(medium_url.replace("https://inaturalist-open-data.s3.amazonaws.com", "https://static.inaturalist.org"))

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def scale_bbox(bbox_xyxy: List[float], old_size: Tuple[int, int], new_size: Tuple[int, int]) -> List[float]:
    old_width, old_height = old_size
    new_width, new_height = new_size
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    x1, y1, x2, y2 = bbox_xyxy
    scaled = [
        round(clamp(x1 * scale_x, 0.0, float(new_width)), 2),
        round(clamp(y1 * scale_y, 0.0, float(new_height)), 2),
        round(clamp(x2 * scale_x, 0.0, float(new_width)), 2),
        round(clamp(y2 * scale_y, 0.0, float(new_height)), 2),
    ]
    if scaled[2] <= scaled[0]:
        scaled[2] = min(float(new_width), round(scaled[0] + 1.0, 2))
    if scaled[3] <= scaled[1]:
        scaled[3] = min(float(new_height), round(scaled[1] + 1.0, 2))
    return scaled


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "wildbiowiki-medium-builder/1.0"})
        _thread_local.session = session
    return session


def download_file(urls: List[str], output_path: Path, timeout: int, max_retries: int) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    last_error: Optional[Exception] = None
    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                session = get_session()
                with session.get(url, timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                tmp_path.replace(output_path)
                return url
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Failed to download {urls[0]} -> {output_path}: {last_error}") from last_error


def resize_source_to_medium(source_image_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with Image.open(source_image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > MEDIUM_LONG_EDGE:
            scale = MEDIUM_LONG_EDGE / float(long_edge)
            resized = image.resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.Resampling.LANCZOS,
            )
        else:
            resized = image.copy()
        resized.save(tmp_path, format="JPEG", quality=90)
    tmp_path.replace(output_path)


def ensure_medium_image(
    source_image_path: Path,
    source_image_url: str,
    output_path: Path,
    timeout: int,
    max_retries: int,
) -> Tuple[Optional[str], str]:
    if output_path.exists():
        return None, "existing_output"

    medium_candidates = build_medium_url_candidates(source_image_url)
    try:
        used_url = download_file(medium_candidates, output_path, timeout=timeout, max_retries=max_retries)
        if used_url == medium_candidates[0]:
            return used_url, "inat_medium"
        return used_url, "inat_medium_host_fallback"
    except Exception:
        resize_source_to_medium(source_image_path, output_path)
        return None, "local_resize_fallback"


def build_metadata_index(metadata_root: Path) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    metadata_paths = sorted(metadata_root.rglob("metadata.jsonl"))
    for metadata_path in metadata_paths:
        for row in load_jsonl(metadata_path):
            image_id = str(row["image_id"])
            if image_id in index:
                continue
            medium_url = derive_medium_url(row["image_url"])
            index[image_id] = {
                "image_url": row["image_url"],
                "medium_url": medium_url,
                "local_path": row["local_path"],
                "scientific_name": row["scientific_name"],
                "taxonomy_class": Path(row["local_path"]).parts[3],
            }
    return index


def validate_existing_record(row: dict, output_image_path: Path) -> bool:
    required_fields = {"image_width", "image_height", "source_image_width", "source_image_height", "image_variant"}
    if not required_fields.issubset(row.keys()):
        return False
    return output_image_path.exists()


def convert_one_image(
    row: dict,
    source_image_path: Path,
    output_image_path: Path,
    metadata_index: Dict[str, dict],
    timeout: int,
    max_retries: int,
) -> dict:
    image_id = str(row["image_id"])
    if image_id not in metadata_index:
        raise KeyError(f"Missing metadata for image_id={image_id}")

    metadata = metadata_index[image_id]
    old_size = get_image_size(source_image_path)

    used_url, variant_source = ensure_medium_image(
        source_image_path=source_image_path,
        source_image_url=metadata["image_url"],
        output_path=output_image_path,
        timeout=timeout,
        max_retries=max_retries,
    )
    new_size = get_image_size(output_image_path)

    converted = dict(row)
    converted["bbox_xyxy"] = scale_bbox(row["bbox_xyxy"], old_size, new_size)
    converted["image_width"] = new_size[0]
    converted["image_height"] = new_size[1]
    converted["source_image_width"] = old_size[0]
    converted["source_image_height"] = old_size[1]
    converted["image_variant"] = "medium"
    converted["medium_image_url"] = used_url
    converted["image_variant_source"] = variant_source
    return converted


def copy_static_files(input_root: Path, output_root: Path) -> None:
    for rel_path in [
        Path("qa"),
        Path("class_id_index.jsonl"),
        Path("pruning_report.json"),
    ]:
        src = input_root / rel_path
        dst = output_root / rel_path
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def process_annotation_file(
    input_root: Path,
    output_root: Path,
    annotation_path: Path,
    metadata_index: Dict[str, dict],
    workers: int,
    timeout: int,
    max_retries: int,
    max_images_per_species: int,
) -> Tuple[List[dict], Dict[str, int]]:
    rows = load_jsonl(annotation_path)
    if max_images_per_species > 0:
        rows = rows[:max_images_per_species]

    rel_annotation_path = annotation_path.relative_to(input_root)
    output_annotation_path = output_root / rel_annotation_path

    existing_rows = {}
    if output_annotation_path.exists():
        for existing in load_jsonl(output_annotation_path):
            existing_rows[str(existing["image_id"])] = existing

    ordered_output_rows: List[Optional[dict]] = [None] * len(rows)
    futures = {}
    reused = 0
    downloaded = 0
    failures = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for idx, row in enumerate(rows):
            image_id = str(row["image_id"])
            output_image_path = output_root / row["image_path"]
            existing = existing_rows.get(image_id)
            if existing and validate_existing_record(existing, output_image_path):
                ordered_output_rows[idx] = existing
                reused += 1
                continue

            source_image_path = input_root / row["image_path"]
            if not source_image_path.exists():
                raise FileNotFoundError(f"Missing source image: {source_image_path}")

            future = executor.submit(
                convert_one_image,
                row,
                source_image_path,
                output_image_path,
                metadata_index,
                timeout,
                max_retries,
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                converted = future.result()
                ordered_output_rows[idx] = converted
                downloaded += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "annotation_path": str(rel_annotation_path),
                        "row": rows[idx],
                        "error": str(exc),
                    }
                )
                print(
                    f"FAILED {rel_annotation_path} image_id={rows[idx]['image_id']}: {exc}",
                    flush=True,
                )

    output_rows = [row for row in ordered_output_rows if row is not None]
    write_jsonl(output_annotation_path, output_rows)
    if failures:
        append_jsonl(output_root / "conversion_failures.jsonl", failures)

    return output_rows, {"reused": reused, "downloaded": downloaded, "failures": len(failures)}


def compute_split_summary(rows: List[dict], split: str) -> dict:
    return {
        "split": split,
        "image_count": len(rows),
        "class_count": len({row["taxonomy_class"] for row in rows}),
        "species_count": len({(row["taxonomy_class"], row["scientific_name"]) for row in rows}),
    }


def build_dataset_summary(output_root: Path, class_summaries: List[dict], split_rows: Dict[str, List[dict]]) -> dict:
    total_images = sum(len(rows) for rows in split_rows.values())
    summary = {
        "dataset_name": "WildBioWiki-QA-medium",
        "source_dataset": "WildBioWiki-QA",
        "image_spec": {
            "provider": "iNaturalist",
            "variant": "medium",
        },
        "total_images": total_images,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "class_count": len(class_summaries),
        "species_count": len(
            {
                (species["taxonomy_class"], species["scientific_name"])
                for class_summary in class_summaries
                for species in class_summary["species"]
            }
        ),
        "class_summaries": class_summaries,
    }

    class_id_index_path = output_root / "class_id_index.jsonl"
    if class_id_index_path.exists():
        summary["class_id_index"] = load_jsonl(class_id_index_path)
    pruning_report_path = output_root / "pruning_report.json"
    if pruning_report_path.exists():
        summary["pruning"] = load_json(pruning_report_path)
    return summary


def main() -> int:
    args = parse_args()
    input_root = (PROJECT_ROOT / args.input_root).resolve()
    metadata_root = (PROJECT_ROOT / args.metadata_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_index = build_metadata_index(metadata_root)
    copy_static_files(input_root, output_root)

    split_rows: Dict[str, List[dict]] = {split: [] for split in SPLITS}
    class_summary_map: Dict[str, dict] = {}

    annotation_paths = sorted((input_root / "annotations").rglob("*.jsonl"))
    for annotation_path in annotation_paths:
        rel_parts = annotation_path.relative_to(input_root / "annotations").parts
        split, taxonomy_class, filename = rel_parts[0], rel_parts[1], rel_parts[2]
        species_slug = Path(filename).stem

        class_summary = class_summary_map.setdefault(
            taxonomy_class,
            {
                "taxonomy_class": taxonomy_class,
                "species": {},
                "splits": {name: {"images": 0, "species": 0} for name in SPLITS},
            },
        )

        output_rows, counters = process_annotation_file(
            input_root=input_root,
            output_root=output_root,
            annotation_path=annotation_path,
            metadata_index=metadata_index,
            workers=args.workers,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_images_per_species=args.max_images_per_species,
        )

        if not output_rows:
            continue

        print(
            f"[{split}] {taxonomy_class}/{species_slug}: "
            f"{len(output_rows)} images, reused={counters['reused']}, downloaded={counters['downloaded']}, failures={counters['failures']}",
            flush=True,
        )

        split_rows[split].extend(output_rows)

        species_summary = class_summary["species"].setdefault(
            species_slug,
            {
                "species_slug": species_slug,
                "taxonomy_class": taxonomy_class,
                "scientific_name": output_rows[0]["scientific_name"],
                "common_name": output_rows[0]["common_name"],
                "taxon_id": output_rows[0]["taxon_id"],
                "class_id": output_rows[0]["class_id"],
                "splits": {name: 0 for name in SPLITS},
                "total_images": 0,
            },
        )
        species_summary["splits"][split] = len(output_rows)
        species_summary["total_images"] = sum(species_summary["splits"].values())

    manifests_dir = output_root / "manifests"
    summaries_dir = output_root / "summaries"
    for split in SPLITS:
        manifest_rows = split_rows[split]
        write_jsonl(manifests_dir / f"{split}.jsonl", manifest_rows)
        write_json(manifests_dir / f"{split}.summary.json", compute_split_summary(manifest_rows, split))

    class_summaries = []
    for taxonomy_class in sorted(class_summary_map.keys()):
        class_summary = class_summary_map[taxonomy_class]
        species_list = sorted(class_summary["species"].values(), key=lambda row: row["species_slug"])
        class_summary["species"] = species_list
        class_summary["species_count"] = len(species_list)
        for split in SPLITS:
            image_count = sum(species["splits"][split] for species in species_list)
            species_count = sum(1 for species in species_list if species["splits"][split] > 0)
            class_summary["splits"][split]["images"] = image_count
            class_summary["splits"][split]["species"] = species_count
        write_json(summaries_dir / f"{taxonomy_class}.summary.json", class_summary)
        class_summaries.append(class_summary)

    dataset_summary = build_dataset_summary(output_root, class_summaries, split_rows)
    write_json(output_root / "dataset_summary.json", dataset_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
