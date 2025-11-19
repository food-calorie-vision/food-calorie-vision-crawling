#!/usr/bin/env python3
"""
Exact-duplicate filter for crawled images.

Keeps the first occurrence of every file hash and copies it into a filtered folder
while writing a stats.yaml report (per-class counts plus duplicate listings).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class FileRecord:
    rel_path: str
    class_name: str
    digest: str
    kept: bool
    duplicate_of: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove exact duplicate images by hash.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw run directory (class subfolders expected).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination folder for filtered images. Defaults to data/2_filtered/<run_id>.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Path to write stats.yaml (default: <output>/stats.yaml).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run identifier used for reporting (defaults to input directory name).",
    )
    parser.add_argument(
        "--hash-algo",
        type=str,
        default="sha256",
        choices=hashlib.algorithms_available,
        help="Hash algorithm for duplicate detection (default: sha256).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect duplicates and report stats without copying/removing any files.",
    )
    return parser.parse_args()


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def compute_digest(path: Path, algo: str) -> str:
    hasher = hashlib.new(algo)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_stats(records: List[FileRecord], run_id: str, input_dir: Path, output_dir: Path, hash_algo: str) -> Dict:
    per_class = defaultdict(lambda: {"input": 0, "kept": 0, "dropped_exact": 0})
    duplicates = []
    for rec in records:
        class_entry = per_class[rec.class_name]
        class_entry["input"] += 1
        if rec.kept:
            class_entry["kept"] += 1
        else:
            class_entry["dropped_exact"] += 1
            duplicates.append(
                {
                    "rel_path": rec.rel_path,
                    "duplicate_of": rec.duplicate_of,
                    "class": rec.class_name,
                    "digest": rec.digest,
                }
            )

    total_input = sum(entry["input"] for entry in per_class.values())
    total_kept = sum(entry["kept"] for entry in per_class.values())
    return {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hash_algorithm": hash_algo,
        "totals": {
            "input_files": total_input,
            "kept": total_kept,
            "dropped_exact": total_input - total_kept,
        },
        "classes": dict(per_class),
        "duplicates": duplicates,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = args.input.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    run_id = args.run_id or input_dir.name
    output_dir = (args.output or (input_dir.parents[1] / "2_filtered" / run_id)).resolve()
    stats_path = (args.stats or (output_dir / "stats.yaml")).resolve()

    logging.info("Run ID: %s", run_id)
    logging.info("Input directory: %s", input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Stats file: %s", stats_path)

    if not args.dry_run:
        ensure_dir(output_dir)

    digest_map: Dict[str, str] = {}
    records: List[FileRecord] = []

    for src in iter_image_files(input_dir):
        rel_path = src.relative_to(input_dir)
        class_name = rel_path.parts[0] if len(rel_path.parts) > 1 else input_dir.name
        digest = compute_digest(src, args.hash_algo)

        if digest not in digest_map:
            digest_map[digest] = str(rel_path)
            record = FileRecord(rel_path=str(rel_path), class_name=class_name, digest=digest, kept=True)
            records.append(record)
            if not args.dry_run:
                dest = output_dir / rel_path
                ensure_dir(dest.parent)
                shutil.copy2(src, dest)
        else:
            record = FileRecord(
                rel_path=str(rel_path),
                class_name=class_name,
                digest=digest,
                kept=False,
                duplicate_of=digest_map[digest],
            )
            records.append(record)
            logging.debug("Duplicate detected: %s -> %s", rel_path, digest_map[digest])

    stats = build_stats(records, run_id, input_dir, output_dir, args.hash_algo)
    ensure_dir(stats_path.parent)
    with stats_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(stats, fp, sort_keys=False, allow_unicode=True)

    kept = stats["totals"]["kept"]
    dropped = stats["totals"]["dropped_exact"]
    logging.info("Completed. Kept %d files, dropped %d duplicates.", kept, dropped)
    if args.dry_run:
        logging.info("Dry-run mode: no files were copied.")


if __name__ == "__main__":
    main()
