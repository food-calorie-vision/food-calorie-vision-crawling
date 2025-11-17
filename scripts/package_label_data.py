#!/usr/bin/env python3
"""
Prepare an export folder (no zip) for Label Studio/CVAT import.

The script copies selected images into `data/exports/<run_id>/images`,
and writes `manifest.txt` listing the relative paths. Move the folder
to the target storage and point Label Studio Local Files storage there.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare folder exports for labeling tools (no compression).")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier (used for folder naming).")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Root directory containing images to label (e.g., data/filtered/<run_id>).",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Optional text file with relative paths to include (one per line).",
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Include only one image per top-level class folder (default: all files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: data/exports/<run_id>).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove output directory before copying.")
    parser.add_argument("--dry-run", action="store_true", help="List files without copying.")
    return parser.parse_args()


def load_manifest(manifest: Path) -> list[Path]:
    with manifest.open("r", encoding="utf-8") as fp:
        return [Path(line.strip()) for line in fp if line.strip()]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def collect_files(images_root: Path, manifest: Path | None, per_class: bool) -> list[Path]:
    if manifest:
        entries = []
        for rel in load_manifest(manifest):
            candidate = (images_root / rel).resolve()
            if candidate.exists() and candidate.is_file() and is_image_file(candidate):
                entries.append(candidate)
        return entries

    if per_class:
        selected = []
        for class_dir in sorted([p for p in images_root.iterdir() if p.is_dir()]):
            files = sorted([f for f in class_dir.iterdir() if f.is_file() and is_image_file(f)])
            if files:
                selected.append(files[0])
        return selected

    return sorted([p for p in images_root.rglob("*") if p.is_file() and is_image_file(p)])


def main() -> None:
    args = parse_args()
    images_root = args.images.resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    manifest_path = args.image_list.resolve() if args.image_list else None
    files = collect_files(images_root, manifest_path, args.per_class)
    if not files:
        raise RuntimeError("No files found to package.")

    output_dir = (args.output_dir or Path("data/exports") / args.run_id).resolve()
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory already exists: {output_dir}. Use --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.txt"
    images_dest = output_dir / "images"

    if args.dry_run:
        print(f"[DRY-RUN] Would copy {len(files)} files into {images_dest}")
        return

    copied = 0
    for file_path in files:
        rel = file_path.relative_to(images_root)
        dest_path = images_dest / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        copied += 1

    with manifest_path.open("w", encoding="utf-8") as fp:
        for file_path in files:
            rel = file_path.relative_to(images_root)
            fp.write(f"{rel.as_posix()}\n")

    print(f"Copied {copied} files to {images_dest}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
