#!/usr/bin/env python3
"""
Post-process YOLO txt labels:
1) keep only the largest bounding box per file
2) replace class_id with the class specified by the parent folder using a CSV map
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the largest YOLO box per file and remap class IDs based on folder names."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Root directory of YOLO txt files.")
    parser.add_argument(
        "--label-map",
        type=Path,
        required=True,
        help="CSV file mapping numeric id,classes (e.g., food_class_pre_label.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination root. Defaults to in-place overwrite.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Error if a folder name is missing from the label map (default: warn and skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without writing files.",
    )
    return parser.parse_args()


def load_label_map(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} must have headers.")
        col_id = reader.fieldnames[0]
        col_name = reader.fieldnames[1] if len(reader.fieldnames) > 1 else None
        for row in reader:
            cls = (row.get(col_name) if col_name else None) or row.get("classes") or row.get("class") or ""
            cls = (cls or "").strip()
            if not cls:
                continue
            class_id = row.get(col_id, "").strip()
            if class_id == "":
                continue
            mapping[cls] = class_id
    if not mapping:
        raise RuntimeError(f"No class entries found in {csv_path}")
    return mapping


def read_boxes(label_file: Path) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    with label_file.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                _, x, y, w, h = parts
                boxes.append((float(x), float(y), float(w), float(h)))
            except ValueError:
                continue
    return boxes


def select_largest_box(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    labels_root = args.labels.resolve()
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    label_map = load_label_map(args.label_map.resolve())
    logging.info("Loaded %d classes from %s", len(label_map), args.label_map)

    output_root = args.output.resolve() if args.output else labels_root
    dry_run = args.dry_run
    processed = 0
    skipped = 0

    for label_file in sorted(labels_root.rglob("*.txt")):
        if not label_file.is_file():
            continue
        rel = label_file.relative_to(labels_root)
        class_name = rel.parts[0]
        class_id = label_map.get(class_name)
        if class_id is None:
            msg = f"Class '{class_name}' missing from label map."
            if args.strict:
                raise KeyError(msg)
            logging.warning("%s Skipping %s", msg, rel)
            skipped += 1
            continue

        boxes = read_boxes(label_file)
        largest = select_largest_box(boxes)
        if largest is None:
            logging.debug("No boxes found in %s", rel)
            skipped += 1
            continue

        dest_path = output_root / rel
        if not dry_run:
            ensure_dir(dest_path.parent)
            with dest_path.open("w", encoding="utf-8") as fp:
                fp.write(f"{class_id} {largest[0]:.6f} {largest[1]:.6f} {largest[2]:.6f} {largest[3]:.6f}\n")
        else:
            logging.info("[dry-run] Would write %s -> class_id=%s", rel, class_id)
        processed += 1

    logging.info("Processed %d label files (skipped %d). Output: %s", processed, skipped, output_root)


if __name__ == "__main__":
    main()
