#!/usr/bin/env python3
"""
Filter and remap YOLO label IDs in-place (or to a separate directory).

Usage examples:
  # Keep YOLO ID 45 and remap it to 0
  python scripts/filter_labels.py \
    --labels labels/3-1_yolo_auto/crawl_test_b \
    --keep-ids 45 \
    --remap-id 0

  # Drop COCO IDs 0-79, shift everything else down by 80 (80->0, 81->1, …)
  python scripts/filter_labels.py \
    --labels labels/4_yolo_validated/crawl_test_b/labels \
    --drop-below 80 \
    --shift-offset 80 \
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter/Remap YOLO txt label files.")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Root directory containing YOLO txt files (mirrors image tree).",
    )
    parser.add_argument(
        "--keep-ids",
        type=int,
        nargs="+",
        help="Optional whitelist of YOLO class IDs to keep. All others are dropped.",
    )
    parser.add_argument("--drop-below", type=int, help="Drop any class IDs lower than this value.")
    parser.add_argument("--drop-above", type=int, help="Drop any class IDs higher than this value.")
    parser.add_argument(
        "--remap-id",
        type=int,
        help="Optional: remap every kept ID to this single ID (e.g., 0 for '국밥').",
    )
    parser.add_argument(
        "--shift-offset",
        type=int,
        default=0,
        help="Subtract this offset from each remaining class ID (e.g., 80 -> 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional directory to write filtered labels. Defaults to in-place edit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    return parser.parse_args()


def iter_label_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.txt"):
        if path.is_file():
            yield path


def validate_args(args: argparse.Namespace) -> None:
    if args.remap_id is not None and args.shift_offset:
        raise ValueError("Use either --remap-id or --shift-offset, not both.")


def filter_lines(
    lines: Sequence[str],
    keep_ids: set[int] | None,
    drop_below: int | None,
    drop_above: int | None,
    remap_id: int | None,
    shift_offset: int,
) -> List[str]:
    filtered: List[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if keep_ids is not None and class_id not in keep_ids:
            continue
        if drop_below is not None and class_id < drop_below:
            continue
        if drop_above is not None and class_id > drop_above:
            continue
        new_id = class_id
        if remap_id is not None:
            new_id = remap_id
        elif shift_offset:
            new_id = class_id - shift_offset
            if new_id < 0:
                continue
        parts[0] = str(new_id)
        filtered.append(" ".join(parts))
    return filtered


def main() -> None:
    args = parse_args()
    validate_args(args)
    labels_root = args.labels.resolve()
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    output_root = (args.output or labels_root).resolve()
    replace_in_place = output_root == labels_root
    keep_ids = set(args.keep_ids) if args.keep_ids else None

    changed_files = 0
    removed_lines = 0
    total_files = 0

    for label_file in iter_label_files(labels_root):
        total_files += 1
        with label_file.open("r", encoding="utf-8") as fp:
            lines = fp.readlines()
        filtered = filter_lines(
            lines,
            keep_ids,
            args.drop_below,
            args.drop_above,
            args.remap_id,
            args.shift_offset,
        )
        removed = max(len(lines) - len(filtered), 0)

        if args.dry_run:
            if removed or (replace_in_place and len(filtered) != len(lines)):
                print(f"[DRY-RUN] {label_file}: kept {len(filtered)} lines, removed {removed}")
            continue

        if replace_in_place:
            target_path = label_file
        else:
            target_path = output_root / label_file.relative_to(labels_root)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(filtered))
            if filtered:
                fp.write("\n")

        changed_files += 1
        removed_lines += removed

    print(
        f"Processed {total_files} files. Updated {changed_files} files, "
        f"removed {removed_lines} detections "
        f"(keep_ids={sorted(keep_ids) if keep_ids else 'ALL'}, "
        f"drop_below={args.drop_below}, drop_above={args.drop_above}, "
        f"shift_offset={args.shift_offset})."
    )


if __name__ == "__main__":
    main()
