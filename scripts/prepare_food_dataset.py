#!/usr/bin/env python3
"""
Convert validated YOLO labels into a food-only dataset directory.

Workflow:
1. Point --source at Label Studio export (must contain images/ and labels/).
2. Copy images into data/datasets/<run_id>/images.
3. Filter label txt files (drop COCO IDs, shift IDs) into .../labels.
4. Optionally create train/val split lists (--val-ratio) and write classes.txt + dataset.yaml.

Example:
  python scripts/prepare_food_dataset.py \
    --run-id crawl_test_b \
    --source labels/yolo_validated/crawl_test_b \
    --label-map food_class.csv \
    --drop-below 80 \
    --shift-offset 80 \
    --val-ratio 0.2 \
    --overwrite
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package validated or filtered labels into a food-only YOLO dataset.")
    parser.add_argument("--run-id", required=True, help="Run identifier (used for output folder naming).")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing validated export (images/ + labels/) or filtered labels (txt files).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Optional directory containing images when --source has only label txt files (e.g., labels/food_only).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets"),
        help="Parent directory where the packaged dataset will live (default: data/datasets).",
    )
    parser.add_argument("--drop-below", type=int, help="Drop any class IDs lower than this value.")
    parser.add_argument("--drop-above", type=int, help="Drop any class IDs higher than this value.")
    parser.add_argument(
        "--shift-offset",
        type=int,
        default=0,
        help="Subtract this offset from remaining IDs (default: 0; set 80 for COCO).",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional CSV/TSV mapping of class_id,label_value (used for names list).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output folder before writing.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Optional fraction of samples to reserve for validation (writes train/val .txt lists).",
    )
    return parser.parse_args()


def load_label_map(path: Path | None) -> Dict[int, str]:
    if path is None:
        return {}
    delimiter = "," if path.suffix.lower() != ".tsv" else "\t"
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            value = ""
            extra = ""
            if len(row) > 1:
                value = row[1].strip()
            if len(row) > 2:
                extra = row[2].strip()
            if key.lower() in {"id", "class_id"}:
                continue
            try:
                mapping[int(key)] = value or extra or key
            except ValueError:
                continue
    return mapping


def copy_images_for_labels(labels_root: Path, images_src: Path, images_dst: Path) -> List[Path]:
    copied: List[Path] = []
    seen: set[Path] = set()
    missing: List[Path] = []

    for label_file in sorted(labels_root.rglob("*.txt")):
        if not label_file.is_file():
            continue
        rel = label_file.relative_to(labels_root)
        rel_parent = rel.parent
        stem = label_file.stem

        image_path: Path | None = None
        for ext in IMAGE_EXTS:
            candidate = images_src / rel_parent / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            # fallback: search entire image tree for matching filename
            matches = list(images_src.rglob(f"{stem}.*"))
            for candidate in matches:
                if candidate.suffix.lower() in IMAGE_EXTS:
                    image_path = candidate
                    break
        if image_path is None:
            missing.append(rel_parent / stem)
            continue

        rel_image = image_path.relative_to(images_src)
        if rel_image in seen:
            continue
        seen.add(rel_image)

        dest_path = images_dst / rel_image
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_path)
        copied.append(rel_image)

    if missing:
        print(f"[warn] Missing images for {len(missing)} labels (first 3 shown): {missing[:3]}")
    return copied


def filter_lines(
    lines: Sequence[str],
    drop_below: int | None,
    drop_above: int | None,
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
        if drop_below is not None and class_id < drop_below:
            continue
        if drop_above is not None and class_id > drop_above:
            continue
        new_id = class_id - shift_offset
        if new_id < 0:
            continue
        parts[0] = str(new_id)
        filtered.append(" ".join(parts))
    return filtered


def process_labels(
    src: Path,
    dst: Path,
    drop_below: int | None,
    drop_above: int | None,
    shift_offset: int,
    label_map: Dict[int, str],
) -> Dict[int, str]:
    dst.mkdir(parents=True, exist_ok=True)
    id_to_name: Dict[int, str] = {}
    for label_file in sorted(src.rglob("*.txt")):
        if not label_file.is_file():
            continue
        rel = label_file.relative_to(src)
        with label_file.open("r", encoding="utf-8") as fp:
            lines = fp.readlines()
        filtered = filter_lines(lines, drop_below, drop_above, shift_offset)
        if not filtered:
            continue
        target_path = dst / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(filtered))
            fp.write("\n")

        for entry in filtered:
            cls_id = int(entry.split()[0])
            original_id = cls_id + shift_offset
            # Prefer explicitly provided mappings for the shifted (new) ID so that
            # label_map files referencing the post-drop IDs take precedence.
            name = label_map.get(cls_id)
            if name is None:
                name = label_map.get(original_id, f"class_{cls_id}")
            id_to_name[cls_id] = name
    return id_to_name


def reindex_class_ids(labels_dir: Path, id_to_name: Dict[int, str]) -> Dict[int, str]:
    sorted_ids = sorted(id_to_name)
    if sorted_ids == list(range(len(sorted_ids))):
        return id_to_name

    mapping = {old_id: new_idx for new_idx, old_id in enumerate(sorted_ids)}
    for label_file in labels_dir.rglob("*.txt"):
        if not label_file.is_file():
            continue
        with label_file.open("r", encoding="utf-8") as fp:
            lines = fp.readlines()
        rewritten = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            if cls not in mapping:
                continue
            parts[0] = str(mapping[cls])
            rewritten.append(" ".join(parts))
        if rewritten:
            with label_file.open("w", encoding="utf-8") as fp:
                fp.write("\n".join(rewritten))
                fp.write("\n")

    new_mapping = {mapping[old]: name for old, name in id_to_name.items()}
    return new_mapping


def write_classes_file(path: Path, id_to_name: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for cls_id in sorted(id_to_name):
            fp.write(f"{id_to_name[cls_id]}\n")


def write_dataset_yaml(path: Path, train_source: Path | str, val_source: Path | str, names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        train_value = train_source.as_posix() if isinstance(train_source, Path) else str(train_source)
        val_value = val_source.as_posix() if isinstance(val_source, Path) else str(val_source)
        fp.write(f"train: {train_value}\n")
        fp.write(f"val: {val_value}\n")
        fp.write(f"nc: {len(names)}\n")
        fp.write("names: [")
        fp.write(", ".join(f'"{name}"' for name in names))
        fp.write("]\n")


def create_split_lists(
    dataset_dir: Path,
    images_root: Path,
    rel_image_paths: Sequence[Path],
    val_ratio: float,
) -> Tuple[Path | None, Path | None]:
    if not rel_image_paths or val_ratio <= 0.0 or val_ratio >= 1.0:
        return None, None
    total = len(rel_image_paths)
    if total < 2:
        return None, None
    rel_sorted = sorted(rel_image_paths)
    val_count = max(1, int(round(total * val_ratio)))
    if val_count >= total:
        val_count = total - 1
    val_subset = rel_sorted[-val_count:]
    train_subset = rel_sorted[:-val_count]
    train_list = dataset_dir / "train.txt"
    val_list = dataset_dir / "val.txt"
    train_list.parent.mkdir(parents=True, exist_ok=True)

    def _write_list(path: Path, subset: Sequence[Path]) -> None:
        with path.open("w", encoding="utf-8") as fp:
            for rel in subset:
                fp.write(f"{(images_root / rel).resolve().as_posix()}\n")

    _write_list(train_list, train_subset)
    _write_list(val_list, val_subset)
    return train_list, val_list


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    images_src = source_root / "images"
    labels_src = source_root / "labels"
    source_is_structured = images_src.exists() and labels_src.exists()
    if not source_is_structured:
        labels_txt_root = source_root
        if not args.image_root:
            raise FileNotFoundError(
                "When --source has only label txt files, provide --image-root pointing to the matching images."
            )
        images_src = args.image_root.resolve()
        labels_src = source_root
    else:
        labels_txt_root = labels_src

    dataset_dir = (args.output_root / args.run_id).resolve()
    images_dst = dataset_dir / "images"
    labels_dst = dataset_dir / "labels"

    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[copy] Images (labels-linked) -> {images_dst}")
    copied_images = copy_images_for_labels(labels_txt_root, images_src, images_dst)
    print(f"[copy] Copied {len(copied_images)} images (skipped unlabeled files)")
    train_list, val_list = create_split_lists(dataset_dir, images_dst, copied_images, args.val_ratio)
    if train_list and val_list:
        print(f"[split] Wrote train/val lists -> {train_list}, {val_list}")

    label_map = load_label_map(args.label_map)
    print("[filter] Processing label txt files...")
    id_to_name = process_labels(
        labels_txt_root,
        labels_dst,
        drop_below=args.drop_below,
        drop_above=args.drop_above,
        shift_offset=args.shift_offset,
        label_map=label_map,
    )
    if not id_to_name:
        raise RuntimeError("No labels remained after filtering. Check drop/shift parameters.")
    id_to_name = reindex_class_ids(labels_dst, id_to_name)
    print(f"[filter] Remaining classes: {len(id_to_name)} (contiguous IDs enforced)")

    classes_path = dataset_dir / "classes.txt"
    write_classes_file(classes_path, id_to_name)
    print(f"[write] classes.txt -> {classes_path}")

    yaml_path = dataset_dir / f"{args.run_id}.yaml"
    names_list = [id_to_name[idx] for idx in sorted(id_to_name)]
    train_source = train_list or images_dst
    val_source = val_list or images_dst
    write_dataset_yaml(yaml_path, train_source, val_source, names_list)
    print(f"[write] dataset yaml -> {yaml_path}")

    print("[done] Food-only dataset ready.")


if __name__ == "__main__":
    main()
