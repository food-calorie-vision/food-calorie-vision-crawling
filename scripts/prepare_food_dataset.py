#!/usr/bin/env python3
"""
Convert validated YOLO labels into a food-only dataset directory.

Workflow:
1.  Point --source at Label Studio export (must contain images/ and labels/).
2.  Use mapping files to translate Label Studio class IDs to final categories.
    - --source-classes: classes.txt from the LS export (maps LS index to name).
    - --master-classes: food_class_pre_label.csv (maps name to a master ID).
    - --label-map: food_class_after_label.csv (maps master ID to the final category name).
3.  Copy (or hardlink) images into data/5_datasets/<run_id>/images.
4.  Create new label files in .../labels with the final category IDs.
5.  Optionally create train/val split lists, then write classes.txt + dataset.yaml.

Example:
  python scripts/prepare_food_dataset.py \
    --run-id crawl_test_b_remapped \
    --source labels/4_export_from_studio/crawl_test_b/source \
    --source-classes labels/4_export_from_studio/crawl_test_b/source/classes.txt \
    --master-classes food_class_pre_label.csv \
    --label-map food_class_after_label.csv \
    --image-root data/1_raw/crawl_test_b \
    --val-ratio 0.2 \
    --overwrite
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.parse import unquote

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package validated or filtered labels into a food-only YOLO dataset.")
    parser.add_argument('--run-id', required=True, help='Run identifier (used for output folder naming).')
    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help='Directory containing validated export (images/ + labels/).',
    )
    parser.add_argument(
        '--source-classes',
        type=Path,
        required=True,
        help='Path to the classes.txt file from the Label Studio export.',
    )
    parser.add_argument(
        '--master-classes',
        type=Path,
        required=True,
        help='Path to the master class mapping file (e.g., food_class_pre_label.csv).',
    )
    parser.add_argument(
        '--label-map',
        type=Path,
        required=True,
        help='Path to the final category mapping CSV (e.g., food_class_after_label.csv).',
    )
    parser.add_argument(
        '--image-root',
        type=Path,
        help='Optional directory containing images when --source has only label txt files.',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path("data/5_datasets"),
        help='Parent directory where the packaged dataset will live (default: data/5_datasets).',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove existing output folder before writing.',
    )
    parser.add_argument(
        '--copy-mode',
        choices=['copy', 'hardlink'],
        default='copy',
        help='copy(기본) 또는 hardlink 방식으로 images/ 파일을 구성합니다. 하드링크는 동일 파티션에서만 작동합니다.',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.0,
        help='Optional fraction of samples to reserve for validation (writes train/val .txt lists).',
    )
    parser.add_argument(
        '--val-list',
        type=Path,
        help='검증 세트를 고정하려면 이미지 경로 목록이 담긴 텍스트 파일을 지정합니다.',
    )
    parser.add_argument(
        '--train-include',
        type=Path,
        help='이 파일에 명시된 이미지가 train 목록에 반드시 포함되도록 강제합니다.',
    )
    parser.add_argument(
        '--train-exclude',
        type=Path,
        help='이 파일에 명시된 이미지는 train 목록에서 제외합니다.',
    )
    # Deprecated arguments, kept for compatibility but ignored
    parser.add_argument('--drop-below', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--drop-above', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--shift-offset', type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_label_map_by_id(path: Path) -> Dict[int, str]:
    delimiter = "," if path.suffix.lower() != ".tsv" else "\t"
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        header = next(reader, None)  # Skip header
        if header and header[0].lower() in {"", "id", "class_id"}:
            pass
        else:
            # No header, reset file pointer
            fp.seek(0)

        for row in reader:
            if not row or len(row) < 2:
                continue
            key, value = row[0].strip(), row[1].strip()
            if not key.isdigit():
                continue
            mapping[int(key)] = value
    return mapping


def load_master_classes(path: Path) -> Dict[str, int]:
    """Loads mapping from class name to master ID."""
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if header and header[0].lower() == "id":
             pass
        else:
            fp.seek(0)
        for row in reader:
            if not row or len(row) < 2:
                continue
            master_id, name = row[0].strip(), row[1].strip()
            if not master_id.isdigit():
                continue
            mapping[name] = int(master_id)
    return mapping

def load_source_classes(path: Path) -> Dict[int, str]:
    """Loads the classes.txt from Label Studio export."""
    with path.open("r", encoding="utf-8") as fp:
        return {i: line.strip() for i, line in enumerate(fp) if line.strip()}


def remap_and_process_labels(
    src_labels: Path,
    dst_labels: Path,
    ls_idx_to_name: Dict[int, str],
    name_to_master_id: Dict[str, int],
    master_id_to_category: Dict[int, str],
) -> Tuple[Dict[int, str], Dict[Path, int]]:
    """
    Remaps labels from LS index to final category index and writes new label files.
    """
    # 1. Get the set of all possible final category names
    all_final_categories = sorted(list(set(master_id_to_category.values())))

    # 2. Create the final mapping from category name to final YOLO ID
    category_to_final_id = {name: i for i, name in enumerate(all_final_categories)}
    final_id_to_category = {i: name for name, i in category_to_final_id.items()}

    image_class_map: Dict[Path, int] = {}
    dst_labels.mkdir(parents=True, exist_ok=True)

    for label_file in sorted(src_labels.rglob("*.txt")):
        if not label_file.is_file():
            continue

        with label_file.open("r", encoding="utf-8") as fp:
            lines = fp.readlines()

        remapped_lines = []
        assigned_class = -1
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                ls_idx = int(parts[0])
                ls_name = ls_idx_to_name.get(ls_idx)
                if ls_name is None:
                    print(f"[warn] Unknown LS class index '{ls_idx}' in {label_file}. Skipping line.")
                    continue

                master_id = name_to_master_id.get(ls_name)
                if master_id is None:
                    print(f"[warn] LS class name '{ls_name}' not found in master class list. Skipping line.")
                    continue

                category_name = master_id_to_category.get(master_id)
                if category_name is None:
                    print(f"[warn] Master ID '{master_id}' for '{ls_name}' not found in final category map. Skipping line.")
                    continue

                final_id = category_to_final_id.get(category_name)
                if final_id is None:
                    # This should not happen if logic is correct
                    print(f"[error] Category '{category_name}' not found in final ID map. This is a bug.")
                    continue

                parts[0] = str(final_id)
                remapped_lines.append(" ".join(parts))
                if assigned_class == -1:
                    assigned_class = final_id
            except (ValueError, IndexError):
                continue

        if remapped_lines:
            rel = label_file.relative_to(src_labels)
            target_path = dst_labels / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as fp:
                fp.write("\n".join(remapped_lines) + "\n")

            rel_key = rel.with_suffix("")
            if assigned_class != -1:
                image_class_map[rel_key] = assigned_class

    return final_id_to_category, image_class_map

def _candidate_name_parts(stem: str) -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append(Path(cleaned.replace("\\", "/")))

    add(stem)
    if "__" in stem:
        suffix = stem.split("__", 1)[1]
        decoded = unquote(suffix)
        add(decoded)
        cleaned = decoded.strip("/ ")
        if cleaned and cleaned != decoded:
            add(cleaned)
        if cleaned:
            leaf = cleaned.split("/")[-1]
            add(leaf)
    return candidates


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            print(f"[warn] Hardlink 실패({{exc}}). copy2로 대체합니다.")
    shutil.copy2(src, dst)


def copy_images_for_labels(
    labels_root: Path,
    images_src: Path,
    images_dst: Path,
    copy_mode: str = "copy",
) -> Tuple[List[Path], Dict[Path, Path]]:
    copied: List[Path] = []
    missing: List[str] = []
    source_lookup: Dict[Path, Path] = {}

    print(f"Searching for images in: {images_src.resolve()}")
    print(f"Reading labels from: {labels_root.resolve()}")

    all_image_files = list(images_src.rglob("*"))
    print(f"Found {len(all_image_files)} total files in image source.")

    for label_file in sorted(labels_root.rglob("*.txt")):
        if not label_file.is_file():
            continue
        rel = label_file.relative_to(labels_root)
        stem = label_file.stem

        image_path: Path | None = None
        candidate_names = _candidate_name_parts(stem)
        
        # More robust recursive search
        for candidate_path in candidate_names:
            base_name = candidate_path.name
            # Find any file in the source tree that matches the base name, regardless of extension
            for ext in IMAGE_EXTS:
                full_name = f"{base_name}{ext}"
                for img_file in all_image_files:
                    if img_file.name == full_name:
                        image_path = img_file
                        break
                if image_path:
                    break
            if image_path:
                break

        if image_path is None:
            decoded_hint = candidate_names[-1].name if candidate_names else stem
            missing.append(decoded_hint)
            continue

        # Use the label file's relative path for the destination to preserve structure
        dest_rel = rel.with_suffix(image_path.suffix.lower())
        dest_path = images_dst / dest_rel
        materialize_image(image_path, dest_path, copy_mode)
        copied.append(dest_rel)
        source_lookup[dest_rel] = image_path

    if missing:
        print(f"[warn] Missing images for {len(missing)} labels (first 3 shown): {missing[:3]}")
    return copied, source_lookup

def normalize_compare_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or cleaned.startswith("#"):
        return ""
    cleaned = cleaned.replace("\\", "/").replace("//", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.strip()

def load_manifest_entries(path: Path | None, remap_base: Path | None = None) -> List[str]:
    if not path or not path.exists():
        if path: print(f"[warn] Manifest 파일을 찾을 수 없어 건너뜀: {path}")
        return []
    entries: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            normalized = normalize_compare_value(raw)
            if not normalized:
                continue
            if remap_base:
                try:
                    abs_path = Path(normalized)
                    if abs_path.is_absolute():
                        normalized = abs_path.relative_to(remap_base.resolve()).as_posix()
                except ValueError:
                    pass
            entries.append(normalized)
    return entries

def build_candidate_keys(rel: Path, images_dst: Path, source_path: Path | None) -> set[str]:
    del images_dst, source_path
    values: set[str] = set()
    rel_posix = rel.as_posix()
    values.add(rel_posix)
    values.add(f"images/{rel_posix}")
    leaf = rel_posix.rsplit("/", 1)[-1]
    values.add(leaf)
    stem = leaf.rsplit(".", 1)[0] if "." in leaf else leaf
    values.add(stem)
    if "__" in leaf:
        suffix = leaf.split("__", 1)[1]
        values.add(suffix)
        if "." in suffix:
            values.add(suffix.rsplit(".", 1)[0])
    if "__" in stem:
        values.add(stem.split("__", 1)[1])
    return {entry for entry in values if entry}


def build_candidate_lookup(rel_candidates: Dict[Path, set[str]]) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = defaultdict(list)
    for rel, keys in rel_candidates.items():
        for key in keys:
            lookup[key].append(rel)
    return lookup


def match_manifest_entries(
    entries: Sequence[str],
    candidate_lookup: Dict[str, List[Path]],
    available: set[Path],
) -> Tuple[List[Path], List[str]]:
    matched, unmatched = [], []
    for entry in entries:
        normalized = normalize_compare_value(entry)
        if not normalized: continue
        
        choices = candidate_lookup.get(normalized, [])
        chosen: Path | None = next((rel for rel in choices if rel in available), None)
            
        if chosen:
            matched.append(chosen)
            available.remove(chosen)
        else:
            unmatched.append(entry)
    return matched, unmatched

def apply_manifest_filter(
    rel_paths: Sequence[Path],
    manifest_entries: Sequence[str],
    rel_candidates: Dict[Path, set[str]],
    include: bool,
) -> Tuple[List[Path], List[str]]:
    if not manifest_entries:
        return list(rel_paths), []
    manifest_set = {normalize_compare_value(e) for e in manifest_entries}
    matched_entries: set[str] = set()
    result: List[Path] = []
    for rel in rel_paths:
        keys = rel_candidates.get(rel, set())
        intersects = keys.intersection(manifest_set)
        if intersects:
            matched_entries.update(intersects)
        if include:
            if intersects:
                result.append(rel)
        else:
            if not intersects:
                result.append(rel)
    unmatched = [e for e in manifest_entries if normalize_compare_value(e) not in matched_entries]
    return result, unmatched

def write_list_file(path: Path, rel_paths: Sequence[Path], images_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rel in rel_paths:
            fp.write(f"{(images_root / rel).resolve().as_posix()}\n")

def write_classes_file(path: Path, id_to_name: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for i in range(len(id_to_name)):
            fp.write(f"{id_to_name[i]}\n")

def write_dataset_yaml(path: Path, train_source: Path | str, val_source: Path | str, names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        dataset_root = path.parent.resolve()
        fp.write(f"path: {dataset_root.as_posix()}\n")
        train_entry = train_source.resolve().as_posix() if isinstance(train_source, Path) else train_source
        val_entry = val_source.resolve().as_posix() if isinstance(val_source, Path) else val_source
        fp.write(f"train: {train_entry}\n")
        fp.write(f"val: {val_entry}\n")
        fp.write("test:\n\n")
        fp.write(f"nc: {len(names)}\n")
        fp.write(f"names: {names}\n")

def stratified_split(
    rel_image_paths: Sequence[Path],
    val_ratio: float,
    class_assignments: Dict[Path, int],
) -> Tuple[List[Path], List[Path]]:
    if not rel_image_paths or not (0 < val_ratio < 1):
        return list(rel_image_paths), []

    groups = defaultdict(list)
    for rel in sorted(rel_image_paths):
        groups[class_assignments.get(rel.with_suffix(""), -1)].append(rel)

    train_subset, val_subset = [], []
    for items in groups.values():
        if len(items) < 2:
            train_subset.extend(items)
            continue
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_subset.extend(items[-val_count:])
        train_subset.extend(items[:-val_count])
    
    return train_subset, val_subset

def main() -> None:
    args = parse_args()
    if args.val_list and args.run_id == args.val_list.parent.name:
        print(f"[error] 'run_id'({args.run_id}) cannot be the same as the base run_id from '--val-list'.")
        sys.exit(1)

    source_root = args.source.resolve()
    images_src = (source_root / "images") if not args.image_root else args.image_root.resolve()
    labels_src = source_root / "labels"
    if not (source_root.exists() and labels_src.exists()):
        raise FileNotFoundError(f"Source directory must contain a 'labels' subdirectory. Path not found: {labels_src}")

    try:
        first_label = next(labels_src.rglob("*.txt"))
    except StopIteration:
        first_label = None
    if first_label is None:
        raise RuntimeError(
            f"No YOLO label files were found under {labels_src}. "
            "Label Studio export may not contain any approved annotations yet. "
            "Complete validation (or re-export with --copy-to-annotations/--copy-predictions) before packaging."
        )

    dataset_dir = (args.output_root / args.run_id).resolve()
    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    images_dst = dataset_dir / "images"
    labels_dst = dataset_dir / "labels"

    # --- Remapping Logic ---
    print("[map] Loading class mapping files...")
    ls_idx_to_name = load_source_classes(args.source_classes)
    name_to_master_id = load_master_classes(args.master_classes)
    master_id_to_category = load_label_map_by_id(args.label_map)

    print("[map] Remapping and processing label files...")
    final_id_to_name, image_class_map = remap_and_process_labels(
        labels_src, labels_dst, ls_idx_to_name, name_to_master_id, master_id_to_category
    )
    if not final_id_to_name:
        raise RuntimeError("No labels remained after remapping. Check mapping files and source labels.")
    print(f"[map] Remapped to {len(final_id_to_name)} final categories.")
    
    print(f"[copy] Images (labels-linked) -> {images_dst}")
    copied_images, source_lookup = copy_images_for_labels(labels_dst, images_src, images_dst, args.copy_mode)
    print(f"[copy] Copied {len(copied_images)} images.")

    rel_candidates = {rel: build_candidate_keys(rel, images_dst, source_lookup.get(rel)) for rel in copied_images}
    candidate_lookup = build_candidate_lookup(rel_candidates)

    val_remap_base = args.val_list.resolve().parent if args.val_list else None
    train_relatives: List[Path] = list(copied_images)
    val_relatives: List[Path] = []
    
    val_entries = load_manifest_entries(args.val_list, remap_base=val_remap_base)
    if val_entries:
        val_relatives, unmatched = match_manifest_entries(val_entries, candidate_lookup, set(train_relatives))
        if unmatched: print(f"[warn] {len(unmatched)} validation entries not found: {unmatched[:3]}")
        if val_relatives: train_relatives = [r for r in train_relatives if r not in set(val_relatives)]
    elif args.val_ratio > 0:
        train_relatives, val_relatives = stratified_split(train_relatives, args.val_ratio, image_class_map)

    include_entries = load_manifest_entries(args.train_include, remap_base=val_remap_base)
    if include_entries:
        train_relatives, unmatched = apply_manifest_filter(train_relatives, include_entries, rel_candidates, include=True)
        if unmatched: print(f"[warn] {len(unmatched)} train include entries not found: {unmatched[:3]}")

    exclude_entries = load_manifest_entries(args.train_exclude, remap_base=val_remap_base)
    if exclude_entries:
        train_relatives, unmatched = apply_manifest_filter(train_relatives, exclude_entries, rel_candidates, include=False)
        if unmatched: print(f"[warn] {len(unmatched)} train exclude entries not found: {unmatched[:3]}")

    if not train_relatives:
        raise RuntimeError("No training images remained after applying filters.")

    train_list_path, val_list_path = None, None
    if val_relatives or include_entries or exclude_entries:
        train_list_path = dataset_dir / "train.txt"
        write_list_file(train_list_path, train_relatives, images_dst)
        print(f"[split] train.txt -> {train_list_path} ({len(train_relatives)} images)")
    if val_relatives:
        val_list_path = dataset_dir / "val.txt"
        write_list_file(val_list_path, val_relatives, images_dst)
        print(f"[split] val.txt -> {val_list_path} ({len(val_relatives)} images)")

    classes_path = dataset_dir / "classes.txt"
    write_classes_file(classes_path, final_id_to_name)
    print(f"[write] classes.txt -> {classes_path}")

    yaml_path = dataset_dir / f"{args.run_id}.yaml"
    names_list = [final_id_to_name[i] for i in sorted(final_id_to_name)]
    write_dataset_yaml(yaml_path, train_list_path or images_dst, val_list_path or images_dst, names_list)
    print(f"[write] dataset yaml -> {yaml_path}")

    print("[done] Food-only dataset ready.")


if __name__ == "__main__":
    main()
