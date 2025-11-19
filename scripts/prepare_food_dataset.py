#!/usr/bin/env python3
"""
Convert validated YOLO labels into a food-only dataset directory.

Workflow:
1. Point --source at Label Studio export (must contain images/ and labels/).
2. Copy (or hardlink with --copy-mode) images into data/5_datasets/<run_id>/images.
3. Filter label txt files (drop COCO IDs, shift IDs) into .../labels.
4. Optionally create train/val split lists (--val-ratio) or reuse manifests (--val-list/--train-include/--train-exclude),
   then write classes.txt + dataset.yaml.

Example:
  python scripts/prepare_food_dataset.py \
    --run-id crawl_test_b \
    --source labels/4_yolo_validated/crawl_test_b \
    --label-map food_class_after_label.csv \
    --drop-below 80 \
    --shift-offset 80 \
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
        default=Path("data/5_datasets"),
        help="Parent directory where the packaged dataset will live (default: data/5_datasets).",
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
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="copy(기본) 또는 hardlink 방식으로 images/ 파일을 구성합니다. 하드링크는 동일 파티션에서만 작동합니다.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Optional fraction of samples to reserve for validation (writes train/val .txt lists).",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        help="검증 세트를 고정하려면 이미지 경로 목록이 담긴 텍스트 파일을 지정합니다 (한 줄당 하나, 상대/절대 경로 모두 허용).",
    )
    parser.add_argument(
        "--train-include",
        type=Path,
        help="이 파일에 명시된 이미지(상대 경로/절대 경로)가 train 목록에 반드시 포함되도록 강제합니다.",
    )
    parser.add_argument(
        "--train-exclude",
        type=Path,
        help="이 파일에 명시된 이미지는 train 목록에서 제외합니다 (validation에는 영향 없음).",
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
            print(f"[warn] Hardlink 실패({exc}). copy2로 대체합니다.")
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

    for label_file in sorted(labels_root.rglob("*.txt")):
        if not label_file.is_file():
            continue
        rel = label_file.relative_to(labels_root)
        rel_parent = rel.parent
        stem = label_file.stem

        image_path: Path | None = None
        candidate_names = _candidate_name_parts(stem)
        for candidate_name in candidate_names:
            candidate_parent = rel_parent / candidate_name.parent
            base = candidate_name.name
            for ext in IMAGE_EXTS:
                candidate = images_src / candidate_parent / f"{base}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path:
                break
        if image_path is None:
            # fallback: search entire image tree for matching filename
            for candidate_name in candidate_names:
                base = candidate_name.name
                matches = list(images_src.rglob(f"{base}.*"))
                match = next((m for m in matches if m.suffix.lower() in IMAGE_EXTS), None)
                if match:
                    image_path = match
                    break
        if image_path is None:
            decoded_hint = candidate_names[-1].name if candidate_names else stem
            missing.append(decoded_hint)
            continue

        dest_rel = rel.parent / f"{rel.stem}{image_path.suffix.lower()}"
        dest_path = images_dst / dest_rel
        materialize_image(image_path, dest_path, copy_mode)
        copied.append(dest_rel)
        source_lookup[dest_rel] = image_path

    if missing:
        print(f"[warn] Missing images for {len(missing)} labels (first 3 shown): {missing[:3]}")
    return copied, source_lookup


def normalize_compare_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("#"):
        return ""
    cleaned = cleaned.replace("\\", "/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.strip()


def load_manifest_entries(path: Path | None, remap_base: Path | None = None) -> List[str]:
    if not path:
        return []
    if not path.exists():
        print(f"[warn] Manifest 파일을 찾을 수 없어 건너뜀: {path}")
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


def add_candidate_variant(values: set[str], candidate: str) -> None:
    normalized = normalize_compare_value(candidate)
    if not normalized:
        return
    values.add(normalized)
    # allow version without leading slash
    if normalized.startswith("/"):
        values.add(normalized.lstrip("/"))
    if "/" in normalized and normalized.startswith("images/"):
        values.add(normalized[len("images/") :])
    stem_candidate = normalized
    if "/" in normalized:
        stem_candidate = normalized.split("/")[-1]
    values.add(stem_candidate)
    if "." in normalized.rsplit("/", 1)[-1]:
        no_ext = normalized.rsplit(".", 1)[0]
        values.add(no_ext)


def build_candidate_keys(rel: Path, images_dst: Path, source_path: Path | None) -> set[str]:
    values: set[str] = set()
    rel_posix = rel.as_posix()
    add_candidate_variant(values, rel_posix)
    add_candidate_variant(values, f"images/{rel_posix}")
    dst_abs = (images_dst / rel).resolve().as_posix()
    add_candidate_variant(values, dst_abs)
    if source_path:
        add_candidate_variant(values, source_path.resolve().as_posix())
    return {entry for entry in values if entry}


def build_candidate_lookup(rel_candidates: Dict[Path, set[str]]) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = {}
    for rel, keys in rel_candidates.items():
        for key in keys:
            lookup.setdefault(key, []).append(rel)
    return lookup


def match_manifest_entries(
    entries: Sequence[str],
    candidate_lookup: Dict[str, List[Path]],
    available: set[Path],
) -> Tuple[List[Path], List[str]]:
    matched: List[Path] = []
    unmatched: List[str] = []
    for entry in entries:
        normalized = normalize_compare_value(entry)
        if not normalized:
            continue
        choices = candidate_lookup.get(normalized, [])
        chosen: Path | None = None
        for rel in choices:
            if rel in available:
                chosen = rel
                available.remove(rel)
                break
        if chosen:
            matched.append(chosen)
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
    manifest_set = {normalize_compare_value(entry) for entry in manifest_entries if normalize_compare_value(entry)}
    matched_entries: set[str] = set()
    result: List[Path] = []
    for rel in rel_paths:
        keys = rel_candidates.get(rel, set())
        intersects = keys & manifest_set
        if include:
            if intersects:
                result.append(rel)
                matched_entries.update(intersects)
        else:
            if intersects:
                matched_entries.update(intersects)
                continue
            result.append(rel)
    unmatched = [entry for entry in manifest_entries if normalize_compare_value(entry) not in matched_entries]
    return result, unmatched


def write_list_file(path: Path, rel_paths: Sequence[Path], images_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rel in rel_paths:
            abs_entry = (images_root / rel).resolve().as_posix()
            fp.write(f"{abs_entry}\n")


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
) -> Tuple[Dict[int, str], Dict[Path, int]]:
    dst.mkdir(parents=True, exist_ok=True)
    id_to_name: Dict[int, str] = {}
    image_class_map: Dict[Path, int] = {}
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

        rel_key = (rel.parent / rel.stem)
        try:
            assigned_class = int(filtered[0].split()[0])
            image_class_map[rel_key] = assigned_class
        except (IndexError, ValueError):
            pass

        for entry in filtered:
            cls_id = int(entry.split()[0])
            original_id = cls_id + shift_offset
            # Prefer explicitly provided mappings for the shifted (new) ID so that
            # label_map files referencing the post-drop IDs take precedence.
            name = label_map.get(cls_id)
            if name is None:
                name = label_map.get(original_id, f"class_{cls_id}")
            id_to_name[cls_id] = name
    return id_to_name, image_class_map


def reindex_class_ids(labels_dir: Path, id_to_name: Dict[int, str]) -> Tuple[Dict[int, str], Dict[int, int]]:
    sorted_ids = sorted(id_to_name)
    identity_map = {idx: idx for idx in sorted_ids}
    if sorted_ids == list(range(len(sorted_ids))):
        return id_to_name, identity_map

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
    return new_mapping, mapping


def write_classes_file(path: Path, id_to_name: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for cls_id in sorted(id_to_name):
            fp.write(f"{id_to_name[cls_id]}\n")


def write_dataset_yaml(path: Path, train_source: Path | str, val_source: Path | str, names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fp:
        dataset_root = path.parent.resolve()
        fp.write(f"path: {dataset_root}\n")

        train_entry = str(train_source.resolve()) if isinstance(train_source, Path) else train_source
        val_entry = str(val_source.resolve()) if isinstance(val_source, Path) else val_source

        fp.write(f"train: {train_entry}\n")
        fp.write(f"val: {val_entry}\n")
        fp.write(f"test: \n\n")

        # Classes
        fp.write(f"nc: {len(names)}\n")
        fp.write(f"names: {names}\n")


def stratified_split(
    rel_image_paths: Sequence[Path],
    val_ratio: float,
    class_assignments: Dict[Path, int],
) -> Tuple[List[Path], List[Path]]:
    if not rel_image_paths or val_ratio <= 0.0 or val_ratio >= 1.0:
        return list(rel_image_paths), []

    groups: Dict[int, List[Path]] = defaultdict(list)
    for rel in sorted(rel_image_paths):
        class_key = rel.with_suffix("")
        cls_id = class_assignments.get(class_key, -1)
        groups[cls_id].append(rel)

    train_subset: List[Path] = []
    val_subset: List[Path] = []
    for items in groups.values():
        if len(items) < 2:
            train_subset.extend(items)
            continue
        class_val_count = max(1, int(round(len(items) * val_ratio)))
        if class_val_count >= len(items):
            class_val_count = len(items) - 1
        if class_val_count <= 0:
            train_subset.extend(items)
            continue
        val_subset.extend(items[-class_val_count:])
        train_subset.extend(items[:-class_val_count])

    if not val_subset or not train_subset:
        total = len(rel_image_paths)
        if total < 2:
            return list(rel_image_paths), []
        rel_sorted = sorted(rel_image_paths)
        val_count = max(1, int(round(total * val_ratio)))
        if val_count >= total:
            val_count = total - 1
        val_subset = rel_sorted[-val_count:]
        train_subset = rel_sorted[:-val_count]
    return train_subset, val_subset

    def _write_list(path: Path, subset: Sequence[Path]) -> None:
        with path.open("w", encoding="utf-8") as fp:
            for rel in subset:
                fp.write(f"{(images_root / rel).resolve().as_posix()}\n")

    _write_list(train_list, train_subset)
    _write_list(val_list, val_subset)
    return train_list, val_list


def main() -> None:
    args = parse_args()

    if args.val_list and args.run_id:
        base_run_id = args.val_list.parent.name
        if base_run_id == args.run_id:
            print(f"[error] 'run_id'({args.run_id})와 'base_run_id'({base_run_id})가 동일합니다.")
            print("        '--val-list'를 사용하는 액티브 러닝 단계에서는 'run_id'를 이전과 다른 새 이름으로 지정해야 합니다.")
            print("        예: --run-id <이전_id>_r2")
            sys.exit(1)

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
        if args.image_root:
            images_src = args.image_root.resolve()

    dataset_dir = (args.output_root / args.run_id).resolve()
    images_dst = dataset_dir / "images"
    labels_dst = dataset_dir / "labels"

    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[copy] Images (labels-linked) -> {images_dst}")
    copied_images, source_lookup = copy_images_for_labels(
        labels_txt_root,
        images_src,
        images_dst,
        copy_mode=args.copy_mode,
    )
    print(f"[copy] Copied {len(copied_images)} images (skipped unlabeled files)")
    label_map = load_label_map(args.label_map)
    print("[filter] Processing label txt files...")
    id_to_name, image_class_map = process_labels(
        labels_txt_root,
        labels_dst,
        drop_below=args.drop_below,
        drop_above=args.drop_above,
        shift_offset=args.shift_offset,
        label_map=label_map,
    )
    if not id_to_name:
        raise RuntimeError("No labels remained after filtering. Check drop/shift parameters.")
    id_to_name, reindex_map = reindex_class_ids(labels_dst, id_to_name)
    image_class_map = {rel: reindex_map.get(cls, cls) for rel, cls in image_class_map.items()}
    print(f"[filter] Remaining classes: {len(id_to_name)} (contiguous IDs enforced)")

    rel_candidates = {
        rel: build_candidate_keys(rel, images_dst, source_lookup.get(rel)) for rel in copied_images
    }
    candidate_lookup = build_candidate_lookup(rel_candidates)

    val_remap_base = args.val_list.resolve().parent if args.val_list else None
    train_relatives: List[Path] = list(copied_images)
    val_relatives: List[Path] = []
    val_entries = load_manifest_entries(args.val_list, remap_base=val_remap_base)
    if val_entries:
        val_relatives, unmatched_val = match_manifest_entries(
            val_entries,
            candidate_lookup,
            set(train_relatives),
        )
        if unmatched_val:
            print(f"[warn] {len(unmatched_val)} validation entries not found: {unmatched_val[:3]}")
        if val_relatives:
            val_rel_set = set(val_relatives)
            train_relatives = [rel for rel in train_relatives if rel not in val_rel_set]
    elif args.val_ratio > 0:
        train_relatives, val_relatives = stratified_split(train_relatives, args.val_ratio, image_class_map)
        if not val_relatives:
            print("[warn] val_ratio 적용에 필요한 표본이 부족해 validation split을 생성하지 못했습니다.")

    include_entries = load_manifest_entries(args.train_include, remap_base=val_remap_base)
    train_relatives, unmatched_include = apply_manifest_filter(
        train_relatives,
        include_entries,
        rel_candidates,
        include=True,
    )
    if include_entries and unmatched_include:
        print(f"[warn] train include 목록 중 {len(unmatched_include)}개를 찾지 못했습니다: {unmatched_include[:3]}")
    exclude_entries = load_manifest_entries(args.train_exclude, remap_base=val_remap_base)
    train_relatives, unmatched_exclude = apply_manifest_filter(
        train_relatives,
        exclude_entries,
        rel_candidates,
        include=False,
    )
    if exclude_entries and unmatched_exclude:
        print(f"[warn] train exclude 목록 중 {len(unmatched_exclude)}개를 찾지 못했습니다: {unmatched_exclude[:3]}")

    if not train_relatives:
        raise RuntimeError("No training images remained after applying include/exclude filters.")

    train_list: Path | None = None
    val_list: Path | None = None
    force_train_manifest = bool(val_relatives or include_entries or exclude_entries or val_entries or args.val_ratio > 0)
    if force_train_manifest:
        train_list = dataset_dir / "train.txt"
        write_list_file(train_list, train_relatives, images_dst)
        print(f"[split] train.txt -> {train_list} ({len(train_relatives)} images)")
    if val_relatives:
        val_list = dataset_dir / "val.txt"
        write_list_file(val_list, val_relatives, images_dst)
        print(f"[split] val.txt -> {val_list} ({len(val_relatives)} images)")

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
