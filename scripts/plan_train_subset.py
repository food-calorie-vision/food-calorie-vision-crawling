#!/usr/bin/env python3
"""
Interactively select how many samples to exclude per class from train.txt.

Outputs an exclude manifest that can be passed to prepare_food_dataset.py
via --train-exclude and optionally rewrites train.txt without the selected samples.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively prune train.txt per class.")
    parser.add_argument("--run-id", required=True, help="Dataset run_id (used for default paths).")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Dataset root (default: data/5_datasets/<run_id>).",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        help="Train manifest path (default: <dataset-dir>/train.txt).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="Images directory (default: <dataset-dir>/images). Used to infer class names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclude manifest output (default: data/meta/<run_id>_exclude.txt).",
    )
    parser.add_argument(
        "--update-train",
        action="store_true",
        help="Rewrite train.txt after exclusions. 기본값은 train.txt를 수정하지 않고 exclude 리스트만 생성합니다.",
    )
    return parser.parse_args()


def load_train_entries(train_list: Path) -> List[str]:
    if not train_list.exists():
        raise FileNotFoundError(f"Train list not found: {train_list}")
    with train_list.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def relative_path(path_str: str, images_root: Path) -> Path:
    path = Path(path_str)
    try:
        return path.relative_to(images_root)
    except ValueError:
        return path


def load_class_names(dataset_dir: Path) -> List[str]:
    classes_txt = dataset_dir / "classes.txt"
    if not classes_txt.exists():
        return []
    with classes_txt.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def resolve_class_name(rel_path: Path, labels_dir: Path, class_names: List[str]) -> str:
    # rel_path can be 'f067da90__햄버거_0001.jpg'
    # We need to find '.../labels/f067da90__햄버거_0001.txt'
    
    # Get the path components without the image extension
    label_stem = rel_path.with_suffix('')
    
    label_path = (labels_dir / label_stem).with_suffix(".txt")
    
    class_id = -1
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as fp:
            first_line = fp.readline().strip()
            if first_line:
                try:
                    class_id = int(first_line.split()[0])
                except (ValueError, IndexError):
                    class_id = -1
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id if class_id >= 0 else 'unknown'}"


def prompt_global_exclusion(max_value: int) -> int:
    while True:
        raw = input(f"클래스별 제외할 이미지 수 입력 (0~{max_value}): ").strip()
        if not raw:
            return 0
        if not raw.isdigit():
            print("정수 값을 입력하세요.")
            continue
        value = int(raw)
        if value < 0 or value > max_value:
            print(f"0 이상 {max_value} 이하의 값을 입력하세요.")
            continue
        return value


def write_list(path: Path, entries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(f"{entry}\n")


def main() -> None:
    args = parse_args()
    dataset_dir = (args.dataset_dir or (Path("data") / "5_datasets" / args.run_id)).resolve()
    train_list = (args.train_list or (dataset_dir / "train.txt")).resolve()
    images_root = (args.images_root or (dataset_dir / "images")).resolve()
    labels_dir = dataset_dir / "labels"
    output_path = (args.output or (Path("data") / "meta" / f"{args.run_id}_exclude.txt")).resolve()

    print(f"[info] dataset_dir: {dataset_dir}")
    print(f"[info] train list : {train_list}")
    print(f"[info] images root: {images_root}")

    entries = load_train_entries(train_list)
    if not entries:
        raise RuntimeError("train.txt가 비어 있습니다.")

    class_names = load_class_names(dataset_dir)

    per_class: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        rel = relative_path(entry, images_root)
        class_name = resolve_class_name(rel, labels_dir, class_names)
        per_class[class_name].append(entry)

    if not per_class:
        print("[info] 분류된 클래스가 없습니다.")
        return

    print("[info] 클래스별 이미지 수:")
    for class_name in sorted(per_class):
        print(f"  - {class_name}: {len(per_class[class_name])}장")

    min_count = min(len(items) for items in per_class.values())
    to_remove = prompt_global_exclusion(min_count)
    if to_remove <= 0:
        print("[info] 제외할 항목이 없어 파일을 생성하지 않습니다.")
        return

    exclude_entries: List[str] = []
    updated_per_class: Dict[str, List[str]] = {}
    for class_name, entries_list in per_class.items():
        if len(entries_list) <= to_remove:
            drop = list(entries_list)
            keep: List[str] = []
        else:
            selected = set(random.sample(entries_list, to_remove))
            drop = [entry for entry in entries_list if entry in selected]
            keep = [entry for entry in entries_list if entry not in selected]
        exclude_entries.extend(drop)
        updated_per_class[class_name] = keep

    write_list(output_path, exclude_entries)
    print(f"[write] 제외 목록 -> {output_path} ({len(exclude_entries)}장)")

    if args.update_train:
        updated_train: List[str] = []
        for class_entries in updated_per_class.values():
            updated_train.extend(class_entries)
        write_list(train_list, updated_train)
        print(f"[write] train.txt 업데이트 -> {train_list} (총 {len(updated_train)}장)")
    else:
        print("[info] train.txt는 수정하지 않았습니다. exclude 목록만 생성되었습니다.")


if __name__ == "__main__":
    main()
