#!/usr/bin/env python3
"""
Convert YOLO txt detections to Label Studio JSON tasks.

Each selected image becomes an entry with `data.image` pointing to a local path
and `predictions` containing bounding boxes.
"""
from __future__ import annotations

import argparse
import json
import csv
import struct
import copy
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import quote

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Label Studio JSON from YOLO labels.")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier (matching filtered images).")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Root images directory (data/2_filtered/<run_id>).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="YOLO labels directory (labels/3-1_yolo_auto/<run_id>).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional file listing relative image paths to include. If omitted, all images are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: data/3_exports/<run_id>/labelstudio.json).",
    )
    parser.add_argument(
        "--document-root",
        type=str,
        help="Optional override for URL prefix served by Label Studio. If omitted, workspace-relative URLs are used.",
    )
    parser.add_argument(
        "--local-prefix",
        type=str,
        default="workspace",
        help="Subfolder mounted under /label-studio/data/local_storage (default: workspace).",
    )
    parser.add_argument(
        "--copy-to-annotations",
        action="store_true",
        help="Also emit annotations mirroring predictions so boxes appear as completed labels.",
    )
    parser.add_argument("--from-name", default="label", help="RectangleLabels name from the project config.")
    parser.add_argument("--to-name", default="image", help="Image tag name from the project config.")
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional CSV/TSV mapping of class_id,label_value (header row optional).",
    )
    return parser.parse_args()


def load_manifest(manifest: Path | None, images_root: Path) -> List[Path]:
    if manifest is None:
        return sorted([p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    selected = []
    with manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            rel = line.strip()
            if not rel:
                continue
            candidate = (images_root / rel).resolve()
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
                selected.append(candidate)
    return selected


def load_label_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter="," if path.suffix.lower() != ".tsv" else "\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if not key or key.lower() in {"id", "class_id"}:
                continue
            mapping[key] = value
    if not mapping:
        raise RuntimeError(f"No entries parsed from label map {path}")
    return mapping


def build_image_url(
    img_path: Path,
    rel_from_images: Path,
    document_root: str | None,
    local_prefix: str,
) -> str:
    if document_root:
        doc_root = document_root
        if "{path}" in doc_root:
            return doc_root.replace("{path}", rel_from_images.as_posix())
        if not doc_root.endswith(("/", "=")):
            doc_root = f"{doc_root}/"
        return f"{doc_root}{rel_from_images.as_posix()}"

    try:
        rel_project = img_path.relative_to(PROJECT_ROOT)
    except ValueError as err:
        raise RuntimeError(
            f"Image path {img_path} must reside inside project root {PROJECT_ROOT} when --document-root is omitted."
        ) from err
    url_path = f"{local_prefix}/{rel_project.as_posix()}".lstrip("/")
    encoded = quote(url_path, safe="/")
    return f"/data/local-files/?d={encoded}"


def _read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fp:
        header = fp.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")
    width = struct.unpack(">I", header[16:20])[0]
    height = struct.unpack(">I", header[20:24])[0]
    return width, height


def _read_jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fp:
        data = fp.read(2)
        if data != b"\xff\xd8":
            raise ValueError("Not a JPEG file")
        while True:
            marker_start = fp.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = fp.read(1)
            while marker == b"\xff":
                marker = fp.read(1)
            if marker in {
                b"\xc0",
                b"\xc1",
                b"\xc2",
                b"\xc3",
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                fp.read(3)
                height, width = struct.unpack(">HH", fp.read(4))
                return width, height
            else:
                length_bytes = fp.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                fp.seek(length - 2, 1)
    raise RuntimeError(f"Unable to determine JPEG size for {path}")


def read_image_size(path: Path) -> tuple[int, int]:
    if PILImage is not None:
        with PILImage.open(path) as img:
            return img.size
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _read_png_size(path)
    if suffix in {".jpg", ".jpeg"}:
        return _read_jpeg_size(path)
    raise RuntimeError(
        f"Cannot determine size for {path} without Pillow. Install Pillow or use PNG/JPEG files."
    )


def load_yolo_boxes(
    label_path: Path,
    img_w: int,
    img_h: int,
    from_name: str,
    to_name: str,
    label_map: Dict[str, str],
    box_prefix: str,
) -> List[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            cls_int = int(cls)
            label_value = label_map.get(str(cls_int), str(cls_int))
            x1 = (x - w / 2) * 100
            y1 = (y - h / 2) * 100
            width = w * 100
            height = h * 100
            boxes.append(
                {
                    "id": f"{box_prefix}_{len(boxes)}",
                    "type": "rectanglelabels",
                    "from_name": from_name,
                    "to_name": to_name,
                    "image_rotation": 0,
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": max(0.0, min(100.0, x1)),
                        "y": max(0.0, min(100.0, y1)),
                        "width": max(0.0, min(100.0, width)),
                        "height": max(0.0, min(100.0, height)),
                        "rotation": 0,
                        "rectanglelabels": [label_value],
                    },
                }
            )
    return boxes


def main() -> None:
    args = parse_args()
    images_root = args.images.resolve()
    labels_root = args.labels.resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    selected_images = load_manifest(args.manifest, images_root)
    if not selected_images:
        raise RuntimeError("No images found to export.")

    label_map = load_label_map(args.label_map) if args.label_map else {}

    tasks = []
    for idx, img_path in enumerate(selected_images):
        rel = img_path.relative_to(images_root)
        label_path = (labels_root / rel).with_suffix(".txt")
        data_image = build_image_url(img_path, rel, args.document_root, args.local_prefix)
        img_w, img_h = read_image_size(img_path)
        boxes = load_yolo_boxes(
            label_path,
            img_w,
            img_h,
            args.from_name,
            args.to_name,
            label_map,
            f"{rel.stem}_{idx}",
        )
        task_entry: dict = {
            "data": {"image": data_image},
            "predictions": [
                {
                    "model_version": "auto_label",
                    "result": boxes,
                }
            ]
            if boxes
            else [],
        }
        if args.copy_to_annotations and boxes:
            task_entry["annotations"] = [
                {
                    "result": copy.deepcopy(boxes),
                    "was_cancelled": False,
                    "ground_truth": False,
                }
            ]
        tasks.append(task_entry)

    default_output = Path("data/3_exports") / args.run_id / "labelstudio.json"
    output_path = (args.output or default_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(tasks, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(tasks)} tasks to {output_path}")


if __name__ == "__main__":
    main()
