#!/usr/bin/env python3
"""
Auto-label filtered images using a YOLO model (Ultralytics).

Outputs YOLO txt annotations mirroring the filtered images directory structure,
stores per-detection confidences in a CSV, and appends low-confidence samples to
labels/meta/review_queue.csv for manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from ultralytics import YOLO  # type: ignore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = Path("models") / "yolo11l.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label images with a YOLO model.")
    parser.add_argument("--images", type=Path, required=True, help="Path to filtered images (class subfolders supported).")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="YOLO weights path (default: models/yolo11l.pt).",
    )
    parser.add_argument("--out", type=Path, help="Root output dir for YOLO label txt files (default: labels/yolo/<run_id>).")
    parser.add_argument("--run-id", type=str, help="Run identifier (defaults to images directory name).")
    parser.add_argument("--confidence", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold.")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size when sending images to YOLO.")
    parser.add_argument("--device", type=str, default="auto", help="Device string for YOLO (e.g., cpu, 0, 0,1).")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.3,
        help="Images whose max detection confidence is below this are queued for manual review.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help="Maximum detections per image to keep (matches YOLO default).",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        help="Path to write per-detection CSV (default: labels/meta/<run_id>_predictions.csv).",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("labels") / "meta" / "review_queue.csv",
        help="Global review queue CSV path (appends rows when needed).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip writing outputs/logging only.")
    return parser.parse_args()


def iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunked(seq: Sequence[Path], size: int) -> Iterable[List[Path]]:
    if size <= 0:
        size = len(seq) or 1
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def write_predictions_csv(csv_path: Path, rows: List[List[str]]) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["run_id", "image_path", "label_path", "class_id", "class_name", "confidence"])
        writer.writerows(rows)


def append_review_queue(csv_path: Path, rows: List[List[str]]) -> None:
    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if not file_exists:
            writer.writerow(["run_id", "image_path", "reason", "max_conf", "total_detections"])
        writer.writerows(rows)


def save_label_file(label_path: Path, detections: List[str], dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(label_path.parent)
    with label_path.open("w", encoding="utf-8") as fp:
        for line in detections:
            fp.write(f"{line}\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    images_root = args.images.resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    run_id = args.run_id or images_root.name
    labels_root = (args.out or (Path("labels") / "yolo" / run_id)).resolve()
    predictions_csv = (args.predictions_csv or (Path("labels") / "meta" / f"{run_id}_predictions.csv")).resolve()
    review_csv = args.review_csv.resolve()

    logging.info("Run ID: %s", run_id)
    logging.info("Images: %s", images_root)
    logging.info("Labels output: %s", labels_root)
    logging.info("Predictions CSV: %s", predictions_csv)
    logging.info("Review queue CSV: %s", review_csv)

    image_paths = iter_images(images_root)
    if not image_paths:
        logging.warning("No image files found under %s", images_root)
        return

    if args.dry_run:
        logging.info("[DRY-RUN] Would run inference on %d images using %s", len(image_paths), args.weights)
        return

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}. "
            "Download yolo11l.pt (YOLOv11 Large) from Ultralytics and place it here, "
            "or pass --weights with a valid .pt file."
        )

    model = YOLO(str(weights_path))
    model_names = model.names if isinstance(model.names, dict) else {idx: name for idx, name in enumerate(model.names)}

    prediction_rows: List[List[str]] = []
    review_rows: List[List[str]] = []
    total_detections = 0

    for batch in chunked(image_paths, args.batch_size):
        results = model.predict(
            source=[str(p) for p in batch],
            conf=args.confidence,
            iou=args.iou,
            imgsz=args.img_size,
            max_det=args.max_detections,
            device=args.device,
            verbose=False,
        )
        for img_path, result in zip(batch, results):
            rel_path = img_path.relative_to(images_root)
            label_path = (labels_root / rel_path).with_suffix(".txt")
            detections: List[str] = []
            max_conf = 0.0
            if result.boxes is not None and result.boxes.xywhn is not None:
                xywhn = result.boxes.xywhn.cpu().tolist()
                clses = result.boxes.cls.cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                for (x, y, w, h), cls_id, conf in zip(xywhn, clses, confs):
                    max_conf = max(max_conf, float(conf))
                    class_id = int(cls_id)
                    class_name = model_names.get(class_id, f"class_{class_id}")
                    detections.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    prediction_rows.append(
                        [
                            run_id,
                            str(img_path),
                            str(label_path),
                            str(class_id),
                            class_name,
                            f"{conf:.4f}",
                        ]
                    )
                    total_detections += 1

            save_label_file(label_path, detections, dry_run=False)

            if not detections or max_conf < args.review_threshold:
                reason = "no_detections" if not detections else f"max_conf_below_{args.review_threshold}"
                review_rows.append(
                    [
                        run_id,
                        str(img_path),
                        reason,
                        f"{max_conf:.4f}",
                        str(len(detections)),
                    ]
                )

    write_predictions_csv(predictions_csv, prediction_rows)
    if review_rows:
        append_review_queue(review_csv, review_rows)
        logging.info("Queued %d images for manual review.", len(review_rows))
    logging.info("Generated %d detections over %d images.", total_detections, len(image_paths))


if __name__ == "__main__":
    main()
