#!/usr/bin/env python3
"""
Summarize class-level performance metrics from auto-label predictions.

Reads labels/3-2_meta/<run_id>_predictions.csv and aggregates detection counts,
confidence statistics, and review queue pressure per class (folder).
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate class-level metrics from predictions CSV.")
    parser.add_argument("--run-id", required=True, help="Run identifier.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Per-detection CSV (default: labels/3-2_meta/<run_id>_predictions.csv).",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("labels") / "3-2_meta" / "review_queue.csv",
        help="Global review queue CSV path (default: labels/3-2_meta/review_queue.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path (default: data/meta/class_performance/<run_id>/<run_id>_class_performance_latest.csv "
            "when history is enabled, otherwise data/meta/<run_id>_class_performance.csv)."
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        help="Directory for timestamped summaries (default: data/meta/class_performance/<run_id>).",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable writing timestamped history files.",
    )
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold used to flag low-confidence detections.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional CSV(id,classes) to map folder names to canonical class IDs/names.",
    )
    return parser.parse_args()


def load_predictions(predictions_path: Path) -> List[Dict[str, str]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")
    with predictions_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def load_review_rows(review_path: Path) -> List[Dict[str, str]]:
    if not review_path or not review_path.exists():
        return []
    with review_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def load_class_map(path: Path | None) -> Dict[str, Tuple[int, str]]:
    if not path:
        return {}
    mapping: Dict[str, Tuple[int, str]] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip().lower()
            label = row[1].strip()
            if key in {"id", "class_id"}:
                continue
            try:
                class_id = int(float(key))
            except ValueError:
                continue
            mapping[label] = (class_id, label)
    return mapping


def summarize(
    run_id: str,
    predictions: List[Dict[str, str]],
    review_rows: List[Dict[str, str]],
    low_conf_threshold: float,
) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}

    def ensure_entry(folder: str) -> Dict[str, object]:
        if folder not in stats:
            stats[folder] = {
                "detections": 0,
                "conf_sum": 0.0,
                "conf_min": None,
                "conf_max": None,
                "low_conf": 0,
                "images_seen": set(),
                "images_with_detections": set(),
                "review_images": set(),
                "review_reasons": Counter(),
            }
        return stats[folder]

    for row in predictions:
        image_path = Path(row.get("image_path", "")).resolve()
        folder = image_path.parent.name or "unknown"
        entry = ensure_entry(folder)
        conf = float(row.get("confidence", 0.0))
        entry["detections"] += 1
        entry["conf_sum"] += conf
        entry["conf_min"] = conf if entry["conf_min"] is None else min(entry["conf_min"], conf)
        entry["conf_max"] = conf if entry["conf_max"] is None else max(entry["conf_max"], conf)
        if conf < low_conf_threshold:
            entry["low_conf"] += 1
        entry["images_seen"].add(str(image_path))
        entry["images_with_detections"].add(str(image_path))

    for row in review_rows:
        if row.get("run_id") != run_id:
            continue
        image_path = Path(row.get("image_path", "")).resolve()
        folder = image_path.parent.name or "unknown"
        entry = ensure_entry(folder)
        entry["images_seen"].add(str(image_path))
        entry["review_images"].add(str(image_path))
        reason = row.get("reason", "unknown")
        entry["review_reasons"][reason] += 1

    return stats


def write_output(
    output_path: Path,
    run_id: str,
    stats: Dict[str, Dict[str, object]],
    class_map: Dict[str, Tuple[int, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_id",
        "folder",
        "class_id",
        "class_name",
        "total_images",
        "images_with_detections",
        "images_without_detections",
        "detections",
        "avg_confidence",
        "min_confidence",
        "max_confidence",
        "low_conf_detections",
        "low_conf_ratio",
        "review_images",
        "top_review_reasons",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for folder in sorted(stats):
            entry = stats[folder]
            images_seen: set[str] = entry["images_seen"]
            images_with_det: set[str] = entry["images_with_detections"]
            review_images: set[str] = entry["review_images"]
            total_images = len(images_seen)
            detections = entry["detections"]
            avg_conf = entry["conf_sum"] / detections if detections else 0.0
            conf_min = entry["conf_min"]
            conf_max = entry["conf_max"]
            low_conf = entry["low_conf"]
            review_reasons: Counter = entry["review_reasons"]
            mapped = class_map.get(folder)
            class_id = mapped[0] if mapped else ""
            class_name = mapped[1] if mapped else folder
            reasons_summary = "; ".join(
                f"{reason}:{count}" for reason, count in review_reasons.most_common(3)
            )
            writer.writerow(
                [
                    run_id,
                    folder,
                    class_id,
                    class_name,
                    total_images,
                    len(images_with_det),
                    max(total_images - len(images_with_det), 0),
                    detections,
                    f"{avg_conf:.4f}" if detections else "",
                    f"{conf_min:.4f}" if conf_min is not None else "",
                    f"{conf_max:.4f}" if conf_max is not None else "",
                    low_conf,
                    f"{(low_conf / detections):.4f}" if detections else "",
                    len(review_images),
                    reasons_summary,
                ]
            )


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    predictions_path = args.predictions or (Path("labels") / "3-2_meta" / f"{run_id}_predictions.csv")
    history_dir = None
    if not args.no_history:
        history_dir = (args.history_dir or (Path("data") / "meta" / "class_performance" / run_id)).resolve()
    if args.output:
        output_path = args.output.resolve()
    else:
        if history_dir:
            output_path = (history_dir / f"{run_id}_class_performance_latest.csv").resolve()
        else:
            output_path = (Path("data") / "meta" / f"{run_id}_class_performance.csv").resolve()
    predictions = load_predictions(predictions_path)
    review_rows = load_review_rows(args.review_csv)
    stats = summarize(run_id, predictions, review_rows, args.low_conf_threshold)
    class_map = load_class_map(args.label_map)
    if history_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f"{run_id}_class_performance_{timestamp}.csv"
        write_output(history_path, run_id, stats, class_map)
        print(f"[write] History snapshot -> {history_path} ({len(stats)} classes)")
    write_output(output_path, run_id, stats, class_map)
    print(f"[write] Latest summary -> {output_path} ({len(stats)} classes)")


if __name__ == "__main__":
    main()
