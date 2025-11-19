#!/usr/bin/env python3
"""
Summarize training metrics exported by train_yolo.py.

Example:
  python scripts/report_training_metrics.py --run-id crawl_test_b --top 5 --conf-top 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-class metrics and confusion matrix CSVs.")
    parser.add_argument("--run-id", required=True, help="Run identifier used when training (e.g., crawl_test_b).")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("data/meta/train_metrics"),
        help="Directory containing *_per_class.csv and *_confusion_matrix.csv files.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["precision", "recall", "ap50", "ap50_95"],
        default="ap50_95",
        help="Metric column used for sorting per-class entries (default: ap50_95).",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of top classes to display (default: 5).")
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending (useful for finding weakest classes).",
    )
    parser.add_argument(
        "--include-confusion",
        action="store_true",
        help="Also summarize largest off-diagonal confusion matrix entries.",
    )
    parser.add_argument(
        "--conf-top",
        type=int,
        default=5,
        help="How many confusion pairs to display when --include-confusion is set (default: 5).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save the sorted per-class metrics as CSV.",
    )
    return parser.parse_args()


def load_per_class_metrics(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Per-class CSV not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows: List[Dict[str, float]] = []
        for row in reader:
            try:
                rows.append(
                    {
                        "class_id": int(row["class_id"]),
                        "class_name": row["class_name"],
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "ap50": float(row["ap50"]),
                        "ap50_95": float(row["ap50_95"]),
                    }
                )
            except KeyError as exc:
                raise KeyError(f"Missing column in per-class CSV: {exc}") from exc
    return rows


def load_confusion_matrix(path: Path) -> Tuple[List[str], List[List[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Confusion matrix CSV not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if not header:
            raise RuntimeError("Confusion matrix CSV is empty.")
        class_names = header[1:]
        rows: List[List[float]] = []
        for row in reader:
            values = [float(value) for value in row[1:]]
            rows.append(values)
    return class_names, rows


def summarize_confusion(
    class_names: Sequence[str],
    matrix: Sequence[Sequence[float]],
    top_n: int,
) -> List[Tuple[str, str, float]]:
    entries: List[Tuple[str, str, float]] = []
    for true_idx, row in enumerate(matrix):
        true_name = class_names[true_idx] if true_idx < len(class_names) else f"class_{true_idx}"
        for pred_idx, value in enumerate(row):
            if true_idx == pred_idx or value <= 0:
                continue
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
            entries.append((true_name, pred_name, value))
    entries.sort(key=lambda item: item[2], reverse=True)
    return entries[:top_n]


def print_per_class_table(rows: Sequence[Dict[str, float]], metric: str, top: int) -> None:
    print(f"\n[Per-class metrics sorted by {metric}]")
    header = f"{'Rank':>4}  {'Class':<20}  {'Precision':>9}  {'Recall':>7}  {'AP50':>7}  {'AP50-95':>8}"
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(rows[:top], start=1):
        print(
            f"{idx:>4}  "
            f"{row['class_name']:<20.20}  "
            f"{row['precision']:>9.3f}  "
            f"{row['recall']:>7.3f}  "
            f"{row['ap50']:>7.3f}  "
            f"{row['ap50_95']:>8.3f}"
        )


def print_confusion_summary(entries: Sequence[Tuple[str, str, float]]) -> None:
    if not entries:
        print("\n[Confusion summary] No off-diagonal entries with positive counts.")
        return
    print("\n[Confusion summary - largest off-diagonal counts]")
    header = f"{'Rank':>4}  {'True class':<20}  {'Pred class':<20}  {'Count':>10}"
    print(header)
    print("-" * len(header))
    for idx, (true_name, pred_name, value) in enumerate(entries, start=1):
        print(f"{idx:>4}  {true_name:<20.20}  {pred_name:<20.20}  {value:>10.0f}")


def save_sorted_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["class_id", "class_name", "precision", "recall", "ap50", "ap50_95"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[output] Saved sorted metrics -> {path}")


def main() -> None:
    args = parse_args()
    per_class_path = args.metrics_dir / f"{args.run_id}_per_class.csv"
    conf_path = args.metrics_dir / f"{args.run_id}_confusion_matrix.csv"

    per_class_rows = load_per_class_metrics(per_class_path)
    sorted_rows = sorted(per_class_rows, key=lambda row: row[args.sort_by], reverse=not args.ascending)
    print_per_class_table(sorted_rows, args.sort_by, args.top)

    if args.output_csv:
        save_sorted_csv(sorted_rows, args.output_csv.resolve())

    if args.include_confusion:
        class_names, matrix = load_confusion_matrix(conf_path)
        summary = summarize_confusion(class_names, matrix, args.conf_top)
        print_confusion_summary(summary)


if __name__ == "__main__":
    main()
