#!/usr/bin/env python3
"""
Visualize top/bottom confidence detections for quick QA.

Reads labels/meta/<run_id>_predictions.csv, selects the highest-confidence N images
and lowest-confidence M images, draws their YOLO boxes on the originals, and saves
annotated copies plus a CSV summary under labels/viz/<run_id>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize high/low confidence YOLO predictions.")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier (e.g., crawl_test_b).")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to filtered images root (default: data/filtered/<run_id>).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to YOLO label root (default: labels/yolo/<run_id>).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to predictions CSV (default: labels/meta/<run_id>_predictions.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Directory to save visualizations (default: labels/viz/<run_id>).",
    )
    parser.add_argument("--top-n", type=int, default=2, help="Number of highest-confidence images to sample.")
    parser.add_argument("--bottom-n", type=int, default=3, help="Number of lowest-confidence images to sample.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_images(df: pd.DataFrame, top_n: int, bottom_n: int) -> List[dict]:
    aggregated = df.groupby("image_path", as_index=False)["confidence"].max()
    selections: List[dict] = []
    seen: set[str] = set()

    for subset, bucket in (
        (aggregated.nlargest(top_n, "confidence"), "top"),
        (aggregated.nsmallest(bottom_n, "confidence"), "bottom"),
    ):
        for _, row in subset.iterrows():
            path_str = row["image_path"]
            if path_str in seen:
                continue
            selections.append({"image_path": path_str, "max_conf": float(row["confidence"]), "bucket": bucket})
            seen.add(path_str)
    return selections


def draw_boxes(img_path: Path, label_path: Path, out_path: Path) -> None:
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if label_path.exists():
        with label_path.open() as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                img_w, img_h = img.size
                cx, cy = x * img_w, y * img_h
                bw, bh = w * img_w, h * img_h
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
                draw.text((x1 + 4, y1 + 4), str(int(cls)), fill=(255, 255, 255))
    ensure_dir(out_path.parent)
    img.save(out_path, format="JPEG", quality=90)


def main() -> None:
    args = parse_args()
    images_root = (args.images or (Path("data/filtered") / args.run_id)).resolve()
    labels_root = (args.labels or (Path("labels/yolo") / args.run_id)).resolve()
    predictions_csv = (args.predictions or (Path("labels/meta") / f"{args.run_id}_predictions.csv")).resolve()
    out_dir = (args.out or (Path("labels/viz") / args.run_id)).resolve()

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    df = pd.read_csv(predictions_csv)
    if df.empty:
        raise RuntimeError("Predictions CSV is empty.")

    df["confidence"] = df["confidence"].astype(float)
    selected = select_images(df, args.top_n, args.bottom_n)
    if not selected:
        raise RuntimeError("No images selected. Adjust top/bottom values.")

    summary_rows = []
    for entry in selected:
        img_path = Path(entry["image_path"]).resolve()
        if not img_path.exists():
            candidate = images_root / Path(entry["image_path"]).name
            if candidate.exists():
                img_path = candidate.resolve()
            else:
                raise FileNotFoundError(f"Image path not found: {entry['image_path']}")

        try:
            rel = img_path.relative_to(images_root)
        except ValueError:
            rel = Path(img_path.name)

        label_path = (labels_root / rel).with_suffix(".txt")
        viz_path = (out_dir / rel).with_suffix(".jpg")
        draw_boxes(img_path, label_path, viz_path)
        summary_rows.append(
            {
                "image_path": str(img_path),
                "max_conf": entry["max_conf"],
                "bucket": entry["bucket"],
                "viz_path": str(viz_path),
            }
        )

    ensure_dir(out_dir)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{args.run_id}_top_bottom.csv", index=False)
    print(f"Saved {len(summary_rows)} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
