#!/usr/bin/env python3
"""
Train a YOLO model (Ultralytics) using the repository's standard config files.

Usage example:
  python scripts/train_yolo.py \
    --config configs/food_poc.yaml \
    --data data/5_datasets/crawl_test_b/crawl_test_b.yaml \
    --run-id crawl_test_b \
    --model models/yolo11l.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from matplotlib import font_manager, pyplot as plt  # type: ignore

YOLO_CONFIG_DIR = Path(".cache") / "ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR.resolve()))

from ultralytics import YOLO  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO models using Ultralytics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/food_poc.yaml"),
        help="Training config YAML (default: configs/food_poc.yaml).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="YOLO dataset YAML. If omitted, tries configs or data/5_datasets/<run-id>/<run-id>.yaml.",
    )
    parser.add_argument("--run-id", help="Run identifier used for dataset inference and logging names.")
    parser.add_argument("--model", type=Path, help="Override model weights path from the config.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--img-size", type=int, help="Override training image size.")
    parser.add_argument("--device", type=str, help="Override device string (cpu, 0, 0,1 ...).")
    parser.add_argument("--project", type=Path, help="Override output project directory (Ultralytics `--project`).")
    parser.add_argument("--name", help="Override run name inside the project directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without launching training.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def render_value(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError:
            return value
    return value


def resolve_dataset_yaml(args: argparse.Namespace, cfg: Dict[str, Any], context: Dict[str, str]) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    candidates: list[Optional[Path]] = [
        args.data,
        Path(render_value(dataset_cfg.get("yaml"), context)) if dataset_cfg.get("yaml") else None,
        Path(render_value(dataset_cfg.get("data_yaml"), context)) if dataset_cfg.get("data_yaml") else None,
        Path(render_value(dataset_cfg.get("dataset_yaml"), context)) if dataset_cfg.get("dataset_yaml") else None,
    ]
    if args.run_id:
        default_ds = Path("data") / "5_datasets" / args.run_id / f"{args.run_id}.yaml"
        candidates.append(default_ds)

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Dataset YAML not found. Pass --data, add dataset.yaml to the config, "
        "or run scripts/prepare_food_dataset.py to generate data/5_datasets/<run-id>/<run-id>.yaml."
    )


def prepare_train_kwargs(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    dataset_yaml: Path,
    context: Dict[str, str],
) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    training_cfg = cfg.get("training", {})
    augmentation_cfg = cfg.get("augmentation", {})
    logging_cfg = cfg.get("logging", {})

    model_path = args.model or training_cfg.get("model") or "yolov8n.pt"

    epochs = args.epochs or training_cfg.get("epochs", 50)
    batch_size = args.batch_size or training_cfg.get("batch_size", 16)
    img_size = args.img_size or training_cfg.get("img_size", 640)
    device = args.device or training_cfg.get("device", "auto")

    project_dir = Path(render_value(args.project or logging_cfg.get("output_dir", "runs/train"), context)).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        args.name
        or context.get("run_id")
        or cfg.get("project_name")
        or dataset_yaml.stem
    )

    train_kwargs: Dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
    }

    optional_training_keys = ["optimizer", "lr0", "lrf", "weight_decay", "patience", "momentum"]
    for key in optional_training_keys:
        if key in training_cfg and training_cfg[key] is not None:
            train_kwargs[key] = training_cfg[key]

    for key, value in augmentation_cfg.items():
        if value is not None:
            train_kwargs[key] = value

    if logging_cfg.get("save_best_only"):
        train_kwargs["save"] = True
        train_kwargs["save_period"] = -1  # disable periodic checkpoints

    if args.resume:
        train_kwargs["resume"] = True

    return train_kwargs, model_path, logging_cfg.get("metrics_file")


def serialize_metrics(metrics: Any) -> Any:
    if isinstance(metrics, dict):
        return {str(k): serialize_metrics(v) for k, v in metrics.items()}
    if isinstance(metrics, (list, tuple)):
        return [serialize_metrics(v) for v in metrics]
    if isinstance(metrics, (int, float, str)) or metrics is None:
        return metrics
    return str(metrics)


def save_metrics(metrics: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(serialize_metrics(metrics), fp, ensure_ascii=False, indent=2)


def _normalize_name_mapping(names: Any) -> List[str]:
    if not names:
        return []
    if isinstance(names, dict):
        ordered = sorted(names.items(), key=lambda item: int(item[0]) if isinstance(item[0], (int, str)) and str(item[0]).isdigit() else item[0])
        return [str(value) for _, value in ordered]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return [str(names)]


def resolve_class_names(trainer: Any, dataset_yaml: Path) -> List[str]:
    names = _normalize_name_mapping(getattr(getattr(trainer, "data", None), "names", None))
    if names:
        return names
    trainer_names = _normalize_name_mapping(getattr(trainer, "names", None))
    if trainer_names:
        return trainer_names
    yaml_names: Sequence[str] = []
    try:
        with dataset_yaml.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        yaml_names = _normalize_name_mapping(data.get("names"))
    except FileNotFoundError:
        yaml_names = []
    return list(yaml_names)


def export_per_class_metrics(metrics: Any, class_names: Sequence[str], output_dir: Path, run_id: str) -> None:
    box = getattr(metrics, "box", None)
    if not box or not len(getattr(box, "p", [])):
        print("[warn] Per-class metrics unavailable (missing precision/recall arrays).")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_id}_per_class.csv"
    rows: List[List[Any]] = []
    for idx, class_name in enumerate(class_names):
        try:
            precision = float(box.p[idx])
            recall = float(box.r[idx])
            ap50 = float(box.ap50[idx])
            ap5095 = float(box.ap[idx])
        except Exception:
            continue
        rows.append([idx, class_name, precision, recall, ap50, ap5095])
    if not rows:
        print("[warn] Per-class metrics export skipped (no rows).")
        return
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["class_id", "class_name", "precision", "recall", "ap50", "ap50_95"])
        writer.writerows(rows)
    print(f"[metrics] Saved per-class metrics -> {csv_path}")


def export_confusion_matrix(validator: Any, class_names: Sequence[str], output_dir: Path, run_id: str) -> None:
    confusion_matrix = getattr(getattr(validator, "confusion_matrix", None), "matrix", None)
    if confusion_matrix is None:
        print("[warn] Confusion matrix unavailable; skipping export.")
        return
    matrix = np.asarray(confusion_matrix)
    if matrix.ndim != 2:
        print("[warn] Confusion matrix has unexpected shape; skipping export.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_id}_confusion_matrix.csv"
    axis_size = matrix.shape[0]
    if axis_size == len(class_names) + 1:
        axis_names = list(class_names) + ["background"]
    else:
        axis_names = [f"class_{i}" for i in range(axis_size)]
    header = ["true\\pred"] + axis_names
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for idx, row in enumerate(matrix):
            row_name = axis_names[idx] if idx < len(axis_names) else f"class_{idx}"
            writer.writerow([row_name] + [float(value) for value in row])
    print(f"[metrics] Saved confusion matrix CSV -> {csv_path}")


def main() -> None:
    font_path = Path("models") / "font" / "NanumGothic.ttf"
    if font_path.exists():
        try:
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = "NanumGothic"
            os.environ.setdefault("ULTRALYTICS_FONT", str(font_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to register NanumGothic font: {exc}")

    args = parse_args()
    cfg = load_yaml_config(args.config)

    context = {
        "run_id": args.run_id or "",
        "project_name": cfg.get("project_name", ""),
    }

    dataset_yaml = resolve_dataset_yaml(args, cfg, context)
    train_kwargs, model_path, metrics_path_str = prepare_train_kwargs(args, cfg, dataset_yaml, context)

    print("[train] Resolved arguments:")
    print(f"  config       : {args.config}")
    print(f"  model        : {model_path}")
    print(f"  dataset yaml : {dataset_yaml}")
    print(f"  project/name : {train_kwargs['project']} / {train_kwargs['name']}")
    print(f"  epochs/batch : {train_kwargs['epochs']} / {train_kwargs['batch']}")
    print(f"  device       : {train_kwargs['device']}")

    if args.dry_run:
        print("[dry-run] Training skipped.")
        return

    model = YOLO(str(model_path))
    results = model.train(**train_kwargs)

    metrics = getattr(results, "metrics", None)
    if metrics_path_str and metrics is not None:
        metrics_path = Path(render_value(metrics_path_str, context)).resolve()
        save_metrics(metrics, metrics_path)
        print(f"[metrics] Saved metrics -> {metrics_path}")

    trainer = getattr(model, "trainer", None)
    reports_dir = Path("data") / "meta" / "train_metrics"
    run_identifier = args.run_id or train_kwargs["name"]
    if trainer:
        class_names = resolve_class_names(trainer, dataset_yaml)
        if class_names:
            det_metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
            if det_metrics:
                export_per_class_metrics(det_metrics, class_names, reports_dir, run_identifier)
            export_confusion_matrix(getattr(trainer, "validator", None), class_names, reports_dir, run_identifier)
        else:
            print("[warn] Could not resolve class names; skipping per-class exports.")

    print("[done] Training run complete.")


if __name__ == "__main__":
    main()
