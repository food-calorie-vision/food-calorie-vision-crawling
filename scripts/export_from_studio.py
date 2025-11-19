#!/usr/bin/env python3
"""
Download a YOLO export from Label Studio and optionally package it into a dataset.

Example:
  python scripts/export_from_studio.py \
    --project-id 9 \
    --run-id crawl_test_b \
    --token $LABEL_STUDIO_TOKEN \
    --image-root data/2_filtered/crawl_test_b \
    --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import getpass
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a Label Studio export and (optionally) build a dataset.")
    parser.add_argument("--run-id", required=True, help="Run identifier. Used for folder naming.")
    parser.add_argument("--project-id", required=True, type=int, help="Label Studio project numeric ID.")
    parser.add_argument("--token", help="Label Studio access token (PAT or legacy). Prompted if omitted.")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Label Studio base URL.")
    parser.add_argument(
        "--export-type",
        default="YOLO",
        type=lambda value: normalize_export_type(value),
        help="Export format requested from Label Studio (only YOLO supported).",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include tasks without annotations (default: download annotated tasks only).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("labels/4_export_from_studio"),
        help="Where to place the downloaded export (default: labels/4_export_from_studio).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Directory containing source images (needed when exports do not include images).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/5_datasets"),
        help="Destination for packaged datasets (default: data/5_datasets).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.3,
        help="Validation split ratio for packaging (default: 0.3).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting an existing dataset directory (default: skip if exists).",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Only download/extract the export and skip dataset packaging.",
    )
    return parser.parse_args()


def resolve_token(explicit: str | None) -> str:
    token = explicit or getpass.getpass("Label Studio access token: ").strip()
    if not token:
        raise RuntimeError("Label Studio token is required.")
    return token


def build_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}"})
    return session


def normalize_export_type(value: str) -> str:
    normalized = value.strip().upper()
    allowed = {"YOLO"}
    if normalized not in allowed:
        raise argparse.ArgumentTypeError(f"Unsupported export type '{value}'. Allowed: {', '.join(sorted(allowed))}")
    return normalized


def download_export(
    session: requests.Session,
    base_url: str,
    project_id: int,
    export_type: str,
    destination: Path,
    include_unlabeled: bool,
) -> Path:
    base = base_url.rstrip("/")
    params = {
        "exportType": export_type,
        "download_all_tasks": "true" if include_unlabeled else "false",
    }
    url = f"{base}/api/projects/{project_id}/export"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, params=params, stream=True) as resp:
        resp.raise_for_status()
        with destination.open("wb") as fp:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fp.write(chunk)
    return destination


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    return target_dir


def detect_export_source(extracted_root: Path) -> Path:
    """
    Find the directory that contains a `labels/` folder inside the extracted export.
    """
    candidate: Optional[Path] = None
    for lbl_dir in extracted_root.rglob("labels"):
        if lbl_dir.is_dir():
            base = lbl_dir.parent
            candidate = base
            break
    if candidate is None:
        raise RuntimeError(f"Could not find a 'labels/' directory inside {extracted_root}")
    return candidate


def package_dataset(
    run_id: str,
    source_dir: Path,
    image_root: Path | None,
    dataset_root: Path,
    val_ratio: float,
    allow_overwrite: bool,
) -> None:
    dataset_dir = dataset_root / run_id
    if dataset_dir.exists() and not allow_overwrite:
        print(f"[skip] Dataset {dataset_dir} exists. Use --allow-overwrite to rebuild.")
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_food_dataset.py"),
        "--run-id",
        run_id,
        "--source",
        str(source_dir),
        "--output-root",
        str(dataset_root),
    ]
    if image_root:
        cmd += ["--image-root", str(image_root)]
    if val_ratio > 0:
        cmd += ["--val-ratio", str(val_ratio)]
    if allow_overwrite:
        cmd.append("--overwrite")
    print(f"[prepare] {' '.join(cmd)}")
    import subprocess

    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)
    session = build_session(token)
    output_dir = (args.output_root / args.run_id).resolve()
    download_dir = output_dir / "exports"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"{args.run_id}_{args.export_type}.zip"
    print(f"[export] Downloading project {args.project_id} ({args.export_type}) -> {zip_path}")
    download_export(
        session,
        args.base_url,
        args.project_id,
        args.export_type,
        zip_path,
        include_unlabeled=args.include_unlabeled,
    )
    print(f"[export] Saved archive: {zip_path}")

    extracted_dir = output_dir / "_extracted"
    extract_zip(zip_path, extracted_dir)
    source_dir = detect_export_source(extracted_dir)
    staged_source = output_dir / "source"
    if staged_source.exists():
        shutil.rmtree(staged_source)
    shutil.copytree(source_dir, staged_source)
    print(f"[export] Located Label Studio export root: {source_dir}")
    print(f"[export] Staged source directory: {staged_source}")

    if args.skip_dataset:
        print("[done] Export downloaded and extracted.")
        return

    if not args.image_root:
        print(
            "[warn] --image-root not provided. If the export lacks images, "
            "prepare_food_dataset may fail. Provide --image-root when needed."
        )

    package_dataset(
        run_id=args.run_id,
        source_dir=staged_source,
        image_root=args.image_root.resolve() if args.image_root else None,
        dataset_root=args.dataset_root.resolve(),
        val_ratio=args.val_ratio,
        allow_overwrite=args.allow_overwrite,
    )
    print("[done] Dataset packaging complete.")


if __name__ == "__main__":
    main()
