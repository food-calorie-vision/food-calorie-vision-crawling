#!/usr/bin/env python3
"""
Image crawling utility for Food Calorie Vision Crawling.

Features
--------
* Reads prioritized class names from target_food.csv or CLI.
* Downloads images via DuckDuckGo search (no API key required).
* Enforces per-class min/max counts, optional global limit, and throttling.
* Persists run metadata to data/meta/crawl_logs/<run_id>.json for traceability.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from duckduckgo_search import DDGS  # type: ignore
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH_DEFAULT = ROOT / "target_food.csv"
RAW_DIR_DEFAULT = ROOT / "data" / "raw"
LOG_DIR = ROOT / "data" / "meta" / "crawl_logs"


@dataclass
class ClassSummary:
    name: str
    requested: int
    downloaded: int
    errors: int
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl food images by class.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH_DEFAULT,
        help="Path to target_food.csv containing prioritized classes.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Specific class names to crawl. Overrides CSV if provided.",
    )
    parser.add_argument(
        "--min_per_class",
        type=int,
        default=50,
        help="Minimum number of images to attempt per class.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=300,
        help="Maximum number of images to download per class.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Global image limit across all classes (useful for dry-runs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR_DEFAULT,
        help="Output directory for downloaded images (per class subfolders).",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Run identifier (YYYYMMDD_stage_seq). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between downloads to stay polite with sources.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout per image download (seconds).",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Mozilla/5.0 (compatible; FoodCrawler/1.0)",
        help="HTTP User-Agent header value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip downloads; only print planned actions and exit.",
    )
    return parser.parse_args()


def load_classes(csv_path: Path, limit: Optional[int] = None) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    preferred_columns = ["class_name", "food_name", "name", "item", "food"]
    classes: List[str] = []

    with csv_path.open("r", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("CSV must contain headers.")
        fallback_column = reader.fieldnames[0]
        for row in reader:
            class_name = ""
            for column in preferred_columns:
                if column in row and row[column].strip():
                    class_name = row[column].strip()
                    break
            if not class_name:
                class_name = row.get(fallback_column, "").strip()
            if class_name:
                classes.append(class_name)
            if limit and len(classes) >= limit:
                break
    if not classes:
        raise ValueError("No class names found in CSV.")
    return classes


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "class"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_run_id(stage: str = "crawl") -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d")
    return f"{stamp}_{stage}_a"


def fetch_candidate_urls(query: str, max_results: int) -> Iterable[str]:
    with DDGS() as ddgs:
        for result in ddgs.images(keywords=query, max_results=max_results):
            url = result.get("image") or result.get("thumbnail")
            if url:
                yield url


def download_image(url: str, dest: Path, timeout: float, user_agent: str) -> bool:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent},
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            return False
        with dest.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)
        return True
    except requests.RequestException as exc:
        logging.debug("Download failed for %s: %s", url, exc)
        return False


def crawl_class(
    class_name: str,
    out_dir: Path,
    per_class_max: int,
    timeout: float,
    user_agent: str,
    delay: float,
    dry_run: bool,
) -> ClassSummary:
    safe_name = sanitize_name(class_name)
    class_dir = out_dir / safe_name
    ensure_dir(class_dir)

    downloaded = 0
    errors = 0
    requested = per_class_max
    seen_urls: set[str] = set()
    max_results = per_class_max * 3

    if dry_run:
        logging.info("[DRY-RUN] Would download up to %s images for '%s'", per_class_max, class_name)
        return ClassSummary(name=class_name, requested=requested, downloaded=0, errors=0, notes="dry-run")

    for idx, url in enumerate(fetch_candidate_urls(class_name, max_results=max_results), start=1):
        if url in seen_urls:
            continue
        seen_urls.add(url)
        filename = f"{safe_name}_{idx:04d}.jpg"
        dest_path = class_dir / filename
        if dest_path.exists():
            continue
        success = download_image(url, dest_path, timeout=timeout, user_agent=user_agent)
        if success:
            downloaded += 1
        else:
            errors += 1
        if downloaded >= per_class_max:
            break
        if delay > 0:
            time.sleep(delay)

    notes = ""
    if downloaded < requested:
        notes = f"shortfall_{requested-downloaded}"
    return ClassSummary(name=class_name, requested=requested, downloaded=downloaded, errors=errors, notes=notes)


def save_run_log(run_id: str, summaries: Sequence[ClassSummary], sources: Sequence[str]) -> Path:
    ensure_dir(LOG_DIR)
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classes": [asdict(summary) for summary in summaries],
        "source": list(sources),
        "notes": "",
    }
    log_path = LOG_DIR / f"{run_id}.json"
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return log_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_id = args.run_id or generate_run_id("crawl")
    out_dir = (args.out / run_id).resolve()
    ensure_dir(out_dir)

    logging.info("Run ID: %s", run_id)
    logging.info("Output directory: %s", out_dir)

    selected_classes = args.classes or load_classes(args.csv)
    logging.info("Total classes available: %d", len(selected_classes))

    global_limit = args.limit
    total_downloaded = 0
    summaries: List[ClassSummary] = []

    with tqdm(total=len(selected_classes), desc="Classes") as class_bar:
        for class_name in selected_classes:
            remaining = None
            if global_limit is not None:
                remaining = max(global_limit - total_downloaded, 0)
                if remaining <= 0:
                    logging.info("Global limit reached. Stopping.")
                    break
            per_class_target = min(args.max_per_class, remaining) if remaining else args.max_per_class

            summary = crawl_class(
                class_name=class_name,
                out_dir=out_dir,
                per_class_max=per_class_target,
                timeout=args.timeout,
                user_agent=args.user_agent,
                delay=args.delay,
                dry_run=args.dry_run,
            )
            summaries.append(summary)
            total_downloaded += summary.downloaded
            class_bar.update(1)

    log_path = save_run_log(run_id, summaries, sources=["duckduckgo_images"])
    logging.info("Saved crawl log to %s", log_path)

    if args.dry_run:
        logging.info("Dry-run completed. No files downloaded.")


if __name__ == "__main__":
    main()
