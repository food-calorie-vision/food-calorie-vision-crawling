#!/usr/bin/env python3
"""
Bootstrap a Label Studio project by run_id.

Steps:
1. Collect images under data/2_filtered/<run_id> (fallback: data/1_raw/<run_id>).
2. Generate a RectangleLabels config using either a CSV map or folder names.
3. Create/patch a Label Studio project via REST API.
4. Import tasks referencing /data/local-files paths served from /label-studio/data/local_storage/<prefix>.

The script is interactive only for the access token prompt; everything else
derives from the provided run_id to keep agent workflows deterministic.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTERED_ROOT = PROJECT_ROOT / "data" / "2_filtered"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "1_raw"
DEFAULT_LOCAL_PREFIX = "workspace"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PALETTE = [
    "#ef4444",
    "#f97316",
    "#facc15",
    "#22c55e",
    "#14b8a6",
    "#0ea5e9",
    "#6366f1",
    "#a855f7",
    "#ec4899",
    "#f43f5e",
]
IMPORT_CHUNK = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/seed a Label Studio project for the given run_id.")
    parser.add_argument("--run-id", required=True, help="Pipeline run identifier (e.g., 20250107_poc_a).")
    parser.add_argument("--token", help="Label Studio access token. Prompted if omitted.")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Label Studio base URL.")
    parser.add_argument("--project-title", help="Override project title (default: Food Vision <run_id>).")
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional CSV listing label names. If omitted, top-level class folders under the images root are used.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_FILTERED_ROOT,
        help="Root directory that stores filtered images grouped by run_id.",
    )
    parser.add_argument(
        "--fallback-images-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Fallback root if filtered images are missing.",
    )
    parser.add_argument("--disable-fallback", action="store_true", help="Do not look at fallback roots for images.")
    parser.add_argument("--limit", type=int, help="Optionally cap the number of images imported.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing tasks before import.")
    parser.add_argument("--dry-run", action="store_true", help="Print the actions without hitting the API.")
    parser.add_argument(
        "--auth-type",
        choices=["auto", "token", "bearer"],
        default="auto",
        help="Authentication scheme: legacy Token, Bearer, or auto-detect.",
    )
    parser.add_argument(
        "--local-prefix",
        default=DEFAULT_LOCAL_PREFIX,
        help="Path prefix inside LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (default: workspace).",
    )
    parser.add_argument(
        "--tasks-json",
        type=Path,
        help="Optional JSON file containing full Label Studio task entries (including predictions/annotations).",
    )
    return parser.parse_args()


def resolve_path(base: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def resolve_token(explicit: str | None) -> str:
    token = explicit or os.environ.get("LABEL_STUDIO_TOKEN")
    if not token:
        token = getpass.getpass("Label Studio access token (입력 시 숨김 처리, Enter=취소): ").strip()
    if not token:
        raise RuntimeError("Label Studio token is required.")
    return token


def build_session(token: str, scheme: str) -> requests.Session:
    session = requests.Session()
    header = f"{scheme} {token}"
    session.headers.update({"Authorization": header, "Content-Type": "application/json"})
    return session


def load_label_names(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label map file not found: {csv_path}")
    names: list[str] = []
    with csv_path.open("r", encoding="utf-8") as fp:
        for idx, raw in enumerate(fp):
            name = raw.strip()
            if not name:
                continue
            if idx == 0 and "food_class" in name.lower():
                # Skip header rows such as "food_class1".
                continue
            names.append(name)
    if not names:
        raise RuntimeError(f"No label names found in {csv_path}")
    return names


def discover_folder_labels(images_dir: Path) -> list[str]:
    labels = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
    if not labels:
        raise RuntimeError(
            f"No class folders found under {images_dir}. Provide --label-map to specify labels explicitly."
        )
    return labels


def build_label_config(labels: Sequence[str]) -> str:
    label_lines = []
    for idx, label in enumerate(labels):
        color = PALETTE[idx % len(PALETTE)]
        label_lines.append(f'    <Label value="{label}" background="{color}" />')
    options = "\n".join(label_lines)
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{options}\n"
        "  </RectangleLabels>\n"
        "</View>"
    )


def choose_images_dir(run_id: str, primary_root: Path, fallback_root: Path | None) -> Path:
    primary = (primary_root / run_id).resolve()
    if primary.exists():
        return primary
    if fallback_root:
        fallback = (fallback_root / run_id).resolve()
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"No images found for run_id '{run_id}' in {primary} or fallback roots.")


def collect_images(images_dir: Path, limit: int | None) -> list[Path]:
    files = sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if limit:
        return files[:limit]
    return files


def images_to_tasks(image_paths: Sequence[Path], run_id: str, local_prefix: str) -> list[dict]:
    tasks: list[dict] = []
    for path in image_paths:
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError as err:
            raise RuntimeError(f"Image path {path} is outside project root {PROJECT_ROOT}") from err
        rel_posix = rel.as_posix()
        encoded = quote(rel_posix, safe="/")
        url_path = f"{local_prefix}/{rel_posix}".lstrip("/")
        encoded = quote(url_path, safe="/")
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d={encoded}",
                    "run_id": run_id,
                    "relative_path": rel_posix,
                }
            }
        )
    return tasks


def load_tasks_json(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise RuntimeError(f"Unsupported JSON structure in {json_path}; expected list or dict.")


def chunked(items: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def find_project(session: requests.Session, base_url: str, title: str) -> dict | None:
    url = f"{base_url}/api/projects/"
    params = {"page_size": 100}
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("results", []):
            if item.get("title") == title:
                return item
        url = payload.get("next")
        params = None
    return None


def create_or_update_project(
    session: requests.Session,
    base_url: str,
    title: str,
    description: str,
    label_config: str,
) -> dict:
    project = find_project(session, base_url, title)
    if project:
        project_id = project["id"]
        update_payload = {}
        if project.get("label_config") != label_config:
            update_payload["label_config"] = label_config
        if description and project.get("description") != description:
            update_payload["description"] = description
        if update_payload:
            resp = session.patch(f"{base_url}/api/projects/{project_id}/", json=update_payload)
            resp.raise_for_status()
            project.update(resp.json())
        return project

    payload = {
        "title": title,
        "label_config": label_config,
        "description": description,
    }
    resp = session.post(f"{base_url}/api/projects/", json=payload)
    resp.raise_for_status()
    return resp.json()


def delete_existing_tasks(session: requests.Session, base_url: str, project_id: int) -> None:
    resp = session.delete(f"{base_url}/api/projects/{project_id}/tasks/")
    if resp.status_code not in (200, 204):
        resp.raise_for_status()


def import_tasks(
    session: requests.Session,
    base_url: str,
    project_id: int,
    tasks: Sequence[dict],
) -> int:
    url = f"{base_url}/api/projects/{project_id}/import"
    imported = 0
    for batch in chunked(tasks, IMPORT_CHUNK):
        resp = session.post(url, json=batch)
        resp.raise_for_status()
        data = resp.json()
        imported += data.get("task_count") or len(batch)
    return imported


def ensure_local_storage(
    session: requests.Session,
    base_url: str,
    project_id: int,
    storage_path: str,
    run_id: str,
) -> None:
    base = base_url.rstrip("/")
    list_url = f"{base}/api/storages/localfiles/"
    params = {"project": project_id}
    resp = session.get(list_url, params=params)
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results", []) if isinstance(payload, dict) else payload
    for item in results:
        if item.get("path") == storage_path:
            return
    data = {
        "title": f"{run_id}-workspace",
        "path": storage_path,
        "regex": ".*",
        "use_blob_urls": False,
        "enable_file_watcher": False,
        "project": project_id,
    }
    resp = session.post(list_url, params=params, json=data)
    resp.raise_for_status()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    token = resolve_token(args.token)

    label_map_path = resolve_path(PROJECT_ROOT, args.label_map) if args.label_map else None
    images_root = resolve_path(PROJECT_ROOT, args.images_root)
    tasks_json_path = resolve_path(PROJECT_ROOT, args.tasks_json) if args.tasks_json else None
    fallback_root = None
    if not args.disable_fallback and args.fallback_images_root:
        fallback_root = resolve_path(PROJECT_ROOT, args.fallback_images_root)

    images_dir = choose_images_dir(args.run_id, images_root, fallback_root)
    if tasks_json_path:
        tasks = load_tasks_json(tasks_json_path)
    else:
        image_paths = collect_images(images_dir, args.limit)
        if not image_paths:
            raise RuntimeError(f"No images found to import under {images_dir}")
        tasks = images_to_tasks(image_paths, args.run_id, args.local_prefix)
    labels = load_label_names(label_map_path) if label_map_path else discover_folder_labels(images_dir)
    label_config = build_label_config(labels)

    project_title = args.project_title or f"Food Vision {args.run_id}"
    description = (
        f"Auto-generated for run_id {args.run_id}. "
        f"Images sourced from {images_dir.relative_to(PROJECT_ROOT)} "
        f"({len(tasks)} tasks)."
    )

    print(f"[Label Studio] Target project: {project_title}")
    print(f"[Label Studio] Base URL: {base_url}")
    print(f"[Label Studio] Images dir: {images_dir}")
    print(f"[Label Studio] Tasks to import: {len(tasks)}")
    if tasks_json_path:
        print(f"[Label Studio] Tasks JSON: {tasks_json_path}")
    if args.label_map:
        print(f"[Label Studio] Label map: {label_map_path} ({len(labels)} labels)")
    else:
        print(f"[Label Studio] Labels derived from class folders ({len(labels)})")

    if args.dry_run:
        print("[Label Studio] Dry-run enabled; skipping API calls.")
        return

    schemes: list[str]
    if args.auth_type == "auto":
        schemes = ["Token", "Bearer"]
    elif args.auth_type == "token":
        schemes = ["Token"]
    else:
        schemes = ["Bearer"]

    last_error: requests.HTTPError | None = None
    for scheme in schemes:
        session = build_session(token, scheme)
        try:
            project = create_or_update_project(session, base_url, project_title, description, label_config)
            project_id = project["id"]
            print(f"[Label Studio] Project ID: {project_id} (auth: {scheme})")
            try:
                rel_images = images_dir.relative_to(PROJECT_ROOT)
            except ValueError as err:
                raise RuntimeError(
                    f"Images dir {images_dir} must be inside project root {PROJECT_ROOT}"
                ) from err
            storage_root = Path("/label-studio/data/local_storage") / args.local_prefix
            storage_path = storage_root / rel_images
            ensure_local_storage(session, base_url, project_id, storage_path.as_posix(), args.run_id)
            print(f"[Label Studio] Local storage registered: {storage_path}")

            if args.overwrite:
                print("[Label Studio] Removing existing tasks...")
                delete_existing_tasks(session, base_url, project_id)

            imported = import_tasks(session, base_url, project_id, tasks)
            print(f"[Label Studio] Imported {imported} tasks into '{project_title}'.")
            break
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 401 and scheme != schemes[-1]:
                print(f"[Label Studio] Authentication failed with scheme '{scheme}'. Retrying with next scheme...")
                last_error = err
                continue
            raise
    else:
        if last_error:
            raise last_error


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(1)
