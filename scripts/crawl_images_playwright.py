#!/usr/bin/env python3
"""
Playwright 기반 DuckDuckGo 이미지 크롤러.
DuckDuckGo는 동적으로 생성된 긴 랜덤 문자열 클래스(예: SZ76bwIlqO8BBoqOLqYV)를 가진 div 안에 이미지를 배치하므로
해당 요소를 JavaScript로 찾아 이미지 URL을 추출합니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote_plus

import requests
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR_DEFAULT = ROOT / "data" / "raw"
LOG_DIR = ROOT / "data" / "meta" / "crawl_logs"


@dataclass
class ClassSummary:
    name: str
    requested: int
    downloaded: int
    errors: int
    notes: str = ""
    folder: str = ""


def load_classes(csv_path: Path, limit: Optional[int] = None, start_from: Optional[str] = None) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    preferred_columns = ["class_name", "food_name", "name", "item", "food"]
    classes: List[str] = []
    start_collecting = start_from is None  # start_from이 None이면 처음부터 수집

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
                # start_from이 지정되었고 아직 시작하지 않았다면, 해당 클래스를 찾을 때까지 건너뛰기
                if not start_collecting and start_from:
                    if class_name == start_from:
                        start_collecting = True
                        logging.info("Starting from class: %s", class_name)
                    else:
                        continue  # 아직 시작 클래스를 찾지 못함
                
                if start_collecting:
                    classes.append(class_name)
                    if limit and len(classes) >= limit:
                        break
    
    if start_from and not start_collecting:
        logging.warning("Start class '%s' not found in CSV. Starting from beginning.", start_from)
    
    if not classes:
        raise ValueError("No class names found in CSV.")
    return classes


def sanitize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name or "").strip()
    sanitized = re.sub(r"[^\w가-힣-]+", "_", normalized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if sanitized:
        return sanitized

    ascii_slug = unicodedata.normalize("NFKD", normalized)
    ascii_slug = ascii_slug.encode("ascii", "ignore").decode("ascii")
    ascii_slug = re.sub(r"[^\w-]+", "_", ascii_slug)
    ascii_slug = re.sub(r"_+", "_", ascii_slug).strip("_")
    if ascii_slug:
        return ascii_slug

    return f"class_{abs(hash(normalized)) & 0xFFFF:04x}"


def resolve_safe_class_names(class_names: Sequence[str]) -> List[Tuple[str, str]]:
    seen: Dict[str, int] = {}
    resolved: List[Tuple[str, str]] = []
    for class_name in class_names:
        base = sanitize_name(class_name)
        count = seen.get(base, 0)
        seen[base] = count + 1
        safe_name = base if count == 0 else f"{base}-{count+1}"
        resolved.append((class_name, safe_name))
    return resolved


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_run_id(stage: str = "crawl_pw") -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d")
    return f"{stamp}_{stage}_a"


def download_image(url: str, dest: Path, timeout: float, user_agent: str) -> bool:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent},
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < 100:
            logging.debug("Download failed for %s: File too small (%s bytes)", url[:80], content_length)
            return False

        with dest.open("wb") as fp:
            bytes_written = 0
            first_chunk = None
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    if first_chunk is None:
                        first_chunk = chunk[:12]
                    fp.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written < 100:
                logging.debug("Download failed for %s: Written file too small (%s bytes)", url[:80], bytes_written)
                dest.unlink(missing_ok=True)
                return False

            if content_type and "image" not in content_type:
                if first_chunk:
                    image_signatures = [
                        b"\xff\xd8\xff",
                        b"\x89PNG\r\n\x1a\n",
                        b"GIF87a",
                        b"GIF89a",
                        b"RIFF",
                    ]
                    is_image = any(first_chunk.startswith(sig) for sig in image_signatures)
                    if not is_image:
                        logging.debug("Download failed for %s: Not an image (Content-Type: %s)", url[:80], content_type)
                        dest.unlink(missing_ok=True)
                        return False

            try:
                from PIL import Image

                with Image.open(dest) as img:
                    width, height = img.size
                    if width <= 50 or height <= 50:
                        logging.debug("Download failed for %s: Image too small (%dx%d, likely an icon)", url[:80], width, height)
                        dest.unlink(missing_ok=True)
                        return False
                    if bytes_written < 2000 and (width <= 64 or height <= 64):
                        logging.debug("Download failed for %s: Small image with small file size (%dx%d, %d bytes)", url[:80], width, height, bytes_written)
                        dest.unlink(missing_ok=True)
                        return False
            except ImportError:
                pass
            except Exception as exc:
                logging.debug("Could not verify image size for %s: %s", url[:80], exc)

        return True
    except requests.exceptions.Timeout:
        logging.debug("Download timeout for %s", url[:80])
        return False
    except requests.exceptions.HTTPError as exc:
        logging.debug("HTTP error %d for %s: %s", exc.response.status_code if exc.response else 0, url[:80], exc)
        return False
    except requests.RequestException as exc:
        logging.debug("Download failed for %s: %s", url[:80], exc)
        return False
    except Exception as exc:
        logging.debug("Unexpected error downloading %s: %s", url[:80], exc)
        dest.unlink(missing_ok=True)
        return False


def save_run_log(run_id: str, summaries: Sequence[ClassSummary], sources: Sequence[str]) -> Path:
    ensure_dir(LOG_DIR)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "classes": [asdict(summary) for summary in summaries],
        "source": list(sources),
        "notes": "",
    }
    attempt_record = {
        "timestamp": timestamp,
        "classes": [asdict(summary) for summary in summaries],
        "source": list(sources),
        "notes": "",
    }
    log_path = LOG_DIR / f"{run_id}.json"
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as fp:
                existing = json.load(fp)
        except json.JSONDecodeError:
            existing = {}
        class_map: Dict[str, Dict[str, object]] = {}
        for entry in existing.get("classes", []):
            class_map[entry.get("name", "")] = entry
        for summary in summaries:
            class_map[summary.name] = asdict(summary)
        payload["classes"] = list(class_map.values())
        existing_sources = set(existing.get("source", []))
        payload["source"] = sorted(existing_sources.union(sources))
        payload["notes"] = existing.get("notes", "")
        history = existing.get("history", [])
        if isinstance(history, list):
            history.append(attempt_record)
        else:
            history = [attempt_record]
        payload["history"] = history
    else:
        payload["history"] = [attempt_record]
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return log_path


def normalize_duckduckgo_src(raw: str | None) -> str | None:
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    # DuckDuckGo srcset strings look like "url 1x, url 2x".
    candidate = candidate.split(",")[0].strip()
    candidate = candidate.split(" ")[0].strip()
    if not candidate or candidate.startswith("data:"):
        return None
    
    # 아이콘 파일 필터링 (.ico 확장자, favicon, icon 관련 URL)
    candidate_lower = candidate.lower()
    if (".ico" in candidate_lower or 
        "favicon" in candidate_lower or 
        "/icon" in candidate_lower or
        candidate_lower.endswith(".ico")):
        return None
    
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    return None


def maybe_accept_consent(page, timeout_ms: int = 5000) -> None:
    try:
        button = page.wait_for_selector("button#onetrust-accept-btn-handler", timeout=timeout_ms)
        button.click()
        logging.debug("Accepted DuckDuckGo consent dialog.")
    except PlaywrightTimeoutError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playwright-based DuckDuckGo image crawler.")
    parser.add_argument("--classes", nargs="+", help="Specific class names. Overrides CSV if provided.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "target_food.csv",
        help="Path to target_food.csv (used when --classes is omitted).",
    )
    parser.add_argument("--min_per_class", type=int, default=50, help="Minimum images to attempt per class.")
    parser.add_argument("--max_per_class", type=int, default=150, help="Maximum images to download per class.")
    parser.add_argument("--limit", type=int, default=None, help="Global image cap (dry-run style).")
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR_DEFAULT,
        help="Output root directory (class subfolders created automatically).",
    )
    parser.add_argument("--run-id", dest="run_id", type=str, help="Run identifier (YYYYMMDD_stage_seq).")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between downloads (seconds).")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout per download.")
    parser.add_argument("--start-from", dest="start_from", type=str, default=None, help="Start crawling from this class name (inclusive).")
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36",
        help="User-Agent header for image downloads.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show Chromium window (default: headless).",
    )
    parser.add_argument("--max-scrolls", type=int, default=60, help="Maximum scroll iterations per class.")
    parser.add_argument("--scroll-pause", type=float, default=1.2, help="Delay between scrolls (seconds).")
    parser.add_argument(
        "--load-timeout",
        type=float,
        default=20.0,
        help="Timeout (seconds) for DuckDuckGo page loads/selectors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip downloads; only report how many URLs were discovered.",
    )
    return parser.parse_args()


def collect_image_urls(
    query: str,
    max_results: int,
    max_scrolls: int,
    scroll_pause: float,
    headless: bool,
    load_timeout: float,
) -> List[str]:
    """
    Use Playwright to scroll DuckDuckGo image results and extract image URLs.
    """
    encoded_query = quote_plus(query)
    target_url = f"https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images"
    unique_urls: List[str] = []
    seen: Set[str] = set()

    logging.info("Opening DuckDuckGo images for '%s'", query)
    with sync_playwright() as p:
        launch_args = ["--no-sandbox"]
        browser = p.chromium.launch(headless=headless, args=launch_args)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        page = context.new_page()

        try:
            # networkidle이 너무 엄격할 수 있으므로 domcontentloaded로 변경하고 재시도 로직 추가
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=load_timeout * 1000)
            except (PlaywrightTimeoutError, Exception) as nav_exc:
                # 네비게이션 에러 발생 시 재시도 (about:blank 리다이렉트 등 처리)
                logging.warning("Navigation error for '%s' (retrying): %s", query, nav_exc)
                try:
                    page.wait_for_timeout(2000)
                    page.goto(target_url, wait_until="domcontentloaded", timeout=load_timeout * 1000)
                except Exception as retry_exc:
                    logging.error("Failed to load DuckDuckGo page for '%s' after retry: %s", query, retry_exc)
                    browser.close()
                    return unique_urls
            
            maybe_accept_consent(page)
            # 이미지가 로드될 시간을 줌
            page.wait_for_timeout(3000)
            # 초기 스크롤로 이미지 로드 유도
            page.mouse.wheel(0, 1000)
            page.wait_for_timeout(2000)
        except Exception as exc:
            # 예상치 못한 에러 처리
            logging.error("Unexpected error loading DuckDuckGo page for '%s': %s", query, exc)
            browser.close()
            return unique_urls

        stagnation = 0
        for scroll_idx in range(max_scrolls):
            # 스크롤 후 이미지가 로드될 시간을 줌
            if scroll_idx > 0:
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(scroll_pause * 1000)
            
            # JavaScript로 동적 클래스(랜덤 문자열)를 가진 div 안의 이미지 찾기
            # 예: SZ76bwIlqO8BBoqOLqYV 같은 클래스를 가진 div
            try:
                image_urls = page.evaluate("""
                    () => {
                        const divs = Array.from(document.querySelectorAll('div[class]'));
                        const imageUrls = [];
                        divs.forEach(div => {
                            const className = div.className || '';
                            // 긴 랜덤 문자열 클래스 (10자 이상, 알파벳/숫자만)를 가진 div 찾기
                            if (typeof className === 'string' && className.length > 10 && /^[A-Za-z0-9]+$/.test(className)) {
                                const imgs = div.querySelectorAll('img');
                                imgs.forEach(img => {
                                    const src = img.src || img.getAttribute('data-src') || img.getAttribute('data-srcset');
                                    if (src && !imageUrls.includes(src)) {
                                        imageUrls.push(src);
                                    }
                                });
                            }
                        });
                        return imageUrls;
                    }
                """)
                
                # JavaScript로 찾은 URL들을 정규화하고 추가
                if image_urls:
                    found_count = 0
                    for url_str in image_urls:
                        normalized = normalize_duckduckgo_src(url_str)
                        if normalized and normalized not in seen:
                            seen.add(normalized)
                            unique_urls.append(normalized)
                            found_count += 1
                            if len(unique_urls) >= max_results:
                                break
                    
                    if found_count > 0:
                        logging.debug("Found %d new image URLs (total: %d/%d)", found_count, len(unique_urls), max_results)
                    
                    if len(unique_urls) >= max_results:
                        break
            except Exception as e:
                logging.debug("Failed to find images in div containers: %s", e)
            
            # 더 이상 새로운 이미지를 찾지 못하면 중단
            if len(unique_urls) >= max_results:
                break
            
            previous_count = len(unique_urls)
            if len(unique_urls) == previous_count:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= 3:
                logging.debug("No new images after %s scrolls for '%s'; stopping.", scroll_idx + 1, query)
                break

        browser.close()

    logging.info("Collected %d candidate URLs for '%s'", len(unique_urls), query)
    return unique_urls[:max_results]


def crawl_class(
    class_name: str,
    safe_name: str,
    out_dir: Path,
    per_class_target: int,
    timeout: float,
    user_agent: str,
    delay: float,
    dry_run: bool,
    max_scrolls: int,
    scroll_pause: float,
    headless: bool,
    load_timeout: float,
) -> ClassSummary:
    class_dir = out_dir / safe_name
    ensure_dir(class_dir)

    urls = collect_image_urls(
        class_name,
        max_results=per_class_target * 4,
        max_scrolls=max_scrolls,
        scroll_pause=scroll_pause,
        headless=headless,
        load_timeout=load_timeout,
    )

    requested = per_class_target
    downloaded = 0
    errors = 0

    if dry_run:
        logging.info("[DRY-RUN] Would download up to %s images for '%s' (found %s URLs)", requested, class_name, len(urls))
        return ClassSummary(name=class_name, requested=requested, downloaded=0, errors=0, notes="dry-run", folder=safe_name)

    if not urls:
        logging.warning("No URLs collected for '%s'", class_name)
        return ClassSummary(name=class_name, requested=requested, downloaded=0, errors=0, notes="no_urls_collected", folder=safe_name)
    
    logging.info("Attempting to download %d images for '%s' (from %d URLs)", min(per_class_target, len(urls)), class_name, len(urls))
    
    for idx, url in enumerate(urls, start=1):
        filename = f"{safe_name}_{idx:04d}.jpg"
        dest_path = class_dir / filename
        if dest_path.exists():
            downloaded += 1
            logging.debug("Skipping existing file: %s", dest_path)
            continue
        success = download_image(url, dest_path, timeout=timeout, user_agent=user_agent)
        if success:
            downloaded += 1
            if downloaded % 10 == 0:
                logging.info("Downloaded %d/%d images for '%s'", downloaded, per_class_target, class_name)
        else:
            errors += 1
            if errors <= 10:  # 처음 몇 개 에러만 로깅
                logging.warning("Failed to download image %d/%d for '%s': %s", idx, len(urls), class_name, url[:100] if url else "None")
        if downloaded >= per_class_target:
            break
        if delay > 0:
            time.sleep(delay)

    notes = ""
    if downloaded < requested:
        notes = f"shortfall_{requested-downloaded}"
    return ClassSummary(name=class_name, requested=requested, downloaded=downloaded, errors=errors, notes=notes, folder=safe_name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.min_per_class > args.max_per_class:
        raise ValueError("--min_per_class cannot exceed --max_per_class")

    run_id = args.run_id or generate_run_id("crawl_pw")
    out_dir = (args.out / run_id).resolve()
    ensure_dir(out_dir)

    logging.info("Run ID: %s", run_id)
    logging.info("Output directory: %s", out_dir)

    selected_classes = args.classes or load_classes(args.csv, start_from=args.start_from)
    resolved_classes = resolve_safe_class_names(selected_classes)
    logging.info("Total classes available: %d", len(resolved_classes))
    if args.start_from:
        logging.info("Starting from class: %s", args.start_from)

    global_limit = args.limit
    total_downloaded = 0
    summaries: List[ClassSummary] = []

    for class_name, safe_name in resolved_classes:
        remaining = None
        if global_limit is not None:
            remaining = max(global_limit - total_downloaded, 0)
            if remaining <= 0:
                logging.info("Global limit reached. Stopping.")
                break
        per_class_target = args.max_per_class
        if remaining is not None:
            per_class_target = min(per_class_target, remaining)
        if per_class_target < args.min_per_class:
            per_class_target = remaining if remaining is not None else args.min_per_class

        summary = crawl_class(
            class_name=class_name,
            safe_name=safe_name,
            out_dir=out_dir,
            per_class_target=per_class_target,
            timeout=args.timeout,
            user_agent=args.user_agent,
            delay=args.delay,
            dry_run=args.dry_run,
            max_scrolls=args.max_scrolls,
            scroll_pause=args.scroll_pause,
            headless=not args.show_browser,
            load_timeout=args.load_timeout,
        )
        summaries.append(summary)
        total_downloaded += summary.downloaded

    log_path = save_run_log(run_id, summaries, sources=["duckduckgo_images_playwright"])
    logging.info("Saved crawl log to %s", log_path)

    if args.dry_run:
        logging.info("Dry-run completed. No files downloaded.")


if __name__ == "__main__":
    main()
