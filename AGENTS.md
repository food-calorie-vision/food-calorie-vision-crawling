# Agent Operations Guide

LLM/agent collaborators use this guide together with `README.md` to keep work on the Food Calorie Vision Crawling project consistent and auditable.

---

## 1. Mission and Context
- Goal: build an automated pipeline that crawls prioritized food images, applies YOLO auto-labeling, performs human-in-the-loop validation, and continually retrains via Active Learning.
- Scope priorities come from `target_food.csv`; always honor its ordering when selecting classes.
- When in doubt, defer to `README.md` for domain intent and use this document for execution standards.

---

## 2. Operating Principles
1. **Traceability first** - log every script invocation, arguments, and data outputs in task notes or PR descriptions.
2. **Deterministic workspaces** - avoid ad-hoc manual tweaks; prefer idempotent scripts under `scripts/`.
3. **UTF-8 everywhere** - store mixed-language documents in UTF-8 and verify encoding in editors.
4. **Data hygiene** - never check raw images or licensed assets into git; keep only metadata, configs, or approved thumbnails.
5. **Fail fast** - validate assumptions with dry-runs (`--limit`, `--debug`) before launching long crawls or training.
6. **Consistent run_id** - name every run `YYYYMMDD_<stage>_<seq>` (e.g., `20250107_poc_a`) and reuse that identifier across folders/logs.

---

## 3. Primary Workflows
### 3.1 Data Crawling (`scripts/crawl_images_playwright.py`)
- Read `target_food.csv`, select classes according to priority columns.
- Generate a run_id before starting and reuse it for all outputs (`data/raw/<run_id>`, logs, labels).
- Mandatory arguments: `--classes`, `--min_per_class`, `--max_per_class`, `--out data/raw/<run_id>`.
- Respect robots.txt and add throttle or delay flags when available.
- After each run, write a summary (`run_id`, class counts, error URLs) to `data/meta/crawl_logs/<run_id>.json`.
- 동일한 run_id를 재실행하면 기존 JSON에 `history` 항목이 append되고 `classes` 요약이 병합되므로 데이터 손실 없이 누적할 수 있습니다.
- Use `--limit` or `--dry-run` during testing to avoid large downloads and confirm CLI arguments.
- Default source는 Playwright로 렌더링한 DuckDuckGo 이미지(`duckduckgo_images_playwright`); 다른 소스를 쓰면 `requirements.txt`와 문서를 업데이트하세요.
- `--min_per_class`는 실제 다운로드 목표 하한으로 사용되며, 전역 limit 때문에 부족할 경우 `notes`에 shortfall이 표시됩니다.
- **클래스별 폴더 생성**: 각 클래스는 `data/raw/<run_id>/<sanitized_class_name>/` 폴더에 저장됩니다. 한글 클래스명은 그대로 유지되며, 파일시스템에 안전하지 않은 문자만 제거됩니다.
- 동일한 sanitize 결과가 충돌하는 경우 자동으로 `-2`, `-3` 접미사가 붙어 덮어쓰기를 방지합니다. 로그의 `folder` 필드를 참고하세요.
- 설치: `pip install -r requirements.txt && playwright install chromium`, 필요 시 `sudo playwright install-deps`.
- **Important**: DuckDuckGo 이미지 탭 DOM이 JavaScript로만 생성되기 때문에 headless 모드(WSL, 무헤드 서버 포함)에서는 `<img>` 태그가 로드되지 않을 수 있습니다. 항상 `--show-browser` 옵션을 켜고 실제 브라우저를 띄운 상태에서 스크롤을 수행하며, Chromium은 `--no-sandbox` 플래그와 함께 실행합니다.
- DuckDuckGo는 동적으로 생성된 긴 랜덤 문자열 클래스(예: `SZ76bwIlqO8BBoqOLqYV`)를 가진 div 안에 이미지를 배치합니다.
- Playwright 크롤러는 JavaScript로 이러한 동적 클래스를 가진 div를 찾아 이미지 URL을 추출합니다.
- Playwright 크롤러는 동의 다이얼로그 자동 처리(`maybe_accept_consent`), 초기 스크롤로 이미지 로딩 유도, 아이콘 필터링을 포함합니다.
- 아이콘 필터링: URL 레벨에서 `.ico`, `favicon`, `/icon` 제외, 다운로드 후 50x50 이하 이미지 자동 삭제

### 3.2 Deduplication and Filtering (`scripts/dedup_filter.py`)
- Inputs: previously crawled folder, outputs to `data/filtered/<run_id>`.
- Required checks: hash duplicates, perceptual similarity, broken/corrupt files, minimum resolution.
- Emit `stats.yaml` capturing files kept/dropped plus reasons; link it in PR or task notes.

### 3.3 Auto Labeling (`scripts/auto_label.py`)
- Use the current best checkpoint under `models/`.
- Store predictions under `labels/yolo/<run_id>` with filenames matching `data/filtered`.
- Export per-image confidences; mark low-confidence samples (`< threshold`) for manual review in `labels/meta/review_queue.csv`.

### 3.4 Human Validation (Label Studio or CVAT)
- Sync auto-labeled data into the chosen tool using project naming convention `food-<run_id>`.
- After validation, export YOLO-format labels back into `labels/yolo_validated/<run_id>` and document reviewer notes.

### 3.5 Training and Active Learning (`scripts/train_yolo.py`)
- Configs live under `configs/`; clone `configs/food_poc.yaml` when creating new datasets.
- Training artifacts belong in `models/<date>_<run_id>.pt` plus `metrics.json`.
- Active Learning loop:
  1. Run inference on the unlabeled pool.
  2. Rank by uncertainty (entropy or low confidence).
  3. Push top-N images back to the validation queue.
  4. Repeat crawl -> filter -> label -> train steps.

---

## 4. Task Execution Checklist
1. Confirm the latest `README.md` and this file before starting.
2. Define deliverables (e.g., "crawl 5 classes x 100 images").
3. Create or update necessary configs (datasets, thresholds, model paths).
4. Run scripts with explicit args; capture stdout and metrics.
5. Update documentation/logs (README sections, `data/source_list.md`, run summaries).
6. If adding code, include brief comments for non-obvious logic and provide usage examples.
7. Validate outputs (spot-check images, run `yolo val`, ensure counts match expectations).

---

## 5. Repository Etiquette
- Honor existing `.gitignore`; never add large binaries unless approved.
- Keep Markdown files concise but informative; tables for metrics, bullet lists for steps.
- Prefer `apply_patch` or similarly structured edits; avoid overwriting files with unrelated content.
- Before any PR or task completion, describe:
  - Scripts used plus key flags
  - Data ranges or classes affected
  - Metrics or validation evidence
  - Follow-up actions required

---

## 6. Quick Reference
| Asset | Purpose |
|-------|---------|
| `README.md` | Product-level overview, pipeline summary, roadmap |
| `AGENTS.md` | (This file) execution standards for autonomous agents |
| `target_food.csv` | Prioritized list of food classes |
| `data/source_list.md` | (Create/update) provenance and licensing log |
| `scripts/*.py` | Automation entry points (crawl, filter, label, export, train) |

Stay within these guardrails to keep every agent handoff predictable and auditable.
