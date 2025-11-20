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
### 3.0 Class Map Management & Runner
- `target_food.csv`는 수집 우선순위를, `food_class_pre_label.csv`/`food_class_after_label.csv`는 **공식 클래스 매핑**을 정의합니다. 두 CSV는 외부 파이프라인에서 공급되므로 재생성하지 말고 최신본을 교체할 때만 TODO/노트에 출처를 남깁니다.
- `food_class_pre_label.csv`(`id,classes`)는 포스트프로세싱·Label Studio export 단계의 기준입니다. `postprocess_labels.py`, `export_labelstudio_json.py`, `prepare_food_dataset.py` 모두 이 파일을 참조합니다.
- `food_class_after_label.csv`는 검수 이후 학습용 상위 카테고리 매핑입니다. 실험용 CSV는 별도로 만들어 사용하되 원본은 그대로 유지합니다.
- 다단계 스크립트를 한 번에 실행하려면 `python scripts/pipeline_runner.py --run-id <run_id>`를 사용하세요. `--steps`로 부분 실행을 지정할 수 있고, `--interactive` 모드에서는 `/set`, `/run`, `/help` 명령으로 `COMMANDS.md`에 정의된 코드 블록을 즉시 실행할 수 있습니다. 민감한 환경 변수(`LABEL_STUDIO_TOKEN`)는 프롬프트 입력으로만 처리합니다.

### 3.1 Crawling (`scripts/crawl_images_playwright.py`)
- run_id를 먼저 정한 뒤 `--out data/1_raw/<run_id>`를 명시해 기본 폴더 구조를 강제합니다(스크립트 기본값은 `data/raw`이므로 반드시 덮어쓰세요).
- 클래스 목록은 `--classes` 또는 `--csv target_food.csv` 중 하나로 전달합니다. `--start-from`으로 CSV 내 특정 클래스부터 재개할 수 있습니다.
- DuckDuckGo는 이미지 탭을 JavaScript로 렌더링하므로 `--show-browser`와 Chromium `--no-sandbox` 조합을 기본으로 하세요. DOM이 뜨지 않는 환경(WSL/리모트 서버)은 headless에서 `<img>` 태그가 아예 생성되지 않습니다.
- 내부 로직은 동적 div 클래스 탐색 + URL 정규화를 수행하며, `.ico`/`favicon`/`/icon` URL 및 50x50 이하 이미지는 자동으로 제거합니다. 추가 정리는 `scripts/remove_icons.py`로 수행합니다.
- 클래스명은 `sanitize_name()`으로 파일시스템 안전 문자열로 변환하며, 충돌 시 `-2`, `-3` 접미사를 붙여 덮어쓰기를 방지합니다.
- 각 실행은 `data/meta/crawl_logs/<run_id>.json`에 기록됩니다. 같은 run_id로 다시 실행하면 `history`에 시도별 로그가 append되고 `classes` 요약은 병합됩니다.
- 필수 설치: `pip install -r requirements.txt`, `playwright install chromium`, 필요 시 `sudo playwright install-deps`. 테스트 시에는 `--limit` 또는 `--dry-run`으로 대량 다운로드를 막습니다.

### 3.2 Deduplication & Filtering (`remove_icons.py`, `dedup_filter.py`)
- 아이콘 정리 이후 `scripts/dedup_filter.py --input data/1_raw/<run_id> --output data/2_filtered/<run_id>`를 실행해 sha256 완전 중복만 제거합니다.
- 결과는 `data/2_filtered/<run_id>/stats.yaml`에 클래스별 input/kept/dropped와 중복 목록으로 남기고, PR/노트에 반드시 첨부합니다.
- `--dry-run`을 사용하면 복사 없이 통계만 확인할 수 있습니다. run_id는 입력 폴더명에서 자동 추론되지만 명시하는 편이 안전합니다.

### 3.3 Auto Labeling & QC (`auto_label.py`, `postprocess_labels.py`, `visualize_predictions.py`, `filter_labels.py`)
- 입력: `--images data/2_filtered/<run_id>`, 기본 가중치는 `models/yolo11l.pt`입니다. 파일이 없으면 에러 메시지가 안내하므로 사전에 다운로드하세요.
- 출력: `labels/3-1_yolo_auto/<run_id>`에 YOLO txt, `labels/3-2_meta/<run_id>_predictions.csv`에 per-detection CSV, 필요 시 `labels/3-2_meta/review_queue.csv`에 저신뢰 샘플을 append합니다(`--review-threshold`).
- `visualize_predictions.py`로 top/bottom confidence 이미지를 `labels/3-3_viz/<run_id>/`에 렌더링해 검수 전에 샘플을 공유하세요.
- **후처리**: `scripts/postprocess_labels.py --labels labels/3-1_yolo_auto/<run_id> --label-map food_class_pre_label.csv`로 각 이미지당 가장 넓은 박스만 남기고, 폴더명 기반 ID를 `food_class_pre_label.csv`의 공식 ID로 덮어씁니다.
- 특정 클래스만 남기거나 COCO 0~79 ID를 제거하려면 `scripts/filter_labels.py`를 사용합니다 (`--keep-ids`, `--drop-below`, `--shift-offset`, `--remap-id`).

### 3.4 Human Validation & Label Studio Tooling
- 프로젝트 명명 규칙: `food-<run_id>`. Label Studio를 구동할 때는 `scripts/run_label_studio.sh`를 사용하면 `/workspace` 마운트와 Local Files 환경 변수가 자동으로 세팅됩니다.
- `/run 8` (`bootstrap_label_studio.py`)는 `data/2_filtered/<run_id>` 또는 fallback 폴더에서 이미지를 읽어 Label Studio 프로젝트 생성, Local Files 스토리지 등록, 기본 라벨링 템플릿 적용까지 자동화합니다.
- Label Studio에서 수동 검수가 끝나면 `/run 9` (`export_from_studio.py`)로 YOLO export zip을 받아 `labels/4_export_from_studio/<run_id>/source`에 stages/manifest를 정리합니다. `--skip-dataset`가 기본이라 데이터셋은 건드리지 않습니다.
- 간단히 리뷰용 샘플을 만들고 싶다면 `scripts/package_label_data.py`로 `data/3_exports/<run_id>`를 생성해 Local Files 스토리지에 동기화합니다.

### 3.5 Packaging (`prepare_food_dataset.py`, `plan_train_subset.py`)
- `/run 10`은 항상 `prepare_food_dataset.py`를 호출해 Label Studio export → `data/5_datasets/<run_id>` 데이터셋 생성까지 처리합니다. 핵심 단계는 다음과 같습니다.
  1. Label Studio `classes.txt` → `food_class_pre_label.csv` → `food_class_after_label.csv` 순으로 ID/이름을 재매핑.
  2. labels/ 폴더 구조를 유지한 채 YOLO txt를 재작성하고, 이미지 파일은 copy/hardlink 모드로 `data/5_datasets/<run_id>/images`에 정렬.
  3. 필요 시 `--val-list`, `--train-include`, `--train-exclude`를 적용해 검증 세트를 고정하고 train manifest를 제어.
  4. `classes.txt`, `train.txt`, `val.txt`, `<run_id>.yaml`까지 완비된 상태로 출력.
- `/run 10` 직후 `scripts/plan_train_subset.py --run-id <run_id>`를 호출해 `data/meta/<run_id>_exclude.txt`를 생성합니다. 이 manifest는 Active Learning 라운드에서 `--train-exclude`로 재사용하며, `--update-train`을 지정하지 않는 한 `train.txt`는 그대로 유지됩니다.

### 3.6 Training & Active Learning (`train_yolo.py`, `report_training_metrics.py`)
- `train_yolo.py`는 `configs/food_poc.yaml`을 기본으로 하며, `--run-id`를 넘기면 `data/5_datasets/<run_id>/<run_id>.yaml`을 자동으로 찾습니다. 다른 데이터셋을 학습하려면 `--data` 인자를 직접 지정하세요.
- 실행 후 결과는 `runs/train/<name>`과 `models/runs/metrics.json`에 저장되고, `data/meta/train_metrics/<run_id>_{per_class,confusion_matrix}.csv`가 자동 생성됩니다. GPU 메모리는 스크립트가 종료 시 비워 줍니다.
- `/run 12` (`report_training_metrics.py`)는 위 CSV를 정렬·필터링하여 취약 클래스를 빠르게 확인하고, Active Learning 우선순위를 정할 때 사용합니다.
- Active Learning run_id는 `YYYYMMDD_al_rX` 규칙을 따릅니다. `/set base_run_id <초기값>` 후 `/run 13`(AL packaging)을 실행하면 기존 `val.txt`를 그대로 유지한 채 train manifest만 갱신합니다. 동일한 train 구성이 감지되면 명시적으로 확인을 받아 중복 생성을 방지합니다.
- 재학습 시 `/run 11`은 run_id에 따라 이전 라운드의 `best.pt`를 자동으로 찾으려 시도하므로, weights 경로를 바꾸고 싶으면 `--model`을 명시하세요.

### 3.7 Runner Etiquette
- Pipeline Runner는 QA/데모용으로 여러 단계를 한 번에 돌려야 할 때만 사용합니다. Active Learning 실무에서는 크롤→검수 타이밍이 다르므로 `/run <번호>` 단위로 관리하세요.
- `/run` 명령이 실패하면 해당 단계만 중지되고 세션은 유지됩니다. 재시도 전에는 run_id, classes_flag, base_run_id 등 컨텍스트 값을 `/context`로 확인하세요.
- 모든 run_id, 명령, 주요 옵션은 `TODO.md` 또는 PR/노트에 남겨 traceability를 확보합니다.

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
| `food_class_pre_label.csv` | `target_food.csv` 기반 YOLO/Label Studio용 ID↔이름 매핑 (자동 생성) |
| `food_class_after_label.csv` | 검수 후 학습 카테고리 매핑 |
| `data/source_list.md` | (Create/update) provenance and licensing log |
| `data/meta/train_metrics/` | 학습 종료 후 자동 저장되는 per-class/혼동행렬 CSV (`report_training_metrics.py` 참조) |
| `scripts/*.py` | Automation entry points (crawl, filter, label, export, train) |
| `scripts/plan_train_subset.py` | 클래스별 train 제외량을 인터랙티브로 선택해 manifest 생성 |
| `scripts/report_training_metrics.py` | 학습 결과 기반 클래스/혼동행렬 리포트 |
| (미사용) `scripts/analyze_class_performance.py` | 과거 자동 라벨 기반 분석 스크립트 (현재 비활성) |
| `python scripts/pipeline_runner.py --interactive` | (QA/데모 용도) `/set run_id`, `/help`, `/run <번호>`로 COMMANDS를 배치 실행 |

Stay within these guardrails to keep every agent handoff predictable and auditable.

---

## 7. Troubleshooting Learnings
- **Class ID and File Path Issues (2025-11-20):** A series of issues were found during the data packaging pipeline (`/run 10`).
  - **Initial Problem:** Class ID mismatch between Label Studio export and project master files (`food_class_pre_label.csv`), causing incorrect label mapping (e.g., "milk" mapping to "kimbap_category").
  - **Fix 1:** The `prepare_food_dataset.py` script was refactored to perform a name-based, multi-step re-mapping instead of relying on numeric IDs.
  - **Additional Finding 1:** The refactored script failed to find any images (`Copied 0 images`). This was because the image search logic was over-simplified and did not recursively search the class-based subdirectories in `data/2_filtered/`.
  - **Fix 2:** The image search logic was corrected to use `rglob` for a recursive search.
  - **Additional Finding 2:** The subsequent script, `plan_train_subset.py`, failed to identify any classes (`class_unknown`). This was due to a bug in how it constructed the label file path from the image file path (e.g., trying to find `image.jpg.txt` instead of `image.txt`).
  - **Fix 3:** The path manipulation logic in `plan_train_subset.py` was corrected.
  - **Learning:** Data pipelines are sensitive to both identifiers (like class IDs) and file paths. When refactoring, it's crucial to verify assumptions about directory structures and the logic used by downstream scripts.
