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
### 3.0 Class Map Management
- `target_food.csv`가 클래스 우선순위의 소스입니다. 내용이 바뀌면 아래 명령으로 `food_class.csv`를 재생성하고, 실행 로그에 사용한 커맨드를 적어 둡니다.
  ```bash
  UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
  from pathlib import Path
  src = Path("target_food.csv")
  lines = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
  dst = Path("food_class.csv")
  with dst.open("w", encoding="utf-8") as fp:
      fp.write("id,classes\n")
      for idx, name in enumerate(lines):
          fp.write(f"{idx},{name}\n")
  print(f"Updated {dst} with {len(lines)} rows.")
  PY
  ```
- `food_class.csv`는 `id,classes` 헤더를 가진 공식 라벨 매핑 테이블입니다. **모든** 라벨 패키징(`prepare_food_dataset.py`), 검수 export(`export_labelstudio_json.py`)는 이 파일을 참조합니다.
- 추가 서브셋(예: 상위 N개 클래스만)이나 실험용 매핑이 필요하면 별도 CSV를 만들되, 원본 `food_class.csv`를 직접 수정하지 말고 PR/노트에 생성 명령을 기록하십시오.

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
- Inputs: `--input data/raw/<run_id>` (아이콘 정리 이후), outputs to `data/filtered/<run_id>`.
- v1 기준으로 sha256 해시가 완전히 동일한 파일만 제거합니다. 퍼셉추얼 비교나 해상도 필터는 후속 버전에서 확장하세요.
- 실행 시 `data/filtered/<run_id>/stats.yaml`이 생성되며 클래스별 input/kept/dropped 집계와 duplicate 목록을 포함하므로 PR/노트에 반드시 첨부합니다.
- `--dry-run`으로 실제 복사 없이 중복 여부만 확인할 수 있습니다.

### 3.3 Auto Labeling (`scripts/auto_label.py`)
- Inputs: `--images data/filtered/<run_id>`, `--weights` 기본값은 `models/yolo11l.pt`. 파일이 없으면 스크립트가 다운로드 안내를 출력하므로 가중치를 준비한 뒤 재실행합니다.
- Outputs: YOLO txt 파일을 `labels/yolo/<run_id>/<relative_path>.txt`에 생성하고, per-detection 정보를 `labels/meta/<run_id>_predictions.csv`로 기록합니다.
- `--review-threshold`보다 낮은 confidence만 검출되었거나 검출이 없는 이미지는 `labels/meta/review_queue.csv`에 run_id/이유/확신도를 append하여 반자동 검수 대기열을 유지합니다.
- 검수 전 샘플 이미지를 확인하려면 `python scripts/visualize_predictions.py --run-id <run_id> --top-n 2 --bottom-n 3`으로 `labels/viz/<run_id>/`에 시각화를 생성합니다.
- 특정 클래스 ID만 남기거나 COCO 기본 클래스(0~79)를 제거하고 싶다면 `scripts/filter_labels.py`를 사용하세요.
  - 예시 1) `--keep-ids 45 --remap-id 0` : 45번만 남기고 0으로 재매핑.
  - 예시 2) `--drop-below 80 --shift-offset 80` : 80번 이상의 음식 클래스만 남기고 80을 빼서 0~N-1로 재정렬.
- `--confidence`, `--iou`, `--batch-size`, `--device`를 상황에 따라 조정하고, 사용한 가중치·옵션을 TODO/노트에 남깁니다.

### 3.4 Human Validation (Label Studio or CVAT)
- Sync auto-labeled data into the chosen tool using project naming convention `food-<run_id>`.
- Label Studio에서 Local Files를 쓸 때는 컨테이너 환경 변수 `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`,
  `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/local_storage`를 반드시 설정하고,
  Storage 경로는 `/label-studio/data/local_storage/<run_id>/images` 형태로 입력합니다.
- Labeling Interface는 `<Image name="image" value="$image"/>` + `<RectangleLabels name="label" toName="image">` 구조를 기본으로 사용합니다.
- JSON pre-annotation의 `data.image` 경로는 `/data/local-files/?d=<run_id>/images/...` 형태여야 하며,
  Label Studio에서 샘플을 수동 라벨링 후 Export하여 구조를 검증한 뒤 가져오십시오.
- After validation, export YOLO-format labels back into `labels/yolo_validated/<run_id>` (또는 새 폴더) and document reviewer notes.
- 검수 결과를 학습용으로 정리하려면 `python scripts/prepare_food_dataset.py --run-id <run_id> --source labels/yolo_validated/<run_id> --label-map food_class.csv --val-ratio 0.2 --overwrite`로
  이미지/라벨 복사 + COCO ID 제거 + train/val split 리스트 + data/datasets/<run_id> 구성까지 자동화할 수 있습니다.

### 3.5 Training and Active Learning (`scripts/train_yolo.py`)
- 학습 파라미터는 `configs/food_poc.yaml`을 기본으로 하며, `dataset.dataset_yaml`에 `{run_id}` 플레이스홀더가 있으므로 `--run-id <run_id>`를 반드시 넘깁니다.
- 실행 예시:
  ```
  UV_CACHE_DIR=.uv-cache uv run python scripts/train_yolo.py \
    --run-id crawl_test_b \
    --config configs/food_poc.yaml \
    --model models/yolo11l.pt \
    --device 0
  ```
  필요 시 `--data data/datasets/<run_id>/<run_id>.yaml`, `--epochs`, `--batch-size`, `--resume` 등을 덮어쓸 수 있습니다.
- 학습 산출물은 기본적으로 `runs/train/<name>` 아래에 저장되고, `logging.metrics_file` 경로(기본: `models/runs/metrics.json`)에도 지표가 기록됩니다. run_id, config, 주요 하이퍼파라미터를 TODO/노트에 남깁니다.
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
| `food_class.csv` | `target_food.csv` 기반 공식 ID↔이름 매핑 (자동 생성) |
| `data/source_list.md` | (Create/update) provenance and licensing log |
| `scripts/*.py` | Automation entry points (crawl, filter, label, export, train) |

Stay within these guardrails to keep every agent handoff predictable and auditable.
