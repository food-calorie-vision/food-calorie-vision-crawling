# Project TODO

Tracked tasks for Food Calorie Vision Crawling. Keep this file current so every agent/user handoff stays aligned.

---

## Agent TODO (English)

### Phase 0 – Foundations
- [x] Pin Python 3.10 baseline, create `requirements.txt`, and document the install command.
- [x] Add `.gitignore` entries for `data/raw`, `data/filtered`, `labels`, and `models` to block large asset commits.
- [x] Create the `data/source_list.md` template (source, license, usage scope, notes).
- [x] Stub or verify `configs/food_poc.yaml` and describe its fields in the README.

### Phase 1 – Data Acquisition
- [x] DuckDuckGo API 기반 `scripts/crawl_images.py`는 호출 차단으로 인해 폐기됐으며 Playwright 버전을 단일 경로로 유지한다.
- [x] Emit crawl logs to `data/meta/crawl_logs/<run_id>.json`.
- [x] Add sampling/dry-run mode (`--limit`) for safer tests.
- [x] Document the run_id naming convention in AGENTS/README.
- [x] Harden DuckDuckGo crawlers (Playwright selector fallback, srcset/protocol normalization, min/max enforcement & docs).
- [x] Implement `scripts/crawl_images_playwright.py` with consent handling, image selector fallback, and scroll-based loading.
  - Note: `--show-browser` option recommended due to headless mode image loading limitations.
- [x] **완료 (2025-11-07)**: DuckDuckGo 동적 클래스 탐색 구현
  - JavaScript로 긴 랜덤 문자열 클래스(예: SZ76bwIlqO8BBoqOLqYV)를 가진 div에서 이미지 URL 추출
  - 불필요한 CSS 셀렉터 시도문 제거, 코드 간소화
- [x] **완료 (2025-11-07)**: 아이콘 파일 필터링 추가
  - URL 레벨 필터링: `.ico`, `favicon`, `/icon` 제외
  - 다운로드 후 이미지 크기 검증: 50x50 이하 자동 삭제
  - `scripts/remove_icons.py` 스크립트 추가 (기존 다운로드된 아이콘 정리용)
- [x] **완료 (2025-11-07)**: 한글 클래스명 폴더 생성 문제 수정
  - `sanitize_name` 함수 개선: 한글 클래스명을 그대로 유지하도록 수정
  - 여러 클래스를 입력해도 각각 별도 폴더에 저장되도록 수정
  - 파일시스템 안전성: 한글, 영문, 숫자, 언더스코어, 하이픈만 허용

### Phase 2 – Cleaning & Labeling
- [x] Implement `scripts/dedup_filter.py` (현재 버전: sha256 기반 완전 중복 제거 + 복사/보고서 생성).
- [x] Produce the `stats.yaml` output format plus a sample 파일(`data/filtered/crawl_test_b/stats.yaml`).
- [x] Implement `scripts/auto_label.py` with YOLO weights, confidence thresholds, per-detection CSV(`labels/meta/<run_id>_predictions.csv`), and review queue export(`labels/meta/review_queue.csv`).
- [x] Document the Label Studio/CVAT import-export workflow (README `Label Studio` 섹션 + 인터페이스 템플릿/Local Files 가이드).
- [x] Add `scripts/visualize_predictions.py` for 상·하위 신뢰도 샘플 시각화.
- [x] Add `scripts/filter_labels.py` to keep/remap reviewed classes (post-GUI cleanup).

### Phase 3 – Training & Active Learning
- [ ] Implement `scripts/train_yolo.py` to read dataset configs and store metrics/checkpoints.
- [ ] Define evaluation criteria (mAP, precision/recall) and the logging format.
- [ ] Add an inference/uncertainty-scoring path (script or YOLO CLI instructions) with documented thresholds.
- [ ] Automate the Active Learning loop orchestration (script or detailed instructions).

### Phase 4 – Ops & QA
- [ ] Add automated lint/test coverage (e.g., `pytest`, `ruff`, formatting) to keep scripts stable.
- [ ] Provide lightweight sample notebooks for analysis/visualization.
- [ ] Prepare deployment/infra notes (Dockerfile, GPU node requirements, storage layout).
- [ ] Establish a release checklist (data snapshot, model registry update, documentation refresh).

---

## 사용자 TODO (Korean)

### 환경 및 준비
- [x] CUDA 가용성과 버전을 확인하고 GPU 0이 정상 인식되는지 점검한다.
- [ ] 현재 작업 목표(크롤 대상 클래스, 최소/최대 장 수)를 정의하고 `YYYYMMDD_stage_seq` 형식의 run_id를 확정한다.

### 데이터 파이프라인 실행
- [x] 정한 run_id로 `scripts/crawl_images_playwright.py`를 `--limit` 또는 `--dry-run` 옵션과 함께 실행해 설정을 검증한다.
- [ ] 본 크롤을 수행하고 결과를 `data/raw/<run_id>`에 저장하며 `data/meta/crawl_logs/`와 `data/source_list.md`에 요약을 남긴다.
- [x] DuckDuckGo rate limit이 반복되면 `playwright install chromium` 및 시스템 의존성 설치 후 `scripts/crawl_images_playwright.py`로 동일한 run_id를 수집해 본다(`--show-browser` 옵션 권장, headless 모드 제한 있음).
- [x] **완료 (2025-11-07)**: Playwright 크롤러 개선 및 테스트
  - 동적 클래스 탐색으로 이미지 URL 수집 성공 (run_id: `20251107_test_div_selector`)
  - 아이콘 필터링으로 품질 개선 (32x32 아이콘 자동 제거)
  - 코드 정리 완료 (불필요한 시도문 제거)
- [ ] `20251107_set_1` 런은 sanitize 버그로 단일 `class/` 폴더에 저장되어 정리됨. 고정된 스크립트로 동일 클래스 세트를 재수집한다.
- [ ] (스크립트가 준비되는 대로) `scripts/dedup_filter.py`와 `scripts/auto_label.py`를 순차 실행하고 사용한 명령/파라미터를 기록한다.
- [ ] Label Studio 혹은 CVAT에서 `food-<run_id>` 프로젝트를 만들어 저신뢰 샘플을 검수하고 `labels/yolo_validated/<run_id>`로 내보낸다.
- [ ] 각 단계에서 발견한 이슈, 메트릭, 후속 조치를 README 또는 작업 노트에 정리한다.
