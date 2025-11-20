# Food Calorie Vision Crawling

운영자는 이 저장소의 파이프라인을 이용해 특정 run_id에 대해 “이미지 수집 → 중복 제거 → 자동 라벨링 → Label Studio 검수 → 데이터셋 패키징 → YOLO 학습 → Active Learning” 순서를 반복 실행할 수 있습니다. 아래 설명은 **최소 설정만으로 전체 파이프라인을 실행**하려는 사용자를 위한 가이드입니다.

---

## 1. 빠른 시작

| 단계 | 설명 |
|------|------|
| 1. 환경 준비 | Python 3.10 이상, CUDA GPU(선택), Playwright/YOLO 가중치 설치 |
| 2. run_id 결정 | `YYYYMMDD_stage_seq` 규칙 (예: `20250301_poc_a`) |
| 3. 파이프라인 실행 | 섹션 3의 단계별 명령을 순서대로 수행 또는 `pipeline_runner.py` 사용 |
| 4. 결과 검증 | Label Studio/metrics 확인 후 Active Learning 루프 반복 |

### 필수 의존성 설치
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # DuckDuckGo 렌더링용

# YOLO 가중치 준비 (예: yolo11l)
curl -L -o models/yolo11l.pt https://github.com/ultralytics/assets/releases/download/v8/yolo11l.pt
```

### 기본 폴더 규칙
- **run_id**: 모든 산출물의 기준 경로 (`data/1_raw/<run_id>`, `labels/3-1_yolo_auto/<run_id>` 등)로 재사용합니다.
- **클래스 우선순위**: `target_food.csv` 순서를 따라야 합니다.
- **공식 매핑 파일**: `food_class_pre_label.csv`(검수 전)과 `food_class_after_label.csv`(검수 후)는 외부 파이프라인이 관리하므로 수정하지 마십시오.

---

## 2. 워크플로 개요

```
1. Crawl images              (scripts/crawl_images_playwright.py)
2. Remove icons + dedup      (scripts/remove_icons.py, scripts/dedup_filter.py)
3. Auto-label + visualize    (scripts/auto_label.py, scripts/postprocess_labels.py, scripts/visualize_predictions.py)
4. Label Studio validation   (scripts/run_label_studio.sh, scripts/bootstrap_label_studio.py)
5. Export + package dataset  (scripts/export_from_studio.py, scripts/prepare_food_dataset.py)
6. Plan train subset         (scripts/plan_train_subset.py)
7. Train YOLO + report       (scripts/train_yolo.py, scripts/report_training_metrics.py)
8. Active Learning           (scripts/prepare_food_dataset.py --val-list, /run 13)
```

각 단계를 수동으로 실행해도 되고, `python scripts/pipeline_runner.py --interactive`로 `/run <번호>` 명령을 호출해 COMMANDS.md에 정의된 스크립트를 자동으로 실행할 수도 있습니다.

---

## 3. 단계별 실행 가이드

### 3.1 run_id 결정
1. `YYYYMMDD_<stage>_<seq>` 규칙으로 run_id를 선택합니다 (예: `20250301_set_a`).
2. 동일 run_id를 Label Studio 프로젝트 이름(`food-<run_id>`), 폴더, 로그 파일 등에 반복 사용합니다.

### 3.2 이미지 크롤링
```bash
python scripts/crawl_images_playwright.py \
  --run-id 20250301_set_a \
  --csv target_food.csv \
  --min-per-class 80 --max-per-class 120 \
  --out data/1_raw/20250301_set_a \
  --show-browser --delay 1.5
```
- DuckDuckGo는 JavaScript로 이미지를 렌더링하므로 `--show-browser`(Chromium + `--no-sandbox`) 옵션을 기본으로 사용합니다.
- 결과 요약은 `data/meta/crawl_logs/<run_id>.json`에 기록됩니다. 파일을 열어 클래스별 shortfall/에러 URL을 확인하세요.

### 3.3 아이콘 제거 및 중복 제거
```bash
python scripts/remove_icons.py data/1_raw/20250301_set_a
python scripts/dedup_filter.py \
  --input data/1_raw/20250301_set_a \
  --output data/2_filtered/20250301_set_a
```
- `dedup_filter.py`는 sha256 완전 중복만 제거하며, 요약을 `data/2_filtered/<run_id>/stats.yaml`에 남깁니다.

### 3.4 자동 라벨링 + 검수 준비
```bash
python scripts/auto_label.py \
  --images data/2_filtered/20250301_set_a \
  --out labels/3-1_yolo_auto/20250301_set_a \
  --weights models/yolo11l.pt \
  --review-threshold 0.3 --device 0

python scripts/postprocess_labels.py \
  --labels labels/3-1_yolo_auto/20250301_set_a \
  --label-map food_class_pre_label.csv

python scripts/visualize_predictions.py \
  --run-id 20250301_set_a --top-n 2 --bottom-n 3
```
- `auto_label.py`는 per-detection CSV(`labels/3-2_meta/<run_id>_predictions.csv`)와 review queue(`labels/3-2_meta/review_queue.csv`)를 생성합니다.
- `postprocess_labels.py`는 이미지당 가장 넓은 박스만 남기고 폴더명을 `food_class_pre_label.csv`의 공식 ID로 매핑합니다.
- `visualize_predictions.py` 결과(`labels/3-3_viz/<run_id>/`)로 검수용 샘플을 빠르게 확인합니다.

### 3.5 Label Studio 기동 및 검수
1. 서버 실행
   ```bash
   bash scripts/run_label_studio.sh
   ```
2. 프로젝트 및 스토리지 자동 구성 (선택)
   ```bash
   python scripts/bootstrap_label_studio.py \
     --run-id 20250301_set_a \
     --label-map food_class_pre_label.csv \
     --images-root data/2_filtered
   ```
3. Label Studio UI에서 `food-<run_id>` 프로젝트를 열고 저신뢰 샘플을 검수합니다.
4. 검수 완료 후 `/run 9` 또는 직접 `scripts/export_from_studio.py`를 실행해 YOLO export를 다운로드합니다.

```bash
python scripts/export_from_studio.py \
  --run-id 20250301_set_a \
  --project-id <ID> \
  --token $LABEL_STUDIO_TOKEN \
  --image-root data/2_filtered/20250301_set_a \
  --skip-dataset
```
- export는 `labels/4_export_from_studio/<run_id>/source`에 저장됩니다.

### 3.6 데이터셋 패키징
```bash
python scripts/prepare_food_dataset.py \
  --run-id 20250301_set_a \
  --source labels/4_export_from_studio/20250301_set_a/source \
  --source-classes labels/4_export_from_studio/20250301_set_a/source/classes.txt \
  --master-classes food_class_pre_label.csv \
  --label-map food_class_after_label.csv \
  --image-root data/2_filtered/20250301_set_a \
  --copy-mode hardlink --val-ratio 0.2 --overwrite
```
- 출력: `data/5_datasets/<run_id>/{images,labels,classes.txt,train.txt,val.txt,<run_id>.yaml}`
- `scripts/plan_train_subset.py --run-id 20250301_set_a`를 실행하면 클래스별 train 제외 후보(`data/meta/<run_id>_exclude.txt`)를 생성합니다.

### 3.7 학습 및 지표 확인
```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_yolo.py \
  --run-id 20250301_set_a \
  --config configs/food_poc.yaml \
  --model models/yolo11l.pt \
  --device 0

python scripts/report_training_metrics.py \
  --run-id 20250301_set_a \
  --include-confusion --top 5
```
- 학습 결과: `runs/train/<name>`, `models/runs/metrics.json`, `data/meta/train_metrics/<run_id>_{per_class,confusion_matrix}.csv`
- `report_training_metrics.py`로 취약 클래스를 정리해 다음 Label Studio 라운드의 우선순위를 선정합니다.
- **주의:** `configs/food_poc.yaml` 파일은 빠른 테스트용(작은 `batch_size`, `epochs`=1)으로 설정되어 있습니다. 실제 학습 시에는 이 값들을 상향 조정해야 합니다.

### 3.8 Active Learning 반복
1. `report_training_metrics.py` 결과를 기반으로 재검수 대상 클래스를 선택합니다.
2. Label Studio에서 신규 라벨을 수집하고 `/run 9`으로 export 받은 뒤 새 run_id(`base_r2`, `base_r3` …)를 할당합니다.
3. 다음 명령으로 validation split을 고정한 채 train manifest만 갱신합니다.
   ```bash
   python scripts/prepare_food_dataset.py \
     --run-id 20250301_set_a_r2 \
     --source labels/4_export_from_studio/20250301_set_a_r2/source \
     --image-root data/2_filtered/20250301_set_a \
     --source-classes labels/4_export_from_studio/20250301_set_a_r2/source/classes.txt \
     --master-classes food_class_pre_label.csv \
     --label-map food_class_after_label.csv \
     --val-list data/5_datasets/20250301_set_a/val.txt \
     --train-exclude data/meta/20250301_set_a_exclude.txt \
     --copy-mode hardlink --overwrite
   ```
4. `/run 11`으로 새 run을 학습하면 이전 라운드의 `best.pt`를 자동으로 이어받습니다. 동일 train manifest가 감지되면 `/run 13`이 경고를 띄우므로 중복 생성을 방지할 수 있습니다.

---

## 4. Pipeline Runner 사용법
전체 단계를 수동으로 입력하는 대신 `pipeline_runner.py`를 사용할 수 있습니다.

```bash
# 대화형 모드
python scripts/pipeline_runner.py --interactive
```

주요 기능:
- `/set run_id <값>`으로 run_id를 변경
- `/help`로 사용 가능한 명령 번호 확인
- `/run <번호>`로 해당 단계 실행 (필요한 환경 변수는 프롬프트로 입력)
- `/run 1` (크롤링) 등 일부 명령어는 클래스, 이미지 수 등 주요 옵션을 직접 입력받아 실행합니다.
- 동일 세션에서 실패 단계만 다시 실행 가능

Active Learning에서는 `/set base_run_id <초기_run_id>` 후 `/run 13`(AL 패키징), `/run 11`(학습) 순으로 사용하는 것을 권장합니다.

---

## 5. 디렉터리 맵
```
data/
  1_raw/<run_id>/            # 크롤링 원본
  2_filtered/<run_id>/       # 중복 제거본 + stats.yaml
  3_exports/<run_id>/        # Label Studio import용 샘플/JSON
  5_datasets/<run_id>/       # 최종 학습 데이터셋
  meta/
    crawl_logs/<run_id>.json
    train_metrics/<run_id>_{per_class,confusion_matrix}.csv
labels/
  3-1_yolo_auto/<run_id>/
  3-2_meta/{predictions.csv,review_queue.csv}
  3-3_viz/<run_id>/
  4_export_from_studio/<run_id>/source/
scripts/  # 모든 실행 스크립트 (crawl, dedup, auto_label, prepare, train ...)
configs/food_poc.yaml
models/{yolo11l.pt,font/NanumGothic.ttf,runs/}
```

---

## 6. 문제 해결
- 클래스 재맵핑, Active Learning exclude, GPU 메모리, 대분류 전처리 등 과거 이슈는 `TROUBLESHOOTING.md`에 정리되어 있습니다.
- 작업 중 발견한 추가 문제는 `TODO.md` 또는 PR/노트에 기록해 traceability를 유지하세요.

---

## 7. 참고 문서
- `AGENTS.md` – 운영 원칙 및 각 스크립트 사용 규칙
- `COMMANDS.md` – `/run <번호>` 명령 모음
- `TODO.md` – 현재/향후 작업 항목
- `TROUBLESHOOTING.md` – 반복 이슈와 해결책

이 README만으로도 전체 파이프라인을 실행할 수 있도록 유지하세요. 단계별 로그, run_id, 주요 하이퍼파라미터는 항상 노트나 PR에 남겨 다음 운영자가 동일한 결과를 재현할 수 있도록 합니다.
