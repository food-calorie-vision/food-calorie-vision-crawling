# Food Calorie Vision Crawling

`target_food.csv`에 정리된 품목 우선순위를 기반으로 곡류·잡곡 등 식품 이미지를 대량으로 수집하고, 자동·반자동 라벨링 파이프라인을 구축해 YOLO 모델과 데이터셋을 동시에 확장하는 프로젝트입니다.

---

## 프로젝트 개요
- 반복적이고 시간이 많이 드는 식품 이미지 수집·라벨링 작업을 자동화합니다.
- 크롤러 → 자동 라벨링 → 반자동 검수 → 재학습을 하나의 파이프라인으로 묶습니다.
- Active Learning 루프를 통해 신뢰도 낮은 샘플을 우선 검수하며 모델과 데이터 품질을 함께 끌어올립니다.

---

## 주요 목표
1. `target_food.csv`에 정의된 우선순위 식품 이미지를 안정적으로 수집하고 메타데이터를 관리합니다.
2. 사전학습된 YOLO(v8/v9/v11) 모델로 박스·클래스를 자동 생성합니다.
3. Label Studio 기반 반자동 검수로 라벨 품질을 확보합니다.
4. 재학습과 Active Learning 루프를 통해 모델 성능을 지속적으로 개선합니다.

### 진행 현황
- **이전 완료**  
  - Playwright 기반 크롤링·전처리·자동 라벨 파이프라인과 Label Studio export/bootstrap 흐름은 재현 가능한 상태입니다.  
  - `prepare_food_dataset.py`에 하드링크(copy-mode)·`--val-list`·`--train-include/--train-exclude` 옵션이 추가되어 validation 고정 및 train 서브셋 재구성이 가능합니다.  
- 학습 결과(`models/runs/<run_id>/results.csv`, `confusion_matrix.png`)를 기준으로 클래스별 성능을 분석하고, `train_yolo.py`가 생성하는 per-class/혼동행렬 CSV를 사용합니다. (기존 자동 라벨 기반 `scripts/analyze_class_performance.py`는 더 이상 사용하지 않습니다.)
- **현재 진행**  
  - 고정된 validation 목록을 유지한 상태에서 `train_yolo.py` 학습을 실행하고, run_id별 실험 로그(하이퍼파라미터/manifest)를 정리해야 합니다.  
  - `data/meta/train_metrics/<run_id>_per_class.csv`와 `/run 12` 리포트를 기반으로 취약 클래스를 선정해 Label Studio 재검수 대상만 재패키징합니다.  
  - README/AGENTS/TODO/COMMANDS의 Active Learning 지침을 계속 업데이트하며 run별 traceability를 유지합니다.
- **향후 계획**  
  - `report_training_metrics.py`로 정리한 per-class/혼동행렬 CSV를 근거로 Label Studio에 재검수/추가 라벨링이 필요한 샘플만 보내고, `export_from_studio.py` → 패키징 → 재학습의 Active Learning 루프를 반복합니다.  
  - 라운드별 run_id(`YYYYMMDD_al_rX`)와 제외 목록을 기록해 실험 추적성을 확보하고, 평가 지표(mAP/precision/recall)를 정량화한 리포트를 꾸준히 쌓습니다.

### 클래스 매핑
- `target_food.csv`는 크롤링 우선순위를 위한 참고 파일입니다. 실제 라벨 매핑은 외부 전처리 결과물인 `food_class_pre_label.csv`, `food_class_after_label.csv`를 그대로 사용하며, 이 저장소에서 재생성하거나 덮어쓰지 않습니다. 두 파일이 업데이트되면 최신본을 받아 교체하고 TODO/노트에 출처를 기록하세요.
- `food_class_pre_label.csv`는 YOLO 자동 라벨·Label Studio 단계에서 사용하는 공식 `id,classes` 매핑입니다.
- `food_class_after_label.csv`는 검수 이후 학습 패키징에서 사용하는 상위 카테고리 매핑입니다. 필요 시 `food_class_subset.csv`처럼 별도 실험 CSV를 만들 수 있지만, 원본 두 파일은 수정하지 않습니다.

---

## 데이터 수집 전략
| 단계        | 대상 클래스 수 | 클래스당 이미지(목표) | 비고                              |
|-------------|----------------|------------------------|-----------------------------------|
| PoC         | 5              | >=50                   | 기장, 보리 등 상위 곡류 우선       |
| 확장 1차    | ~50            | >=100                  | 소비 빈도/선호도 기반 식품        |
| Full(선택)  | ~1234          | 100~200                | 중요도에 따라 차등 수집           |

- 한 클래스당 300장 이상 바운딩 박스 확보를 장기 목표로 설정합니다.
- 크롤링 소스: Google Images, Naver 이미지, 블로그 리뷰, Food-101, UEC-Food 등 공개 데이터셋.
- 사용 허가와 용도(연구/비상업)는 `data/source_list.md`에 기록합니다.

---

## 파이프라인
1. **이미지 크롤링** – `scripts/crawl_images_playwright.py`  
   - DuckDuckGo API 호출이 차단된 환경을 고려해 Playwright 기반 렌더링 방식만 유지합니다.  
   - 설치: `pip install -r requirements.txt && playwright install chromium`  
   - 시스템 의존성: `sudo playwright install-deps` 또는 `sudo apt-get install libnss3 libnspr4 libasound2t64`  
   - **중요**: headless 모드에서는 이미지 탭 DOM이 제대로 렌더링되지 않아 `<img>` 태그를 찾지 못할 수 있으므로(특히 WSL·무헤드 리눅스 서버) 항상 `--show-browser` 옵션을 사용하고, 브라우저 런치는 `--no-sandbox` 플래그와 함께 수행됩니다.  
   - **동적 클래스 탐색**: DuckDuckGo가 랜덤 문자열 클래스로 이미지를 감싸므로 JavaScript로 해당 div를 찾고 이미지 URL을 추출합니다.  
   - **아이콘 필터링**: `.ico`, `favicon`, `/icon` URL은 제외하며, 다운로드 후 50x50 이하 이미지도 자동 삭제합니다.  
   - `--min_per_class`는 전역 limit을 초과하지 않는 범위에서 실제 다운로드 목표 하한으로 적용되며, 부족분은 크롤 로그 `notes`에 `shortfall_*`로 기록됩니다.
2. **전처리 / 중복 제거** – `scripts/dedup_filter.py`  
   - 입력: `--input data/1_raw/<run_id>` (클래스별 폴더 구조 유지)  
   - 출력: `data/2_filtered/<run_id>/<class_name>`에 정확히 동일한 파일을 제외한 버전이 복사됩니다.  
   - 현재 버전은 해시 기반(sha256) 완전 중복만 제거하며, 실행 결과 요약을 `data/2_filtered/<run_id>/stats.yaml`에 기록합니다(클래스별 input/kept/dropped 및 duplicates 목록 포함).  
   - `--dry-run`으로 어떤 파일이 중복인지 확인만 할 수 있고, 동일한 run_id로 여러 번 돌려도 stats가 최신 상태로 갱신됩니다.
   
   **아이콘 정리** – `scripts/remove_icons.py`  
   기존에 다운로드된 작은 아이콘 파일(50x50 이하) 정리용 스크립트
3. **초기 자동 라벨링** – `scripts/auto_label.py`  
   - 입력: `--images data/2_filtered/<run_id>` (필터링 완료본)  
   - 프로젝트 표준은 **YOLOv11 Large (`yolo11l.pt`)**이며, 기본 경로(`models/yolo11l.pt`)에 파일이 없으면 스크립트가 다운로드 안내를 출력합니다. 필요 시 `--weights`로 다른 경로를 지정할 수 있습니다.
   - 추론 결과는 `labels/3-1_yolo_auto/<run_id>/...` 구조에 YOLO txt로 저장됩니다.  
   - 모든 박스는 YOLO 포맷(`class_id x_center y_center width height`)으로 저장되며, per-detection confidence는 `labels/3-2_meta/<run_id>_predictions.csv`에 기록됩니다.  
   - 이미지별 최대 confidence가 `--review-threshold`보다 낮거나 검출이 없으면 `labels/3-2_meta/review_queue.csv`에 run_id/이유를 append해 수동 검수 대기열을 유지합니다.  
   - 검수 전 샘플 확인이 필요하면 `python scripts/visualize_predictions.py --run-id <run_id> --top-n 2 --bottom-n 3`을 실행해 고신뢰/저신뢰 이미지를 `labels/3-3_viz/<run_id>/`에 렌더링할 수 있습니다.
  - Label Studio에서 사람이 수정한 이미지의 라벨만 활용하려면, GUI export 결과를 `labels/4_yolo_validated/<run_id>`로 저장하거나 `python scripts/filter_labels.py --labels labels/3-1_yolo_auto/<run_id> --keep-ids <ids> --remap-id <new_id>`로 검수된 클래스만 남깁니다.
   - `--batch-size`, `--device`, `--confidence`, `--iou` 등 YOLO 옵션을 CLI에서 조정할 수 있습니다.
  - 추론이 끝나면 `python scripts/postprocess_labels.py --labels labels/3-1_yolo_auto/<run_id> --label-map food_class_pre_label.csv`로 각 이미지의 최대 bounding box만 남기고 class_id를 `food_class_pre_label.csv`에 정의된 ID로 재매핑하세요. 이후 `export_labelstudio_json.py --label-map`을 사용하면 Label Studio UI에는 “국밥” 같은 문자열 라벨이 표시됩니다.
4. **반자동 검수** – Label Studio  
  자동 생성 결과를 불러와 수정·삭제·추가 수행  
  - `/run 7`(`./scripts/run_label_studio.sh`)로 컨테이너 실행  
  - `/run 8`(`bootstrap_label_studio.py`)로 프로젝트 초기화  
  - `/run 9`(`export_from_studio.py`)로 YOLO export 다운로드 및 패키징  
  - 필요 시 `/run 10`(`prepare_food_dataset.py`)로 추가 패키징/val split 수행
5. **모델 학습** – `scripts/train_yolo.py`  
  검수 완료 데이터를 사용해 YOLO 가중치 갱신
6. **Active Learning 루프**  
  낮은 신뢰도 샘플을 우선 검수하고 3~5단계를 반복

### Pipeline Runner (QA/Batch 전용)
`scripts/pipeline_runner.py`(COMMANDS #99)는 회귀 테스트나 데모처럼 “여러 단계 한 번에 실행”이 필요할 때만 사용합니다. Active Learning 실무 흐름에서는 Label Studio 검수 대기 시간이 있으므로 각 `/run` 명령을 개별적으로 실행하세요.

```bash
# 전체 파이프라인 (크롤 → 정리 → 자동 라벨 → Export → Viz → 패키징 → 학습)
python scripts/pipeline_runner.py \
  --run-id 20251107_set_1 \
  --classes 기장 보리 조 \
  --min-per-class 80 \
  --max-per-class 120 \
  --show-browser

# 정제된 이미지에서 자동 라벨~패키징만 재실행
python scripts/pipeline_runner.py \
  --run-id crawl_test_b \
  --steps cleanup auto-label export visualize package \
  --dry-run  # 실제 실행 대신 커맨드만 확인
```

단계 이름: `crawl`, `cleanup`, `auto-label`, `postprocess`, `export`, `ls-export`, `visualize`, `package`, `train`.  
`ls-export` 스텝은 `scripts/export_from_studio.py`를 호출해 Label Studio 프로젝트에서 토큰을 입력받고 YOLO export를 다운로드한 뒤, `labels/4_export_from_studio/<run_id>/source`를 생성하고 `prepare_food_dataset.py`를 한 번 더 호출합니다.  
`--package-source`로 다른 라벨 디렉터리를 지정할 수 있으며, `--steps`로 부분 실행을 제어합니다.  
명령어 모음만 보고 싶으면 `python scripts/pipeline_runner.py --interactive` (또는 인자 없이 실행) 후 초기 run_id를 입력하고 `/set run_id <값>`, `/help`(도움말), 번호 입력으로 명령을 확인하고 `/run <번호>`로 실제 실행, `/exit`로 종료하면 `COMMANDS.md` 기준 번호별 명령어를 run_id가 치환된 상태로 활용할 수 있습니다. `/run`을 사용할 때 `$LABEL_STUDIO_TOKEN` 같은 환경 변수가 필요하면 자동으로 값을 입력하라는 프롬프트가 뜨며(`LABEL_STUDIO_TOKEN`은 getpass로 숨김 입력), 외부 서비스(예: Label Studio)가 내려가 있으면 해당 명령만 실패하고 인터랙티브 세션은 그대로 유지됩니다.

### 실행 로그 및 run_id 규칙
- 모든 자동화 작업은 `YYYYMMDD_<stage>_<seq>` 형식 run_id를 사용합니다. 예: `20250107_poc_a`.
- 각 크롤링 결과 요약은 `data/meta/crawl_logs/<run_id>.json`으로 저장하고, 소스 및 라이선스 변경 사항은 `data/source_list.md`에 함께 기록합니다.
- 동일한 run_id를 여러 번 실행하면 `history` 배열에 시도별 기록이 누적되고, `classes` 항목은 클래스별 최신 요약으로 병합됩니다.
- 로그의 각 클래스 항목에는 실제 폴더명(`folder`)이 함께 기록되어 sanitize 충돌이나 수동 정리를 추적할 수 있습니다.
- 실행한 스크립트와 옵션은 `AGENTS.md` 체크리스트와 `TODO.md` 진행상황에 반영해 추적합니다.

---

## 디렉터리 구조(초안)
```
project/
|-- data/
|   |-- 1_raw/                  # 크롤링 원본
|   |   `-- <run_id>/           # 예: 20251107_set_1
|   |       |-- 기장/           # 클래스별 폴더 (한글 클래스명 유지)
|   |       |-- 보리/
|   |       `-- 조/
|   |-- 2_filtered/             # 중복·저화질 제거본
|   |-- 3_exports/              # Label Studio 배포본
|   |-- 5_datasets/             # 학습 패키지
|   `-- splits/                 # train/val/test
|-- data/meta/train_metrics/    # 학습 후 작성되는 per-class/혼동행렬 CSV
|-- labels/
|   |-- 3-1_yolo_auto/          # 초기 YOLO 추론 결과
|   |-- 3-2_meta/               # 예측 CSV, review queue
|   |-- 3-3_viz/                # 고신뢰/저신뢰 샘플 시각화
|   `-- 4_yolo_validated/       # 검수 완료본
|-- models/                     # 체크포인트(.pt)
|-- scripts/
|   |-- crawl_images_playwright.py
|   |-- remove_icons.py         # 아이콘 파일 정리 스크립트
|   |-- dedup_filter.py
|   |-- auto_label.py
|   |-- visualize_predictions.py  # 상/하위 confidence 샘플 시각화
|   |-- package_label_data.py     # Label Studio 로컬 스토리지용 폴더 export 생성
|   |-- filter_labels.py          # 라벨 ID 필터/재매핑
|   |-- prepare_food_dataset.py   # 검수본 → 학습 데이터 패키징
|   `-- train_yolo.py
`-- notebooks/                  # 실험용 Jupyter/Colab
```

---

## 시작하기
### 요구 사항
- Python 3.10 (가상환경 권장)
- PyTorch + CUDA 12.x 환경
- `pip install -r requirements.txt`
- Matplotlib 한글 경고 방지를 위해 `models/font/NanumGothic.ttf` 폰트를 번들로 제공하며, `train_yolo.py`에서 자동으로 등록됩니다.
- Ultralytics YOLO 가중치(`yolo11l.pt` 등)를 `models/` 디렉터리에 두거나, `--weights`로 직접 경로를 지정하세요. 파일이 없으면 스크립트가 다운로드 안내를 출력합니다.
- Playwright 기반 크롤러 사용 시:
  - `playwright install chromium` (브라우저 바이너리 설치)
  - `sudo playwright install-deps` 또는 `sudo apt-get install libnss3 libnspr4 libasound2t64` (시스템 의존성)
- 선택: Label Studio 서버

### 기본 실행 예시
```bash
# 의존성 설치
pip install -r requirements.txt

python scripts/crawl_images_playwright.py \
  --run-id 20251107_set_1 \
  --classes 기장 보리 조 \
  --min_per_class 50 \
  --max_per_class 80 \
  --delay 1.5 \
  --show-browser \
  --dry-run  # 테스트 시

# 아이콘 파일 정리 (선택사항)
python scripts/remove_icons.py data/1_raw/<run_id>

# 전처리 (완전 중복 제거 + stats.yaml 생성)
python scripts/dedup_filter.py \
  --input data/1_raw/<run_id> \
  --output data/2_filtered/<run_id>

# 자동 라벨링
python scripts/auto_label.py \
  --images data/2_filtered/<run_id> \
  --out labels/3-1_yolo_auto/<run_id> \
  --confidence 0.4 \
  --review-threshold 0.3 \
  --device 0  # 또는 cpu

# Label Studio 로컬 스토리지 export 생성 (zip 필요 없음)
python scripts/package_label_data.py \
  --run-id <run_id> \
  --images data/2_filtered/<run_id> \
  --overwrite  # 기존 export/<run_id> 폴더가 있다면 덮어쓰기

# export 결과를 Label Studio 데이터 루트로 이동 (예: rsync / cp)
rsync -av data/3_exports/<run_id>/images/ /home/pollux/label-studio/data/local_storage/<run_id>/images/

# 자동 라벨 JSON 생성 (Local Files 경로 기준 document root 지정)
    python scripts/export_labelstudio_json.py \
      --run-id <run_id> \
      --images data/2_filtered/<run_id> \
      --labels labels/3-1_yolo_auto/<run_id> \
      --document-root "/data/local-files/?d=<run_id>/images/" \
  --from-name label \
  --to-name image \
  --output data/3_exports/<run_id>/labelstudio.json

# (선택) 특정 ID만 별도 CSV로 정리하고 싶은 경우
cat <<'EOF' > food_class_subset.csv
45,국밥
EOF
python scripts/export_labelstudio_json.py \
  --run-id <run_id> \
  --images data/2_filtered/<run_id> \
  --labels labels/3-1_yolo_auto/<run_id> \
  --document-root "/data/local-files/?d=<run_id>/images/" \
  --label-map food_class_subset.csv \
  --output data/3_exports/<run_id>/labelstudio.json

`--document-root`에는 `/data/local-files/?d=<run_id>/images/`처럼 STORAGE 경로에
맞는 하위 폴더까지 포함시켜 주세요. 여러 run을 같은 프로젝트에서 관리하거나
특수한 경로 매핑이 필요한 경우 `{path}` 플레이스홀더를 활용할 수도 있습니다.
예: `--document-root "/data/local-files/?d=<run_id>/images/{path}"`.

# Label Studio UI에서 수동으로 프로젝트/스토리지 구성
bash scripts/run_label_studio.sh \
  label-studio start --no-browser --username admin --password admin

스크립트는 `data/label-studio/`를 자동 생성하고 권한까지 맞춰 주므로
별도의 `chmod` 없이 컨테이너를 기동할 수 있습니다. 필요 시
`LABEL_STUDIO_PORT`, `LABEL_STUDIO_IMAGE`, `LABEL_STUDIO_CONTAINER`,
`LABEL_STUDIO_LOCAL_ALIAS` 환경변수로 포트·이미지·마운트 별칭을 조정하세요.

1. 브라우저에서 Label Studio 접속 → 새 프로젝트 생성
2. Settings → Storage → Add Local files → Path에 `/label-studio/data/local_storage/<run_id>/images`
   입력 후 저장 → “Sync storage” 실행
3. Data → Import에서 `data/3_exports/<run_id>/labelstudio.json` 업로드
   (pre-annotation으로 자동 라벨 박스가 붙습니다)

Labeling Interface 탭에는 최소 아래 템플릿을 붙여 주세요. `<RectangleLabels name="label" toName="image">`
와 `<Image name="image" value="$image"/>` 구조가 스크립트 기본값과 직접 연결됩니다.

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="국밥" background="#FFB020"/>
    <Label value="삼각김밥" background="#2A9D8F"/>
    <!-- 필요한 클래스만큼 Label을 추가 -->
  </RectangleLabels>
</View>
```

Label Studio의 기본 템플릿(`<Image name="image">`, `<RectangleLabels name="label">`)
과 일치하도록 스크립트 기본값이 이미 설정되어 있으므로, 특별히 다른
설정을 쓰지 않는다면 추가 옵션 없이 그대로 실행하면 됩니다.
레이블 이름을 바꾼 경우에만 `--from-name`, `--to-name`을 맞춰 주고,
숫자 ID 대신 사람이 읽을 수 있는 라벨명을 쓰고 싶을 때 `--label-map`
CSV를 제공하면 됩니다. 가장 확실한 방법은 Label Studio에서 샘플을
수동으로 라벨링 후 Export하여 JSON 구조를 확인하고 동일하게 맞추는 것입니다.

검수가 끝나면 Label Studio Export(zip)를 `labels/4_yolo_validated/<run_id>/`
같은 별도 폴더에 풀어 두고, `labels/`/`images/`/`classes.txt` 등을 그대로
보관하세요. 이후 `scripts/filter_labels.py`로 COCO 기본 클래스(0~79)를 제거하고
음식 클래스만 0~N-1로 재매핑한 뒤 학습 데이터로 사용할 수 있습니다.

# 검수 결과를 음식 전용 데이터셋으로 패키징
python scripts/prepare_food_dataset.py \
  --run-id <run_id> \
  --source labels/4_yolo_validated/<run_id> \
  --label-map food_class_after_label.csv \
  --copy-mode hardlink \
  --val-ratio 0.3 \
  --overwrite

위 스크립트는 `labels/4_yolo_validated/<run_id>/images`, `labels/.../labels`
구조를 `data/5_datasets/<run_id>/`로 복사하고, COCO 0~79 ID를 제거한 뒤
자동으로 `classes.txt`, `<run_id>.yaml`, `train.txt`, `val.txt`를 생성합니다.
`--copy-mode hardlink`는 동일 파티션 안에서만 동작하며, 공간 절약이 필요한 경우에 사용합니다.
`--val-ratio`를 지정하면 자동으로 train/val split을 생성하며, 생성된 `train.txt`/`val.txt`는 이후 Active Learning 라운드에서 `--val-list`나 `--train-include` 옵션으로 재사용할 수 있습니다.
생성된 `<run_id>.yaml`은 `train_yolo.py` 또는 Ultralytics CLI에서 그대로 사용할 수 있습니다.

### Pull → Packaging → Training(초기 루프)
1. **Pull(Label Studio Export)** – `/run 9` 또는 `scripts/export_from_studio.py`로 검수 완료 라벨을 다운로드해 `labels/4_export_from_studio/<run_id>/source`만 최신화합니다. `--skip-dataset`이 기본이므로 데이터셋은 그대로 보존되며, 필요 시 `--allow-overwrite`를 추가해도 실제 패키징은 `/run 10`에서 수행됩니다.
2. **초기 Packaging** – `/run 10`은 위 source를 받아 `prepare_food_dataset.py`로 `data/5_datasets/<run_id>`를 **항상 `--overwrite`로** 재생성하고, 이어서 `plan_train_subset.py`로 `data/meta/<run_id>_exclude.txt`를 생성합니다. 이 단계가 “baseline 데이터셋”을 결정하므로 다른 용도(예: AL 패키징)에는 사용하지 마세요. exclude 파일은 train 리스트를 수정하지 않고 “제외 후보”만 기록하므로, AL 단계에서 그대로 재사용할 수 있습니다.
3. **Train** – `/run 11`은 기본적으로 `data/5_datasets/<run_id>/<run_id>.yaml`을 사용해 최초 학습을 수행합니다. Active Learning 라운드(run_id가 `_rN`)에서는 이전 라운드의 best.pt(또는 base run의 best.pt)를 자동으로 이어받아 fine-tune합니다. 학습 결과(`models/runs/<run_id>/results.csv`, confusion matrix)를 기반으로 취약 클래스를 분석하며, `train_yolo.py`가 저장한 `data/meta/train_metrics/<run_id>_per_class.csv`, `<run_id>_confusion_matrix.csv`는 `/run 12`(`scripts/report_training_metrics.py`)로 바로 요약할 수 있습니다.

### Active Learning 루프(검수 반복)
1. **추가 Pull** – 취약 클래스만 선별해 Label Studio에서 추가 검수를 진행한 뒤 `/run 9`으로 다시 export를 내려받습니다. 기존 run을 덮어쓰고 싶으면 `--allow-overwrite`를, 버전을 쌓고 싶으면 새 run_id(`crawl_test_b_r2` 등)를 명시하세요.
2. **AL Packaging** – `/run 13`은 `/set base_run_id <최초_run>` 이후 실행합니다. `base_run_id`의 `val.txt`를 `--val-list`로 넘겨 검증 세트를 고정하고, `data/meta/<새_run_id>_include.txt`/`..._exclude.txt`를 기반으로 train에 넣을 샘플만 구성합니다. include/exclude 파일이 없으면 전체 train을 유지한 채 exclude만 적용할 수 있습니다.
   - 동일한 train 구성(이전 run과 같은 `train.txt`)이 감지되면 `/run 13`에서 경고 후 진행 여부를 물으므로, 잘못된 중복 생성을 방지하려면 `y/N` 프롬프트에 유의하세요.
3. **Train(라운드 선택)** – 학습 스크립트를 실행할 때 이번이 최초 학습인지 확인하고, AL 라운드라면 `data/5_datasets/<새_run_id>/<새_run_id>.yaml`을 명시합니다. 이렇게 하면 validation split은 고정한 채 train만 갱신됩니다.

# (액티브 러닝) 기존 val 고정 + 추가 train만 재패키징
python scripts/prepare_food_dataset.py \
  --run-id <round_run_id> \
  --source labels/4_export_from_studio/<round_run_id>/source \
  --image-root data/2_filtered/<base_run_id> \
  --label-map food_class_after_label.csv \
  --copy-mode hardlink \
  --val-list data/5_datasets/<base_run_id>/val.txt \
  --train-exclude data/meta/<base_run_id>_exclude.txt \
  --output-root data/5_datasets \
  --overwrite

Active Learning 라운드에서는 최초 학습 세트의 `val.txt`를 `--val-list`로 넘겨 검증 세트를 그대로 유지하고,
기본적으로 `data/meta/<base_run_id>_exclude.txt`를 그대로 재사용해 train 구성을 유지합니다.
추가로 포함/제외해야 할 이미지가 있다면 별도 manifest(`data/meta/<round_run_id>_include.txt` 등)를 작성하고,
명령어에 `--train-include/--train-exclude`를 추가해 override하세요. 하드링크 모드를 쓰면 디스크 사용량 없이
train manifest만 갱신할 수 있고, include 목록 없이 exclude만 넘기면 기존 train에서 일부 샘플만 제외할 수 있습니다.
이렇게 생성된 `data/5_datasets/<round_run_id>`는 `train_yolo.py`로 다음 라운드 학습을 진행할 때 사용합니다.

# (액티브 러닝) 클래스별 train 제외 수를 인터랙티브로 선택
python scripts/plan_train_subset.py \
  --run-id <run_id> \
  --dataset-dir data/5_datasets/<run_id> \
  --output data/meta/<run_id>_exclude.txt

`plan_train_subset.py`는 기존 `train.txt`를 읽어 클래스별 이미지 수를 표시한 뒤 “모든 클래스에서 공통으로 제외할 장수”를 입력받아
각 클래스에서 동일한 개수만큼 무작위로 제외합니다. 제외된 파일은 `data/meta/<run_id>_exclude.txt`로 기록되며, 기본적으로 `train.txt`
는 그대로 유지됩니다. train 목록까지 갱신하고 싶다면 `--update-train` 플래그를 별도로 지정하세요.

# 검수 전 샘플 시각화 (상/하위 confidence)
python scripts/visualize_predictions.py \
  --run-id <run_id> \
  --top-n 2 \
  --bottom-n 3

# 학습 결과 기반 클래스 분석
- `train_yolo.py` 실행 시 `data/meta/train_metrics/<run_id>_per_class.csv`(precision/recall/AP)와 `<run_id>_confusion_matrix.csv`가 자동 생성됩니다.  
- `/run 13`(`scripts/report_training_metrics.py`)을 이용하면 위 CSV를 정렬·요약해 바로 확인할 수 있고, 필요 시 `--include-confusion`으로 혼동행렬 요약도 함께 출력합니다.  
- Notebook에서는 이 CSV와 `models/runs/<run_id>/results.csv`를 함께 로드해 pandas/matplotlib으로 시각화하여 취약 클래스를 선정하세요.  
- 기존 `scripts/analyze_class_performance.py`(자동 라벨 predictions 기반)는 사용하지 않습니다.

# 검수된 클래스만 남기기 (예: YOLO ID 45 -> 0)
python scripts/filter_labels.py \
  --labels labels/3-1_yolo_auto/<run_id> \
  --keep-ids 45 \
  --remap-id 0

# COCO 0~79 ID 제거 후 음식 클래스(80+)를 0~N-1로 재매핑
python scripts/filter_labels.py \
  --labels labels/4_yolo_validated/<run_id>/labels \
  --drop-below 80 \
  --shift-offset 80 \
  --output labels/food_only/<run_id>

# 검수된 데이터로 학습
UV_CACHE_DIR=.uv-cache uv run python scripts/train_yolo.py \
  --run-id <run_id> \
  --config configs/food_poc.yaml \
  --model models/yolo11l.pt \
  --device 0
```
(명령어와 옵션은 실제 구현에 맞춰 조정하십시오.)

---

## 기술 스택
- **Detection**: YOLOv8 / YOLOv9 / YOLOv11
- **Annotation**: Label Studio
- **언어·라이브러리**: Python, requests, beautifulsoup4, pillow, pandas 등
- **인프라**: Docker, Kubernetes GPU 노드, 공유 PVC 스토리지(계획)

---

## Roadmap
- [ ] PoC: 5개 클래스 x >=50장 크롤링 및 자동 라벨링
- [ ] YOLO 기반 자동 라벨 스크립트 완성
- [ ] Label Studio 워크플로 가이드 정리
- [ ] Active Learning 자동화 스크립트 구현
- [ ] 상위 빈도 식품 50개로 확장
- [ ] 최종 1,234 클래스 타겟으로 스케일업
초기 패키징 직후 `/run 10` 명령에 이 스크립트가 자동 포함되어 있으므로, 제외하고 싶은 장수를 한 번만 입력해 두면 모든 클래스에서 동일하게 제외됩니다. `train.txt`는 그대로 두고 `data/meta/<run_id>_exclude.txt`만 생성되므로, `/run 13` 실행 시 `--train-exclude`로 넘겨 재사용하면 됩니다. 추가 제외가 필요하면 나중에 다시 실행해 새로운 manifest를 만들 수 있습니다.
