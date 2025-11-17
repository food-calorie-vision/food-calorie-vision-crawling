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
3. Label Studio / CVAT 등을 활용한 반자동 검수로 라벨 품질을 확보합니다.
4. 재학습과 Active Learning 루프를 통해 모델 성능을 지속적으로 개선합니다.

### 클래스 매핑
- `target_food.csv`를 수정하면 `food_class.csv`를 꼭 재생성하세요. 이 파일만이 `id,classes` 매핑의 단일 소스입니다.
- 재생성 명령 예시(실행한 커맨드를 TODO/로그에 기록):
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
- 실험용 서브셋이 필요하면 `food_class_subset.csv`처럼 별도 파일을 만들고, `food_class.csv` 원본은 그대로 유지합니다. `label_map.csv`는 향후 특수 매핑용으로 남겨두고 현재 파이프라인에서는 사용하지 않습니다.

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
   - 입력: `--input data/raw/<run_id>` (클래스별 폴더 구조 유지)  
   - 출력: `data/filtered/<run_id>/<class_name>`에 정확히 동일한 파일을 제외한 버전이 복사됩니다.  
   - 현재 버전은 해시 기반(sha256) 완전 중복만 제거하며, 실행 결과 요약을 `data/filtered/<run_id>/stats.yaml`에 기록합니다(클래스별 input/kept/dropped 및 duplicates 목록 포함).  
   - `--dry-run`으로 어떤 파일이 중복인지 확인만 할 수 있고, 동일한 run_id로 여러 번 돌려도 stats가 최신 상태로 갱신됩니다.
   
   **아이콘 정리** – `scripts/remove_icons.py`  
   기존에 다운로드된 작은 아이콘 파일(50x50 이하) 정리용 스크립트
3. **초기 자동 라벨링** – `scripts/auto_label.py`  
   - 입력: `--images data/filtered/<run_id>` (필터링 완료본)  
   - 프로젝트 표준은 **YOLOv11 Large (`yolo11l.pt`)**이며, 기본 경로(`models/yolo11l.pt`)에 파일이 없으면 스크립트가 다운로드 안내를 출력합니다. 필요 시 `--weights`로 다른 경로를 지정할 수 있습니다.
   - 추론 결과는 `labels/yolo/<run_id>/...` 구조에 YOLO txt로 저장됩니다.  
   - 모든 박스는 YOLO 포맷(`class_id x_center y_center width height`)으로 저장되며, per-detection confidence는 `labels/meta/<run_id>_predictions.csv`에 기록됩니다.  
   - 이미지별 최대 confidence가 `--review-threshold`보다 낮거나 검출이 없으면 `labels/meta/review_queue.csv`에 run_id/이유를 append해 수동 검수 대기열을 유지합니다.  
   - 검수 전 샘플 확인이 필요하면 `python scripts/visualize_predictions.py --run-id <run_id> --top-n 2 --bottom-n 3`을 실행해 고신뢰/저신뢰 이미지를 `labels/viz/<run_id>/`에 렌더링할 수 있습니다.
   - Label Studio/CVAT에서 사람이 수정한 이미지의 라벨만 활용하려면, GUI export 결과를 `labels/yolo_validated/<run_id>`로 저장하거나 `python scripts/filter_labels.py --labels labels/yolo/<run_id> --keep-ids <ids> --remap-id <new_id>`로 검수된 클래스만 남깁니다.
   - `--batch-size`, `--device`, `--confidence`, `--iou` 등 YOLO 옵션을 CLI에서 조정할 수 있습니다.
4. **반자동 검수** – Label Studio / CVAT  
   자동 생성 결과를 불러와 수정·삭제·추가 수행
5. **모델 학습** – `scripts/train_yolo.py`  
   검수 완료 데이터를 사용해 YOLO 가중치 갱신
6. **Active Learning 루프**  
   낮은 신뢰도 샘플을 우선 검수하고 3~5단계를 반복

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
|   |-- raw/                    # 크롤링 원본
|   |   `-- <run_id>/           # 예: 20251107_set_1
|   |       |-- 기장/           # 클래스별 폴더 (한글 클래스명 유지)
|   |       |-- 보리/
|   |       `-- 조/
|   |-- filtered/               # 중복·저화질 제거본
|   `-- splits/                 # train/val/test
|-- labels/
|   |-- yolo/                   # YOLO txt 라벨
|   `-- meta/                   # 클래스 매핑·통계
|-- models/                     # 체크포인트(.pt)
|-- scripts/
|   |-- crawl_images_playwright.py
|   |-- remove_icons.py         # 아이콘 파일 정리 스크립트
|   |-- dedup_filter.py
|   |-- auto_label.py
|   |-- visualize_predictions.py  # 상/하위 confidence 샘플 시각화
|   |-- package_label_data.py     # Label Studio/CVAT 로컬 스토리지용 폴더 export 생성
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
- Ultralytics YOLO 가중치(`yolo11l.pt` 등)를 `models/` 디렉터리에 두거나, `--weights`로 직접 경로를 지정하세요. 파일이 없으면 스크립트가 다운로드 안내를 출력합니다.
- Playwright 기반 크롤러 사용 시:
  - `playwright install chromium` (브라우저 바이너리 설치)
  - `sudo playwright install-deps` 또는 `sudo apt-get install libnss3 libnspr4 libasound2t64` (시스템 의존성)
- 선택: Label Studio 또는 CVAT 서버

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
python scripts/remove_icons.py data/raw/<run_id>

# 전처리 (완전 중복 제거 + stats.yaml 생성)
python scripts/dedup_filter.py \
  --input data/raw/<run_id> \
  --output data/filtered/<run_id>

# 자동 라벨링
python scripts/auto_label.py \
  --images data/filtered/<run_id> \
  --out labels/yolo/<run_id> \
  --confidence 0.4 \
  --review-threshold 0.3 \
  --device 0  # 또는 cpu

# Label Studio/CVAT 로컬 스토리지 export 생성 (zip 필요 없음)
python scripts/package_label_data.py \
  --run-id <run_id> \
  --images data/filtered/<run_id> \
  --overwrite  # 기존 export/<run_id> 폴더가 있다면 덮어쓰기

# export 결과를 Label Studio 데이터 루트로 이동 (예: rsync / cp)
rsync -av data/exports/<run_id>/images/ /home/pollux/label-studio/data/local_storage/<run_id>/images/

# 자동 라벨 JSON 생성 (Local Files 경로 기준 document root 지정)
python scripts/export_labelstudio_json.py \
  --run-id <run_id> \
  --images data/filtered/<run_id> \
  --labels labels/yolo/<run_id> \
  --document-root "/data/local-files/?d=<run_id>/images/" \
  --from-name label \
  --to-name image \
  --output data/exports/<run_id>/labelstudio.json

# (선택) 특정 ID만 별도 CSV로 정리하고 싶은 경우
cat <<'EOF' > food_class_subset.csv
45,국밥
EOF
python scripts/export_labelstudio_json.py \
  --run-id <run_id> \
  --images data/filtered/<run_id> \
  --labels labels/yolo/<run_id> \
  --document-root "/data/local-files/?d=<run_id>/images/" \
  --label-map food_class_subset.csv \
  --output data/exports/<run_id>/labelstudio.json

`--document-root`에는 `/data/local-files/?d=<run_id>/images/`처럼 STORAGE 경로에
맞는 하위 폴더까지 포함시켜 주세요. 여러 run을 같은 프로젝트에서 관리하거나
특수한 경로 매핑이 필요한 경우 `{path}` 플레이스홀더를 활용할 수도 있습니다.
예: `--document-root "/data/local-files/?d=<run_id>/images/{path}"`.

# Label Studio UI에서 수동으로 프로젝트/스토리지 구성
docker run --rm -it -p 8080:8080 \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/local_storage \
  -v /home/pollux/label-studio/data:/label-studio/data \
  heartexlabs/label-studio:latest \
  label-studio start --no-browser --username admin --password admin

1. 브라우저에서 Label Studio 접속 → 새 프로젝트 생성
2. Settings → Storage → Add Local files → Path에 `/label-studio/data/local_storage/<run_id>/images`
   입력 후 저장 → “Sync storage” 실행
3. Data → Import에서 `data/exports/<run_id>/labelstudio.json` 업로드
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

검수가 끝나면 Label Studio Export(zip)를 `labels/yolo_validated/<run_id>/`
같은 별도 폴더에 풀어 두고, `labels/`/`images/`/`classes.txt` 등을 그대로
보관하세요. 이후 `scripts/filter_labels.py`로 COCO 기본 클래스(0~79)를 제거하고
음식 클래스만 0~N-1로 재매핑한 뒤 학습 데이터로 사용할 수 있습니다.

# 검수 결과를 음식 전용 데이터셋으로 패키징
python scripts/prepare_food_dataset.py \
  --run-id <run_id> \
  --source labels/yolo_validated/<run_id> \
  --label-map food_class.csv \
  --val-ratio 0.2 \
  --overwrite

위 스크립트는 `labels/yolo_validated/<run_id>/images`, `labels/.../labels`
구조를 `data/datasets/<run_id>/`로 복사하고, COCO 0~79 ID를 제거한 뒤
자동으로 `classes.txt`, `<run_id>.yaml`, `train.txt`, `val.txt`를 생성합니다.
`--val-ratio`를 지정하지 않으면 train/val 모두 전체 이미지 경로를 사용합니다.
생성된 `<run_id>.yaml`은 `train_yolo.py` 또는 Ultralytics CLI에서 그대로 사용할 수 있습니다.

# 검수 전 샘플 시각화 (상/하위 confidence)
python scripts/visualize_predictions.py \
  --run-id <run_id> \
  --top-n 2 \
  --bottom-n 3

# 검수된 클래스만 남기기 (예: YOLO ID 45 -> 0)
python scripts/filter_labels.py \
  --labels labels/yolo/<run_id> \
  --keep-ids 45 \
  --remap-id 0

# COCO 0~79 ID 제거 후 음식 클래스(80+)를 0~N-1로 재매핑
python scripts/filter_labels.py \
  --labels labels/yolo_validated/<run_id>/labels \
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
- **Annotation**: Label Studio, CVAT
- **언어·라이브러리**: Python, requests, beautifulsoup4, pillow, pandas 등
- **인프라**: Docker, Kubernetes GPU 노드, 공유 PVC 스토리지(계획)

---

## Roadmap
- [ ] PoC: 5개 클래스 x >=50장 크롤링 및 자동 라벨링
- [ ] YOLO 기반 자동 라벨 스크립트 완성
- [ ] Label Studio / CVAT 워크플로 가이드 정리
- [ ] Active Learning 자동화 스크립트 구현
- [ ] 상위 빈도 식품 50개로 확장
- [ ] 최종 1,234 클래스 타겟으로 스케일업
