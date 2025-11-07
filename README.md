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
1. **이미지 크롤링** – `scripts/crawl_images.py`  
   입력: `target_food.csv`의 클래스 목록, 예시 옵션 `--min_per_class 50 --max_per_class 300`
2. **전처리 / 중복 제거** – `scripts/dedup_filter.py`  
   해시·유사도 기반 중복 제거, 깨진 파일 필터링
3. **초기 자동 라벨링** – `scripts/auto_label.py`  
   YOLO 모델로 박스·클래스·확률(score) 생성
4. **반자동 검수** – Label Studio / CVAT  
   자동 생성 결과를 불러와 수정·삭제·추가 수행
5. **모델 학습** – `scripts/train_yolo.py`  
   검수 완료 데이터를 사용해 YOLO 가중치 갱신
6. **Active Learning 루프**  
   낮은 신뢰도 샘플을 우선 검수하고 3~5단계를 반복

### 실행 로그 및 run_id 규칙
- 모든 자동화 작업은 `YYYYMMDD_<stage>_<seq>` 형식 run_id를 사용합니다. 예: `20250107_poc_a`.
- 각 크롤링 결과 요약은 `data/meta/crawl_logs/<run_id>.json`으로 저장하고, 소스 및 라이선스 변경 사항은 `data/source_list.md`에 함께 기록합니다.
- 실행한 스크립트와 옵션은 `AGENTS.md` 체크리스트와 `TODO.md` 진행상황에 반영해 추적합니다.

---

## 디렉터리 구조(초안)
```
project/
|-- data/
|   |-- raw/         # 크롤링 원본
|   |-- filtered/    # 중복·저화질 제거본
|   `-- splits/      # train/val/test
|-- labels/
|   |-- yolo/        # YOLO txt 라벨
|   `-- meta/        # 클래스 매핑·통계
|-- models/          # 체크포인트(.pt)
|-- scripts/
|   |-- crawl_images.py
|   |-- dedup_filter.py
|   |-- auto_label.py
|   |-- export_to_yolo.py
|   `-- train_yolo.py
`-- notebooks/       # 실험용 Jupyter/Colab
```

---

## 시작하기
### 요구 사항
- Python 3.10 (가상환경 권장)
- PyTorch + CUDA 12.x 환경
- `pip install -r requirements.txt`
- 선택: Label Studio 또는 CVAT 서버

### 기본 실행 예시
```bash
# 의존성 설치
pip install -r requirements.txt

# 이미지 크롤링 (dry-run 예시)
python scripts/crawl_images.py \
  --classes 기장 보리 조 \
  --min_per_class 50 \
  --max_per_class 100 \
  --limit 30 \
  --dry-run

# PoC: 특정 클래스만 크롤링
python scripts/crawl_images.py \
  --classes 기장 보리 조 등 \
  --min_per_class 50

# 전처리
python scripts/dedup_filter.py \
  --input data/raw \
  --output data/filtered

# 자동 라벨링
python scripts/auto_label.py \
  --images data/filtered \
  --weights models/base_yolo.pt \
  --out labels/yolo

# 검수된 데이터로 학습
python scripts/train_yolo.py \
  --data configs/food_poc.yaml
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
