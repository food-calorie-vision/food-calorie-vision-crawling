# Troubleshooting Guide

운영 중 반복해서 발생했던 이슈와 해결 방법을 요약했습니다. 새 작업을 시작하기 전 동일 증상이 있는지 먼저 확인하세요.

---

## 1. Label Studio ↔ Master Class 맵핑 불일치

### 증상
- Label Studio에서 확인한 라벨과 `prepare_food_dataset.py` 결과 라벨이 완전히 다른 상위 카테고리로 바뀌어 학습 시 대규모 오분류 발생 (예: “우유”가 “김밥류”로 분류).
- `classes.txt`와 `food_class_pre_label.csv` 사이의 ID가 서로 다른데도, 숫자만 맞춰 매핑하면서 뒤바뀜.

### 원인
- Label Studio는 프로젝트 내 클래스 이름을 **사전순으로 0부터 재할당**합니다.
- 우리 파이프라인은 `food_class_pre_label.csv`에 있는 **고정 master ID**를 기준으로 동작합니다.
- 기존 `prepare_food_dataset.py`는 Label Studio ID를 그대로 master ID로 착각해 잘못된 매핑을 수행했습니다.

### 해결
1. `prepare_food_dataset.py`를 전면 개편해 이름 기반 다단계 매핑을 적용했습니다.
   - Label Studio ID → 클래스 이름 (`classes.txt`)
   - 클래스 이름 → master ID (`food_class_pre_label.csv`)
   - master ID → 학습용 상위 카테고리 (`food_class_after_label.csv`)
   - 최종 카테고리 → YOLO ID 재할당
2. `/run 10` 명령(prepare_food_dataset 단계)에 `--source-classes`, `--master-classes`, `--label-map` 인자를 강제해 모든 호출이 위 매핑 체계를 사용하도록 했습니다.

### 추가 발견 & 교정
- **이미지 미검출(`Copied 0 images`)**: 리팩터링 시 이미지 탐색을 상위 폴더만 검색하도록 단순화하면서 class별 하위 디렉터리를 찾지 못했습니다 → `copy_images_for_labels()`를 `rglob` 기반 검색으로 수정.
- **`plan_train_subset.py`가 `class_unknown`만 출력**: 이미지 경로에서 `.jpg.txt` 형태로 레이블을 찾으려 해 실패함 → 확장자 제거 로직을 수정해 `<image>.txt`를 정확히 찾도록 변경.

### 교훈
- 클래스/ID/경로를 다루는 코드 변경 시 구조 가정이 깨졌는지 끝까지 검증합니다. 단순화보다는 `rglob`, 명시적 매핑 등 안전한 접근을 선택하세요.

---

## 2. Active Learning 패키징에서 제외 목록이 적용되지 않음

### 증상
- `/run 13` 실행 시 `prepare_food_dataset.py`가 “train exclude entries not found” 경고를 반복 출력하고, `data/meta/<base_run>_exclude.txt`에 기록된 이미지가 그대로 train에 남음.

### 원인
- baseline exclude 파일은 `images/<hash>__클래스_0001.jpg` 형태를 보존하지만, AL 패키징은 새 run의 이미지 경로를 `class_name/file.jpg` 구조로 다시 만들기 때문에 문자열이 일치하지 않았습니다.

### 해결
- `build_candidate_keys()`를 확장해 각 이미지에 대해 `rel path`, `images/<rel>`, 파일명, `클래스명_번호`, `__` 분리 suffix 등 다양한 비교 키를 생성했습니다.
- `apply_manifest_filter()`는 include/exclude 여부와 상관없이 매칭된 키를 기록하고, 매칭 실패 시에만 경고하도록 수정해 기존 exclude 파일을 그대로 재사용할 수 있게 했습니다.

---

## 3. YOLO 학습 반복 시 GPU 메모리 미해제

### 증상
- `/run 11`을 여러 번 실행하면 `nvidia-smi`에 이전 학습의 GPU 메모리가 남아 있어 중복 실행처럼 보임.
- CTRL+C로 강제 종료하면 바로 해제되지만 정상 종료 시에는 캐시가 남음.

### 원인
- PyTorch는 프로세스가 살아 있는 동안 CUDA 캐시를 유지합니다. 학습 후 `torch.cuda.empty_cache()`를 호출하지 않아 캐시가 잔류했습니다.

### 해결
- `scripts/train_yolo.py` 종료 직전에 `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`, `gc.collect()`를 호출해 GPU/CPU 메모리를 즉시 반환하도록 했습니다.

---

## 4. 대분류 맵핑 데이터 전처리 오류

### 증상
- 대규모 클래스 CSV(`db-preprocessing` 리포)에서 추출한 맵핑이 제대로 적용되지 않아 혼동행렬에서 특정 클래스가 엉뚱한 대분류로 몰림.

### 원인
1. 클래스 이름 문자열에 포함된 공백을 제거하지 않아 키가 일치하지 않았습니다.
2. 애매한 분류 항목을 별도 검토하지 않아 잘못된 대분류로 묶였습니다.

### 해결
- CSV 로딩 시 `.str.replace(' ', '')` 등 공백 제거 전처리를 추가하고, 혼동행렬을 주기적으로 분석해 예외 항목을 재검토했습니다.
- `food_class_after_label.csv` 갱신 시 출처와 변경 이유를 기록해 추후 감사가 가능하도록 했습니다.

---

문제 재현 시 이 문서를 먼저 확인한 뒤, 추가 사례와 해결책은 `TODO.md` 또는 노트에 이어서 기록해 주세요.
