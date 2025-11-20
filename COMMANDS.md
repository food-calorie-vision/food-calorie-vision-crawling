# Pipeline Command Reference

## 0. Run IDs / Prep
- 모든 단계는 `--run-id`를 사용하며 `YYYYMMDD_stage_seq` 형식을 권장합니다.
- 아래 명령의 `{run_id}`는 실제 run_id로 교체하세요. (인터랙티브 모드에서는 `/set run_id <값>`으로 설정)
- 클래스 서브셋이 필요하면 `target_food.csv`를 복사해 별도 CSV를 만든 뒤 `--classes` 또는 `--classes-csv`로 지정하세요.

## 1. Crawling (Playwright)
```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/crawl_images_playwright.py \
  --run-id {run_id} \
  {classes_flag} \
  --min_per_class {min_per_class} \
  --max_per_class {max_per_class} \
  --out data/1_raw {crawl_extra_flags}
```
- 인터랙티브 모드에서는 `/run 1` 실행 시 클래스, CSV 경로, 최소/최대 장수, 브라우저 표시, limit 등을 입력받아 위 옵션이 자동으로 채워집니다. `--limit`과 `--dry-run`은 필요 시 직접 지정하세요.

## 2. Cleanup (Icons + Dedup)
```bash
python scripts/remove_icons.py data/1_raw/{run_id}
python scripts/dedup_filter.py --input data/1_raw/{run_id} --output data/2_filtered/{run_id}
```

## 3. Auto Label
```bash
python scripts/auto_label.py \
  --images data/2_filtered/{run_id} \
  --out labels/3-1_yolo_auto/{run_id} \
  --weights models/yolo11l.pt \
  --run-id {run_id} \
  --device 0 \
  --confidence 0.4 \
  --review-threshold 0.3
```

## 4. Post-process (Largest box + class remap)
```bash
python scripts/postprocess_labels.py \
  --labels labels/3-1_yolo_auto/{run_id} \
  --label-map food_class_pre_label.csv
```
- class_id가 `food_class_pre_label.csv`의 ID로 재작성됩니다.

## 5. Export for Label Studio
```bash
python scripts/package_label_data.py \
  --run-id {run_id} \
  --images data/2_filtered/{run_id} \
  --output-dir data/3_exports/{run_id} \
  --overwrite

python scripts/export_labelstudio_json.py \
  --run-id {run_id} \
  --images data/2_filtered/{run_id} \
  --labels labels/3-1_yolo_auto/{run_id} \
  --label-map food_class_pre_label.csv \
  --output data/3_exports/{run_id}/labelstudio.json
```
- JSON은 기본적으로 `/data/local-files/?d=workspace/...` 경로를 만들어 Label Studio 컨테이너와 동일하게 맞춰 줍니다. 다른 접두사가 필요하면 `--local-prefix`나 `--document-root`를 지정하세요.

## 6. Visualization (Optional QA)
```bash
python scripts/visualize_predictions.py \
  --run-id {run_id} \
  --images data/2_filtered/{run_id} \
  --labels labels/3-1_yolo_auto/{run_id} \
  --predictions labels/3-2_meta/{run_id}_predictions.csv \
  --out labels/3-3_viz/{run_id}
```

## 7. Run Label Studio (Docker)
```bash
bash scripts/run_label_studio.sh \
  label-studio start --no-browser --username admin --password admin
```
- 스크립트가 `data/label-studio/`를 자동 생성하고 권한을 맞춰 주기 때문에 Permission 오류 없이 기동됩니다. 포트, 이미지, 컨테이너 이름 등은 환경변수(`LABEL_STUDIO_PORT`, `LABEL_STUDIO_IMAGE` 등)로 조절할 수 있습니다.

## 8. Bootstrap Label Studio Project
```bash
python scripts/bootstrap_label_studio.py \
  --run-id {run_id} \
  --tasks-json data/3_exports/{run_id}/labelstudio.json \
  --overwrite
```
- Access token을 입력하거나 `--token`/`LABEL_STUDIO_TOKEN`으로 넘기면 프로젝트를 생성/갱신하고 이미지를 `workspace/data/...` 경로로 import합니다. 기본 라벨 목록은 `data/2_filtered/{run_id}` 하위 폴더명에서 추출되며, 다른 매핑이 필요하면 `--label-map <csv>`를 추가하세요.
- `--copy-predictions` 플래그를 `/run 5` or pipeline export 단계에 추가하면 모델 예측을 annotations로 복사하여 Label Studio에서 즉시 확정된 상태로 표시할 수 있습니다. 기본값은 predictions만 포함하여 사람 검수 후 “Copy to annotation” 버튼으로 확정하는 방식입니다.

## 9. Pull Label Studio Export
```bash
python scripts/export_from_studio.py \
  --project-id {ls_project_id} \
  --run-id {run_id} \
  --token $LABEL_STUDIO_TOKEN \
  --image-root data/2_filtered/{run_id} \
  --val-ratio 0.3 \
  --skip-dataset
```
- Label Studio export(zip)을 `labels/4_export_from_studio/{run_id}`에 내려받아 `_extracted/`와 `source/`를 준비합니다. 기본적으로 `--skip-dataset`을 켜 두어 초기 데이터셋을 보존하고, 패키징은 `/run 10`에서 명시적으로 실행하세요. 필요 시 `--allow-overwrite`를 추가하면 기존 dataset 폴더를 덮어쓰도록 설정됩니다.
- `LABEL_STUDIO_TOKEN` 환경 변수가 필요하며, 인터랙티브 모드에서 `/run 9`을 호출하면 값 입력을 자동으로 요청합니다. Label Studio 인스턴스(`localhost:8080`)가 실행 중이어야 합니다.
- `ls_project_id` 컨텍스트가 없으면 `/run 9` 실행 시 프로젝트 번호 입력을 요청합니다. Label Studio UI 주소(`/projects/<ID>/...`)에서 확인 가능합니다.
- 기본적으로 완료된(annotation) 태스크만 내려받으며, 미라벨링 task도 포함하려면 `--include-unlabeled`를 추가하세요.

## 10. Packaging (after Label Studio)
```bash
python scripts/prepare_food_dataset.py \
  --run-id {run_id} \
  --source labels/4_export_from_studio/{run_id}/source \
  --image-root data/2_filtered/{run_id} \
  --source-classes labels/4_export_from_studio/{run_id}/source/classes.txt \
  --master-classes food_class_pre_label.csv \
  --label-map food_class_after_label.csv \
  --copy-mode hardlink \
  --val-ratio 0.3 \
  --output-root data/5_datasets \
  --overwrite

python scripts/plan_train_subset.py \
  --run-id {run_id} \
  --dataset-dir data/5_datasets/{run_id} \
  --output data/meta/{run_id}_exclude.txt
```
- 이 단계는 baseline 데이터셋을 확정하는 용도로만 사용하며, Label Studio와 Master DB의 클래스 ID 불일치 문제를 해결하는 재맵핑(re-mapping)을 수행합니다. `prepare_food_dataset.py` 호출 시 항상 `--overwrite`를 포함해 기존 `data/5_datasets/{run_id}`를 재생성합니다. 이후 AL 패키징은 `/run 13`을 통해 별도 run_id(`{run_id}_r2` 등)로 관리하세요.
- 동일 파티션이라면 `--copy-mode hardlink`로 디스크 사용량을 줄이세요. 다른 파티션일 경우 자동으로 copy2 모드가 사용됩니다.
- 최초 패키징 시에는 `--val-ratio`로 train/val split을 생성해 `data/5_datasets/{run_id}/train.txt`와 `val.txt`를 만듭니다. 이후 Active Learning에서는 13번 항목을 사용해 `val.txt`를 고정한 채 train 목록만 갱신하세요.
- 두 번째 명령인 `plan_train_subset.py`는 방금 생성된 `train.txt`를 읽어 클래스별 이미지 수를 표시한 뒤 “모든 클래스에서 제외할 장수”를 한 번만 입력받아 `data/meta/{run_id}_exclude.txt`를 생성합니다. 기본적으로 `train.txt`는 수정되지 않으며, 필요 시 `--update-train` 플래그를 추가해 갱신할 수 있습니다. 제외를 원치 않으면 Enter를 눌러 0으로 두면 됩니다.
- 추후 manifest 기반 제어가 필요하면 `--val-list`, `--train-include`, `--train-exclude`를 추가로 지정할 수 있습니다.

## 11. Training
```bash
python scripts/train_yolo.py \
  --config configs/food_poc.yaml \
  --data data/5_datasets/{run_id}/{run_id}.yaml \
  --run-id {run_id} \
  {train_model_flag} \
  --device 0
```

## 12. Training Metrics Report
```bash
python scripts/report_training_metrics.py \
  --run-id {run_id} \
  --sort-by ap50_95 \
  --top 5 \
  --include-confusion \
  --conf-top 5 \
  --output-csv data/meta/train_metrics/{run_id}_per_class_sorted.csv
```
- `train_yolo.py`가 저장한 `data/meta/train_metrics/{run_id}_per_class.csv`와 `{run_id}_confusion_matrix.csv`를 읽어 클래스별 성능을 요약합니다.
- `--sort-by`로 정렬 기준(precision/recall/ap50/ap50_95)을 지정하고, `--ascending`을 붙이면 약한 클래스 순으로 정렬됩니다.
- `--include-confusion`을 지정하면 혼동행렬에서 가장 큰 off-diagonal 항목을 보여 주므로 검수/추가 라벨링 우선순위를 정할 수 있습니다.

## 13. Active Learning Packaging
```bash
python scripts/prepare_food_dataset.py \
  --run-id {run_id} \
  --source labels/4_export_from_studio/{base_run_id}/source \
  --image-root data/2_filtered/{base_run_id} \
  --source-classes labels/4_export_from_studio/{base_run_id}/source/classes.txt \
  --master-classes food_class_pre_label.csv \
  --label-map food_class_after_label.csv \
  --copy-mode hardlink \
  --val-list data/5_datasets/{base_run_id}/val.txt \
  --train-exclude data/meta/{base_run_id}_exclude.txt \
  --output-root data/5_datasets \
  --overwrite
```
- `/set base_run_id <값>`으로 최초 학습 세트(run_id)를 지정하면 `{base_run_id}` 플레이스홀더가 채워집니다.
- `--val-list`는 최초 split의 `val.txt`를 그대로 재사용해 검증 세트를 고정하고, `--train-exclude`에는 초기 Packaging(#10)에서 생성한 `data/meta/{base_run_id}_exclude.txt`를 그대로 넘겨 train 구성을 유지합니다. 라운드별로 별도 include/exclude manifest가 필요하면 직접 파일을 만든 뒤 명령어에 추가하세요.
- `/run 13` 실행 시 직전 run과 동일한 `train.txt` 구성이 감지되면 경고 후 `y/N` 프롬프트로 진행 여부를 묻습니다. 테스트 환경에서는 `y`로 통과할 수 있지만 실제 운용에서는 중복 생성을 피하세요.
- 이 명령은 Active Learning 라운드에서 “추가 검수분만 train에 합치는” 용도로 사용하고, 기본 패키징(#10)과 혼동하지 마세요.

## 14. Train Subset Planner
```bash
python scripts/plan_train_subset.py \
  --run-id {run_id} \
  --dataset-dir data/5_datasets/{base_run_id} \
  --output data/meta/{run_id}_exclude.txt
```
- 기존 `train.txt`를 읽어 클래스별 이미지 수를 표시하고, 제외할 장수를 입력하면 `data/meta/{run_id}_exclude.txt`를 생성합니다.
- 기본적으로 train.txt는 유지되며, `/run 13` 실행 전 제외 manifest만 준비하고 싶을 때 사용합니다.
- train 목록까지 바로 반영하려면 `--update-train` 플래그를 함께 전달하세요.

## 99. Pipeline Runner (Batch / QA 전용)
```bash
# 전체 실행(데모/회귀 테스트 용도)
python scripts/pipeline_runner.py --run-id {run_id} --classes 기장 보리 조 --show-browser

# 필요한 단계만 지정
python scripts/pipeline_runner.py --run-id {run_id} --steps auto-label postprocess export --dry-run
```
- Active Learning 실무 흐름에서는 각 단계를 개별 `/run` 명령으로 실행하세요. 이 러너는 새 환경 재현이나 회귀 테스트에만 사용합니다.