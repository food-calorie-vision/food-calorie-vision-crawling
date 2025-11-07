# Project TODO

Tracked tasks for Food Calorie Vision Crawling. Update this file as work progresses to keep LLM/agent contributors aligned.

---

## Phase 0 – Foundations
- [x] Pin Python 3.10 baseline, create `requirements.txt`, and document install command.
- [ ] Verify CUDA availability/version on target infra (sandbox lacks `nvidia-smi`; run on host).
- [x] Generate `.gitignore` entries for data outputs (`data/raw`, `data/filtered`, `labels`, `models`) to prevent large asset commits.
- [x] Create `data/source_list.md` template (columns: source, license, usage scope, notes).
- [x] Stub `configs/food_poc.yaml` (or verify existing) and document fields in README.

## Phase 1 – Data Acquisition
- [x] Finalize `scripts/crawl_images.py` CLI (input CSV parsing, throttling, retries).
- [x] Implement crawl logging to `data/meta/crawl_logs/<run_id>.json`.
- [x] Add sampling/dry-run mode (`--limit`) for safer tests.
- [x] Define run_id naming convention and reflect in AGENTS/README.

## Phase 2 – Cleaning & Labeling
- [ ] Implement `scripts/dedup_filter.py` (hash + perceptual dedupe, corrupt detection).
- [ ] Produce `stats.yaml` output format and sample file.
- [ ] Implement `scripts/auto_label.py` with YOLO weights loading, confidence thresholds, and `labels/meta/review_queue.csv` export.
- [ ] Document Label Studio/CVAT import-export workflow (screenshots or steps).

## Phase 3 – Training & Active Learning
- [ ] Implement `scripts/train_yolo.py` wrapper that reads dataset config and stores metrics + checkpoints.
- [ ] Define evaluation criteria (mAP, precision/recall) and logging format.
- [ ] Add inference script or reuse YOLO CLI for uncertainty scoring; document thresholds.
- [ ] Automate Active Learning loop orchestration (script or instructions).

## Phase 4 – Ops & QA
- [ ] Add automated lint/test (e.g., `pytest`, `ruff`, or formatting) to ensure scripts stay stable.
- [ ] Provide sample notebooks (exploratory analysis, visualization) with lightweight data.
- [ ] Prepare deployment/infra notes (Dockerfile, GPU node requirements, storage layout).
- [ ] Establish release checklist (data snapshot, model registry update, documentation refresh).
