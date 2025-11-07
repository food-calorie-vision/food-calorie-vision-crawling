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
### 3.1 Data Crawling (`scripts/crawl_images.py`)
- Read `target_food.csv`, select classes according to priority columns.
- Generate a run_id before starting and reuse it for all outputs (`data/raw/<run_id>`, logs, labels).
- Mandatory arguments: `--classes`, `--min_per_class`, `--max_per_class`, `--out data/raw/<run_id>`.
- Respect robots.txt and add throttle or delay flags when available.
- After each run, write a summary (`run_id`, class counts, error URLs) to `data/meta/crawl_logs/<timestamp>.json`.
- Use `--limit` or `--dry-run` during testing to avoid large downloads and confirm CLI arguments.
- Current default source is DuckDuckGo (`duckduckgo_search`); update `requirements.txt` if you swap sources.

### 3.2 Deduplication and Filtering (`scripts/dedup_filter.py`)
- Inputs: previously crawled folder, outputs to `data/filtered/<run_id>`.
- Required checks: hash duplicates, perceptual similarity, broken/corrupt files, minimum resolution.
- Emit `stats.yaml` capturing files kept/dropped plus reasons; link it in PR or task notes.

### 3.3 Auto Labeling (`scripts/auto_label.py`)
- Use the current best checkpoint under `models/`.
- Store predictions under `labels/yolo/<run_id>` with filenames matching `data/filtered`.
- Export per-image confidences; mark low-confidence samples (`< threshold`) for manual review in `labels/meta/review_queue.csv`.

### 3.4 Human Validation (Label Studio or CVAT)
- Sync auto-labeled data into the chosen tool using project naming convention `food-<run_id>`.
- After validation, export YOLO-format labels back into `labels/yolo_validated/<run_id>` and document reviewer notes.

### 3.5 Training and Active Learning (`scripts/train_yolo.py`)
- Configs live under `configs/`; clone `configs/food_poc.yaml` when creating new datasets.
- Training artifacts belong in `models/<date>_<run_id>.pt` plus `metrics.json`.
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
| `data/source_list.md` | (Create/update) provenance and licensing log |
| `scripts/*.py` | Automation entry points (crawl, filter, label, export, train) |

Stay within these guardrails to keep every agent handoff predictable and auditable.
