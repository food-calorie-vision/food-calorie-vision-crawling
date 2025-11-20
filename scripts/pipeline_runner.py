#!/usr/bin/env python3
"""
End-to-end pipeline runner that stitches together the existing scripts.

Each stage can be executed independently via the --steps flag, or the
default ordering (crawl -> cleanup -> auto-label -> postprocess -> export
-> visualize -> package -> train) can be run as a batch.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_STEPS = [
    "crawl",
    "cleanup",
    "auto-label",
    "postprocess",
    "export",
    "ls-export",
    "visualize",
    "package",
    "train",
]

SENSITIVE_ENV_VARS = {"LABEL_STUDIO_TOKEN"}


@dataclass
class PipelineOptions:
    run_id: str
    steps: List[str]
    classes: List[str] = field(default_factory=list)
    classes_csv: Path = Path("target_food.csv")
    start_from: str | None = None
    min_per_class: int = 50
    max_per_class: int = 120
    crawl_limit: int | None = None
    crawl_delay: float = 1.5
    crawl_show_browser: bool = False
    auto_device: str = "auto"
    auto_batch: int = 16
    auto_conf: float = 0.4
    auto_iou: float = 0.5
    review_threshold: float = 0.3
    auto_weights: Path = Path("models") / "yolo11l.pt"
    export_per_class_only: bool = False
    package_source: Path | None = None
    val_ratio: float = 0.3
    postprocess_label_map: Path = Path("food_class_pre_label.csv")
    package_label_map: Path = Path("food_class_after_label.csv")
    train_config: Path = Path("configs/food_poc.yaml")
    train_model: Path | None = None
    dry_run: bool = False
    skip_icons: bool = False
    interactive: bool = False
    copy_predictions: bool = False
    ls_project_id: int | None = None
    ls_base_url: str = "http://localhost:8080"
    ls_token: str | None = None


class PipelineRunner:
    def __init__(self, options: PipelineOptions) -> None:
        self.opts = options
        self.root = Path(__file__).resolve().parents[1]
        self.python = Path(sys.executable)
        self.raw_dir = self.root / "data" / "1_raw" / self.opts.run_id
        self.filtered_dir = self.root / "data" / "2_filtered" / self.opts.run_id
        self.export_dir = self.root / "data" / "3_exports" / self.opts.run_id
        self.auto_label_dir = self.root / "labels" / "3-1_yolo_auto" / self.opts.run_id
        self.predictions_csv = self.root / "labels" / "3-2_meta" / f"{self.opts.run_id}_predictions.csv"
        self.review_csv = self.root / "labels" / "3-2_meta" / "review_queue.csv"
        self.viz_dir = self.root / "labels" / "3-3_viz" / self.opts.run_id
        self.validated_dir = self.opts.package_source or (
            self.root / "labels" / "4_export_from_studio" / self.opts.run_id / "source"
        )
        self.dataset_root = self.root / "data" / "5_datasets"

    def run(self) -> None:
        for step in self.opts.steps:
            handler = getattr(self, f"step_{step.replace('-', '_')}", None)
            if not handler:
                raise SystemExit(f"Unknown step '{step}'. Valid: {', '.join(DEFAULT_STEPS)}")
            print(f"\n=== Running step: {step} ===")
            handler()

    def step_crawl(self) -> None:
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "crawl_images_playwright.py"),
            "--run-id",
            self.opts.run_id,
            "--out",
            str(self.raw_dir),
            "--min_per_class",
            str(self.opts.min_per_class),
            "--max_per_class",
            str(self.opts.max_per_class),
            "--delay",
            str(self.opts.crawl_delay),
        ]
        if self.opts.crawl_limit:
            cmd += ["--limit", str(self.opts.crawl_limit)]
        if self.opts.start_from:
            cmd += ["--start-from", self.opts.start_from]

        if self.opts.classes:
            cmd += ["--classes", *self.opts.classes]
        else:
            cmd += ["--csv", str(self.opts.classes_csv)]

        if self.opts.crawl_show_browser:
            cmd.append("--show-browser")

        self._run(cmd)

    def step_cleanup(self) -> None:
        if not self.skip_icons and self.raw_dir.exists():
            cmd = [
                str(self.python),
                str(self.root / "scripts" / "remove_icons.py"),
                str(self.raw_dir),
            ]
            self._run(cmd)
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "dedup_filter.py"),
            "--input",
            str(self.raw_dir),
            "--output",
            str(self.filtered_dir),
            "--run-id",
            self.opts.run_id,
        ]
        self._run(cmd)

    def step_auto_label(self) -> None:
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "auto_label.py"),
            "--images",
            str(self.filtered_dir),
            "--out",
            str(self.auto_label_dir),
            "--weights",
            str(self.opts.auto_weights),
            "--run-id",
            self.opts.run_id,
            "--batch-size",
            str(self.opts.auto_batch),
            "--device",
            self.opts.auto_device,
            "--confidence",
            str(self.opts.auto_conf),
            "--iou",
            str(self.opts.auto_iou),
            "--review-threshold",
            str(self.opts.review_threshold),
            "--predictions-csv",
            str(self.predictions_csv),
            "--review-csv",
            str(self.review_csv),
        ]
        self._run(cmd)

    def step_postprocess(self) -> None:
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "postprocess_labels.py"),
            "--labels",
            str(self.auto_label_dir),
            "--label-map",
            str(self.opts.postprocess_label_map),
        ]
        self._run(cmd)

    def step_export(self) -> None:
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "package_label_data.py"),
            "--run-id",
            self.opts.run_id,
            "--images",
            str(self.filtered_dir),
            "--output-dir",
            str(self.export_dir),
            "--overwrite",
        ]
        if self.opts.export_per_class_only:
            cmd.append("--per-class")
        self._run(cmd)

        cmd = [
            str(self.python),
            str(self.root / "scripts" / "export_labelstudio_json.py"),
            "--run-id",
            self.opts.run_id,
            "--images",
            str(self.filtered_dir),
            "--labels",
            str(self.auto_label_dir),
            "--label-map",
            str(self.opts.postprocess_label_map),
            "--output",
            str(self.export_dir / "labelstudio.json"),
        ]
        if self.opts.copy_predictions:
            cmd.append("--copy-to-annotations")
        self._run(cmd)

    def step_visualize(self) -> None:
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "visualize_predictions.py"),
            "--run-id",
            self.opts.run_id,
            "--images",
            str(self.filtered_dir),
            "--labels",
            str(self.auto_label_dir),
            "--predictions",
            str(self.predictions_csv),
            "--out",
            str(self.viz_dir),
        ]
        self._run(cmd)

    def step_ls_export(self) -> None:
        if not self.opts.ls_project_id:
            print("[warn] --ls-project-id not set. Skipping Label Studio export.")
            return
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "export_from_studio.py"),
            "--project-id",
            str(self.opts.ls_project_id),
            "--run-id",
            self.opts.run_id,
            "--base-url",
            self.opts.ls_base_url,
            "--image-root",
            str(self.filtered_dir),
            "--dataset-root",
            str(self.dataset_root),
            "--val-ratio",
            str(self.opts.val_ratio),
            "--skip-dataset",
        ]
        if self.opts.ls_token:
            cmd += ["--token", self.opts.ls_token]
        self._run(cmd)

    def step_package(self) -> None:
        if not self.validated_dir.exists():
            print(f"[warn] Package source {self.validated_dir} missing. Skipping packaging.")
            return
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "prepare_food_dataset.py"),
            "--run-id",
            self.opts.run_id,
            "--source",
            str(self.validated_dir),
            "--image-root",
            str(self.filtered_dir),
            "--source-classes",
            str(self.validated_dir / "classes.txt"),
            "--master-classes",
            str(self.opts.postprocess_label_map),
            "--label-map",
            str(self.opts.package_label_map),
            "--output-root",
            str(self.dataset_root),
            "--val-ratio",
            str(self.opts.val_ratio),
            "--overwrite",
        ]
        self._run(cmd)

    def step_train(self) -> None:
        dataset_yaml = self.dataset_root / self.opts.run_id / f"{self.opts.run_id}.yaml"
        if not dataset_yaml.exists():
            print(f"[warn] Dataset yaml {dataset_yaml} missing. Skipping training.")
            return
        cmd = [
            str(self.python),
            str(self.root / "scripts" / "train_yolo.py"),
            "--config",
            str(self.opts.train_config),
            "--data",
            str(dataset_yaml),
            "--run-id",
            self.opts.run_id,
        ]
        if self.opts.train_model:
            cmd += ["--model", str(self.opts.train_model)]
        self._run(cmd)

    def _run(self, cmd: Iterable[str]) -> None:
        printable = " ".join(shlex.quote(str(part)) for part in cmd)
        if self.opts.dry_run:
            print(f"[dry-run] {printable}")
            return
        print(f"[exec] {printable}")
        subprocess.run(cmd, check=True)

    @property
    def skip_icons(self) -> bool:
        return self.opts.skip_icons


def load_command_sections(md_path: Path) -> OrderedDict[str, dict[str, object]]:
    sections: OrderedDict[str, dict[str, str]] = OrderedDict()
    if not md_path.exists():
        return sections
    lines = md_path.read_text(encoding="utf-8").splitlines()
    current_key: str | None = None
    current_body: List[str] = []
    current_blocks: List[str] = []
    in_code = False
    code_buffer: List[str] = []
    for line in lines:
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_buffer = []
            else:
                in_code = False
                block_text = "\n".join(code_buffer).strip()
                if block_text and current_key:
                    current_blocks.append(block_text)
                code_buffer = []
            continue
        if in_code:
            code_buffer.append(line)
            continue
        if line.startswith("## "):
            header = line[3:].strip()
            key = None
            title = header
            if "." in header:
                left, right = header.split(".", 1)
                if left.strip().isdigit():
                    key = left.strip()
                    title = right.strip()
            if key is None:
                key = str(len(sections) + 1)
            if current_key is not None:
                sections[current_key]["body"] = "\n".join(current_body).strip()
                sections[current_key]["blocks"] = list(current_blocks)
            sections[key] = {"title": title, "body": "", "blocks": []}
            current_key = key
            current_body = []
            current_blocks = []
        elif current_key:
            current_body.append(line)
    if current_key:
        sections[current_key]["body"] = "\n".join(current_body).strip()
        sections[current_key]["blocks"] = list(current_blocks)
    return sections


INVALID_RUN_ID_CHARS = {"/", "\\"}


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def apply_context(text: str, context: Dict[str, str]) -> str:
    return text.format_map(SafeFormatDict(context))


def set_context_value(context: Dict[str, str], key: str, raw_value: str) -> bool:
    value = (raw_value or "").strip()
    if not value:
        if key == "train_model_flag":
            context[key] = ""
            return True
        print(f"{key} 값이 없어 명령 실행을 건너뜁니다.")
        return False
    if key == "run_id" and any(char in value for char in INVALID_RUN_ID_CHARS):
        print("run_id 값에는 '/'나 '\\' 문자를 사용할 수 없습니다. 다른 run_id를 입력하세요.")
        return False
    context[key] = value
    return True


def ensure_context_placeholders(text: str, context: Dict[str, str]) -> bool:
    placeholders = re.findall(r"{([^{}]+)}", text)
    missing = [key for key in placeholders if not context.get(key)]
    for key in missing:
        if key == "train_model_flag":
            context[key] = ""
            continue
        prompt = f"{key} 값 입력 (Enter=취소): "
        try:
            value = input(prompt)
        except (EOFError, KeyboardInterrupt):
            value = ""
        if not set_context_value(context, key, value):
            return False
    return True


def hash_train_manifest(dataset_dir: Path) -> str | None:
    train_file = dataset_dir / "train.txt"
    if not train_file.exists():
        return None
    entries: List[str] = []
    with train_file.open("r", encoding="utf-8") as fp:
        for line in fp:
            trimmed = line.strip()
            if trimmed:
                entries.append(trimmed)
    if not entries:
        return None
    canonical = "\n".join(sorted(set(entries)))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def extract_env_vars(command_text: str) -> List[str]:
    matches = re.findall(r"(?<!\\)\$([A-Za-z_][A-Za-z0-9_]*)", command_text)
    seen = set()
    ordered: List[str] = []
    for name in matches:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def ensure_env(command_text: str, env_cache: Dict[str, str]) -> Dict[str, str] | None:
    needed = extract_env_vars(command_text)
    if not needed:
        return os.environ.copy()
    env = os.environ.copy()
    for name in needed:
        value = env_cache.get(name) or os.environ.get(name)
        if not value:
            prompt = f"{name} 환경변수 값 입력 (Enter=취소): "
            try:
                if name in SENSITIVE_ENV_VARS:
                    value = getpass.getpass(prompt)
                else:
                    value = input(prompt)
            except (EOFError, KeyboardInterrupt):
                value = ""
            value = value.strip()
            if not value:
                print(f"{name} 값이 없어 명령 실행을 건너뜁니다.")
                return None
            env_cache[name] = value
        env[name] = value
    return env


def compute_next_al_run_id(
    base_run_id: str,
    project_root: Path,
    current_run_id: str | None = None,
) -> str:
    dataset_root = project_root / "data" / "5_datasets"
    pattern = re.compile(rf"^{re.escape(base_run_id)}_r(\d+)$")
    max_suffix = 1
    if current_run_id:
        current_path = dataset_root / current_run_id
        match = pattern.match(current_run_id)
        if match and current_path.exists():
            max_suffix = max(max_suffix, int(match.group(1)))
    if dataset_root.exists():
        for entry in dataset_root.iterdir():
            match = pattern.match(entry.name)
            if match:
                max_suffix = max(max_suffix, int(match.group(1)))
    next_suffix = max_suffix + 1 if max_suffix >= 2 else 2
    return f"{base_run_id}_r{next_suffix}"


def run_interactive_menu(defaults: PipelineOptions | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    commands_md = root / "COMMANDS.md"
    sections = load_command_sections(commands_md)
    if not sections:
        print("COMMANDS.md 파일을 찾을 수 없습니다. README.md를 참고하세요.")
    context: Dict[str, str] = {"run_id": "", "ls_project_id": "", "train_model_flag": ""}
    context["initial_run_id"] = ""
    try:
        initial = input("초기 run_id (없으면 Enter): ").strip()
    except (EOFError, KeyboardInterrupt):
        initial = ""
    if initial:
        if not set_context_value(context, "run_id", initial):
            print("run_id 초기값이 설정되지 않았습니다. /set run_id <값>으로 다시 지정하세요.")
        else:
            context.setdefault("base_run_id", context["run_id"])
            context["initial_run_id"] = context["run_id"]
    if defaults:
        if defaults.run_id and "default_run_id" not in context:
            context["default_run_id"] = defaults.run_id
        if defaults.run_id and "base_run_id" not in context:
            context["base_run_id"] = defaults.run_id
    default_classes_flag = "--classes 기장 보리 조"
    if defaults and defaults.classes:
        default_classes_flag = "--classes " + " ".join(defaults.classes)
    elif defaults and defaults.classes_csv:
        default_classes_flag = f'--classes-csv "{defaults.classes_csv}"'
    context.setdefault("classes_flag", default_classes_flag)
    context.setdefault("min_per_class", str(defaults.min_per_class if defaults else 80))
    context.setdefault("max_per_class", str(defaults.max_per_class if defaults else 120))
    context.setdefault("crawl_extra_flags", "--show-browser" if (defaults is None or defaults.crawl_show_browser) else "")
    print("인터랙티브 명령 안내입니다. 번호를 입력하면 명령을 확인하고, /run <번호>로 실제 실행, /set run_id <값>으로 갱신, /exit 로 종료하세요.")
    python_exec = shlex.quote(str(Path(sys.executable)))
    env_cache: Dict[str, str] = {}
    while True:
        try:
            user_input = input("pipeline> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"/help", "help"}:
            print("사용 가능한 섹션:")
            for key, info in sections.items():
                print(f"  {key}. {info['title']}")
            print("번호를 입력하면 해당 명령을 표시하고, /exit 로 종료합니다.")
        elif lowered in {"/context", "context"}:
            print("현재 컨텍스트:", context)
        elif lowered.startswith("/set"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("사용법: /set <key> <value>")
                continue
            key, raw_value = parts[1], parts[2]
            if not set_context_value(context, key, raw_value):
                continue
            print(f"{key} = {context[key]}")
            if key == "run_id" and "base_run_id" not in context:
                context["base_run_id"] = context["run_id"]
                context.setdefault("default_run_id", context["run_id"])
        elif lowered in {"/exit", "exit", "quit"}:
            break
        elif user_input in sections:
            info = sections[user_input]
            print(f"\n[{user_input}] {info['title']}")
            body = info["body"].strip()
            if body:
                print(apply_context(body, context))
            blocks = info.get("blocks") or []
            if blocks:
                for block in blocks:
                    formatted_block = apply_context(block, context)
                    print("```bash")
                    print(formatted_block)
                    print("```")
            if not body and not blocks:
                print("(명령이 정의되어 있지 않습니다)")
            print()
        elif lowered.startswith("/run"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or parts[1] not in sections:
                print("사용법: /run <번호>")
                continue

            command_number = parts[1]
            info = sections[command_number]
            blocks = info.get("blocks") or []
            if not blocks:
                print("실행 가능한 명령이 없습니다.")
                continue

            run_context = context.copy()
            run_context.setdefault("train_model_flag", "")
            base_run_for_14: str | None = None
            parent_dataset_run: str | None = None
            parent_dataset_hash: str | None = None

            if command_number == "1":
                class_flag_default = context.get("classes_flag", "--classes 기장 보리 조")
                run_context.setdefault("classes_flag", class_flag_default)
                run_context.setdefault("min_per_class", context.get("min_per_class", "80"))
                run_context.setdefault("max_per_class", context.get("max_per_class", "120"))
                run_context.setdefault("crawl_extra_flags", context.get("crawl_extra_flags", "--show-browser"))

                class_input = input("수집할 클래스 입력 (공백 구분, Enter=CSV 사용): ").strip()
                if class_input:
                    run_context["classes_flag"] = "--classes " + class_input
                else:
                    csv_default = defaults.classes_csv if (defaults and defaults.classes_csv) else "target_food.csv"
                    csv_path = input(f"클래스 CSV 경로 (Enter={csv_default}): ").strip() or str(csv_default)
                    run_context["classes_flag"] = f'--classes-csv "{csv_path}"'

                min_default = run_context["min_per_class"]
                min_input = input(f"클래스당 최소 이미지 수 (Enter={min_default}): ").strip()
                if min_input:
                    run_context["min_per_class"] = min_input

                max_default = run_context["max_per_class"]
                max_input = input(f"클래스당 최대 이미지 수 (Enter={max_default}): ").strip()
                if max_input:
                    run_context["max_per_class"] = max_input

                flags: List[str] = []
                show_ans = input("크롤 시 브라우저 표시할까요? (Y/n): ").strip().lower()
                if show_ans in {"", "y", "yes"}:
                    flags.append("--show-browser")
                limit_input = input("전체 다운로드 한도 (Enter=제한 없음): ").strip()
                if limit_input.isdigit():
                    flags.append(f"--limit {limit_input}")
                start_from = input("특정 클래스부터 시작 (Enter=전체): ").strip()
                if start_from:
                    flags.append(f'--start-from "{start_from}"')
                run_context["crawl_extra_flags"] = " ".join(flags)

            if command_number in {"13", "14"}:
                base_run_id = context.get("base_run_id") or context.get("run_id")
                if not base_run_id:
                    print("[error] 현재 base_run_id/run_id가 설정되지 않았습니다. 먼저 /set base_run_id <값> 또는 /set run_id <값>으로 설정하세요.")
                    continue

                new_run_id = compute_next_al_run_id(base_run_id, root, context.get("run_id"))

                print("[info] 액티브 러닝을 위해 새 run_id를 생성합니다.")
                print(f"       base_run_id: {base_run_id}")
                print(f"       new_run_id : {new_run_id}")

                run_context["run_id"] = new_run_id
                run_context["base_run_id"] = base_run_id
                base_run_for_14 = base_run_id
                dataset_root = root / "data" / "5_datasets"
                parent_candidate: str | None = None
                if "_r" in new_run_id:
                    prefix, suffix = new_run_id.rsplit("_r", 1)
                    if suffix.isdigit():
                        idx = int(suffix)
                        if idx > 2:
                            parent_candidate = f"{prefix}_r{idx - 1}"
                        else:
                            parent_candidate = prefix
                else:
                    parent_candidate = base_run_id
                if parent_candidate:
                    parent_dir = dataset_root / parent_candidate
                    parent_hash = hash_train_manifest(parent_dir)
                    if parent_hash:
                        parent_dataset_run = parent_candidate
                        parent_dataset_hash = parent_hash

            if command_number == "11":
                dataset_root = root / "data" / "5_datasets"
                base_run_id = context.get("base_run_id")
                default_run = (
                    context.get("initial_run_id")
                    or context.get("default_run_id")
                    or context.get("run_id")
                    or base_run_id
                    or ""
                )
                selected_run: str | None = None

                available_runs: List[str] = []
                if dataset_root.exists():
                    available_runs = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
                if available_runs:
                    print("======= 사용 가능한 데이터셋(run_id) =======")
                    for idx, run_name in enumerate(available_runs, start=1):
                        print(f"{idx}. {run_name}")
                    print("==========================================")
                print("예시 입력: 'crawl_test_b_r2' 또는 'r2'(base_run_id + suffix)")
                prompt_default = default_run or (available_runs[0] if available_runs else "")
                prompt_label = prompt_default or "<필수 입력>"
                print(f"Enter를 누르면 기본값 {prompt_label or '<필수 입력>'}로 학습합니다.")
                choice = input("학습에 사용할 run_id 입력: ").strip()
                normalized_choice = choice
                if not normalized_choice:
                    normalized_choice = prompt_default
                elif base_run_id and normalized_choice.startswith("r") and not normalized_choice.startswith(base_run_id):
                    normalized_choice = f"{base_run_id}_{normalized_choice}"
                if normalized_choice and normalized_choice not in available_runs:
                    suffix = normalized_choice.lstrip("_")
                    matches = [run for run in available_runs if run.endswith(f"_{suffix}")]
                    if len(matches) == 1:
                        normalized_choice = matches[0]
                selected_run = normalized_choice

                if not selected_run:
                    print("[error] 학습 run_id를 결정하지 못했습니다. /set run_id <값> 또는 /set base_run_id <값>으로 설정하세요.")
                    continue

                dataset_yaml = dataset_root / selected_run / f"{selected_run}.yaml"
                if not dataset_yaml.exists():
                    print(f"[error] 데이터셋 YAML을 찾을 수 없습니다: {dataset_yaml}")
                    continue

                run_context["run_id"] = selected_run

                run_context["train_model_flag"] = ""
                parent_weights = None
                next_parent = None
                if selected_run and "_r" in selected_run:
                    base_prefix, suffix = selected_run.rsplit("_r", 1)
                    if suffix.isdigit():
                        idx = int(suffix)
                        if idx > 2:
                            next_parent = f"{base_prefix}_r{idx - 1}"
                        else:
                            next_parent = base_prefix
                elif base_run_id and selected_run != base_run_id:
                    next_parent = base_run_id

                if next_parent:
                    candidate = root / "models" / "runs" / next_parent / "weights" / "best.pt"
                    if candidate.exists():
                        parent_weights = candidate
                if parent_weights:
                    run_context["train_model_flag"] = f"--model {parent_weights}"
                    print(f"[info] Using pretrained weights: {parent_weights}")

            if command_number == "12":
                metrics_dir = root / "data" / "meta" / "train_metrics"
                available_runs: List[str] = []
                if metrics_dir.exists():
                    for path in sorted(metrics_dir.glob("*_per_class.csv")):
                        name = path.name.replace("_per_class.csv", "")
                        available_runs.append(name)
                default_run = context.get("run_id") or (available_runs[0] if available_runs else "")
                if available_runs:
                    print("======= 사용 가능한 학습 run_id =======")
                    for idx, run_name in enumerate(available_runs, start=1):
                        print(f"{idx}. {run_name}")
                    print("=======================================")
                prompt_label = default_run or "<필수 입력>"
                print(f"Enter를 누르면 기본값 {prompt_label}로 리포트를 생성합니다.")
                choice = input("리포트에 사용할 run_id 입력: ").strip()
                if not choice:
                    choice = default_run
                if choice and available_runs and choice not in available_runs:
                    matches = [name for name in available_runs if name.endswith(choice)]
                    if len(matches) == 1:
                        choice = matches[0]
                if not choice:
                    print("[error] 리포트 run_id를 결정하지 못했습니다. /set run_id <값>으로 설정하세요.")
                    continue
                run_context["run_id"] = choice

            for block in blocks:
                if not ensure_context_placeholders(block, run_context):
                    continue
                command_text = apply_context(block, run_context)
                env = ensure_env(command_text, env_cache)
                if env is None:
                    continue
                prepared = re.sub(r"(^|\n)python(\s)", rf"\1{python_exec}\2", command_text)
                print("[run] 명령 실행:")
                print(prepared)
                try:
                    subprocess.run(["bash", "-lc", prepared], check=True, env=env)
                except subprocess.CalledProcessError as exc:
                    print(f"[run] 명령이 실패했습니다 (exit {exc.returncode}). 로그를 확인하세요.")
            if command_number in {"13", "14"} and base_run_for_14:
                context["base_run_id"] = context.get("base_run_id") or base_run_for_14
                context["run_id"] = run_context.get("run_id", context.get("run_id"))
                if parent_dataset_hash:
                    dataset_root = root / "data" / "5_datasets"
                    new_dataset_dir = dataset_root / run_context["run_id"]
                    new_hash = hash_train_manifest(new_dataset_dir)
                    if new_hash and new_hash == parent_dataset_hash:
                        prompt = (
                            f"[warn] 새 데이터셋({run_context['run_id']})이 {parent_dataset_run}과 동일합니다. "
                            "진행하시겠습니까? (y/N): "
                        )
                        answer = input(prompt).strip().lower()
                        if answer not in {"y", "yes"}:
                            if new_dataset_dir.exists():
                                shutil.rmtree(new_dataset_dir, ignore_errors=True)
                            print("[info] 동일한 데이터셋을 삭제했습니다. 새 run_id를 다시 생성하세요.")
                            continue
                        print("[info] 사용자 확인으로 동일한 데이터셋을 유지합니다.")
            if command_number == "11":
                context["run_id"] = run_context.get("run_id", context.get("run_id"))
                context["train_model_flag"] = run_context.get("train_model_flag", context.get("train_model_flag", ""))
            if command_number == "12":
                context["run_id"] = run_context.get("run_id", context.get("run_id"))
            if command_number == "1":
                context["classes_flag"] = run_context.get("classes_flag", context.get("classes_flag", ""))
                context["min_per_class"] = run_context.get("min_per_class", context.get("min_per_class", "80"))
                context["max_per_class"] = run_context.get("max_per_class", context.get("max_per_class", "120"))
                context["crawl_extra_flags"] = run_context.get("crawl_extra_flags", context.get("crawl_extra_flags", ""))
        else:
            print("알 수 없는 입력입니다. /help 로 목록을 보거나 /exit 로 종료하세요.")


def parse_args() -> PipelineOptions:
    parser = argparse.ArgumentParser(description="Run the Food Calorie Vision pipeline end-to-end.")
    parser.add_argument("--run-id", help="Run identifier (YYYYMMDD_stage_seq).")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=DEFAULT_STEPS,
        help=f"Pipeline steps to execute. Default: {', '.join(DEFAULT_STEPS)}",
    )
    parser.add_argument("--classes", nargs="+", default=[], help="Explicit class names (overrides CSV).")
    parser.add_argument("--classes-csv", type=Path, default=Path("target_food.csv"), help="CSV for target classes.")
    parser.add_argument("--start-from", type=str, help="Start crawling from this class (inclusive).")
    parser.add_argument("--min-per-class", type=int, default=50, help="Minimum crawl attempts per class.")
    parser.add_argument("--max-per-class", type=int, default=120, help="Maximum crawl attempts per class.")
    parser.add_argument("--crawl-limit", type=int, help="Global crawl limit (optional).")
    parser.add_argument("--crawl-delay", type=float, default=1.5, help="Delay between downloads.")
    parser.add_argument("--show-browser", action="store_true", help="Open Chromium window during crawl.")
    parser.add_argument("--skip-icons", action="store_true", help="Skip icon removal step.")
    parser.add_argument("--auto-device", default="auto", help="YOLO inference device (auto, cpu, 0, ...).")
    parser.add_argument("--auto-batch", type=int, default=16, help="YOLO batch size.")
    parser.add_argument("--auto-conf", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--auto-iou", type=float, default=0.5, help="YOLO IOU threshold.")
    parser.add_argument("--review-threshold", type=float, default=0.3, help="Review queue max confidence threshold.")
    parser.add_argument("--auto-weights", type=Path, default=Path("models") / "yolo11l.pt", help="YOLO weights path.")
    parser.add_argument("--per-class-export", action="store_true", help="Package only one image per class for exports.")
    parser.add_argument("--package-source", type=Path, help="Override default validated labels location.")
    parser.add_argument("--val-ratio", type=float, default=0.3, help="Validation split ratio for dataset packaging.")
    parser.add_argument(
        "--postprocess-label-map",
        type=Path,
        default=Path("food_class_pre_label.csv"),
        help="Label map CSV for postprocess remapping (default: food_class_pre_label.csv).",
    )
    parser.add_argument(
        "--package-label-map",
        type=Path,
        help="Label map CSV for dataset packaging (default: food_class_after_label.csv).",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help="(Deprecated) Alias for --package-label-map.",
    )
    parser.add_argument("--train-config", type=Path, default=Path("configs/food_poc.yaml"), help="Training config file.")
    parser.add_argument("--train-model", type=Path, help="Override model weights path for training.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--interactive", action="store_true", help="Open interactive command menu.")
    parser.add_argument(
        "--copy-predictions",
        action="store_true",
        help="Also copy YOLO predictions into annotations when exporting Label Studio JSON.",
    )
    parser.add_argument("--ls-project-id", type=int, help="Label Studio project ID for automated exports.")
    parser.add_argument("--ls-base-url", default="http://localhost:8080", help="Label Studio base URL.")
    parser.add_argument("--ls-token", help="Label Studio access token (PAT or legacy).")

    default_args = None
    if len(sys.argv) == 1:
        default_args = ["--interactive"]

    args = parser.parse_args(default_args)

    if not args.interactive and not args.run_id:
        parser.error("--run-id is required unless --interactive is used.")

    root = Path(__file__).resolve().parents[1]
    package_label_map = args.package_label_map or args.label_map or Path("food_class_after_label.csv")
    postprocess_label_map = args.postprocess_label_map

    if postprocess_label_map and not postprocess_label_map.is_absolute():
        postprocess_label_map = root / postprocess_label_map
    if package_label_map and not package_label_map.is_absolute():
        package_label_map = root / package_label_map

    return PipelineOptions(
        run_id=args.run_id or "",
        steps=[step.lower() for step in args.steps],
        classes=args.classes,
        classes_csv=args.classes_csv,
        start_from=args.start_from,
        min_per_class=args.min_per_class,
        max_per_class=args.max_per_class,
        crawl_limit=args.crawl_limit,
        crawl_delay=args.crawl_delay,
        crawl_show_browser=args.show_browser,
        auto_device=args.auto_device,
        auto_batch=args.auto_batch,
        auto_conf=args.auto_conf,
        auto_iou=args.auto_iou,
        review_threshold=args.review_threshold,
        auto_weights=args.auto_weights,
        export_per_class_only=args.per_class_export,
        package_source=args.package_source,
        val_ratio=args.val_ratio,
        postprocess_label_map=postprocess_label_map,
        package_label_map=package_label_map,
        train_config=args.train_config,
        train_model=args.train_model,
        dry_run=args.dry_run,
        skip_icons=args.skip_icons,
        interactive=args.interactive,
        copy_predictions=args.copy_predictions,
        ls_project_id=args.ls_project_id,
        ls_base_url=args.ls_base_url,
        ls_token=args.ls_token,
    )


def main() -> None:
    options = parse_args()
    if options.interactive:
        run_interactive_menu(options)
        return
    runner = PipelineRunner(options)
    runner.run()


if __name__ == "__main__":
    main()
