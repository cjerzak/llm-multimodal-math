#!/usr/bin/env python3
"""
run_all.py - Configuration, DAG definition, and pipeline orchestration for the
multimodal multiplication benchmark.

This file serves two roles:
1. Configuration / inspection utility (`--show-config`, `--show-models`, `--python-cmd`)
2. Resource-aware pipeline scheduler, invoked by `run_all.sh`
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell environment

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "Scripts"
PAPER_DIR = REPO_ROOT / "PaperTexFolder"
FIGURES_DIR = PAPER_DIR / "Figures"
RESULTS_DIR = REPO_ROOT / "SavedResults"
TIMING_LOG = RESULTS_DIR / "pipeline_timing.csv"
DEFAULT_LOG_DIR = REPO_ROOT / "Logs"
DEFAULT_LOCK_DIR = REPO_ROOT / "Tmp" / "locks"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core.TinkerStartup import (
    create_tinker_service_client,
    format_tinker_startup_config,
    load_tinker_startup_config,
)

# =============================================================================
# PROBLEM SIZE CONFIGURATION
# =============================================================================

MULTIMODAL_COUNT = 10000
HDS_TARGET_COUNT = 1000
LORA_TRAINING_COUNT = 1000
SPLIT_RATIOS = "70/15/15"
MULTIMODAL_COMPLEXITY_MIN = 10
MULTIMODAL_COMPLEXITY_MAX = 324
HDS_COMPLEXITY_MIN = 12
HDS_COMPLEXITY_MAX = 324

LORA_EPOCHS = 3
LORA_EARLY_STOP = 2
LORA_BATCH_SIZE = 16
LORA_TOKENIZE_WORKERS = 4

FINGERPRINT_BATCH_SIZE = 30

MODELS = [
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
]

RESOURCE_PROFILES: Dict[str, Dict[str, int]] = {
    "studio": {"tinker_api": 2, "local_cpu": 3, "merge_io": 1},
    "m4": {"tinker_api": 2, "local_cpu": 3, "merge_io": 1},
    "mini": {"tinker_api": 2, "local_cpu": 3, "merge_io": 1},
    "pop": {"tinker_api": 2, "local_cpu": 3, "merge_io": 1},
    "default": {"tinker_api": 2, "local_cpu": 3, "merge_io": 1},
}

PARTIAL_CANONICAL_FIGURE_JOBS = {
    "merge_results_macros",
    "merge_nudge_appendix",
    "merge_similarity_appendix",
    "merge_fingerprint_appendix",
    "merge_embedding_appendix",
}

PHASES = ("data", "experiment", "figures")


@dataclass(frozen=True)
class PipelineJob:
    job_id: str
    deps: Tuple[str, ...]
    command: Tuple[str, ...]
    phase: str
    resource_class: str
    resource_units: int = 1
    artifact_group: str = "default"
    description: str = ""
    order: int = 0

    @property
    def command_display(self) -> str:
        return shlex.join(self.command)


@dataclass
class RunningJob:
    job: PipelineJob
    process: subprocess.Popen[str]
    started_monotonic: float
    started_wall: float
    stdout_path: Path
    stderr_path: Path
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread


def get_python_cmd() -> str:
    """Get the Python command to use for pipeline jobs."""
    env_python = os.getenv("PYTHON")
    if env_python:
        return env_python

    miniconda_python = Path.home() / "miniconda3" / "envs" / "tm_env" / "bin" / "python"
    miniforge_python = Path.home() / "miniforge3" / "envs" / "tm_env" / "bin" / "python"
    if miniconda_python.exists():
        return str(miniconda_python)
    if miniforge_python.exists():
        return str(miniforge_python)
    return sys.executable


def validate_tinker_runtime() -> Optional[str]:
    """Return an error message if the current interpreter cannot initialize Tinker."""
    startup_config = load_tinker_startup_config()
    startup_settings = format_tinker_startup_config(startup_config)
    refresh_hint = (
        "Refresh tm_env with:\n"
        "  conda env update -n tm_env --prune -f environment.yml\n"
        "  conda run -n tm_env python -m pip install --upgrade -r requirements-tm_env.txt"
    )

    try:
        import tinker
    except Exception as exc:
        return (
            f"ERROR: Tinker runtime preflight failed for interpreter {sys.executable}.\n"
            f"Unable to import `tinker`: {type(exc).__name__}: {exc}\n"
            f"Startup settings: {startup_settings}\n"
            f"{refresh_hint}"
        )

    service_client = None
    try:
        service_client = create_tinker_service_client(
            tinker_module=tinker,
            config=startup_config,
        )
    except Exception as exc:
        return (
            f"ERROR: Tinker runtime preflight failed for interpreter {sys.executable}.\n"
            "Unable to initialize `tinker.ServiceClient()` during `init_api`: "
            f"{type(exc).__name__}: {exc}\n"
            f"Startup settings: {startup_settings}\n"
            f"{refresh_hint}"
        )
    finally:
        if service_client is not None and hasattr(service_client, "close"):
            try:
                service_client.close()
            except Exception:
                pass

    return None


def get_resource_profile(machine_tag: str) -> Dict[str, int]:
    return dict(RESOURCE_PROFILES.get(machine_tag.lower(), RESOURCE_PROFILES["default"]))


def get_config() -> dict:
    """Get all configuration as a dictionary."""
    return {
        "repo_root": str(REPO_ROOT),
        "scripts_dir": str(SCRIPTS_DIR),
        "paper_dir": str(PAPER_DIR),
        "results_dir": str(RESULTS_DIR),
        "timing_log": str(TIMING_LOG),
        "python_cmd": get_python_cmd(),
        "data_generation": {
            "multimodal_count": MULTIMODAL_COUNT,
            "multimodal_complexity_min": MULTIMODAL_COMPLEXITY_MIN,
            "multimodal_complexity_max": MULTIMODAL_COMPLEXITY_MAX,
            "hds_target_count": HDS_TARGET_COUNT,
            "hds_complexity_min": HDS_COMPLEXITY_MIN,
            "hds_complexity_max": HDS_COMPLEXITY_MAX,
            "lora_training_count": LORA_TRAINING_COUNT,
            "split_ratios": SPLIT_RATIOS,
        },
        "lora_training": {
            "epochs": LORA_EPOCHS,
            "early_stop_patience": LORA_EARLY_STOP,
            "batch_size": LORA_BATCH_SIZE,
            "tokenize_workers": LORA_TOKENIZE_WORKERS,
        },
        "fingerprinting": {
            "batch_size": FINGERPRINT_BATCH_SIZE,
        },
        "orchestration": {
            "resource_profiles": RESOURCE_PROFILES,
        },
        "models": MODELS,
    }


def _figure_path(name: str) -> str:
    return str(FIGURES_DIR / name)


def _add_job(
    jobs: List[PipelineJob],
    *,
    job_id: str,
    deps: Sequence[str],
    command: Sequence[str],
    phase: str,
    resource_class: str,
    resource_units: int = 1,
    artifact_group: str = "default",
    description: str = "",
) -> None:
    jobs.append(
        PipelineJob(
            job_id=job_id,
            deps=tuple(deps),
            command=tuple(command),
            phase=phase,
            resource_class=resource_class,
            resource_units=resource_units,
            artifact_group=artifact_group,
            description=description,
            order=len(jobs),
        )
    )


def build_pipeline_jobs(python_cmd: Optional[str] = None) -> List[PipelineJob]:
    """Build the full pipeline DAG."""
    py = python_cmd or get_python_cmd()
    jobs: List[PipelineJob] = []

    # Phase 1: data generation
    _add_job(
        jobs,
        job_id="hds_gen",
        deps=(),
        command=(
            py,
            "Scripts/generators/GenerateHDS.py",
            "--count",
            "1000",
            "--complexity-min",
            str(HDS_COMPLEXITY_MIN),
            "--complexity-max",
            str(HDS_COMPLEXITY_MAX),
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Generate HDS/Traps tabular data",
    )
    _add_job(
        jobs,
        job_id="multimodal_grid",
        deps=(),
        command=(
            py,
            "Scripts/generators/GenerateMathText.py",
            "--count",
            "10000",
            "--skip-text-files",
            "--force-regenerate-grid",
            "--complexity-min",
            str(MULTIMODAL_COMPLEXITY_MIN),
            "--complexity-max",
            str(MULTIMODAL_COMPLEXITY_MAX),
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Generate shared multimodal problem grid",
    )
    _add_job(
        jobs,
        job_id="math_images",
        deps=("multimodal_grid",),
        command=(
            py,
            "Scripts/generators/GenerateMathImages.py",
            "--count",
            "10000",
            "--complexity-min",
            str(MULTIMODAL_COMPLEXITY_MIN),
            "--complexity-max",
            str(MULTIMODAL_COMPLEXITY_MAX),
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Render multimodal image dataset",
    )
    _add_job(
        jobs,
        job_id="math_audio",
        deps=("multimodal_grid",),
        command=(
            py,
            "Scripts/generators/GenerateMathAudio.py",
            "--count",
            "10000",
            "--complexity-min",
            str(MULTIMODAL_COMPLEXITY_MIN),
            "--complexity-max",
            str(MULTIMODAL_COMPLEXITY_MAX),
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Render multimodal audio dataset",
    )
    _add_job(
        jobs,
        job_id="math_text",
        deps=("multimodal_grid",),
        command=(
            py,
            "Scripts/generators/GenerateMathText.py",
            "--count",
            "10000",
            "--complexity-min",
            str(MULTIMODAL_COMPLEXITY_MIN),
            "--complexity-max",
            str(MULTIMODAL_COMPLEXITY_MAX),
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Write multimodal text dataset",
    )
    _add_job(
        jobs,
        job_id="lora_data",
        deps=("hds_gen",),
        command=(
            py,
            "Scripts/generators/GenerateLoRATrainingData.py",
            "--count",
            "1000",
            "--exclude-test-problems",
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Generate LoRA training data",
    )
    _add_job(
        jobs,
        job_id="hds_images",
        deps=("hds_gen",),
        command=(
            py,
            "Scripts/generators/GenerateHDSImages.py",
            "--dataset",
            "HDSv2",
            "--split",
            "test",
        ),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Render HDS test images",
    )
    _add_job(
        jobs,
        job_id="traps_images",
        deps=("hds_gen",),
        command=(py, "Scripts/generators/GenerateHDSImages.py", "--dataset", "Trapsv2"),
        phase="data",
        resource_class="local_cpu",
        artifact_group="saved-data",
        description="Render Traps images",
    )

    # Phase 2: experiments
    fingerprint_jobs = [
        ("fp_hds_text_30b", ("hds_gen",), ("HDSv2", "test", "text", MODELS[0], None)),
        ("fp_hds_all_text_30b", ("hds_gen",), ("HDSv2", "all", "text", MODELS[0], None)),
        ("fp_traps_text_30b", ("hds_gen",), ("Trapsv2", "all", "text", MODELS[0], None)),
        ("fp_hds_text_235b", ("hds_gen",), ("HDSv2", "test", "text", MODELS[1], None)),
        ("fp_hds_all_text_235b", ("hds_gen",), ("HDSv2", "all", "text", MODELS[1], None)),
        ("fp_traps_text_235b", ("hds_gen",), ("Trapsv2", "all", "text", MODELS[1], None)),
        ("fp_hds_image_30b", ("hds_images",), ("HDSv2", "test", "image", MODELS[0], None)),
        ("fp_traps_image_30b", ("traps_images",), ("Trapsv2", "all", "image", MODELS[0], None)),
        ("fp_hds_image_235b", ("hds_images",), ("HDSv2", "test", "image", MODELS[1], None)),
        ("fp_traps_image_235b", ("traps_images",), ("Trapsv2", "all", "image", MODELS[1], None)),
        ("fp_hds_text_30b_style", ("hds_gen",), ("HDSv2", "test", "text", MODELS[0], "style_mismatch")),
        ("fp_hds_image_30b_style", ("hds_images",), ("HDSv2", "test", "image", MODELS[0], "style_mismatch")),
        ("fp_hds_text_235b_style", ("hds_gen",), ("HDSv2", "test", "text", MODELS[1], "style_mismatch")),
        ("fp_hds_image_235b_style", ("hds_images",), ("HDSv2", "test", "image", MODELS[1], "style_mismatch")),
    ]
    for job_id, deps, (dataset, split, modality, model, profile) in fingerprint_jobs:
        command = [
            py,
            "Scripts/experiments/BaselineFingerprint.py",
            "--model",
            model,
            "--dataset",
            dataset,
            "--split",
            split,
            "--batch-size",
            "30",
            "--async",
            "--concurrency",
            "4",
            "--template-mode",
            "multi",
            "--template-profile",
            "balanced" if profile is None else profile,
        ]
        if modality != "text":
            command.extend(["--modality", modality])
        if profile is not None:
            command.extend(["--output-tag", profile])
        _add_job(
            jobs,
            job_id=job_id,
            deps=deps,
            command=command,
            phase="experiment",
            resource_class="tinker_api",
            artifact_group="saved-results",
            description=f"Baseline fingerprinting ({dataset} {split} {modality})",
        )

    contrastive_jobs = [
        ("contrastive_hds_text_30b", ("hds_gen",), MODELS[0], "text"),
        ("contrastive_hds_image_30b", ("hds_images",), MODELS[0], "image"),
        ("contrastive_hds_text_235b", ("hds_gen",), MODELS[1], "text"),
        ("contrastive_hds_image_235b", ("hds_images",), MODELS[1], "image"),
    ]
    for job_id, deps, model, modality in contrastive_jobs:
        command = [
            py,
            "Scripts/experiments/ContrastiveStepProbe.py",
            "--model",
            model,
            "--dataset",
            "HDSv2",
            "--split",
            "test",
            "--modality",
            modality,
            "--template-mode",
            "multi",
            "--template-profile",
            "harder",
        ]
        _add_job(
            jobs,
            job_id=job_id,
            deps=deps,
            command=command,
            phase="experiment",
            resource_class="tinker_api",
            artifact_group="saved-results",
            description=f"Contrastive step probing ({modality})",
        )

    _add_job(
        jobs,
        job_id="lora_train_30b",
        deps=("lora_data",),
        command=(
            py,
            "Scripts/experiments/TrainHeuristicLoRAs.py",
            "--model",
            MODELS[0],
            "--heuristics",
            "RC",
            "DD",
            "OT",
            "STYLE",
            "--epochs",
            "3",
            "--early-stop-patience",
            "2",
            "--batch-size",
            "16",
            "--parallel",
        ),
        phase="experiment",
        resource_class="tinker_api",
        resource_units=2,
        artifact_group="saved-results",
        description="Train 30B LoRA adapters",
    )
    _add_job(
        jobs,
        job_id="lora_train_30b_seed123",
        deps=("lora_data",),
        command=(
            py,
            "Scripts/experiments/TrainHeuristicLoRAs.py",
            "--model",
            MODELS[0],
            "--heuristics",
            "RC",
            "DD",
            "OT",
            "--epochs",
            "3",
            "--early-stop-patience",
            "2",
            "--batch-size",
            "16",
            "--parallel",
            "--seed",
            "123",
            "--seed-control",
        ),
        phase="experiment",
        resource_class="tinker_api",
        resource_units=2,
        artifact_group="saved-results",
        description="Train 30B seed-control LoRA adapters",
    )
    _add_job(
        jobs,
        job_id="lora_train_235b",
        deps=("lora_data",),
        command=(
            py,
            "Scripts/experiments/TrainHeuristicLoRAs.py",
            "--model",
            MODELS[1],
            "--heuristics",
            "RC",
            "DD",
            "OT",
            "STYLE",
            "--epochs",
            "3",
            "--early-stop-patience",
            "2",
            "--batch-size",
            "16",
            "--parallel",
        ),
        phase="experiment",
        resource_class="tinker_api",
        resource_units=2,
        artifact_group="saved-results",
        description="Train 235B LoRA adapters",
    )
    _add_job(
        jobs,
        job_id="lora_train_235b_seed123",
        deps=("lora_data",),
        command=(
            py,
            "Scripts/experiments/TrainHeuristicLoRAs.py",
            "--model",
            MODELS[1],
            "--heuristics",
            "RC",
            "DD",
            "OT",
            "--epochs",
            "3",
            "--early-stop-patience",
            "2",
            "--batch-size",
            "16",
            "--parallel",
            "--seed",
            "123",
            "--seed-control",
        ),
        phase="experiment",
        resource_class="tinker_api",
        resource_units=2,
        artifact_group="saved-results",
        description="Train 235B seed-control LoRA adapters",
    )

    nudge_jobs = [
        ("nudge_text_30b", ("lora_train_30b",), MODELS[0], "text"),
        ("nudge_image_30b", ("lora_train_30b", "hds_images"), MODELS[0], "image"),
        ("nudge_text_235b", ("lora_train_235b",), MODELS[1], "text"),
        ("nudge_image_235b", ("lora_train_235b", "hds_images"), MODELS[1], "image"),
    ]
    for job_id, deps, model, modality in nudge_jobs:
        command = [
            py,
            "Scripts/experiments/LoRANudgeTest.py",
            "--model",
            model,
            "--split",
            "test",
            "--modality",
            modality,
            "--template-mode",
            "multi",
            "--include-style-control",
        ]
        if modality == "text":
            command.extend(["--batch-size", "1"])
        _add_job(
            jobs,
            job_id=job_id,
            deps=deps,
            command=command,
            phase="experiment",
            resource_class="tinker_api",
            artifact_group="saved-results",
            description=f"LoRA nudge evaluation ({modality})",
        )

    gradient_jobs = [
        ("gradient_30b", ("lora_train_30b", "lora_train_30b_seed123"), MODELS[0]),
        ("gradient_235b", ("lora_train_235b", "lora_train_235b_seed123"), MODELS[1]),
    ]
    for job_id, deps, model in gradient_jobs:
        _add_job(
            jobs,
            job_id=job_id,
            deps=deps,
            command=(py, "Scripts/experiments/GradientOrthogonality.py", "--model", model),
            phase="experiment",
            resource_class="local_cpu",
            artifact_group="saved-results",
            description="Gradient / update-geometry analysis",
        )

    # Phase 3: figures and paper artifacts
    macro_fragments = {
        "fp_macros_30b": _figure_path("results_macros.30b.fingerprint.tex"),
        "nudge_macros_30b": _figure_path("results_macros.30b.nudge.tex"),
        "gradient_macros_30b": _figure_path("results_macros.30b.gradient.tex"),
        "fp_macros_235b": _figure_path("results_macros.235b.fingerprint.tex"),
        "nudge_macros_235b": _figure_path("results_macros.235b.nudge.tex"),
        "gradient_macros_235b": _figure_path("results_macros.235b.gradient.tex"),
    }
    nudge_appendix_fragments = {
        "nudge_appendix_30b": _figure_path("appendix_nudge_examples.30b.tex"),
        "nudge_appendix_235b": _figure_path("appendix_nudge_examples.235b.tex"),
    }
    similarity_fragments = {
        "gradient_appendix_30b": _figure_path("appendix_similarity_matrix.30b.tex"),
        "gradient_appendix_235b": _figure_path("appendix_similarity_matrix.235b.tex"),
    }
    fingerprint_appendix_fragments = {
        "fingerprint_appendix_30b": _figure_path("appendix_template_variability.30b.tex"),
        "fingerprint_appendix_235b": _figure_path("appendix_template_variability.235b.tex"),
    }
    embedding_appendix_fragments = {
        "embedding_appendix_30b": _figure_path("appendix_embedding_results.30b.tex"),
        "embedding_appendix_235b": _figure_path("appendix_embedding_results.235b.tex"),
    }
    probe_hds_dataset = "HDSv2"

    _add_job(
        jobs,
        job_id="fingerprint_figs_30b",
        deps=("fp_hds_text_30b", "fp_traps_text_30b", "fp_hds_image_30b", "fp_traps_image_30b"),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--model",
            "Qwen3-VL-30B-A3B",
            "--output-type",
            "fingerprint-figures",
            "--probe-hds-dataset",
            probe_hds_dataset,
        ),
        phase="figures",
        resource_class="local_cpu",
        artifact_group="paper-figures",
        description="Generate canonical 30B fingerprint figures",
    )

    macro_jobs = [
        (
            "fp_macros_30b",
            (
                "fp_hds_text_30b",
                "fp_hds_text_30b_style",
                "fp_hds_all_text_30b",
                "fp_traps_text_30b",
                "fp_hds_image_30b",
                "fp_hds_image_30b_style",
                "fp_traps_image_30b",
                "contrastive_hds_text_30b",
                "contrastive_hds_image_30b",
            ),
            "Qwen3-VL-30B-A3B",
            "fingerprint-macros",
            macro_fragments["fp_macros_30b"],
        ),
        (
            "nudge_macros_30b",
            ("nudge_text_30b", "nudge_image_30b"),
            "Qwen3-VL-30B-A3B",
            "nudge-macros",
            macro_fragments["nudge_macros_30b"],
        ),
        (
            "gradient_macros_30b",
            ("gradient_30b",),
            "Qwen3-VL-30B-A3B",
            "gradient-macros",
            macro_fragments["gradient_macros_30b"],
        ),
        (
            "fp_macros_235b",
            (
                "fp_hds_text_235b",
                "fp_hds_text_235b_style",
                "fp_hds_all_text_235b",
                "fp_traps_text_235b",
                "fp_hds_image_235b",
                "fp_hds_image_235b_style",
                "fp_traps_image_235b",
                "contrastive_hds_text_235b",
                "contrastive_hds_image_235b",
            ),
            "Qwen3-VL-235B-A22B",
            "fingerprint-macros",
            macro_fragments["fp_macros_235b"],
        ),
        (
            "nudge_macros_235b",
            ("nudge_text_235b", "nudge_image_235b"),
            "Qwen3-VL-235B-A22B",
            "nudge-macros",
            macro_fragments["nudge_macros_235b"],
        ),
        (
            "gradient_macros_235b",
            ("gradient_235b",),
            "Qwen3-VL-235B-A22B",
            "gradient-macros",
            macro_fragments["gradient_macros_235b"],
        ),
    ]
    for job_id, deps, model, output_type, output_path in macro_jobs:
        _add_job(
            jobs,
            job_id=job_id,
            deps=deps,
            command=(
                py,
                "Scripts/analysis/GenerateResultsFigures.py",
                "--model",
                model,
                "--output-type",
                output_type,
                "--probe-hds-dataset",
                probe_hds_dataset,
                "--output-path",
                output_path,
            ),
            phase="figures",
            resource_class="local_cpu",
            artifact_group="paper-fragments",
            description=f"Generate {output_type} fragment for {model}",
        )

    appendix_jobs = [
        (
            "nudge_appendix_30b",
            ("nudge_text_30b", "nudge_image_30b"),
            "Qwen3-VL-30B-A3B",
            "nudge-appendix",
            nudge_appendix_fragments["nudge_appendix_30b"],
        ),
        (
            "nudge_appendix_235b",
            ("nudge_text_235b", "nudge_image_235b"),
            "Qwen3-VL-235B-A22B",
            "nudge-appendix",
            nudge_appendix_fragments["nudge_appendix_235b"],
        ),
        (
            "gradient_appendix_30b",
            ("gradient_30b",),
            "Qwen3-VL-30B-A3B",
            "gradient-appendix",
            similarity_fragments["gradient_appendix_30b"],
        ),
        (
            "gradient_appendix_235b",
            ("gradient_235b",),
            "Qwen3-VL-235B-A22B",
            "gradient-appendix",
            similarity_fragments["gradient_appendix_235b"],
        ),
        (
            "fingerprint_appendix_30b",
            ("fp_hds_text_30b", "fp_hds_text_30b_style", "fp_hds_image_30b", "fp_hds_image_30b_style"),
            "Qwen3-VL-30B-A3B",
            "fingerprint-appendix",
            fingerprint_appendix_fragments["fingerprint_appendix_30b"],
        ),
        (
            "fingerprint_appendix_235b",
            ("fp_hds_text_235b", "fp_hds_text_235b_style", "fp_hds_image_235b", "fp_hds_image_235b_style"),
            "Qwen3-VL-235B-A22B",
            "fingerprint-appendix",
            fingerprint_appendix_fragments["fingerprint_appendix_235b"],
        ),
        (
            "embedding_appendix_30b",
            ("fp_hds_image_30b",),
            "Qwen3-VL-30B-A3B",
            "embedding-appendix",
            embedding_appendix_fragments["embedding_appendix_30b"],
        ),
        (
            "embedding_appendix_235b",
            ("fp_hds_image_235b",),
            "Qwen3-VL-235B-A22B",
            "embedding-appendix",
            embedding_appendix_fragments["embedding_appendix_235b"],
        ),
    ]
    for job_id, deps, model, output_type, output_path in appendix_jobs:
        _add_job(
            jobs,
            job_id=job_id,
            deps=deps,
            command=(
                py,
                "Scripts/analysis/GenerateResultsFigures.py",
                "--model",
                model,
                "--output-type",
                output_type,
                "--probe-hds-dataset",
                probe_hds_dataset,
                "--output-path",
                output_path,
            ),
            phase="figures",
            resource_class="local_cpu",
            artifact_group="paper-fragments",
            description=f"Generate {output_type} fragment for {model}",
        )

    _add_job(
        jobs,
        job_id="training_appendix",
        deps=("lora_data",),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--output-type",
            "training-appendix",
            "--output-path",
            _figure_path("appendix_training_examples.tex"),
        ),
        phase="figures",
        resource_class="merge_io",
        artifact_group="paper-canonical",
        description="Generate canonical training appendix",
    )
    _add_job(
        jobs,
        job_id="merge_results_macros",
        deps=tuple(macro_fragments.keys()),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--output-type",
            "merge-macros",
            "--output-path",
            _figure_path("results_macros.tex"),
            "--fragment-path",
            macro_fragments["fp_macros_30b"],
            "--fragment-path",
            macro_fragments["nudge_macros_30b"],
            "--fragment-path",
            macro_fragments["gradient_macros_30b"],
            "--fragment-path",
            macro_fragments["fp_macros_235b"],
            "--fragment-path",
            macro_fragments["nudge_macros_235b"],
            "--fragment-path",
            macro_fragments["gradient_macros_235b"],
        ),
        phase="figures",
        resource_class="merge_io",
        artifact_group="paper-canonical",
        description="Merge all macro fragments into the canonical paper macro file",
    )
    _add_job(
        jobs,
        job_id="merge_nudge_appendix",
        deps=tuple(nudge_appendix_fragments.keys()),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--output-type",
            "merge-nudge-appendix",
            "--output-path",
            _figure_path("appendix_nudge_examples.tex"),
            "--fragment-path",
            nudge_appendix_fragments["nudge_appendix_30b"],
            "--fragment-path",
            nudge_appendix_fragments["nudge_appendix_235b"],
        ),
        phase="figures",
        resource_class="merge_io",
        artifact_group="paper-canonical",
        description="Merge model-specific nudge appendix fragments",
    )
    _add_job(
        jobs,
        job_id="merge_similarity_appendix",
        deps=tuple(similarity_fragments.keys()),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--output-type",
            "merge-gradient-appendix",
            "--output-path",
            _figure_path("appendix_similarity_matrix.tex"),
            "--fragment-path",
            similarity_fragments["gradient_appendix_30b"],
            "--fragment-path",
            similarity_fragments["gradient_appendix_235b"],
        ),
        phase="figures",
        resource_class="merge_io",
        artifact_group="paper-canonical",
        description="Merge model-specific similarity appendix fragments",
    )
    _add_job(
        jobs,
        job_id="merge_fingerprint_appendix",
        deps=tuple(fingerprint_appendix_fragments.keys()),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--output-type",
            "merge-fingerprint-appendix",
            "--output-path",
            _figure_path("appendix_template_variability.tex"),
            "--fragment-path",
            fingerprint_appendix_fragments["fingerprint_appendix_30b"],
            "--fragment-path",
            fingerprint_appendix_fragments["fingerprint_appendix_235b"],
        ),
        phase="figures",
        resource_class="merge_io",
        artifact_group="paper-canonical",
        description="Merge model-specific template-variability appendix fragments",
    )
    _add_job(
        jobs,
        job_id="merge_embedding_appendix",
        deps=tuple(embedding_appendix_fragments.keys()),
        command=(
            py,
            "Scripts/analysis/GenerateResultsFigures.py",
            "--output-type",
            "merge-embedding-appendix",
            "--output-path",
            _figure_path("appendix_embedding_results.tex"),
            "--fragment-path",
            embedding_appendix_fragments["embedding_appendix_30b"],
            "--fragment-path",
            embedding_appendix_fragments["embedding_appendix_235b"],
        ),
        phase="figures",
        resource_class="merge_io",
        artifact_group="paper-canonical",
        description="Merge model-specific embedding appendix fragments",
    )

    _validate_job_catalog(jobs)
    return jobs


def _validate_job_catalog(jobs: Sequence[PipelineJob]) -> None:
    seen: Set[str] = set()
    catalog_ids = {job.job_id for job in jobs}
    for job in jobs:
        if job.job_id in seen:
            raise ValueError(f"Duplicate job id: {job.job_id}")
        seen.add(job.job_id)
        if job.phase not in PHASES:
            raise ValueError(f"Unknown phase for job {job.job_id}: {job.phase}")
        if job.resource_units <= 0:
            raise ValueError(f"Job {job.job_id} must request at least one resource unit")
        for dep in job.deps:
            if dep not in catalog_ids:
                raise ValueError(f"Job {job.job_id} depends on unknown job {dep}")


def _job_matches_model_filter(job: PipelineJob, model_size: Optional[str]) -> bool:
    if not model_size:
        return True
    normalized = model_size.lower()
    if normalized not in {"30b", "235b"}:
        raise ValueError(f"Unknown model size filter: {model_size}")
    if job.job_id in PARTIAL_CANONICAL_FIGURE_JOBS:
        return False
    if "_30b" not in job.job_id and "_235b" not in job.job_id:
        return True
    return f"_{normalized}" in job.job_id


def select_pipeline_jobs(
    *,
    machine_tag: str,
    phase: str,
    skip_lora_training: bool = False,
    model_size: Optional[str] = None,
    python_cmd: Optional[str] = None,
) -> Tuple[List[PipelineJob], Set[str], Dict[str, int]]:
    catalog = build_pipeline_jobs(python_cmd=python_cmd)
    catalog_by_id = {job.job_id: job for job in catalog}

    if phase not in (*PHASES, "all"):
        raise ValueError(f"Unknown phase '{phase}'. Use: data, experiment, figures, or all")

    selected = [
        job
        for job in catalog
        if phase == "all" or job.phase == phase
    ]

    if skip_lora_training:
        selected = [job for job in selected if not job.job_id.startswith("lora_train_")]

    selected = [job for job in selected if _job_matches_model_filter(job, model_size)]

    selected_ids = {job.job_id for job in selected}
    precompleted: Set[str] = set()
    for job in selected:
        for dep in job.deps:
            if dep not in selected_ids:
                if dep not in catalog_by_id:
                    raise ValueError(f"Job {job.job_id} depends on unknown job {dep}")
                precompleted.add(dep)

    _validate_selected_graph(selected, precompleted)
    return selected, precompleted, get_resource_profile(machine_tag)


def _validate_selected_graph(jobs: Sequence[PipelineJob], precompleted: Set[str]) -> None:
    selected_ids = {job.job_id for job in jobs}
    remaining = {job.job_id: {dep for dep in job.deps if dep in selected_ids} for job in jobs}
    ready = [job_id for job_id, deps in remaining.items() if not deps]
    completed = set(precompleted)
    topo_completed: Set[str] = set()

    while ready:
        current = ready.pop()
        topo_completed.add(current)
        for job_id, deps in remaining.items():
            if current in deps:
                deps.remove(current)
                if not deps and job_id not in topo_completed and job_id not in ready:
                    ready.append(job_id)

    unresolved = selected_ids - topo_completed
    if unresolved:
        raise ValueError(f"Selected job graph is cyclic or unresolved: {sorted(unresolved)}")


def format_dry_run(
    jobs: Sequence[PipelineJob],
    *,
    resource_profile: Mapping[str, int],
    precompleted: Set[str],
) -> str:
    lines = [
        "",
        "═══════════════════════════════════════════════════════════════════════════════",
        "DRY RUN - PIPELINE PREVIEW",
        "═══════════════════════════════════════════════════════════════════════════════",
        "",
        f"Resource pools: {dict(resource_profile)}",
        "",
        f"Jobs to execute ({len(jobs)} total):",
        "",
    ]
    for job in jobs:
        dep_text = ", ".join(job.deps) if job.deps else "(none - starts immediately)"
        lines.extend(
            [
                f"[{job.job_id}]",
                f"  Phase: {job.phase}",
                f"  Waits for: {dep_text}",
                f"  Resource: {job.resource_class} x{job.resource_units}",
                f"  Artifact Group: {job.artifact_group}",
                f"  Command: {job.command_display}",
                "",
            ]
        )

    if precompleted:
        lines.append("Pre-satisfied dependencies:")
        for dep in sorted(precompleted):
            lines.append(f"  {dep}")
        lines.append("")

    lines.extend(
        [
            "═══════════════════════════════════════════════════════════════════════════════",
            f"Total: {len(jobs)} jobs",
            "═══════════════════════════════════════════════════════════════════════════════",
        ]
    )
    return "\n".join(lines)


def load_resumable_completed_jobs(lock_dir: Path, jobs: Sequence[PipelineJob]) -> Set[str]:
    """Load completed selected jobs from an existing lock directory."""
    if not lock_dir.exists():
        return set()

    selected_ids = {job.job_id for job in jobs}
    completed: Set[str] = set()
    for path in lock_dir.glob("*.done"):
        if path.stem in selected_ids:
            completed.add(path.stem)
    return completed


def init_lock_dir(lock_dir: Path, completed_markers: Iterable[str], *, resume: bool) -> None:
    if resume:
        lock_dir.mkdir(parents=True, exist_ok=True)
        failed_marker = lock_dir / "FAILED"
        if failed_marker.exists():
            failed_marker.unlink()
    else:
        shutil.rmtree(lock_dir, ignore_errors=True)
        lock_dir.mkdir(parents=True, exist_ok=True)

    for dep in completed_markers:
        (lock_dir / f"{dep}.done").touch()


def mark_done(lock_dir: Path, job_id: str) -> None:
    (lock_dir / f"{job_id}.done").touch()


def signal_failure(lock_dir: Path) -> None:
    (lock_dir / "FAILED").touch()


class PipelineScheduler:
    """Simple resource-aware DAG scheduler."""

    def __init__(
        self,
        *,
        jobs: Sequence[PipelineJob],
        resource_profile: Mapping[str, int],
        precompleted: Iterable[str],
        resumed_completed: Iterable[str],
        repo_root: Path,
        logs_dir: Path,
        lock_dir: Path,
        resume: bool,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.jobs = list(jobs)
        self.total_jobs = len(self.jobs)
        self.job_slots = {job.job_id: idx + 1 for idx, job in enumerate(self.jobs)}
        self.jobs_by_id = {job.job_id: job for job in self.jobs}
        self.job_depths = self._compute_job_depths()
        self.repo_root = repo_root
        self.logs_dir = logs_dir
        self.jobs_log_dir = logs_dir / "jobs"
        self.lock_dir = lock_dir
        self.resume = resume
        self.env = dict(env or os.environ)
        self.resource_profile = dict(resource_profile)
        self.available_resources = dict(resource_profile)
        self.precompleted = set(precompleted)
        self.resumed_completed = set(resumed_completed)
        self.completed: Set[str] = self.precompleted | self.resumed_completed
        self.selected_completed: Set[str] = set(self.resumed_completed)
        self.pending: Dict[str, PipelineJob] = {
            job.job_id: job for job in self.jobs if job.job_id not in self.selected_completed
        }
        self.ready_queue: List[str] = []
        self.ready_set: Set[str] = set()
        self.running: Dict[str, RunningJob] = {}
        self.failed = False
        self.exit_code = 0
        self.stdout_lock = threading.Lock()
        self.stderr_lock = threading.Lock()
        self.joblog_lock = threading.Lock()
        self.out_log_path = logs_dir / "pipeline_out.log"
        self.err_log_path = logs_dir / "pipeline_err.log"
        self.joblog_path = logs_dir / "pipeline_joblog.log"
        self._out_handle = None
        self._err_handle = None
        self._joblog_handle = None

    def run(self) -> int:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_log_dir.mkdir(parents=True, exist_ok=True)
        init_lock_dir(self.lock_dir, self.completed, resume=self.resume)

        if not self.resume:
            if self.out_log_path.exists():
                self.out_log_path.unlink()
            if self.err_log_path.exists():
                self.err_log_path.unlink()
            if self.joblog_path.exists():
                self.joblog_path.unlink()

        with open(self.out_log_path, "a", buffering=1) as out_handle, open(
            self.err_log_path, "a", buffering=1
        ) as err_handle, open(self.joblog_path, "a", buffering=1) as joblog_handle:
            self._out_handle = out_handle
            self._err_handle = err_handle
            self._joblog_handle = joblog_handle
            if self.resume:
                resume_banner = (
                    f"# RESUMED {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                    f"completed={len(self.selected_completed)}/{self.total_jobs}\n"
                )
                self._out_handle.write(resume_banner)
                self._err_handle.write(resume_banner)
                self._out_handle.flush()
                self._err_handle.flush()
            if self.joblog_path.stat().st_size == 0:
                self._joblog_handle.write(
                    "job_id\tphase\tresource_class\tresource_units\tstart_time\tend_time\tduration_seconds\texit_code\tcommand\n"
                )

            self._refresh_ready_queue()
            while self.pending or self.running or self.ready_queue:
                if not self.failed:
                    self._dispatch_ready_jobs()

                self._finalize_completed_jobs()

                if self.failed and not self.running:
                    break

                if not self.running and not self.ready_queue and self.pending:
                    unresolved = sorted(self.pending)
                    raise RuntimeError(f"Scheduler stalled with unresolved jobs: {unresolved}")

                time.sleep(0.1)

        return self.exit_code

    def _refresh_ready_queue(self) -> None:
        for job in self.jobs:
            if job.job_id not in self.pending:
                continue
            if job.job_id in self.ready_set:
                continue
            if all(dep in self.completed for dep in job.deps):
                self.ready_queue.append(job.job_id)
                self.ready_set.add(job.job_id)

    def _dispatch_ready_jobs(self) -> None:
        while True:
            startable = [
                job_id
                for job_id in self.ready_queue
                if self._can_start(self.pending[job_id])
            ]
            if not startable:
                break

            job_id = min(startable, key=self._ready_priority)
            self.ready_queue.remove(job_id)
            self.ready_set.remove(job_id)
            self._start_job(self.pending[job_id])
            self._refresh_ready_queue()

    def _can_start(self, job: PipelineJob) -> bool:
        available = self.available_resources.get(job.resource_class, 0)
        return available >= job.resource_units

    def _ready_priority(self, job_id: str) -> Tuple[int, int, int]:
        job = self.pending[job_id]
        # Prefer continuing already-unlocked branches before starting new heavy roots.
        return (-self.job_depths.get(job_id, 0), job.resource_units, job.order)

    def _compute_job_depths(self) -> Dict[str, int]:
        depths: Dict[str, int] = {}

        def depth(job_id: str) -> int:
            cached = depths.get(job_id)
            if cached is not None:
                return cached

            deps = [
                dep
                for dep in self.jobs_by_id[job_id].deps
                if dep in self.jobs_by_id
            ]
            if not deps:
                depths[job_id] = 0
                return 0

            value = 1 + max(depth(dep) for dep in deps)
            depths[job_id] = value
            return value

        for job_id in self.jobs_by_id:
            depth(job_id)
        return depths

    def _start_job(self, job: PipelineJob) -> None:
        self.available_resources[job.resource_class] -= job.resource_units
        self.pending.pop(job.job_id, None)

        stdout_path = self.jobs_log_dir / f"{job.job_id}.out.log"
        stderr_path = self.jobs_log_dir / f"{job.job_id}.err.log"
        started_monotonic = time.monotonic()
        started_wall = time.time()
        child_env = dict(self.env)
        child_env.setdefault("PYTHONUNBUFFERED", "1")

        self._log_status(
            f"[{time.strftime('%H:%M:%S')}] [{self._job_progress_label(job.job_id)}] STARTING "
            f"(resource={job.resource_class} x{job.resource_units})"
        )

        process = subprocess.Popen(
            list(job.command),
            cwd=self.repo_root,
            env=child_env,
            # Pipeline jobs are non-interactive; don't inherit a potentially revoked SSH tty.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        stdout_thread = threading.Thread(
            target=self._stream_pipe,
            args=(process.stdout, stdout_path, self._out_handle, self.stdout_lock, job.job_id),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._stream_pipe,
            args=(process.stderr, stderr_path, self._err_handle, self.stderr_lock, job.job_id),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        self.running[job.job_id] = RunningJob(
            job=job,
            process=process,
            started_monotonic=started_monotonic,
            started_wall=started_wall,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_thread=stdout_thread,
            stderr_thread=stderr_thread,
        )

    def _stream_pipe(
        self,
        stream,
        job_log_path: Path,
        aggregate_handle,
        aggregate_lock: threading.Lock,
        job_id: str,
    ) -> None:
        if stream is None:
            return
        with open(job_log_path, "a", buffering=1) as job_handle:
            for line in stream:
                job_handle.write(line)
                with aggregate_lock:
                    aggregate_handle.write(f"[{job_id}] {line}")
                    aggregate_handle.flush()

    def _finalize_completed_jobs(self) -> None:
        for job_id, running in list(self.running.items()):
            return_code = running.process.poll()
            if return_code is None:
                continue

            running.stdout_thread.join(timeout=1)
            running.stderr_thread.join(timeout=1)
            self.running.pop(job_id, None)
            self.available_resources[running.job.resource_class] += running.job.resource_units

            finished_wall = time.time()
            duration_seconds = time.monotonic() - running.started_monotonic
            self._write_joblog_entry(
                running.job,
                running.started_wall,
                finished_wall,
                duration_seconds,
                return_code,
            )

            if return_code == 0:
                self.completed.add(job_id)
                self.selected_completed.add(job_id)
                mark_done(self.lock_dir, job_id)
                self._log_status(
                    f"[{time.strftime('%H:%M:%S')}] [{self._job_progress_label(job_id)}] COMPLETE "
                    f"({duration_seconds:.1f}s)"
                )
                self._refresh_ready_queue()
                continue

            self._log_status(
                f"[{time.strftime('%H:%M:%S')}] "
                f"[{self._job_progress_label(job_id)}] FAILED (exit code: {return_code})",
                stderr=True,
            )
            if not self.failed:
                self.failed = True
                self.exit_code = return_code
                signal_failure(self.lock_dir)
                self._terminate_running_jobs(exclude=job_id)

    def _terminate_running_jobs(self, *, exclude: Optional[str] = None) -> None:
        for job_id, running in self.running.items():
            if job_id == exclude:
                continue
            if running.process.poll() is not None:
                continue
            self._log_status(
                f"[{time.strftime('%H:%M:%S')}] "
                f"[{self._job_progress_label(job_id)}] TERMINATING due to upstream failure",
                stderr=True,
            )
            self._kill_process_group(running.process, sig=signal.SIGTERM)

    def abort(self) -> None:
        """Terminate all running jobs."""
        self.failed = True
        if self.exit_code == 0:
            self.exit_code = 1
        signal_failure(self.lock_dir)
        self._terminate_running_jobs()

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[str], *, sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return

    def _write_joblog_entry(
        self,
        job: PipelineJob,
        started_wall: float,
        finished_wall: float,
        duration_seconds: float,
        exit_code: int,
    ) -> None:
        with self.joblog_lock:
            self._joblog_handle.write(
                "\t".join(
                    [
                        job.job_id,
                        job.phase,
                        job.resource_class,
                        str(job.resource_units),
                        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_wall)),
                        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(finished_wall)),
                        f"{duration_seconds:.3f}",
                        str(exit_code),
                        job.command_display,
                    ]
                )
                + "\n"
            )
            self._joblog_handle.flush()

    def _log_status(self, message: str, *, stderr: bool = False) -> None:
        handle = self._err_handle if stderr else self._out_handle
        lock = self.stderr_lock if stderr else self.stdout_lock
        with lock:
            print(message, file=sys.stderr if stderr else sys.stdout, flush=True)
            handle.write(message + "\n")
            handle.flush()

    def _job_progress_label(self, job_id: str) -> str:
        slot = self.job_slots[job_id]
        return f"{job_id}, {slot} of {self.total_jobs}"


def run_pipeline(
    *,
    machine_tag: str,
    phase: str,
    dry_run: bool,
    skip_lora_training: bool,
    model_size: Optional[str],
    resume: bool,
    logs_dir: Path,
    lock_dir: Path,
) -> int:
    jobs, precompleted, resource_profile = select_pipeline_jobs(
        machine_tag=machine_tag,
        phase=phase,
        skip_lora_training=skip_lora_training,
        model_size=model_size,
    )

    if resume and not lock_dir.exists():
        print(
            f"WARNING: Resume requested but lock directory does not exist: {lock_dir}. "
            "Starting with 0 completed jobs.",
            file=sys.stderr,
        )
    resumed_completed = load_resumable_completed_jobs(lock_dir, jobs) if resume else set()

    if dry_run:
        print(
            format_dry_run(
                jobs,
                resource_profile=resource_profile,
                precompleted=precompleted,
            )
        )
        return 0

    if phase in {"experiment", "all"} and not os.getenv("TINKER_API_KEY"):
        print("ERROR: TINKER_API_KEY not set. Please add it to .env or export it.", file=sys.stderr)
        return 1
    if phase in {"experiment", "all"}:
        preflight_error = validate_tinker_runtime()
        if preflight_error:
            print(preflight_error, file=sys.stderr)
            return 1

    print("")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print("STARTING DEPENDENCY-AWARE PIPELINE")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print("")
    print(f"Machine: {machine_tag}")
    print(f"Phase: {phase}")
    print(f"Jobs: {len(jobs)}")
    print(f"Resource pools: {resource_profile}")
    print(f"Lock directory: {lock_dir}")
    if resume:
        print(f"Resume: YES ({len(resumed_completed)}/{len(jobs)} jobs already marked done)")
        print(f"Remaining jobs at start: {len(jobs) - len(resumed_completed)}/{len(jobs)}")
    if skip_lora_training:
        print("Skip LoRA training: YES (using cached adapters)")
    if model_size:
        print(f"Model Size Filter: {model_size}")
    if precompleted:
        print(f"Pre-satisfied dependencies: {', '.join(sorted(precompleted))}")
    print("")

    scheduler = PipelineScheduler(
        jobs=jobs,
        resource_profile=resource_profile,
        precompleted=precompleted,
        resumed_completed=resumed_completed,
        repo_root=REPO_ROOT,
        logs_dir=logs_dir,
        lock_dir=lock_dir,
        resume=resume,
    )

    try:
        exit_code = scheduler.run()
    except KeyboardInterrupt:
        scheduler.abort()
        signal_failure(lock_dir)
        print("\nPipeline interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        scheduler.abort()
        print(f"\nPipeline orchestration error: {exc}", file=sys.stderr)
        return 1

    print("")
    print("═══════════════════════════════════════════════════════════════════════════════")
    if exit_code == 0:
        print("PIPELINE COMPLETE")
    else:
        print(f"PIPELINE FAILED (exit code: {exit_code})")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print("")
    print(f"Logs available in: {logs_dir}/")
    print("  - pipeline_joblog.log  (scheduler job log)")
    print("  - pipeline_out.log     (stdout)")
    print("  - pipeline_err.log     (stderr)")
    print("  - jobs/                (per-job logs)")
    print("")

    completed = sorted(scheduler.selected_completed)
    print("═══ PIPELINE STATUS ═══")
    print(f"Lock directory: {lock_dir}")
    print(f"Status: {'FAILED' if exit_code else 'COMPLETE'}")
    print(f"Completed jobs: {len(completed)}/{len(jobs)}")
    if completed:
        print("")
        print("Completed:")
        for job_id in completed:
            print(f"  {job_id}")
    print("")
    return exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configuration and orchestration utility for the multimodal multiplication pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --show-config
  python run_all.py --show-models
  python run_all.py --python-cmd

  # Called by run_all.sh for execution:
  python run_all.py --run-pipeline --machine M4 --phase all
        """,
    )
    parser.add_argument("--show-config", action="store_true", help="Print full configuration as JSON")
    parser.add_argument("--show-models", action="store_true", help="Print list of models to evaluate")
    parser.add_argument("--python-cmd", action="store_true", help="Print Python command path")
    parser.add_argument("--run-pipeline", action="store_true", help="Execute the pipeline scheduler")
    parser.add_argument("--machine", type=str, default="default", help="Machine tag for resource profile")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=(*PHASES, "all"),
        help="Which phase of the pipeline to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview jobs without executing them")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing lock-dir .done markers instead of starting fresh",
    )
    parser.add_argument(
        "--skip-lora-training",
        action="store_true",
        help="Skip LoRA training jobs and assume cached adapters exist",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=("30b", "235b"),
        default=None,
        help="Run only jobs for the specified model size",
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOG_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR, help=argparse.SUPPRESS)
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not any([args.show_config, args.show_models, args.python_cmd, args.run_pipeline]):
        parser.print_help()
        return 0

    if args.show_config:
        print(json.dumps(get_config(), indent=2))

    if args.show_models:
        for model in MODELS:
            print(model)

    if args.python_cmd:
        print(get_python_cmd())

    if args.run_pipeline:
        return run_pipeline(
            machine_tag=args.machine,
            phase=args.phase,
            dry_run=args.dry_run,
            resume=args.resume,
            skip_lora_training=args.skip_lora_training,
            model_size=args.model_size,
            logs_dir=args.logs_dir,
            lock_dir=args.lock_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
