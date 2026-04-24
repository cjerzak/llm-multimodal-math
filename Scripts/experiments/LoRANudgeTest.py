#!/usr/bin/env python3
"""
LoRANudgeTest.py

Test if applying heuristic-specific LoRAs changes model performance on HDS problems.

This script:
1. Loads the base model and trained LoRA adapters
2. Evaluates each on HDS problems
3. Identifies "flips" where LoRA changes correctness
4. Analyzes whether LoRAs improve performance on their target heuristic

Usage:
    TINKER_API_KEY="..." python Scripts/LoRANudgeTest.py
    TINKER_API_KEY="..." python Scripts/LoRANudgeTest.py --split test
    TINKER_API_KEY="..." python Scripts/LoRANudgeTest.py --split all
    TINKER_API_KEY="..." python Scripts/LoRANudgeTest.py --split test --modality image
    TINKER_API_KEY="..." python Scripts/LoRANudgeTest.py --gen-max-tokens 256
"""

# =============================================================================
# MODEL CONFIGURATION - Using VLM for both text and image modalities
# =============================================================================
# Using the same VLM for both modalities enables apples-to-apples cross-modal
# comparison and allows LoRAs to be applied to both text and image inputs.
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
CONFIDENCE_THRESHOLD = 0.7
DEFAULT_GEN_MAX_TOKENS = 2048
PROGRESS_HEARTBEAT_ITEMS = 10
PROGRESS_HEARTBEAT_SECONDS = 60.0
TEXT_PROMPT_TEMPLATE = (
    "What is {a} × {b}? "
    "Show your work step by step, then give the final answer as \"Answer: <number>\"."
)
IMAGE_PROMPT_TEMPLATE = (
    "What is the answer to this multiplication problem? "
    "Show your work step by step, then give the final answer as \"Answer: <number>\"."
)
# =============================================================================

import os
import sys
import csv
import json
import time
import argparse
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable, cast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths (experiments/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
HDS_PATH = REPO_ROOT / "SavedData" / "HDSv2.csv"
HDS_IMAGES_DIR = REPO_ROOT / "SavedData" / "HDSv2Images"
TRAINING_SUMMARY_PATH = REPO_ROOT / "SavedResults" / "lora_training" / "training_summary.json"
OUTPUT_DIR = REPO_ROOT / "SavedResults" / "nudge_test"

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))

from core.TinkerClient import (
    VisionTinkerClient, get_multi_heuristic_templates,
    extract_answer, extract_answer_enhanced,
    _call_with_timeout, build_tinker_sampling_retry_config,
    get_effective_heuristic_template_metadata,
    get_heuristic_template_profile,
    set_heuristic_template_mode,
    set_heuristic_template_profile,
    get_heuristic_template_mode,
    get_heuristic_template_seed,
    validate_active_heuristic_templates,
)
from core.Logging import tprint

LOSS_DETECTION_SEMANTICS = "loss_argmin_over_aggregated_losses"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for compact heartbeat logging."""
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _ProgressHeartbeat:
    """Emit fixed-cadence progress heartbeats for long-running loops."""

    def __init__(
        self,
        label: str,
        total: int,
        item_interval: Optional[int] = None,
        time_interval_seconds: Optional[float] = None,
        emit_fn: Optional[Callable[[str], None]] = None,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.label = label
        self.total = total
        self.item_interval = item_interval or PROGRESS_HEARTBEAT_ITEMS
        self.time_interval_seconds = (
            time_interval_seconds
            if time_interval_seconds is not None
            else PROGRESS_HEARTBEAT_SECONDS
        )
        self.emit_fn = emit_fn or tprint
        self.now_fn = now_fn or time.monotonic

        now = self.now_fn()
        self.started_at = now
        self.last_logged_at = now
        self.last_logged_completed = 0

    def log_start(self, stage_label: str) -> None:
        """Log the start of a long-running stage."""
        self.emit_fn(
            f"  {stage_label} for {self.total} problems "
            f"(heartbeat every {self.item_interval} items or {int(self.time_interval_seconds)}s)..."
        )

    def maybe_log(self, completed: int, *, force: bool = False) -> None:
        """Emit a heartbeat when count/time thresholds or completion are reached."""
        if completed <= 0 and not force:
            return

        now = self.now_fn()
        should_log = (
            force
            or completed >= self.total
            or completed - self.last_logged_completed >= self.item_interval
            or now - self.last_logged_at >= self.time_interval_seconds
        )
        if not should_log or completed == self.last_logged_completed:
            return

        elapsed = _format_elapsed(now - self.started_at)
        self.emit_fn(f"    {self.label}: {completed}/{self.total} (elapsed {elapsed})")
        self.last_logged_completed = completed
        self.last_logged_at = now


def _get_template_context() -> Dict[str, Any]:
    """Return active heuristic-template metadata for saved artifacts."""
    return get_effective_heuristic_template_metadata(
        profile=get_heuristic_template_profile(),
        mode=get_heuristic_template_mode(),
        seed=get_heuristic_template_seed(),
    )


def _write_run_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    """Persist a run manifest beside saved nudge artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tprint(f"Saved run manifest to {manifest_path}")


@dataclass
class HDSRow:
    """A row from HDS.csv."""
    id: str
    a: int
    b: int
    product: int
    target_heuristic: str
    design_family: str = ""
    canonical_target_heuristic: str = ""


@dataclass
class EvaluationResult:
    """Result of evaluating a single problem."""
    hds_id: str
    a: int
    b: int
    product: int
    target_heuristic: str
    model_answer: Optional[int]
    is_correct: bool
    perplexity_losses: Dict[str, float]
    detected_heuristic: str
    # Neutral baseline for delta loss computation
    neutral_loss: Optional[float] = None
    delta_losses: Optional[Dict[str, float]] = None
    # Extraction metadata for output quality tracking
    extraction_confidence: Optional[float] = None
    extraction_strategy: Optional[str] = None
    is_truncated: Optional[bool] = None
    # Generation inputs/outputs for detailed JSONL logging
    prompt_text: Optional[str] = None
    generation_text: Optional[str] = None
    image_path: Optional[str] = None
    input_modality: Optional[str] = None


@dataclass
class NudgeResult:
    """Comparison between base and LoRA model."""
    hds_id: str
    a: int
    b: int
    target_heuristic: str
    base_correct: Optional[bool]
    base_correctness: str
    base_detected: str
    lora_correct: Optional[bool]
    lora_correctness: str
    lora_detected: str
    is_flip: Optional[bool]
    flip_type: str  # "improved", "degraded", "unchanged", "unknown"
    correctness_flip: Optional[bool]
    correctness_flip_type: str
    detection_flip: bool
    detection_flip_type: str


def _is_confident_eval(result: EvaluationResult, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Return True if extraction metadata indicates a reliable answer."""
    if result.extraction_confidence is None:
        return False
    if result.is_truncated is True:
        return False
    return result.extraction_confidence >= threshold


def _correctness_label(result: EvaluationResult) -> str:
    """Map evaluation correctness into correct/incorrect/unknown based on confidence."""
    if not _is_confident_eval(result):
        return "unknown"
    return "correct" if result.is_correct else "incorrect"


def _label_to_bool(label: str) -> Optional[bool]:
    """Convert correctness label into bool, preserving unknown."""
    if label == "correct":
        return True
    if label == "incorrect":
        return False
    return None


def _compute_correctness_flip(base_label: str, lora_label: str) -> Tuple[Optional[bool], str]:
    """Compute correctness flip flag and type from two correctness labels."""
    if base_label == "unknown" or lora_label == "unknown":
        return None, "unknown"
    if base_label == lora_label:
        return False, "unchanged"
    if base_label == "incorrect" and lora_label == "correct":
        return True, "improved"
    return True, "degraded"


def load_hds(split: str = "all") -> List[HDSRow]:
    """Load HDS dataset, optionally filtered by split.

    Args:
        split: Split to load ("train", "val", "test", or "all")

    Returns:
        List of HDSRow objects.
    """
    rows = []
    with open(HDS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check split filter
            if split != "all":
                row_split = row.get('split', '')
                if row_split != split:
                    continue

            rows.append(HDSRow(
                id=row['id'],
                a=int(row['a']),
                b=int(row['b']),
                product=int(row['product']),
                target_heuristic=row.get('design_family') or row['target_heuristic'],
                design_family=row.get('design_family') or row['target_heuristic'],
                canonical_target_heuristic=row.get('canonical_target_heuristic') or row['target_heuristic'],
            ))
    return rows


def load_adapter_paths(model_name: Optional[str] = None) -> Dict[str, Dict[str, Optional[str]]]:
    """Load trained LoRA adapter and state paths from training summary.

    Args:
        model_name: Full model name (e.g., "Qwen/Qwen3-VL-30B-A3B-Instruct").
                    If None, uses legacy default path.
    """
    if model_name:
        # Use model-specific training summary
        model_slug = model_name.split("/")[-1].replace("-Instruct", "")
        training_summary_path = REPO_ROOT / "SavedResults" / f"lora_training_{model_slug}" / "training_summary.json"
    else:
        # Fallback to legacy path
        training_summary_path = TRAINING_SUMMARY_PATH

    with open(training_summary_path, 'r') as f:
        summary = json.load(f)

    paths: Dict[str, Dict[str, Optional[str]]] = {}
    for h, data in summary['heuristics'].items():
        adapter_path = data.get('adapter_path')
        state_path = data.get('state_path')
        if adapter_path or state_path:
            paths[h] = {
                "adapter_path": adapter_path,
                "state_path": state_path
            }

    return paths


def _select_checkpoint_from_list(
    checkpoints: List[Any],
    adapter_name: str,
    model_name: str,
    checkpoint_type: str
) -> Optional[Any]:
    """Pick the newest checkpoint that matches adapter/model/size constraints."""
    # Determine model size threshold based on model name
    is_30b = "30B" in model_name
    if checkpoint_type == "sampler":
        size_max = 3_000_000_000 if is_30b else 15_000_000_000
        size_min = 0 if is_30b else 5_000_000_000
    else:  # training
        size_max = 10_000_000_000 if is_30b else 30_000_000_000
        size_min = 0 if is_30b else 15_000_000_000

    if checkpoint_type == "sampler":
        matches = [
            c for c in checkpoints
            if adapter_name in c.checkpoint_id
            and "sampler_weights" in c.checkpoint_id
            and size_min <= c.size_bytes <= size_max
        ]
    else:
        matches = [
            c for c in checkpoints
            if adapter_name in c.checkpoint_id
            and c.checkpoint_id.endswith("_state")
            and size_min <= c.size_bytes <= size_max
        ]

    if matches:
        return sorted(matches, key=lambda c: c.time, reverse=True)[0]
    return None


def discover_checkpoint_from_registry(
    service_client,
    adapter_name: str,
    model_name: str,
    checkpoint_type: str = "training"
) -> Optional[str]:
    """Discover a valid checkpoint path from Tinker's user checkpoint registry.

    This is a fallback mechanism when stored checkpoint paths expire or become
    inaccessible. It searches the user's available checkpoints by matching
    adapter name and model size.

    Args:
        service_client: Tinker ServiceClient instance
        adapter_name: Name of the adapter (e.g., "rc_lora", "dd_lora", "ot_lora")
        model_name: Full model name (e.g., "Qwen/Qwen3-VL-30B-A3B-Instruct")
        checkpoint_type: "training" for state paths (load_state), "sampler" for adapter paths

    Returns:
        Most recent matching checkpoint path, or None if not found.
    """
    try:
        rc = service_client.create_rest_client()
        result = rc.list_user_checkpoints(limit=100).result()
        checkpoints = result.checkpoints if hasattr(result, 'checkpoints') else result

        best = _select_checkpoint_from_list(checkpoints, adapter_name, model_name, checkpoint_type)
        if best:
            tprint(f"  [checkpoint discovery] Found match for {adapter_name}, using: {best.tinker_path}")
            return best.tinker_path

        tprint(f"  [checkpoint discovery] No matching checkpoints found for {adapter_name} ({checkpoint_type})")
        return None

    except Exception as e:
        tprint(f"  [checkpoint discovery] Error querying registry: {e}")
        return None


async def discover_checkpoint_from_registry_async(
    service_client,
    adapter_name: str,
    model_name: str,
    checkpoint_type: str = "training"
) -> Optional[str]:
    """Async checkpoint discovery using list_user_checkpoints_async."""
    try:
        rc = service_client.create_rest_client()
        result = await rc.list_user_checkpoints_async(limit=100)
        checkpoints = result.checkpoints if hasattr(result, 'checkpoints') else result

        best = _select_checkpoint_from_list(checkpoints, adapter_name, model_name, checkpoint_type)
        if best:
            tprint(f"  [checkpoint discovery] Found match for {adapter_name}, using: {best.tinker_path}")
            return best.tinker_path

        tprint(f"  [checkpoint discovery] No matching checkpoints found for {adapter_name} ({checkpoint_type})")
        return None

    except Exception as e:
        tprint(f"  [checkpoint discovery] Error querying registry (async): {e}")
        return None


def _is_infrastructure_error(exc: Exception) -> bool:
    """Return True when a failure is auth/network/session related."""
    infrastructure_types = (TimeoutError, ConnectionError)
    if isinstance(exc, infrastructure_types):
        return True

    try:
        import tinker
    except Exception:
        tinker = None

    if tinker is not None:
        typed_infra = tuple(
            exc_type
            for exc_type in (
                getattr(tinker, "APITimeoutError", None),
                getattr(tinker, "APIConnectionError", None),
                getattr(tinker, "AuthenticationError", None),
            )
            if exc_type is not None
        )
        if typed_infra and isinstance(exc, typed_infra):
            return True

        api_status_error = getattr(tinker, "APIStatusError", None)
        if api_status_error is not None and isinstance(exc, api_status_error):
            status_code = getattr(exc, "status_code", None)
            if status_code in {401, 408, 409, 429}:
                return True
            if isinstance(status_code, int) and status_code >= 500:
                return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "invalid jwt",
            "authentication",
            "connection error",
            "request timed out",
            "heartbeat",
        )
    )


def _should_fallback_to_checkpoint_registry(exc: Exception) -> bool:
    """Return True when the failure looks like a stale checkpoint reference."""
    if _is_infrastructure_error(exc):
        return False

    if isinstance(exc, (FileNotFoundError, KeyError)):
        return True

    try:
        import tinker
    except Exception:
        tinker = None

    if tinker is not None:
        api_status_error = getattr(tinker, "APIStatusError", None)
        if api_status_error is not None and isinstance(exc, api_status_error):
            status_code = getattr(exc, "status_code", None)
            if status_code in {400, 404}:
                return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "checkpoint",
            "tinker://",
            "not found",
            "missing",
            "path",
            "sampler_weights",
            "_state",
        )
    )


def _raise_nonrecoverable_checkpoint_error(
    action: str,
    adapter_name: str,
    checkpoint_path: str,
    exc: Exception,
) -> None:
    """Raise a contextual error when registry fallback should not run."""
    reason = "infrastructure/auth-related" if _is_infrastructure_error(exc) else "non-checkpoint"
    raise RuntimeError(
        f"Failed to {action} for '{adapter_name}' from {checkpoint_path}; "
        f"skipping registry fallback because the error was {reason} "
        f"({type(exc).__name__}: {exc})"
    ) from exc


class NudgeTester:
    """Test LoRA nudge effects on HDS problems.

    Uses VisionTinkerClient which supports both text-only and image+text inputs,
    allowing the same LoRAs to be tested on both modalities.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        adapter_paths: Optional[Dict[str, str]] = None,
        state_paths: Optional[Dict[str, str]] = None,
        generation_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
        prompt_template: str = TEXT_PROMPT_TEMPLATE,
        score_max_in_flight: int = 4
    ):
        """Initialize with Tinker API via VisionTinkerClient (which supports text-only).

        Args:
            api_key: Tinker API key
            model_name: Model to use (default: VLM)
            adapter_paths: Dict mapping adapter names (e.g., "rc_lora") to their Tinker checkpoint paths
            state_paths: Dict mapping adapter names to training state paths for load_state()
            generation_max_tokens: Max tokens for generation
            prompt_template: Prompt template for generation
            score_max_in_flight: Max forward() batches to keep in flight (pipelined)
        """
        # Use VisionTinkerClient for unified text/image support
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.client = VisionTinkerClient(
            model_name=self.model_name,
            api_key=api_key,
            verbose=True
        )

        # Store adapter checkpoint paths for sampling and state loading
        self.adapter_paths = adapter_paths or {}
        self.state_paths = state_paths or {}

        self.generation_max_tokens = generation_max_tokens
        self.prompt_template = prompt_template
        self.score_max_in_flight = score_max_in_flight

        # Cache for sampling clients
        self._sampling_clients: Dict[str, Any] = {}
        self._training_clients: Dict[str, Any] = {}
        self._sampling_locks: Dict[str, asyncio.Lock] = {}
        self._training_locks: Dict[str, asyncio.Lock] = {}

    def _get_sampling_client(self, adapter_name: Optional[str] = None):
        """Get or create a sampling client for base model or LoRA.

        For base model: Creates fresh training client and saves weights.
        For LoRA adapters: Loads from previously saved checkpoint using the adapter path.

        If the stored checkpoint reference looks stale or missing, automatically
        falls back to discovering a valid checkpoint from the user registry.
        """
        key = adapter_name or "base"

        if key not in self._sampling_clients:
            if adapter_name:
                # Try to get checkpoint path from stored paths or discover from registry
                checkpoint_path = self.adapter_paths.get(adapter_name)
                if not checkpoint_path:
                    tprint(f"  No stored adapter path for '{adapter_name}', searching registry...")
                    checkpoint_path = discover_checkpoint_from_registry(
                        self.client.service_client, adapter_name, self.model_name, "sampler"
                    )
                    if checkpoint_path:
                        self.adapter_paths[adapter_name] = checkpoint_path

                if checkpoint_path:
                    tprint(f"  Loading sampling client for {key} from checkpoint: {checkpoint_path}")
                    try:
                        self._sampling_clients[key] = self.client.service_client.create_sampling_client(
                            model_path=checkpoint_path,
                            retry_config=build_tinker_sampling_retry_config(),
                        )
                    except Exception as e:
                        if not _should_fallback_to_checkpoint_registry(e):
                            _raise_nonrecoverable_checkpoint_error(
                                "create sampling client",
                                adapter_name,
                                checkpoint_path,
                                e,
                            )
                        tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry...")
                        discovered_path = discover_checkpoint_from_registry(
                            self.client.service_client, adapter_name, self.model_name, "sampler"
                        )
                        if discovered_path and discovered_path != checkpoint_path:
                            tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                            try:
                                self._sampling_clients[key] = self.client.service_client.create_sampling_client(
                                    model_path=discovered_path,
                                    retry_config=build_tinker_sampling_retry_config(),
                                )
                                self.adapter_paths[adapter_name] = discovered_path
                                tprint(f"  [fallback] Successfully loaded from discovered path")
                            except Exception as e2:
                                raise RuntimeError(
                                    f"Failed to create sampling client for '{adapter_name}' from both stored path "
                                    f"({checkpoint_path}) and discovered path ({discovered_path})"
                                ) from e2
                        else:
                            raise RuntimeError(
                                f"Failed to create sampling client for '{adapter_name}' from {checkpoint_path} "
                                f"and no fallback checkpoint found in registry"
                            ) from e
                else:
                    # No checkpoint found - fall through to training client approach
                    tprint(f"  No checkpoint found for {key}, creating via training client...")
                    training_client = self._get_training_client(adapter_name)
                    self._sampling_clients[key] = training_client.save_weights_and_get_sampling_client(
                        name=f"nudge_test_{key}",
                        retry_config=build_tinker_sampling_retry_config(),
                    )
            else:
                # Base model: create fresh training client and save weights
                tprint(f"  Creating sampling client for {key}...")
                training_client = self._get_training_client(adapter_name)
                self._sampling_clients[key] = training_client.save_weights_and_get_sampling_client(
                    name=f"nudge_test_{key}",
                    retry_config=build_tinker_sampling_retry_config(),
                )

        return self._sampling_clients[key]

    def _get_training_client(self, adapter_name: Optional[str] = None):
        """Get training client for perplexity computation with optional LoRA.

        Creates a fresh training client and loads LoRA weights via load_state()
        if an adapter checkpoint path is available. This enables adapter-aware
        perplexity probes to measure how LoRA changes heuristic preferences.

        If the stored checkpoint reference looks stale or missing, automatically
        falls back to discovering a valid checkpoint from the user registry.
        """
        key = adapter_name or "base"

        if key not in self._training_clients:
            # Create fresh training client
            training_client = self.client.service_client.create_lora_training_client(
                base_model=self.model_name,
                rank=32
            )

            # Load LoRA weights if adapter specified
            if adapter_name:
                checkpoint_path = self.state_paths.get(adapter_name)
                if not checkpoint_path:
                    # Try to discover from registry
                    tprint(f"  No stored state path for '{adapter_name}', searching registry...")
                    checkpoint_path = discover_checkpoint_from_registry(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if checkpoint_path:
                        self.state_paths[adapter_name] = checkpoint_path
                    else:
                        raise KeyError(f"Adapter state path not found for '{adapter_name}' and no fallback available")

                tprint(f"  Loading LoRA weights for perplexity: {adapter_name} from {checkpoint_path}")
                try:
                    training_client.load_state(checkpoint_path).result()
                except Exception as e:
                    if not _should_fallback_to_checkpoint_registry(e):
                        _raise_nonrecoverable_checkpoint_error(
                            "load LoRA state",
                            adapter_name,
                            checkpoint_path,
                            e,
                        )
                    tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry...")
                    discovered_path = discover_checkpoint_from_registry(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if discovered_path and discovered_path != checkpoint_path:
                        tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                        try:
                            # Need a fresh training client since the previous one may be in a bad state
                            training_client = self.client.service_client.create_lora_training_client(
                                base_model=self.model_name,
                                rank=32
                            )
                            training_client.load_state(discovered_path).result()
                            # Success! Update stored path for future use
                            self.state_paths[adapter_name] = discovered_path
                            tprint(f"  [fallback] Successfully loaded from discovered path")
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to load LoRA state for '{adapter_name}' from both stored path "
                                f"({checkpoint_path}) and discovered path ({discovered_path})"
                            ) from e2
                    else:
                        raise RuntimeError(
                            f"Failed to load LoRA state for '{adapter_name}' from {checkpoint_path} "
                            f"and no fallback checkpoint found in registry"
                        ) from e

            self._training_clients[key] = training_client

        return self._training_clients[key]

    async def _get_training_client_async(self, adapter_name: Optional[str] = None):
        """Async-safe training client getter using async ServiceClient APIs."""
        key = adapter_name or "base"
        lock = self._training_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._training_clients:
                return self._training_clients[key]

            training_client = await self.client.service_client.create_lora_training_client_async(
                base_model=self.model_name,
                rank=32
            )

            if adapter_name:
                checkpoint_path = self.state_paths.get(adapter_name)
                if not checkpoint_path:
                    tprint(f"  No stored state path for '{adapter_name}', searching registry (async)...")
                    checkpoint_path = await discover_checkpoint_from_registry_async(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if checkpoint_path:
                        self.state_paths[adapter_name] = checkpoint_path
                    else:
                        raise KeyError(f"Adapter state path not found for '{adapter_name}' and no fallback available")

                tprint(f"  Loading LoRA weights for perplexity: {adapter_name} from {checkpoint_path} (async)")
                try:
                    await training_client.load_state_async(checkpoint_path)
                except Exception as e:
                    if not _should_fallback_to_checkpoint_registry(e):
                        _raise_nonrecoverable_checkpoint_error(
                            "load LoRA state",
                            adapter_name,
                            checkpoint_path,
                            e,
                        )
                    tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry (async)...")
                    discovered_path = await discover_checkpoint_from_registry_async(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if discovered_path and discovered_path != checkpoint_path:
                        tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                        try:
                            training_client = await self.client.service_client.create_lora_training_client_async(
                                base_model=self.model_name,
                                rank=32
                            )
                            await training_client.load_state_async(discovered_path)
                            self.state_paths[adapter_name] = discovered_path
                            tprint(f"  [fallback] Successfully loaded from discovered path")
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to load LoRA state for '{adapter_name}' from both stored path "
                                f"({checkpoint_path}) and discovered path ({discovered_path})"
                            ) from e2
                    else:
                        raise RuntimeError(
                            f"Failed to load LoRA state for '{adapter_name}' from {checkpoint_path} "
                            f"and no fallback checkpoint found in registry"
                        ) from e

            self._training_clients[key] = training_client
            return training_client

    async def _get_sampling_client_async(self, adapter_name: Optional[str] = None):
        """Async-safe sampling client getter that avoids sync calls inside event loops."""
        key = adapter_name or "base"
        lock = self._sampling_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._sampling_clients:
                return self._sampling_clients[key]

            if adapter_name:
                checkpoint_path = self.adapter_paths.get(adapter_name)
                if not checkpoint_path:
                    tprint(f"  No stored adapter path for '{adapter_name}', searching registry (async)...")
                    checkpoint_path = await discover_checkpoint_from_registry_async(
                        self.client.service_client, adapter_name, self.model_name, "sampler"
                    )
                    if checkpoint_path:
                        self.adapter_paths[adapter_name] = checkpoint_path

                if checkpoint_path:
                    tprint(f"  Loading sampling client for {key} from checkpoint: {checkpoint_path} (async)")
                    try:
                        self._sampling_clients[key] = await self.client.service_client.create_sampling_client_async(
                            model_path=checkpoint_path,
                            retry_config=build_tinker_sampling_retry_config(),
                        )
                    except Exception as e:
                        if not _should_fallback_to_checkpoint_registry(e):
                            _raise_nonrecoverable_checkpoint_error(
                                "create sampling client",
                                adapter_name,
                                checkpoint_path,
                                e,
                            )
                        tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry (async)...")
                        discovered_path = await discover_checkpoint_from_registry_async(
                            self.client.service_client, adapter_name, self.model_name, "sampler"
                        )
                        if discovered_path and discovered_path != checkpoint_path:
                            tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                            try:
                                self._sampling_clients[key] = await self.client.service_client.create_sampling_client_async(
                                    model_path=discovered_path,
                                    retry_config=build_tinker_sampling_retry_config(),
                                )
                                self.adapter_paths[adapter_name] = discovered_path
                                tprint(f"  [fallback] Successfully loaded from discovered path")
                            except Exception as e2:
                                raise RuntimeError(
                                    f"Failed to create sampling client for '{adapter_name}' from both stored path "
                                    f"({checkpoint_path}) and discovered path ({discovered_path})"
                                ) from e2
                        else:
                            raise RuntimeError(
                                f"Failed to create sampling client for '{adapter_name}' from {checkpoint_path} "
                                f"and no fallback checkpoint found in registry"
                            ) from e
                else:
                    tprint(f"  No checkpoint found for {key}, creating via training client (async)...")
                    training_client = await self._get_training_client_async(adapter_name)
                    self._sampling_clients[key] = await training_client.save_weights_and_get_sampling_client_async(
                        name=f"nudge_test_{key}",
                        retry_config=build_tinker_sampling_retry_config(),
                    )
            else:
                tprint(f"  Creating sampling client for {key} (async)...")
                training_client = await self._get_training_client_async(adapter_name)
                self._sampling_clients[key] = await training_client.save_weights_and_get_sampling_client_async(
                    name=f"nudge_test_{key}",
                    retry_config=build_tinker_sampling_retry_config(),
                )

            return self._sampling_clients[key]

    def compute_perplexity_losses(self, a: int, b: int, adapter_name: Optional[str] = None) -> Dict:
        """
        Compute perplexity losses using template averaging and neutral baseline.

        Returns:
            Dict with 'losses', 'neutral_loss', and 'delta_losses'
        """
        client = self._get_training_client(adapter_name)
        template_count = sum(len(t) for t in get_multi_heuristic_templates(0, 0).values())
        batch_size = template_count + 1

        results = self.client.compute_heuristic_losses_multi(
            problems=[(a, b)],
            batch_size=batch_size,
            include_neutral=True,
            training_client=client
        )
        if results:
            return results[0]

        return {
            'losses': {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
            'neutral_loss': float('inf'),
            'delta_losses': None
        }

    def generate_answer(
        self, a: int, b: int, adapter_name: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Generate model's answer for a multiplication problem.

        Returns:
            Tuple of (answer, raw_text) where raw_text is the model output
        """
        try:
            sampler = self._get_sampling_client(adapter_name)
            prompt_text = self.prompt_template.format(a=a, b=b)
            model_input = None
            if hasattr(self.client, "build_text_generation_input"):
                model_input = self.client.build_text_generation_input(prompt_text)
            if model_input is None:
                tokens = self.client.tokenizer.encode(prompt_text)
                model_input = self.client.tinker.types.ModelInput.from_ints(tokens=tokens)

            sampling_params = self.client.tinker.types.SamplingParams(
                max_tokens=self.generation_max_tokens,
                temperature=0.0
            )

            future = sampler.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            )
            result = _call_with_timeout(future, operation="sampling generation")

            # Extract text
            if hasattr(result, 'samples') and len(result.samples) > 0:
                sample = result.samples[0]
                if hasattr(sample, 'tokens'):
                    output_tokens = sample.tokens
                    text = self.client.tokenizer.decode(output_tokens)
                elif hasattr(sample, 'token_ids'):
                    output_tokens = sample.token_ids
                    text = self.client.tokenizer.decode(output_tokens)
                elif hasattr(sample, 'text'):
                    text = sample.text
                else:
                    text = str(sample)
            elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                output_tokens = result.sequences[0].tokens
                text = self.client.tokenizer.decode(output_tokens)
            elif hasattr(result, 'completions'):
                text = result.completions[0]
            elif isinstance(result, list):
                text = str(result[0])
            else:
                text = str(result)

            return extract_answer(text), text

        except Exception as e:
            tprint(f"    Warning: Generation failed: {e}")
            return None, None

    def _evaluate_single(self, row: HDSRow, adapter_name: Optional[str] = None) -> EvaluationResult:
        """Evaluate a single HDS problem."""
        # Get perplexity losses (now includes neutral baseline)
        probe_result = self.compute_perplexity_losses(row.a, row.b, adapter_name)

        # Extract from new return format
        losses = probe_result.get('losses', {})
        neutral_loss = probe_result.get('neutral_loss')
        delta_losses = probe_result.get('delta_losses')

        # Determine detected heuristic
        if losses and all(v < float('inf') for v in losses.values()):
            detected = min(losses, key=losses.get)
        else:
            detected = "UNKNOWN"

        # Generate answer and use enhanced extraction with validation
        prompt_text = self.prompt_template.format(a=row.a, b=row.b)
        _, text = self.generate_answer(row.a, row.b, adapter_name)

        extraction_result = None
        answer = None
        if text:
            extraction_result = extract_answer_enhanced(text, a=row.a, b=row.b)
            answer = extraction_result.answer

        is_correct = answer == row.product if answer is not None else False

        return EvaluationResult(
            hds_id=row.id,
            a=row.a,
            b=row.b,
            product=row.product,
            target_heuristic=row.target_heuristic,
            model_answer=answer,
            is_correct=is_correct,
            perplexity_losses=losses,
            detected_heuristic=detected,
            neutral_loss=neutral_loss,
            delta_losses=delta_losses,
            extraction_confidence=extraction_result.confidence if extraction_result else None,
            extraction_strategy=extraction_result.strategy if extraction_result else None,
            is_truncated=extraction_result.is_truncated if extraction_result else None,
            prompt_text=prompt_text,
            generation_text=text,
            input_modality="text"
        )

    def evaluate_hds(
        self,
        hds: List[HDSRow],
        adapter_name: Optional[str] = None,
        verbose: bool = True,
        max_workers: int = 1,
        batch_size: int = 1,
        use_async: bool = True,
        concurrency: Optional[int] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate all HDS problems with given model/adapter.

        Args:
            hds: List of HDS problems
            adapter_name: Optional LoRA adapter name
            verbose: Print progress
            max_workers: Number of parallel workers (1 = sequential)
            batch_size: Batch size for perplexity probes (1 = no batching, >1 = multi-problem batching)

        Returns:
            List of EvaluationResult objects
        """
        if use_async:
            async_concurrency = concurrency or max(1, max_workers)
            return asyncio.run(self._evaluate_hds_async(
                hds,
                adapter_name,
                verbose,
                batch_size,
                async_concurrency
            ))

        label = adapter_name or "base"
        if verbose:
            tprint(f"\nEvaluating {label} model on {len(hds)} problems...")
            if max_workers > 1:
                tprint(f"  Using {max_workers} parallel workers")
            if batch_size > 1:
                tprint(f"  Using batch size {batch_size} for perplexity probes")

        # Use batched evaluation if batch_size > 1
        if batch_size > 1:
            return self._evaluate_hds_batched(hds, adapter_name, verbose, batch_size)

        results: List[EvaluationResult]
        if max_workers <= 1:
            # Sequential processing
            results = []
            for i, row in enumerate(hds):
                if verbose:
                    tprint(f"  [{i+1}/{len(hds)}] {row.id}: {row.a} × {row.b} = {row.product} (target: {row.target_heuristic})")

                result = self._evaluate_single(row, adapter_name)
                results.append(result)

                if verbose:
                    status = "✓" if result.is_correct else "✗"
                    match = "=" if result.detected_heuristic.upper() == row.target_heuristic else "≠"
                    tprint(f"         {status} answer={result.model_answer}, detected={result.detected_heuristic} {match} target")
        else:
            # Parallel processing
            results_opt: List[Optional[EvaluationResult]] = [None] * len(hds)
            completed = 0

            def process_one(idx_row):
                idx, row = idx_row
                return idx, self._evaluate_single(row, adapter_name)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_one, (i, row)): i for i, row in enumerate(hds)}

                for future in as_completed(futures):
                    idx, result = future.result()
                    results_opt[idx] = result
                    completed += 1

                    if verbose and completed % max(1, len(hds) // 10) == 0:
                        tprint(f"  [{completed}/{len(hds)}] Processing...")

            results = cast(List[EvaluationResult], results_opt)

        correct = sum(1 for r in results if r.is_correct)
        if verbose:
            tprint(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        return results

    async def _evaluate_hds_async(
        self,
        hds: List[HDSRow],
        adapter_name: Optional[str],
        verbose: bool,
        batch_size: int,
        concurrency: int
    ) -> List[EvaluationResult]:
        """Async text evaluation using sample_async + forward_async pipelining."""
        if not hds:
            return []

        client = self.client
        sampling_client = await self._get_sampling_client_async(adapter_name)
        tokenizer = client.tokenizer
        tinker = client.tinker
        TIMEOUT_SECONDS = 120

        semaphore = asyncio.Semaphore(max(1, concurrency))
        completed = [0]
        total = len(hds)
        generation_heartbeat: Optional[_ProgressHeartbeat] = None
        if verbose:
            generation_heartbeat = _ProgressHeartbeat("Generation progress", total)
            generation_heartbeat.log_start("Starting async generation")

        async def _generate_one(idx: int, row: HDSRow):
            def _mark_generation_progress() -> None:
                completed[0] += 1
                if generation_heartbeat is not None:
                    generation_heartbeat.maybe_log(completed[0])

            async with semaphore:
                prompt_text = self.prompt_template.format(a=row.a, b=row.b)
                model_input = client.build_text_generation_input(prompt_text)
                if model_input is None:
                    tokens = tokenizer.encode(prompt_text)
                    model_input = tinker.types.ModelInput.from_ints(tokens=tokens)

                sampling_params = tinker.types.SamplingParams(
                    max_tokens=self.generation_max_tokens,
                    temperature=0.0
                )

                try:
                    result = await asyncio.wait_for(
                        sampling_client.sample_async(
                            prompt=model_input,
                            sampling_params=sampling_params,
                            num_samples=1
                        ),
                        timeout=TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    if verbose:
                        tprint(f"    Timeout for {row.id} after {TIMEOUT_SECONDS}s")
                    _mark_generation_progress()
                    return idx, None, None, None
                except Exception as e:
                    if verbose:
                        tprint(f"    Warning: Async generation failed for {row.id}: {type(e).__name__}: {e}")
                    _mark_generation_progress()
                    return idx, None, None, None

                text = None
                if hasattr(result, 'samples') and len(result.samples) > 0:
                    sample = result.samples[0]
                    if hasattr(sample, 'tokens'):
                        text = tokenizer.decode(sample.tokens)
                    elif hasattr(sample, 'token_ids'):
                        text = tokenizer.decode(sample.token_ids)
                    elif hasattr(sample, 'text'):
                        text = sample.text
                    else:
                        text = str(sample)
                elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                    text = tokenizer.decode(result.sequences[0].tokens)
                elif hasattr(result, 'completions'):
                    text = result.completions[0]
                else:
                    text = str(result) if result else None

                extraction = extract_answer_enhanced(text, a=row.a, b=row.b) if text else None
                answer = extraction.answer if extraction else None

                _mark_generation_progress()

                return idx, answer, text, extraction

        gen_tasks = [_generate_one(i, row) for i, row in enumerate(hds)]
        gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

        gen_map: Dict[int, Tuple[Optional[int], Optional[str], Optional[Any]]] = {}
        for res in gen_results:
            if isinstance(res, Exception):
                continue
            idx, answer, text, extraction = res
            gen_map[idx] = (answer, text, extraction)

        score_heartbeat: Optional[_ProgressHeartbeat] = None
        if verbose:
            tprint(f"  Computing perplexity probes asynchronously (batch_size={batch_size}, max_in_flight={self.score_max_in_flight})...")
            score_heartbeat = _ProgressHeartbeat("Scoring progress", total)
            score_heartbeat.log_start("Starting async scoring")

        training_client = await self._get_training_client_async(adapter_name)
        perplexity_map: Dict[int, Dict[str, Any]] = {}
        for batch_start in range(0, total, batch_size):
            batch_rows = hds[batch_start:batch_start + batch_size]
            batch_problems = [(r.a, r.b) for r in batch_rows]
            batch_results = await client.compute_heuristic_losses_multi_async(
                batch_problems,
                batch_size=batch_size,
                include_neutral=True,
                training_client=training_client,
                max_in_flight=self.score_max_in_flight
            )
            for local_idx, res in enumerate(batch_results):
                perplexity_map[batch_start + local_idx] = res
            if score_heartbeat is not None:
                score_heartbeat.maybe_log(min(total, batch_start + len(batch_rows)))

        results: List[EvaluationResult] = []
        for i, row in enumerate(hds):
            model_answer, text, extraction_result = gen_map.get(i, (None, None, None))
            perplexity_losses = {}
            neutral_loss = None
            delta_losses = None
            detected = "UNKNOWN"
            detection_confidence = 0.0

            perp_data = perplexity_map.get(i, {})
            perplexity_losses = perp_data.get("losses", {})
            neutral_loss = perp_data.get("neutral_loss")
            delta_losses = perp_data.get("delta_losses")

            if perplexity_losses and all(v < float('inf') for v in perplexity_losses.values()):
                detected = min(perplexity_losses, key=perplexity_losses.get)
                sorted_losses = sorted(perplexity_losses.values())
                if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
                    gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
                    detection_confidence = min(1.0, gap * 2)

            is_correct = model_answer == row.product if model_answer is not None else False

            results.append(EvaluationResult(
                hds_id=row.id,
                a=row.a,
                b=row.b,
                product=row.product,
                target_heuristic=row.target_heuristic,
                model_answer=model_answer,
                is_correct=is_correct,
                perplexity_losses=perplexity_losses,
                detected_heuristic=detected,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                extraction_confidence=extraction_result.confidence if extraction_result else None,
                extraction_strategy=extraction_result.strategy if extraction_result else None,
                is_truncated=extraction_result.is_truncated if extraction_result else None,
                prompt_text=self.prompt_template.format(a=row.a, b=row.b),
                generation_text=text,
                input_modality="text"
            ))

        if verbose:
            correct = sum(1 for r in results if r.is_correct)
            tprint(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        return results

    def _evaluate_hds_batched(
        self,
        hds: List[HDSRow],
        adapter_name: Optional[str] = None,
        verbose: bool = True,
        batch_size: int = 30
    ) -> List[EvaluationResult]:
        """
        Evaluate HDS problems with multi-problem batching for perplexity probes.

        Batches perplexity probes across multiple problems, then runs generation per-problem.
        """
        client = self._get_training_client(adapter_name)

        # Step 1: Batch all perplexity probes (now includes neutral baseline)
        problems = [(row.a, row.b) for row in hds]
        if verbose:
            tprint(f"  Computing perplexity probes for {len(problems)} problems (with neutral baseline)...")

        # Compute batched perplexity losses
        all_probe_results = self._compute_perplexity_losses_multi(
            problems, adapter_name, batch_size, include_neutral=True
        )

        if verbose:
            tprint(f"  Perplexity probes complete. Running generation for each problem...")

        # Step 2: Run generation and build results
        results = []
        for i, (row, probe_result) in enumerate(zip(hds, all_probe_results)):
            if verbose and (i + 1) % max(1, len(hds) // 10) == 0:
                tprint(f"  [{i+1}/{len(hds)}] Generating answers...")

            # Extract losses from new return format
            losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')

            # Determine detected heuristic
            if losses and all(v < float('inf') for v in losses.values()):
                detected = min(losses, key=losses.get)
            else:
                detected = "UNKNOWN"

            # Generate answer and use enhanced extraction with validation
            prompt_text = self.prompt_template.format(a=row.a, b=row.b)
            _, text = self.generate_answer(row.a, row.b, adapter_name)

            extraction_result = None
            answer = None
            if text:
                extraction_result = extract_answer_enhanced(text, a=row.a, b=row.b)
                answer = extraction_result.answer

            is_correct = answer == row.product if answer is not None else False

            results.append(EvaluationResult(
                hds_id=row.id,
                a=row.a,
                b=row.b,
                product=row.product,
                target_heuristic=row.target_heuristic,
                model_answer=answer,
                is_correct=is_correct,
                perplexity_losses=losses,
                detected_heuristic=detected,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                extraction_confidence=extraction_result.confidence if extraction_result else None,
                extraction_strategy=extraction_result.strategy if extraction_result else None,
                is_truncated=extraction_result.is_truncated if extraction_result else None,
                prompt_text=prompt_text,
                generation_text=text,
                input_modality="text"
            ))

        correct = sum(1 for r in results if r.is_correct)
        if verbose:
            tprint(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        return results

    def _compute_perplexity_losses_multi(
        self,
        problems: List[Tuple[int, int]],
        adapter_name: Optional[str] = None,
        batch_size: int = 30,
        include_neutral: bool = True
    ) -> List[Dict[str, Any]]:
        """Compute perplexity losses for multiple problems using the shared client path."""
        if not problems:
            return []

        import asyncio
        training_client = self._get_training_client(adapter_name)
        return asyncio.run(self.client.compute_heuristic_losses_multi_async(
            problems,
            batch_size=batch_size,
            include_neutral=include_neutral,
            training_client=training_client,
            max_in_flight=self.score_max_in_flight
        ))

    def compare_base_vs_lora(
        self,
        hds: List[HDSRow],
        lora_name: str,
        base_results: List[EvaluationResult],
        verbose: bool = True,
        max_workers: int = 1,
        batch_size: int = 1
    ) -> Tuple[List[EvaluationResult], List[NudgeResult]]:
        """Compare base model to a specific LoRA adapter."""
        # Evaluate with LoRA
        lora_results = self.evaluate_hds(hds, adapter_name=lora_name, verbose=verbose, max_workers=max_workers, batch_size=batch_size)

        # Compare results
        nudge_results = []
        for base_r, lora_r, row in zip(base_results, lora_results, hds):
            base_label = _correctness_label(base_r)
            lora_label = _correctness_label(lora_r)
            correctness_flip, correctness_flip_type = _compute_correctness_flip(base_label, lora_label)

            detection_flip = base_r.detected_heuristic != lora_r.detected_heuristic
            detection_flip_type = "detection_change" if detection_flip else "unchanged"

            nudge_results.append(NudgeResult(
                hds_id=row.id,
                a=row.a,
                b=row.b,
                target_heuristic=row.target_heuristic,
                base_correct=_label_to_bool(base_label),
                base_correctness=base_label,
                base_detected=base_r.detected_heuristic,
                lora_correct=_label_to_bool(lora_label),
                lora_correctness=lora_label,
                lora_detected=lora_r.detected_heuristic,
                is_flip=correctness_flip,
                flip_type=correctness_flip_type,
                correctness_flip=correctness_flip,
                correctness_flip_type=correctness_flip_type,
                detection_flip=detection_flip,
                detection_flip_type=detection_flip_type
            ))

        return lora_results, nudge_results


class ImageNudgeTester:
    """
    Test LoRA nudge effects on HDS problems using image modality.

    Uses VisionTinkerClient to evaluate model behavior on rendered problem images.
    With unified VLM backbone, LoRA adapters trained on text can now be applied
    to image inputs for cross-modal transfer testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        images_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        adapter_paths: Optional[Dict[str, str]] = None,
        state_paths: Optional[Dict[str, str]] = None,
        generation_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
        prompt_text: str = IMAGE_PROMPT_TEMPLATE,
        score_max_in_flight: int = 4
    ):
        """Initialize with Vision Tinker API.

        Args:
            api_key: Tinker API key
            images_dir: Directory containing HDS problem images
            model_name: Model to use (default: VLM)
            adapter_paths: Dict mapping adapter names (e.g., "rc_lora") to their Tinker checkpoint paths
            state_paths: Dict mapping adapter names to training state paths for load_state()
            generation_max_tokens: Max tokens for generation
            prompt_text: Prompt to use for image generation
            score_max_in_flight: Max forward() batches to keep in flight (pipelined)
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.client = VisionTinkerClient(
            model_name=self.model_name,
            api_key=api_key,
            verbose=True
        )
        self.images_dir = images_dir or HDS_IMAGES_DIR

        # Store adapter checkpoint paths for sampling and state loading
        self.adapter_paths = adapter_paths or {}
        self.state_paths = state_paths or {}

        self.generation_max_tokens = generation_max_tokens
        self.prompt_text = prompt_text
        self.score_max_in_flight = score_max_in_flight

        # Cache for training clients (for LoRA testing)
        self._training_clients: Dict[str, Any] = {}
        self._sampling_clients: Dict[str, Any] = {}
        self._training_locks: Dict[str, asyncio.Lock] = {}
        self._sampling_locks: Dict[str, asyncio.Lock] = {}

    def _get_image_path(self, hds_id: str) -> Path:
        """Get image path for a problem ID."""
        return self.images_dir / f"{hds_id}.png"

    def _get_training_client(self, adapter_name: Optional[str] = None):
        """Get training client for perplexity computation with optional LoRA.

        Creates a fresh training client and loads LoRA weights via load_state()
        if an adapter checkpoint path is available. This enables adapter-aware
        perplexity probes to measure how LoRA changes heuristic preferences.

        If the stored checkpoint reference looks stale or missing, automatically
        falls back to discovering a valid checkpoint from the user registry.
        """
        key = adapter_name or "base"

        if key not in self._training_clients:
            # Create fresh training client
            training_client = self.client.service_client.create_lora_training_client(
                base_model=self.model_name,
                rank=32
            )

            # Load LoRA weights if adapter specified
            if adapter_name:
                checkpoint_path = self.state_paths.get(adapter_name)
                if not checkpoint_path:
                    # Try to discover from registry
                    tprint(f"  No stored state path for '{adapter_name}', searching registry...")
                    checkpoint_path = discover_checkpoint_from_registry(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if checkpoint_path:
                        self.state_paths[adapter_name] = checkpoint_path
                    else:
                        raise KeyError(f"Adapter state path not found for '{adapter_name}' and no fallback available")

                tprint(f"  Loading LoRA weights for perplexity: {adapter_name} from {checkpoint_path}")
                try:
                    training_client.load_state(checkpoint_path).result()
                except Exception as e:
                    if not _should_fallback_to_checkpoint_registry(e):
                        _raise_nonrecoverable_checkpoint_error(
                            "load LoRA state",
                            adapter_name,
                            checkpoint_path,
                            e,
                        )
                    tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry...")
                    discovered_path = discover_checkpoint_from_registry(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if discovered_path and discovered_path != checkpoint_path:
                        tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                        try:
                            # Need a fresh training client since the previous one may be in a bad state
                            training_client = self.client.service_client.create_lora_training_client(
                                base_model=self.model_name,
                                rank=32
                            )
                            training_client.load_state(discovered_path).result()
                            # Success! Update stored path for future use
                            self.state_paths[adapter_name] = discovered_path
                            tprint(f"  [fallback] Successfully loaded from discovered path")
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to load LoRA state for '{adapter_name}' from both stored path "
                                f"({checkpoint_path}) and discovered path ({discovered_path})"
                            ) from e2
                    else:
                        raise RuntimeError(
                            f"Failed to load LoRA state for '{adapter_name}' from {checkpoint_path} "
                            f"and no fallback checkpoint found in registry"
                        ) from e

            self._training_clients[key] = training_client

        return self._training_clients[key]

    def _get_sampling_client(self, adapter_name: Optional[str] = None):
        """Get or create a sampling client for image modality (sync version)."""
        key = adapter_name or "base"

        if key not in self._sampling_clients:
            if adapter_name:
                checkpoint_path = self.adapter_paths.get(adapter_name)
                if not checkpoint_path:
                    tprint(f"  No stored adapter path for '{adapter_name}', searching registry...")
                    checkpoint_path = discover_checkpoint_from_registry(
                        self.client.service_client, adapter_name, self.model_name, "sampler"
                    )
                    if checkpoint_path:
                        self.adapter_paths[adapter_name] = checkpoint_path

                if checkpoint_path:
                    tprint(f"  Loading sampling client for {key} from checkpoint: {checkpoint_path}")
                    try:
                        self._sampling_clients[key] = self.client.service_client.create_sampling_client(
                            model_path=checkpoint_path,
                            retry_config=build_tinker_sampling_retry_config(),
                        )
                    except Exception as e:
                        if not _should_fallback_to_checkpoint_registry(e):
                            _raise_nonrecoverable_checkpoint_error(
                                "create sampling client",
                                adapter_name,
                                checkpoint_path,
                                e,
                            )
                        tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry...")
                        discovered_path = discover_checkpoint_from_registry(
                            self.client.service_client, adapter_name, self.model_name, "sampler"
                        )
                        if discovered_path and discovered_path != checkpoint_path:
                            tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                            try:
                                self._sampling_clients[key] = self.client.service_client.create_sampling_client(
                                    model_path=discovered_path,
                                    retry_config=build_tinker_sampling_retry_config(),
                                )
                                self.adapter_paths[adapter_name] = discovered_path
                                tprint(f"  [fallback] Successfully loaded from discovered path")
                            except Exception as e2:
                                raise RuntimeError(
                                    f"Failed to create sampling client for '{adapter_name}' from both stored path "
                                    f"({checkpoint_path}) and discovered path ({discovered_path})"
                                ) from e2
                        else:
                            raise RuntimeError(
                                f"Failed to create sampling client for '{adapter_name}' from {checkpoint_path} "
                                f"and no fallback checkpoint found in registry"
                            ) from e
                else:
                    tprint(f"  No checkpoint found for {key}, creating via training client...")
                    training_client = self._get_training_client(adapter_name)
                    self._sampling_clients[key] = training_client.save_weights_and_get_sampling_client(
                        name=f"nudge_test_{key}",
                        retry_config=build_tinker_sampling_retry_config(),
                    )
            else:
                tprint(f"  Creating sampling client for {key}...")
                training_client = self._get_training_client(adapter_name)
                self._sampling_clients[key] = training_client.save_weights_and_get_sampling_client(
                    name=f"nudge_test_{key}",
                    retry_config=build_tinker_sampling_retry_config(),
                )

        return self._sampling_clients[key]

    async def _get_training_client_async(self, adapter_name: Optional[str] = None):
        """Async-safe training client getter for image modality."""
        key = adapter_name or "base"
        lock = self._training_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._training_clients:
                return self._training_clients[key]

            training_client = await self.client.service_client.create_lora_training_client_async(
                base_model=self.model_name,
                rank=32
            )

            if adapter_name:
                checkpoint_path = self.state_paths.get(adapter_name)
                if not checkpoint_path:
                    tprint(f"  No stored state path for '{adapter_name}', searching registry (async)...")
                    checkpoint_path = await discover_checkpoint_from_registry_async(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if checkpoint_path:
                        self.state_paths[adapter_name] = checkpoint_path
                    else:
                        raise KeyError(f"Adapter state path not found for '{adapter_name}' and no fallback available")

                tprint(f"  Loading LoRA weights for perplexity: {adapter_name} from {checkpoint_path} (async)")
                try:
                    await training_client.load_state_async(checkpoint_path)
                except Exception as e:
                    if not _should_fallback_to_checkpoint_registry(e):
                        _raise_nonrecoverable_checkpoint_error(
                            "load LoRA state",
                            adapter_name,
                            checkpoint_path,
                            e,
                        )
                    tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry (async)...")
                    discovered_path = await discover_checkpoint_from_registry_async(
                        self.client.service_client, adapter_name, self.model_name, "training"
                    )
                    if discovered_path and discovered_path != checkpoint_path:
                        tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                        try:
                            training_client = await self.client.service_client.create_lora_training_client_async(
                                base_model=self.model_name,
                                rank=32
                            )
                            await training_client.load_state_async(discovered_path)
                            self.state_paths[adapter_name] = discovered_path
                            tprint(f"  [fallback] Successfully loaded from discovered path")
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to load LoRA state for '{adapter_name}' from both stored path "
                                f"({checkpoint_path}) and discovered path ({discovered_path})"
                            ) from e2
                    else:
                        raise RuntimeError(
                            f"Failed to load LoRA state for '{adapter_name}' from {checkpoint_path} "
                            f"and no fallback checkpoint found in registry"
                        ) from e

            self._training_clients[key] = training_client
            return training_client

    async def _get_sampling_client_async(self, adapter_name: Optional[str] = None):
        """Async-safe sampling client getter for image modality."""
        key = adapter_name or "base"
        lock = self._sampling_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._sampling_clients:
                return self._sampling_clients[key]

            if adapter_name:
                checkpoint_path = self.adapter_paths.get(adapter_name)
                if not checkpoint_path:
                    tprint(f"  No stored adapter path for '{adapter_name}', searching registry (async)...")
                    checkpoint_path = await discover_checkpoint_from_registry_async(
                        self.client.service_client, adapter_name, self.model_name, "sampler"
                    )
                    if checkpoint_path:
                        self.adapter_paths[adapter_name] = checkpoint_path

                if checkpoint_path:
                    tprint(f"  Loading sampling client for {key} from checkpoint: {checkpoint_path} (async)")
                    try:
                        self._sampling_clients[key] = await self.client.service_client.create_sampling_client_async(
                            model_path=checkpoint_path,
                            retry_config=build_tinker_sampling_retry_config(),
                        )
                    except Exception as e:
                        if not _should_fallback_to_checkpoint_registry(e):
                            _raise_nonrecoverable_checkpoint_error(
                                "create sampling client",
                                adapter_name,
                                checkpoint_path,
                                e,
                            )
                        tprint(f"  [fallback] Stored path failed ({type(e).__name__}), searching registry (async)...")
                        discovered_path = await discover_checkpoint_from_registry_async(
                            self.client.service_client, adapter_name, self.model_name, "sampler"
                        )
                        if discovered_path and discovered_path != checkpoint_path:
                            tprint(f"  [fallback] Trying discovered path: {discovered_path}")
                            try:
                                self._sampling_clients[key] = await self.client.service_client.create_sampling_client_async(
                                    model_path=discovered_path,
                                    retry_config=build_tinker_sampling_retry_config(),
                                )
                                self.adapter_paths[adapter_name] = discovered_path
                                tprint(f"  [fallback] Successfully loaded from discovered path")
                            except Exception as e2:
                                raise RuntimeError(
                                    f"Failed to create sampling client for '{adapter_name}' from both stored path "
                                    f"({checkpoint_path}) and discovered path ({discovered_path})"
                                ) from e2
                        else:
                            raise RuntimeError(
                                f"Failed to create sampling client for '{adapter_name}' from {checkpoint_path} "
                                f"and no fallback checkpoint found in registry"
                            ) from e
                else:
                    tprint(f"  No checkpoint found for {key}, creating via training client (async)...")
                    training_client = await self._get_training_client_async(adapter_name)
                    self._sampling_clients[key] = await training_client.save_weights_and_get_sampling_client_async(
                        name=f"nudge_test_{key}",
                        retry_config=build_tinker_sampling_retry_config(),
                    )
            else:
                tprint(f"  Creating sampling client for {key} (async)...")
                training_client = await self._get_training_client_async(adapter_name)
                self._sampling_clients[key] = await training_client.save_weights_and_get_sampling_client_async(
                    name=f"nudge_test_{key}",
                    retry_config=build_tinker_sampling_retry_config(),
                )

            return self._sampling_clients[key]

    def _evaluate_single(self, row: HDSRow, adapter_name: Optional[str] = None) -> EvaluationResult:
        """Evaluate a single HDS problem using image input with optional LoRA adapter."""
        image_path = self._get_image_path(row.id)
        prompt_text = self.prompt_text
        generation_text = None

        # Initialize results
        perplexity_losses = {}
        detected_heuristic = "UNKNOWN"
        detection_confidence = 0.0
        model_answer = None
        is_correct = False
        extraction_result = None

        if image_path.exists():
            # Compute perplexity losses for each heuristic using image (with neutral baseline)
            # Get LoRA-aware training client if adapter specified
            training_client = self._get_training_client(adapter_name)
            probe_result = self.client.compute_heuristic_losses_with_image_batched(
                image_path, row.a, row.b, training_client=training_client, include_neutral=True
            )

            # Extract from new return format
            perplexity_losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')

            # Determine preferred heuristic from perplexity
            if perplexity_losses and all(v < float('inf') for v in perplexity_losses.values()):
                detected_heuristic = min(perplexity_losses, key=perplexity_losses.get)

                # Compute confidence from loss gap
                sorted_losses = sorted(perplexity_losses.values())
                if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
                    gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
                    detection_confidence = min(1.0, gap * 2)

            # Generate answer from image using vision model
            # adapter_name is passed to get_sampling_client internally
            adapter_path = self.adapter_paths.get(adapter_name) if adapter_name else None
            model_answer, text = self.client.generate_with_image(
                image_path, row.a, row.b,
                with_reasoning=True,
                max_tokens=self.generation_max_tokens,
                adapter_name=adapter_name or "base",
                adapter_path=adapter_path,
                prompt_text=self.prompt_text
            )
            if text:
                generation_text = text
                extraction_result = extract_answer_enhanced(text, a=row.a, b=row.b)
                model_answer = extraction_result.answer
            is_correct = model_answer == row.product if model_answer is not None else False
        else:
            tprint(f"    Warning: Image not found: {image_path}")
            neutral_loss = None
            delta_losses = None

        return EvaluationResult(
            hds_id=row.id,
            a=row.a,
            b=row.b,
            product=row.product,
            target_heuristic=row.target_heuristic,
            model_answer=model_answer,
            is_correct=is_correct,
            perplexity_losses=perplexity_losses,
            detected_heuristic=detected_heuristic,
            neutral_loss=neutral_loss,
            delta_losses=delta_losses,
            extraction_confidence=extraction_result.confidence if extraction_result else None,
            extraction_strategy=extraction_result.strategy if extraction_result else None,
            is_truncated=extraction_result.is_truncated if extraction_result else None,
            prompt_text=prompt_text,
            generation_text=generation_text,
            image_path=str(image_path),
            input_modality="image"
        )

    def evaluate_hds(
        self,
        hds: List[HDSRow],
        adapter_name: Optional[str] = None,
        verbose: bool = True,
        max_workers: int = 1,
        batch_size: int = 1,
        use_async: bool = True,
        concurrency: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Evaluate HDS problems using image inputs with optional LoRA adapter."""
        if use_async:
            async_concurrency = concurrency or max(1, max_workers)
            return asyncio.run(self._evaluate_hds_async(
                hds,
                adapter_name,
                verbose,
                batch_size,
                async_concurrency
            ))

        label = adapter_name or "base"
        if verbose:
            tprint(f"\nEvaluating {label} model on {len(hds)} image problems...")
            if batch_size > 1:
                tprint(f"  Using batch size {batch_size} for perplexity probes")

        if batch_size > 1:
            return self._evaluate_hds_batched(hds, adapter_name, verbose, batch_size)

        results: List[EvaluationResult]
        if max_workers <= 1:
            results = []
            for i, row in enumerate(hds):
                if verbose:
                    tprint(f"  [{i+1}/{len(hds)}] {row.id}: {row.a} × {row.b} = {row.product} (target: {row.target_heuristic})")
                result = self._evaluate_single(row, adapter_name)
                results.append(result)
                if verbose:
                    match = "=" if result.detected_heuristic.upper() == row.target_heuristic else "≠"
                    tprint(f"         detected={result.detected_heuristic} {match} target")
        else:
            results_opt: List[Optional[EvaluationResult]] = [None] * len(hds)
            completed = 0

            def process_one(idx_row):
                idx, row = idx_row
                return idx, self._evaluate_single(row, adapter_name)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_one, (i, row)): i for i, row in enumerate(hds)}
                for future in as_completed(futures):
                    idx, result = future.result()
                    results_opt[idx] = result
                    completed += 1
                    if verbose and completed % max(1, len(hds) // 10) == 0:
                        tprint(f"  [{completed}/{len(hds)}] Processing...")

            results = cast(List[EvaluationResult], results_opt)

        if verbose:
            # Report detection distribution (no accuracy since no generation)
            detected_counts = Counter(r.detected_heuristic for r in results)
            tprint(f"  Detection distribution: {dict(detected_counts)}")

        return results

    async def _evaluate_hds_async(
        self,
        hds: List[HDSRow],
        adapter_name: Optional[str],
        verbose: bool,
        batch_size: int,
        concurrency: int
    ) -> List[EvaluationResult]:
        """Async image evaluation using sample_async + forward_async pipelines."""
        if not hds:
            return []

        client = self.client
        tokenizer = client.tokenizer
        tinker = client.tinker
        sampling_client = await self._get_sampling_client_async(adapter_name)
        TIMEOUT_SECONDS = 180

        semaphore = asyncio.Semaphore(max(1, concurrency))
        completed = [0]
        total = len(hds)
        generation_heartbeat: Optional[_ProgressHeartbeat] = None
        if verbose:
            generation_heartbeat = _ProgressHeartbeat("Image generation progress", total)
            generation_heartbeat.log_start("Starting async image generation")

        async def _generate_one(idx: int, row: HDSRow):
            def _mark_generation_progress() -> None:
                completed[0] += 1
                if generation_heartbeat is not None:
                    generation_heartbeat.maybe_log(completed[0])

            async with semaphore:
                image_path = self._get_image_path(row.id)
                if not image_path.exists():
                    if verbose:
                        tprint(f"    Warning: Image not found: {image_path}")
                    _mark_generation_progress()
                    return idx, None, None, None, str(image_path)

                # Build multimodal prompt
                model_input = None
                try:
                    if client._renderer is not None:
                        messages = [
                            {
                                'role': 'user',
                                'content': [
                                    {'type': 'image', 'image': str(image_path)},
                                    {'type': 'text', 'text': self.prompt_text},
                                ]
                            }
                        ]
                        model_input = client._renderer.build_generation_prompt(messages)
                    else:
                        image_bytes = client._load_image_bytes(image_path)
                        image_format = client._get_image_format(image_path)
                        model_input = client._build_multimodal_input_for_sampling(
                            image_bytes, image_format, self.prompt_text
                        )
                except Exception as e:
                    if verbose:
                        tprint(f"    Warning: Failed to build multimodal input for {image_path}: {e}")
                    _mark_generation_progress()
                    return idx, None, None, None, str(image_path)

                sampling_params = tinker.types.SamplingParams(
                    max_tokens=self.generation_max_tokens,
                    temperature=0.0
                )

                try:
                    result = await asyncio.wait_for(
                        sampling_client.sample_async(
                            prompt=model_input,
                            sampling_params=sampling_params,
                            num_samples=1
                        ),
                        timeout=TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    if verbose:
                        tprint(f"    Timeout for {row.id} after {TIMEOUT_SECONDS}s")
                    _mark_generation_progress()
                    return idx, None, None, None, str(image_path)
                except Exception as e:
                    if verbose:
                        tprint(f"    Warning: Async image generation failed for {row.id}: {type(e).__name__}: {e}")
                    _mark_generation_progress()
                    return idx, None, None, None, str(image_path)

                text = None
                if hasattr(result, 'samples') and len(result.samples) > 0:
                    sample = result.samples[0]
                    if hasattr(sample, 'tokens'):
                        text = tokenizer.decode(sample.tokens)
                    elif hasattr(sample, 'token_ids'):
                        text = tokenizer.decode(sample.token_ids)
                    elif hasattr(sample, 'text'):
                        text = sample.text
                    else:
                        text = str(sample)
                elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                    text = tokenizer.decode(result.sequences[0].tokens)
                elif hasattr(result, 'completions'):
                    text = result.completions[0]
                else:
                    text = str(result) if result else None

                extraction = extract_answer_enhanced(text, a=row.a, b=row.b) if text else None
                answer = extraction.answer if extraction else None

                _mark_generation_progress()

                return idx, answer, text, extraction, str(image_path)

        gen_tasks = [_generate_one(i, row) for i, row in enumerate(hds)]
        gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

        gen_map: Dict[int, Tuple[Optional[int], Optional[str], Optional[Any], str]] = {}
        for res in gen_results:
            if isinstance(res, Exception):
                continue
            idx, answer, text, extraction, img_path = res
            gen_map[idx] = (answer, text, extraction, img_path)

        score_heartbeat: Optional[_ProgressHeartbeat] = None
        if verbose:
            tprint(f"  Computing image perplexity probes asynchronously (batch_size={batch_size}, max_in_flight={self.score_max_in_flight})...")
            score_heartbeat = _ProgressHeartbeat("Image scoring progress", total)
            score_heartbeat.log_start("Starting async image scoring")

        training_client = await self._get_training_client_async(adapter_name)
        perplexity_map: Dict[int, Dict[str, Any]] = {}
        problems = [(self._get_image_path(row.id), row.a, row.b) for row in hds]
        for batch_start in range(0, total, batch_size):
            batch = problems[batch_start:batch_start + batch_size]
            batch_results = await client.compute_heuristic_losses_multi_image_async(
                batch,
                batch_size=batch_size,
                include_neutral=True,
                training_client=training_client,
                max_in_flight=self.score_max_in_flight
            )
            for local_idx, res in enumerate(batch_results):
                perplexity_map[batch_start + local_idx] = res
            if score_heartbeat is not None:
                score_heartbeat.maybe_log(min(total, batch_start + len(batch)))

        results: List[EvaluationResult] = []
        for i, row in enumerate(hds):
            model_answer, text, extraction_result, img_path = gen_map.get(i, (None, None, None, str(self._get_image_path(row.id))))

            perp_data = perplexity_map.get(i, {})
            perplexity_losses = perp_data.get("losses", {})
            neutral_loss = perp_data.get("neutral_loss")
            delta_losses = perp_data.get("delta_losses")

            detected_heuristic = "UNKNOWN"
            detection_confidence = 0.0
            if perplexity_losses and all(v < float('inf') for v in perplexity_losses.values()):
                detected_heuristic = min(perplexity_losses, key=perplexity_losses.get)
                sorted_losses = sorted(perplexity_losses.values())
                if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
                    gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
                    detection_confidence = min(1.0, gap * 2)

            is_correct = model_answer == row.product if model_answer is not None else False

            results.append(EvaluationResult(
                hds_id=row.id,
                a=row.a,
                b=row.b,
                product=row.product,
                target_heuristic=row.target_heuristic,
                model_answer=model_answer,
                is_correct=is_correct,
                perplexity_losses=perplexity_losses,
                detected_heuristic=detected_heuristic,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                extraction_confidence=extraction_result.confidence if extraction_result else None,
                extraction_strategy=extraction_result.strategy if extraction_result else None,
                is_truncated=extraction_result.is_truncated if extraction_result else None,
                prompt_text=self.prompt_text,
                generation_text=text,
                image_path=img_path,
                input_modality="image"
            ))

        if verbose:
            detected_counts = Counter(r.detected_heuristic for r in results)
            tprint(f"  Detection distribution: {dict(detected_counts)}")

        return results

    def _compute_perplexity_losses_multi(
        self,
        rows: List[HDSRow],
        adapter_name: Optional[str] = None,
        batch_size: int = 30,
        include_neutral: bool = True
    ) -> List[Dict[str, Any]]:
        """Compute batched image perplexity losses with optional LoRA adapter."""
        import asyncio
        training_client = self._get_training_client(adapter_name)
        problems = [(self._get_image_path(row.id), row.a, row.b) for row in rows]
        return asyncio.run(self.client.compute_heuristic_losses_multi_image_async(
            problems,
            batch_size=batch_size,
            include_neutral=include_neutral,
            training_client=training_client,
            max_in_flight=self.score_max_in_flight
        ))

    def _evaluate_hds_batched(
        self,
        hds: List[HDSRow],
        adapter_name: Optional[str] = None,
        verbose: bool = True,
        batch_size: int = 30
    ) -> List[EvaluationResult]:
        """
        Evaluate image HDS problems with multi-problem batching for perplexity probes.

        Batches image perplexity probes across multiple problems, then runs
        generation per-problem (generation cannot be batched).
        """
        if verbose:
            tprint(f"  Computing perplexity probes for {len(hds)} image problems (with neutral baseline)...")

        all_probe_results = self._compute_perplexity_losses_multi(
            hds, adapter_name, batch_size, include_neutral=True
        )

        if verbose:
            tprint("  Perplexity probes complete. Running generation for each problem...")

        results: List[EvaluationResult] = []
        for i, (row, probe_result) in enumerate(zip(hds, all_probe_results)):
            if verbose and (i + 1) % max(1, len(hds) // 10) == 0:
                tprint(f"  [{i+1}/{len(hds)}] Generating answers...")

            image_path = self._get_image_path(row.id)
            prompt_text = self.prompt_text
            generation_text = None

            perplexity_losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')

            detected_heuristic = "UNKNOWN"
            detection_confidence = 0.0
            if perplexity_losses and all(v < float('inf') for v in perplexity_losses.values()):
                detected_heuristic = min(perplexity_losses, key=perplexity_losses.get)
                sorted_losses = sorted(perplexity_losses.values())
                if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
                    gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
                    detection_confidence = min(1.0, gap * 2)

            model_answer = None
            is_correct = False
            extraction_result = None

            if image_path.exists():
                adapter_path = self.adapter_paths.get(adapter_name) if adapter_name else None
                model_answer, text = self.client.generate_with_image(
                    image_path, row.a, row.b,
                    with_reasoning=True,
                    max_tokens=self.generation_max_tokens,
                    adapter_name=adapter_name or "base",
                    adapter_path=adapter_path,
                    prompt_text=self.prompt_text
                )
                if text:
                    generation_text = text
                    extraction_result = extract_answer_enhanced(text, a=row.a, b=row.b)
                    model_answer = extraction_result.answer
                is_correct = model_answer == row.product if model_answer is not None else False
            else:
                if verbose:
                    tprint(f"    Warning: Image not found: {image_path}")

            results.append(EvaluationResult(
                hds_id=row.id,
                a=row.a,
                b=row.b,
                product=row.product,
                target_heuristic=row.target_heuristic,
                model_answer=model_answer,
                is_correct=is_correct,
                perplexity_losses=perplexity_losses,
                detected_heuristic=detected_heuristic,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                extraction_confidence=extraction_result.confidence if extraction_result else None,
                extraction_strategy=extraction_result.strategy if extraction_result else None,
                is_truncated=extraction_result.is_truncated if extraction_result else None,
                prompt_text=prompt_text,
                generation_text=generation_text,
                image_path=str(image_path),
                input_modality="image"
            ))

        if verbose:
            detected_counts = Counter(r.detected_heuristic for r in results)
            tprint(f"  Detection distribution: {dict(detected_counts)}")

        return results

    def compare_with_text_results(
        self,
        hds: List[HDSRow],
        text_results: List[EvaluationResult],
        verbose: bool = True,
        max_workers: int = 1,
        batch_size: int = 1
    ) -> Tuple[List[EvaluationResult], List[NudgeResult]]:
        """
        Compare image modality results with text modality results.

        Returns image results and "nudge" results showing detection changes.
        """
        # Evaluate with image modality
        image_results = self.evaluate_hds(
            hds,
            verbose=verbose,
            max_workers=max_workers,
            batch_size=batch_size
        )

        # Create lookup for text results
        text_by_id = {r.hds_id: r for r in text_results}

        # Compare and create nudge results
        nudge_results = []
        for img_r in image_results:
            text_r = text_by_id.get(img_r.hds_id)
            if text_r is None:
                continue

            # Check if detection changed between modalities
            base_label = _correctness_label(text_r)
            lora_label = _correctness_label(img_r)
            correctness_flip, correctness_flip_type = _compute_correctness_flip(base_label, lora_label)

            detection_flip = text_r.detected_heuristic != img_r.detected_heuristic
            detection_flip_type = "detection_change" if detection_flip else "unchanged"

            nudge_results.append(NudgeResult(
                hds_id=img_r.hds_id,
                a=img_r.a,
                b=img_r.b,
                target_heuristic=img_r.target_heuristic,
                base_correct=_label_to_bool(base_label),  # Text modality as "base"
                base_correctness=base_label,
                base_detected=text_r.detected_heuristic,
                lora_correct=_label_to_bool(lora_label),  # Image modality as "lora" (for comparison)
                lora_correctness=lora_label,
                lora_detected=img_r.detected_heuristic,
                is_flip=correctness_flip,
                flip_type=correctness_flip_type,
                correctness_flip=correctness_flip,
                correctness_flip_type=correctness_flip_type,
                detection_flip=detection_flip,
                detection_flip_type=detection_flip_type
            ))

        if verbose:
            flips = sum(1 for r in nudge_results if r.detection_flip)
            tprint(f"  Detection changes (text→image): {flips}/{len(nudge_results)}")

        return image_results, nudge_results

    def compare_base_vs_lora(
        self,
        hds: List[HDSRow],
        lora_name: str,
        base_results: List[EvaluationResult],
        verbose: bool = True,
        max_workers: int = 1,
        batch_size: int = 1
    ) -> Tuple[List[EvaluationResult], List[NudgeResult]]:
        """Compare base model to a specific LoRA adapter on image inputs."""
        # Evaluate with LoRA
        lora_results = self.evaluate_hds(
            hds,
            adapter_name=lora_name,
            verbose=verbose,
            max_workers=max_workers,
            batch_size=batch_size
        )

        # Compare results (note: for images we compare detection changes, not accuracy)
        nudge_results = []
        for base_r, lora_r, row in zip(base_results, lora_results, hds):
            base_label = _correctness_label(base_r)
            lora_label = _correctness_label(lora_r)
            correctness_flip, correctness_flip_type = _compute_correctness_flip(base_label, lora_label)

            detection_flip = base_r.detected_heuristic != lora_r.detected_heuristic
            detection_flip_type = "detection_change" if detection_flip else "unchanged"

            nudge_results.append(NudgeResult(
                hds_id=row.id,
                a=row.a,
                b=row.b,
                target_heuristic=row.target_heuristic,
                base_correct=_label_to_bool(base_label),
                base_correctness=base_label,
                base_detected=base_r.detected_heuristic,
                lora_correct=_label_to_bool(lora_label),
                lora_correctness=lora_label,
                lora_detected=lora_r.detected_heuristic,
                is_flip=correctness_flip,
                flip_type=correctness_flip_type,
                correctness_flip=correctness_flip,
                correctness_flip_type=correctness_flip_type,
                detection_flip=detection_flip,
                detection_flip_type=detection_flip_type
            ))

        if verbose:
            flips = sum(1 for r in nudge_results if r.detection_flip)
            tprint(f"  Detection changes: {flips}/{len(nudge_results)}")

        return lora_results, nudge_results


def analyze_nudge_results(
    heuristic: str,
    nudge_results: List[NudgeResult]
) -> Dict:
    """Analyze nudge results for a specific heuristic LoRA."""
    correctness_unknown = sum(1 for r in nudge_results if r.correctness_flip is None)
    correctness_known = len(nudge_results) - correctness_unknown
    analysis: Dict[str, Any] = {
        "heuristic": heuristic,
        "total_problems": len(nudge_results),
        "flips": {
            "total": sum(1 for r in nudge_results if r.correctness_flip),
            "improved": sum(1 for r in nudge_results if r.correctness_flip_type == "improved"),
            "degraded": sum(1 for r in nudge_results if r.correctness_flip_type == "degraded"),
            "unchanged": sum(1 for r in nudge_results if r.correctness_flip_type == "unchanged"),
            "unknown": correctness_unknown,
            "known_total": correctness_known,
        },
        "detection_flips": {
            "total": sum(1 for r in nudge_results if r.detection_flip),
            "unchanged": sum(1 for r in nudge_results if not r.detection_flip),
        },
        "flip_matrix": {
            "det_change": {"corr_change": 0, "corr_no_change": 0, "corr_unknown": 0},
            "det_no_change": {"corr_change": 0, "corr_no_change": 0, "corr_unknown": 0},
        },
        "by_design_family": {},
    }

    # 2x2 matrix: detection change x correctness change (unknown tracked separately)
    for r in nudge_results:
        det_key = "det_change" if r.detection_flip else "det_no_change"
        if r.correctness_flip is None:
            corr_key = "corr_unknown"
        else:
            corr_key = "corr_change" if r.correctness_flip else "corr_no_change"
        analysis["flip_matrix"][det_key][corr_key] += 1

    # Analyze by design family
    for family in ["RC", "DD", "OT"]:
        target_results = [r for r in nudge_results if r.target_heuristic == family]
        if not target_results:
            continue

        base_known = [r for r in target_results if r.base_correct is not None]
        lora_known = [r for r in target_results if r.lora_correct is not None]
        paired_known = [r for r in target_results if r.base_correct is not None and r.lora_correct is not None]

        analysis["by_design_family"][family] = {
            "total": len(target_results),
            "base_correct": sum(1 for r in target_results if r.base_correct is True),
            "lora_correct": sum(1 for r in target_results if r.lora_correct is True),
            "base_unknown": sum(1 for r in target_results if r.base_correct is None),
            "lora_unknown": sum(1 for r in target_results if r.lora_correct is None),
            "base_known_total": len(base_known),
            "lora_known_total": len(lora_known),
            "paired_known_total": len(paired_known),
            "improved": sum(1 for r in target_results if r.correctness_flip_type == "improved"),
            "degraded": sum(1 for r in target_results if r.correctness_flip_type == "degraded"),
        }

    # Detection changes
    detection_changes: Counter[str] = Counter()
    for r in nudge_results:
        if r.base_detected != r.lora_detected:
            detection_changes[f"{r.base_detected}->{r.lora_detected}"] += 1
    analysis["detection_changes"] = dict(detection_changes)

    return analysis


def print_nudge_analysis(all_analysis: Dict[str, Dict]):
    """Print formatted nudge analysis."""
    tprint("\n" + "=" * 60)
    tprint("NUDGE TEST RESULTS")
    tprint("=" * 60)

    for heuristic, analysis in all_analysis.items():
        tprint(f"\n{heuristic} LoRA:")
        tprint("-" * 40)

        flips = analysis["flips"]
        tprint(f"  Correctness flips: {flips['total']} total")
        tprint(f"    Improved: {flips['improved']} (base wrong -> LoRA correct)")
        tprint(f"    Degraded: {flips['degraded']} (base correct -> LoRA wrong)")
        tprint(f"    Unchanged: {flips.get('unchanged', 0)}")
        tprint(f"    Unknown: {flips.get('unknown', 0)} (low-confidence extractions)")

        det_flips = analysis.get("detection_flips", {})
        if det_flips:
            tprint(f"\n  Detection flips: {det_flips.get('total', 0)} total")

        tprint(f"\n  By design family:")
        for target, stats in analysis["by_design_family"].items():
            base_total = stats.get("base_known_total", stats["total"])
            lora_total = stats.get("lora_known_total", stats["total"])
            base_acc = stats["base_correct"] / base_total * 100 if base_total else 0.0
            lora_acc = stats["lora_correct"] / lora_total * 100 if lora_total else 0.0
            delta = lora_acc - base_acc
            tprint(f"    {target}: base {base_acc:.1f}% -> LoRA {lora_acc:.1f}% ({delta:+.1f}%)")

        if analysis["detection_changes"]:
            tprint(f"\n  Detection changes:")
            for change, count in analysis["detection_changes"].items():
                tprint(f"    {change}: {count}")

        matrix = analysis.get("flip_matrix")
        if matrix:
            tprint(f"\n  Detection × correctness matrix:")
            tprint(f"    det_change & corr_change: {matrix['det_change']['corr_change']}")
            tprint(f"    det_change & corr_no_change: {matrix['det_change']['corr_no_change']}")
            tprint(f"    det_no_change & corr_change: {matrix['det_no_change']['corr_change']}")
            tprint(f"    det_no_change & corr_no_change: {matrix['det_no_change']['corr_no_change']}")
            tprint(f"    corr_unknown (det_change): {matrix['det_change']['corr_unknown']}")
            tprint(f"    corr_unknown (det_no_change): {matrix['det_no_change']['corr_unknown']}")


def save_nudge_results(
    all_results: Dict[str, List[NudgeResult]],
    all_analysis: Dict[str, Dict],
    output_dir: Path
):
    """Save nudge test results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results CSV
    csv_path = output_dir / "nudge_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "lora", "hds_id", "a", "b", "target_heuristic",
            "base_correct", "base_correctness", "base_detected",
            "lora_correct", "lora_correctness", "lora_detected",
            "is_flip", "flip_type",
            "correctness_flip", "correctness_flip_type",
            "detection_flip", "detection_flip_type"
        ])
        writer.writeheader()
        for lora, results in all_results.items():
            for r in results:
                row = asdict(r)
                row["lora"] = lora
                writer.writerow(row)
    tprint(f"Saved results to {csv_path}")

    # Save analysis JSON
    json_path = output_dir / "nudge_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(all_analysis, f, indent=2)
    tprint(f"Saved analysis to {json_path}")


def save_detailed_nudge_results(
    base_results: List[EvaluationResult],
    all_lora_results: Dict[str, List[EvaluationResult]],
    all_nudge_results: Dict[str, List[NudgeResult]],
    hds: List[HDSRow],
    output_dir: Path
):
    """
    Save detailed nudge test results including model answers and perplexity to JSONL.

    This supplementary file preserves data that would otherwise be discarded,
    enabling re-analysis of model outputs without re-running experiments.
    """
    from datetime import datetime, timezone
    template_context = _get_template_context()

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "nudge_details.jsonl"

    with open(jsonl_path, 'w') as f:
        for i, row in enumerate(hds):
            base_r = base_results[i]

            record: Dict[str, Any] = {
                "hds_id": row.id,
                "a": row.a,
                "b": row.b,
                "product": row.product,
                "target_heuristic": row.target_heuristic,
                "design_family": row.design_family or row.target_heuristic,
                "canonical_target_heuristic": row.canonical_target_heuristic or row.target_heuristic,
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "detection_semantics": LOSS_DETECTION_SEMANTICS,
                **template_context,
                "base_evaluation": {
                    "model_answer": base_r.model_answer,
                    "is_correct": base_r.is_correct,
                    "correctness_label": _correctness_label(base_r),
                    "input": {
                        "modality": base_r.input_modality,
                        "prompt_text": base_r.prompt_text,
                        "image_path": base_r.image_path
                    },
                    "generation": {
                        "text": base_r.generation_text
                    },
                    "perplexity": base_r.perplexity_losses,
                    "neutral_loss": base_r.neutral_loss,
                    "delta_losses": base_r.delta_losses,
                    "detected_heuristic": base_r.detected_heuristic,
                    "loss_best_heuristic": base_r.detected_heuristic,
                    "detection_semantics": LOSS_DETECTION_SEMANTICS,
                },
                "lora_evaluations": {}
            }

            for heuristic, lora_results in all_lora_results.items():
                lora_r = lora_results[i]
                nudge_results = all_nudge_results.get(heuristic, [])
                nudge_r = nudge_results[i] if i < len(nudge_results) else None
                lora_name = f"{heuristic.lower()}_lora"

                record["lora_evaluations"][lora_name] = {
                    "model_answer": lora_r.model_answer,
                    "is_correct": lora_r.is_correct,
                    "correctness_label": _correctness_label(lora_r),
                    "input": {
                        "modality": lora_r.input_modality,
                        "prompt_text": lora_r.prompt_text,
                        "image_path": lora_r.image_path
                    },
                    "generation": {
                        "text": lora_r.generation_text
                    },
                    "perplexity": lora_r.perplexity_losses,
                    "neutral_loss": lora_r.neutral_loss,
                    "delta_losses": lora_r.delta_losses,
                    "detected_heuristic": lora_r.detected_heuristic,
                    "loss_best_heuristic": lora_r.detected_heuristic,
                    "detection_semantics": LOSS_DETECTION_SEMANTICS,
                    "is_flip": nudge_r.is_flip if nudge_r else None,
                    "flip_type": nudge_r.flip_type if nudge_r else None,
                    "correctness_flip": nudge_r.correctness_flip if nudge_r else None,
                    "correctness_flip_type": nudge_r.correctness_flip_type if nudge_r else None,
                    "detection_flip": nudge_r.detection_flip if nudge_r else None,
                    "detection_flip_type": nudge_r.detection_flip_type if nudge_r else None
                }

            f.write(json.dumps(record) + "\n")

    tprint(f"Saved detailed results to {jsonl_path}")


def main():
    """Run nudge test experiment."""
    parser = argparse.ArgumentParser(description="Test LoRA nudge effects on HDS problems")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="HDS split to evaluate (default: test)")
    parser.add_argument("--modality", type=str, default="text",
                        choices=["text", "image"],
                        help="Input modality (default: text)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model to use (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for perplexity probes (default: 1, no batching; >1 enables multi-problem batching)")
    parser.add_argument("--gen-max-tokens", type=int, default=DEFAULT_GEN_MAX_TOKENS,
                        help=f"Max tokens for generation (default: {DEFAULT_GEN_MAX_TOKENS})")
    parser.add_argument("--template-mode", type=str, default="multi",
                        choices=["single", "multi"],
                        help="Heuristic template mode: single or multi (default: multi)")
    parser.add_argument("--template-seed", type=int, default=None,
                        help="Seed for single-template selection (default: 0)")
    parser.add_argument("--template-profile", type=str, default=None,
                        choices=["balanced", "style_mismatch", "crosswired_stress"],
                        help="Heuristic template profile (default: balanced)")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Optional suffix for the SavedResults directory")
    parser.add_argument("--include-style-control", action="store_true",
                        help="Also evaluate a STYLE control adapter if one is available")
    parser.add_argument("--score-in-flight", type=int, default=4,
                        help="Max forward() batches in flight for pipelined scoring (default: 4)")
    args = parser.parse_args()

    if args.template_mode is not None or args.template_seed is not None:
        mode = args.template_mode or get_heuristic_template_mode()
        set_heuristic_template_mode(mode, seed=args.template_seed)
    if args.template_profile is not None:
        set_heuristic_template_profile(args.template_profile)
    template_validation = validate_active_heuristic_templates(
        expected_profile=args.template_profile or get_heuristic_template_profile(),
        expected_mode=args.template_mode or get_heuristic_template_mode(),
        expected_seed=args.template_seed if args.template_seed is not None else get_heuristic_template_seed(),
    )
    if not template_validation["is_valid"]:
        failure_reasons = []
        if not template_validation["config_matches"]:
            failure_reasons.append(
                f"Expected {template_validation['expected']}, got {template_validation['active']}."
            )
        if template_validation["validation_errors"]:
            failure_reasons.append(
                "Template audit errors: "
                + "; ".join(template_validation["validation_errors"])
            )
        raise RuntimeError(
            "Active heuristic template configuration failed validation. "
            + " ".join(failure_reasons)
        )

    # Force sequential processing to prevent race conditions in Tinker API
    # Parallel processing with shared TinkerClient can cause image/text mixing
    if args.max_workers > 1:
        tprint("WARNING: Parallel processing disabled for Tinker API calls to prevent race conditions")
        tprint("         (shared client state can cause request mixing)")
        args.max_workers = 1

    # Use unified VLM model for both modalities (enables apples-to-apples comparison)
    model_name = args.model

    tprint("=" * 60)
    tprint("LoRA Method Nudge Test")
    tprint("=" * 60)
    tprint(f"Model: {model_name}")
    tprint(f"Split: {args.split}")
    tprint(f"Modality: {args.modality}")
    tprint(f"Generation max tokens: {args.gen_max_tokens}")
    tprint(f"Template mode: {get_heuristic_template_mode()} (seed={get_heuristic_template_seed()})")
    tprint(f"Template profile: {get_heuristic_template_profile()}")
    tprint()

    # Warn if evaluating on training data
    if args.split == "train":
        tprint("WARNING: Evaluating on TRAINING split - results may be biased!")
        tprint()
    elif args.split == "all":
        tprint("NOTE: Evaluating on ALL splits (includes train data)")
        tprint()

    # Check for API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        tprint("ERROR: TINKER_API_KEY not set")
        sys.exit(1)

    # Load HDS with split filter
    tprint("Loading HDS dataset...")
    hds = load_hds(split=args.split)
    tprint(f"  Loaded {len(hds)} problems (split={args.split})")

    if len(hds) == 0:
        tprint("ERROR: No problems found for the specified split.")
        tprint("  The HDS.csv file may not have a 'split' column yet.")
        tprint("  Run: python Scripts/GenerateHDS.py --count 99")
        sys.exit(1)

    start_time = time.time()

    # Handle image modality with LoRA support (unified VLM enables cross-modal LoRA testing)
    if args.modality == "image":
        # Load adapter paths (same LoRAs work for both modalities with unified VLM)
        tprint("\nLoading trained adapter info...")
        raw_adapter_paths = load_adapter_paths(model_name=args.model)
        tprint(f"  Found adapters: {list(raw_adapter_paths.keys())}")

        missing_paths = [
            h for h, info in raw_adapter_paths.items()
            if not info.get("adapter_path") or not info.get("state_path")
        ]
        if missing_paths:
            tprint(f"ERROR: Missing adapter/state paths for {missing_paths}. Re-run TrainHeuristicLoRAs.py to save checkpoints.")
            sys.exit(1)

        # Convert to adapter_name -> path format (e.g., "rc_lora" -> path)
        adapter_paths_by_name = {
            f"{h.lower()}_lora": info["adapter_path"]
            for h, info in raw_adapter_paths.items()
            if info.get("adapter_path")
        }
        state_paths_by_name = {
            f"{h.lower()}_lora": info["state_path"]
            for h, info in raw_adapter_paths.items()
            if info.get("state_path")
        }

        tprint("\nInitializing image nudge tester...")
        tester = ImageNudgeTester(
            api_key=api_key,
            model_name=model_name,
            adapter_paths=adapter_paths_by_name,
            state_paths=state_paths_by_name,
            generation_max_tokens=args.gen_max_tokens,
            prompt_text=IMAGE_PROMPT_TEMPLATE,
            score_max_in_flight=args.score_in_flight
        )

        # Evaluate base model first
        tprint("\n" + "=" * 60)
        tprint("Evaluating Base Model (Image Modality)")
        tprint("=" * 60)
        if args.max_workers > 1:
            tprint(f"Using {args.max_workers} parallel workers")

        base_results = tester.evaluate_hds(
            hds,
            adapter_name=None,
            verbose=True,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )

        # Evaluate each LoRA
        all_nudge_results = {}
        all_lora_results = {}
        all_analysis = {}
        heuristics_to_evaluate = ["RC", "DD", "OT"]
        if args.include_style_control:
            heuristics_to_evaluate.append("STYLE")

        for heuristic in heuristics_to_evaluate:
            if heuristic not in raw_adapter_paths:
                tprint(f"\nSkipping {heuristic} - no adapter found")
                continue

            tprint(f"\n{'='*60}")
            tprint(f"Testing {heuristic} LoRA Nudge (Image Modality)")
            tprint("=" * 60)

            lora_results, nudge_results = tester.compare_base_vs_lora(
                hds=hds,
                lora_name=f"{heuristic.lower()}_lora",
                base_results=base_results,
                verbose=True,
                max_workers=args.max_workers,
                batch_size=args.batch_size
            )

            all_nudge_results[heuristic] = nudge_results
            all_lora_results[heuristic] = lora_results
            all_analysis[heuristic] = analyze_nudge_results(heuristic, nudge_results)

        elapsed = time.time() - start_time
        tprint(f"\nTotal time: {elapsed:.1f}s")

        # Print analysis
        print_nudge_analysis(all_analysis)

        # Save results (use model-specific subdirectory)
        tprint("\n" + "=" * 60)
        tprint("Saving Results")
        tprint("=" * 60)
        # Extract short model name for path (e.g., "Qwen3-VL-30B-A3B" from full path)
        model_slug = args.model.split("/")[-1].replace("-Instruct", "")
        dir_name = f"split_{args.split}_image"
        if args.output_tag:
            dir_name = f"{dir_name}_{args.output_tag}"
        output_dir = REPO_ROOT / "SavedResults" / f"nudge_test_{model_slug}" / dir_name
        save_nudge_results(all_nudge_results, all_analysis, output_dir)
        save_detailed_nudge_results(base_results, all_lora_results, all_nudge_results, hds, output_dir)
        _write_run_manifest(
            output_dir,
            {
                "script": "Scripts/experiments/LoRANudgeTest.py",
                "model": model_name,
                "split": args.split,
                "modality": args.modality,
                "output_tag": args.output_tag,
                "include_style_control": args.include_style_control,
                "elapsed_seconds": elapsed,
                "evaluated_loras": sorted(all_analysis.keys()),
                "detection_semantics": LOSS_DETECTION_SEMANTICS,
                "cli_args": vars(args),
                **template_validation,
            },
        )

    else:
        # Text modality - original behavior with LoRA testing
        # Load adapter paths (model-specific)
        tprint("\nLoading trained adapter info...")
        raw_adapter_paths = load_adapter_paths(model_name=args.model)
        tprint(f"  Found adapters: {list(raw_adapter_paths.keys())}")

        missing_paths = [
            h for h, info in raw_adapter_paths.items()
            if not info.get("adapter_path") or not info.get("state_path")
        ]
        if missing_paths:
            tprint(f"ERROR: Missing adapter/state paths for {missing_paths}. Re-run TrainHeuristicLoRAs.py to save checkpoints.")
            sys.exit(1)

        # Convert to adapter_name -> path format (e.g., "rc_lora" -> path)
        adapter_paths_by_name = {
            f"{h.lower()}_lora": info["adapter_path"]
            for h, info in raw_adapter_paths.items()
            if info.get("adapter_path")
        }
        state_paths_by_name = {
            f"{h.lower()}_lora": info["state_path"]
            for h, info in raw_adapter_paths.items()
            if info.get("state_path")
        }

        # Initialize tester with adapter paths so it can load trained LoRAs
        tprint("\nInitializing nudge tester...")
        tester = NudgeTester(
            api_key=api_key,
            model_name=model_name,
            adapter_paths=adapter_paths_by_name,
            state_paths=state_paths_by_name,
            generation_max_tokens=args.gen_max_tokens,
            prompt_template=TEXT_PROMPT_TEMPLATE,
            score_max_in_flight=args.score_in_flight
        )

        # Evaluate base model first
        tprint("\n" + "=" * 60)
        tprint("Evaluating Base Model")
        tprint("=" * 60)
        if args.max_workers > 1:
            tprint(f"Using {args.max_workers} parallel workers")
        if args.batch_size > 1:
            tprint(f"Using batch size {args.batch_size} for perplexity probes")
        base_results = tester.evaluate_hds(hds, adapter_name=None, verbose=True, max_workers=args.max_workers, batch_size=args.batch_size)

        # Evaluate each LoRA
        all_nudge_results = {}
        all_lora_results = {}
        all_analysis = {}
        heuristics_to_evaluate = ["RC", "DD", "OT"]
        if args.include_style_control:
            heuristics_to_evaluate.append("STYLE")

        for heuristic in heuristics_to_evaluate:
            if heuristic not in raw_adapter_paths:
                tprint(f"\nSkipping {heuristic} - no adapter found")
                continue

            tprint(f"\n{'='*60}")
            tprint(f"Testing {heuristic} LoRA Nudge")
            tprint("=" * 60)

            lora_results, nudge_results = tester.compare_base_vs_lora(
                hds=hds,
                lora_name=f"{heuristic.lower()}_lora",
                base_results=base_results,
                verbose=True,
                max_workers=args.max_workers,
                batch_size=args.batch_size
            )

            all_nudge_results[heuristic] = nudge_results
            all_lora_results[heuristic] = lora_results
            all_analysis[heuristic] = analyze_nudge_results(heuristic, nudge_results)

        elapsed = time.time() - start_time
        tprint(f"\nTotal time: {elapsed:.1f}s")

        # Print analysis
        print_nudge_analysis(all_analysis)

        # Save results (use model-specific subdirectory)
        tprint("\n" + "=" * 60)
        tprint("Saving Results")
        tprint("=" * 60)
        # Extract short model name for path (e.g., "Qwen3-VL-30B-A3B" from full path)
        model_slug = args.model.split("/")[-1].replace("-Instruct", "")
        dir_name = f"split_{args.split}"
        if args.output_tag:
            dir_name = f"{dir_name}_{args.output_tag}"
        output_dir = REPO_ROOT / "SavedResults" / f"nudge_test_{model_slug}" / dir_name
        save_nudge_results(all_nudge_results, all_analysis, output_dir)
        save_detailed_nudge_results(base_results, all_lora_results, all_nudge_results, hds, output_dir)
        _write_run_manifest(
            output_dir,
            {
                "script": "Scripts/experiments/LoRANudgeTest.py",
                "model": model_name,
                "split": args.split,
                "modality": args.modality,
                "output_tag": args.output_tag,
                "include_style_control": args.include_style_control,
                "elapsed_seconds": elapsed,
                "evaluated_loras": sorted(all_analysis.keys()),
                "detection_semantics": LOSS_DETECTION_SEMANTICS,
                "cli_args": vars(args),
                **template_validation,
            },
        )

    tprint("\n" + "=" * 60)
    tprint("Done!")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
