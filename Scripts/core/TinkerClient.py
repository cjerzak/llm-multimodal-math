#!/usr/bin/env python3
"""
TinkerClient.py

Shared Tinker API client for fingerprinting and LoRA experiments.

This module provides:
- TinkerClient: Unified API client with tokenizer, forward pass, and sampling
- HEURISTIC_TEMPLATES: Perplexity probe templates for OT, DD, RC
- extract_answer: Parse numerical answer from generated text
- compute_weighted_loss: Compute loss from logprobs

Usage:
    from TinkerClient import TinkerClient, HEURISTIC_TEMPLATES

    client = TinkerClient(model_name="Qwen/Qwen3-4B-Instruct-2507")
    loss = client.compute_perplexity("Some text to evaluate")
    answer, trace = client.generate(42, 17, with_reasoning=True)
"""

import json
import math
import os
import re
import sys
import time
import hashlib
import concurrent.futures
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any, TypeVar, Iterable
from dataclasses import dataclass, field

import numpy as np

from core.TinkerStartup import (
    TinkerStartupConfig,
    create_tinker_service_client,
    format_tinker_startup_config,
    load_tinker_startup_config,
)

# =============================================================================
# API TIMEOUT CONFIGURATION
# =============================================================================
# Timeout for inference API calls (forward pass, sampling)
# Max observed: ~80 sec for 235B model, so 600 sec (10 min) is generous buffer
API_CALL_TIMEOUT = 600  # seconds

# Generic type for return values
T = TypeVar('T')


def _call_with_timeout(
    future: Any,
    timeout: float = API_CALL_TIMEOUT,
    operation: str = "API call"
) -> Any:
    """
    Execute a future with timeout.

    Args:
        future: A future object with .result() method
        timeout: Timeout in seconds
        operation: Description of operation for error messages

    Returns:
        The result of future.result()

    Raises:
        TimeoutError: If the call times out
    """
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"{operation} timed out after {timeout}s")


def build_tinker_sampling_retry_config(
    progress_timeout: float = API_CALL_TIMEOUT,
) -> Any:
    """Build an SDK-native retry config for sampling requests."""
    from tinker.lib.retry_handler import RetryConfig

    return RetryConfig(progress_timeout=progress_timeout)


def _run_startup_phase(
    *,
    phase_name: str,
    model_name: str,
    startup_config: TinkerStartupConfig,
    verbose: bool,
    action: Callable[[], T],
) -> T:
    """Run an initialization phase with structured logging and actionable errors."""
    start_time = time.perf_counter()
    if verbose:
        tprint(f"  {phase_name}: start")
    try:
        result = action()
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        if verbose:
            tprint(f"  {phase_name}: failed after {elapsed:.1f}s")
        raise RuntimeError(
            (
                f"Tinker client startup failed during `{phase_name}` for model "
                f"`{model_name}` after {elapsed:.1f}s using interpreter `{sys.executable}`. "
                f"Startup settings: {format_tinker_startup_config(startup_config)}. "
                f"{type(exc).__name__}: {exc}"
            )
        ) from exc
    elapsed = time.perf_counter() - start_time
    if verbose:
        tprint(f"  {phase_name}: done in {elapsed:.1f}s")
    return result


# =============================================================================
# EXTRACTION RESULT DATACLASS
# =============================================================================
@dataclass
class ExtractionResult:
    """
    Result of answer extraction with confidence metadata.

    Attributes:
        answer: Extracted integer answer, or None if extraction failed
        confidence: Confidence score (0.0-1.0) indicating extraction reliability
        strategy: Strategy used for extraction ("boxed", "final_marker",
                  "answer_marker", "equation", "fallback", "none", "contaminated")
        is_truncated: Whether the input text appears truncated
        raw_match: The raw text that was matched and parsed
        is_contaminated: Whether the trace appears to solve a different problem
                        (neither expected operand appears in the text)

    Confidence scoring:
        - 0.95: LaTeX \\boxed{} match
        - 0.90: "Final answer is X" markers
        - 0.85: "Answer/result: X" markers
        - 0.70: Last equation result "= X"
        - 0.50: Last significant number fallback (not truncated)
        - 0.20: Fallback with truncation detected
        - 0.10: Contaminated (operands not found in trace)
        - 0.0: No extraction
    """
    answer: Optional[int]
    confidence: float
    strategy: str
    is_truncated: bool
    raw_match: Optional[str] = None
    is_contaminated: bool = False

    @property
    def is_confident(self) -> bool:
        """True if extraction has high confidence (>= 0.7) and not contaminated."""
        return self.confidence >= 0.7 and not self.is_contaminated

    @property
    def is_valid(self) -> bool:
        """True if an answer was extracted."""
        return self.answer is not None


@dataclass(frozen=True)
class ParsedTokenCount:
    """Token count parsed from an API error and the count's scope."""

    count: int
    scope: str  # "image" or "total"

# Add Scripts to path for imports when run directly
_SCRIPT_DIR = Path(__file__).parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.Logging import tprint, log_detail


# =============================================================================
# DEFAULT MODEL CONFIGURATION
# =============================================================================
# Using VLM for both text and image to enable apples-to-apples cross-modal comparison
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_VISION_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_LORA_RANK = 32

# Vision model image token count (empirically determined)
# Default fallback - actual count depends on image dimensions
VISION_IMAGE_TOKEN_COUNT = 84

# Calibrated image token counts by dimension (width, height)
# These were determined by probing the Tinker API with actual images
# Counter-intuitively, larger images may compress to fewer tokens
IMAGE_TOKEN_MAP = {
    (411, 67): 80,   # Smaller width HDS/Traps images
    (450, 67): 84,   # Medium width HDS/Traps images
    (490, 67): 66,   # Larger width HDS/Traps images
}


def get_image_dimensions(image_path) -> tuple:
    """
    Get image dimensions (width, height) from file.

    Uses PIL if available, falls back to file command.
    """
    from pathlib import Path
    image_path = Path(image_path)

    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except ImportError:
        pass

    # Fallback to file command
    try:
        import subprocess
        result = subprocess.run(['file', str(image_path)], capture_output=True, text=True)
        import re
        match = re.search(r'(\d+) x (\d+)', result.stdout)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except Exception:
        pass

    return None


def get_image_token_count(image_path) -> int:
    """
    Get the token count for an image based on its dimensions.

    Uses calibrated lookup table, falls back to default if unknown.
    """
    dims = get_image_dimensions(image_path)
    if dims and dims in IMAGE_TOKEN_MAP:
        return IMAGE_TOKEN_MAP[dims]
    return VISION_IMAGE_TOKEN_COUNT

# Chat-format templates for Qwen3-VL style prompts
CHAT_TEXT_PREFIX = "<|im_start|>user\n"
CHAT_TEXT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_VISION_PREFIX = "<|im_start|>user\n<|vision_start|>"
CHAT_VISION_SUFFIX = "<|vision_end|><|im_end|>\n<|im_start|>assistant\n"
CHAT_ASSISTANT_SUFFIX = "<|im_end|>"


# =============================================================================
# HEURISTIC TEMPLATE SELECTION
# =============================================================================
HEURISTIC_TEMPLATE_MODE_ENV = "HEURISTIC_TEMPLATE_MODE"
HEURISTIC_TEMPLATE_SEED_ENV = "HEURISTIC_TEMPLATE_SEED"
HEURISTIC_TEMPLATE_PROFILE_ENV = "HEURISTIC_TEMPLATE_PROFILE"
DEFAULT_HEURISTIC_TEMPLATE_MODE = "multi"
DEFAULT_HEURISTIC_TEMPLATE_SEED = 0
DEFAULT_HEURISTIC_TEMPLATE_PROFILE = "balanced"
MAX_TEMPLATE_TOKEN_SPREAD = 7
MAX_NEAR_DUPLICATE_RATIO = 0.94


@dataclass(frozen=True)
class HeuristicTemplateSpec:
    """Structured template metadata to keep style and heuristic cues separate."""

    heuristic: str
    style_family: str
    text: str
    semantic_cues: Tuple[str, ...] = field(default_factory=tuple)


_PROFILE_KIND = {
    "balanced": "balanced",
    "style_mismatch": "style_only",
    "crosswired_stress": "crosswired_stress",
}

_REQUIRED_CUES = {
    "OT": (
        "column",
        "columns",
        "digit by digit",
        "right to left",
        "rightmost",
        "ones place",
        "ones digit",
        "carry",
    ),
    "DD": (
        "decomposition",
        "split",
        "place value",
        "partial products",
        "distributive",
        "addends",
        "tens and ones",
    ),
    "RC": (
        "round",
        "base",
        "compensate",
        "offset",
        "adjust",
        "correction",
    ),
}

_FORBIDDEN_CUES = {
    "OT": (
        "decomposition",
        "partial products",
        "distributive",
        "split and combine",
        "split",
        "place value",
        "round",
        "base",
        "compensate",
        "offset",
        "adjust",
    ),
    "DD": (
        "column",
        "columns",
        "digit by digit",
        "right to left",
        "rightmost",
        "carry",
        "round",
        "base",
        "compensate",
        "offset",
        "adjust",
    ),
    "RC": (
        "column",
        "columns",
        "digit by digit",
        "right to left",
        "rightmost",
        "carry",
        "decomposition",
        "partial products",
        "distributive",
        "split and combine",
        "split",
        "place value",
        "addends",
        "tens and ones",
    ),
}


def _template_spec(
    heuristic: str,
    style_family: str,
    text: str,
    *semantic_cues: str,
) -> HeuristicTemplateSpec:
    """Create a structured heuristic template spec."""
    return HeuristicTemplateSpec(
        heuristic=heuristic,
        style_family=style_family,
        text=text,
        semantic_cues=tuple(semantic_cues),
    )


def _balanced_template_specs() -> Dict[str, List[HeuristicTemplateSpec]]:
    """Balanced default template bank used by the main fingerprint analysis."""
    return {
        "OT": [
            _template_spec("OT", "tagged", "Column method: start with the ones digits", "column", "ones digit"),
            _template_spec("OT", "plain", "Digit by digit: multiply the ones place first", "digit by digit", "ones place"),
            _template_spec("OT", "formal", "Standard algorithm: work through the columns from right to left", "columns", "right to left"),
            _template_spec("OT", "compact", "Schoolbook multiplication: begin with the rightmost column", "rightmost", "column"),
        ],
        "DD": [
            _template_spec("DD", "tagged", "Decomposition: split one factor into place-value parts", "decomposition", "place-value"),
            _template_spec("DD", "plain", "Partial products: break a factor into tens and ones", "partial products", "tens and ones"),
            _template_spec("DD", "formal", "Distributive method: expand one operand into addends", "distributive", "addends"),
            _template_spec("DD", "compact", "Split-and-combine: compute the partial products, then add them", "split-and-combine", "partial products"),
        ],
        "RC": [
            _template_spec("RC", "tagged", "Round and adjust: use a nearby round base, then compensate", "round", "base", "compensate"),
            _template_spec("RC", "plain", "Base approximation: round to a convenient value, then correct", "base", "round", "correct"),
            _template_spec("RC", "formal", "Compensation method: use a nearby base and fix the offset", "compensation", "base", "offset"),
            _template_spec("RC", "compact", "Near-base strategy: replace with a round number, then adjust back", "near-base", "round", "adjust"),
        ],
    }


def _style_mismatch_template_specs() -> Dict[str, List[HeuristicTemplateSpec]]:
    """Style-only ablation: wording family shifts while heuristic semantics stay fixed."""
    return {
        "OT": [
            _template_spec("OT", "structured", "Structured plan: work through the columns from the ones place", "columns", "ones place"),
            _template_spec("OT", "plain", "Plain wording: multiply digit by digit from right to left", "digit by digit", "right to left"),
            _template_spec("OT", "compact", "Compact prompt: begin with the rightmost column and carry as needed", "rightmost", "column", "carry"),
            _template_spec("OT", "formal", "Formal framing: use the standard column steps starting at the ones digit", "column", "ones digit"),
        ],
        "DD": [
            _template_spec("DD", "structured", "Structured plan: split one factor into place-value parts", "split", "place-value"),
            _template_spec("DD", "plain", "Plain wording: break a factor into tens and ones, then add the partial products", "tens and ones", "partial products"),
            _template_spec("DD", "compact", "Compact prompt: expand one operand into addends and combine the products", "addends", "expand"),
            _template_spec("DD", "formal", "Formal framing: use a distributive decomposition before summing the parts", "distributive", "decomposition"),
        ],
        "RC": [
            _template_spec("RC", "structured", "Structured plan: use a nearby round base, then correct the offset", "round", "base", "offset"),
            _template_spec("RC", "plain", "Plain wording: round to a convenient value and compensate afterward", "round", "compensate"),
            _template_spec("RC", "compact", "Compact prompt: anchor on a nearby base and adjust back at the end", "base", "adjust"),
            _template_spec("RC", "formal", "Formal framing: use a base approximation, then apply the correction", "base", "approximation", "correction"),
        ],
    }


def _crosswired_stress_template_specs() -> Dict[str, List[HeuristicTemplateSpec]]:
    """Lexical-confound stress test retained from the older cross-wired ablation."""
    return {
        "OT": [
            _template_spec("OT", "structured", "Decomposition frame: treat the columns as separate place-value parts", "columns", "place-value"),
            _template_spec("OT", "plain", "Partial-products style: begin at the ones column and carry across", "partial-products", "ones column", "carry"),
            _template_spec("OT", "formal", "Distributive wording: move right-to-left through the written columns", "distributive", "right-to-left", "columns"),
            _template_spec("OT", "compact", "Split-and-combine phrasing: write the ones column first, then carry", "split-and-combine", "ones column", "carry"),
        ],
        "DD": [
            _template_spec("DD", "structured", "Round-and-adjust style: use tens-and-ones parts instead of a base", "tens-and-ones", "base"),
            _template_spec("DD", "plain", "Compensation framing: expand one operand into addends and combine", "compensation", "addends"),
            _template_spec("DD", "formal", "Near-base wording: split a factor into place-value pieces and add", "near-base", "place-value"),
            _template_spec("DD", "compact", "Base-method style: rewrite one factor in parts, then sum the products", "base-method", "parts"),
        ],
        "RC": [
            _template_spec("RC", "structured", "Column-method style: pick a nearby round base, then correct the offset", "column-method", "round", "base", "offset"),
            _template_spec("RC", "plain", "Digit-by-digit framing: use the nearest round number and compensate", "digit-by-digit", "round", "compensate"),
            _template_spec("RC", "formal", "Schoolbook tone: take a nearby base first, then adjust the result", "schoolbook", "base", "adjust"),
            _template_spec("RC", "compact", "Right-to-left style: anchor on a round base and then fix the difference", "right-to-left", "round", "base"),
        ],
    }


def _heuristic_template_spec_bank(profile: str) -> Dict[str, List[HeuristicTemplateSpec]]:
    """Return the structured template bank for a given profile."""
    if profile == "style_mismatch":
        return _style_mismatch_template_specs()
    if profile == "crosswired_stress":
        return _crosswired_stress_template_specs()
    return _balanced_template_specs()


def _serialize_template_spec_bank(
    bank: Dict[str, List[HeuristicTemplateSpec]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert template specs into a JSON-serializable structure."""
    return {
        heuristic: [
            {
                "heuristic": spec.heuristic,
                "style_family": spec.style_family,
                "text": spec.text,
                "semantic_cues": list(spec.semantic_cues),
            }
            for spec in specs
        ]
        for heuristic, specs in bank.items()
    }


def _normalize_template_text(text: str) -> str:
    """Normalize template text for cue checks and duplicate detection."""
    lowered = text.lower().replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(lowered.split())


def _iter_template_specs(
    bank: Dict[str, List[HeuristicTemplateSpec]]
) -> Iterable[Tuple[str, int, HeuristicTemplateSpec]]:
    """Yield flattened template specs with their heuristic key and index."""
    for heuristic, specs in bank.items():
        for idx, spec in enumerate(specs):
            yield heuristic, idx, spec


def _count_template_tokens(text: str) -> int:
    """Count coarse-grained word tokens for template-length validation."""
    return len(_normalize_template_text(text).split())


def _audit_heuristic_template_bank(
    profile: str,
    bank: Dict[str, List[HeuristicTemplateSpec]],
) -> Dict[str, Any]:
    """Validate cue separation, uniqueness, and length spread for a template bank."""
    errors: List[str] = []
    warnings: List[str] = []
    profile_kind = _PROFILE_KIND.get(profile, "balanced")
    cue_separation_enforced = profile_kind != "crosswired_stress"

    normalized_seen: Dict[str, Tuple[str, int]] = {}
    flattened = list(_iter_template_specs(bank))

    for heuristic, idx, spec in flattened:
        if spec.heuristic != heuristic:
            errors.append(
                f"{profile}:{heuristic}[{idx}] has mismatched heuristic metadata {spec.heuristic}"
            )
        normalized = _normalize_template_text(spec.text)
        prior = normalized_seen.get(normalized)
        if prior is not None:
            errors.append(
                f"{profile}:{heuristic}[{idx}] duplicates template text from {prior[0]}[{prior[1]}]"
            )
        else:
            normalized_seen[normalized] = (heuristic, idx)

        if cue_separation_enforced:
            required = _REQUIRED_CUES.get(heuristic, ())
            if required and not any(term in normalized for term in required):
                errors.append(
                    f"{profile}:{heuristic}[{idx}] is missing a required {heuristic} cue: {spec.text}"
                )
            forbidden = _FORBIDDEN_CUES.get(heuristic, ())
            leaked = [term for term in forbidden if term in normalized]
            if leaked:
                errors.append(
                    f"{profile}:{heuristic}[{idx}] leaks cross-heuristic cues {leaked}: {spec.text}"
                )

    for i, (heuristic_a, idx_a, spec_a) in enumerate(flattened):
        norm_a = _normalize_template_text(spec_a.text)
        for heuristic_b, idx_b, spec_b in flattened[i + 1:]:
            norm_b = _normalize_template_text(spec_b.text)
            similarity = SequenceMatcher(None, norm_a, norm_b).ratio()
            if similarity >= MAX_NEAR_DUPLICATE_RATIO:
                errors.append(
                    f"{profile}:{heuristic_a}[{idx_a}] is near-duplicate of "
                    f"{heuristic_b}[{idx_b}] (similarity={similarity:.2f})"
                )

    token_counts = [_count_template_tokens(spec.text) for _, _, spec in flattened]
    token_spread = max(token_counts) - min(token_counts) if token_counts else 0
    if token_spread > MAX_TEMPLATE_TOKEN_SPREAD:
        errors.append(
            f"{profile} token-length spread {token_spread} exceeds {MAX_TEMPLATE_TOKEN_SPREAD}"
        )

    if not cue_separation_enforced:
        warnings.append(
            "Cue-separation checks are intentionally disabled for crosswired_stress."
        )

    return {
        "profile_kind": profile_kind,
        "cue_separation_enforced": cue_separation_enforced,
        "cue_separation_valid": cue_separation_enforced and not any(
            "leaks cross-heuristic cues" in error or "missing a required" in error
            for error in errors
        ),
        "token_length_spread": token_spread,
        "validation_errors": errors,
        "validation_warnings": warnings,
    }


def _normalize_template_mode(mode: Optional[str]) -> str:
    """Normalize template mode values to 'single' or 'multi'."""
    if not mode:
        return DEFAULT_HEURISTIC_TEMPLATE_MODE
    lowered = mode.strip().lower()
    if lowered in ("multi", "full", "all", "many"):
        return "multi"
    if lowered in ("single", "one", "1"):
        return "single"
    return DEFAULT_HEURISTIC_TEMPLATE_MODE


def get_heuristic_template_mode() -> str:
    """Return the active heuristic template mode."""
    return _normalize_template_mode(os.getenv(HEURISTIC_TEMPLATE_MODE_ENV))


def get_heuristic_template_seed() -> int:
    """Return the seed used for single-template selection."""
    raw = os.getenv(HEURISTIC_TEMPLATE_SEED_ENV)
    if raw is None or raw == "":
        return DEFAULT_HEURISTIC_TEMPLATE_SEED
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_HEURISTIC_TEMPLATE_SEED


def _normalize_template_profile(profile: Optional[str]) -> str:
    """Normalize heuristic template profile values."""
    if not profile:
        return DEFAULT_HEURISTIC_TEMPLATE_PROFILE
    lowered = profile.strip().lower()
    if lowered in ("balanced", "default", "matched"):
        return "balanced"
    if lowered in ("style", "style-mismatch", "style_mismatch", "mismatch", "ablation"):
        return "style_mismatch"
    if lowered in ("crosswired", "crosswired-stress", "crosswired_stress", "stress", "stress-test", "stress_test"):
        return "crosswired_stress"
    return DEFAULT_HEURISTIC_TEMPLATE_PROFILE


def get_heuristic_template_profile() -> str:
    """Return the active heuristic template profile."""
    return _normalize_template_profile(os.getenv(HEURISTIC_TEMPLATE_PROFILE_ENV))


def get_heuristic_template_bank(profile: Optional[str] = None) -> Dict[str, List[str]]:
    """Return the raw heuristic template bank for a profile."""
    resolved_profile = (
        get_heuristic_template_profile()
        if profile is None
        else _normalize_template_profile(profile)
    )
    spec_bank = _heuristic_template_spec_bank(resolved_profile)
    return {
        heuristic: [spec.text for spec in specs]
        for heuristic, specs in spec_bank.items()
    }


def _compute_template_bank_hash(payload: Dict[str, Any]) -> str:
    """Compute a stable short hash for template-bank provenance."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def get_effective_heuristic_template_metadata(
    profile: Optional[str] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return resolved heuristic-template metadata for manifests and JSONL output."""
    resolved_profile = (
        get_heuristic_template_profile()
        if profile is None
        else _normalize_template_profile(profile)
    )
    resolved_mode = (
        get_heuristic_template_mode()
        if mode is None
        else _normalize_template_mode(mode)
    )
    resolved_seed = get_heuristic_template_seed() if seed is None else int(seed)
    spec_bank = _heuristic_template_spec_bank(resolved_profile)
    bank_payload = {
        "profile": resolved_profile,
        "profile_kind": _PROFILE_KIND.get(resolved_profile, "balanced"),
        "mode": resolved_mode,
        "seed": resolved_seed,
        "template_bank": _serialize_template_spec_bank(spec_bank),
    }
    return {
        "template_profile": resolved_profile,
        "template_profile_kind": _PROFILE_KIND.get(resolved_profile, "balanced"),
        "template_mode": resolved_mode,
        "template_seed": resolved_seed,
        "template_bank_hash": _compute_template_bank_hash(bank_payload),
    }


def resolve_heuristic_templates(
    a: int,
    b: int,
    profile: Optional[str] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Resolve heuristic templates for explicit settings without reading env state."""
    resolved_profile = (
        get_heuristic_template_profile()
        if profile is None
        else _normalize_template_profile(profile)
    )
    resolved_mode = (
        get_heuristic_template_mode()
        if mode is None
        else _normalize_template_mode(mode)
    )
    resolved_seed = get_heuristic_template_seed() if seed is None else int(seed)
    templates = get_heuristic_template_bank(resolved_profile)

    if resolved_mode == "multi":
        return templates

    single_templates: Dict[str, List[str]] = {}
    for h_name, options in templates.items():
        idx = _stable_template_index(resolved_seed, a, b, h_name, len(options))
        single_templates[h_name] = [options[idx]]
    return single_templates


def validate_active_heuristic_templates(
    expected_profile: Optional[str] = None,
    expected_mode: Optional[str] = None,
    expected_seed: Optional[int] = None,
    sample_problem: Tuple[int, int] = (37, 42),
) -> Dict[str, Any]:
    """Validate that the active template configuration matches the requested settings."""
    resolved_profile = _normalize_template_profile(expected_profile or get_heuristic_template_profile())
    resolved_mode = _normalize_template_mode(expected_mode or get_heuristic_template_mode())
    resolved_seed = get_heuristic_template_seed() if expected_seed is None else int(expected_seed)
    spec_bank = _heuristic_template_spec_bank(resolved_profile)
    audit = _audit_heuristic_template_bank(resolved_profile, spec_bank)
    expected_templates = resolve_heuristic_templates(
        sample_problem[0],
        sample_problem[1],
        profile=resolved_profile,
        mode=resolved_mode,
        seed=resolved_seed,
    )
    active_templates = get_multi_heuristic_templates(sample_problem[0], sample_problem[1])
    config_matches = active_templates == expected_templates
    is_valid = config_matches and not audit["validation_errors"]
    return {
        "is_valid": is_valid,
        "config_matches": config_matches,
        "expected": expected_templates,
        "active": active_templates,
        **audit,
        **get_effective_heuristic_template_metadata(
            profile=resolved_profile,
            mode=resolved_mode,
            seed=resolved_seed,
        ),
    }


def set_heuristic_template_mode(mode: str, seed: Optional[int] = None) -> None:
    """Set template mode and optional seed for the current process."""
    os.environ[HEURISTIC_TEMPLATE_MODE_ENV] = _normalize_template_mode(mode)
    if seed is not None:
        os.environ[HEURISTIC_TEMPLATE_SEED_ENV] = str(seed)


def set_heuristic_template_profile(profile: str) -> None:
    """Set the heuristic template profile for the current process."""
    os.environ[HEURISTIC_TEMPLATE_PROFILE_ENV] = _normalize_template_profile(profile)


def _stable_template_index(seed: int, a: int, b: int, heuristic: str, count: int) -> int:
    """Pick a stable pseudo-random index based on inputs and seed."""
    if count <= 1:
        return 0
    payload = f"{seed}|{heuristic}|{a}|{b}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


# =============================================================================
# CONTRASTIVE TEMPLATE SELECTION
# =============================================================================
CONTRASTIVE_TEMPLATE_MODE_ENV = "CONTRASTIVE_TEMPLATE_MODE"
CONTRASTIVE_TEMPLATE_SEED_ENV = "CONTRASTIVE_TEMPLATE_SEED"
CONTRASTIVE_TEMPLATE_PROFILE_ENV = "CONTRASTIVE_TEMPLATE_PROFILE"
DEFAULT_CONTRASTIVE_TEMPLATE_MODE = "single"
DEFAULT_CONTRASTIVE_TEMPLATE_SEED = 0
DEFAULT_CONTRASTIVE_TEMPLATE_PROFILE = "standard"


def _normalize_contrastive_template_mode(mode: Optional[str]) -> str:
    """Normalize contrastive template mode values to 'single' or 'multi'."""
    if not mode:
        return DEFAULT_CONTRASTIVE_TEMPLATE_MODE
    lowered = mode.strip().lower()
    if lowered in ("multi", "full", "all", "many"):
        return "multi"
    if lowered in ("single", "one", "1"):
        return "single"
    return DEFAULT_CONTRASTIVE_TEMPLATE_MODE


def get_contrastive_template_mode() -> str:
    """Return the active contrastive template mode."""
    return _normalize_contrastive_template_mode(os.getenv(CONTRASTIVE_TEMPLATE_MODE_ENV))


def get_contrastive_template_seed() -> int:
    """Return the seed used for contrastive single-template selection."""
    raw = os.getenv(CONTRASTIVE_TEMPLATE_SEED_ENV)
    if raw is None or raw == "":
        return DEFAULT_CONTRASTIVE_TEMPLATE_SEED
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_CONTRASTIVE_TEMPLATE_SEED


def _normalize_contrastive_template_profile(profile: Optional[str]) -> str:
    """Normalize contrastive template profile values."""
    if not profile:
        return DEFAULT_CONTRASTIVE_TEMPLATE_PROFILE
    lowered = profile.strip().lower()
    if lowered in ("standard", "default", "baseline"):
        return "standard"
    if lowered in ("hard", "harder", "carry", "robust"):
        return "harder"
    return DEFAULT_CONTRASTIVE_TEMPLATE_PROFILE


def get_contrastive_template_profile() -> str:
    """Return the active contrastive template profile."""
    return _normalize_contrastive_template_profile(os.getenv(CONTRASTIVE_TEMPLATE_PROFILE_ENV))


def set_contrastive_template_mode(mode: str, seed: Optional[int] = None) -> None:
    """Set contrastive template mode and optional seed for the current process."""
    os.environ[CONTRASTIVE_TEMPLATE_MODE_ENV] = _normalize_contrastive_template_mode(mode)
    if seed is not None:
        os.environ[CONTRASTIVE_TEMPLATE_SEED_ENV] = str(seed)


def set_contrastive_template_profile(profile: str) -> None:
    """Set the contrastive template profile for the current process."""
    os.environ[CONTRASTIVE_TEMPLATE_PROFILE_ENV] = _normalize_contrastive_template_profile(profile)


# =============================================================================
# HEURISTIC TEMPLATES
# =============================================================================
def _nearest_base(n: int) -> int:
    """Find nearest round base for RC heuristic."""
    bases = [10, 50, 100, 200, 500, 1000]
    return min(bases, key=lambda b: abs(n - b))


def get_problem_prompt(a: int, b: int) -> str:
    """
    Build the user prompt for perplexity probes.

    We score loss on assistant continuation tokens, so this should contain
    the problem statement only.
    """
    return f"What is {a} × {b}?"


def _heuristic_template_bank(profile: str) -> Dict[str, List[str]]:
    """Return the heuristic template bank for a given profile."""
    return get_heuristic_template_bank(profile)


def get_heuristic_templates(a: int, b: int) -> Dict[str, str]:
    """
    Generate single continuation templates for each heuristic.

    These are intended as assistant continuations after the problem prompt.
    Operand-free for consistency across text and image modalities.

    Args:
        a: First operand (unused - kept for API compatibility)
        b: Second operand (unused - kept for API compatibility)

    Returns:
        Dict mapping heuristic name to continuation template
    """
    bank = get_heuristic_template_bank(get_heuristic_template_profile())
    return {heuristic: templates[0] for heuristic, templates in bank.items()}


# Legacy alias
HEURISTIC_TEMPLATES: Callable[[int, int], Dict[str, str]] = get_heuristic_templates


def get_multi_heuristic_templates(a: int, b: int) -> Dict[str, List[str]]:
    """
    Generate multiple perplexity probe templates per heuristic.

    DESIGN: Operand-free templates that don't mention specific operand values.
    This ensures identical templates for both text and image modalities,
    so any observed perplexity differences come from the input modality itself
    (text vs image) rather than template differences.

    Length-matched (~35-50 chars), style-matched (all imperative).
    Each template is an assistant continuation: "[Method]: [generic action]".

    Template mode can be switched via HEURISTIC_TEMPLATE_MODE:
      - "single": pick one template per heuristic (default). Selection is
        deterministic per (a, b, heuristic) using HEURISTIC_TEMPLATE_SEED.
      - "multi": return all templates for averaging.

    Args:
        a: First operand (unused - kept for API compatibility)
        b: Second operand (unused - kept for API compatibility)

    Returns:
        Dict mapping heuristic name to list of continuation templates
    """
    return resolve_heuristic_templates(
        a,
        b,
        profile=get_heuristic_template_profile(),
        mode=get_heuristic_template_mode(),
        seed=get_heuristic_template_seed(),
    )


def get_neutral_baseline_template(a: int, b: int) -> str:
    """
    Get neutral baseline template for computing Δloss.

    Used to distinguish "model prefers heuristic X" from "model dislikes all heuristics".
    The neutral baseline is a generic step-by-step preamble with no heuristic-specific cues.

    DESIGN: Operand-free template for both text and image modalities.
    This ensures any observed perplexity differences come from the input modality itself.

    Args:
        a: First operand (unused - kept for API compatibility)
        b: Second operand (unused - kept for API compatibility)

    Returns:
        Neutral baseline template string
    """
    return "Let me solve this multiplication problem step by step"


def _trailing_zero_count(value: int) -> int:
    """Count trailing zeros in a non-negative integer (treat 0 as having 1 trailing zero)."""
    if value == 0:
        return 1
    count = 0
    while value % 10 == 0:
        count += 1
        value //= 10
    return count


def _nearby_wrong_value(value: int) -> int:
    """
    Return a nearby integer that differs from value and preserves trailing zeros when present.

    - If value ends with k zeros, perturb by 10^k to keep the same trailing-zero count.
    - Otherwise, perturb by 1.
    - Prefer subtracting when it keeps the same digit length and non-negative.
    """
    if value < 0:
        value = abs(value)
    zeros = _trailing_zero_count(value) if value % 10 == 0 else 0
    step = 10 ** zeros if zeros > 0 else 1
    candidate = value - step
    if candidate >= 0 and len(str(candidate)) == len(str(value)):
        return candidate
    return value + step


def get_contrastive_step_templates(a: int, b: int) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate contrastive (correct vs incorrect) step templates for each heuristic.

    Each template pair is identical in style and differs only in the numeric step
    values, to isolate method grounding from surface-form confounds.

    Template mode can be switched via CONTRASTIVE_TEMPLATE_MODE:
      - "single": pick one template pair per heuristic (default). Selection is
        deterministic per (a, b, heuristic) using CONTRASTIVE_TEMPLATE_SEED.
      - "multi": return all template pairs for averaging.
    """
    a_ones = a % 10
    b_ones = b % 10
    ones_prod = a_ones * b_ones
    wrong_ones_prod = _nearby_wrong_value(ones_prod)
    ones_digit = ones_prod % 10
    ones_carry = ones_prod // 10
    wrong_ones_digit = wrong_ones_prod % 10
    wrong_ones_carry = wrong_ones_prod // 10

    carry_plausible_ones_prod = ones_prod - 10 if ones_carry > 0 else ones_prod + 10
    if carry_plausible_ones_prod < 0:
        carry_plausible_ones_prod = ones_prod + 10
    carry_plausible_ones_digit = carry_plausible_ones_prod % 10
    carry_plausible_ones_carry = carry_plausible_ones_prod // 10

    a_tens = (a // 10) * 10
    a_rest = a - a_tens
    b_tens = (b // 10) * 10
    b_rest = b - b_tens

    dd_correct_a = a_tens * b
    dd_wrong_a = _nearby_wrong_value(dd_correct_a)
    dd_place_value_slip_a = dd_correct_a // 10 if dd_correct_a % 10 == 0 else dd_correct_a * 10

    dd_correct_b = a * b_tens
    dd_wrong_b = _nearby_wrong_value(dd_correct_b)
    dd_place_value_slip_b = dd_correct_b // 10 if dd_correct_b % 10 == 0 else dd_correct_b * 10

    base_a = _nearest_base(a)
    diff_a = abs(a - base_a)
    rc_adjust_a = diff_a * b
    rc_adjust_a_wrong = _nearby_wrong_value(rc_adjust_a)
    rc_adjust_a_harder = rc_adjust_a + b if rc_adjust_a != 0 else b

    base_b = _nearest_base(b)
    diff_b = abs(b - base_b)
    rc_adjust_b = diff_b * a
    rc_adjust_b_wrong = _nearby_wrong_value(rc_adjust_b)
    rc_adjust_b_harder = rc_adjust_b + a if rc_adjust_b != 0 else a
    standard_templates = {
        "OT": [
            (
                f"Ones column: {a_ones} × {b_ones} = {ones_prod}; write {ones_digit}, carry {ones_carry}",
                f"Ones column: {a_ones} × {b_ones} = {wrong_ones_prod}; write {wrong_ones_digit}, carry {wrong_ones_carry}",
            ),
            (
                f"Start with ones place: {a_ones} × {b_ones} = {ones_prod}; write {ones_digit}, carry {ones_carry}",
                f"Start with ones place: {a_ones} × {b_ones} = {wrong_ones_prod}; write {wrong_ones_digit}, carry {wrong_ones_carry}",
            ),
        ],
        "DD": [
            (
                f"Decompose {a} = {a_tens} + {a_rest}; compute {a_tens} × {b} = {dd_correct_a}",
                f"Decompose {a} = {a_tens} + {a_rest}; compute {a_tens} × {b} = {dd_wrong_a}",
            ),
            (
                f"Decompose {b} = {b_tens} + {b_rest}; compute {a} × {b_tens} = {dd_correct_b}",
                f"Decompose {b} = {b_tens} + {b_rest}; compute {a} × {b_tens} = {dd_wrong_b}",
            ),
        ],
        "RC": [
            (
                f"Rounding {a} to {base_a}: adjustment {diff_a} × {b} = {rc_adjust_a}",
                f"Rounding {a} to {base_a}: adjustment {diff_a} × {b} = {rc_adjust_a_wrong}",
            ),
            (
                f"Rounding {b} to {base_b}: adjustment {diff_b} × {a} = {rc_adjust_b}",
                f"Rounding {b} to {base_b}: adjustment {diff_b} × {a} = {rc_adjust_b_wrong}",
            ),
        ],
    }

    harder_templates = {
        "OT": standard_templates["OT"] + [
            (
                f"Ones column: {a_ones} × {b_ones} = {ones_prod}; write {ones_digit}, carry {ones_carry}",
                f"Ones column: {a_ones} × {b_ones} = {carry_plausible_ones_prod}; write {carry_plausible_ones_digit}, carry {carry_plausible_ones_carry}",
            ),
            (
                f"Rightmost column: {a_ones} × {b_ones} = {ones_prod}; record {ones_digit}, carry {ones_carry}",
                f"Rightmost column: {a_ones} × {b_ones} = {carry_plausible_ones_prod}; record {carry_plausible_ones_digit}, carry {carry_plausible_ones_carry}",
            ),
        ],
        "DD": standard_templates["DD"] + [
            (
                f"Decompose {a} = {a_tens} + {a_rest}; compute {a_tens} × {b} = {dd_correct_a}",
                f"Decompose {a} = {a_tens} + {a_rest}; compute {a_tens} × {b} = {dd_place_value_slip_a}",
            ),
            (
                f"Decompose {b} = {b_tens} + {b_rest}; compute {a} × {b_tens} = {dd_correct_b}",
                f"Decompose {b} = {b_tens} + {b_rest}; compute {a} × {b_tens} = {dd_place_value_slip_b}",
            ),
        ],
        "RC": standard_templates["RC"] + [
            (
                f"Rounding {a} to {base_a}: adjustment {diff_a} × {b} = {rc_adjust_a}",
                f"Rounding {a} to {base_a}: adjustment {diff_a} × {b} = {rc_adjust_a_harder}",
            ),
            (
                f"Rounding {b} to {base_b}: adjustment {diff_b} × {a} = {rc_adjust_b}",
                f"Rounding {b} to {base_b}: adjustment {diff_b} × {a} = {rc_adjust_b_harder}",
            ),
        ],
    }

    profile = get_contrastive_template_profile()
    templates = harder_templates if profile == "harder" else standard_templates

    if get_contrastive_template_mode() == "multi":
        return templates

    seed = get_contrastive_template_seed()
    single_templates: Dict[str, List[Tuple[str, str]]] = {}
    for h_name, options in templates.items():
        idx = _stable_template_index(seed, a, b, f"contrastive:{h_name}", len(options))
        single_templates[h_name] = [options[idx]]
    return single_templates


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Number pattern for extraction (handles commas and decimals)
# Order matters: comma-formatted first, then plain numbers
# Matches: 1,200 | 1,200.00 | 1200 | 1200.0 | 2.5e3
_NUM_PATTERN = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'

def _strip_markdown_emphasis(text: str) -> str:
    """Strip Markdown emphasis markers around numeric tokens."""
    if not text:
        return text

    emphasis_patterns = [
        r'\*\*(' + _NUM_PATTERN + r')\*\*',  # **123**
        r'__(' + _NUM_PATTERN + r')__',      # __123__
        r'\*(' + _NUM_PATTERN + r')\*',      # *123*
        r'_(' + _NUM_PATTERN + r')_',        # _123_
    ]
    for pattern in emphasis_patterns:
        text = re.sub(pattern, r'\1', text)
    return text


def _parse_num(s: str) -> Optional[int]:
    """Parse a number string to int, handling commas, decimals, and scientific notation."""
    try:
        cleaned = s.replace(',', '')
        # Handle scientific notation
        return int(float(cleaned))
    except (ValueError, TypeError, OverflowError):
        return None


def detect_truncation(text: str) -> bool:
    """
    Detect if text appears truncated mid-computation.

    Checks the last portion of text for indicators that the model output
    was cut off before completing, such as:
    - Unclosed LaTeX math blocks
    - Trailing operators or equals signs
    - Step headers without content
    - Explicit truncation markers

    Args:
        text: Model output text to check

    Returns:
        True if text appears truncated, False otherwise
    """
    if not text:
        return True

    truncation_patterns = [
        r'\[\s*$',                    # Unclosed LaTeX display math
        r'\\\[\s*$',                  # Unclosed LaTeX inline (escaped)
        r'=\s*\d{1,2}\s*$',           # Suspiciously short number at end after =
        r'\.\.\.\s*$',                # Explicit truncation ellipsis
        r'###\s*Step\s*\d+[^#]*$',    # Ends mid-step header
        r'[+\-*/×]\s*$',              # Ends with operator
        r'=\s*$',                     # Ends with bare equals
        r'\(\s*$',                    # Unclosed parenthesis
        r'\{\s*$',                    # Unclosed brace
    ]

    # Check last 50 chars for truncation markers
    tail = text[-50:] if len(text) > 50 else text
    return any(re.search(p, tail) for p in truncation_patterns)


def is_reasonable_answer(val: int, a: Optional[int], b: Optional[int]) -> bool:
    """
    Check if extracted value is plausible for a × b.

    Uses a 10x margin to catch most computation errors while rejecting
    obvious hallucinations (e.g., model outputs "2491000000" for 47 × 53).

    Args:
        val: Extracted answer value
        a: First operand (optional)
        b: Second operand (optional)

    Returns:
        True if val is within reasonable range, or if a/b not provided
    """
    if a is None or b is None:
        return True  # Can't validate without operands
    if val is None:
        return False

    expected = a * b
    # Allow 10x margin (catches computation errors, rejects hallucinations)
    lower = max(1, expected // 10)
    upper = expected * 10
    return lower <= val <= upper


def _extract_fallback_answer(
    text: str,
    a: Optional[int],
    b: Optional[int],
    is_truncated: bool
) -> Optional[int]:
    """
    Extract answer using fallback strategy (last significant number).

    Used for contaminated traces where we still want to capture some value
    for debugging purposes, but with very low confidence.

    Args:
        text: Generated text
        a: First operand (for validation, may be None)
        b: Second operand (for validation, may be None)
        is_truncated: Whether text appears truncated

    Returns:
        Extracted integer or None
    """
    all_numbers = re.findall(_NUM_PATTERN, text)
    if not all_numbers:
        return None

    # Try last few numbers, prefer significant ones
    for num_str in reversed(all_numbers[-3:]):
        result = _parse_num(num_str)
        if result is not None and result >= 10:
            return result

    # Try last number even if small
    result = _parse_num(all_numbers[-1])
    return result


def extract_answer_enhanced(
    text: str,
    a: Optional[int] = None,
    b: Optional[int] = None
) -> ExtractionResult:
    """
    Extract numerical answer with confidence scoring and validation.

    Uses a priority-ordered multi-strategy approach:
    1. LaTeX \\boxed{} - highest confidence (0.95)
    2. Extended LaTeX (\\mathbf{}, inline $...$) - confidence 0.92
    3. "Final answer is X" markers - confidence 0.90
    4. Answer/result/product markers - confidence 0.85
    5. Last equation result "= X" - confidence 0.70
    6. Last significant number fallback - confidence 0.50 (or 0.20 if truncated)

    Also detects contamination: if operands are provided but neither appears
    in the trace, the model may be solving the wrong problem.

    Args:
        text: Generated text containing the answer
        a: First operand for answer validation (optional)
        b: Second operand for answer validation (optional)

    Returns:
        ExtractionResult with answer, confidence, strategy, and truncation status
    """
    if not text:
        return ExtractionResult(None, 0.0, "none", True, is_contaminated=False)

    text = text.strip()
    is_truncated = detect_truncation(text)
    text = _strip_markdown_emphasis(text)

    # Contamination detection: check if expected operands appear in the trace
    # If neither operand is present, the model may be solving a different problem
    is_contaminated = False
    if a is not None and b is not None:
        # Check for operand presence (allow for formatted numbers like "1,900")
        a_str = str(a)
        b_str = str(b)
        text_normalized = text.replace(",", "")  # Remove thousands separators
        operand_present = a_str in text_normalized or b_str in text_normalized
        if not operand_present:
            is_contaminated = True
            # Return early with contaminated flag - extracted answer is unreliable
            # Still try to extract for debugging purposes, but with low confidence
            fallback_result = _extract_fallback_answer(text, a, b, is_truncated)
            if fallback_result is not None:
                return ExtractionResult(
                    fallback_result, 0.10, "contaminated", is_truncated,
                    raw_match=None, is_contaminated=True
                )
            return ExtractionResult(None, 0.0, "contaminated", is_truncated, is_contaminated=True)

    # Strategy 1: LaTeX \boxed{} - highest priority
    boxed = re.findall(r'\\boxed\{(' + _NUM_PATTERN + r')\}', text)
    if boxed:
        result = _parse_num(boxed[-1])
        if result is not None and is_reasonable_answer(result, a, b):
            return ExtractionResult(result, 0.95, "boxed", is_truncated, boxed[-1])

    # Strategy 1.5: Extended LaTeX patterns
    latex_patterns = [
        (r'\\mathbf\{(' + _NUM_PATTERN + r')\}', 0.92),
        (r'\\textbf\{(' + _NUM_PATTERN + r')\}', 0.92),
        (r'\$\s*(' + _NUM_PATTERN + r')\s*\$', 0.90),
    ]
    for pattern, conf in latex_patterns:
        matches = re.findall(pattern, text)
        if matches:
            result = _parse_num(matches[-1])
            if result is not None and is_reasonable_answer(result, a, b):
                return ExtractionResult(result, conf, "latex", is_truncated, matches[-1])

    # Strategy 2: "Final answer" indicators - high priority
    final_patterns = [
        r'(?:final|therefore|thus|so|hence)\s*(?:the\s+)?(?:answer|result|product)\s*(?:is|=|:)\s*(' + _NUM_PATTERN + r')',
        r'(?:answer|result|product)\s*[:=]\s*\\boxed\{(' + _NUM_PATTERN + r')\}',
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result = _parse_num(matches[-1])
            if result is not None and is_reasonable_answer(result, a, b):
                return ExtractionResult(result, 0.90, "final_marker", is_truncated, matches[-1])

    # Strategy 3: Explicit answer markers
    answer_patterns = [
        r'(?:answer|result|total|product)\s*(?:is|=|:)\s*(' + _NUM_PATTERN + r')',
        r'(?:equals?|is)\s+(' + _NUM_PATTERN + r')(?:\s*[.!?\n]|$)',
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result = _parse_num(matches[-1])
            if result is not None and is_reasonable_answer(result, a, b):
                return ExtractionResult(result, 0.85, "answer_marker", is_truncated, matches[-1])

    # Strategy 4: Equation results "a × b = X" or just "= X" at end
    equation_patterns = [
        r'×\s*\d+\s*=\s*(' + _NUM_PATTERN + r')',      # "47 × 53 = 2491"
        r'\*\s*\d+\s*=\s*(' + _NUM_PATTERN + r')',     # "47 * 53 = 2491"
        r'=\s*(' + _NUM_PATTERN + r')\s*$',            # "= 2491" at end
        r'=\s*(' + _NUM_PATTERN + r')(?:\s*[.!?\n])',  # "= 2491." or "= 2491\n"
    ]
    for pattern in equation_patterns:
        matches = re.findall(pattern, text)
        if matches:
            result = _parse_num(matches[-1])
            if result is not None and is_reasonable_answer(result, a, b):
                return ExtractionResult(result, 0.70, "equation", is_truncated, matches[-1])

    # Strategy 5: Fallback - find all numbers, prefer last significant one
    # Reduce confidence if truncated (higher risk of picking intermediate)
    fallback_confidence = 0.20 if is_truncated else 0.50

    all_numbers = re.findall(_NUM_PATTERN, text)
    if all_numbers:
        # Try to find last significant number (>= 10)
        for num_str in reversed(all_numbers):
            result = _parse_num(num_str)
            if result is not None and result >= 10:
                if is_reasonable_answer(result, a, b):
                    return ExtractionResult(result, fallback_confidence, "fallback", is_truncated, num_str)

        # Try last number even if small
        result = _parse_num(all_numbers[-1])
        if result is not None:
            return ExtractionResult(result, fallback_confidence * 0.8, "fallback", is_truncated, all_numbers[-1])

    return ExtractionResult(None, 0.0, "none", is_truncated)


def extract_answer(text: str) -> Optional[int]:
    """
    Extract numerical answer from generated text.

    This is a backward-compatible wrapper around extract_answer_enhanced().
    For new code, prefer extract_answer_enhanced() which provides confidence
    scoring and truncation detection.

    Args:
        text: Generated text containing the answer

    Returns:
        Integer answer or None if not found
    """
    result = extract_answer_enhanced(text)
    return result.answer


def compute_weighted_loss(logprobs_raw, weights: List[int]) -> float:
    """
    Compute weighted average loss from logprobs.

    Args:
        logprobs_raw: Raw logprobs from Tinker API (may need .tolist())
        weights: Weight for each token position

    Returns:
        Weighted average loss (negative log probability)
    """
    if hasattr(logprobs_raw, 'tolist'):
        logprobs = np.array(logprobs_raw.tolist())
    else:
        logprobs = np.array(logprobs_raw)
    weights_arr = np.array(weights)
    return float(-np.dot(logprobs, weights_arr) / weights_arr.sum())


def _build_chat_text_datum(tinker_module, tokenizer, prompt: str):
    """
    Build a chat-formatted Datum for text-only perplexity.

    Computes loss only on the user prompt tokens, masking chat wrapper tokens.
    Returns (datum, weights) or (None, None) if prompt is too short.
    """
    prefix_tokens = tokenizer.encode(CHAT_TEXT_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(CHAT_TEXT_SUFFIX, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    if not prompt_tokens:
        return None, None

    full_tokens = prefix_tokens + prompt_tokens + suffix_tokens
    if len(full_tokens) < 2:
        return None, None

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    mask = [0] * len(prefix_tokens) + [1] * len(prompt_tokens) + [0] * len(suffix_tokens)
    weights = mask[1:]
    if sum(weights) == 0:
        return None, None

    model_input = tinker_module.types.ModelInput.from_ints(tokens=input_tokens)
    datum = tinker_module.types.Datum(
        model_input=model_input,
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )
    return datum, weights


def _build_chat_text_model_input(tinker_module, tokenizer, prompt: str):
    """
    Build chat-formatted ModelInput for text generation.

    Returns ModelInput or None if the prompt is empty.
    """
    prefix_tokens = tokenizer.encode(CHAT_TEXT_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(CHAT_TEXT_SUFFIX, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    if not prompt_tokens:
        return None

    full_tokens = prefix_tokens + prompt_tokens + suffix_tokens
    return tinker_module.types.ModelInput.from_ints(tokens=full_tokens)


def build_chat_prompt_response_tokens(
    tokenizer,
    prompt: str,
    response: str,
    include_assistant_end: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build chat-formatted tokens and weights for a prompt/response pair.

    Returns (input_tokens, target_tokens, weights) with weights applied only
    to assistant response tokens (and optional assistant end token).
    """
    prefix_tokens = tokenizer.encode(CHAT_TEXT_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(CHAT_TEXT_SUFFIX, add_special_tokens=False)
    end_tokens = tokenizer.encode(CHAT_ASSISTANT_SUFFIX, add_special_tokens=False) if include_assistant_end else []
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    if not prompt_tokens or not response_tokens:
        return [], [], []

    full_tokens = prefix_tokens + prompt_tokens + suffix_tokens + response_tokens + end_tokens
    if len(full_tokens) < 2:
        return [], [], []

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    mask = (
        [0] * (len(prefix_tokens) + len(prompt_tokens) + len(suffix_tokens))
        + [1] * len(response_tokens)
        + [1] * len(end_tokens)
    )
    weights = mask[1:]
    if sum(weights) == 0:
        return [], [], []

    return input_tokens, target_tokens, weights


def _build_chat_prompt_response_datum(
    tinker_module,
    tokenizer,
    prompt: str,
    response: str,
    include_assistant_end: bool = True
):
    """
    Build a chat-formatted Datum for a prompt/response pair.

    Returns (datum, weights) or (None, None) if inputs are empty.
    """
    input_tokens, target_tokens, weights = build_chat_prompt_response_tokens(
        tokenizer,
        prompt,
        response,
        include_assistant_end=include_assistant_end
    )
    if not input_tokens:
        return None, None

    model_input = tinker_module.types.ModelInput.from_ints(tokens=input_tokens)
    datum = tinker_module.types.Datum(
        model_input=model_input,
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )
    return datum, weights


# =============================================================================
# TINKER CLIENT
# =============================================================================
class TinkerClient:
    """
    Unified Tinker API client for fingerprinting and LoRA experiments.

    Handles:
    - API initialization and authentication
    - Tokenizer loading with pad_token fallback
    - Forward pass for perplexity computation
    - Sampling for text generation
    - LoRA training client management

    Example:
        client = TinkerClient()

        # Compute perplexity
        loss = client.compute_perplexity("Some text")

        # Generate answer
        answer, trace = client.generate(42, 17, with_reasoning=True)

        # Get heuristic perplexities
        losses = client.compute_heuristic_losses(42, 17)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        lora_rank: int = DEFAULT_LORA_RANK,
        api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Tinker client.

        Args:
            model_name: HuggingFace model name (must be Tinker-supported)
            lora_rank: LoRA rank for training clients
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
            verbose: Print initialization messages
        """
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TINKER_API_KEY not set. "
                "Set via environment variable or pass api_key parameter."
            )

        self.model_name = model_name
        self.lora_rank = lora_rank
        self.verbose = verbose

        # Lazy-loaded components
        self._tinker: Any = None
        self._service_client: Any = None
        self._tokenizer: Any = None
        self._training_client: Any = None
        self._sampling_client: Any = None
        self._sampling_clients: Dict[str, Any] = {}
        self._hf_token: Optional[str] = None
        self._startup_config = load_tinker_startup_config()

        # Initialize
        _run_startup_phase(
            phase_name="init_api",
            model_name=self.model_name,
            startup_config=self._startup_config,
            verbose=self.verbose,
            action=self._init_api,
        )
        _run_startup_phase(
            phase_name="init_tokenizer",
            model_name=self.model_name,
            startup_config=self._startup_config,
            verbose=self.verbose,
            action=self._init_tokenizer,
        )

    def _init_api(self):
        """Initialize Tinker API and HuggingFace authentication."""
        import tinker

        # HuggingFace authentication: huggingface_hub auto-detects HF_TOKEN env var,
        # so explicit login() is not needed and can cause conflicts in parallel execution
        self._hf_token = os.getenv("HF_TOKEN")
        if self._hf_token:
            if self.verbose:
                tprint("  HF_TOKEN detected (auto-authenticated)")
        elif self.verbose:
            tprint("  Warning: HF_TOKEN not set - gated models may not be accessible")

        self._tinker = tinker
        self._service_client = create_tinker_service_client(
            tinker_module=tinker,
            api_key=self.api_key,
            config=self._startup_config,
        )

    def _init_tokenizer(self):
        """Initialize tokenizer with pad_token fallback."""
        from transformers import AutoTokenizer

        if self.verbose:
            tprint(f"  Loading tokenizer for {self.model_name}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self._hf_token
        )

        # Handle models without pad token (GPT-2, GPT-Neo, etc.)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def tinker(self):
        """Access to tinker module."""
        return self._tinker

    @property
    def tokenizer(self):
        """Access to tokenizer."""
        return self._tokenizer

    @property
    def service_client(self):
        """Access to Tinker service client."""
        return self._service_client

    def _build_text_datum(self, prompt: str) -> Tuple[Optional[Any], Optional[List[int]]]:
        """Build chat-formatted Datum for text-only loss computation."""
        return _build_chat_text_datum(self._tinker, self._tokenizer, prompt)

    def _build_text_response_datum(
        self,
        prompt: str,
        response: str
    ) -> Tuple[Optional[Any], Optional[List[int]]]:
        """Build chat-formatted Datum scoring only assistant continuation tokens."""
        return _build_chat_prompt_response_datum(self._tinker, self._tokenizer, prompt, response)

    def build_text_generation_input(self, prompt: str) -> Optional[Any]:
        """Build chat-formatted ModelInput for text generation."""
        return _build_chat_text_model_input(self._tinker, self._tokenizer, prompt)

    def get_training_client(self, force_new: bool = False):
        """
        Get or create a LoRA training client.

        Args:
            force_new: Force creation of new client

        Returns:
            Tinker LoRA training client
        """
        if self._training_client is None or force_new:
            if self.verbose:
                tprint("  Initializing LoRA training client...")
            self._training_client = self._service_client.create_lora_training_client(
                base_model=self.model_name,
                rank=self.lora_rank
            )
        return self._training_client

    def get_sampling_client(self, adapter_name: str = "base", force_new: bool = False):
        """
        Get or create a sampling client.

        Args:
            adapter_name: Name for the adapter checkpoint
            force_new: Force creation of new client

        Returns:
            Tinker sampling client
        """
        cache_key = f"name:{adapter_name}"
        if force_new or cache_key not in self._sampling_clients:
            if self.verbose:
                tprint(f"  Initializing sampling client ({adapter_name})...")
            training = self.get_training_client()
            sampling_client = training.save_weights_and_get_sampling_client(
                name=adapter_name,
                retry_config=build_tinker_sampling_retry_config(),
            )
            self._sampling_clients[cache_key] = sampling_client
        self._sampling_client = self._sampling_clients[cache_key]
        return self._sampling_client

    def compute_perplexity(self, prompt: str) -> float:
        """
        Compute perplexity/loss for a prompt using forward pass.

        Args:
            prompt: Text to evaluate

        Returns:
            Average loss (lower = more likely under model)
        """
        try:
            client = self.get_training_client()
            datum, weights = self._build_text_datum(prompt)
            if datum is None or weights is None:
                return float('inf')

            # Forward pass (no gradients) with timeout
            future = client.forward([datum], "cross_entropy")
            result = _call_with_timeout(future, operation="perplexity forward pass")

            # Extract loss
            if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > 0:
                logprobs_raw = result.loss_fn_outputs[0]['logprobs']
                return compute_weighted_loss(logprobs_raw, weights)
            elif hasattr(result, 'loss'):
                return float(result.loss)
            else:
                return float('inf')

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Perplexity computation timed out: {e}")
            return float('inf')
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Perplexity computation failed: {e}")
            return float('inf')

    def compute_heuristic_losses(self, a: int, b: int) -> Dict[str, float]:
        """
        Compute perplexity losses for all heuristic templates.

        Loss is measured on assistant continuation tokens given the problem prompt.

        Batches all 3 templates (OT, DD, RC) into a single forward call
        for efficiency (66% reduction in API calls vs sequential).

        Args:
            a: First operand
            b: Second operand

        Returns:
            Dict mapping heuristic name to loss value
        """
        templates = get_heuristic_templates(a, b)
        prompt = get_problem_prompt(a, b)

        try:
            client = self.get_training_client()

            # Prepare all 3 datums for batched call
            datums: List[Any] = []
            heuristic_names: List[str] = []
            weights_list: List[Optional[List[int]]] = []

            for h_name, continuation in templates.items():
                datum, weights = self._build_text_response_datum(prompt, continuation)
                if datum is None:
                    heuristic_names.append(h_name)
                    weights_list.append(None)
                    continue
                datums.append(datum)
                heuristic_names.append(h_name)
                weights_list.append(weights)

            if not datums:
                return {h: float('inf') for h in templates.keys()}

            # Single batched API call for all 3 templates with timeout
            future = client.forward(datums, "cross_entropy")
            result = _call_with_timeout(future, operation="batched heuristic forward pass")

            # Extract losses from batch result
            losses = {}
            datum_idx = 0
            for i, h_name in enumerate(heuristic_names):
                weights = weights_list[i]
                if weights is None:
                    losses[h_name] = float('inf')
                elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    losses[h_name] = compute_weighted_loss(logprobs, weights)
                    datum_idx += 1
                else:
                    # API returned fewer outputs than expected - don't increment datum_idx
                    # as it would cause subsequent heuristics to get wrong loss values
                    losses[h_name] = float('inf')

            return losses

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Batched perplexity computation timed out: {e}")
            return {h: float('inf') for h in templates.keys()}
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Batched perplexity computation failed: {e}")
            return {h: float('inf') for h in templates.keys()}

    def detect_heuristic(self, a: int, b: int) -> Tuple[str, float]:
        """
        Detect preferred heuristic from perplexity probe.

        Args:
            a: First operand
            b: Second operand

        Returns:
            (detected_heuristic, confidence) tuple
        """
        losses = self.compute_heuristic_losses(a, b)

        if not losses or not all(v < float('inf') for v in losses.values()):
            return "UNKNOWN", 0.0

        # Find lowest loss = preferred heuristic
        best = min(losses, key=lambda h: losses[h])

        # Compute confidence from loss gap
        sorted_losses = sorted(losses.values())
        if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
            gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
            confidence = min(1.0, gap * 2)
        else:
            confidence = 0.5

        return best, confidence

    def compute_heuristic_losses_with_baseline(self, a: int, b: int) -> Dict:
        """
        Compute perplexity losses using multi-template averaging and neutral baseline.

        Loss is measured on assistant continuation tokens given the problem prompt.

        This implements the full PerplexityProbe methodology:
        1. Averages losses across templates per heuristic (count depends on template mode)
        2. Computes neutral baseline loss
        3. Returns Δloss (loss - baseline) for each heuristic

        The Δloss values are more interpretable than raw losses because they
        measure preference relative to a neutral preamble rather than absolute values.

        Args:
            a: First operand
            b: Second operand

        Returns:
            Dict with keys:
                'losses': Dict[str, float] - Average raw loss per heuristic
                'neutral_loss': float - Baseline loss
                'delta_losses': Dict[str, float] - Loss relative to baseline
                'per_template_losses': Dict[str, Dict] - Per-template prompts and losses
                    Keys like 'OT_0', 'DD_1', 'NEUTRAL', values are {'prompt': str, 'loss': float}
                'best_heuristic': str - Heuristic with lowest loss
                'confidence': float - Detection confidence (0-1)
        """
        multi_templates = get_multi_heuristic_templates(a, b)
        neutral_template = get_neutral_baseline_template(a, b)
        prompt = get_problem_prompt(a, b)

        try:
            client = self.get_training_client()

            # Prepare all datums: templates per heuristic + neutral baseline
            datums: List[Any] = []
            metadata: List[Tuple[str, int, Optional[List[int]], str]] = []  # (heuristic_name, template_idx, weights, continuation)

            # Add heuristic templates
            for h_name, templates in multi_templates.items():
                for t_idx, continuation in enumerate(templates):
                    datum, weights = self._build_text_response_datum(prompt, continuation)
                    if datum is None:
                        metadata.append((h_name, t_idx, None, continuation))
                        continue

                    datums.append(datum)
                    metadata.append((h_name, t_idx, weights, continuation))

            # Add neutral baseline
            neutral_datum, neutral_weights = self._build_text_response_datum(prompt, neutral_template)
            if neutral_datum is not None:
                datums.append(neutral_datum)
                metadata.append(("NEUTRAL", 0, neutral_weights, neutral_template))
            else:
                metadata.append(("NEUTRAL", 0, None, neutral_template))

            if not datums:
                return self._empty_baseline_result()

            # Single batched API call for all templates with timeout
            future = client.forward(datums, "cross_entropy")
            result = _call_with_timeout(future, operation="multi-template forward pass")

            # Extract losses and group by heuristic
            heuristic_losses: Dict[str, List[float]] = {h: [] for h in multi_templates.keys()}
            per_template_losses: Dict[str, Dict[str, Any]] = {}  # template_id -> {"prompt": ..., "loss": ...}
            neutral_loss = float('inf')
            datum_idx = 0

            for h_name, t_idx, weights, continuation in metadata:
                template_id = f"{h_name}_{t_idx}" if h_name != "NEUTRAL" else "NEUTRAL"

                if weights is None:
                    per_template_losses[template_id] = {"prompt": continuation, "loss": float('inf')}
                    if h_name != "NEUTRAL":
                        heuristic_losses[h_name].append(float('inf'))
                    continue

                if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                    per_template_losses[template_id] = {"prompt": continuation, "loss": loss}
                    if h_name == "NEUTRAL":
                        neutral_loss = loss
                    else:
                        heuristic_losses[h_name].append(loss)
                    datum_idx += 1
                else:
                    # API returned fewer outputs than expected - don't increment datum_idx
                    per_template_losses[template_id] = {"prompt": continuation, "loss": float('inf')}
                    if h_name != "NEUTRAL":
                        heuristic_losses[h_name].append(float('inf'))

            # Compute average loss per heuristic
            avg_losses: Dict[str, float] = {}
            for h, losses in heuristic_losses.items():
                valid_loss_values = [l for l in losses if l < float('inf')]
                avg_losses[h] = sum(valid_loss_values) / len(valid_loss_values) if valid_loss_values else float('inf')

            # Compute delta losses relative to neutral baseline
            delta_losses = {h: loss - neutral_loss for h, loss in avg_losses.items()}

            # Find best heuristic and compute confidence
            valid_loss_map = {h: l for h, l in avg_losses.items() if l < float('inf')}
            if valid_loss_map:
                best_heuristic = min(valid_loss_map, key=lambda h: valid_loss_map[h])
                sorted_vals = sorted(valid_loss_map.values())
                if len(sorted_vals) >= 2 and sorted_vals[1] > 0:
                    gap = (sorted_vals[1] - sorted_vals[0]) / sorted_vals[1]
                    confidence = min(1.0, gap * 2)
                else:
                    confidence = 0.5
            else:
                best_heuristic = "UNKNOWN"
                confidence = 0.0

            return {
                'losses': avg_losses,
                'neutral_loss': neutral_loss,
                'delta_losses': delta_losses,
                'per_template_losses': per_template_losses,
                'best_heuristic': best_heuristic,
                'confidence': confidence
            }

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Multi-template perplexity computation timed out: {e}")
            return self._empty_baseline_result()
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Multi-template perplexity computation failed: {e}")
            return self._empty_baseline_result()

    def compute_contrastive_step_losses(self, a: int, b: int) -> Dict[str, Any]:
        """
        Compute contrastive losses for correct vs incorrect heuristic steps.

        Loss is measured on assistant continuation tokens given the problem prompt.
        Returns averaged losses per heuristic across contrastive variants.
        """
        templates = get_contrastive_step_templates(a, b)
        prompt = get_problem_prompt(a, b)

        try:
            client = self.get_training_client()

            datums: List[Any] = []
            metadata: List[Tuple[str, int, str, Optional[List[int]], str]] = []

            for h_name, pairs in templates.items():
                for t_idx, (correct, incorrect) in enumerate(pairs):
                    for label, continuation in (("correct", correct), ("incorrect", incorrect)):
                        datum, weights = self._build_text_response_datum(prompt, continuation)
                        if datum is None:
                            metadata.append((h_name, t_idx, label, None, continuation))
                            continue
                        datums.append(datum)
                        metadata.append((h_name, t_idx, label, weights, continuation))

            if not datums:
                return self._empty_contrastive_result()

            future = client.forward(datums, "cross_entropy")
            result = _call_with_timeout(future, operation="contrastive step forward pass")

            correct_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            incorrect_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            per_template_losses: Dict[str, Dict[str, Any]] = {}

            datum_idx = 0
            for h_name, t_idx, label, weights, continuation in metadata:
                template_id = f"{h_name}_{t_idx}_{label.upper()}"
                if weights is None:
                    loss = float('inf')
                elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                    datum_idx += 1
                else:
                    loss = float('inf')

                per_template_losses[template_id] = {"prompt": continuation, "loss": loss}
                if label == "correct":
                    correct_losses[h_name].append(loss)
                else:
                    incorrect_losses[h_name].append(loss)

            avg_correct = {}
            avg_incorrect = {}
            delta_losses = {}
            for h in templates.keys():
                correct_vals = [l for l in correct_losses[h] if l < float('inf')]
                incorrect_vals = [l for l in incorrect_losses[h] if l < float('inf')]
                avg_correct[h] = sum(correct_vals) / len(correct_vals) if correct_vals else float('inf')
                avg_incorrect[h] = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else float('inf')
                delta_losses[h] = avg_incorrect[h] - avg_correct[h]

            return {
                "correct_losses": avg_correct,
                "incorrect_losses": avg_incorrect,
                "delta_losses": delta_losses,
                "per_template_losses": per_template_losses
            }

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Contrastive step computation timed out: {e}")
            return self._empty_contrastive_result()
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Contrastive step computation failed: {e}")
            return self._empty_contrastive_result()

    def _empty_baseline_result(self) -> Dict:
        """Return empty result structure for compute_heuristic_losses_with_baseline."""
        return {
            'losses': {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
            'neutral_loss': float('inf'),
            'delta_losses': {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
            'per_template_losses': {},
            'best_heuristic': "UNKNOWN",
            'confidence': 0.0
        }

    def _empty_contrastive_result(self) -> Dict[str, Any]:
        """Return empty result structure for contrastive step probing."""
        losses = {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')}
        return {
            "correct_losses": losses.copy(),
            "incorrect_losses": losses.copy(),
            "delta_losses": losses.copy(),
            "per_template_losses": {}
        }

    def compute_heuristic_losses_multi(
        self,
        problems: List[Tuple[int, int]],
        batch_size: int = 30,
        include_neutral: bool = True,
        training_client=None
    ) -> List[Dict[str, Any]]:
        """
        Compute heuristic losses for multiple problems in batched API calls.

        Batches heuristic probe templates per problem (count depends on template mode)
        into larger API calls for speed while matching the single-problem probe design.

        Args:
            problems: List of (a, b) tuples
            batch_size: Max datums per API call (default 30 = ~2 problems × 12 templates)
            include_neutral: Include neutral baseline template for delta loss computation
            training_client: Optional training client (e.g., LoRA-loaded) to use for forward pass

        Returns:
            List of dicts, one per problem, each containing:
                'losses': Dict[str, float] - average loss per heuristic
                'neutral_loss': float - neutral baseline loss (if include_neutral)
                'delta_losses': Dict[str, float] - loss relative to neutral (if include_neutral)
                'per_template_losses': Dict[str, Dict] - per-template prompts and losses
                'best_heuristic': str - heuristic with lowest average loss
                'confidence': float - detection confidence (0-1)
        """
        if not problems:
            return []

        all_results: List[Dict[str, Any]] = [{} for _ in range(len(problems))]
        client = training_client if training_client is not None else self.get_training_client()

        def _empty_result() -> Dict[str, Any]:
            return {
                "losses": {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
                "neutral_loss": float('inf') if include_neutral else None,
                "delta_losses": None,
                "per_template_losses": {},
                "best_heuristic": "UNKNOWN",
                "confidence": 0.0
            }

        template_count = sum(len(t) for t in get_multi_heuristic_templates(0, 0).values())
        templates_per_problem = template_count + (1 if include_neutral else 0)
        problems_per_batch = max(1, batch_size // templates_per_problem)

        for batch_start in range(0, len(problems), problems_per_batch):
            batch_problems = problems[batch_start:batch_start + problems_per_batch]

            # Prepare datums for all problems × all templates
            datums: List[Any] = []
            metadata: List[Tuple[int, str, int, Optional[List[int]], str]] = []
            batch_results: List[Dict[str, Any]] = []

            for local_idx, (a, b) in enumerate(batch_problems):
                problem_prompt = get_problem_prompt(a, b)
                multi_templates = get_multi_heuristic_templates(a, b)
                batch_results.append({
                    "losses": {},
                    "neutral_loss": None,
                    "delta_losses": None,
                    "per_template_losses": {},
                    "_heuristic_losses": {h: [] for h in multi_templates.keys()}
                })

                for h_name, templates in multi_templates.items():
                    for t_idx, continuation in enumerate(templates):
                        datum, weights = self._build_text_response_datum(problem_prompt, continuation)
                        if datum is None:
                            metadata.append((local_idx, h_name, t_idx, None, continuation))
                            continue

                        datums.append(datum)
                        metadata.append((local_idx, h_name, t_idx, weights, continuation))

                if include_neutral:
                    neutral_prompt = get_neutral_baseline_template(a, b)
                    datum, weights = self._build_text_response_datum(problem_prompt, neutral_prompt)
                    if datum is None:
                        metadata.append((local_idx, "NEUTRAL", 0, None, neutral_prompt))
                    else:
                        datums.append(datum)
                        metadata.append((local_idx, "NEUTRAL", 0, weights, neutral_prompt))

            if not datums:
                for local_idx in range(len(batch_results)):
                    all_results[batch_start + local_idx] = _empty_result()
                continue

            try:
                future = client.forward(datums, "cross_entropy")
                result = _call_with_timeout(future, operation="batch forward pass")

                datum_idx = 0
                for local_idx, h_name, t_idx, weights, continuation in metadata:
                    template_id = "NEUTRAL" if h_name == "NEUTRAL" else f"{h_name}_{t_idx}"
                    res = batch_results[local_idx]
                    if weights is None:
                        res["per_template_losses"][template_id] = {"prompt": continuation, "loss": float('inf')}
                        if h_name == "NEUTRAL":
                            res["neutral_loss"] = float('inf')
                        else:
                            res["_heuristic_losses"][h_name].append(float('inf'))
                        continue

                    if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                        logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                        loss = compute_weighted_loss(logprobs, weights)
                        res["per_template_losses"][template_id] = {"prompt": continuation, "loss": loss}
                        if h_name == "NEUTRAL":
                            res["neutral_loss"] = loss
                        else:
                            res["_heuristic_losses"][h_name].append(loss)
                        datum_idx += 1
                    else:
                        res["per_template_losses"][template_id] = {"prompt": continuation, "loss": float('inf')}
                        if h_name == "NEUTRAL":
                            res["neutral_loss"] = float('inf')
                        else:
                            res["_heuristic_losses"][h_name].append(float('inf'))

            except TimeoutError as e:
                if self.verbose:
                    tprint(f"    Warning: Batched multi-problem computation timed out: {e}")
                for res in batch_results:
                    res.clear()
                    res.update(_empty_result())
            except Exception as e:
                if self.verbose:
                    tprint(f"    Warning: Batched multi-problem computation failed: {e}")
                for res in batch_results:
                    res.clear()
                    res.update(_empty_result())

            for local_idx, res in enumerate(batch_results):
                heuristic_losses = res.pop("_heuristic_losses", None)
                if heuristic_losses is not None:
                    avg_losses: Dict[str, float] = {}
                    for h, losses in heuristic_losses.items():
                        valid = [l for l in losses if l < float('inf')]
                        avg_losses[h] = sum(valid) / len(valid) if valid else float('inf')
                    res["losses"] = avg_losses

                neutral = res.get("neutral_loss")
                if include_neutral and neutral is not None and neutral < float('inf'):
                    res["delta_losses"] = {h: loss - neutral for h, loss in res["losses"].items()}
                else:
                    res["delta_losses"] = None

                valid_loss_map = {h: l for h, l in res["losses"].items() if l < float('inf')}
                if valid_loss_map:
                    best_heuristic = min(valid_loss_map, key=lambda h: valid_loss_map[h])
                    sorted_vals = sorted(valid_loss_map.values())
                    if len(sorted_vals) >= 2 and sorted_vals[1] > 0:
                        gap = (sorted_vals[1] - sorted_vals[0]) / sorted_vals[1]
                        confidence = min(1.0, gap * 2)
                    else:
                        confidence = 0.5
                else:
                    best_heuristic = "UNKNOWN"
                    confidence = 0.0
                res["best_heuristic"] = best_heuristic
                res["confidence"] = confidence

                all_results[batch_start + local_idx] = res

        return all_results

    def generate(
        self,
        a: int,
        b: int,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None,
        adapter_name: str = "base"
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Generate model's answer for a multiplication problem.

        Args:
            a: First operand
            b: Second operand
            with_reasoning: Include step-by-step reasoning
            max_tokens: Max tokens to generate (auto-set based on reasoning)
            adapter_name: Adapter checkpoint name

        Returns:
            (answer, text) tuple - answer as int or None, text is model output
        """
        try:
            sampler = self.get_sampling_client(adapter_name)

            if with_reasoning:
                prompt_text = f"What is {a} × {b}? Show your work step by step, then give the final answer."
                max_tokens = max_tokens or 2048
            else:
                prompt_text = f"What is {a} × {b}? Answer with just the number."
                max_tokens = max_tokens or 2048

            model_input = self.build_text_generation_input(prompt_text)
            if model_input is None:
                return None, None

            # Sampling params
            sampling_params = self._tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic
            )

            # Generate
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
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'token_ids'):
                    output_tokens = sample.token_ids
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'text'):
                    text = sample.text
                else:
                    text = str(sample)
            elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                output_tokens = result.sequences[0].tokens
                text = self._tokenizer.decode(output_tokens)
            elif hasattr(result, 'completions'):
                text = result.completions[0]
            elif isinstance(result, list):
                text = str(result[0])
            else:
                text = str(result)

            # Parse answer
            answer = extract_answer_enhanced(text, a=a, b=b).answer
            trace = text

            return answer, trace

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Generation failed: {e}")
            return None, None

    async def generate_async(
        self,
        a: int,
        b: int,
        sampler,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Async generation for a multiplication problem using sample_async().

        Args:
            a: First operand
            b: Second operand
            sampler: Pre-created sampling client (shared across async calls)
            with_reasoning: Include step-by-step reasoning
            max_tokens: Max tokens to generate

        Returns:
            (answer, text) tuple - answer as int or None, text is model output
        """
        try:
            if with_reasoning:
                prompt_text = f"What is {a} × {b}? Show your work step by step, then give the final answer."
                max_tokens = max_tokens or 2048
            else:
                prompt_text = f"What is {a} × {b}? Answer with just the number."
                max_tokens = max_tokens or 2048

            model_input = self.build_text_generation_input(prompt_text)
            if model_input is None:
                return None, None

            # Sampling params
            sampling_params = self._tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic
            )

            # Generate using async API
            result = await sampler.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            )

            # Extract text
            if hasattr(result, 'samples') and len(result.samples) > 0:
                sample = result.samples[0]
                if hasattr(sample, 'tokens'):
                    output_tokens = sample.tokens
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'token_ids'):
                    output_tokens = sample.token_ids
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'text'):
                    text = sample.text
                else:
                    text = str(sample)
            elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                output_tokens = result.sequences[0].tokens
                text = self._tokenizer.decode(output_tokens)
            elif hasattr(result, 'completions'):
                text = result.completions[0]
            elif isinstance(result, list):
                text = str(result[0])
            else:
                text = str(result)

            # Parse answer
            answer = extract_answer_enhanced(text, a=a, b=b).answer
            return answer, text

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Async generation failed: {e}")
            return None, None

    def forward_backward(
        self,
        text: str,
        apply_gradients: bool = True,
        response: Optional[str] = None,
        use_chat_format: bool = True
    ) -> Tuple[float, object]:
        """
        Run forward-backward pass on text.

        Args:
            text: Prompt text (or raw text if use_chat_format is False)
            apply_gradients: Whether this is for training (True) or just loss (False)
            response: Optional assistant response for prompt/response training
            use_chat_format: Whether to wrap prompt in chat template

        Returns:
            (loss, result) tuple
        """
        client = self.get_training_client()
        datum = None
        weights = None

        if response is not None:
            datum, weights = _build_chat_prompt_response_datum(
                self._tinker,
                self._tokenizer,
                text,
                response
            )
        elif use_chat_format:
            datum, weights = self._build_text_datum(text)
        else:
            tokens = self._tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < 2:
                return float('inf'), None

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            weights = [1] * len(target_tokens)

            model_input = self._tinker.types.ModelInput.from_ints(tokens=input_tokens)
            datum = self._tinker.types.Datum(
                model_input=model_input,
                loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
            )

        if datum is None or weights is None:
            return float('inf'), None

        if apply_gradients:
            future = client.forward_backward([datum], "cross_entropy")
        else:
            future = client.forward([datum], "cross_entropy")
        result = future.result()

        loss = float('inf')
        if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > 0:
            logprobs_raw = result.loss_fn_outputs[0]['logprobs']
            loss = compute_weighted_loss(logprobs_raw, weights)

        return loss, result

    def optim_step(self, learning_rate: float = 1e-4):
        """
        Apply optimizer step after forward_backward.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        client = self.get_training_client()
        adam_params = self._tinker.types.AdamParams(learning_rate=learning_rate)
        future = client.optim_step(adam_params)
        future.result()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def create_client(
    model_name: str = DEFAULT_MODEL_NAME,
    verbose: bool = True
) -> TinkerClient:
    """
    Create a TinkerClient with common defaults.

    Args:
        model_name: HuggingFace model name
        verbose: Print initialization messages

    Returns:
        Initialized TinkerClient
    """
    return TinkerClient(model_name=model_name, verbose=verbose)


# =============================================================================
# IMAGE HEURISTIC TEMPLATES
# =============================================================================
def get_image_heuristic_templates(a: int, b: int) -> Dict[str, List[str]]:
    """
    Generate text continuation templates for image-based perplexity probes.

    These templates are shown AFTER an image of "a × b = ?" and we measure
    how "natural" each continuation feels to the model.

    Args:
        a: First operand (shown in image)
        b: Second operand (shown in image)

    Returns:
        Dict mapping heuristic name to continuation templates
    """
    return get_multi_heuristic_templates(a, b)


def get_image_neutral_baseline_template(a: int, b: int) -> str:
    """
    Generate neutral baseline template for image-based perplexity probes.

    This template provides a neutral starting point to compute delta losses
    for image modality, matching the text modality neutral baseline.

    Args:
        a: First operand (shown in image)
        b: Second operand (shown in image)

    Returns:
        Neutral continuation text
    """
    return get_neutral_baseline_template(a, b)


def get_image_contrastive_step_templates(a: int, b: int) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate contrastive step templates for image-based probing.

    These are identical to text templates, used as continuations after the image.
    """
    return get_contrastive_step_templates(a, b)


# =============================================================================
# VISION TINKER CLIENT
# =============================================================================
class VisionTinkerClient:
    """
    Vision-enabled Tinker API client for multimodal fingerprinting.

    Extends TinkerClient functionality to support image inputs via
    ImageChunk and multimodal ModelInput construction.

    Example:
        client = VisionTinkerClient()

        # Compute perplexity with image context
        loss = client.compute_perplexity_with_image(
            image_path="/path/to/image.png",
            text_continuation="First, multiply 7 by 6"
        )

        # Get heuristic perplexities for an image problem
        losses = client.compute_heuristic_losses_with_image(
            image_path="/path/to/image.png",
            a=47, b=36
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VISION_MODEL,
        lora_rank: int = DEFAULT_LORA_RANK,
        api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Vision Tinker client.

        Args:
            model_name: HuggingFace vision model name
            lora_rank: LoRA rank for training clients
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
            verbose: Print initialization messages
        """
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TINKER_API_KEY not set. "
                "Set via environment variable or pass api_key parameter."
            )

        self.model_name = model_name
        self.lora_rank = lora_rank
        self.verbose = verbose

        # Lazy-loaded components
        self._tinker: Any = None
        self._service_client: Any = None
        self._tokenizer: Any = None
        self._training_client: Any = None
        self._sampling_client: Any = None
        self._sampling_clients: Dict[str, Any] = {}
        self._hf_token: Optional[str] = None
        self._image_processor: Any = None
        self._renderer: Any = None
        self._image_token_cache: Dict[str, int] = {}
        self._image_dimension_token_cache: Dict[Tuple[int, int], int] = {}
        self._training_client_lock: Any = None  # asyncio.Lock, lazily created
        self._startup_config = load_tinker_startup_config()

        # Initialize
        _run_startup_phase(
            phase_name="init_api",
            model_name=self.model_name,
            startup_config=self._startup_config,
            verbose=self.verbose,
            action=self._init_api,
        )
        _run_startup_phase(
            phase_name="init_tokenizer",
            model_name=self.model_name,
            startup_config=self._startup_config,
            verbose=self.verbose,
            action=self._init_tokenizer,
        )
        _run_startup_phase(
            phase_name="init_renderer",
            model_name=self.model_name,
            startup_config=self._startup_config,
            verbose=self.verbose,
            action=self._init_renderer,
        )

    def _init_api(self):
        """Initialize Tinker API."""
        import tinker

        self._hf_token = os.getenv("HF_TOKEN")
        if self._hf_token and self.verbose:
            tprint("  HF_TOKEN detected (auto-authenticated)")

        self._tinker = tinker
        self._service_client = create_tinker_service_client(
            tinker_module=tinker,
            api_key=self.api_key,
            config=self._startup_config,
        )

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        from transformers import AutoTokenizer

        if self.verbose:
            tprint(f"  Loading tokenizer for {self.model_name}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self._hf_token
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _init_renderer(self):
        """Initialize Qwen3-VL renderer and image processor for proper multimodal input."""
        try:
            from tinker_cookbook import renderers
            from tinker_cookbook.image_processing_utils import get_image_processor

            if self.verbose:
                tprint(f"  Loading image processor for {self.model_name}...")

            self._image_processor = get_image_processor(self.model_name)
            self._renderer = renderers.Qwen3VLInstructRenderer(
                self._tokenizer, self._image_processor
            )

            if self.verbose:
                tprint("  Renderer initialized successfully")

        except ImportError as e:
            if self.verbose:
                tprint(f"  Warning: Could not import tinker_cookbook: {e}")
                tprint("  Falling back to manual multimodal input construction")
            self._renderer = None
            self._image_processor = None
        except Exception as e:
            if self.verbose:
                tprint(f"  Warning: Failed to initialize renderer: {e}")
                tprint("  Falling back to manual multimodal input construction")
            self._renderer = None
            self._image_processor = None

    @property
    def tinker(self):
        """Access to tinker module."""
        return self._tinker

    @property
    def tokenizer(self):
        """Access to tokenizer."""
        return self._tokenizer

    @property
    def service_client(self):
        """Access to Tinker service client."""
        return self._service_client

    def _build_text_datum(self, prompt: str) -> Tuple[Optional[Any], Optional[List[int]]]:
        """Build chat-formatted Datum for text-only loss computation."""
        return _build_chat_text_datum(self._tinker, self._tokenizer, prompt)

    def _build_text_response_datum(
        self,
        prompt: str,
        response: str
    ) -> Tuple[Optional[Any], Optional[List[int]]]:
        """Build chat-formatted Datum scoring only assistant continuation tokens."""
        return _build_chat_prompt_response_datum(self._tinker, self._tokenizer, prompt, response)

    def build_text_generation_input(self, prompt: str) -> Optional[Any]:
        """Build chat-formatted ModelInput for text generation."""
        model_input = None
        if self._renderer is not None:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                model_input = self._renderer.build_generation_prompt(messages)
            except Exception as e:
                if self.verbose:
                    tprint(f"  Warning: Renderer text prompt failed: {e}")
                model_input = None

        if model_input is None:
            model_input = _build_chat_text_model_input(self._tinker, self._tokenizer, prompt)

        return model_input

    def get_training_client(self, force_new: bool = False):
        """Get or create a LoRA training client."""
        if self._training_client is None or force_new:
            if self.verbose:
                tprint("  Initializing Vision LoRA training client...")
            self._training_client = self._service_client.create_lora_training_client(
                base_model=self.model_name,
                rank=self.lora_rank
            )
        return self._training_client

    async def get_training_client_async(self, force_new: bool = False):
        """
        Get or create a LoRA training client (async version for use in async context).

        Uses create_lora_training_client_async() to avoid sync/async mixing warnings.
        Protected by asyncio.Lock to prevent race conditions.
        """
        import asyncio
        # Lazily create the lock if not yet created
        if self._training_client_lock is None:
            self._training_client_lock = asyncio.Lock()

        async with self._training_client_lock:
            if self._training_client is None or force_new:
                if self.verbose:
                    tprint("  Initializing Vision LoRA training client (async)...")
                self._training_client = await self._service_client.create_lora_training_client_async(
                    base_model=self.model_name,
                    rank=self.lora_rank
                )
            return self._training_client

    def get_sampling_client(
        self,
        adapter_name: str = "base",
        force_new: bool = False,
        adapter_path: Optional[str] = None
    ):
        """
        Get or create a sampling client.

        Args:
            adapter_name: Name for the adapter checkpoint
            force_new: Force creation of new client
            adapter_path: Optional LoRA checkpoint path (overrides adapter_name)

        Returns:
            Tinker sampling client
        """
        cache_key = f"path:{adapter_path}" if adapter_path else f"name:{adapter_name}"
        if force_new or cache_key not in self._sampling_clients:
            if adapter_path:
                if self.verbose:
                    tprint(f"  Initializing Vision sampling client from checkpoint: {adapter_path}")
                sampling_client = self._service_client.create_sampling_client(
                    model_path=adapter_path,
                    retry_config=build_tinker_sampling_retry_config(),
                )
            else:
                if self.verbose:
                    tprint(f"  Initializing Vision sampling client ({adapter_name})...")
                training = self.get_training_client()
                sampling_client = training.save_weights_and_get_sampling_client(
                    name=adapter_name,
                    retry_config=build_tinker_sampling_retry_config(),
                )
            self._sampling_clients[cache_key] = sampling_client

        self._sampling_client = self._sampling_clients[cache_key]
        return self._sampling_client

    def _resolve_image_generation_prompt(
        self,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None,
        prompt_text: Optional[str] = None
    ) -> Tuple[str, int]:
        """Resolve prompt text and max tokens for image generation."""
        if prompt_text is None:
            if with_reasoning:
                prompt_text = (
                    "Look at this multiplication problem. Show your work step by step, "
                    "then give the final answer."
                )
                max_tokens = max_tokens or 2048
            else:
                prompt_text = (
                    "What is the answer to this multiplication problem? "
                    "Answer with just the number."
                )
                max_tokens = max_tokens or 2048
        else:
            max_tokens = max_tokens or 2048
        return prompt_text, max_tokens

    def _build_image_generation_input(
        self,
        image_path: Path,
        prompt_text: str,
        a: int,
        b: int
    ) -> Any:
        """Build multimodal input for image generation."""
        if self._renderer is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            model_input = self._renderer.build_generation_prompt(messages)

            if self.verbose:
                tprint(f"    Using renderer for image: {image_path}")
                tprint(f"    Expected operands in output: {a} × {b} = {a * b}")
                log_detail(
                    "image_generation.log",
                    f"DEBUG: renderer_mode=True, image_path={image_path}, expected={a}×{b}={a*b}",
                )
            return model_input

        image_bytes = self._load_image_bytes(image_path)
        image_format = self._get_image_format(image_path)
        model_input = self._build_multimodal_input_for_sampling(
            image_bytes, image_format, prompt_text
        )

        if self.verbose:
            tprint(
                f"    Image input: {len(image_bytes)} bytes, format={image_format}, path={image_path}"
            )
            tprint("    Using vision tokens: <|vision_start|>...<|vision_end|>")
            tprint(f"    Expected operands in output: {a} × {b} = {a * b}")
            log_detail(
                "image_generation.log",
                "DEBUG: vision_tokens=True, "
                f"image_bytes={len(image_bytes)}, format={image_format}, "
                f"model_input_type={type(model_input)}, expected={a}×{b}={a*b}",
            )
        return model_input

    def _extract_text_from_sampling_result(self, result: Any) -> str:
        """Decode sampling output into text across Tinker result variants."""
        if hasattr(result, "samples") and len(result.samples) > 0:
            sample = result.samples[0]
            if hasattr(sample, "tokens"):
                return self._tokenizer.decode(sample.tokens)
            if hasattr(sample, "token_ids"):
                return self._tokenizer.decode(sample.token_ids)
            if hasattr(sample, "text"):
                return sample.text
            return str(sample)
        if hasattr(result, "sequences") and len(result.sequences) > 0:
            return self._tokenizer.decode(result.sequences[0].tokens)
        if hasattr(result, "completions"):
            return result.completions[0]
        if isinstance(result, list):
            return str(result[0])
        return str(result)

    # =========================================================================
    # TEXT-ONLY METHODS (for unified text/image experiments on VLM)
    # =========================================================================

    def compute_perplexity(self, prompt: str) -> float:
        """
        Compute perplexity/loss for a text-only prompt using forward pass.

        This allows the VLM to be used for text-only inputs, enabling
        apples-to-apples comparison with image inputs.

        Args:
            prompt: Text to evaluate

        Returns:
            Average loss (lower = more likely under model)
        """
        try:
            client = self.get_training_client()
            datum, weights = self._build_text_datum(prompt)
            if datum is None or weights is None:
                return float('inf')

            # Forward pass (no gradients) with timeout
            future = client.forward([datum], "cross_entropy")
            result = _call_with_timeout(future, operation="perplexity forward pass")

            # Extract loss
            if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > 0:
                logprobs_raw = result.loss_fn_outputs[0]['logprobs']
                return compute_weighted_loss(logprobs_raw, weights)
            elif hasattr(result, 'loss'):
                return float(result.loss)
            else:
                return float('inf')

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Text perplexity computation timed out: {e}")
            return float('inf')
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Text perplexity computation failed: {e}")
            return float('inf')

    def compute_heuristic_losses(self, a: int, b: int) -> Dict[str, float]:
        """
        Compute perplexity losses for all heuristic templates (text-only).

        Loss is measured on assistant continuation tokens given the problem prompt.

        Batches all 3 templates (OT, DD, RC) into a single forward call.

        Args:
            a: First operand
            b: Second operand

        Returns:
            Dict mapping heuristic name to loss value
        """
        templates = get_heuristic_templates(a, b)
        prompt = get_problem_prompt(a, b)

        try:
            client = self.get_training_client()

            # Prepare all 3 datums for batched call
            datums: List[Any] = []
            heuristic_names: List[str] = []
            weights_list: List[Optional[List[int]]] = []

            for h_name, continuation in templates.items():
                datum, weights = self._build_text_response_datum(prompt, continuation)
                if datum is None:
                    heuristic_names.append(h_name)
                    weights_list.append(None)
                    continue

                datums.append(datum)
                heuristic_names.append(h_name)
                weights_list.append(weights)

            if not datums:
                return {h: float('inf') for h in templates.keys()}

            # Single batched API call for all 3 templates with timeout
            future = client.forward(datums, "cross_entropy")
            result = _call_with_timeout(future, operation="text heuristic forward pass")

            # Extract losses from batch result
            losses = {}
            datum_idx = 0
            for i, h_name in enumerate(heuristic_names):
                weights = weights_list[i]
                if weights is None:
                    losses[h_name] = float('inf')
                elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    losses[h_name] = compute_weighted_loss(logprobs, weights)
                    datum_idx += 1
                else:
                    # API returned fewer outputs than expected - don't increment datum_idx
                    losses[h_name] = float('inf')

            return losses

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Text heuristic computation timed out: {e}")
            return {h: float('inf') for h in templates.keys()}
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Text heuristic computation failed: {e}")
            return {h: float('inf') for h in templates.keys()}

    def detect_heuristic(self, a: int, b: int) -> Tuple[str, float]:
        """
        Detect preferred heuristic from text-only perplexity probe.

        Args:
            a: First operand
            b: Second operand

        Returns:
            (detected_heuristic, confidence) tuple
        """
        losses = self.compute_heuristic_losses(a, b)

        if not losses or not all(v < float('inf') for v in losses.values()):
            return "UNKNOWN", 0.0

        # Find lowest loss = preferred heuristic
        best = min(losses, key=lambda h: losses[h])

        # Compute confidence from loss gap
        sorted_losses = sorted(losses.values())
        if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
            gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
            confidence = min(1.0, gap * 2)
        else:
            confidence = 0.5

        return best, confidence

    def compute_heuristic_losses_with_baseline(self, a: int, b: int) -> Dict:
        """
        Compute perplexity losses using multi-template averaging and neutral baseline.

        Loss is measured on assistant continuation tokens given the problem prompt.

        This implements the full PerplexityProbe methodology:
        1. Averages losses across templates per heuristic (count depends on template mode)
        2. Computes neutral baseline loss
        3. Returns Δloss (loss - baseline) for each heuristic

        Args:
            a: First operand
            b: Second operand

        Returns:
            Dict with keys:
                'losses': Dict[str, float] - Average raw loss per heuristic
                'neutral_loss': float - Baseline loss
                'delta_losses': Dict[str, float] - Loss relative to baseline
                'per_template_losses': Dict[str, Dict] - Per-template prompts and losses
                    Keys like 'OT_0', 'DD_1', 'NEUTRAL', values are {'prompt': str, 'loss': float}
                'best_heuristic': str - Heuristic with lowest loss
                'confidence': float - Detection confidence (0-1)
        """
        multi_templates = get_multi_heuristic_templates(a, b)
        neutral_template = get_neutral_baseline_template(a, b)
        prompt = get_problem_prompt(a, b)

        try:
            client = self.get_training_client()

            # Prepare all datums: templates per heuristic + neutral baseline
            datums: List[Any] = []
            metadata: List[Tuple[str, int, Optional[List[int]], str]] = []  # (heuristic_name, template_idx, weights, continuation)

            # Add heuristic templates
            for h_name, templates in multi_templates.items():
                for t_idx, continuation in enumerate(templates):
                    datum, weights = self._build_text_response_datum(prompt, continuation)
                    if datum is None:
                        metadata.append((h_name, t_idx, None, continuation))
                        continue

                    datums.append(datum)
                    metadata.append((h_name, t_idx, weights, continuation))

            # Add neutral baseline
            neutral_datum, neutral_weights = self._build_text_response_datum(prompt, neutral_template)
            if neutral_datum is not None:
                datums.append(neutral_datum)
                metadata.append(("NEUTRAL", 0, neutral_weights, neutral_template))
            else:
                metadata.append(("NEUTRAL", 0, None, neutral_template))

            if not datums:
                return self._empty_baseline_result()

            # Single batched API call for all templates with timeout
            future = client.forward(datums, "cross_entropy")
            result = _call_with_timeout(future, operation="multi-template forward pass")

            # Extract losses and group by heuristic
            heuristic_losses: Dict[str, List[float]] = {h: [] for h in multi_templates.keys()}
            per_template_losses: Dict[str, Dict[str, Any]] = {}  # template_id -> {"prompt": ..., "loss": ...}
            neutral_loss = float('inf')
            datum_idx = 0

            for h_name, t_idx, weights, continuation in metadata:
                template_id = f"{h_name}_{t_idx}" if h_name != "NEUTRAL" else "NEUTRAL"

                if weights is None:
                    per_template_losses[template_id] = {"prompt": continuation, "loss": float('inf')}
                    if h_name != "NEUTRAL":
                        heuristic_losses[h_name].append(float('inf'))
                    continue

                if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                    per_template_losses[template_id] = {"prompt": continuation, "loss": loss}
                    if h_name == "NEUTRAL":
                        neutral_loss = loss
                    else:
                        heuristic_losses[h_name].append(loss)
                    datum_idx += 1
                else:
                    # API returned fewer outputs than expected - don't increment datum_idx
                    per_template_losses[template_id] = {"prompt": continuation, "loss": float('inf')}
                    if h_name != "NEUTRAL":
                        heuristic_losses[h_name].append(float('inf'))

            # Compute average loss per heuristic
            avg_losses: Dict[str, float] = {}
            for h, losses in heuristic_losses.items():
                valid_loss_values = [l for l in losses if l < float('inf')]
                avg_losses[h] = sum(valid_loss_values) / len(valid_loss_values) if valid_loss_values else float('inf')

            # Compute delta losses relative to neutral baseline
            delta_losses = {h: loss - neutral_loss for h, loss in avg_losses.items()}

            # Find best heuristic and compute confidence
            valid_loss_map = {h: l for h, l in avg_losses.items() if l < float('inf')}
            if valid_loss_map:
                best_heuristic = min(valid_loss_map, key=lambda h: valid_loss_map[h])
                sorted_vals = sorted(valid_loss_map.values())
                if len(sorted_vals) >= 2 and sorted_vals[1] > 0:
                    gap = (sorted_vals[1] - sorted_vals[0]) / sorted_vals[1]
                    confidence = min(1.0, gap * 2)
                else:
                    confidence = 0.5
            else:
                best_heuristic = "UNKNOWN"
                confidence = 0.0

            return {
                'losses': avg_losses,
                'neutral_loss': neutral_loss,
                'delta_losses': delta_losses,
                'per_template_losses': per_template_losses,
                'best_heuristic': best_heuristic,
                'confidence': confidence
            }

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Multi-template perplexity computation timed out: {e}")
            return self._empty_baseline_result()
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Multi-template perplexity computation failed: {e}")
            return self._empty_baseline_result()

    def compute_contrastive_step_losses(self, a: int, b: int) -> Dict[str, Any]:
        """
        Compute contrastive losses for correct vs incorrect heuristic steps.

        Loss is measured on assistant continuation tokens given the problem prompt.
        Returns averaged losses per heuristic across contrastive variants.
        """
        templates = get_contrastive_step_templates(a, b)
        prompt = get_problem_prompt(a, b)

        try:
            client = self.get_training_client()

            datums: List[Any] = []
            metadata: List[Tuple[str, int, str, Optional[List[int]], str]] = []

            for h_name, pairs in templates.items():
                for t_idx, (correct, incorrect) in enumerate(pairs):
                    for label, continuation in (("correct", correct), ("incorrect", incorrect)):
                        datum, weights = self._build_text_response_datum(prompt, continuation)
                        if datum is None:
                            metadata.append((h_name, t_idx, label, None, continuation))
                            continue
                        datums.append(datum)
                        metadata.append((h_name, t_idx, label, weights, continuation))

            if not datums:
                return self._empty_contrastive_result()

            future = client.forward(datums, "cross_entropy")
            result = _call_with_timeout(future, operation="contrastive step forward pass")

            correct_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            incorrect_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            per_template_losses: Dict[str, Dict[str, Any]] = {}

            datum_idx = 0
            for h_name, t_idx, label, weights, continuation in metadata:
                template_id = f"{h_name}_{t_idx}_{label.upper()}"
                if weights is None:
                    loss = float('inf')
                elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                    datum_idx += 1
                else:
                    loss = float('inf')

                per_template_losses[template_id] = {"prompt": continuation, "loss": loss}
                if label == "correct":
                    correct_losses[h_name].append(loss)
                else:
                    incorrect_losses[h_name].append(loss)

            avg_correct = {}
            avg_incorrect = {}
            delta_losses = {}
            for h in templates.keys():
                correct_vals = [l for l in correct_losses[h] if l < float('inf')]
                incorrect_vals = [l for l in incorrect_losses[h] if l < float('inf')]
                avg_correct[h] = sum(correct_vals) / len(correct_vals) if correct_vals else float('inf')
                avg_incorrect[h] = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else float('inf')
                delta_losses[h] = avg_incorrect[h] - avg_correct[h]

            return {
                "correct_losses": avg_correct,
                "incorrect_losses": avg_incorrect,
                "delta_losses": delta_losses,
                "per_template_losses": per_template_losses
            }

        except TimeoutError as e:
            if self.verbose:
                tprint(f"    Warning: Contrastive step computation timed out: {e}")
            return self._empty_contrastive_result()
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Contrastive step computation failed: {e}")
            return self._empty_contrastive_result()

    def _empty_baseline_result(self) -> Dict:
        """Return empty result structure for compute_heuristic_losses_with_baseline."""
        return {
            'losses': {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
            'neutral_loss': float('inf'),
            'delta_losses': {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
            'per_template_losses': {},
            'best_heuristic': "UNKNOWN",
            'confidence': 0.0
        }

    def _empty_contrastive_result(self) -> Dict[str, Any]:
        """Return empty result structure for contrastive step probing."""
        losses = {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')}
        return {
            "correct_losses": losses.copy(),
            "incorrect_losses": losses.copy(),
            "delta_losses": losses.copy(),
            "per_template_losses": {}
        }

    def compute_heuristic_losses_multi(
        self,
        problems: List[Tuple[int, int]],
        batch_size: int = 30,
        include_neutral: bool = True,
        training_client=None
    ) -> List[Dict[str, Any]]:
        """
        Compute heuristic losses for multiple problems in batched API calls.

        Batches heuristic probe templates per problem (count depends on template mode)
        into larger API calls for speed while matching the single-problem probe design.

        Args:
            problems: List of (a, b) tuples
            batch_size: Max datums per API call (default 30 = ~2 problems × 12 templates)
            include_neutral: Include neutral baseline template for delta loss computation
            training_client: Optional training client (e.g., LoRA-loaded) to use for forward pass

        Returns:
            List of dicts, one per problem, each containing:
                'losses': Dict[str, float] - average loss per heuristic
                'neutral_loss': float - neutral baseline loss (if include_neutral)
                'delta_losses': Dict[str, float] - loss relative to neutral (if include_neutral)
                'per_template_losses': Dict[str, Dict] - per-template prompts and losses
                'best_heuristic': str - heuristic with lowest average loss
                'confidence': float - detection confidence (0-1)
        """
        if not problems:
            return []

        all_results: List[Dict[str, Any]] = [{} for _ in range(len(problems))]
        client = training_client if training_client is not None else self.get_training_client()

        def _empty_result() -> Dict[str, Any]:
            return {
                "losses": {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
                "neutral_loss": float('inf') if include_neutral else None,
                "delta_losses": None,
                "per_template_losses": {},
                "best_heuristic": "UNKNOWN",
                "confidence": 0.0
            }

        template_count = sum(len(t) for t in get_multi_heuristic_templates(0, 0).values())
        templates_per_problem = template_count + (1 if include_neutral else 0)
        problems_per_batch = max(1, batch_size // templates_per_problem)

        for batch_start in range(0, len(problems), problems_per_batch):
            batch_problems = problems[batch_start:batch_start + problems_per_batch]

            # Prepare datums for all problems × all templates
            datums: List[Any] = []
            metadata: List[Tuple[int, str, int, Optional[List[int]], str]] = []
            batch_results: List[Dict[str, Any]] = []

            for local_idx, (a, b) in enumerate(batch_problems):
                problem_prompt = get_problem_prompt(a, b)
                multi_templates = get_multi_heuristic_templates(a, b)
                batch_results.append({
                    "losses": {},
                    "neutral_loss": None,
                    "delta_losses": None,
                    "per_template_losses": {},
                    "_heuristic_losses": {h: [] for h in multi_templates.keys()}
                })

                for h_name, templates in multi_templates.items():
                    for t_idx, continuation in enumerate(templates):
                        datum, weights = self._build_text_response_datum(problem_prompt, continuation)
                        if datum is None:
                            metadata.append((local_idx, h_name, t_idx, None, continuation))
                            continue

                        datums.append(datum)
                        metadata.append((local_idx, h_name, t_idx, weights, continuation))

                if include_neutral:
                    neutral_prompt = get_neutral_baseline_template(a, b)
                    datum, weights = self._build_text_response_datum(problem_prompt, neutral_prompt)
                    if datum is None:
                        metadata.append((local_idx, "NEUTRAL", 0, None, neutral_prompt))
                    else:
                        datums.append(datum)
                        metadata.append((local_idx, "NEUTRAL", 0, weights, neutral_prompt))

            if not datums:
                for local_idx in range(len(batch_results)):
                    all_results[batch_start + local_idx] = _empty_result()
                continue

            try:
                future = client.forward(datums, "cross_entropy")
                result = _call_with_timeout(future, operation="batch forward pass")

                datum_idx = 0
                for local_idx, h_name, t_idx, weights, continuation in metadata:
                    template_id = "NEUTRAL" if h_name == "NEUTRAL" else f"{h_name}_{t_idx}"
                    res = batch_results[local_idx]
                    if weights is None:
                        res["per_template_losses"][template_id] = {"prompt": continuation, "loss": float('inf')}
                        if h_name == "NEUTRAL":
                            res["neutral_loss"] = float('inf')
                        else:
                            res["_heuristic_losses"][h_name].append(float('inf'))
                        continue

                    if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                        logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                        loss = compute_weighted_loss(logprobs, weights)
                        res["per_template_losses"][template_id] = {"prompt": continuation, "loss": loss}
                        if h_name == "NEUTRAL":
                            res["neutral_loss"] = loss
                        else:
                            res["_heuristic_losses"][h_name].append(loss)
                        datum_idx += 1
                    else:
                        res["per_template_losses"][template_id] = {"prompt": continuation, "loss": float('inf')}
                        if h_name == "NEUTRAL":
                            res["neutral_loss"] = float('inf')
                        else:
                            res["_heuristic_losses"][h_name].append(float('inf'))

            except TimeoutError as e:
                if self.verbose:
                    tprint(f"    Warning: Batched multi-problem computation timed out: {e}")
                for res in batch_results:
                    res.clear()
                    res.update(_empty_result())
            except Exception as e:
                if self.verbose:
                    tprint(f"    Warning: Batched multi-problem computation failed: {e}")
                for res in batch_results:
                    res.clear()
                    res.update(_empty_result())

            for local_idx, res in enumerate(batch_results):
                heuristic_losses = res.pop("_heuristic_losses", None)
                if heuristic_losses is not None:
                    avg_losses: Dict[str, float] = {}
                    for h, losses in heuristic_losses.items():
                        valid = [l for l in losses if l < float('inf')]
                        avg_losses[h] = sum(valid) / len(valid) if valid else float('inf')
                    res["losses"] = avg_losses

                neutral = res.get("neutral_loss")
                if include_neutral and neutral is not None and neutral < float('inf'):
                    res["delta_losses"] = {h: loss - neutral for h, loss in res["losses"].items()}
                else:
                    res["delta_losses"] = None

                valid_loss_map = {h: l for h, l in res["losses"].items() if l < float('inf')}
                if valid_loss_map:
                    best_heuristic = min(valid_loss_map, key=lambda h: valid_loss_map[h])
                    sorted_vals = sorted(valid_loss_map.values())
                    if len(sorted_vals) >= 2 and sorted_vals[1] > 0:
                        gap = (sorted_vals[1] - sorted_vals[0]) / sorted_vals[1]
                        confidence = min(1.0, gap * 2)
                    else:
                        confidence = 0.5
                else:
                    best_heuristic = "UNKNOWN"
                    confidence = 0.0
                res["best_heuristic"] = best_heuristic
                res["confidence"] = confidence

                all_results[batch_start + local_idx] = res

        return all_results

    async def compute_heuristic_losses_multi_async(
        self,
        problems: List[Tuple[int, int]],
        batch_size: int = 30,
        include_neutral: bool = True,
        training_client=None,
        max_in_flight: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Async version of compute_heuristic_losses_multi for use in async contexts.

        Uses forward_async() to avoid sync/async mixing warnings.
        Adds pipelined submission with a bounded number of in-flight
        forward_async() calls to avoid idle clock cycles on Tinker.

        Args:
            problems: List of (a, b) tuples
            batch_size: Max datums per API call (default 30 = ~2 problems × 12 templates)
            include_neutral: Include neutral baseline template for delta loss computation
            training_client: Optional training client (e.g., LoRA-loaded) to use for forward pass
            max_in_flight: Max number of forward_async requests in flight

        Returns:
            List of dicts, one per problem, each containing:
                'losses': Dict[str, float] - average loss per heuristic
                'neutral_loss': float - neutral baseline loss (if include_neutral)
                'delta_losses': Dict[str, float] - loss relative to neutral (if include_neutral)
                'per_template_losses': Dict[str, Dict] - per-template prompts and losses
                'best_heuristic': str - heuristic with lowest average loss
                'confidence': float - detection confidence (0-1)
        """
        import asyncio

        if not problems:
            return []

        all_results: List[Dict[str, Any]] = [{} for _ in range(len(problems))]
        # Use async client initialization to avoid sync/async mixing
        client = training_client if training_client is not None else await self.get_training_client_async()
        max_in_flight = max(1, max_in_flight)

        def _empty_result() -> Dict[str, Any]:
            return {
                "losses": {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')},
                "neutral_loss": float('inf') if include_neutral else None,
                "delta_losses": None,
                "per_template_losses": {},
                "best_heuristic": "UNKNOWN",
                "confidence": 0.0
            }

        template_count = sum(len(t) for t in get_multi_heuristic_templates(0, 0).values())
        templates_per_problem = template_count + (1 if include_neutral else 0)
        problems_per_batch = max(1, batch_size // templates_per_problem)

        semaphore = asyncio.Semaphore(max_in_flight)
        pending = set()

        async def submit_batch(batch_start, batch_problems):
            # Prepare datums for all problems × all templates
            datums: List[Any] = []
            metadata: List[Tuple[int, str, int, Optional[List[int]], str]] = []
            batch_results: List[Dict[str, Any]] = []

            for local_idx, (a, b) in enumerate(batch_problems):
                problem_prompt = get_problem_prompt(a, b)
                multi_templates = get_multi_heuristic_templates(a, b)
                batch_results.append({
                    "losses": {},
                    "neutral_loss": None,
                    "delta_losses": None,
                    "per_template_losses": {},
                    "_heuristic_losses": {h: [] for h in multi_templates.keys()}
                })

                for h_name, templates in multi_templates.items():
                    for t_idx, continuation in enumerate(templates):
                        datum, weights = self._build_text_response_datum(problem_prompt, continuation)
                        if datum is None:
                            metadata.append((local_idx, h_name, t_idx, None, continuation))
                            continue

                        datums.append(datum)
                        metadata.append((local_idx, h_name, t_idx, weights, continuation))

                if include_neutral:
                    neutral_prompt = get_neutral_baseline_template(a, b)
                    datum, weights = self._build_text_response_datum(problem_prompt, neutral_prompt)
                    if datum is None:
                        metadata.append((local_idx, "NEUTRAL", 0, None, neutral_prompt))
                    else:
                        datums.append(datum)
                        metadata.append((local_idx, "NEUTRAL", 0, weights, neutral_prompt))

            if not datums:
                return batch_start, [ _empty_result() for _ in range(len(batch_problems)) ]

            async with semaphore:
                try:
                    # Use async forward with a wall-clock timeout and let Tinker handle retries.
                    api_future = await asyncio.wait_for(
                        client.forward_async(datums, "cross_entropy"),
                        timeout=API_CALL_TIMEOUT
                    )
                    # Double-await pattern: await API future instead of blocking .result()
                    result = await asyncio.wait_for(api_future, timeout=API_CALL_TIMEOUT)

                    datum_idx = 0
                    for local_idx, h_name, t_idx, weights, continuation in metadata:
                        template_id = "NEUTRAL" if h_name == "NEUTRAL" else f"{h_name}_{t_idx}"
                        res = batch_results[local_idx]
                        if weights is None:
                            res["per_template_losses"][template_id] = {"prompt": continuation, "loss": float('inf')}
                            if h_name == "NEUTRAL":
                                res["neutral_loss"] = float('inf')
                            else:
                                res["_heuristic_losses"][h_name].append(float('inf'))
                            continue

                        if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                            logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                            loss = compute_weighted_loss(logprobs, weights)
                            res["per_template_losses"][template_id] = {"prompt": continuation, "loss": loss}
                            if h_name == "NEUTRAL":
                                res["neutral_loss"] = loss
                            else:
                                res["_heuristic_losses"][h_name].append(loss)
                            datum_idx += 1
                        else:
                            res["per_template_losses"][template_id] = {"prompt": continuation, "loss": float('inf')}
                            if h_name == "NEUTRAL":
                                res["neutral_loss"] = float('inf')
                            else:
                                res["_heuristic_losses"][h_name].append(float('inf'))

                except (asyncio.TimeoutError, TimeoutError) as e:
                    if self.verbose:
                        tprint(f"    Warning: Async batched computation timed out: {e}")
                    for res in batch_results:
                        res.clear()
                        res.update(_empty_result())
                except Exception as e:
                    if self.verbose:
                        tprint(f"    Warning: Async batched computation failed: {e}")
                    for res in batch_results:
                        res.clear()
                        res.update(_empty_result())

            # Finalize each problem in batch
            finalized: List[Dict[str, Any]] = []
            for res in batch_results:
                heuristic_losses = res.pop("_heuristic_losses", None)
                if heuristic_losses is not None:
                    avg_losses: Dict[str, float] = {}
                    for h, losses in heuristic_losses.items():
                        valid = [l for l in losses if l < float('inf')]
                        avg_losses[h] = sum(valid) / len(valid) if valid else float('inf')
                    res["losses"] = avg_losses

                neutral = res.get("neutral_loss")
                if include_neutral and neutral is not None and neutral < float('inf'):
                    res["delta_losses"] = {h: loss - neutral for h, loss in res["losses"].items()}
                else:
                    res["delta_losses"] = None

                valid_loss_map = {h: l for h, l in res["losses"].items() if l < float('inf')}
                if valid_loss_map:
                    best_heuristic = min(valid_loss_map, key=lambda h: valid_loss_map[h])
                    sorted_vals = sorted(valid_loss_map.values())
                    if len(sorted_vals) >= 2 and sorted_vals[1] > 0:
                        gap = (sorted_vals[1] - sorted_vals[0]) / sorted_vals[1]
                        confidence = min(1.0, gap * 2)
                    else:
                        confidence = 0.5
                else:
                    best_heuristic = "UNKNOWN"
                    confidence = 0.0
                res["best_heuristic"] = best_heuristic
                res["confidence"] = confidence
                finalized.append(res)

            return batch_start, finalized

        async def drain_pending(pending_tasks: set):
            if not pending_tasks:
                return set(), []
            done, remaining = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            results = [task.result() for task in done]
            return set(remaining), results

        for batch_start in range(0, len(problems), problems_per_batch):
            batch_problems = problems[batch_start:batch_start + problems_per_batch]
            pending.add(asyncio.create_task(submit_batch(batch_start, batch_problems)))

            if len(pending) >= max_in_flight:
                pending, finished = await drain_pending(pending)
                for batch_start_finished, batch_results in finished:
                    for local_idx, res in enumerate(batch_results):
                        all_results[batch_start_finished + local_idx] = res

        # Drain remaining batches
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                batch_start_finished, batch_results = task.result()
                for local_idx, res in enumerate(batch_results):
                    all_results[batch_start_finished + local_idx] = res

        return all_results

    def generate(
        self,
        a: int,
        b: int,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None,
        adapter_name: str = "base"
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Generate model's answer for a multiplication problem (text-only).

        Args:
            a: First operand
            b: Second operand
            with_reasoning: Include step-by-step reasoning
            max_tokens: Max tokens to generate (auto-set based on reasoning)
            adapter_name: Adapter checkpoint name

        Returns:
            (answer, trace) tuple - answer as int or None, trace if reasoning
        """
        try:
            sampler = self.get_sampling_client(adapter_name)

            if with_reasoning:
                prompt_text = f"What is {a} × {b}? Show your work step by step, then give the final answer."
                max_tokens = max_tokens or 2048
            else:
                prompt_text = f"What is {a} × {b}? Answer with just the number."
                max_tokens = max_tokens or 2048

            model_input = self.build_text_generation_input(prompt_text)
            if model_input is None:
                return None, None

            # Sampling params
            sampling_params = self._tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic
            )

            # Generate
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
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'token_ids'):
                    output_tokens = sample.token_ids
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'text'):
                    text = sample.text
                else:
                    text = str(sample)
            elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                output_tokens = result.sequences[0].tokens
                text = self._tokenizer.decode(output_tokens)
            elif hasattr(result, 'completions'):
                text = result.completions[0]
            elif isinstance(result, list):
                text = str(result[0])
            else:
                text = str(result)

            # Parse answer
            answer = extract_answer(text)
            trace = text if with_reasoning else None

            return answer, trace

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Text generation failed: {e}")
            return None, None

    async def generate_async(
        self,
        a: int,
        b: int,
        sampler,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Async generation for a multiplication problem using sample_async().

        Args:
            a: First operand
            b: Second operand
            sampler: Pre-created sampling client (shared across async calls)
            with_reasoning: Include step-by-step reasoning
            max_tokens: Max tokens to generate

        Returns:
            (answer, text) tuple - answer as int or None, text is model output
        """
        try:
            if with_reasoning:
                prompt_text = f"What is {a} × {b}? Show your work step by step, then give the final answer."
                max_tokens = max_tokens or 2048
            else:
                prompt_text = f"What is {a} × {b}? Answer with just the number."
                max_tokens = max_tokens or 2048

            model_input = self.build_text_generation_input(prompt_text)
            if model_input is None:
                return None, None

            # Sampling params
            sampling_params = self._tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic
            )

            # Generate using async API
            result = await sampler.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            )

            # Extract text
            if hasattr(result, 'samples') and len(result.samples) > 0:
                sample = result.samples[0]
                if hasattr(sample, 'tokens'):
                    output_tokens = sample.tokens
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'token_ids'):
                    output_tokens = sample.token_ids
                    text = self._tokenizer.decode(output_tokens)
                elif hasattr(sample, 'text'):
                    text = sample.text
                else:
                    text = str(sample)
            elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                output_tokens = result.sequences[0].tokens
                text = self._tokenizer.decode(output_tokens)
            elif hasattr(result, 'completions'):
                text = result.completions[0]
            elif isinstance(result, list):
                text = str(result[0])
            else:
                text = str(result)

            # Parse answer
            answer = extract_answer_enhanced(text, a=a, b=b).answer
            return answer, text

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Async generation failed: {e}")
            return None, None

    def forward_backward(
        self,
        text: str,
        apply_gradients: bool = True,
        response: Optional[str] = None,
        use_chat_format: bool = True
    ) -> Tuple[float, object]:
        """
        Run forward-backward pass on text (for LoRA training).

        Args:
            text: Prompt text (or raw text if use_chat_format is False)
            apply_gradients: Whether this is for training (True) or just loss (False)
            response: Optional assistant response for prompt/response training
            use_chat_format: Whether to wrap prompt in chat template

        Returns:
            (loss, result) tuple
        """
        client = self.get_training_client()
        datum = None
        weights = None

        if response is not None:
            datum, weights = _build_chat_prompt_response_datum(
                self._tinker,
                self._tokenizer,
                text,
                response
            )
        elif use_chat_format:
            datum, weights = self._build_text_datum(text)
        else:
            tokens = self._tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < 2:
                return float('inf'), None

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            weights = [1] * len(target_tokens)

            model_input = self._tinker.types.ModelInput.from_ints(tokens=input_tokens)
            datum = self._tinker.types.Datum(
                model_input=model_input,
                loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
            )

        if datum is None or weights is None:
            return float('inf'), None

        if apply_gradients:
            future = client.forward_backward([datum], "cross_entropy")
        else:
            future = client.forward([datum], "cross_entropy")
        result = future.result()

        loss = float('inf')
        if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > 0:
            logprobs_raw = result.loss_fn_outputs[0]['logprobs']
            loss = compute_weighted_loss(logprobs_raw, weights)

        return loss, result

    def optim_step(self, learning_rate: float = 1e-4):
        """
        Apply optimizer step after forward_backward.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        client = self.get_training_client()
        adam_params = self._tinker.types.AdamParams(learning_rate=learning_rate)
        future = client.optim_step(adam_params)
        future.result()

    # =========================================================================
    # IMAGE+TEXT METHODS (original VisionTinkerClient functionality)
    # =========================================================================

    def _load_image_bytes(self, image_path: Path) -> bytes:
        """Load image file as bytes."""
        with open(image_path, 'rb') as f:
            return f.read()

    def _get_image_format(self, image_path: Path) -> str:
        """Determine image format from file extension."""
        ext = image_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            return 'jpeg'
        elif ext == '.png':
            return 'png'
        else:
            # Default to PNG, let API handle errors
            return 'png'

    def _build_multimodal_input(
        self,
        image_bytes: bytes,
        image_format: str,
        text_tokens: List[int]
    ):
        """
        Build a multimodal ModelInput with image + text (for forward pass/perplexity).

        Deprecated: Use chat-formatted inputs with vision tokens for Qwen3-VL.

        Args:
            image_bytes: Raw image bytes
            image_format: 'png' or 'jpeg'
            text_tokens: Tokenized text to append after image

        Returns:
            ModelInput with image chunk and text tokens
        """
        # Create image chunk
        img_chunk = self._tinker.types.ImageChunk(
            data=image_bytes,
            format=image_format
        )

        # Build model input (immutable pattern - must reassign)
        model_input = self._tinker.types.ModelInput.empty()
        model_input = model_input.append(img_chunk)

        # Append text tokens one by one
        for token in text_tokens:
            model_input = model_input.append_int(token)

        return model_input

    def _build_multimodal_input_for_sampling(
        self,
        image_bytes: bytes,
        image_format: str,
        prompt_text: str
    ):
        """
        Build a multimodal ModelInput with proper chat format for sampling.

        For Qwen3-VL models, the correct format is:
        <|im_start|>user
        <|vision_start|>[IMAGE]<|vision_end|>[TEXT]<|im_end|>
        <|im_start|>assistant

        Args:
            image_bytes: Raw image bytes
            image_format: 'png' or 'jpeg'
            prompt_text: User prompt text (e.g., "What is in this image?")

        Returns:
            ModelInput ready for sampling with VLM
        """
        # Build the chat-formatted prompt with vision tokens
        prefix = "<|im_start|>user\n<|vision_start|>"
        suffix = f"<|vision_end|>{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        # Encode prefix and suffix tokens
        prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

        # Create chunks list for ModelInput
        chunks = [
            self._tinker.types.EncodedTextChunk(tokens=prefix_tokens),
            self._tinker.types.ImageChunk(data=image_bytes, format=image_format),
            self._tinker.types.EncodedTextChunk(tokens=suffix_tokens),
        ]

        # Build ModelInput from chunks
        model_input = self._tinker.types.ModelInput(chunks=chunks)

        return model_input

    def _extract_token_count_from_error(self, error_msg: str) -> Optional[ParsedTokenCount]:
        """Extract a token count and whether it refers to image-only or total tokens."""
        message = str(error_msg)
        image_patterns = (
            r'Expected\s+\d+\s+tokens,\s+got\s+(\d+)\s+from image',
            r'got\s+(\d+)\s+from image',
            r'image[^0-9]+(\d+)\s+tokens',
        )
        for pattern in image_patterns:
            image_match = re.search(pattern, message, flags=re.IGNORECASE)
            if image_match:
                return ParsedTokenCount(count=int(image_match.group(1)), scope="image")

        total_patterns = (
            r'token_count=(\d+)',
            r'expected_tokens=(\d+)',
        )
        for pattern in total_patterns:
            total_match = re.search(pattern, message, flags=re.IGNORECASE)
            if total_match:
                return ParsedTokenCount(count=int(total_match.group(1)), scope="total")

        return None

    def _resolve_image_tokens_from_error(
        self,
        parsed: Optional[ParsedTokenCount],
        prefix_len: int,
        suffix_input_len: int
    ) -> Optional[int]:
        """Translate a parsed API token count into the image token count we should use."""
        if parsed is None or parsed.count <= 0:
            return None

        if parsed.scope == "image":
            return parsed.count

        adjusted = parsed.count - (prefix_len + suffix_input_len)
        return adjusted if adjusted > 0 else None

    def _ensure_image_token_caches(self) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """Ensure image token caches exist even on test doubles constructed via __new__."""
        if not hasattr(self, "_image_token_cache") or self._image_token_cache is None:
            self._image_token_cache = {}
        if (
            not hasattr(self, "_image_dimension_token_cache")
            or self._image_dimension_token_cache is None
        ):
            self._image_dimension_token_cache = {}
        return self._image_token_cache, self._image_dimension_token_cache

    def _get_cached_image_token_count(self, image_path: Path) -> Optional[int]:
        """Look up a cached image token count by path first, then by image dimensions."""
        path_cache, dimension_cache = self._ensure_image_token_caches()

        cache_key = str(image_path)
        cached = path_cache.get(cache_key)
        if cached is not None:
            return cached

        dims = get_image_dimensions(image_path)
        if dims is not None:
            return dimension_cache.get(dims)
        return None

    def _store_image_token_count(self, image_path: Path, token_count: int) -> None:
        """Store a resolved token count by both path and image dimensions."""
        if token_count <= 0:
            return

        path_cache, dimension_cache = self._ensure_image_token_caches()
        path_cache[str(image_path)] = token_count

        dims = get_image_dimensions(image_path)
        if dims is not None:
            dimension_cache[dims] = token_count

    def _get_initial_image_token_count(self, image_path: Path) -> int:
        """Return the best available token guess before any probing."""
        cached = self._get_cached_image_token_count(image_path)
        if cached is not None:
            return cached
        return get_image_token_count(image_path)

    def _build_vision_datum(
        self,
        image_bytes: bytes,
        image_format: str,
        text_continuation: str,
        image_token_count: Optional[int] = None
    ) -> Tuple[Optional[Any], Optional[List[int]], Optional[int], Optional[int]]:
        """Build a Datum for vision perplexity from pre-loaded image bytes."""
        prefix_tokens = self._tokenizer.encode(CHAT_VISION_PREFIX, add_special_tokens=False)
        wrapper_tokens = self._tokenizer.encode(CHAT_VISION_SUFFIX, add_special_tokens=False)
        continuation_tokens = self._tokenizer.encode(text_continuation, add_special_tokens=False)
        if not continuation_tokens:
            return None, None, None, None

        suffix_tokens = wrapper_tokens + continuation_tokens
        if len(suffix_tokens) < 2:
            return None, None, None, None

        suffix_input_tokens = suffix_tokens[:-1]
        target_suffix_tokens = suffix_tokens[1:]

        mask = [0] * len(wrapper_tokens) + [1] * len(continuation_tokens)
        weights_suffix = mask[1:]
        if sum(weights_suffix) == 0:
            return None, None, None, None

        prefix_len = len(prefix_tokens)
        suffix_input_len = len(suffix_input_tokens)
        image_tokens = image_token_count or VISION_IMAGE_TOKEN_COUNT

        model_input = self._tinker.types.ModelInput(
            chunks=[
                self._tinker.types.EncodedTextChunk(tokens=prefix_tokens),
                self._tinker.types.ImageChunk(
                    data=image_bytes,
                    format=image_format,
                    expected_tokens=image_tokens
                ),
                self._tinker.types.EncodedTextChunk(tokens=suffix_input_tokens),
            ]
        )

        full_target = [0] * prefix_len + [0] * image_tokens + target_suffix_tokens
        full_weights = [0] * prefix_len + [0] * image_tokens + weights_suffix

        datum = self._tinker.types.Datum(
            model_input=model_input,
            loss_fn_inputs=dict(
                weights=full_weights,
                target_tokens=full_target
            )
        )

        return datum, full_weights, prefix_len, suffix_input_len

    def _resolve_image_token_count(
        self,
        image_path: Path,
        image_bytes: bytes,
        image_format: str,
        text_continuation: str,
        training_client=None,
        force_refresh: bool = False
    ) -> int:
        """Resolve the image token count with a single forward pass."""
        if not force_refresh:
            cached = self._get_cached_image_token_count(image_path)
            if cached is not None:
                return cached

        image_tokens = self._get_initial_image_token_count(image_path)
        datum, weights, prefix_len, suffix_input_len = self._build_vision_datum(
            image_bytes=image_bytes,
            image_format=image_format,
            text_continuation=text_continuation,
            image_token_count=image_tokens
        )
        if datum is None or weights is None or prefix_len is None or suffix_input_len is None:
            self._store_image_token_count(image_path, image_tokens)
            return image_tokens

        client = training_client if training_client is not None else self.get_training_client()

        try:
            future = client.forward([datum], "cross_entropy")
            _call_with_timeout(future, operation="image token calibration")
            self._store_image_token_count(image_path, image_tokens)
            return image_tokens
        except TimeoutError:
            # On timeout, just use the dimension-based estimate
            self._store_image_token_count(image_path, image_tokens)
            return image_tokens
        except Exception as api_error:
            parsed = self._extract_token_count_from_error(str(api_error))
            adjusted = self._resolve_image_tokens_from_error(
                parsed,
                prefix_len=prefix_len,
                suffix_input_len=suffix_input_len
            )
            if adjusted is not None:
                self._store_image_token_count(image_path, adjusted)
                if self.verbose:
                    source = parsed.scope if parsed is not None else "unknown"
                    tprint(f"  Adjusted image token count to {adjusted} for {image_path} ({source})")
                return adjusted
            if self.verbose:
                tprint(f"  Warning: Unable to resolve image token count for {image_path}")
            self._store_image_token_count(image_path, image_tokens)
            return image_tokens

    def compute_perplexity_with_image(
        self,
        image_path: Path,
        text_continuation: str,
        image_token_count: Optional[int] = None,
        adapter_name: Optional[str] = None,
        training_client=None
    ) -> float:
        """
        Compute perplexity of text continuation given image context.

        The model sees a chat-formatted user image followed by an assistant continuation.
        We compute loss only on the continuation tokens, masking wrapper and image tokens.

        Args:
            image_path: Path to image file
            text_continuation: Text to evaluate after image
            image_token_count: Number of tokens image expands to (auto-detected if None)
            adapter_name: Optional LoRA adapter name (deprecated, use training_client)
            training_client: Optional pre-configured training client (e.g., with LoRA loaded)

        Returns:
            Average loss on text tokens (lower = more natural)
        """
        try:
            # Use provided training client or get default (base model)
            client = training_client if training_client is not None else self.get_training_client()

            # Load image
            image_bytes = self._load_image_bytes(image_path)
            image_format = self._get_image_format(image_path)

            # DIAGNOSTIC: Verify image is loaded correctly
            if len(image_bytes) == 0:
                raise ValueError(f"Image loaded with 0 bytes: {image_path}")
            if self.verbose:
                tprint(f"    Image loaded: {len(image_bytes)} bytes, format={image_format}, path={image_path}")

            # Tokenize chat scaffolding + continuation
            prefix_tokens = self._tokenizer.encode(CHAT_VISION_PREFIX, add_special_tokens=False)
            wrapper_tokens = self._tokenizer.encode(CHAT_VISION_SUFFIX, add_special_tokens=False)
            continuation_tokens = self._tokenizer.encode(text_continuation, add_special_tokens=False)
            if not continuation_tokens:
                return float('inf')

            suffix_tokens = wrapper_tokens + continuation_tokens
            if len(suffix_tokens) < 2:
                return float('inf')

            suffix_input_tokens = suffix_tokens[:-1]
            target_suffix_tokens = suffix_tokens[1:]

            mask = [0] * len(wrapper_tokens) + [1] * len(continuation_tokens)
            weights_suffix = mask[1:]
            if sum(weights_suffix) == 0:
                return float('inf')

            prefix_len = len(prefix_tokens)
            suffix_input_len = len(suffix_input_tokens)
            # Use dimension-based lookup if no explicit count provided
            if image_token_count is not None:
                image_tokens = image_token_count
            else:
                image_tokens = self._get_initial_image_token_count(image_path)

            for attempt in range(2):  # Max 2 attempts
                model_input = self._tinker.types.ModelInput(
                    chunks=[
                        self._tinker.types.EncodedTextChunk(tokens=prefix_tokens),
                        self._tinker.types.ImageChunk(
                            data=image_bytes, format=image_format, expected_tokens=image_tokens
                        ),
                        self._tinker.types.EncodedTextChunk(tokens=suffix_input_tokens),
                    ]
                )

                full_target = [0] * prefix_len + [0] * image_tokens + target_suffix_tokens
                full_weights = [0] * prefix_len + [0] * image_tokens + weights_suffix

                datum = self._tinker.types.Datum(
                    model_input=model_input,
                    loss_fn_inputs=dict(
                        weights=full_weights,
                        target_tokens=full_target
                    )
                )

                try:
                    # Forward pass (no gradients) with timeout
                    future = client.forward([datum], "cross_entropy")
                    result = _call_with_timeout(future, operation="image perplexity forward pass")

                    if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > 0:
                        logprobs_raw = result.loss_fn_outputs[0]['logprobs']
                        self._store_image_token_count(image_path, image_tokens)
                        return compute_weighted_loss(logprobs_raw, full_weights)

                    self._store_image_token_count(image_path, image_tokens)
                    return float('inf')

                except TimeoutError as e:
                    if self.verbose:
                        tprint(f"    Warning: Image perplexity computation timed out: {e}")
                    return float('inf')
                except Exception as api_error:
                    # Try to extract actual token count from error
                    parsed = self._extract_token_count_from_error(str(api_error))
                    corrected_tokens = self._resolve_image_tokens_from_error(
                        parsed,
                        prefix_len=prefix_len,
                        suffix_input_len=suffix_input_len
                    )
                    if corrected_tokens is not None and attempt == 0:
                        image_tokens = corrected_tokens
                        self._store_image_token_count(image_path, image_tokens)
                        if self.verbose:
                            source = parsed.scope if parsed is not None else "unknown"
                            tprint(
                                f"    Adjusted image token count to {image_tokens} "
                                f"for {image_path} ({source})"
                            )
                        continue  # Retry with corrected count
                    raise  # Re-raise on second attempt

            self._store_image_token_count(image_path, image_tokens)
            return float('inf')

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Vision perplexity failed: {e}")
            return float('inf')

    def _is_finite_probe_loss(self, value: Optional[float]) -> bool:
        """Return whether a probe loss is finite."""
        return value is not None and math.isfinite(float(value))

    def _compute_best_heuristic_summary(self, losses: Dict[str, float]) -> Tuple[str, float]:
        """Return legacy best-heuristic argmin and confidence."""
        valid_loss_map = {h: l for h, l in losses.items() if self._is_finite_probe_loss(l)}
        if not valid_loss_map:
            return ("UNKNOWN", 0.0)

        best_heuristic = min(valid_loss_map, key=lambda h: valid_loss_map[h])
        sorted_vals = sorted(valid_loss_map.values())
        if len(sorted_vals) >= 2 and sorted_vals[1] > 0:
            gap = (sorted_vals[1] - sorted_vals[0]) / sorted_vals[1]
            confidence = min(1.0, gap * 2)
        else:
            confidence = 0.5
        return (best_heuristic, confidence)

    def _build_image_probe_result(
        self,
        losses: Dict[str, float],
        neutral_loss: Optional[float],
        per_template_losses: Dict[str, Dict[str, Any]],
        *,
        image_token_count: Optional[int],
        resolution_status: Optional[str] = None,
        resolution_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a canonical image-probe payload with explicit resolution metadata."""
        delta_losses = None
        if self._is_finite_probe_loss(neutral_loss):
            delta_losses = {
                h: loss - neutral_loss
                for h, loss in losses.items()
            }

        probe_resolved = self._is_finite_probe_loss(neutral_loss) and all(
            self._is_finite_probe_loss(losses.get(h))
            for h in ("OT", "DD", "RC")
        )
        best_heuristic, confidence = self._compute_best_heuristic_summary(losses)
        if resolution_status is None:
            resolution_status = "ok" if probe_resolved else "unresolved"
        elif not probe_resolved and resolution_status in {"ok", "retry_resolved", "fallback_resolved"}:
            resolution_status = "unresolved"

        return {
            "losses": losses,
            "neutral_loss": neutral_loss,
            "delta_losses": delta_losses,
            "per_template_losses": per_template_losses,
            "best_heuristic": best_heuristic,
            "confidence": confidence,
            "probe_resolved": probe_resolved,
            "probe_resolution_status": resolution_status,
            "probe_resolution_error": resolution_error,
            "probe_image_token_count": image_token_count,
        }

    def compute_heuristic_losses_with_image(
        self,
        image_path: Path,
        a: int,
        b: int,
        adapter_name: Optional[str] = None,
        training_client=None,
        include_neutral: bool = True
    ) -> Dict:
        """
        Compute perplexity losses for all heuristic text continuations given image.

        The image shows "a × b = ?" and we test how natural each heuristic
        continuation feels as a response.

        Args:
            image_path: Path to problem image
            a: First operand (shown in image)
            b: Second operand (shown in image)
            adapter_name: Optional LoRA adapter name (deprecated, use training_client)
            training_client: Optional pre-configured training client (e.g., with LoRA loaded)
            include_neutral: Include neutral baseline for delta loss computation

        Returns:
            Dict with 'losses', 'neutral_loss', and 'delta_losses' keys
        """
        templates = get_image_heuristic_templates(a, b)
        losses: Dict[str, float] = {}
        per_template_losses: Dict[str, Dict[str, Any]] = {}

        # Process each heuristic template (multi-template averaging)
        # Note: Image token count is auto-detected on first call
        for h_name, continuations in templates.items():
            h_losses: List[float] = []
            for t_idx, continuation in enumerate(continuations):
                loss = self.compute_perplexity_with_image(
                    image_path=image_path,
                    text_continuation=continuation,
                    adapter_name=adapter_name,
                    training_client=training_client
                )
                per_template_losses[f"{h_name}_{t_idx}"] = {
                    "prompt": continuation,
                    "loss": loss
                }
                if loss < float('inf'):
                    h_losses.append(loss)
            losses[h_name] = sum(h_losses) / len(h_losses) if h_losses else float('inf')

        # Compute neutral baseline
        neutral_loss = None

        if include_neutral:
            neutral_template = get_image_neutral_baseline_template(a, b)
            neutral_loss = self.compute_perplexity_with_image(
                image_path=image_path,
                text_continuation=neutral_template,
                adapter_name=adapter_name,
                training_client=training_client
            )
            per_template_losses["NEUTRAL"] = {
                "prompt": neutral_template,
                "loss": neutral_loss
            }
        cached_image_tokens = self._get_cached_image_token_count(image_path)
        return self._build_image_probe_result(
            losses,
            neutral_loss,
            per_template_losses,
            image_token_count=cached_image_tokens,
        )

    def compute_contrastive_step_losses_with_image(
        self,
        image_path: Path,
        a: int,
        b: int,
        adapter_name: Optional[str] = None,
        training_client=None
    ) -> Dict[str, Any]:
        """
        Compute contrastive losses for correct vs incorrect heuristic steps given image.

        This sequentially scores each contrastive template using image context.
        """
        templates = get_image_contrastive_step_templates(a, b)
        correct_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
        incorrect_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
        per_template_losses: Dict[str, Dict[str, Any]] = {}

        for h_name, pairs in templates.items():
            for t_idx, (correct, incorrect) in enumerate(pairs):
                for label, continuation in (("correct", correct), ("incorrect", incorrect)):
                    loss = self.compute_perplexity_with_image(
                        image_path=image_path,
                        text_continuation=continuation,
                        adapter_name=adapter_name,
                        training_client=training_client
                    )
                    template_id = f"{h_name}_{t_idx}_{label.upper()}"
                    per_template_losses[template_id] = {"prompt": continuation, "loss": loss}
                    if label == "correct":
                        correct_losses[h_name].append(loss)
                    else:
                        incorrect_losses[h_name].append(loss)

        avg_correct = {}
        avg_incorrect = {}
        delta_losses = {}
        for h in templates.keys():
            correct_vals = [l for l in correct_losses[h] if l < float('inf')]
            incorrect_vals = [l for l in incorrect_losses[h] if l < float('inf')]
            avg_correct[h] = sum(correct_vals) / len(correct_vals) if correct_vals else float('inf')
            avg_incorrect[h] = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else float('inf')
            delta_losses[h] = avg_incorrect[h] - avg_correct[h]

        return {
            "correct_losses": avg_correct,
            "incorrect_losses": avg_incorrect,
            "delta_losses": delta_losses,
            "per_template_losses": per_template_losses
        }

    def compute_contrastive_step_losses_with_image_batched(
        self,
        image_path: Path,
        a: int,
        b: int,
        adapter_name: Optional[str] = None,
        training_client=None
    ) -> Dict[str, Any]:
        """
        Compute contrastive losses for correct vs incorrect heuristic steps given image,
        batching all templates into a single forward() call.
        """
        templates = get_image_contrastive_step_templates(a, b)
        correct_losses = {h: float('inf') for h in templates.keys()}
        incorrect_losses = {h: float('inf') for h in templates.keys()}
        delta_losses = {h: float('inf') for h in templates.keys()}
        per_template_losses: Dict[str, Dict[str, Any]] = {}

        try:
            image_bytes = self._load_image_bytes(image_path)
            image_format = self._get_image_format(image_path)
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Image load failed: {e}")
            return {
                "correct_losses": correct_losses,
                "incorrect_losses": incorrect_losses,
                "delta_losses": delta_losses,
                "per_template_losses": per_template_losses
            }

        client = training_client if training_client is not None else self.get_training_client()
        image_tokens = self._get_initial_image_token_count(image_path)

        def build_datums(token_count: int):
            datums: List[Any] = []
            metadata: List[Tuple[str, int, str, Optional[List[int]], str]] = []

            for h_name, pairs in templates.items():
                for t_idx, (correct, incorrect) in enumerate(pairs):
                    for label, continuation in (("correct", correct), ("incorrect", incorrect)):
                        datum, weights, _, _ = self._build_vision_datum(
                            image_bytes=image_bytes,
                            image_format=image_format,
                            text_continuation=continuation,
                            image_token_count=token_count
                        )
                        if datum is None or weights is None:
                            metadata.append((h_name, t_idx, label, None, continuation))
                            continue
                        datums.append(datum)
                        metadata.append((h_name, t_idx, label, weights, continuation))

            return datums, metadata

        def run_batch(token_count: int):
            batch_datums, metadata = build_datums(token_count)
            if not batch_datums:
                return {h: [] for h in templates.keys()}, {h: [] for h in templates.keys()}, {}

            future = client.forward(batch_datums, "cross_entropy")
            result = _call_with_timeout(future, operation="image contrastive batch forward pass")

            batch_correct: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            batch_incorrect: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            batch_per_template: Dict[str, Dict[str, Any]] = {}

            datum_idx = 0
            for h_name, t_idx, label, weights, continuation in metadata:
                template_id = f"{h_name}_{t_idx}_{label.upper()}"
                if weights is None:
                    loss = float('inf')
                elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                    datum_idx += 1
                else:
                    loss = float('inf')

                batch_per_template[template_id] = {"prompt": continuation, "loss": loss}
                if label == "correct":
                    batch_correct[h_name].append(loss)
                else:
                    batch_incorrect[h_name].append(loss)

            return batch_correct, batch_incorrect, batch_per_template

        try:
            batch_correct, batch_incorrect, per_template_losses = run_batch(image_tokens)
        except Exception as batch_error:
            if self.verbose:
                tprint(f"    Warning: Batched contrastive vision failed: {batch_error}")
            probe_text = next(iter(templates.values()))[0][0]
            image_tokens = self._resolve_image_token_count(
                image_path=image_path,
                image_bytes=image_bytes,
                image_format=image_format,
                text_continuation=probe_text,
                training_client=training_client,
                force_refresh=True
            )
            try:
                batch_correct, batch_incorrect, per_template_losses = run_batch(image_tokens)
            except Exception as retry_error:
                if self.verbose:
                    tprint(f"    Warning: Batched retry failed: {retry_error}")
                return self.compute_contrastive_step_losses_with_image(
                    image_path=image_path,
                    a=a,
                    b=b,
                    adapter_name=adapter_name,
                    training_client=training_client
                )

        self._store_image_token_count(image_path, image_tokens)

        for h in templates.keys():
            correct_vals = [l for l in batch_correct[h] if l < float('inf')]
            incorrect_vals = [l for l in batch_incorrect[h] if l < float('inf')]
            correct_losses[h] = sum(correct_vals) / len(correct_vals) if correct_vals else float('inf')
            incorrect_losses[h] = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else float('inf')
            delta_losses[h] = incorrect_losses[h] - correct_losses[h]

        return {
            "correct_losses": correct_losses,
            "incorrect_losses": incorrect_losses,
            "delta_losses": delta_losses,
            "per_template_losses": per_template_losses
        }

    def compute_heuristic_losses_with_image_batched(
        self,
        image_path: Path,
        a: int,
        b: int,
        adapter_name: Optional[str] = None,
        training_client=None,
        include_neutral: bool = True
    ) -> Dict:
        """
        Compute perplexity losses for all heuristic text continuations given image,
        batching all templates into a single forward() call.
        """
        templates = get_image_heuristic_templates(a, b)
        losses = {h: float('inf') for h in templates.keys()}
        neutral_loss = None
        delta_losses = None
        per_template_losses: Dict[str, Dict[str, Any]] = {}

        try:
            image_bytes = self._load_image_bytes(image_path)
            image_format = self._get_image_format(image_path)
        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Image load failed: {e}")
            return self._build_image_probe_result(
                losses,
                float('inf') if include_neutral else None,
                per_template_losses,
                image_token_count=None,
                resolution_status="image_load_failed",
                resolution_error=str(e),
            )

        client = training_client if training_client is not None else self.get_training_client()
        image_tokens = self._get_initial_image_token_count(image_path)
        resolution_status = "ok"
        resolution_error: Optional[str] = None

        neutral_template = None
        if include_neutral:
            neutral_template = get_image_neutral_baseline_template(a, b)

        def build_datums(token_count: int):
            datums: List[Any] = []
            metadata: List[Tuple[str, int, Optional[List[int]], str]] = []

            for h_name, continuations in templates.items():
                for t_idx, continuation in enumerate(continuations):
                    datum, weights, _, _ = self._build_vision_datum(
                        image_bytes=image_bytes,
                        image_format=image_format,
                        text_continuation=continuation,
                        image_token_count=token_count
                    )
                    if datum is None or weights is None:
                        metadata.append((h_name, t_idx, None, continuation))
                        continue
                    datums.append(datum)
                    metadata.append((h_name, t_idx, weights, continuation))

            if include_neutral and neutral_template is not None:
                datum, weights, _, _ = self._build_vision_datum(
                    image_bytes=image_bytes,
                    image_format=image_format,
                    text_continuation=neutral_template,
                    image_token_count=token_count
                )
                if datum is None or weights is None:
                    metadata.append(("NEUTRAL", 0, None, neutral_template))
                else:
                    datums.append(datum)
                    metadata.append(("NEUTRAL", 0, weights, neutral_template))

            return datums, metadata

        def run_batch(token_count: int):
            batch_datums, metadata = build_datums(token_count)
            if not batch_datums:
                empty_neutral = float('inf') if include_neutral else None
                return {h: [] for h in templates.keys()}, empty_neutral, {}

            future = client.forward(batch_datums, "cross_entropy")
            result = _call_with_timeout(future, operation="image heuristic batch forward pass")

            batch_heuristic_losses: Dict[str, List[float]] = {h: [] for h in templates.keys()}
            batch_neutral = None
            batch_per_template: Dict[str, Dict[str, Any]] = {}
            datum_idx = 0

            for name, t_idx, weights, continuation in metadata:
                template_id = "NEUTRAL" if name == "NEUTRAL" else f"{name}_{t_idx}"

                if weights is None:
                    batch_per_template[template_id] = {"prompt": continuation, "loss": float('inf')}
                    if name == "NEUTRAL":
                        batch_neutral = float('inf')
                    else:
                        batch_heuristic_losses[name].append(float('inf'))
                    continue

                if hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                else:
                    loss = float('inf')

                batch_per_template[template_id] = {"prompt": continuation, "loss": loss}
                if name == "NEUTRAL":
                    batch_neutral = loss
                else:
                    batch_heuristic_losses[name].append(loss)
                datum_idx += 1

            if include_neutral and batch_neutral is None:
                batch_neutral = float('inf')

            return batch_heuristic_losses, batch_neutral, batch_per_template

        try:
            heuristic_losses, neutral_loss, per_template_losses = run_batch(image_tokens)
        except Exception as batch_error:
            if self.verbose:
                tprint(f"    Warning: Batched vision perplexity failed: {batch_error}")
            resolution_status = "retry_resolved"
            resolution_error = str(batch_error)
            probe_text = neutral_template or next(iter(templates.values()))[0]
            image_tokens = self._resolve_image_token_count(
                image_path=image_path,
                image_bytes=image_bytes,
                image_format=image_format,
                text_continuation=probe_text,
                training_client=training_client,
                force_refresh=True
            )
            try:
                heuristic_losses, neutral_loss, per_template_losses = run_batch(image_tokens)
            except Exception as retry_error:
                if self.verbose:
                    tprint(f"    Warning: Batched retry failed: {retry_error}")
                fallback = self.compute_heuristic_losses_with_image(
                    image_path=image_path,
                    a=a,
                    b=b,
                    adapter_name=adapter_name,
                    training_client=training_client,
                    include_neutral=include_neutral
                )
                fallback.setdefault("probe_resolution_error", str(retry_error))
                if fallback.get("probe_resolved"):
                    fallback["probe_resolution_status"] = "fallback_resolved"
                else:
                    fallback["probe_resolution_status"] = "fallback_unresolved"
                return fallback

        self._store_image_token_count(image_path, image_tokens)

        for h, h_losses in heuristic_losses.items():
            valid = [l for l in h_losses if l < float('inf')]
            losses[h] = sum(valid) / len(valid) if valid else float('inf')
        return self._build_image_probe_result(
            losses,
            neutral_loss,
            per_template_losses,
            image_token_count=image_tokens,
            resolution_status=resolution_status,
            resolution_error=resolution_error,
        )

    def compute_heuristic_losses_multi_image(
        self,
        problems: List[Tuple[Path, int, int]],
        batch_size: int = 30,
        include_neutral: bool = True,
        training_client=None
    ) -> List[Dict[str, Any]]:
        """
        Compute heuristic losses for multiple image problems in batched API calls.

        Each problem includes an image path and operands (a, b). This batches
        all heuristic templates across multiple images into fewer forward calls.
        """
        if not problems:
            return []

        normalized: List[Tuple[Path, int, int]] = []
        for image_path, a, b in problems:
            normalized.append((Path(image_path), a, b))

        heuristic_names = ("OT", "DD", "RC")
        all_results: List[Dict[str, Any]] = [
            {
                "losses": {h: float('inf') for h in heuristic_names},
                "neutral_loss": float('inf') if include_neutral else None,
                "delta_losses": None,
                "per_template_losses": {},
                "best_heuristic": "UNKNOWN",
                "confidence": 0.0,
                "_heuristic_losses": {h: [] for h in heuristic_names},
                "probe_resolved": False,
                "probe_resolution_status": "uninitialized",
                "probe_resolution_error": None,
                "probe_image_token_count": None,
            }
            for _ in normalized
        ]

        client = training_client if training_client is not None else self.get_training_client()
        template_count = sum(len(t) for t in get_image_heuristic_templates(0, 0).values())
        templates_per_problem = template_count + (1 if include_neutral else 0)
        problems_per_batch = max(1, batch_size // templates_per_problem)

        for batch_start in range(0, len(normalized), problems_per_batch):
            batch = normalized[batch_start:batch_start + problems_per_batch]
            image_ctx: Dict[int, Dict[str, Any]] = {}
            resolution_status_by_idx: Dict[int, str] = {}
            resolution_error_by_idx: Dict[int, Optional[str]] = {}

            for local_idx, (image_path, a, b) in enumerate(batch):
                global_idx = batch_start + local_idx
                if not image_path.exists():
                    if self.verbose:
                        tprint(f"    Warning: Image not found: {image_path}")
                    all_results[global_idx]["probe_resolution_status"] = "image_missing"
                    all_results[global_idx]["probe_resolution_error"] = f"Image not found: {image_path}"
                    continue
                try:
                    image_bytes = self._load_image_bytes(image_path)
                    image_format = self._get_image_format(image_path)
                except Exception as e:
                    if self.verbose:
                        tprint(f"    Warning: Image load failed: {image_path} ({e})")
                    all_results[global_idx]["probe_resolution_status"] = "image_load_failed"
                    all_results[global_idx]["probe_resolution_error"] = str(e)
                    continue

                token_guess = self._get_initial_image_token_count(image_path)
                image_ctx[global_idx] = {
                    "image_path": image_path,
                    "image_bytes": image_bytes,
                    "image_format": image_format,
                    "a": a,
                    "b": b,
                    "token_count": token_guess
                }
                resolution_status_by_idx[global_idx] = "ok"
                resolution_error_by_idx[global_idx] = None

            def build_datums(token_overrides: Optional[Dict[int, int]] = None):
                datums: List[Any] = []
                metadata: List[Tuple[int, str, str, Optional[List[int]], str]] = []

                for global_idx, ctx in image_ctx.items():
                    token_count = ctx["token_count"]
                    if token_overrides and global_idx in token_overrides:
                        token_count = token_overrides[global_idx]

                    templates = get_image_heuristic_templates(ctx["a"], ctx["b"])
                    for h_name, continuations in templates.items():
                        for t_idx, continuation in enumerate(continuations):
                            template_id = f"{h_name}_{t_idx}"
                            datum, weights, _, _ = self._build_vision_datum(
                                image_bytes=ctx["image_bytes"],
                                image_format=ctx["image_format"],
                                text_continuation=continuation,
                                image_token_count=token_count
                            )
                            if datum is None or weights is None:
                                metadata.append((global_idx, h_name, template_id, None, continuation))
                                continue
                            datums.append(datum)
                            metadata.append((global_idx, h_name, template_id, weights, continuation))

                    if include_neutral:
                        neutral_template = get_image_neutral_baseline_template(ctx["a"], ctx["b"])
                        datum, weights, _, _ = self._build_vision_datum(
                            image_bytes=ctx["image_bytes"],
                            image_format=ctx["image_format"],
                            text_continuation=neutral_template,
                            image_token_count=token_count
                        )
                        if datum is None or weights is None:
                            metadata.append((global_idx, "NEUTRAL", "NEUTRAL", None, neutral_template))
                            continue
                        datums.append(datum)
                        metadata.append((global_idx, "NEUTRAL", "NEUTRAL", weights, neutral_template))

                return datums, metadata

            def run_batch(token_overrides: Optional[Dict[int, int]] = None):
                datums, metadata = build_datums(token_overrides)
                if not datums:
                    return None, None

                future = client.forward(datums, "cross_entropy")
                result = _call_with_timeout(future, operation="image multi-batch forward pass")
                return result, metadata

            resolved_token_counts = {
                global_idx: ctx["token_count"] for global_idx, ctx in image_ctx.items()
            }

            try:
                result, metadata = run_batch()
            except Exception as batch_error:
                if self.verbose:
                    tprint(f"    Warning: Image multi-batch failed: {batch_error}")
                for global_idx in image_ctx:
                    resolution_status_by_idx[global_idx] = "retry_resolved"
                    resolution_error_by_idx[global_idx] = str(batch_error)
                token_overrides: Dict[int, int] = {}
                for global_idx, ctx in image_ctx.items():
                    probe_text = get_image_neutral_baseline_template(ctx["a"], ctx["b"])
                    if not include_neutral:
                        probe_text = next(iter(get_image_heuristic_templates(ctx["a"], ctx["b"]).values()))[0]
                    token_overrides[global_idx] = self._resolve_image_token_count(
                        image_path=ctx["image_path"],
                        image_bytes=ctx["image_bytes"],
                        image_format=ctx["image_format"],
                        text_continuation=probe_text,
                        training_client=training_client,
                        force_refresh=True
                    )
                resolved_token_counts.update(token_overrides)
                try:
                    result, metadata = run_batch(token_overrides)
                except Exception as retry_error:
                    if self.verbose:
                        tprint(f"    Warning: Image multi-batch retry failed: {retry_error}")
                    # Fall back to per-image computation for this batch.
                    for global_idx, ctx in image_ctx.items():
                        probe = self.compute_heuristic_losses_with_image_batched(
                            image_path=ctx["image_path"],
                            a=ctx["a"],
                            b=ctx["b"],
                            training_client=training_client,
                            include_neutral=include_neutral
                        )
                        if probe.get("probe_resolution_error") is None:
                            probe["probe_resolution_error"] = str(retry_error)
                        if probe.get("probe_resolved"):
                            probe["probe_resolution_status"] = "fallback_resolved"
                        else:
                            probe["probe_resolution_status"] = "fallback_unresolved"
                        all_results[global_idx] = probe
                    continue

            if result is None or metadata is None:
                for local_idx in range(len(batch)):
                    global_idx = batch_start + local_idx
                    if global_idx in image_ctx:
                        token_count = resolved_token_counts.get(global_idx, image_ctx[global_idx]["token_count"])
                    else:
                        token_count = None
                    all_results[global_idx] = self._build_image_probe_result(
                        {h: float('inf') for h in heuristic_names},
                        float('inf') if include_neutral else None,
                        {},
                        image_token_count=token_count,
                        resolution_status=resolution_status_by_idx.get(global_idx, "unresolved"),
                        resolution_error=resolution_error_by_idx.get(global_idx),
                    )
                continue

            for global_idx, ctx in image_ctx.items():
                self._store_image_token_count(
                    ctx["image_path"],
                    resolved_token_counts.get(global_idx, ctx["token_count"])
                )

            datum_idx = 0
            for global_idx, name, template_id, weights, continuation in metadata:
                res = all_results[global_idx]
                if weights is None:
                    loss = float('inf')
                elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                    logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                    loss = compute_weighted_loss(logprobs, weights)
                    datum_idx += 1
                else:
                    loss = float('inf')
                res["per_template_losses"][template_id] = {
                    "prompt": continuation,
                    "loss": loss,
                }

                if name == "NEUTRAL":
                    res["neutral_loss"] = loss
                else:
                    res["_heuristic_losses"][name].append(loss)

            for local_idx in range(len(batch)):
                global_idx = batch_start + local_idx
                res = all_results[global_idx]
                heuristic_losses = res.pop("_heuristic_losses", None)
                if heuristic_losses is not None:
                    avg_losses: Dict[str, float] = {}
                    for h, losses in heuristic_losses.items():
                        valid = [l for l in losses if l < float('inf')]
                        avg_losses[h] = sum(valid) / len(valid) if valid else float('inf')
                    res["losses"] = avg_losses

                all_results[global_idx] = self._build_image_probe_result(
                    res["losses"],
                    res.get("neutral_loss"),
                    res.get("per_template_losses", {}),
                    image_token_count=resolved_token_counts.get(
                        global_idx,
                        image_ctx.get(global_idx, {}).get("token_count"),
                    ),
                    resolution_status=resolution_status_by_idx.get(global_idx),
                    resolution_error=resolution_error_by_idx.get(global_idx),
                )

        return all_results

    async def compute_heuristic_losses_multi_image_async(
        self,
        problems: List[Tuple[Path, int, int]],
        batch_size: int = 30,
        include_neutral: bool = True,
        training_client=None,
        max_in_flight: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Async version of compute_heuristic_losses_multi_image for use in async contexts.

        Uses forward_async() to avoid sync/async mixing warnings and pipelines
        requests with a bounded number of in-flight batches.
        """
        import asyncio

        if not problems:
            return []

        normalized: List[Tuple[Path, int, int]] = []
        for image_path, a, b in problems:
            normalized.append((Path(image_path), a, b))

        heuristic_names = ("OT", "DD", "RC")

        # Use async client initialization to avoid sync/async mixing
        client = training_client if training_client is not None else await self.get_training_client_async()
        max_in_flight = max(1, max_in_flight)
        template_count = sum(len(t) for t in get_image_heuristic_templates(0, 0).values())
        templates_per_problem = template_count + (1 if include_neutral else 0)
        problems_per_batch = max(1, batch_size // templates_per_problem)

        def _empty_result() -> Dict[str, Any]:
            return {
                "losses": {h: float('inf') for h in heuristic_names},
                "neutral_loss": float('inf') if include_neutral else None,
                "delta_losses": None,
                "per_template_losses": {},
                "best_heuristic": "UNKNOWN",
                "confidence": 0.0,
                "_heuristic_losses": {h: [] for h in heuristic_names},
                "probe_resolved": False,
                "probe_resolution_status": "uninitialized",
                "probe_resolution_error": None,
                "probe_image_token_count": None,
            }

        all_results: List[Dict[str, Any]] = [_empty_result() for _ in normalized]

        semaphore = asyncio.Semaphore(max_in_flight)
        pending_batches: set = set()

        async def submit_batch(batch_start: int, batch: List[Tuple[Path, int, int]]):
            batch_results: List[Dict[str, Any]] = [_empty_result() for _ in batch]
            image_ctx: Dict[int, Dict[str, Any]] = {}
            resolution_status_by_idx: Dict[int, str] = {}
            resolution_error_by_idx: Dict[int, Optional[str]] = {}

            for local_idx, (image_path, a, b) in enumerate(batch):
                if not image_path.exists():
                    if self.verbose:
                        tprint(f"    Warning: Image not found: {image_path}")
                    batch_results[local_idx]["probe_resolution_status"] = "image_missing"
                    batch_results[local_idx]["probe_resolution_error"] = f"Image not found: {image_path}"
                    continue
                try:
                    image_bytes = self._load_image_bytes(image_path)
                    image_format = self._get_image_format(image_path)
                except Exception as e:
                    if self.verbose:
                        tprint(f"    Warning: Image load failed: {image_path} ({e})")
                    batch_results[local_idx]["probe_resolution_status"] = "image_load_failed"
                    batch_results[local_idx]["probe_resolution_error"] = str(e)
                    continue

                token_guess = self._get_initial_image_token_count(image_path)
                image_ctx[local_idx] = {
                    "image_path": image_path,
                    "image_bytes": image_bytes,
                    "image_format": image_format,
                    "a": a,
                    "b": b,
                    "token_count": token_guess
                }
                resolution_status_by_idx[local_idx] = "ok"
                resolution_error_by_idx[local_idx] = None

            def build_datums(token_overrides: Optional[Dict[int, int]] = None):
                datums: List[Any] = []
                metadata: List[Tuple[int, str, str, Optional[List[int]], str]] = []

                for local_idx, ctx in image_ctx.items():
                    token_count = ctx["token_count"]
                    if token_overrides and local_idx in token_overrides:
                        token_count = token_overrides[local_idx]

                    templates = get_image_heuristic_templates(ctx["a"], ctx["b"])
                    for h_name, continuations in templates.items():
                        for t_idx, continuation in enumerate(continuations):
                            template_id = f"{h_name}_{t_idx}"
                            datum, weights, _, _ = self._build_vision_datum(
                                image_bytes=ctx["image_bytes"],
                                image_format=ctx["image_format"],
                                text_continuation=continuation,
                                image_token_count=token_count
                            )
                            if datum is None or weights is None:
                                metadata.append((local_idx, h_name, template_id, None, continuation))
                                continue
                            datums.append(datum)
                            metadata.append((local_idx, h_name, template_id, weights, continuation))

                    if include_neutral:
                        neutral_template = get_image_neutral_baseline_template(ctx["a"], ctx["b"])
                        datum, weights, _, _ = self._build_vision_datum(
                            image_bytes=ctx["image_bytes"],
                            image_format=ctx["image_format"],
                            text_continuation=neutral_template,
                            image_token_count=token_count
                        )
                        if datum is None or weights is None:
                            metadata.append((local_idx, "NEUTRAL", "NEUTRAL", None, neutral_template))
                            continue
                        datums.append(datum)
                        metadata.append((local_idx, "NEUTRAL", "NEUTRAL", weights, neutral_template))

                return datums, metadata

            async def run_batch_async(token_overrides: Optional[Dict[int, int]] = None):
                datums, metadata = build_datums(token_overrides)
                if not datums:
                    return None, None

                api_future = await asyncio.wait_for(
                    client.forward_async(datums, "cross_entropy"),
                    timeout=API_CALL_TIMEOUT
                )
                # Double-await pattern to avoid blocking the event loop
                result = await asyncio.wait_for(api_future, timeout=API_CALL_TIMEOUT)
                return result, metadata

            resolved_token_counts = {
                local_idx: ctx["token_count"] for local_idx, ctx in image_ctx.items()
            }

            async with semaphore:
                try:
                    result, metadata = await run_batch_async()
                except Exception as batch_error:
                    if self.verbose:
                        tprint(f"    Warning: Async image multi-batch failed: {batch_error}")
                    for local_idx in image_ctx:
                        resolution_status_by_idx[local_idx] = "retry_resolved"
                        resolution_error_by_idx[local_idx] = str(batch_error)
                    token_overrides: Dict[int, int] = {}
                    for local_idx, ctx in image_ctx.items():
                        probe_text = get_image_neutral_baseline_template(ctx["a"], ctx["b"])
                        if not include_neutral:
                            probe_text = next(iter(get_image_heuristic_templates(ctx["a"], ctx["b"]).values()))[0]
                        token_overrides[local_idx] = self._resolve_image_token_count(
                            image_path=ctx["image_path"],
                            image_bytes=ctx["image_bytes"],
                            image_format=ctx["image_format"],
                            text_continuation=probe_text,
                            training_client=training_client,
                            force_refresh=True
                        )
                    resolved_token_counts.update(token_overrides)
                    try:
                        result, metadata = await run_batch_async(token_overrides)
                    except Exception as retry_error:
                        if self.verbose:
                            tprint(f"    Warning: Async image multi-batch retry failed: {retry_error}")
                        for local_idx, ctx in image_ctx.items():
                            probe = self.compute_heuristic_losses_with_image_batched(
                                image_path=ctx["image_path"],
                                a=ctx["a"],
                                b=ctx["b"],
                                training_client=training_client,
                                include_neutral=include_neutral
                            )
                            if probe.get("probe_resolution_error") is None:
                                probe["probe_resolution_error"] = str(retry_error)
                            if probe.get("probe_resolved"):
                                probe["probe_resolution_status"] = "fallback_resolved"
                            else:
                                probe["probe_resolution_status"] = "fallback_unresolved"
                            batch_results[local_idx] = probe
                        return batch_start, batch_results

                if result is None or metadata is None:
                    for local_idx in range(len(batch)):
                        if local_idx in image_ctx:
                            token_count = resolved_token_counts.get(local_idx, image_ctx[local_idx]["token_count"])
                        else:
                            token_count = None
                        batch_results[local_idx] = self._build_image_probe_result(
                            {h: float('inf') for h in heuristic_names},
                            float('inf') if include_neutral else None,
                            {},
                            image_token_count=token_count,
                            resolution_status=resolution_status_by_idx.get(local_idx, "unresolved"),
                            resolution_error=resolution_error_by_idx.get(local_idx),
                        )
                    return batch_start, batch_results

                for local_idx, ctx in image_ctx.items():
                    self._store_image_token_count(
                        ctx["image_path"],
                        resolved_token_counts.get(local_idx, ctx["token_count"])
                    )

                datum_idx = 0
                for local_idx, name, template_id, weights, continuation in metadata:
                    res = batch_results[local_idx]
                    if weights is None:
                        loss = float('inf')
                    elif hasattr(result, 'loss_fn_outputs') and len(result.loss_fn_outputs) > datum_idx:
                        logprobs = result.loss_fn_outputs[datum_idx]['logprobs']
                        loss = compute_weighted_loss(logprobs, weights)
                        datum_idx += 1
                    else:
                        loss = float('inf')
                    res["per_template_losses"][template_id] = {
                        "prompt": continuation,
                        "loss": loss,
                    }

                    if name == "NEUTRAL":
                        res["neutral_loss"] = loss
                    else:
                        res["_heuristic_losses"][name].append(loss)

            for local_idx, res in enumerate(batch_results):
                heuristic_losses = res.pop("_heuristic_losses", None)
                if heuristic_losses is not None:
                    avg_losses: Dict[str, float] = {}
                    for h, losses in heuristic_losses.items():
                        valid = [l for l in losses if l < float('inf')]
                        avg_losses[h] = sum(valid) / len(valid) if valid else float('inf')
                    res["losses"] = avg_losses

                batch_results[local_idx] = self._build_image_probe_result(
                    res["losses"],
                    res.get("neutral_loss"),
                    res.get("per_template_losses", {}),
                    image_token_count=resolved_token_counts.get(
                        local_idx,
                        image_ctx.get(local_idx, {}).get("token_count"),
                    ),
                    resolution_status=resolution_status_by_idx.get(local_idx),
                    resolution_error=resolution_error_by_idx.get(local_idx),
                )

            return batch_start, batch_results

        for batch_start in range(0, len(normalized), problems_per_batch):
            batch = normalized[batch_start:batch_start + problems_per_batch]
            pending_batches.add(asyncio.create_task(submit_batch(batch_start, batch)))

            if len(pending_batches) >= max_in_flight:
                done, pending_batches = await asyncio.wait(pending_batches, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    batch_start_finished, batch_results = task.result()
                    for local_idx, res in enumerate(batch_results):
                        all_results[batch_start_finished + local_idx] = res

        while pending_batches:
            done, pending_batches = await asyncio.wait(pending_batches, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                batch_start_finished, batch_results = task.result()
                for local_idx, res in enumerate(batch_results):
                    all_results[batch_start_finished + local_idx] = res

        return all_results

    def detect_heuristic_from_image(
        self,
        image_path: Path,
        a: int,
        b: int
    ) -> Tuple[str, float]:
        """
        Detect preferred heuristic from image perplexity probe.

        Args:
            image_path: Path to problem image
            a: First operand
            b: Second operand

        Returns:
            (detected_heuristic, confidence) tuple
        """
        result = self.compute_heuristic_losses_with_image(image_path, a, b, include_neutral=False)
        losses = result.get('losses', {})

        if not losses or not all(v < float('inf') for v in losses.values()):
            return "UNKNOWN", 0.0

        # Find lowest loss = preferred heuristic
        best = min(losses, key=lambda h: losses[h])

        # Compute confidence from loss gap
        sorted_losses = sorted(losses.values())
        if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
            gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
            confidence = min(1.0, gap * 2)
        else:
            confidence = 0.5

        return best, confidence

    def generate_with_image(
        self,
        image_path: Path,
        a: int,
        b: int,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None,
        adapter_name: str = "base",
        adapter_path: Optional[str] = None,
        prompt_text: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Generate model's answer for a multiplication problem shown in an image.

        Args:
            image_path: Path to problem image showing "a × b = ?"
            a: First operand (for prompt construction)
            b: Second operand (for prompt construction)
            with_reasoning: Include step-by-step reasoning
            max_tokens: Max tokens to generate (auto-set based on reasoning)
            adapter_name: Adapter checkpoint name
            adapter_path: Optional LoRA checkpoint path (overrides adapter_name)

        Returns:
            (answer, trace) tuple - answer as int or None, trace if reasoning
        """
        try:
            # Reuse sampling clients per adapter to avoid reinitializing for every image.
            sampler = self.get_sampling_client(
                adapter_name=adapter_name,
                adapter_path=adapter_path
            )

            prompt_text, max_tokens = self._resolve_image_generation_prompt(
                with_reasoning=with_reasoning,
                max_tokens=max_tokens,
                prompt_text=prompt_text,
            )
            model_input = self._build_image_generation_input(image_path, prompt_text, a, b)

            # Sampling params
            sampling_params = self._tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic
            )

            # Generate
            future = sampler.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            )
            result = _call_with_timeout(future, operation="sampling generation")

            text = self._extract_text_from_sampling_result(result)

            # Parse answer
            answer = extract_answer(text)
            trace = text if with_reasoning else None

            # Contamination detection: check if trace mentions expected operands
            # If model is solving wrong problem (e.g., "7 × 8" instead of "47 × 53"),
            # the trace won't contain the expected operands
            operand_in_trace = str(a) in text or str(b) in text
            expected_answer_in_trace = str(a * b) in text.replace(",", "")  # Handle "39,996" format

            if self.verbose:
                tprint(f"    Gen: answer={answer}")
                if not operand_in_trace:
                    tprint(f"    WARNING: Trace may be contaminated - expected operands {a}, {b} not found in output!")
                    tprint(f"    Trace preview: {text[:200]}...")
                # Detailed output to log file for debugging
                log_detail("image_generation.log",
                    f"answer={answer}, operand_check={operand_in_trace}, "
                    f"expected_answer_check={expected_answer_in_trace}, text={text[:300]}...")

            return answer, trace

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Image generation failed: {e}")
            return None, None

    async def generate_with_image_async(
        self,
        image_path: Path,
        a: int,
        b: int,
        sampler,
        with_reasoning: bool = False,
        max_tokens: Optional[int] = None,
        prompt_text: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """Async image generation variant for reuse in async experiment runners."""
        try:
            prompt_text, max_tokens = self._resolve_image_generation_prompt(
                with_reasoning=with_reasoning,
                max_tokens=max_tokens,
                prompt_text=prompt_text,
            )
            model_input = self._build_image_generation_input(image_path, prompt_text, a, b)

            sampling_params = self._tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0
            )
            result = await sampler.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            )

            text = self._extract_text_from_sampling_result(result)
            answer = extract_answer(text)
            return answer, text

        except Exception as e:
            if self.verbose:
                tprint(f"    Warning: Async image generation failed: {e}")
            return None, None


def create_vision_client(
    model_name: str = DEFAULT_VISION_MODEL,
    verbose: bool = True
) -> VisionTinkerClient:
    """
    Create a VisionTinkerClient with common defaults.

    Args:
        model_name: HuggingFace vision model name
        verbose: Print initialization messages

    Returns:
        Initialized VisionTinkerClient
    """
    return VisionTinkerClient(model_name=model_name, verbose=verbose)
