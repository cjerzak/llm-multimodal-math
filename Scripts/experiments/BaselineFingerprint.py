#!/usr/bin/env python3
from __future__ import annotations
"""
BaselineFingerprint.py

Run baseline fingerprinting on the HDS dataset using Tinker API.

This script:
1. Loads the HDS (Heuristic-Disagreement Set)
2. Runs perplexity probes on each problem
3. Evaluates model accuracy on each problem
4. Compares detected heuristics to expected heuristics
5. Saves results and generates summary statistics
"""

# =============================================================================
# MODEL CONFIGURATION - Using VLM for both text and image modalities
# =============================================================================
# Using the same VLM for both modalities enables apples-to-apples cross-modal comparison.
# The VLM can process text-only inputs, so we use it for both text and image fingerprinting.
#
# Default: Qwen3-VL-30B (MoE Vision-Language model)
# =============================================================================

import os
import sys
import csv
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths (experiments/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))

from core.FingerprintParsers import (
    Problem, Heuristic, FingerprintResult,
    PerplexityProbe, ErrorShapeParser, TraceClassifier,
    PrototypeEmbeddingClassifier,
    HeuristicFingerprinter
)
from core.DatasetSplits import get_hds_splits, SPLIT_SEED
from core.TinkerClient import (
    TinkerClient, VisionTinkerClient,
    extract_answer, extract_answer_enhanced, ExtractionResult,
    compute_weighted_loss, DEFAULT_MODEL_NAME, DEFAULT_VISION_MODEL,
    build_tinker_sampling_retry_config,
    get_effective_heuristic_template_metadata,
    get_heuristic_template_profile,
    get_heuristic_template_mode,
    get_heuristic_template_seed,
    set_heuristic_template_mode,
    set_heuristic_template_profile,
    validate_active_heuristic_templates,
)
from core.Logging import tprint


_HEURISTIC_VALUE_TO_NAME = {
    "ones_then_tens": "OT",
    "decomposition": "DD",
    "rounding_compensation": "RC",
}
LOSS_DETECTION_SEMANTICS = "loss_argmin_over_aggregated_losses"
LEXICAL_PROBE_KIND = "lexical_preamble"
EMBEDDING_DETECTION_SEMANTICS = "trace_prototype_cosine_with_style_centroid"
EMBEDDING_SUPPORT_SEMANTICS = "softmax_over_cosine_similarity_with_style"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_PROTOTYPE_SAMPLE_CAP = 256


def heuristic_from_label(label: Optional[str]) -> Optional[Heuristic]:
    """Map a heuristic label or value back to the enum."""
    if not label:
        return None
    if label in Heuristic.__members__:
        return Heuristic[label]
    upper = label.upper()
    if upper in Heuristic.__members__:
        return Heuristic[upper]
    mapped = _HEURISTIC_VALUE_TO_NAME.get(label)
    if mapped and mapped in Heuristic.__members__:
        return Heuristic[mapped]
    return None


def _resolve_loss_based_detection(
    losses: Dict[str, float],
    best_heuristic: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Tuple[Heuristic, float]:
    """Resolve the lowest-loss heuristic and confidence from aggregated losses."""
    finite_losses = {
        heuristic: float(loss)
        for heuristic, loss in losses.items()
        if isinstance(loss, (int, float)) and loss < float("inf")
    }
    if not finite_losses:
        return Heuristic.UNKNOWN, 0.0

    resolved = heuristic_from_label(best_heuristic) if best_heuristic else None
    if resolved is None or resolved == Heuristic.UNKNOWN:
        resolved = min(finite_losses, key=finite_losses.get)
        resolved = Heuristic[resolved]

    if confidence is not None:
        return resolved, float(confidence)

    sorted_losses = sorted(finite_losses.values())
    if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
        gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
        return resolved, min(1.0, gap * 2)
    return resolved, 0.5


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite float when possible."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def _compute_soft_support_mass(
    losses: Dict[str, Any],
    neutral_loss: Optional[Any],
) -> Optional[Dict[str, float]]:
    """Return normalized support mass over heuristics plus neutral, or None if unresolved."""
    resolved_losses: Dict[str, float] = {}
    for heuristic in ("DD", "OT", "RC"):
        parsed = _finite_float(losses.get(heuristic))
        if parsed is None:
            return None
        resolved_losses[heuristic] = parsed

    neutral = _finite_float(neutral_loss)
    if neutral is None:
        return None
    resolved_losses["NEUTRAL"] = neutral

    best_loss = min(resolved_losses.values())
    shifted = {
        name: math.exp(-(loss - best_loss))
        for name, loss in resolved_losses.items()
    }
    denom = sum(shifted.values())
    if denom <= 0:
        return None
    return {name: value / denom for name, value in shifted.items()}


def _get_template_context() -> Dict[str, Any]:
    """Return active heuristic-template metadata for saved artifacts."""
    return get_effective_heuristic_template_metadata(
        profile=get_heuristic_template_profile(),
        mode=get_heuristic_template_mode(),
        seed=get_heuristic_template_seed(),
    )


def _extract_probe_metadata(probe_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize optional probe metadata emitted by runtime scoring paths."""
    probe_result = probe_result or {}
    losses = probe_result.get("losses", {}) or {}
    neutral_loss = probe_result.get("neutral_loss")
    support_mass = _compute_soft_support_mass(losses, neutral_loss)
    probe_resolved = probe_result.get("probe_resolved")
    if probe_resolved is None:
        probe_resolved = support_mass is not None

    return {
        "probe_resolved": bool(probe_resolved),
        "probe_resolution_status": probe_result.get("probe_resolution_status"),
        "probe_resolution_error": probe_result.get("probe_resolution_error"),
        "probe_image_token_count": probe_result.get("probe_image_token_count"),
        "support_mass": support_mass,
    }


def _normalize_embedding_support_mass(payload: Any) -> Optional[Dict[str, float]]:
    """Parse a STYLE-aware embedding support-mass dict when present."""
    if not isinstance(payload, dict):
        return None
    labels = ("DD", "OT", "RC", "STYLE")
    parsed: Dict[str, float] = {}
    for label in labels:
        value = _finite_float(payload.get(label))
        if value is None:
            return None
        parsed[label] = value
    return parsed


def _analyze_trace_outputs(
    trace: Optional[str],
    extraction_result: Optional[ExtractionResult],
    trace_classifier: TraceClassifier,
    embedding_classifier: Optional[PrototypeEmbeddingClassifier],
) -> Dict[str, Any]:
    """Compute trace- and embedding-based heuristic signals from a generated trace."""
    trace_heuristic = None
    trace_confidence = None
    embedding_heuristic = None
    embedding_confidence = None
    embedding_margin = None
    embedding_support_mass = None
    embedding_model = None
    embedding_resolved = None
    embedding_resolution_status = None

    if trace:
        trace_result = trace_classifier.fingerprint(trace)
        if trace_result.heuristic != Heuristic.UNKNOWN:
            trace_heuristic = trace_result.heuristic.value
            trace_confidence = trace_result.confidence

    if embedding_classifier is None:
        return {
            "trace_heuristic": trace_heuristic,
            "trace_confidence": trace_confidence,
            "embedding_heuristic": embedding_heuristic,
            "embedding_confidence": embedding_confidence,
            "embedding_margin": embedding_margin,
            "embedding_support_mass": embedding_support_mass,
            "embedding_model": embedding_model,
            "embedding_resolved": embedding_resolved,
            "embedding_resolution_status": embedding_resolution_status,
        }

    if not trace or not trace.strip():
        embedding_resolved = False
        embedding_resolution_status = "trace_missing"
    elif extraction_result is not None and extraction_result.is_contaminated:
        embedding_resolved = False
        embedding_resolution_status = "contaminated_trace"
    else:
        embedding_result = embedding_classifier.fingerprint(trace)
        details = embedding_result.details or {}
        embedding_model = str(details.get("model") or embedding_classifier.model_name_or_path)
        embedding_resolved = bool(details.get("resolved", embedding_result.heuristic != Heuristic.UNKNOWN))
        embedding_resolution_status = str(details.get("status") or ("ok" if embedding_resolved else "unresolved"))
        embedding_support_mass = _normalize_embedding_support_mass(details.get("support_mass"))
        embedding_margin = _finite_float(details.get("margin"))
        embedding_confidence = embedding_result.confidence if embedding_resolved else None
        if embedding_resolved and embedding_result.heuristic != Heuristic.UNKNOWN:
            embedding_heuristic = embedding_result.heuristic.name

    return {
        "trace_heuristic": trace_heuristic,
        "trace_confidence": trace_confidence,
        "embedding_heuristic": embedding_heuristic,
        "embedding_confidence": embedding_confidence,
        "embedding_margin": embedding_margin,
        "embedding_support_mass": embedding_support_mass,
        "embedding_model": embedding_model,
        "embedding_resolved": embedding_resolved,
        "embedding_resolution_status": embedding_resolution_status,
    }


def _build_fingerprinting_result(
    *,
    row: HDSRow,
    detected_heuristic: Heuristic,
    detection_confidence: float,
    model_answer: Optional[int],
    error_heuristic: Optional[str],
    error_confidence: Optional[float],
    perplexity_losses: Dict[str, float],
    trace: Optional[str],
    trace_signals: Dict[str, Any],
    neutral_loss: Optional[float] = None,
    delta_losses: Optional[Dict[str, float]] = None,
    per_template_losses: Optional[Dict[str, Dict[str, Any]]] = None,
    extraction_result: Optional[ExtractionResult] = None,
    probe_metadata: Optional[Dict[str, Any]] = None,
) -> "FingerprintingResult":
    """Construct a canonical FingerprintingResult from shared runtime pieces."""
    probe_metadata = probe_metadata or _extract_probe_metadata(None)
    return FingerprintingResult(
        hds_id=row.id,
        a=row.a,
        b=row.b,
        product=row.product,
        target_heuristic=row.target_heuristic,
        design_family=row.design_family or row.target_heuristic,
        canonical_target_heuristic=row.canonical_target_heuristic or row.target_heuristic,
        detected_heuristic=detected_heuristic.name if detected_heuristic != Heuristic.UNKNOWN else "UNKNOWN",
        detection_confidence=detection_confidence,
        model_answer=model_answer,
        is_correct=model_answer == row.product if model_answer is not None else False,
        error_delta=model_answer - row.product if model_answer is not None else None,
        error_heuristic=error_heuristic,
        error_confidence=error_confidence,
        perplexity_losses=perplexity_losses,
        trace=trace,
        trace_heuristic=trace_signals.get("trace_heuristic"),
        trace_confidence=trace_signals.get("trace_confidence"),
        neutral_loss=neutral_loss,
        delta_losses=delta_losses,
        per_template_losses=per_template_losses,
        extraction_confidence=extraction_result.confidence if extraction_result else None,
        extraction_strategy=extraction_result.strategy if extraction_result else None,
        is_truncated=extraction_result.is_truncated if extraction_result else None,
        is_contaminated=extraction_result.is_contaminated if extraction_result else None,
        probe_resolved=probe_metadata["probe_resolved"],
        probe_resolution_status=probe_metadata["probe_resolution_status"],
        probe_resolution_error=probe_metadata["probe_resolution_error"],
        probe_image_token_count=probe_metadata["probe_image_token_count"],
        embedding_heuristic=trace_signals.get("embedding_heuristic"),
        embedding_confidence=trace_signals.get("embedding_confidence"),
        embedding_margin=trace_signals.get("embedding_margin"),
        embedding_support_mass=trace_signals.get("embedding_support_mass"),
        embedding_model=trace_signals.get("embedding_model"),
        embedding_resolved=trace_signals.get("embedding_resolved"),
        embedding_resolution_status=trace_signals.get("embedding_resolution_status"),
    )


def _write_run_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    """Persist a run manifest beside saved experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tprint(f"Saved run manifest to {manifest_path}")


def _initialize_embedding_classifier(
    model_name_or_path: Optional[str],
    *,
    cache_dir: Path,
    batch_size: int,
    prototype_sample_cap: int,
) -> Optional[PrototypeEmbeddingClassifier]:
    """Build and preflight the optional embedding detector before long-running generation."""
    if not model_name_or_path:
        return None

    tprint(f"Initializing embedding detector: {model_name_or_path}")
    classifier = PrototypeEmbeddingClassifier(
        model_name_or_path,
        cache_dir=cache_dir,
        batch_size=batch_size,
        prototype_sample_cap=prototype_sample_cap,
    )

    try:
        prototype_pack = classifier.warmup()
    except Exception as exc:
        offline_flag = os.getenv("HF_HUB_OFFLINE", "").strip() or "0"
        raise RuntimeError(
            "Failed to initialize the trace embedding detector before generation. "
            f"Model={model_name_or_path}, HF_HUB_OFFLINE={offline_flag}. "
            "If you intend to run offline, pre-download the embedding model first or "
            "point --embedding-model to a complete local checkpoint. Otherwise rerun "
            "with HF_HUB_OFFLINE=0 for the initial download."
        ) from exc

    prototype_counts = prototype_pack.get("prototype_counts", {})
    total_examples = sum(int(count) for count in prototype_counts.values())
    tprint(
        "Embedding detector ready "
        f"({total_examples} prototype traces across {len(prototype_counts)} labels)"
    )
    return classifier


@dataclass
class HDSRow:
    """A row from HDS/Traps CSV."""
    id: str
    a: int
    b: int
    product: int
    category: str
    notes: str
    design_family: str = ""
    canonical_target_heuristic: str = ""
    canonical_target_margin: float = 0.0
    ot_cost: float = 0.0
    dd_cost: float = 0.0
    rc_cost: float = 0.0
    heuristic_definition_version: str = ""
    target_heuristic: str = ""  # Legacy alias; use design_family for lexical analysis
    ot_score: float = 0.0
    dd_score: float = 0.0
    rc_score: float = 0.0
    split: str = ""  # Optional split field


@dataclass
class FingerprintingResult:
    """Result of fingerprinting a single problem."""
    hds_id: str
    a: int
    b: int
    product: int
    target_heuristic: str  # Legacy alias; mirrors design_family for lexical runs
    detected_heuristic: str
    detection_confidence: float
    model_answer: Optional[int]
    is_correct: bool
    error_delta: Optional[int]
    error_heuristic: Optional[str]
    error_confidence: Optional[float]  # Confidence from ErrorShapeParser
    perplexity_losses: Dict[str, float]
    trace: Optional[str]
    trace_heuristic: Optional[str]
    trace_confidence: Optional[float]  # Confidence from TraceClassifier
    embedding_heuristic: Optional[str] = None
    embedding_confidence: Optional[float] = None
    embedding_margin: Optional[float] = None
    embedding_support_mass: Optional[Dict[str, float]] = None
    embedding_model: Optional[str] = None
    embedding_resolved: Optional[bool] = None
    embedding_resolution_status: Optional[str] = None
    # New fields for template averaging with neutral baseline
    neutral_loss: Optional[float] = None
    delta_losses: Optional[Dict[str, float]] = None
    per_template_losses: Optional[Dict[str, Dict[str, Any]]] = None  # template_id -> {"prompt": ..., "loss": ...}
    # Extraction metadata for output quality tracking
    extraction_confidence: Optional[float] = None  # 0.0-1.0 confidence in answer extraction
    extraction_strategy: Optional[str] = None  # Strategy used: "boxed", "answer_marker", etc.
    is_truncated: Optional[bool] = None  # Whether model output was truncated
    is_contaminated: Optional[bool] = None  # Whether trace appears to solve wrong problem
    design_family: str = ""
    canonical_target_heuristic: str = ""
    probe_kind: str = LEXICAL_PROBE_KIND
    probe_resolved: Optional[bool] = None
    probe_resolution_status: Optional[str] = None
    probe_resolution_error: Optional[str] = None
    probe_image_token_count: Optional[int] = None


def load_hds(path: Path) -> List[HDSRow]:
    """Load HDS or Traps from CSV (auto-detects format)."""
    def _float_value(raw: Optional[str], default: float = 0.0) -> float:
        if raw in (None, ""):
            return default
        return float(raw)

    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_cost_model_fields = 'canonical_target_heuristic' in row
            has_score_fields = 'ot_score' in row

            if has_cost_model_fields:
                design_family = row.get('design_family') or row.get('target_heuristic', '')
                canonical_target = row.get('canonical_target_heuristic') or row.get('target_heuristic', '')
                ot_cost = _float_value(row.get('ot_cost'))
                dd_cost = _float_value(row.get('dd_cost'))
                rc_cost = _float_value(row.get('rc_cost'))
                cost_values = {'OT': ot_cost, 'DD': dd_cost, 'RC': rc_cost}
                max_cost = max(cost_values.values())
                min_cost = min(cost_values.values())
                span = max(max_cost - min_cost, 1e-6)
                pseudo_scores = {
                    heuristic: (max_cost - cost) / span
                    for heuristic, cost in cost_values.items()
                }
                rows.append(HDSRow(
                    id=row['id'],
                    a=int(row['a']),
                    b=int(row['b']),
                    product=int(row['product']),
                    category=row.get('category') or row.get('trap_type', 'unknown'),
                    notes=row.get('notes', ''),
                    design_family=design_family,
                    canonical_target_heuristic=canonical_target,
                    canonical_target_margin=_float_value(row.get('canonical_target_margin')),
                    ot_cost=ot_cost,
                    dd_cost=dd_cost,
                    rc_cost=rc_cost,
                    heuristic_definition_version=row.get('heuristic_definition_version', ''),
                    target_heuristic=design_family,
                    ot_score=_float_value(row.get('ot_score'), pseudo_scores['OT']),
                    dd_score=_float_value(row.get('dd_score'), pseudo_scores['DD']),
                    rc_score=_float_value(row.get('rc_score'), pseudo_scores['RC']),
                    split=row.get('split', '')
                ))
            elif has_score_fields:
                rows.append(HDSRow(
                    id=row['id'],
                    a=int(row['a']),
                    b=int(row['b']),
                    product=int(row['product']),
                    category=row['category'],
                    notes=row['notes'],
                    design_family=row['target_heuristic'],
                    canonical_target_heuristic=row['target_heuristic'],
                    heuristic_definition_version='legacy_applicability_v1',
                    target_heuristic=row['target_heuristic'],
                    ot_score=float(row['ot_score']),
                    dd_score=float(row['dd_score']),
                    rc_score=float(row['rc_score']),
                    split=row.get('split', '')
                ))
            else:
                rows.append(HDSRow(
                    id=row['id'],
                    a=int(row['a']),
                    b=int(row['b']),
                    product=int(row['product']),
                    category=row.get('trap_type', 'unknown'),
                    notes=row.get('notes', ''),
                    design_family=row.get('design_family') or row['target_heuristic'],
                    canonical_target_heuristic=row.get('canonical_target_heuristic') or row['target_heuristic'],
                    heuristic_definition_version=row.get('heuristic_definition_version', 'legacy_applicability_v1'),
                    target_heuristic=row.get('design_family') or row['target_heuristic'],
                    ot_score=0.5,
                    dd_score=0.5,
                    rc_score=0.5,
                    split=row.get('split', '')
                ))
    return rows


def select_rows_for_split(
    all_rows: List[HDSRow],
    split: str,
    dataset_name: str,
) -> List[HDSRow]:
    """Select rows for a requested split, preferring explicit CSV split columns."""
    if split == "all":
        return all_rows

    if any(getattr(row, "split", "").strip() for row in all_rows):
        return [row for row in all_rows if getattr(row, "split", "").strip() == split]

    if dataset_name.upper() in {"HDS", "HDSV2"}:
        hds_dicts = [{"id": r.id, "a": r.a, "b": r.b, "product": r.product,
                      "target_heuristic": r.target_heuristic, "ot_score": r.ot_score,
                      "dd_score": r.dd_score, "rc_score": r.rc_score,
                      "category": r.category, "notes": r.notes} for r in all_rows]
        splits = get_hds_splits(hds_dicts)
        return [HDSRow(**d) for d in splits.get(split, [])]

    return [r for r in all_rows if getattr(r, 'split', split) == split]


def resolve_images_dir(
    dataset_name: str,
    images_dir_arg: Optional[str] = None,
    has_custom_csv: bool = False,
) -> Optional[Path]:
    """Resolve the image directory for image-modality fingerprinting."""
    if images_dir_arg:
        return Path(images_dir_arg)
    return REPO_ROOT / "SavedData" / f"{dataset_name}Images"


class TinkerFingerprinter:
    """
    Fingerprinter that uses Tinker API for perplexity probes and generation.

    Uses VisionTinkerClient which supports both text-only and image+text inputs.
    This allows using the same VLM model for both modalities.
    """
    supports_multi_problem_batching = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_classifier: Optional[PrototypeEmbeddingClassifier] = None,
    ):
        """Initialize with Tinker API via VisionTinkerClient (which supports text-only)."""
        # Use VisionTinkerClient for unified text/image support
        self.modality = "text"
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.client = VisionTinkerClient(
            model_name=self.model_name,
            api_key=api_key,
            verbose=True
        )

        # Initialize parsers
        self.error_parser = ErrorShapeParser()
        self.trace_classifier = TraceClassifier()
        self.embedding_classifier = embedding_classifier

    def compute_perplexity(self, prompt: str) -> float:
        """Compute perplexity/loss for a prompt using forward pass."""
        return self.client.compute_perplexity(prompt)

    def compute_heuristic_losses_multi(
        self,
        rows: List[HDSRow],
        batch_size: int = 30,
        include_neutral: bool = True
    ) -> List[Dict[str, Any]]:
        """Compute batched heuristic losses for text modality."""
        problems = [(row.a, row.b) for row in rows]
        return self.client.compute_heuristic_losses_multi(
            problems,
            batch_size=batch_size,
            include_neutral=include_neutral
        )

    def generate_answer(
        self,
        a: int,
        b: int,
        with_reasoning: bool = False,
        hds_id: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """Generate model's answer for a multiplication problem (hds_id ignored)."""
        _ = hds_id
        return self.client.generate(a, b, with_reasoning=with_reasoning)

    def fingerprint_problem(self, row: HDSRow, use_perplexity: bool = True,
                            use_generation: bool = True) -> FingerprintingResult:
        """
        Run full fingerprinting on a single HDS problem.

        Uses template averaging with neutral baseline for more robust
        perplexity-based heuristic detection.
        """
        problem = Problem(row.a, row.b)

        # Initialize results
        perplexity_losses = {}
        neutral_loss = None
        delta_losses = None
        per_template_losses = None
        detected_heuristic = Heuristic.UNKNOWN
        detection_confidence = 0.0
        model_answer = None
        trace = None
        error_heuristic = None
        error_confidence = None

        # Method 1: Perplexity probe using template averaging with baseline
        if use_perplexity:
            probe_result = self.client.compute_heuristic_losses_with_baseline(row.a, row.b)

            # Extract results from the enriched probe result
            perplexity_losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')
            per_template_losses = probe_result.get('per_template_losses')
            probe_metadata = _extract_probe_metadata(probe_result)

            detected_heuristic, detection_confidence = _resolve_loss_based_detection(
                perplexity_losses,
                best_heuristic=probe_result.get("best_heuristic"),
                confidence=probe_result.get("confidence"),
            )
        else:
            probe_metadata = _extract_probe_metadata(None)

        # Method 2: Generate answer and analyze
        extraction_result = None
        if use_generation:
            _, trace = self.generate_answer(row.a, row.b, with_reasoning=True)

            # Use enhanced extraction with operand validation for better accuracy
            if trace:
                extraction_result = extract_answer_enhanced(trace, a=row.a, b=row.b)
                model_answer = extraction_result.answer

            # Analyze error if wrong using ErrorShapeParser
            if model_answer is not None and model_answer != row.product:
                error_result = self.error_parser.fingerprint(problem, model_answer)
                if error_result.heuristic != Heuristic.UNKNOWN:
                    error_heuristic = error_result.heuristic.value
                    error_confidence = error_result.confidence

            # Analyze trace if available using TraceClassifier
        trace_signals = _analyze_trace_outputs(
            trace,
            extraction_result,
            self.trace_classifier,
            self.embedding_classifier,
        )

        return _build_fingerprinting_result(
            row=row,
            detected_heuristic=detected_heuristic,
            detection_confidence=detection_confidence,
            model_answer=model_answer,
            error_heuristic=error_heuristic,
            error_confidence=error_confidence,
            perplexity_losses=perplexity_losses,
            trace=trace,
            trace_signals=trace_signals,
            neutral_loss=neutral_loss,
            delta_losses=delta_losses,
            per_template_losses=per_template_losses,
            extraction_result=extraction_result,
            probe_metadata=probe_metadata,
        )


class ImageFingerprinter:
    """
    Fingerprinter for image modality using VisionTinkerClient.

    Uses perplexity probes on image + text continuation pairs.
    """
    supports_multi_problem_batching = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        images_dir: Optional[Path] = None,
        embedding_classifier: Optional[PrototypeEmbeddingClassifier] = None,
    ):
        """Initialize with Vision Tinker API."""
        self.modality = "image"
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.client = VisionTinkerClient(
            model_name=self.model_name,
            api_key=api_key,
            verbose=True
        )
        self.images_dir = images_dir or (REPO_ROOT / "SavedData" / "HDSImages")
        self.traps_images_dir = REPO_ROOT / "SavedData" / "TrapsImages"

        # Initialize parsers
        self.error_parser = ErrorShapeParser()
        self.trace_classifier = TraceClassifier()
        self.embedding_classifier = embedding_classifier

    def _get_image_path(self, hds_id: str) -> Path:
        """Get image path for a problem ID."""
        # Try HDS images first, then Traps
        hds_path = self.images_dir / f"{hds_id}.png"
        if hds_path.exists():
            return hds_path
        traps_path = self.traps_images_dir / f"{hds_id}.png"
        if traps_path.exists():
            return traps_path
        return hds_path  # Return HDS path even if missing (will fail gracefully)

    def generate_answer(
        self,
        a: int,
        b: int,
        with_reasoning: bool = False,
        hds_id: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """Generate model's answer for a multiplication problem from its image (requires hds_id)."""
        if not hds_id:
            return None, None
        image_path = self._get_image_path(hds_id)
        if not image_path.exists():
            return None, None
        return self.client.generate_with_image(image_path, a, b, with_reasoning=with_reasoning)

    def compute_heuristic_losses_multi(
        self,
        rows: List[HDSRow],
        batch_size: int = 30,
        include_neutral: bool = True
    ) -> List[Dict[str, Any]]:
        """Compute batched heuristic losses for image modality."""
        problems = [(self._get_image_path(row.id), row.a, row.b) for row in rows]
        return self.client.compute_heuristic_losses_multi_image(
            problems,
            batch_size=batch_size,
            include_neutral=include_neutral
        )

    def fingerprint_problem(self, row: HDSRow, use_perplexity: bool = True,
                            use_generation: bool = True) -> FingerprintingResult:
        """
        Run image-based fingerprinting on a single HDS problem.

        Now supports generation via generate_with_image() to get actual model answers.
        """
        # Get image path
        image_path = self._get_image_path(row.id)

        # Initialize results
        perplexity_losses = {}
        per_template_losses = None
        detected_heuristic = Heuristic.UNKNOWN
        detection_confidence = 0.0
        model_answer = None
        trace = None
        error_heuristic = None
        error_confidence = None

        # Method 1: Perplexity probe with image (with neutral baseline)
        neutral_loss = None
        delta_losses = None
        if use_perplexity and image_path.exists():
            probe_result = self.client.compute_heuristic_losses_with_image_batched(
                image_path, row.a, row.b, include_neutral=True
            )

            # Extract from new return format
            perplexity_losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')

            # Construct per_template_losses for detailed logging
            per_template_losses = probe_result.get('per_template_losses')
            probe_metadata = _extract_probe_metadata(probe_result)

            detected_heuristic, detection_confidence = _resolve_loss_based_detection(
                perplexity_losses,
                best_heuristic=probe_result.get("best_heuristic"),
                confidence=probe_result.get("confidence"),
            )
        elif not image_path.exists():
            tprint(f"    Warning: Image not found: {image_path}")
            probe_metadata = _extract_probe_metadata(
                {
                    "losses": perplexity_losses,
                    "neutral_loss": neutral_loss,
                    "probe_resolved": False,
                    "probe_resolution_status": "image_missing",
                    "probe_resolution_error": f"Image not found: {image_path}",
                }
            )
        else:
            probe_metadata = _extract_probe_metadata(None)

        # Method 2: Generate answer from image (with retry for empty traces)
        extraction_result = None
        if use_generation and image_path.exists():
            max_retries = 2
            for attempt in range(max_retries):
                _, trace = self.client.generate_with_image(
                    image_path, row.a, row.b, with_reasoning=True
                )

                # Check for empty/null trace and retry
                if trace and trace.strip() and trace.strip() != "<|im_end|>":
                    break
                elif attempt < max_retries - 1:
                    tprint(f"    Retrying (attempt {attempt + 2}/{max_retries}): empty trace")

            # Use enhanced extraction with operand validation for better accuracy
            if trace:
                extraction_result = extract_answer_enhanced(trace, a=row.a, b=row.b)
                model_answer = extraction_result.answer

                # Log contamination warning if detected
                if extraction_result.is_contaminated:
                    tprint(f"    WARNING: Trace appears contaminated (operands {row.a}, {row.b} not found)")

            # Analyze error if wrong using ErrorShapeParser
            if model_answer is not None and model_answer != row.product:
                problem = Problem(row.a, row.b)
                error_result = self.error_parser.fingerprint(problem, model_answer)
                if error_result.heuristic != Heuristic.UNKNOWN:
                    error_heuristic = error_result.heuristic.value
                    error_confidence = error_result.confidence

        trace_signals = _analyze_trace_outputs(
            trace,
            extraction_result,
            self.trace_classifier,
            self.embedding_classifier,
        )

        return _build_fingerprinting_result(
            row=row,
            detected_heuristic=detected_heuristic,
            detection_confidence=detection_confidence,
            model_answer=model_answer,
            error_heuristic=error_heuristic,
            error_confidence=error_confidence,
            perplexity_losses=perplexity_losses,
            trace=trace,
            trace_signals=trace_signals,
            neutral_loss=neutral_loss,
            delta_losses=delta_losses,
            per_template_losses=per_template_losses,
            extraction_result=extraction_result,
            probe_metadata=probe_metadata,
        )


class MockFingerprinter:
    """
    supports_multi_problem_batching = False
    Mock fingerprinter for testing without API calls.
    Uses heuristic scores to simulate model behavior.
    """

    def __init__(self, embedding_classifier: Optional[PrototypeEmbeddingClassifier] = None):
        self.modality = "text"
        self.error_parser = ErrorShapeParser()
        self.trace_classifier = TraceClassifier()
        self.embedding_classifier = embedding_classifier

    def fingerprint_problem(self, row: HDSRow, **kwargs) -> FingerprintingResult:
        """Simulate fingerprinting based on heuristic scores."""
        import random

        # Simulate perplexity based on heuristic scores (lower score = lower loss)
        # Invert scores since high applicability = low perplexity
        base_loss = 2.0
        perplexity_losses = {
            "OT": base_loss - row.ot_score * 0.5 + random.gauss(0, 0.1),
            "DD": base_loss - row.dd_score * 0.5 + random.gauss(0, 0.1),
            "RC": base_loss - row.rc_score * 0.5 + random.gauss(0, 0.1),
        }

        # Detected heuristic is the one with lowest loss
        detected = min(perplexity_losses, key=lambda h: perplexity_losses[h])

        # Simulate accuracy (higher for problems matching detected heuristic)
        if detected == row.target_heuristic:
            is_correct = random.random() < 0.85
        else:
            is_correct = random.random() < 0.65

        model_answer = row.product if is_correct else row.product + random.choice([10, -10, 1, -1, 100])

        return _build_fingerprinting_result(
            row=row,
            detected_heuristic=Heuristic[detected],
            detection_confidence=0.7 + random.gauss(0, 0.1),
            model_answer=model_answer,
            error_heuristic=None,
            error_confidence=None,
            perplexity_losses=perplexity_losses,
            trace=None,
            trace_signals=_analyze_trace_outputs(
                None,
                None,
                self.trace_classifier,
                self.embedding_classifier,
            ),
        )


def run_fingerprinting(
    hds: List[HDSRow],
    fingerprinter,
    verbose: bool = True,
    max_workers: int = 1,
    batch_size: int = 1,
    output_dir: Optional[Path] = None,
    use_async_backend: bool = True,
    concurrency: int = 4,
    score_max_in_flight: int = 4,
) -> List[FingerprintingResult]:
    """
    Run fingerprinting on all HDS problems.

    By default this delegates to the async pipeline (sample_async + forward_async
    with bounded concurrency) to avoid idle Tinker clock cycles.

    Args:
        hds: List of HDS problems to fingerprint
        fingerprinter: Fingerprinter instance
        verbose: Print progress messages
        max_workers: Number of parallel workers (1 = sequential)
        batch_size: Batch size for perplexity probes (1 = no batching, >1 = multi-problem batching)
        output_dir: If provided, stream results to JSONL as they complete (crash-safe)

    Returns:
        List of FingerprintingResult objects
    """
    if use_async_backend:
        return run_fingerprinting_async(
            hds,
            fingerprinter,
            verbose=verbose,
            batch_size=batch_size,
            output_dir=output_dir,
            concurrency=concurrency,
            score_max_in_flight=score_max_in_flight
        )

    # Use multi-problem batching when the fingerprinter supports it.
    supports_multi = getattr(fingerprinter, "supports_multi_problem_batching", True)
    if batch_size > 1 and supports_multi and hasattr(fingerprinter, 'compute_heuristic_losses_multi'):
        # Use parallel generation if max_workers > 1
        if max_workers > 1:
            return run_fingerprinting_batched_parallel(
                hds, fingerprinter, verbose, batch_size, output_dir, max_workers
            )
        return run_fingerprinting_batched(hds, fingerprinter, verbose, batch_size, output_dir)
    if batch_size > 1 and not supports_multi and verbose:
        tprint("  Multi-problem batching not supported; running per-problem.")

    # Set up streaming writer if output_dir provided
    writer = StreamingResultWriter(output_dir) if output_dir else None

    try:
        if max_workers <= 1:
            # Sequential processing (original behavior)
            results = []
            for i, row in enumerate(hds):
                if verbose:
                    tprint(f"  [{i+1}/{len(hds)}] {row.id}: {row.a} × {row.b} = {row.product} (target: {row.target_heuristic})")

                result = fingerprinter.fingerprint_problem(row)
                results.append(result)

                # Stream result immediately
                if writer:
                    writer.write(result)

                if verbose:
                    status = "✓" if result.is_correct else "✗"
                    match = "=" if result.detected_heuristic.upper() == row.target_heuristic else "≠"
                    tprint(f"         {status} answer={result.model_answer}, detected={result.detected_heuristic} {match} target")
            return results

        # Parallel processing
        if verbose:
            tprint(f"  Running with {max_workers} parallel workers...")

        results = [None] * len(hds)
        completed = 0

        def process_one(idx_row):
            idx, row = idx_row
            return idx, fingerprinter.fingerprint_problem(row)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_one, (i, row)): i for i, row in enumerate(hds)}

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1

                # Stream result immediately
                if writer:
                    writer.write(result)

                if verbose and completed % max(1, len(hds) // 20) == 0:
                    row = hds[idx]
                    status = "✓" if result.is_correct else "✗"
                    match = "=" if result.detected_heuristic.upper() == row.target_heuristic else "≠"
                    tprint(f"  [{completed}/{len(hds)}] {row.id}: {status} detected={result.detected_heuristic} {match} target")

        return results
    finally:
        if writer:
            writer.close()


def run_fingerprinting_batched(
    hds: List[HDSRow],
    fingerprinter,
    verbose: bool = True,
    batch_size: int = 30,
    output_dir: Optional[Path] = None
) -> List[FingerprintingResult]:
    """
    Run fingerprinting with multi-problem batching for perplexity probes.

    Batches perplexity probes across multiple problems into single API calls,
    then runs generation per-problem (generation can't be easily batched).

    Now includes neutral baseline computation for delta loss analysis.

    Args:
        hds: List of HDS problems to fingerprint
        fingerprinter: TinkerFingerprinter instance (must have client attribute)
        verbose: Print progress messages
        batch_size: Number of datums per batch (problems × templates including neutral)
        output_dir: If provided, stream results to JSONL as they complete (crash-safe)

    Returns:
        List of FingerprintingResult objects
    """
    if verbose:
        tprint(f"  Running with multi-problem batching (batch_size={batch_size})...")

    # Set up streaming writer if output_dir provided
    writer = StreamingResultWriter(output_dir) if output_dir else None

    try:
        # Step 1: Batch all perplexity probes (now includes neutral baseline)
        if verbose:
            tprint(f"  Computing perplexity probes for {len(hds)} problems (with neutral baseline)...")

        all_probe_results = fingerprinter.compute_heuristic_losses_multi(
            hds, batch_size=batch_size, include_neutral=True
        )

        if verbose:
            tprint(f"  Perplexity probes complete. Running generation for each problem...")

        # Step 2: Run generation and build results for each problem
        results = []
        for i, (row, probe_result) in enumerate(zip(hds, all_probe_results)):
            if verbose and (i + 1) % max(1, len(hds) // 10) == 0:
                tprint(f"  [{i+1}/{len(hds)}] Generating answers...")

            # Extract losses from new return format
            perplexity_losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')
            per_template_losses = probe_result.get('per_template_losses')
            probe_metadata = _extract_probe_metadata(probe_result)

            detected_heuristic, detection_confidence = _resolve_loss_based_detection(
                perplexity_losses,
                best_heuristic=probe_result.get("best_heuristic"),
                confidence=probe_result.get("confidence"),
            )

            # Generate answer
            _, trace = fingerprinter.generate_answer(
                row.a,
                row.b,
                with_reasoning=True,
                hds_id=row.id
            )

            # Use enhanced extraction with operand validation for better accuracy
            extraction_result = None
            if trace:
                extraction_result = extract_answer_enhanced(trace, a=row.a, b=row.b)
                model_answer = extraction_result.answer
            else:
                model_answer = None

            # Analyze error if wrong
            error_heuristic = None
            error_confidence = None
            if model_answer is not None and model_answer != row.product:
                problem = Problem(row.a, row.b)
                error_result = fingerprinter.error_parser.fingerprint(problem, model_answer)
                if error_result.heuristic != Heuristic.UNKNOWN:
                    error_heuristic = error_result.heuristic.value
                    error_confidence = error_result.confidence

            trace_signals = _analyze_trace_outputs(
                trace,
                extraction_result,
                fingerprinter.trace_classifier,
                getattr(fingerprinter, "embedding_classifier", None),
            )

            result = _build_fingerprinting_result(
                row=row,
                detected_heuristic=detected_heuristic,
                detection_confidence=detection_confidence,
                model_answer=model_answer,
                error_heuristic=error_heuristic,
                error_confidence=error_confidence,
                perplexity_losses=perplexity_losses,
                trace=trace,
                trace_signals=trace_signals,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                per_template_losses=per_template_losses,
                extraction_result=extraction_result,
                probe_metadata=probe_metadata,
            )
            results.append(result)

            # Stream result immediately
            if writer:
                writer.write(result)

        return results
    finally:
        if writer:
            writer.close()


def run_fingerprinting_batched_parallel(
    hds: List[HDSRow],
    fingerprinter,
    verbose: bool = True,
    batch_size: int = 30,
    output_dir: Optional[Path] = None,
    max_workers: int = 4
) -> List[FingerprintingResult]:
    """
    Run fingerprinting with batched perplexity AND parallel generation.

    This combines multi-problem batching for perplexity probes with
    parallel generation using thread-local sampling clients.

    Args:
        hds: List of HDS problems to fingerprint
        fingerprinter: TinkerFingerprinter instance (must have client attribute)
        verbose: Print progress messages
        batch_size: Number of datums per batch for perplexity probes
        output_dir: If provided, stream results to JSONL as they complete
        max_workers: Number of parallel generation workers (default: 4)

    Returns:
        List of FingerprintingResult objects
    """
    if verbose:
        tprint(f"  Running with batched perplexity + {max_workers} parallel generation workers...")

    # Thread-local storage for sampling clients
    _thread_local = threading.local()

    def _get_thread_sampler():
        """Get or create a sampling client for this thread."""
        if not hasattr(_thread_local, 'sampler'):
            # Create a truly independent sampling client for this thread
            # Using unique thread ID as adapter name to avoid cache collisions
            thread_id = threading.get_ident()
            adapter_name = f"base_parallel_{thread_id}"
            if fingerprinter.client.verbose:
                tprint(f"    Creating sampling client for thread {thread_id}...")
            _thread_local.sampler = fingerprinter.client.get_sampling_client(adapter_name)
        return _thread_local.sampler

    def _generate_for_problem(args):
        """Generate answer for a single problem using thread-local sampler."""
        idx, row = args
        try:
            sampler = _get_thread_sampler()

            # Build generation input
            prompt_text = f"What is {row.a} × {row.b}? Show your work step by step, then give the final answer."
            model_input = fingerprinter.client.build_text_generation_input(prompt_text)
            if model_input is None:
                return idx, None, "model_input is None"

            # Sampling params
            sampling_params = fingerprinter.client._tinker.types.SamplingParams(
                max_tokens=2048,
                temperature=0.0
            )

            # Generate using thread-local sampler
            from core.TinkerClient import _call_with_timeout
            future = sampler.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            )
            result = _call_with_timeout(future, operation="parallel generation")

            # Extract text with better error handling
            text = None
            if hasattr(result, 'samples') and len(result.samples) > 0:
                sample = result.samples[0]
                if hasattr(sample, 'tokens'):
                    text = fingerprinter.client._tokenizer.decode(sample.tokens)
                elif hasattr(sample, 'token_ids'):
                    text = fingerprinter.client._tokenizer.decode(sample.token_ids)
                elif hasattr(sample, 'text'):
                    text = sample.text
                else:
                    text = str(sample)
            elif hasattr(result, 'sequences') and len(result.sequences) > 0:
                output_tokens = result.sequences[0].tokens
                text = fingerprinter.client._tokenizer.decode(output_tokens)
            elif hasattr(result, 'completions'):
                text = result.completions[0]
            elif isinstance(result, list) and len(result) > 0:
                text = str(result[0])
            else:
                # Last resort - try to convert to string
                text = str(result)
                if verbose:
                    tprint(f"    Warning: Unknown result format for problem {idx}: {type(result)}")

            return idx, text, None
        except Exception as e:
            import traceback
            return idx, None, f"{str(e)}\n{traceback.format_exc()}"

    # Set up streaming writer if output_dir provided
    writer = StreamingResultWriter(output_dir) if output_dir else None

    try:
        # Step 1: Batch all perplexity probes (same as before)
        if verbose:
            tprint(f"  Computing perplexity probes for {len(hds)} problems (batched)...")

        all_probe_results = fingerprinter.compute_heuristic_losses_multi(
            hds, batch_size=batch_size, include_neutral=True
        )

        if verbose:
            tprint(f"  Perplexity complete. Running {len(hds)} generations with {max_workers} workers...")

        # Step 2: Run all generations in parallel
        generation_results = [None] * len(hds)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generate_for_problem, (i, row)): i
                      for i, row in enumerate(hds)}

            for future in as_completed(futures):
                idx, trace, error = future.result()
                generation_results[idx] = (trace, error)
                completed += 1
                if verbose and completed % max(1, len(hds) // 5) == 0:
                    tprint(f"    [{completed}/{len(hds)}] generations complete...")

        # Log any generation errors
        errors = [(i, err) for i, (_, err) in enumerate(generation_results) if err]
        if errors and verbose:
            tprint(f"  Warning: {len(errors)} generation errors:")
            for i, err in errors[:3]:  # Show first 3 errors
                tprint(f"    Problem {i}: {err[:200] if err else 'None'}")

        if verbose:
            tprint(f"  All generations complete. Building results...")

        # Step 3: Build results combining probe + generation data
        results = []
        for i, (row, probe_result) in enumerate(zip(hds, all_probe_results)):
            trace, gen_error = generation_results[i]

            # Extract losses from probe result
            perplexity_losses = probe_result.get('losses', {})
            neutral_loss = probe_result.get('neutral_loss')
            delta_losses = probe_result.get('delta_losses')
            per_template_losses = probe_result.get('per_template_losses')
            probe_metadata = _extract_probe_metadata(probe_result)

            detected_heuristic, detection_confidence = _resolve_loss_based_detection(
                perplexity_losses,
                best_heuristic=probe_result.get("best_heuristic"),
                confidence=probe_result.get("confidence"),
            )

            # Process generation result
            extraction_result = None
            model_answer = None
            if trace:
                extraction_result = extract_answer_enhanced(trace, a=row.a, b=row.b)
                model_answer = extraction_result.answer

            # Analyze error if wrong
            error_heuristic = None
            error_confidence = None
            if model_answer is not None and model_answer != row.product:
                problem = Problem(row.a, row.b)
                error_result = fingerprinter.error_parser.fingerprint(problem, model_answer)
                if error_result.heuristic != Heuristic.UNKNOWN:
                    error_heuristic = error_result.heuristic.value
                    error_confidence = error_result.confidence

            trace_signals = _analyze_trace_outputs(
                trace,
                extraction_result,
                fingerprinter.trace_classifier,
                getattr(fingerprinter, "embedding_classifier", None),
            )

            result = _build_fingerprinting_result(
                row=row,
                detected_heuristic=detected_heuristic,
                detection_confidence=detection_confidence,
                model_answer=model_answer,
                error_heuristic=error_heuristic,
                error_confidence=error_confidence,
                perplexity_losses=perplexity_losses,
                trace=trace,
                trace_signals=trace_signals,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                per_template_losses=per_template_losses,
                extraction_result=extraction_result,
                probe_metadata=probe_metadata,
            )
            results.append(result)

            # Stream result immediately
            if writer:
                writer.write(result)

        return results
    finally:
        if writer:
            writer.close()


def run_fingerprinting_async(
    hds: List[HDSRow],
    fingerprinter,
    verbose: bool = True,
    batch_size: int = 30,
    output_dir: Optional[Path] = None,
    concurrency: int = 4,
    score_max_in_flight: int = 4
) -> List[FingerprintingResult]:
    """
    Run fingerprinting using async generation for maximum throughput.

    Uses asyncio with sample_async() for concurrent generation, which is
    ~35% faster than ThreadPoolExecutor-based parallel workers.

    Args:
        hds: List of HDS problems to fingerprint
        fingerprinter: TinkerFingerprinter instance
        verbose: Whether to print progress
        batch_size: Batch size for perplexity computation
        output_dir: Optional output directory for streaming results
        concurrency: Number of concurrent async generations (default: 4)
        score_max_in_flight: Max forward() batches in flight for perplexity scoring

    Returns:
        List of FingerprintingResult for each problem
    """
    return asyncio.run(_run_fingerprinting_async_impl(
        hds, fingerprinter, verbose, batch_size, output_dir, concurrency, score_max_in_flight
    ))


async def _run_fingerprinting_async_impl(
    hds: List[HDSRow],
    fingerprinter,
    verbose: bool,
    batch_size: int,
    output_dir: Optional[Path],
    concurrency: int,
    score_max_in_flight: int
) -> List[FingerprintingResult]:
    """Async implementation of fingerprinting.

    Uses direct sample_async() calls for maximum performance.
    This approach is ~35% faster than ThreadPoolExecutor-based parallel workers.
    """
    from core.TinkerClient import extract_answer_enhanced

    results = []
    total = len(hds)
    TIMEOUT_SECONDS = 120  # 2 minute timeout per generation

    # Setup streaming writer if output_dir provided
    writer = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = StreamingResultWriter(output_dir)

    try:
        if verbose:
            tprint(f"  Running async fingerprinting with concurrency={concurrency}...")
            tprint(f"  Timeout per generation: {TIMEOUT_SECONDS}s")

        # Get references to client internals (matching working test script structure)
        client = fingerprinter.client
        service_client = client._service_client
        tokenizer = client._tokenizer
        tinker = client._tinker

        # Create a single sampling client for all async calls (using async version)
        if verbose:
            tprint(f"  Creating async sampling client...")
        sampling_client = await service_client.create_sampling_client_async(
            base_model=client.model_name,
            retry_config=build_tinker_sampling_retry_config(),
        )
        if verbose:
            tprint(f"  Sampling client created successfully")

        modality = getattr(fingerprinter, "modality", "text")

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = [0]  # Use list for mutable counter in closure

        if modality == "image":
            async def _generate_one_async_image(
                row: HDSRow,
                idx: int,
            ) -> Tuple[int, Optional[int], Optional[str], Optional[ExtractionResult], str]:
                """Generate for one problem using image-aware async sampling."""
                async with semaphore:
                    image_path = fingerprinter._get_image_path(row.id)
                    if not image_path.exists():
                        if verbose:
                            tprint(f"    Warning: Image not found: {image_path}")
                        return idx, None, None, None, str(image_path)

                    try:
                        answer, text = await asyncio.wait_for(
                            client.generate_with_image_async(
                                image_path=image_path,
                                a=row.a,
                                b=row.b,
                                sampler=sampling_client,
                                with_reasoning=True,
                            ),
                            timeout=TIMEOUT_SECONDS,
                        )
                        extraction = (
                            extract_answer_enhanced(text, a=row.a, b=row.b)
                            if text else None
                        )

                        completed_count[0] += 1
                        if verbose and completed_count[0] % max(1, total // 5) == 0:
                            tprint(f"    Image generation progress: {completed_count[0]}/{total}")

                        return idx, answer, text, extraction, str(image_path)

                    except asyncio.TimeoutError:
                        if verbose:
                            tprint(f"    Timeout for {row.id} after {TIMEOUT_SECONDS}s")
                        return idx, None, None, None, str(image_path)
                    except Exception as e:
                        if verbose:
                            tprint(
                                f"    Warning: Async image generation failed for {row.id}: "
                                f"{type(e).__name__}: {e}"
                            )
                        return idx, None, None, None, str(image_path)

            if verbose:
                tprint(f"  Submitting {total} async image generation tasks...")

            generation_results = await asyncio.gather(
                *[_generate_one_async_image(row, i) for i, row in enumerate(hds)],
                return_exceptions=True,
            )

            if verbose:
                tprint(f"  Processing {len(generation_results)} image generation results...")

            gen_map: Dict[int, Tuple[Optional[int], Optional[str], Optional[ExtractionResult], str]] = {}
            for r in generation_results:
                if isinstance(r, Exception):
                    continue
                idx, answer, trace, extraction_result, image_path = r
                gen_map[idx] = (answer, trace, extraction_result, image_path)

            if verbose:
                tprint(
                    f"  Computing image perplexity probes in batches of {batch_size} "
                    f"(max_in_flight={score_max_in_flight})..."
                )

            perplexity_map: Dict[int, Dict[str, Any]] = {}
            problems = [(fingerprinter._get_image_path(row.id), row.a, row.b) for row in hds]
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = problems[batch_start:batch_end]
                batch_results = await fingerprinter.client.compute_heuristic_losses_multi_image_async(
                    batch,
                    batch_size=batch_size,
                    include_neutral=True,
                    max_in_flight=score_max_in_flight,
                )

                for local_idx, res in enumerate(batch_results):
                    perplexity_map[batch_start + local_idx] = res

                if verbose and batch_end % max(1, total // 5) == 0:
                    tprint(f"    Image perplexity: {batch_end}/{total} problems processed")

            if verbose:
                tprint("  Building final image results...")

            for i, row in enumerate(hds):
                model_answer, trace, extraction_result, _ = gen_map.get(
                    i,
                    (None, None, None, str(fingerprinter._get_image_path(row.id))),
                )

                perp_data = perplexity_map.get(i, {})
                perplexity_losses = perp_data.get("losses", {})
                neutral_loss = perp_data.get("neutral_loss")
                delta_losses = perp_data.get("delta_losses")
                per_template_losses = perp_data.get("per_template_losses", {})
                probe_metadata = _extract_probe_metadata(perp_data)
                detected_heuristic, detection_confidence = _resolve_loss_based_detection(
                    perplexity_losses,
                    best_heuristic=perp_data.get("best_heuristic"),
                    confidence=perp_data.get("confidence"),
                )

                error_heuristic = None
                error_confidence = None
                if model_answer is not None and model_answer != row.product:
                    problem = Problem(row.a, row.b)
                    error_result = fingerprinter.error_parser.fingerprint(problem, model_answer)
                    if error_result.heuristic != Heuristic.UNKNOWN:
                        error_heuristic = error_result.heuristic.value
                        error_confidence = error_result.confidence

                trace_signals = _analyze_trace_outputs(
                    trace,
                    extraction_result,
                    fingerprinter.trace_classifier,
                    getattr(fingerprinter, "embedding_classifier", None),
                )

                result = _build_fingerprinting_result(
                    row=row,
                    detected_heuristic=detected_heuristic,
                    detection_confidence=detection_confidence,
                    model_answer=model_answer,
                    error_heuristic=error_heuristic,
                    error_confidence=error_confidence,
                    perplexity_losses=perplexity_losses,
                    trace=trace,
                    trace_signals=trace_signals,
                    neutral_loss=neutral_loss,
                    delta_losses=delta_losses,
                    per_template_losses=per_template_losses,
                    extraction_result=extraction_result,
                    probe_metadata=probe_metadata,
                )
                results.append(result)

                if writer:
                    writer.write(result)

            if verbose:
                tprint(f"  Async fingerprinting complete: {len(results)} results")

            return results

        async def _generate_one_async(row: HDSRow, idx: int) -> Tuple[int, Optional[int], Optional[str]]:
            """Generate for one problem using async API - direct sample_async() call."""
            async with semaphore:
                try:
                    prompt_text = f"What is {row.a} × {row.b}? Show your work step by step, then give the final answer."
                    model_input = client.build_text_generation_input(prompt_text)

                    if model_input is None:
                        return idx, None, None

                    sampling_params = tinker.types.SamplingParams(
                        max_tokens=2048,
                        temperature=0.0
                    )

                    result = await asyncio.wait_for(
                        sampling_client.sample_async(
                            prompt=model_input,
                            sampling_params=sampling_params,
                            num_samples=1
                        ),
                        timeout=TIMEOUT_SECONDS
                    )

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

                    answer = None
                    if text:
                        extraction = extract_answer_enhanced(text, a=row.a, b=row.b)
                        answer = extraction.answer

                    completed_count[0] += 1
                    if verbose and completed_count[0] % max(1, total // 5) == 0:
                        tprint(f"    Generation progress: {completed_count[0]}/{total}")

                    return idx, answer, text

                except asyncio.TimeoutError:
                    if verbose:
                        tprint(f"    Timeout for {row.id} after {TIMEOUT_SECONDS}s")
                    return idx, None, None
                except Exception as e:
                    if verbose:
                        tprint(f"    Warning: Async generation failed for {row.id}: {type(e).__name__}: {e}")
                    return idx, None, None

        if verbose:
            tprint(f"  Submitting {total} async generation tasks...")
            import sys
            sys.stdout.flush()

        tasks = [_generate_one_async(row, i) for i, row in enumerate(hds)]

        if verbose:
            tprint(f"  Awaiting {len(tasks)} tasks with asyncio.gather()...")
            import sys
            sys.stdout.flush()

        generation_results = await asyncio.gather(*tasks, return_exceptions=True)

        if verbose:
            tprint(f"  Processing {len(generation_results)} generation results...")

        gen_map: Dict[int, Tuple[Optional[int], Optional[str]]] = {}
        for r in generation_results:
            if isinstance(r, Exception):
                continue
            idx, answer, trace = r
            gen_map[idx] = (answer, trace)

        if verbose:
            tprint(f"  Computing perplexity probes in batches of {batch_size}...")

        perplexity_map: Dict[int, Dict[str, Any]] = {}
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_rows = hds[batch_start:batch_end]

            batch_problems = [(r.a, r.b) for r in batch_rows]
            batch_results = await fingerprinter.client.compute_heuristic_losses_multi_async(
                batch_problems,
                batch_size=batch_size,
                include_neutral=True,
                max_in_flight=score_max_in_flight
            )

            for local_idx, res in enumerate(batch_results):
                perplexity_map[batch_start + local_idx] = res

            if verbose and batch_end % max(1, total // 5) == 0:
                tprint(f"    Perplexity: {batch_end}/{total} problems processed")

        if verbose:
            tprint(f"  Building final results...")

        for i, row in enumerate(hds):
            model_answer, trace = gen_map.get(i, (None, None))

            extraction_result = None
            if trace:
                extraction_result = extract_answer_enhanced(trace, a=row.a, b=row.b)

            perp_data = perplexity_map.get(i, {})
            perplexity_losses = perp_data.get("losses", {})
            neutral_loss = perp_data.get("neutral_loss", 0.0)
            delta_losses = perp_data.get("delta_losses", {})
            per_template_losses = perp_data.get("per_template_losses", {})
            probe_metadata = _extract_probe_metadata(perp_data)
            detected_heuristic, detection_confidence = _resolve_loss_based_detection(
                perplexity_losses,
                best_heuristic=perp_data.get("best_heuristic"),
                confidence=perp_data.get("confidence"),
            )

            error_heuristic = None
            error_confidence = None
            if model_answer is not None and model_answer != row.product:
                problem = Problem(row.a, row.b)
                error_result = fingerprinter.error_parser.fingerprint(
                    problem, model_answer
                )
                if error_result.heuristic != Heuristic.UNKNOWN:
                    error_heuristic = error_result.heuristic.value
                    error_confidence = error_result.confidence

            trace_signals = _analyze_trace_outputs(
                trace,
                extraction_result,
                fingerprinter.trace_classifier,
                getattr(fingerprinter, "embedding_classifier", None),
            )

            result = _build_fingerprinting_result(
                row=row,
                detected_heuristic=detected_heuristic,
                detection_confidence=detection_confidence,
                model_answer=model_answer,
                error_heuristic=error_heuristic,
                error_confidence=error_confidence,
                perplexity_losses=perplexity_losses,
                trace=trace,
                trace_signals=trace_signals,
                neutral_loss=neutral_loss,
                delta_losses=delta_losses,
                per_template_losses=per_template_losses,
                extraction_result=extraction_result,
                probe_metadata=probe_metadata,
            )
            results.append(result)

            if writer:
                writer.write(result)

        if verbose:
            tprint(f"  Async fingerprinting complete: {len(results)} results")

        return results

    finally:
        if writer:
            writer.close()


def analyze_results(results: List[FingerprintingResult]) -> Dict:
    """Analyze fingerprinting results."""
    unknown_count = sum(1 for r in results if r.detected_heuristic.upper() == "UNKNOWN")
    nonfinite_loss_count = 0
    resolved_probe_count = 0
    for r in results:
        heuristic_losses = list((r.perplexity_losses or {}).values())
        has_nonfinite = not heuristic_losses or any(not math.isfinite(float(loss)) for loss in heuristic_losses)
        if r.neutral_loss is not None:
            has_nonfinite = has_nonfinite or (not math.isfinite(float(r.neutral_loss)))
        if has_nonfinite:
            nonfinite_loss_count += 1
        if _compute_soft_support_mass(r.perplexity_losses or {}, r.neutral_loss) is not None:
            resolved_probe_count += 1

    unknown_rate = (unknown_count / len(results)) if results else 0.0
    if unknown_rate > 0.50:
        quality_status = "invalid"
    elif unknown_rate > 0.10:
        quality_status = "degraded"
    else:
        quality_status = "ok"

    analysis: Dict[str, Any] = {
        "total": len(results),
        "correct": sum(1 for r in results if r.is_correct),
        "accuracy": sum(1 for r in results if r.is_correct) / len(results) if results else 0,
        "probe_kind": LEXICAL_PROBE_KIND,
        "family_match_rate": (
            sum(1 for r in results if r.detected_heuristic.upper() == (r.design_family or r.target_heuristic))
            / len(results)
            if results else 0
        ),
        "detection_semantics": LOSS_DETECTION_SEMANTICS,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "nonfinite_loss_count": nonfinite_loss_count,
        "nonfinite_loss_rate": (nonfinite_loss_count / len(results)) if results else 0.0,
        "resolved_probe_count": resolved_probe_count,
        "resolved_probe_rate": (resolved_probe_count / len(results)) if results else 0.0,
        "quality_status": quality_status,
        "support_measure_semantics": "softmax_over_negative_loss_with_neutral",
        **_get_template_context(),
    }

    by_design_family: Dict[str, Dict[str, float]] = {}
    for r in results:
        family = r.design_family or r.target_heuristic
        if family not in by_design_family:
            by_design_family[family] = {
                "total": 0,
                "correct": 0,
                "family_match": 0,
                "resolved_probe_count": 0,
                "_target_support_values": [],
                "embedding_resolved_count": 0,
                "embedding_detection_hits": 0,
                "_embedding_target_support_values": [],
            }
        by_design_family[family]["total"] += 1
        if r.is_correct:
            by_design_family[family]["correct"] += 1
        if r.detected_heuristic.upper() == family:
            by_design_family[family]["family_match"] += 1
        support_mass = _compute_soft_support_mass(r.perplexity_losses or {}, r.neutral_loss)
        if support_mass is not None:
            by_design_family[family]["resolved_probe_count"] += 1
            by_design_family[family]["_target_support_values"].append(support_mass.get(family, 0.0))
        embedding_support_mass = _normalize_embedding_support_mass(r.embedding_support_mass)
        if embedding_support_mass is not None and r.embedding_resolved:
            by_design_family[family]["embedding_resolved_count"] += 1
            by_design_family[family]["_embedding_target_support_values"].append(
                embedding_support_mass.get(family, 0.0)
            )
            if (r.embedding_heuristic or "").upper() == family:
                by_design_family[family]["embedding_detection_hits"] += 1

    for family, family_stats in by_design_family.items():
        family_stats["accuracy"] = family_stats["correct"] / family_stats["total"] if family_stats["total"] > 0 else 0
        family_stats["family_match_rate"] = (
            family_stats["family_match"] / family_stats["total"]
            if family_stats["total"] > 0 else 0
        )
        target_support_values = family_stats.pop("_target_support_values", [])
        family_stats["coverage_rate"] = (
            family_stats["resolved_probe_count"] / family_stats["total"]
            if family_stats["total"] > 0 else 0
        )
        family_stats["coverage_se"] = (
            math.sqrt(family_stats["coverage_rate"] * (1 - family_stats["coverage_rate"]) / family_stats["total"])
            if family_stats["total"] > 0 else 0
        )
        if target_support_values:
            family_stats["target_support_mean"] = sum(target_support_values) / len(target_support_values)
            if len(target_support_values) >= 2:
                mean_value = family_stats["target_support_mean"]
                sum_sq = sum((float(value) - mean_value) ** 2 for value in target_support_values)
                sample_std = math.sqrt(sum_sq / (len(target_support_values) - 1))
                family_stats["target_support_se"] = sample_std / math.sqrt(len(target_support_values))
            else:
                family_stats["target_support_se"] = 0.0
        else:
            family_stats["target_support_mean"] = None
            family_stats["target_support_se"] = None
        embedding_target_support_values = family_stats.pop("_embedding_target_support_values", [])
        family_stats["embedding_coverage_rate"] = (
            family_stats["embedding_resolved_count"] / family_stats["total"]
            if family_stats["total"] > 0 else 0
        )
        family_stats["embedding_coverage_se"] = (
            math.sqrt(
                family_stats["embedding_coverage_rate"]
                * (1 - family_stats["embedding_coverage_rate"])
                / family_stats["total"]
            )
            if family_stats["total"] > 0 else 0
        )
        if family_stats["embedding_resolved_count"] > 0:
            family_stats["embedding_detection_rate"] = (
                family_stats["embedding_detection_hits"] / family_stats["embedding_resolved_count"]
            )
            family_stats["embedding_detection_se"] = math.sqrt(
                family_stats["embedding_detection_rate"]
                * (1 - family_stats["embedding_detection_rate"])
                / family_stats["embedding_resolved_count"]
            )
        else:
            family_stats["embedding_detection_rate"] = 0.0
            family_stats["embedding_detection_se"] = 0.0
        if embedding_target_support_values:
            family_stats["embedding_target_support_mean"] = (
                sum(embedding_target_support_values) / len(embedding_target_support_values)
            )
            if len(embedding_target_support_values) >= 2:
                mean_value = family_stats["embedding_target_support_mean"]
                sum_sq = sum((float(value) - mean_value) ** 2 for value in embedding_target_support_values)
                sample_std = math.sqrt(sum_sq / (len(embedding_target_support_values) - 1))
                family_stats["embedding_target_support_se"] = (
                    sample_std / math.sqrt(len(embedding_target_support_values))
                )
            else:
                family_stats["embedding_target_support_se"] = 0.0
        else:
            family_stats["embedding_target_support_mean"] = None
            family_stats["embedding_target_support_se"] = None

    analysis["by_design_family"] = by_design_family

    confusion: Dict[Tuple[str, str], int] = {}
    for r in results:
        key = ((r.design_family or r.target_heuristic), r.detected_heuristic.upper())
        confusion[key] = confusion.get(key, 0) + 1

    analysis["confusion_matrix"] = confusion

    # Perplexity analysis (if available)
    losses_by_heuristic: Dict[str, List[float]] = {"OT": [], "DD": [], "RC": []}
    delta_losses_by_heuristic: Dict[str, List[float]] = {"OT": [], "DD": [], "RC": []}
    neutral_losses: List[float] = []

    for r in results:
        if r.perplexity_losses:
            for h, loss in r.perplexity_losses.items():
                if loss < float('inf'):
                    losses_by_heuristic[h].append(loss)

        # Collect delta losses (new template-averaged values)
        if r.delta_losses:
            for h, delta in r.delta_losses.items():
                if delta is not None and abs(delta) < float('inf'):
                    delta_losses_by_heuristic[h].append(delta)

        # Collect neutral baseline losses
        if r.neutral_loss is not None and r.neutral_loss < float('inf'):
            neutral_losses.append(r.neutral_loss)

    analysis["avg_perplexity"] = {
        h: sum(losses) / len(losses) if losses else None
        for h, losses in losses_by_heuristic.items()
    }

    # Add delta loss analysis (relative to neutral baseline)
    analysis["avg_delta_loss"] = {
        h: sum(deltas) / len(deltas) if deltas else None
        for h, deltas in delta_losses_by_heuristic.items()
    }

    # Add average neutral baseline loss
    analysis["avg_neutral_loss"] = sum(neutral_losses) / len(neutral_losses) if neutral_losses else None

    # Error-based detection analysis (ErrorShapeParser)
    error_detections = [r for r in results if r.error_heuristic is not None]
    error_detection_by_heuristic: Dict[str, Dict[str, Any]] = {}
    for r in error_detections:
        if r.error_heuristic is None:
            continue
        h = r.error_heuristic.upper()
        if h not in error_detection_by_heuristic:
            error_detection_by_heuristic[h] = {"count": 0, "confidences": []}
        error_detection_by_heuristic[h]["count"] += 1
        if r.error_confidence is not None:
            error_detection_by_heuristic[h]["confidences"].append(r.error_confidence)

    for h, error_stats in error_detection_by_heuristic.items():
        error_stats["avg_confidence"] = sum(error_stats["confidences"]) / len(error_stats["confidences"]) if error_stats["confidences"] else None

    analysis["error_detection"] = {
        "total_with_errors": len([r for r in results if not r.is_correct]),
        "errors_with_heuristic_detected": len(error_detections),
        "by_heuristic": {h: {"count": s["count"], "avg_confidence": s["avg_confidence"]}
                        for h, s in error_detection_by_heuristic.items()}
    }

    # Trace-based detection analysis (TraceClassifier)
    trace_detections = [r for r in results if r.trace_heuristic is not None]
    trace_detection_by_heuristic: Dict[str, Dict[str, Any]] = {}
    for r in trace_detections:
        if r.trace_heuristic is None:
            continue
        h = r.trace_heuristic.upper()
        if h not in trace_detection_by_heuristic:
            trace_detection_by_heuristic[h] = {"count": 0, "confidences": []}
        trace_detection_by_heuristic[h]["count"] += 1
        if r.trace_confidence is not None:
            trace_detection_by_heuristic[h]["confidences"].append(r.trace_confidence)

    for h, trace_stats in trace_detection_by_heuristic.items():
        trace_stats["avg_confidence"] = sum(trace_stats["confidences"]) / len(trace_stats["confidences"]) if trace_stats["confidences"] else None

    analysis["trace_detection"] = {
        "total_with_traces": len([r for r in results if r.trace is not None]),
        "traces_with_heuristic_detected": len(trace_detections),
        "by_heuristic": {h: {"count": s["count"], "avg_confidence": s["avg_confidence"]}
                        for h, s in trace_detection_by_heuristic.items()}
    }

    embedding_detections = [r for r in results if r.embedding_resolved and r.embedding_heuristic is not None]
    embedding_detection_by_heuristic: Dict[str, Dict[str, Any]] = {}
    for r in embedding_detections:
        heuristic = (r.embedding_heuristic or "").upper()
        if not heuristic:
            continue
        if heuristic not in embedding_detection_by_heuristic:
            embedding_detection_by_heuristic[heuristic] = {"count": 0, "confidences": [], "margins": []}
        embedding_detection_by_heuristic[heuristic]["count"] += 1
        if r.embedding_confidence is not None:
            embedding_detection_by_heuristic[heuristic]["confidences"].append(r.embedding_confidence)
        if r.embedding_margin is not None:
            embedding_detection_by_heuristic[heuristic]["margins"].append(r.embedding_margin)

    for heuristic, stats in embedding_detection_by_heuristic.items():
        stats["avg_confidence"] = (
            sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else None
        )
        stats["avg_margin"] = (
            sum(stats["margins"]) / len(stats["margins"]) if stats["margins"] else None
        )

    resolved_embedding_count = sum(1 for r in results if r.embedding_resolved)
    analysis["embedding_detection"] = {
        "total_with_traces": len([r for r in results if r.trace is not None]),
        "resolved_embeddings": resolved_embedding_count,
        "resolved_embedding_rate": (resolved_embedding_count / len(results)) if results else 0.0,
        "embeddings_with_heuristic_detected": len(embedding_detections),
        "detection_semantics": EMBEDDING_DETECTION_SEMANTICS,
        "support_measure_semantics": EMBEDDING_SUPPORT_SEMANTICS,
        "by_heuristic": {
            h: {
                "count": s["count"],
                "avg_confidence": s["avg_confidence"],
                "avg_margin": s["avg_margin"],
            }
            for h, s in embedding_detection_by_heuristic.items()
        },
    }

    template_variability: Dict[str, List[float]] = {"OT": [], "DD": [], "RC": []}
    for r in results:
        if not r.per_template_losses:
            continue
        grouped: Dict[str, List[float]] = {"OT": [], "DD": [], "RC": []}
        for template_id, payload in r.per_template_losses.items():
            heuristic = template_id.split("_", 1)[0].upper()
            if heuristic not in grouped:
                continue
            loss = payload.get("loss")
            if isinstance(loss, (int, float)) and loss < float("inf"):
                grouped[heuristic].append(float(loss))
        for heuristic, losses in grouped.items():
            if len(losses) < 2:
                continue
            mean_loss = sum(losses) / len(losses)
            variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
            template_variability[heuristic].append(variance ** 0.5)

    analysis["template_variability"] = {
        heuristic: {
            "count": len(stds),
            "mean_within_problem_std": (sum(stds) / len(stds)) if stds else None,
        }
        for heuristic, stds in template_variability.items()
    }
    analysis["confidence_definition"] = {
        "name": "margin_gap",
        "formula": "confidence = min(1.0, 2 * ((loss_2 - loss_1) / loss_2))",
    }

    return analysis


def print_analysis(analysis: Dict):
    """Print formatted analysis."""
    tprint()
    tprint("=" * 60)
    tprint("FINGERPRINTING RESULTS")
    tprint("=" * 60)
    tprint()

    tprint(f"Overall Accuracy: {analysis['accuracy']:.1%} ({analysis['correct']}/{analysis['total']})")
    tprint(f"Lexical Family Match Rate: {analysis['family_match_rate']:.1%}")
    tprint()

    tprint("By Design Family:")
    tprint("-" * 40)
    for h, stats in sorted(analysis['by_design_family'].items()):
        tprint(f"  {h}: accuracy={stats['accuracy']:.1%}, family_match={stats['family_match_rate']:.1%} ({stats['total']} items)")
    tprint()

    tprint("Design-Family Confusion Matrix:")
    tprint("-" * 40)
    tprint("  (rows=design family, cols=detected)")
    heuristics = ["RC", "DD", "OT"]
    tprint("         " + "  ".join(f"{h:>6}" for h in heuristics) + "  UNKN")
    for target in heuristics:
        row = [analysis['confusion_matrix'].get((target, det), 0) for det in heuristics]
        unknown = analysis['confusion_matrix'].get((target, "UNKNOWN"), 0)
        tprint(f"  {target}:   " + "  ".join(f"{v:>6}" for v in row) + f"  {unknown:>4}")
    tprint()

    if analysis.get('avg_perplexity'):
        tprint("Average Perplexity by Heuristic Template:")
        tprint("-" * 40)
        for h, avg in analysis['avg_perplexity'].items():
            if avg is not None:
                tprint(f"  {h}: {avg:.4f}")
        tprint()

    # Show delta loss analysis (relative to neutral baseline)
    if analysis.get('avg_delta_loss'):
        tprint("Average Δ-Loss (relative to neutral baseline):")
        tprint("-" * 40)
        neutral = analysis.get('avg_neutral_loss')
        if neutral is not None:
            tprint(f"  Neutral baseline: {neutral:.4f}")
        for h, delta in analysis['avg_delta_loss'].items():
            if delta is not None:
                # Negative delta = lower than baseline = preferred
                pref = "← preferred" if delta < 0 else ""
                tprint(f"  {h}: {delta:+.4f} {pref}")
        tprint()

    # Show error-based detection analysis (ErrorShapeParser)
    if analysis.get('error_detection'):
        ed = analysis['error_detection']
        tprint("Error-Based Detection (ErrorShapeParser):")
        tprint("-" * 40)
        tprint(f"  Wrong answers: {ed['total_with_errors']}")
        tprint(f"  Heuristic detected: {ed['errors_with_heuristic_detected']}")
        if ed.get('by_heuristic'):
            for h, stats in sorted(ed['by_heuristic'].items()):
                conf = f" (avg conf: {stats['avg_confidence']:.2f})" if stats.get('avg_confidence') else ""
                tprint(f"    {h}: {stats['count']}{conf}")
        tprint()

    # Show trace-based detection analysis (TraceClassifier)
    if analysis.get('trace_detection'):
        td = analysis['trace_detection']
        tprint("Trace-Based Detection (TraceClassifier):")
        tprint("-" * 40)
        tprint(f"  Problems with traces: {td['total_with_traces']}")
        tprint(f"  Heuristic detected: {td['traces_with_heuristic_detected']}")
        if td.get('by_heuristic'):
            for h, stats in sorted(td['by_heuristic'].items()):
                conf = f" (avg conf: {stats['avg_confidence']:.2f})" if stats.get('avg_confidence') else ""
                tprint(f"    {h}: {stats['count']}{conf}")
        tprint()

    if analysis.get('embedding_detection'):
        ed = analysis['embedding_detection']
        tprint("Trace-Embedding Detection (PrototypeEmbeddingClassifier):")
        tprint("-" * 40)
        tprint(f"  Problems with traces: {ed['total_with_traces']}")
        tprint(f"  Resolved embeddings: {ed['resolved_embeddings']}")
        if ed.get('by_heuristic'):
            for h, stats in sorted(ed['by_heuristic'].items()):
                extras = []
                if stats.get('avg_confidence') is not None:
                    extras.append(f"avg conf: {stats['avg_confidence']:.2f}")
                if stats.get('avg_margin') is not None:
                    extras.append(f"avg margin: {stats['avg_margin']:.2f}")
                suffix = f" ({', '.join(extras)})" if extras else ""
                tprint(f"    {h}: {stats['count']}{suffix}")
        tprint()


def save_results(results: List[FingerprintingResult], analysis: Dict, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as CSV
    csv_path = output_dir / "fingerprint_results.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['hds_id', 'a', 'b', 'product', 'target_heuristic',
                      'design_family', 'canonical_target_heuristic', 'probe_kind',
                      'detected_heuristic', 'detection_confidence',
                      'loss_best_heuristic', 'loss_best_confidence', 'detection_semantics',
                      'model_answer',
                      'is_correct', 'error_delta', 'error_heuristic', 'error_confidence',
                      'trace_heuristic', 'trace_confidence',
                      'embedding_heuristic', 'embedding_confidence', 'embedding_margin',
                      'embedding_model', 'embedding_resolved', 'embedding_resolution_status',
                      'neutral_loss',
                      'extraction_confidence', 'extraction_strategy', 'is_truncated',
                      'probe_resolved', 'probe_resolution_status',
                      'probe_resolution_error', 'probe_image_token_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k in fieldnames}
            row["loss_best_heuristic"] = r.detected_heuristic
            row["loss_best_confidence"] = r.detection_confidence
            row["detection_semantics"] = LOSS_DETECTION_SEMANTICS
            writer.writerow(row)
    tprint(f"Saved results to {csv_path}")

    # Save analysis as JSON
    json_path = output_dir / "fingerprint_analysis.json"
    # Convert tuple keys to strings for JSON
    json_analysis = analysis.copy()
    json_analysis['confusion_matrix'] = {
        f"{k[0]}->{k[1]}": v for k, v in analysis['confusion_matrix'].items()
    }
    with open(json_path, 'w') as f:
        json.dump(json_analysis, f, indent=2)
    tprint(f"Saved analysis to {json_path}")


def _format_detail_record(r: FingerprintingResult) -> dict:
    """Format a FingerprintingResult as a detailed JSON record."""
    from datetime import datetime, timezone
    template_context = _get_template_context()
    support_mass = _compute_soft_support_mass(r.perplexity_losses or {}, r.neutral_loss)

    return {
        "hds_id": r.hds_id,
        "a": r.a,
        "b": r.b,
        "product": r.product,
        "target_heuristic": r.target_heuristic,
        "design_family": r.design_family or r.target_heuristic,
        "canonical_target_heuristic": r.canonical_target_heuristic or r.target_heuristic,
        "probe_kind": r.probe_kind,
        "detected_heuristic": r.detected_heuristic,
        "detection_semantics": LOSS_DETECTION_SEMANTICS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **template_context,
        "generation": {
            "trace": r.trace,
            "model_answer": r.model_answer,
            "is_correct": r.is_correct,
            "prompt_type": "reasoning" if r.trace else "direct"
        },
        "extraction": {
            "confidence": r.extraction_confidence,
            "strategy": r.extraction_strategy,
            "is_truncated": r.is_truncated,
            "is_contaminated": r.is_contaminated
        },
        "perplexity": {
            "templates": r.per_template_losses or {},
            "aggregated": r.perplexity_losses,
            "delta_losses": r.delta_losses,
            "neutral_loss": r.neutral_loss,
            "support_mass": support_mass,
            "probe_resolved": r.probe_resolved if r.probe_resolved is not None else (support_mass is not None),
            "probe_resolution_status": r.probe_resolution_status,
            "probe_resolution_error": r.probe_resolution_error,
            "image_token_count": r.probe_image_token_count,
            "loss_best_heuristic": r.detected_heuristic,
            "loss_best_confidence": r.detection_confidence,
            "detection_semantics": LOSS_DETECTION_SEMANTICS,
            "support_measure_semantics": "softmax_over_negative_loss_with_neutral",
        },
        "error_analysis": {
            "error_heuristic": r.error_heuristic,
            "error_confidence": r.error_confidence,
            "error_delta": r.error_delta
        },
        "trace_analysis": {
            "trace_heuristic": r.trace_heuristic,
            "trace_confidence": r.trace_confidence
        },
        "embedding_analysis": {
            "embedding_heuristic": r.embedding_heuristic,
            "embedding_confidence": r.embedding_confidence,
            "embedding_margin": r.embedding_margin,
            "support_mass": r.embedding_support_mass,
            "model": r.embedding_model,
            "resolved": r.embedding_resolved if r.embedding_resolved is not None else (r.embedding_support_mass is not None),
            "resolution_status": r.embedding_resolution_status,
            "detection_semantics": EMBEDDING_DETECTION_SEMANTICS,
            "support_measure_semantics": EMBEDDING_SUPPORT_SEMANTICS,
        }
    }


class StreamingResultWriter:
    """Write fingerprint results incrementally to JSONL for crash safety."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "fingerprint_details.jsonl"
        self.file = open(self.jsonl_path, 'w')
        self.count = 0

    def write(self, result: FingerprintingResult):
        """Write single result, flush immediately for crash safety."""
        record = _format_detail_record(result)
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        self.count += 1

    def close(self):
        self.file.close()
        tprint(f"Streamed {self.count} results to {self.jsonl_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def save_detailed_results(results: List[FingerprintingResult], output_dir: Path):
    """
    Save detailed results including traces and per-template losses to JSONL.

    This supplementary file preserves data that would otherwise be discarded,
    enabling re-analysis of model outputs without re-running experiments.

    Note: When streaming is enabled, this is called only as a fallback.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "fingerprint_details.jsonl"

    with open(jsonl_path, 'w') as f:
        for r in results:
            record = _format_detail_record(r)
            f.write(json.dumps(record) + "\n")

    tprint(f"Saved detailed results to {jsonl_path}")


def main():
    """Run baseline fingerprinting."""
    import argparse

    parser = argparse.ArgumentParser(description="Run fingerprinting on HDS or Traps dataset")
    parser.add_argument("--dataset", type=str, default="HDSv2",
                        choices=["HDS", "HDSv2", "Traps", "Trapsv2"],
                        help="Dataset to use (default: HDSv2)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Custom CSV path (overrides --dataset)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--modality", type=str, default="text",
                        choices=["text", "image"],
                        help="Input modality: text or image (default: text)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Override image directory for image modality")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model to use (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for perplexity probes (default: 1, no batching; >1 enables multi-problem batching)")
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="Use async generation (~35%% faster than parallel workers)")
    parser.add_argument("--no-async", dest="use_async", action="store_false",
                        help="Force legacy synchronous path (not recommended)")
    parser.set_defaults(use_async=True)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Async concurrency limit (default: 4, used with --async)")
    parser.add_argument("--score-in-flight", type=int, default=4,
                        help="Max forward() batches in flight for async scoring (default: 4)")
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
    parser.add_argument("--embedding-model", type=str, default=os.getenv("FINGERPRINT_EMBED_MODEL"),
                        help="Optional local HF embedding model or path for trace prototype classification")
    parser.add_argument("--embedding-batch-size", type=int,
                        default=int(os.getenv("FINGERPRINT_EMBED_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)),
                        help=f"Batch size for embedding inference (default: env or {DEFAULT_EMBEDDING_BATCH_SIZE})")
    parser.add_argument("--embedding-cache-dir", type=str,
                        default=str(REPO_ROOT / "Tmp" / "embedding_prototypes"),
                        help="Cache directory for fitted embedding prototypes")
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

    # Parallel processing now supported with thread-local sampling clients
    # (requires batch_size > 1 for optimal performance)
    if args.max_workers > 1 and args.batch_size <= 1:
        tprint("NOTE: For parallel generation, recommend using --batch-size 30")
        tprint("      (parallel mode uses batched perplexity + parallel generation)")
        args.batch_size = 30  # Auto-enable batching for parallel mode

    # Determine dataset path
    if args.csv:
        dataset_path = Path(args.csv)
        dataset_name = dataset_path.stem
    else:
        dataset_name = args.dataset
        dataset_path = REPO_ROOT / "SavedData" / f"{dataset_name}.csv"

    # Use specified model for both modalities (VLM supports both text and image)
    model_display = args.model

    tprint("=" * 60)
    tprint(f"Baseline Fingerprinting on {dataset_name}")
    tprint("=" * 60)
    tprint()
    tprint(f"Model: {model_display}")
    tprint(f"Modality: {args.modality}")
    tprint(f"Split: {args.split}")
    tprint(f"Template mode: {get_heuristic_template_mode()} (seed={get_heuristic_template_seed()})")
    tprint(f"Template profile: {get_heuristic_template_profile()}")
    if args.embedding_model:
        tprint(f"Embedding detector: {args.embedding_model}")
    tprint()

    # Load dataset
    tprint(f"Loading dataset from {dataset_path}...")
    all_rows = load_hds(dataset_path)
    tprint(f"Loaded {len(all_rows)} total problems")

    hds = select_rows_for_split(all_rows, args.split, dataset_name)
    if args.split == "all":
        tprint(f"Using all problems: {len(hds)}")
    else:
        tprint(f"Using {args.split} split: {len(hds)} problems")
    tprint()

    embedding_classifier = _initialize_embedding_classifier(
        args.embedding_model,
        cache_dir=Path(args.embedding_cache_dir),
        batch_size=args.embedding_batch_size,
        prototype_sample_cap=DEFAULT_EMBEDDING_PROTOTYPE_SAMPLE_CAP,
    )

    # Check for API key
    api_key = os.getenv("TINKER_API_KEY")
    if api_key:
        tprint("TINKER_API_KEY found - using real Tinker API")
        try:
            if args.modality == "text":
                fingerprinter = TinkerFingerprinter(
                    api_key,
                    model_name=args.model,
                    embedding_classifier=embedding_classifier,
                )
            else:
                # Image modality - use vision fingerprinter (same VLM model)
                images_dir = resolve_images_dir(
                    dataset_name,
                    images_dir_arg=args.images_dir,
                    has_custom_csv=bool(args.csv),
                )
                fingerprinter = ImageFingerprinter(
                    api_key,
                    model_name=args.model,
                    images_dir=images_dir,
                    embedding_classifier=embedding_classifier,
                )
            use_real_api = True
        except Exception as e:
            tprint(f"Warning: Failed to initialize Tinker: {e}")
            tprint("Falling back to mock fingerprinter")
            fingerprinter = MockFingerprinter(embedding_classifier=embedding_classifier)
            use_real_api = False
    else:
        tprint("TINKER_API_KEY not found - using mock fingerprinter")
        fingerprinter = MockFingerprinter(embedding_classifier=embedding_classifier)
        use_real_api = False

    # Compute output directory early for streaming (crash-safe incremental output)
    split_suffix = f"_{args.split}" if args.split != "all" else ""
    modality_suffix = f"_{args.modality}" if args.modality != "text" else ""
    # Extract short model name for path (e.g., "Qwen3-VL-30B-A3B" from full path)
    model_slug = args.model.split("/")[-1].replace("-Instruct", "")
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    output_dir = REPO_ROOT / "SavedResults" / (
        f"fingerprint_{dataset_name.lower()}{split_suffix}{modality_suffix}{tag_suffix}_{model_slug}"
    )

    tprint()
    tprint("Running fingerprinting...")
    tprint("-" * 40)
    if args.use_async:
        tprint(f"Using async generation (concurrency={args.concurrency})")
        tprint(f"Async scoring batches in flight: {args.score_in_flight}")
    elif args.max_workers > 1:
        tprint(f"Using {args.max_workers} parallel workers")
    if args.batch_size > 1:
        tprint(f"Using batch size {args.batch_size} for perplexity probes")
    tprint(f"Streaming results to: {output_dir}/fingerprint_details.jsonl")

    start_time = time.time()

    # Dispatch to appropriate fingerprinting function
    if args.use_async:
        # Async generation (~35% faster than parallel workers)
        results = run_fingerprinting_async(
            hds, fingerprinter, verbose=True,
            batch_size=args.batch_size,
            output_dir=output_dir,
            concurrency=args.concurrency,
            score_max_in_flight=args.score_in_flight
        )
    else:
        # Standard fingerprinting (parallel or sequential)
        results = run_fingerprinting(
            hds, fingerprinter, verbose=True,
            max_workers=args.max_workers, batch_size=args.batch_size,
            output_dir=output_dir,
            use_async_backend=False,
            concurrency=args.concurrency,
            score_max_in_flight=args.score_in_flight
        )
    elapsed = time.time() - start_time

    tprint()
    if len(hds) > 0:
        tprint(f"Completed in {elapsed:.1f}s ({elapsed/len(hds):.2f}s per problem)")
    else:
        tprint(f"No problems to fingerprint. Check split filter.")

    # Analyze
    analysis = analyze_results(results)
    analysis["model"] = model_display
    analysis["dataset"] = dataset_name
    analysis["split"] = args.split
    analysis["modality"] = args.modality
    analysis["template_mode"] = get_heuristic_template_mode()
    analysis["template_profile"] = get_heuristic_template_profile()
    analysis["template_seed"] = get_heuristic_template_seed()
    analysis["template_bank_hash"] = template_validation["template_bank_hash"]
    analysis["output_tag"] = args.output_tag
    analysis["elapsed_seconds"] = elapsed
    analysis["use_real_api"] = use_real_api
    analysis["embedding_model"] = args.embedding_model
    analysis["embedding_detection_semantics"] = EMBEDDING_DETECTION_SEMANTICS

    # Print analysis
    print_analysis(analysis)

    # Save summary results (CSV + JSON analysis)
    # Note: detailed JSONL was already streamed during run_fingerprinting()
    save_results(results, analysis, output_dir)
    run_manifest = {
        "script": "Scripts/experiments/BaselineFingerprint.py",
        "dataset": dataset_name,
        "split": args.split,
        "modality": args.modality,
        "model": model_display,
        "output_tag": args.output_tag,
        "use_real_api": use_real_api,
        "embedding_model": args.embedding_model,
        "elapsed_seconds": elapsed,
        "detection_semantics": LOSS_DETECTION_SEMANTICS,
        "cli_args": vars(args),
        **template_validation,
    }
    _write_run_manifest(output_dir, run_manifest)

    tprint("=" * 60)
    tprint("Done!")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
