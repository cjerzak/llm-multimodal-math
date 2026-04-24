#!/usr/bin/env python3
"""
GenerateResultsFigures.py

Generate figures and LaTeX macros from fingerprinting results for the paper.
Outputs to PaperTexFolder/Figures/ by default.

Supports fine-grained output generation for parallel pipeline execution:
  --model MODEL          Only process this model's results
  --output-type TYPE     fingerprint-figures|fingerprint-macros|nudge-macros|
                         nudge-appendix|gradient-macros|gradient-appendix|
                         embedding-appendix|training-appendix|merge-macros|
                         merge-nudge-appendix|merge-gradient-appendix|
                         merge-embedding-appendix|all
  --output-path PATH     Override the destination for single-file outputs
  --fragment-path PATH   Input fragment path for merge output types (repeatable)
  --append-macros        Append to existing macros file (legacy/manual usage)
"""

import argparse
import fcntl
import json
import csv
import re
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Sequence, Union, TypedDict

# Paths (analysis/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "SavedResults"
OUTPUT_DIR = REPO_ROOT / "PaperTexFolder" / "Figures"

# Add Scripts to path for imports when run directly
import sys
sys.path.insert(0, str(SCRIPTS_DIR))
from core.Logging import tprint
from core.GenerateMathHelpers import CANONICAL_COMPLEXITY_BANDS
from core.NudgeTaxonomy import classify_error_taxonomy, summarize_taxonomy

# Bootstrap configuration
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
PRIMARY_LORA_HEURISTICS = ("RC", "DD", "OT")
CONTROL_LORA_HEURISTICS = ("STYLE",)
PRIMARY_LORA_RESULT_LABELS = set(PRIMARY_LORA_HEURISTICS)
PRIMARY_LORA_DETAIL_LABELS = {f"{heuristic.lower()}_lora" for heuristic in PRIMARY_LORA_HEURISTICS}

# Font size constants for publication-quality figures
FONT_TINY = 10       # Small annotations in multi-panel figures (was 7-8)
FONT_SMALL = 12      # Legends, bar value labels (was 8-10)
FONT_BASE = 14       # Axis tick labels (was 9-13)
FONT_LABEL = 16      # Axis labels (xlabel/ylabel) (was 11-12)
FONT_TITLE = 18      # Panel/figure titles (was 12-14)
FONT_SUPTITLE = 22   # Super-titles for multi-panel figures (was 14-20)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ModuloInvariantAnalysis(TypedDict):
    errors: Dict[str, float]
    correct: Dict[str, float]
    error_count: int
    correct_count: int


def get_hds_dataset_size() -> int:
    """Get actual HDS dataset size from CSV file.

    Returns the total number of problems in HDS.csv, independent of
    which subset was fingerprinted. This ensures HDSAllCount reflects
    the true dataset size.
    """
    for hds_path in (REPO_ROOT / "SavedData" / "HDSv2.csv", REPO_ROOT / "SavedData" / "HDS.csv"):
        if hds_path.exists():
            with open(hds_path, 'r') as f:
                return sum(1 for _ in f) - 1  # Subtract header row
    return 0


def get_traps_dataset_size() -> int:
    """Get actual Traps dataset size from CSV file."""
    for traps_path in (REPO_ROOT / "SavedData" / "Trapsv2.csv", REPO_ROOT / "SavedData" / "Traps.csv"):
        if traps_path.exists():
            with open(traps_path, 'r') as f:
                return sum(1 for _ in f) - 1  # Subtract header row
    return 0


def get_multimodal_dataset_size() -> int:
    """Get paired multimodal dataset size from the shared grid or modality fallbacks."""
    candidates = [
        REPO_ROOT / "SavedData" / "SharedMultimodalGrid.csv",
        REPO_ROOT / "SavedData" / "TextGrid.csv",
        REPO_ROOT / "SavedData" / "ImageGrid.csv",
        REPO_ROOT / "SavedData" / "AudioGrid.csv",
    ]
    for path in candidates:
        if path.exists():
            with open(path, 'r') as f:
                return max(0, sum(1 for _ in f) - 1)
    return 0


def get_hds_split_composition() -> Dict[str, Dict[str, int]]:
    """Count HDS rows by split and design family from the canonical CSV."""
    counts: Dict[str, Dict[str, int]] = {}
    hds_path = REPO_ROOT / "SavedData" / "HDSv2.csv"
    if not hds_path.exists():
        hds_path = REPO_ROOT / "SavedData" / "HDS.csv"
    if not hds_path.exists():
        return counts

    with open(hds_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "all")
            target = row.get("design_family") or row.get("target_heuristic", "UNKNOWN")
            split_counts = counts.setdefault(split, {})
            split_counts[target] = split_counts.get(target, 0) + 1
    return counts


def get_dataset_csv_path(dataset_name: str) -> Path:
    """Return the canonical SavedData CSV path for a dataset label."""
    return REPO_ROOT / "SavedData" / f"{dataset_name}.csv"


def bootstrap_ci(
    values: Sequence[Union[int, float]],
    statistic: str = "mean",
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: List of values to bootstrap
        statistic: "mean" or "proportion"
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95)

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci)
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.array(values)
    n = len(arr)

    if statistic == "proportion":
        point_estimate = float(np.mean(arr))
    else:
        point_estimate = float(np.mean(arr))

    # Bootstrap samples
    bootstrap_stats: List[float] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=n, replace=True)
        bootstrap_stats.append(float(np.mean(sample)))

    # Compute CI
    alpha = 1 - ci_level
    lower_ci = float(np.percentile(bootstrap_stats, alpha/2 * 100))
    upper_ci = float(np.percentile(bootstrap_stats, (1 - alpha/2) * 100))

    return (point_estimate, lower_ci, upper_ci)


def compute_detection_rate_ci(
    results: List[dict],
    target_heuristic: str
) -> Tuple[float, float, float]:
    """Compute detection rate with bootstrap CI for a specific heuristic.

    Args:
        results: List of result dicts with 'target_heuristic' and 'detected_heuristic'
        target_heuristic: The heuristic to compute detection rate for

    Returns:
        Tuple of (rate, lower_ci, upper_ci) as percentages
    """
    # Filter to design-family bucket for lexical preamble probes
    target_results = [r for r in results if get_design_family(r) == target_heuristic]
    if not target_results:
        return (0.0, 0.0, 0.0)

    # Binary vector: 1 if detected == target, 0 otherwise
    correct = [1 if get_loss_detected_heuristic(r) == target_heuristic else 0
               for r in target_results]

    rate, lower, upper = bootstrap_ci(correct, statistic="proportion")
    return (rate * 100, lower * 100, upper * 100)


def parse_bool(value: Any) -> bool:
    """Parse a boolean value from mixed CSV/JSON inputs."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return False


def parse_correctness_label(value: Any) -> Optional[bool]:
    """Parse correctness labels into bool, preserving unknown."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("", "unknown", "none", "nan"):
            return None
        if lowered in ("correct", "true", "1", "yes", "y"):
            return True
        if lowered in ("incorrect", "false", "0", "no", "n"):
            return False
    return None


def safe_float(value: Any) -> Optional[float]:
    """Parse a finite float from mixed inputs, or return None."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(val):
        return val
    return None


def compute_row_complexity_c(row: Dict[str, Any]) -> Optional[float]:
    """Compute complexity C for a dataset row, preferring serialized complexity_c."""
    serialized = safe_float(row.get("complexity_c"))
    if serialized is not None:
        return serialized

    digit_total = safe_float(row.get("digit_total"))
    nonzero_total = safe_float(row.get("nonzero_total"))
    if digit_total is not None and nonzero_total is not None:
        return digit_total * nonzero_total

    try:
        a = abs(int(row.get("a")))
        b = abs(int(row.get("b")))
    except (TypeError, ValueError):
        return None

    digit_total_int = len(str(a)) + len(str(b))
    nonzero_total_int = sum(ch != "0" for ch in f"{a}{b}")
    return float(digit_total_int * nonzero_total_int)


def compute_probe_dataset_summary(
    dataset_name: str,
    split: str = "test",
    result_rows: Optional[Sequence[Dict[str, Any]]] = None,
    detail_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarize the evaluated probe dataset count and mean complexity."""
    dataset_path = get_dataset_csv_path(dataset_name)
    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "count": 0,
        "mean_c": None,
        "mean_c_display": "[MISSING]",
        "dataset_path": str(dataset_path),
    }
    if not dataset_path.exists():
        return summary

    evaluated_ids = {
        str(row.get("hds_id")).strip()
        for row in (result_rows or [])
        if row.get("hds_id") is not None and str(row.get("hds_id")).strip()
    }
    evaluated_ids.update(
        str(row.get("hds_id")).strip()
        for row in (detail_rows or [])
        if row.get("hds_id") is not None and str(row.get("hds_id")).strip()
    )

    complexities: List[float] = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = str(row.get("id", "")).strip()
            if evaluated_ids and row_id not in evaluated_ids:
                continue
            if not evaluated_ids and split != "all":
                row_split = str(row.get("split", "")).strip()
                if row_split and row_split != split:
                    continue
            complexity_c = compute_row_complexity_c(row)
            if complexity_c is not None:
                complexities.append(complexity_c)

    if complexities:
        mean_c = float(np.mean(np.array(complexities, dtype=float)))
        summary["count"] = len(complexities)
        summary["mean_c"] = mean_c
        summary["mean_c_display"] = f"{mean_c:.1f}"

    return summary


def normalize_heuristic_label(value: Any) -> str:
    """Normalize serialized heuristic labels to uppercase names."""
    if value is None:
        return "UNKNOWN"
    label = str(value).strip().upper()
    return label if label else "UNKNOWN"


def get_design_family(record: Dict[str, Any]) -> str:
    """Return the lexical design family for a result or detail record."""
    return normalize_heuristic_label(record.get("design_family") or record.get("target_heuristic"))


def get_canonical_target(record: Dict[str, Any]) -> str:
    """Return the canonical cost-model target for a result or detail record."""
    return normalize_heuristic_label(record.get("canonical_target_heuristic") or record.get("target_heuristic"))


def get_family_match_rate(analysis: Optional[Dict[str, Any]]) -> float:
    """Return lexical family-match rate with backward compatibility."""
    if not analysis:
        return 0.0
    return float(analysis.get("family_match_rate", analysis.get("heuristic_match", 0.0)) or 0.0)


def get_family_breakdown(analysis: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return lexical family grouping with backward compatibility."""
    if not analysis:
        return {}
    return analysis.get("by_design_family", analysis.get("by_target_heuristic", {})) or {}


def get_family_detection_rate(stats: Optional[Dict[str, Any]]) -> Optional[float]:
    """Return the family-level detection rate across HDS and HDSv2 schemas."""
    if not stats:
        return None
    value = stats.get("family_match_rate", stats.get("detection_rate"))
    return float(value) if isinstance(value, (int, float)) else safe_float(value)


def get_canonical_breakdown(analysis: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return grounded canonical-target grouping with backward compatibility."""
    if not analysis:
        return {}
    return analysis.get("by_canonical_target_heuristic", analysis.get("by_target_heuristic", {})) or {}


def get_loss_detected_heuristic(row: Dict[str, Any]) -> str:
    """Return the canonical loss-based detected heuristic from a flat result row."""
    explicit = row.get("loss_best_heuristic")
    fallback = row.get("detected_heuristic")
    detected = normalize_heuristic_label(explicit or fallback)
    if explicit and fallback and normalize_heuristic_label(explicit) != normalize_heuristic_label(fallback):
        raise ValueError(
            "Fingerprint results CSV contains inconsistent loss-best and detected heuristics: "
            f"{explicit!r} vs {fallback!r}"
        )
    return detected


def get_loss_detection_confidence(row: Dict[str, Any]) -> float:
    """Return the canonical loss-based detection confidence from a flat result row."""
    explicit = safe_float(row.get("loss_best_confidence"))
    if explicit is not None:
        return explicit
    fallback = safe_float(row.get("detection_confidence"))
    return fallback if fallback is not None else 0.0


def infer_loss_best_heuristic_from_detail(record: Dict[str, Any]) -> str:
    """Recompute the lowest-loss heuristic from a fingerprint detail record."""
    aggregated = record.get("perplexity", {}).get("aggregated", {})
    finite_losses = {
        heuristic: value
        for heuristic, value in (
            (heuristic, safe_float(loss))
            for heuristic, loss in aggregated.items()
        )
        if value is not None
    }
    if not finite_losses:
        return "UNKNOWN"

    best = min(finite_losses, key=finite_losses.get)
    serialized = record.get("perplexity", {}).get("loss_best_heuristic")
    if serialized is not None and normalize_heuristic_label(serialized) != best:
        raise ValueError(
            "Fingerprint detail record contains inconsistent serialized loss-best heuristic: "
            f"{serialized!r} vs recomputed {best!r}"
        )
    return best


def compute_support_mass_from_detail(record: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Return stored or recomputed soft support mass over heuristics plus neutral."""
    def _finite_detail_float(value: Any) -> Optional[float]:
        parsed = safe_float(value)
        if parsed is None or not math.isfinite(parsed):
            return None
        return float(parsed)

    stored = (record.get("perplexity") or {}).get("support_mass")
    if isinstance(stored, dict):
        parsed_stored = {
            key: _finite_detail_float(value)
            for key, value in stored.items()
        }
        if all(parsed_stored.get(key) is not None for key in ("DD", "OT", "RC", "NEUTRAL")):
            return {key: float(parsed_stored[key]) for key in ("DD", "OT", "RC", "NEUTRAL")}

    aggregated = (record.get("perplexity") or {}).get("aggregated") or {}
    resolved_losses: Dict[str, float] = {}
    for heuristic in ("DD", "OT", "RC"):
        parsed = _finite_detail_float(aggregated.get(heuristic))
        if parsed is None:
            return None
        resolved_losses[heuristic] = parsed

    neutral_loss = _finite_detail_float((record.get("perplexity") or {}).get("neutral_loss"))
    if neutral_loss is None:
        return None
    resolved_losses["NEUTRAL"] = neutral_loss

    best_loss = min(resolved_losses.values())
    shifted = {
        name: math.exp(-(loss - best_loss))
        for name, loss in resolved_losses.items()
    }
    denom = sum(shifted.values())
    if denom <= 0:
        return None
    return {name: value / denom for name, value in shifted.items()}


def infer_embedding_best_heuristic_from_detail(record: Dict[str, Any]) -> str:
    """Return the embedding detector's best label from a detail record."""
    serialized = (record.get("embedding_analysis") or {}).get("embedding_heuristic")
    if serialized is not None:
        return normalize_heuristic_label(serialized)

    support_mass = compute_embedding_support_mass_from_detail(record)
    if not support_mass:
        return "UNKNOWN"
    return max(support_mass, key=support_mass.get)


def compute_embedding_support_mass_from_detail(record: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Return STYLE-aware embedding support mass when available."""
    stored = (record.get("embedding_analysis") or {}).get("support_mass")
    if not isinstance(stored, dict):
        return None

    parsed: Dict[str, float] = {}
    for key in ("DD", "OT", "RC", "STYLE"):
        value = safe_float(stored.get(key))
        if value is None:
            return None
        parsed[key] = float(value)
    return parsed


def is_resolved_embedding_detail(record: Dict[str, Any]) -> bool:
    """Return whether a detail row has a resolved embedding-side trace classifier result."""
    explicit = (record.get("embedding_analysis") or {}).get("resolved")
    if explicit is not None:
        return parse_bool(explicit)
    return compute_embedding_support_mass_from_detail(record) is not None


def is_resolved_probe_detail(record: Dict[str, Any]) -> bool:
    """Return whether a detail row has a fully resolved heuristic+neutral probe."""
    explicit = (record.get("perplexity") or {}).get("probe_resolved")
    if explicit is not None:
        return parse_bool(explicit)
    return compute_support_mass_from_detail(record) is not None


def get_soft_target_stats(analysis: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return paper-facing soft target stats with backward compatibility."""
    if not analysis:
        return {}

    table3_stats = analysis.get("table3_alt_stats")
    if isinstance(table3_stats, dict) and table3_stats.get("soft_target_stats"):
        return table3_stats.get("soft_target_stats") or {}

    if analysis.get("soft_target_stats"):
        return analysis.get("soft_target_stats") or {}

    by_family = analysis.get("by_design_family", analysis.get("by_target_heuristic", {})) or {}
    fallback: Dict[str, Dict[str, Any]] = {}
    for heuristic in ("DD", "OT", "RC"):
        family_stats = by_family.get(heuristic, {}) or {}
        fallback[heuristic] = {
            "coverage_rate": safe_float(family_stats.get("coverage_rate")) or 0.0,
            "coverage_se": safe_float(family_stats.get("coverage_se")) or 0.0,
            "resolved_n": safe_float(family_stats.get("resolved_probe_count")) or 0.0,
            "total_n": safe_float(family_stats.get("total")) or 0.0,
            "target_support": safe_float(family_stats.get("target_support_mean")) or 0.0,
            "target_support_se": safe_float(family_stats.get("target_support_se")) or 0.0,
        }
    return fallback


def get_soft_target_support_series(analysis: Optional[Dict[str, Any]], heuristics: Sequence[str]) -> List[float]:
    """Return per-family target support percentages for paper-facing plots."""
    stats = get_soft_target_stats(analysis)
    return [float((stats.get(h, {}) or {}).get("target_support", 0.0)) * 100 for h in heuristics]


def _filter_nudge_analysis_by_heuristics(
    analysis: Optional[Dict[str, Any]],
    heuristics: Sequence[str],
) -> Dict[str, Any]:
    """Return a filtered nudge-analysis mapping for selected LoRAs."""
    if not analysis:
        return {}
    allowed = {normalize_heuristic_label(heuristic) for heuristic in heuristics}
    return {
        heuristic: payload
        for heuristic, payload in analysis.items()
        if normalize_heuristic_label(heuristic) in allowed
    }


def _filter_nudge_result_rows(
    rows: Optional[List[Dict[str, Any]]],
    heuristics: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return nudge CSV rows for selected LoRAs only."""
    if not rows:
        return []
    allowed = {normalize_heuristic_label(heuristic) for heuristic in heuristics}
    return [
        row for row in rows
        if normalize_heuristic_label(row.get("lora")) in allowed
    ]


def _filter_nudge_detail_rows(
    rows: Optional[List[Dict[str, Any]]],
    allowed_lora_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return nudge detail rows with only the selected LoRA evaluations."""
    if not rows:
        return []
    allowed = set(allowed_lora_keys)
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        lora_evaluations = dict(copied.get("lora_evaluations") or {})
        copied["lora_evaluations"] = {
            key: value for key, value in lora_evaluations.items()
            if key in allowed
        }
        filtered.append(copied)
    return filtered


def compute_binary_se(flags: Sequence[bool]) -> float:
    """Compute standard error (percentage points) for a binary proportion."""
    n = len(flags)
    if n == 0:
        return 0.0
    p = sum(1 for f in flags if f) / n
    return math.sqrt(p * (1 - p) / n) * 100


def mean_and_se(values: Sequence[Union[int, float]]) -> Tuple[float, float]:
    """Compute mean and standard error for a list of numeric values."""
    cleaned = [v for v in (safe_float(v) for v in values) if v is not None]
    if not cleaned:
        return (0.0, 0.0)
    arr = np.array(cleaned, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return (mean, 0.0)
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return (mean, se)


def extract_perplexity_values(
    details: Sequence[dict],
    heuristics: Sequence[str] = ("DD", "OT", "RC")
) -> Dict[str, List[float]]:
    """Extract per-problem perplexity values from details JSONL entries."""
    values = {h: [] for h in heuristics}
    for row in details:
        agg = row.get("perplexity", {}).get("aggregated", {})
        for h in heuristics:
            val = safe_float(agg.get(h))
            if val is not None:
                values[h].append(val)
    return values


def extract_loss_pairs(
    details: Sequence[dict],
    num_key: str,
    denom_key: str
) -> List[Tuple[float, float]]:
    """Extract paired loss values for ratio-based SE calculations."""
    pairs = []
    for row in details:
        agg = row.get("perplexity", {}).get("aggregated", {})
        num = safe_float(agg.get(num_key))
        denom = safe_float(agg.get(denom_key))
        if num is None or denom is None:
            continue
        pairs.append((num, denom))
    return pairs


def build_perplexity_by_id(details: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    """Map hds_id to aggregated perplexity values."""
    by_id: Dict[str, Dict[str, float]] = {}
    for row in details:
        hds_id = row.get("hds_id")
        if not hds_id:
            continue
        agg = row.get("perplexity", {}).get("aggregated", {})
        parsed = {}
        for h in ("DD", "OT", "RC"):
            val = safe_float(agg.get(h))
            if val is None:
                break
            parsed[h] = val
        if len(parsed) == 3:
            by_id[str(hds_id)] = parsed
    return by_id


def percent_ratio_se(pairs: Sequence[Tuple[float, float]]) -> float:
    """Delta-method SE for 100 * ((num - denom) / denom)."""
    if len(pairs) < 2:
        return 0.0
    arr = np.array(pairs, dtype=float)
    num = arr[:, 0]
    denom = arr[:, 1]
    mean_num = float(np.mean(num))
    mean_denom = float(np.mean(denom))
    if mean_denom == 0:
        return 0.0
    var_num = float(np.var(num, ddof=1) / len(num))
    var_denom = float(np.var(denom, ddof=1) / len(denom))
    cov = float(np.cov(num, denom, ddof=1)[0, 1] / len(num))
    d_num = 100 / mean_denom
    d_denom = -100 * mean_num / (mean_denom ** 2)
    var = (d_num ** 2) * var_num + (d_denom ** 2) * var_denom + 2 * d_num * d_denom * cov
    return math.sqrt(var) if var > 0 else 0.0


def compute_table3_stats_from_jsonl(
    records: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Compute Table 3 statistics directly from JSONL records.

    This provides an alternative computation path that doesn't rely on
    fingerprint_analysis.json, useful for:
    - Real-time stats from in-progress jobs
    - Verification of pre-computed statistics
    - Fallback when analysis.json is missing

    Args:
        records: List of JSONL records from fingerprint_details.jsonl

    Returns:
        Dict with keys: n, accuracy, accuracy_se, neutral_loss,
        delta_losses (DD/OT/RC), detection_rates (DD/OT/RC + SEs)
        Returns None if records is empty.
    """
    if not records:
        return None

    n = len(records)

    # Accuracy
    correct_flags = [r.get('generation', {}).get('is_correct', False) for r in records]
    accuracy = sum(correct_flags) / n
    accuracy_se = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0.0

    # Perplexity aggregates
    dd_losses: List[float] = []
    ot_losses: List[float] = []
    rc_losses: List[float] = []
    neutral_losses: List[float] = []

    for r in records:
        perp = r.get('perplexity', {})
        agg = perp.get('aggregated', {})

        dd = safe_float(agg.get('DD'))
        ot = safe_float(agg.get('OT'))
        rc = safe_float(agg.get('RC'))
        neutral = safe_float(perp.get('neutral_loss'))

        if dd is not None:
            dd_losses.append(dd)
        if ot is not None:
            ot_losses.append(ot)
        if rc is not None:
            rc_losses.append(rc)
        if neutral is not None:
            neutral_losses.append(neutral)

    avg_dd, se_dd = mean_and_se(dd_losses)
    avg_ot, se_ot = mean_and_se(ot_losses)
    avg_rc, se_rc = mean_and_se(rc_losses)
    avg_neutral, se_neutral = mean_and_se(neutral_losses)

    # Delta losses (relative to neutral), computed per-example when available.
    delta_dd_values: List[float] = []
    delta_ot_values: List[float] = []
    delta_rc_values: List[float] = []
    for r in records:
        delta_losses = r.get("perplexity", {}).get("delta_losses", {}) or {}
        dd_delta = safe_float(delta_losses.get("DD"))
        ot_delta = safe_float(delta_losses.get("OT"))
        rc_delta = safe_float(delta_losses.get("RC"))
        if dd_delta is not None:
            delta_dd_values.append(dd_delta)
        if ot_delta is not None:
            delta_ot_values.append(ot_delta)
        if rc_delta is not None:
            delta_rc_values.append(rc_delta)

    delta_dd = (sum(delta_dd_values) / len(delta_dd_values)) if delta_dd_values else (avg_dd - avg_neutral if avg_neutral else 0.0)
    delta_ot = (sum(delta_ot_values) / len(delta_ot_values)) if delta_ot_values else (avg_ot - avg_neutral if avg_neutral else 0.0)
    delta_rc = (sum(delta_rc_values) / len(delta_rc_values)) if delta_rc_values else (avg_rc - avg_neutral if avg_neutral else 0.0)

    resolved_probe_count = sum(1 for record in records if is_resolved_probe_detail(record))
    resolved_probe_rate = resolved_probe_count / n if n > 0 else 0.0
    resolved_probe_rate_se = math.sqrt(resolved_probe_rate * (1 - resolved_probe_rate) / n) if n > 0 else 0.0
    resolved_embedding_count = sum(1 for record in records if is_resolved_embedding_detail(record))
    resolved_embedding_rate = resolved_embedding_count / n if n > 0 else 0.0
    resolved_embedding_rate_se = math.sqrt(
        resolved_embedding_rate * (1 - resolved_embedding_rate) / n
    ) if n > 0 else 0.0

    # Legacy argmin detection rates by design family
    detection_rates: Dict[str, Dict[str, float]] = {}
    soft_target_stats: Dict[str, Dict[str, float]] = {}
    embedding_soft_target_stats: Dict[str, Dict[str, float]] = {}
    for h in ['DD', 'OT', 'RC']:
        target_records = [r for r in records if get_design_family(r) == h]
        if target_records:
            detected = sum(
                1
                for r in target_records
                if infer_loss_best_heuristic_from_detail(r) == h
            )
            rate = detected / len(target_records)
            se = math.sqrt(rate * (1 - rate) / len(target_records))
            detection_rates[h] = {'rate': rate, 'se': se, 'n': float(len(target_records))}

            resolved_target_records = []
            target_support_values: List[float] = []
            for record in target_records:
                support_mass = compute_support_mass_from_detail(record)
                if support_mass is None:
                    continue
                resolved_target_records.append(record)
                target_support_values.append(support_mass[h])

            coverage_rate = len(resolved_target_records) / len(target_records)
            coverage_se = math.sqrt(coverage_rate * (1 - coverage_rate) / len(target_records))
            target_support_mean, target_support_se = mean_and_se(target_support_values)
            soft_target_stats[h] = {
                "coverage_rate": coverage_rate,
                "coverage_se": coverage_se,
                "resolved_n": float(len(resolved_target_records)),
                "total_n": float(len(target_records)),
                "target_support": target_support_mean,
                "target_support_se": target_support_se,
            }

            resolved_embedding_records = []
            embedding_target_support_values: List[float] = []
            embedding_detected = 0
            for record in target_records:
                support_mass = compute_embedding_support_mass_from_detail(record)
                if support_mass is None:
                    continue
                resolved_embedding_records.append(record)
                embedding_target_support_values.append(support_mass[h])
                if infer_embedding_best_heuristic_from_detail(record) == h:
                    embedding_detected += 1

            embedding_coverage_rate = len(resolved_embedding_records) / len(target_records)
            embedding_coverage_se = math.sqrt(
                embedding_coverage_rate * (1 - embedding_coverage_rate) / len(target_records)
            )
            embedding_target_support_mean, embedding_target_support_se = mean_and_se(embedding_target_support_values)
            if resolved_embedding_records:
                embedding_detection_rate = embedding_detected / len(resolved_embedding_records)
                embedding_detection_se = math.sqrt(
                    embedding_detection_rate
                    * (1 - embedding_detection_rate)
                    / len(resolved_embedding_records)
                )
            else:
                embedding_detection_rate = 0.0
                embedding_detection_se = 0.0

            embedding_soft_target_stats[h] = {
                "coverage_rate": embedding_coverage_rate,
                "coverage_se": embedding_coverage_se,
                "resolved_n": float(len(resolved_embedding_records)),
                "total_n": float(len(target_records)),
                "target_support": embedding_target_support_mean,
                "target_support_se": embedding_target_support_se,
                "detection_rate": embedding_detection_rate,
                "detection_se": embedding_detection_se,
            }
        else:
            detection_rates[h] = {'rate': 0.0, 'se': 0.0, 'n': 0.0}
            soft_target_stats[h] = {
                "coverage_rate": 0.0,
                "coverage_se": 0.0,
                "resolved_n": 0.0,
                "total_n": 0.0,
                "target_support": 0.0,
                "target_support_se": 0.0,
            }
            embedding_soft_target_stats[h] = {
                "coverage_rate": 0.0,
                "coverage_se": 0.0,
                "resolved_n": 0.0,
                "total_n": 0.0,
                "target_support": 0.0,
                "target_support_se": 0.0,
                "detection_rate": 0.0,
                "detection_se": 0.0,
            }

    return {
        'n': n,
        'accuracy': accuracy,
        'accuracy_se': accuracy_se,
        'neutral_loss': avg_neutral,
        'neutral_loss_se': se_neutral,
        'avg_perplexity': {'DD': avg_dd, 'OT': avg_ot, 'RC': avg_rc},
        'perplexity_se': {'DD': se_dd, 'OT': se_ot, 'RC': se_rc},
        'delta_loss': {'DD': delta_dd, 'OT': delta_ot, 'RC': delta_rc},
        'detection_rates': detection_rates,
        'resolved_probe_count': resolved_probe_count,
        'resolved_probe_rate': resolved_probe_rate,
        'resolved_probe_rate_se': resolved_probe_rate_se,
        'resolved_embedding_count': resolved_embedding_count,
        'resolved_embedding_rate': resolved_embedding_rate,
        'resolved_embedding_rate_se': resolved_embedding_rate_se,
        'soft_target_stats': soft_target_stats,
        'embedding_soft_target_stats': embedding_soft_target_stats,
        'support_measure_semantics': 'softmax_over_negative_loss_with_neutral',
        'embedding_support_measure_semantics': 'softmax_over_cosine_similarity_with_style',
    }


def generate_jsonl_alt_macros(
    text_details: Optional[List[Dict[str, Any]]],
    image_details: Optional[List[Dict[str, Any]]],
    model_suffix: str = ""
) -> str:
    """Generate Table 3 macros with 'Alt' suffix from JSONL files.

    These macros mirror the standard Table 3 macros but are computed
    directly from JSONL records rather than fingerprint_analysis.json.

    Args:
        text_details: JSONL records for text modality
        image_details: JSONL records for image modality
        model_suffix: Model-specific suffix (e.g., "ThirtyB")

    Returns:
        LaTeX macro definitions string
    """
    macros: List[str] = []
    suffix_label = f" - {model_suffix}" if model_suffix else ""
    macros.append(f"% Alternative Table 3 macros (computed from JSONL){suffix_label}")
    macros.append("% These 'Alt' macros are computed directly from fingerprint_details.jsonl")
    macros.append("")

    # Text modality
    text_stats = compute_table3_stats_from_jsonl(text_details) if text_details else None
    if text_stats:
        n = text_stats['n']
        acc = text_stats['accuracy'] * 100
        acc_se = text_stats['accuracy_se'] * 100
        neutral = text_stats['neutral_loss']
        resolved_rate = text_stats['resolved_probe_rate'] * 100
        resolved_rate_se = text_stats['resolved_probe_rate_se'] * 100
        resolved_count = text_stats['resolved_probe_count']

        macros.append(f"\\newcommand{{\\HDSTestCountAlt{model_suffix}}}{{{n}}}")
        macros.append(f"\\newcommand{{\\HDSTestAccuracyAlt{model_suffix}}}{{{acc:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestAccuracyAltSE{model_suffix}}}{{{acc_se:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestNeutralLossAlt{model_suffix}}}{{{neutral:.4f}}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedCountAlt{model_suffix}}}{{{resolved_count}}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedCoverageAlt{model_suffix}}}{{{resolved_rate:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedCoverageAltSE{model_suffix}}}{{{resolved_rate_se:.1f}\\%}}")

        # Delta losses
        delta = text_stats['delta_loss']
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\HDSTestDelta{h}Alt{model_suffix}}}{{{delta[h]:+.4f}}}")

        # Legacy argmin detection rates
        det = text_stats['detection_rates']
        for h in ['DD', 'OT', 'RC']:
            rate = det[h]['rate'] * 100
            se = det[h]['se'] * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionAlt{model_suffix}}}{{{rate:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionAltSE{model_suffix}}}{{{se:.1f}\\%}}")

        # Paper-facing coverage + soft target support
        soft_stats = text_stats['soft_target_stats']
        for h in ['DD', 'OT', 'RC']:
            coverage = soft_stats[h]['coverage_rate'] * 100
            coverage_se = soft_stats[h]['coverage_se'] * 100
            resolved_n = int(soft_stats[h]['resolved_n'])
            support = soft_stats[h]['target_support'] * 100
            support_se = soft_stats[h]['target_support_se'] * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}CoverageAlt{model_suffix}}}{{{coverage:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}CoverageAltSE{model_suffix}}}{{{coverage_se:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}ResolvedCountAlt{model_suffix}}}{{{resolved_n}}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TargetSupportAlt{model_suffix}}}{{{support:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TargetSupportAltSE{model_suffix}}}{{{support_se:.1f}\\%}}")

        embedding_resolved_rate = text_stats['resolved_embedding_rate'] * 100
        embedding_resolved_rate_se = text_stats['resolved_embedding_rate_se'] * 100
        embedding_resolved_count = text_stats['resolved_embedding_count']
        macros.append(f"\\newcommand{{\\HDSTestResolvedTraceEmbedCountAlt{model_suffix}}}{{{embedding_resolved_count}}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedTraceEmbedCoverageAlt{model_suffix}}}{{{embedding_resolved_rate:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedTraceEmbedCoverageAltSE{model_suffix}}}{{{embedding_resolved_rate_se:.1f}\\%}}")

        embedding_stats = text_stats['embedding_soft_target_stats']
        for h in ['DD', 'OT', 'RC']:
            coverage = embedding_stats[h]['coverage_rate'] * 100
            coverage_se = embedding_stats[h]['coverage_se'] * 100
            resolved_n = int(embedding_stats[h]['resolved_n'])
            support = embedding_stats[h]['target_support'] * 100
            support_se = embedding_stats[h]['target_support_se'] * 100
            detection = embedding_stats[h]['detection_rate'] * 100
            detection_se = embedding_stats[h]['detection_se'] * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedCoverageAlt{model_suffix}}}{{{coverage:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedCoverageAltSE{model_suffix}}}{{{coverage_se:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedResolvedCountAlt{model_suffix}}}{{{resolved_n}}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedTargetSupportAlt{model_suffix}}}{{{support:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedTargetSupportAltSE{model_suffix}}}{{{support_se:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedDetectionAlt{model_suffix}}}{{{detection:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedDetectionAltSE{model_suffix}}}{{{detection_se:.1f}\\%}}")

    macros.append("")

    # Image modality
    image_stats = compute_table3_stats_from_jsonl(image_details) if image_details else None
    if image_stats:
        n = image_stats['n']
        acc = image_stats['accuracy'] * 100
        acc_se = image_stats['accuracy_se'] * 100
        neutral = image_stats['neutral_loss']
        resolved_rate = image_stats['resolved_probe_rate'] * 100
        resolved_rate_se = image_stats['resolved_probe_rate_se'] * 100
        resolved_count = image_stats['resolved_probe_count']

        macros.append(f"\\newcommand{{\\HDSTestCountImageAlt{model_suffix}}}{{{n}}}")
        macros.append(f"\\newcommand{{\\HDSTestAccuracyImageAlt{model_suffix}}}{{{acc:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestAccuracyImageAltSE{model_suffix}}}{{{acc_se:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestNeutralLossImageAlt{model_suffix}}}{{{neutral:.4f}}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedCountImageAlt{model_suffix}}}{{{resolved_count}}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedCoverageImageAlt{model_suffix}}}{{{resolved_rate:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedCoverageImageAltSE{model_suffix}}}{{{resolved_rate_se:.1f}\\%}}")

        # Delta losses
        delta = image_stats['delta_loss']
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\HDSTestDelta{h}ImageAlt{model_suffix}}}{{{delta[h]:+.4f}}}")

        # Legacy argmin detection rates
        det = image_stats['detection_rates']
        for h in ['DD', 'OT', 'RC']:
            rate = det[h]['rate'] * 100
            se = det[h]['se'] * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImageAlt{model_suffix}}}{{{rate:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImageAltSE{model_suffix}}}{{{se:.1f}\\%}}")

        # Paper-facing coverage + soft target support
        soft_stats = image_stats['soft_target_stats']
        for h in ['DD', 'OT', 'RC']:
            coverage = soft_stats[h]['coverage_rate'] * 100
            coverage_se = soft_stats[h]['coverage_se'] * 100
            resolved_n = int(soft_stats[h]['resolved_n'])
            support = soft_stats[h]['target_support'] * 100
            support_se = soft_stats[h]['target_support_se'] * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}CoverageImageAlt{model_suffix}}}{{{coverage:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}CoverageImageAltSE{model_suffix}}}{{{coverage_se:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}ResolvedCountImageAlt{model_suffix}}}{{{resolved_n}}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TargetSupportImageAlt{model_suffix}}}{{{support:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TargetSupportImageAltSE{model_suffix}}}{{{support_se:.1f}\\%}}")

        embedding_resolved_rate = image_stats['resolved_embedding_rate'] * 100
        embedding_resolved_rate_se = image_stats['resolved_embedding_rate_se'] * 100
        embedding_resolved_count = image_stats['resolved_embedding_count']
        macros.append(f"\\newcommand{{\\HDSTestResolvedTraceEmbedCountImageAlt{model_suffix}}}{{{embedding_resolved_count}}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedTraceEmbedCoverageImageAlt{model_suffix}}}{{{embedding_resolved_rate:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\HDSTestResolvedTraceEmbedCoverageImageAltSE{model_suffix}}}{{{embedding_resolved_rate_se:.1f}\\%}}")

        embedding_stats = image_stats['embedding_soft_target_stats']
        for h in ['DD', 'OT', 'RC']:
            coverage = embedding_stats[h]['coverage_rate'] * 100
            coverage_se = embedding_stats[h]['coverage_se'] * 100
            resolved_n = int(embedding_stats[h]['resolved_n'])
            support = embedding_stats[h]['target_support'] * 100
            support_se = embedding_stats[h]['target_support_se'] * 100
            detection = embedding_stats[h]['detection_rate'] * 100
            detection_se = embedding_stats[h]['detection_se'] * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedCoverageImageAlt{model_suffix}}}{{{coverage:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedCoverageImageAltSE{model_suffix}}}{{{coverage_se:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedResolvedCountImageAlt{model_suffix}}}{{{resolved_n}}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedTargetSupportImageAlt{model_suffix}}}{{{support:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedTargetSupportImageAltSE{model_suffix}}}{{{support_se:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedDetectionImageAlt{model_suffix}}}{{{detection:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}TraceEmbedDetectionImageAltSE{model_suffix}}}{{{detection_se:.1f}\\%}}")

    return "\n".join(macros)


# Valid output types for --output-type flag
VALID_OUTPUT_TYPES = [
    'all',
    'fingerprint-figures',
    'fingerprint-macros',
    'fingerprint-appendix',
    'nudge-macros',
    'nudge-appendix',
    'gradient-macros',
    'gradient-appendix',
    'embedding-appendix',
    'training-appendix',
    'merge-macros',
    'merge-fingerprint-appendix',
    'merge-nudge-appendix',
    'merge-gradient-appendix',
    'merge-embedding-appendix',
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fine-grained figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate figures and LaTeX macros from experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output types:
  all                 Generate everything (default)
  fingerprint-figures Generate perplexity/detection/confusion figures
  fingerprint-macros  Generate fingerprint-related LaTeX macros
  fingerprint-appendix Generate template-variability appendix
  nudge-macros        Generate nudge test LaTeX macros
  nudge-appendix      Generate nudge examples appendix
  gradient-macros     Generate gradient/orthogonality LaTeX macros
  gradient-appendix   Generate similarity matrix appendix
  embedding-appendix  Generate embedding-based fingerprint appendix
  training-appendix   Generate training examples appendix
  merge-macros        Merge macro fragments into the canonical macro file
  merge-fingerprint-appendix Merge model-specific template-variability appendix fragments
  merge-nudge-appendix Merge model-specific nudge appendix fragments
  merge-gradient-appendix Merge model-specific similarity appendix fragments
  merge-embedding-appendix Merge model-specific embedding appendix fragments

Examples:
  # Generate everything for a specific model
  python GenerateResultsFigures.py --model Qwen3-VL-30B-A3B

  # Generate only fingerprint figures (for 30B)
  python GenerateResultsFigures.py --model Qwen3-VL-30B-A3B --output-type fingerprint-figures

  # Generate macros for 235B model, appending to existing file
  python GenerateResultsFigures.py --model Qwen3-VL-235B-A22B --output-type fingerprint-macros --append-macros
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Only process this model (e.g., Qwen3-VL-30B-A3B). If not specified, discovers all models.'
    )

    parser.add_argument(
        '--output-type',
        type=str,
        default='all',
        choices=VALID_OUTPUT_TYPES,
        help='Type of output to generate (default: all)'
    )

    parser.add_argument(
        '--append-macros',
        action='store_true',
        help='Append to existing macros file instead of overwriting'
    )

    parser.add_argument(
        '--output-path',
        type=Path,
        default=None,
        help='Override the destination for single-file outputs and merge outputs'
    )

    parser.add_argument(
        '--fragment-path',
        action='append',
        default=[],
        help='Fragment path to merge for merge-* output types (repeatable)'
    )

    parser.add_argument(
        '--probe-hds-dataset',
        type=str,
        default='HDSv2',
        help='Dataset label backing the main held-out fingerprint tables (default: HDSv2)'
    )

    return parser.parse_args()


def write_macros_with_lock(
    content: str,
    append: bool = False,
    macro_path: Optional[Path] = None,
) -> None:
    """Write macros to file with file locking for concurrent safety.

    Args:
        content: The macro content to write
        append: If True, append to existing file; if False, overwrite
    """
    macro_path = macro_path or (OUTPUT_DIR / "results_macros.tex")
    mode = 'a' if append else 'w'

    with open(macro_path, mode) as f:
        # Acquire exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            if not append:
                f.write("% Auto-generated LaTeX macros for fingerprinting results\n")
                f.write("% Generated by Scripts/analysis/GenerateResultsFigures.py\n\n")
            f.write(content)
            if not content.endswith('\n'):
                f.write('\n')
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    tprint(f"{'Appended' if append else 'Wrote'} macros to {macro_path}")


def write_text_artifact(content: str, output_path: Path) -> None:
    """Write a single text artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
        if content and not content.endswith('\n'):
            f.write('\n')
    tprint(f"Wrote artifact to {output_path}")


LATEX_LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
LATEX_TABLE_PATTERN = re.compile(
    r"\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}",
    flags=re.DOTALL,
)


def validate_unique_latex_labels(content: str, context: str) -> None:
    """Fail fast when generated LaTeX contains duplicate labels."""
    counts: Dict[str, int] = {}
    for label in LATEX_LABEL_PATTERN.findall(content):
        counts[label] = counts.get(label, 0) + 1
    duplicates = sorted(label for label, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate LaTeX labels in {context}: {duplicates}")


def merge_text_fragments(
    fragment_paths: Sequence[Path],
    output_path: Path,
    dedupe_newcommands: bool = False,
) -> None:
    """Merge text fragments in the provided order."""
    missing = [str(path) for path in fragment_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fragment(s) for merge: {missing}")

    merged_parts = []
    seen_macros = set()
    newcommand_pattern = re.compile(r"\\newcommand\{\\([^}]+)\}")
    for fragment_path in fragment_paths:
        fragment_text = fragment_path.read_text().rstrip()
        if not dedupe_newcommands:
            merged_parts.append(fragment_text)
            continue

        filtered_lines = []
        for line in fragment_text.splitlines():
            match = newcommand_pattern.match(line.strip())
            if match:
                macro_name = match.group(1)
                if macro_name in seen_macros:
                    continue
                seen_macros.add(macro_name)
            filtered_lines.append(line)
        merged_parts.append("\n".join(filtered_lines).rstrip())

    merged_content = "\n\n".join(part for part in merged_parts if part)
    write_text_artifact(merged_content, output_path)


def infer_fragment_model_label(fragment_path: Path, fragment_text: Optional[str] = None) -> str:
    """Infer a short model label from a fragment path or embedded comment."""
    if fragment_text:
        for line in fragment_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("% Model:"):
                return stripped.split(":", 1)[1].strip()

    match = re.search(r"(\d+)b", fragment_path.name, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}B"
    return fragment_path.stem


def extract_latex_tables(fragment_text: str, context: str) -> List[str]:
    """Extract top-level LaTeX table blocks from a fragment."""
    tables = LATEX_TABLE_PATTERN.findall(fragment_text)
    if not tables:
        raise ValueError(f"No LaTeX tables found in {context}")
    return tables


def extract_table_body_lines(table_text: str, context: str) -> List[str]:
    """Extract table rows between \\midrule and \\bottomrule."""
    match = re.search(r"\\midrule\s*(.*?)\s*\\bottomrule", table_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not locate table body in {context}")
    return [line.rstrip() for line in match.group(1).splitlines() if line.strip()]


def extract_comment_values(fragment_text: str, prefix: str) -> List[str]:
    """Return comment suffixes for lines that start with the given prefix."""
    values = []
    for line in fragment_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            values.append(stripped[len(prefix):].strip())
    return values


def build_grouped_appendix_table(
    *,
    comment_lines: Sequence[str],
    caption: str,
    label: str,
    column_spec: str,
    header_lines: Sequence[str],
    grouped_rows: Sequence[Tuple[str, Sequence[str]]],
    column_count: int,
) -> str:
    """Build a single canonical appendix table with model-grouped rows."""
    lines = list(comment_lines)
    if lines:
        lines.append("")
    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
    ])
    lines.extend(header_lines)
    for index, (model_label, rows) in enumerate(grouped_rows):
        if index > 0:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{column_count}}}{{@{{}}l@{{}}}}{{\\textbf{{{model_label}}}}} \\\\"
        )
        lines.extend(rows)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def merge_nudge_appendix_fragments(
    fragment_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Merge model-specific nudge appendix fragments into one canonical table."""
    missing = [str(path) for path in fragment_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fragment(s) for merge: {missing}")

    grouped_rows: List[Tuple[str, List[str]]] = []
    paired_modes: set[bool] = set()
    for fragment_path in fragment_paths:
        fragment_text = fragment_path.read_text()
        tables = extract_latex_tables(fragment_text, str(fragment_path))
        if len(tables) != 1:
            raise ValueError(
                f"Expected exactly one nudge appendix table in {fragment_path}, found {len(tables)}"
            )
        model_label = infer_fragment_model_label(fragment_path, fragment_text)
        grouped_rows.append((model_label, extract_table_body_lines(tables[0], str(fragment_path))))
        paired_modes.add("\\multicolumn{2}{c}{Image}" in tables[0])

    if len(paired_modes) != 1:
        raise ValueError("Cannot merge nudge appendix fragments with mixed text-only and paired-modality formats")

    has_image_columns = paired_modes.pop()
    if has_image_columns:
        content = build_grouped_appendix_table(
            comment_lines=[
                "% Auto-generated merged nudge test examples (text vs image)",
                "% Generated by Scripts/analysis/GenerateResultsFigures.py",
            ],
            caption="LoRA Nudge Test Examples by Model: Text vs Image Modality (lowest-loss heuristic)",
            label="tab:nudge-examples",
            column_spec="@{}lccccc@{}",
            header_lines=[
                "\\toprule",
                "Problem & Target & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Image} \\\\",
                "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
                "        &        & Base & LoRA & Base & LoRA \\\\",
                "\\midrule",
            ],
            grouped_rows=grouped_rows,
            column_count=6,
        )
    else:
        content = build_grouped_appendix_table(
            comment_lines=[
                "% Auto-generated merged nudge test examples",
                "% Generated by Scripts/analysis/GenerateResultsFigures.py",
            ],
            caption="LoRA Nudge Test Results by Model: Base Model vs Adapter Behavior",
            label="tab:nudge-examples",
            column_spec="@{}lcccccc@{}",
            header_lines=[
                "\\toprule",
                "Problem & Target & Base & RC LoRA & DD LoRA & OT LoRA \\\\",
                "\\midrule",
            ],
            grouped_rows=grouped_rows,
            column_count=6,
        )

    validate_unique_latex_labels(content, str(output_path))
    write_text_artifact(content, output_path)


def merge_template_variability_appendix_fragments(
    fragment_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Merge template-variability appendix fragments into one canonical table."""
    missing = [str(path) for path in fragment_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fragment(s) for merge: {missing}")

    grouped_rows: List[Tuple[str, List[str]]] = []
    for fragment_path in fragment_paths:
        fragment_text = fragment_path.read_text()
        tables = extract_latex_tables(fragment_text, str(fragment_path))
        if len(tables) != 1:
            raise ValueError(
                f"Expected exactly one template-variability table in {fragment_path}, found {len(tables)}"
            )
        model_label = infer_fragment_model_label(fragment_path, fragment_text)
        grouped_rows.append((model_label, extract_table_body_lines(tables[0], str(fragment_path))))

    content = build_grouped_appendix_table(
        comment_lines=[
            "% Auto-generated merged template-variability appendix",
            "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        ],
        caption="Template-variability robustness summary by model (mean within-problem standard deviation across paraphrase losses)",
        label="tab:template-variability",
        column_spec="@{}lcccc@{}",
        header_lines=[
            "\\toprule",
            "Profile & DD std & OT std & RC std & Mean std \\\\",
            "\\midrule",
        ],
        grouped_rows=grouped_rows,
        column_count=5,
    )

    validate_unique_latex_labels(content, str(output_path))
    write_text_artifact(content, output_path)


def merge_embedding_results_appendix_fragments(
    fragment_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Merge embedding appendix fragments into one canonical snippet."""
    missing = [str(path) for path in fragment_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fragment(s) for merge: {missing}")

    grouped_rows: List[Tuple[str, List[str]]] = []
    summary_clauses: List[str] = []
    detection_clauses: List[str] = []

    for fragment_path in fragment_paths:
        fragment_text = fragment_path.read_text()
        tables = extract_latex_tables(fragment_text, str(fragment_path))
        if len(tables) != 1:
            raise ValueError(
                f"Expected exactly one embedding appendix table in {fragment_path}, found {len(tables)}"
            )
        model_label = infer_fragment_model_label(fragment_path, fragment_text)
        model_suffix = get_model_macro_suffix(model_label)
        grouped_rows.append((model_label, extract_table_body_lines(tables[0], str(fragment_path))))
        summary_clauses.append(
            f"{model_label}: DD/OT contribute "
            f"\\HDSTestTraceEmbedDDCountImage{model_suffix}{{}} / "
            f"\\HDSTestTraceEmbedOTCountImage{model_suffix}{{}} of "
            f"\\HDSTestTraceEmbedResolvedTotalImage{model_suffix}{{}} resolved traces, "
            f"with RC/STYLE at \\HDSTestTraceEmbedRCCountImage{model_suffix}{{}} / "
            f"\\HDSTestTraceEmbedSTYLECountImage{model_suffix}{{}}"
        )
        detection_clauses.append(
            f"{model_label}: DD \\HDSTestDDTraceEmbedDetectionImageAlt{model_suffix}{{}}, "
            f"OT \\HDSTestOTTraceEmbedDetectionImageAlt{model_suffix}{{}}, "
            f"RC \\HDSTestRCTraceEmbedDetectionImageAlt{model_suffix}{{}}"
        )

    table = build_grouped_appendix_table(
        comment_lines=[],
        caption="Embedding-based trace fingerprint summary by model (prototype classifier, image modality only)",
        label="tab:embedding-fingerprint",
        column_spec="@{}lccc@{}",
        header_lines=[
            "\\toprule",
            "Slice & Coverage & Target support & Detection \\\\",
            "\\midrule",
        ],
        grouped_rows=grouped_rows,
        column_count=4,
    ).rstrip()

    content = "\n".join([
        "% Auto-generated merged embedding fingerprint appendix",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        "",
        "\\paragraph{Embedding-based trace summary.} "
        "Prototype-embedding trace summaries are currently available only for the image "
        "fingerprint runs in the canonical artifacts. "
        + "; ".join(summary_clauses)
        + ".",
        "Across target-labeled image items, DD and OT remain stronger than RC in the "
        "embedding-based detector ("
        + "; ".join(detection_clauses)
        + ").",
        "",
        table,
        "",
    ])

    validate_unique_latex_labels(content, str(output_path))
    write_text_artifact(content, output_path)


def merge_similarity_appendix_fragments(
    fragment_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Merge similarity appendix fragments into canonical grouped tables."""
    missing = [str(path) for path in fragment_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fragment(s) for merge: {missing}")

    similarity_rows: List[Tuple[str, List[str]]] = []
    seed_rows: List[Tuple[str, List[str]]] = []
    average_comments: List[str] = []
    baseline_comments: List[str] = []
    orthogonality_note = False

    for fragment_path in fragment_paths:
        fragment_text = fragment_path.read_text()
        tables = extract_latex_tables(fragment_text, str(fragment_path))
        if len(tables) not in (1, 2):
            raise ValueError(
                f"Expected one or two similarity appendix tables in {fragment_path}, found {len(tables)}"
            )
        model_label = infer_fragment_model_label(fragment_path, fragment_text)
        similarity_rows.append((model_label, extract_table_body_lines(tables[0], str(fragment_path))))

        averages = extract_comment_values(fragment_text, "% Average off-diagonal similarity:")
        average_comments.extend([f"% {model_label} average off-diagonal similarity: {value}" for value in averages])

        orthogonality_lines = extract_comment_values(
            fragment_text,
            "% Values near 0 indicate orthogonal (independent) adapter directions",
        )
        orthogonality_note = orthogonality_note or bool(orthogonality_lines)

        if len(tables) == 2:
            seed_rows.append((model_label, extract_table_body_lines(tables[1], str(fragment_path))))
            baselines = extract_comment_values(fragment_text, "% Primary cross-heuristic baseline pairs:")
            baseline_comments.extend(
                [f"% {model_label} primary cross-heuristic baseline pairs: {value}" for value in baselines]
            )

    content_parts = [
        build_grouped_appendix_table(
            comment_lines=[
                "% Auto-generated merged cosine similarity matrix",
                "% Generated by Scripts/analysis/GenerateResultsFigures.py",
            ],
            caption="Adapter Weight Cosine Similarity Matrix by model (cosine over concatenated flattened LoRA A/B factors across all modules)",
            label="tab:similarity-matrix",
            column_spec="@{}lccc@{}",
            header_lines=[
                "\\toprule",
                " & DD & OT & RC \\\\",
                "\\midrule",
            ],
            grouped_rows=similarity_rows,
            column_count=4,
        ).rstrip(),
    ]

    if average_comments:
        content_parts.extend(average_comments)
    if orthogonality_note:
        content_parts.append("% Values near 0 indicate orthogonal (independent) adapter directions")

    if seed_rows:
        content_parts.append("")
        content_parts.append(
            build_grouped_appendix_table(
                comment_lines=[],
                caption="Effective-Update Seed-Control Summary by model (same-heuristic unseeded vs seeded controls compared against primary unseeded cross-heuristic pairs)",
                label="tab:similarity-seed-controls",
                column_spec="@{}lc@{}",
                header_lines=[
                    "\\toprule",
                    "Pair & Cosine \\\\",
                    "\\midrule",
                ],
                grouped_rows=seed_rows,
                column_count=2,
            ).rstrip()
        )
        if baseline_comments:
            content_parts.extend(baseline_comments)

    content = "\n".join(part for part in content_parts if part) + "\n"
    validate_unique_latex_labels(content, str(output_path))
    write_text_artifact(content, output_path)


def should_emit_appendix_labels(output_path: Optional[Path], canonical_name: str) -> bool:
    """Emit canonical labels only for canonical appendix artifacts."""
    if output_path is None:
        return True
    return output_path.name == canonical_name


def discover_model_variants() -> List[str]:
    """Discover available model variants from results directories.

    Returns list of model slugs found (e.g., ['Qwen3-VL-30B-A3B', 'Qwen3-VL-235B-A22B']).
    """
    model_slugs = set()

    # Look for fingerprint directories with model suffix
    for path in RESULTS_DIR.glob("fingerprint_*"):
        if path.is_dir():
            # Extract model slug from path like "fingerprint_hds_test_Qwen3-VL-30B-A3B"
            parts = path.name.split("_")
            # Find the model part (typically contains "VL" or is after known parts)
            for i, part in enumerate(parts):
                if "VL" in part or "Qwen" in part or "Llama" in part:
                    model_slug = "_".join(parts[i:])
                    model_slugs.add(model_slug)
                    break

    # Also check lora_training directories
    for path in RESULTS_DIR.glob("lora_training_*"):
        if path.is_dir():
            model_slug = path.name.replace("lora_training_", "")
            if model_slug and not model_slug.startswith("seed"):
                model_slugs.add(model_slug)

    return sorted(model_slugs)


def get_model_macro_suffix(model_slug: Optional[str]) -> str:
    """Convert model slug to LaTeX-safe macro suffix.

    Args:
        model_slug: Model identifier (e.g., "Qwen3-VL-30B-A3B")

    Returns:
        LaTeX-safe suffix (e.g., "ThirtyB", "TwoThirtyFiveB")
    """
    if not model_slug:
        return ""
    # Map known model sizes to readable suffixes
    if "30B" in model_slug:
        return "ThirtyB"
    elif "235B" in model_slug:
        return "TwoThirtyFiveB"
    elif "4B" in model_slug:
        return "FourB"
    elif "8B" in model_slug:
        return "EightB"
    elif "70B" in model_slug:
        return "SeventyB"
    else:
        # Fallback: convert digits/markers so the macro name stays TeX-safe.
        return macro_safe_token(model_slug)


def get_model_plot_label(model_slug: Optional[str]) -> str:
    """Short label for plots (e.g., 30B, 235B)."""
    if not model_slug:
        return "Legacy"
    match = re.search(r"(\d+)B", model_slug)
    if match:
        return f"{match.group(1)}B"
    return model_slug


def get_model_sort_key(model_slug: Optional[str]) -> int:
    """Numeric sort key based on model size (e.g., 30, 235)."""
    if not model_slug:
        return 0
    match = re.search(r"(\d+)B", model_slug)
    if match:
        return int(match.group(1))
    return 999


_NUMBER_WORDS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    13: "Thirteen",
    14: "Fourteen",
    15: "Fifteen",
    16: "Sixteen",
    17: "Seventeen",
    18: "Eighteen",
    19: "Nineteen",
    20: "Twenty",
    30: "Thirty",
    40: "Forty",
    50: "Fifty",
    60: "Sixty",
    70: "Seventy",
    80: "Eighty",
    90: "Ninety",
}


def number_to_words(n: int) -> str:
    """Convert a non-negative integer to ASCII words for macro-safe names."""
    if n < 0:
        raise ValueError("number_to_words expects non-negative integers")
    if n in _NUMBER_WORDS:
        return _NUMBER_WORDS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        tens_word = _NUMBER_WORDS[tens * 10]
        ones_word = _NUMBER_WORDS[ones] if ones else ""
        return tens_word + ones_word
    return "".join(_NUMBER_WORDS[int(ch)] for ch in str(n))


def macro_safe_token(label: str) -> str:
    """Map a label to a TeX-safe macro token (letters only)."""
    def repl(match: re.Match[str]) -> str:
        return number_to_words(int(match.group(0)))

    cleaned = re.sub(r"\d+", repl, label)
    return re.sub(r"[^A-Za-z]", "", cleaned)


def macro_safe_range_label(label: str) -> str:
    """Map numeric ranges (e.g., 9-16) to TeX-safe macro tokens."""
    return macro_safe_token(label.replace("-", "To").replace("+", "Plus"))


def _is_missing_model_file(path: Path, model_slug: Optional[str], label: str) -> bool:
    """Return True if model-scoped file is missing (caller should skip gracefully)."""
    if model_slug and not path.exists():
        tprint(f"  [SKIP] Missing {label} for {model_slug}: {path}")
        return True
    return False


def _build_fingerprint_result_dir(
    dataset: str,
    split: Optional[str] = None,
    modality: str = "text",
    model_slug: Optional[str] = None,
    output_tag: Optional[str] = None,
) -> Path:
    """Build a SavedResults path for fingerprint outputs."""
    name = f"fingerprint_{dataset.lower()}"
    if split and split != "all":
        name += f"_{split}"
    if modality and modality != "text":
        name += f"_{modality}"
    if output_tag:
        name += f"_{output_tag}"
    if model_slug:
        name += f"_{model_slug}"
    return RESULTS_DIR / name


def _dataset_result_aliases(dataset: str) -> List[str]:
    """Return new-first SavedResults dataset aliases for a logical dataset name."""
    normalized = dataset.lower()
    aliases = [normalized]
    if normalized == "hds":
        aliases.insert(0, "hdsv2")
    elif normalized == "traps":
        aliases.insert(0, "trapsv2")
    elif normalized == "hdsv2":
        aliases.append("hds")
    elif normalized == "trapsv2":
        aliases.append("traps")
    return list(dict.fromkeys(aliases))


def _candidate_fingerprint_artifact_paths(
    dataset: str,
    filename: str,
    split: Optional[str] = None,
    modality: str = "text",
    model_slug: Optional[str] = None,
    output_tag: Optional[str] = None,
) -> List[Path]:
    """Build candidate artifact paths across model-aware and legacy layouts."""
    candidate_paths: List[Path] = []
    result_dir = _build_fingerprint_result_dir(
        dataset,
        split=split,
        modality=modality,
        model_slug=model_slug,
        output_tag=output_tag,
    )
    candidate_paths.append(result_dir / filename)
    if not output_tag:
        candidate_paths.append(
            _build_fingerprint_result_dir(
                dataset,
                split=split,
                modality=modality,
            ) / filename
        )
    return candidate_paths


def _resolve_fingerprint_artifact_path(
    dataset: str,
    filename: str,
    split: Optional[str] = None,
    modality: str = "text",
    model_slug: Optional[str] = None,
    output_tag: Optional[str] = None,
    allow_alias_fallback: bool = True,
) -> Path:
    """Resolve a fingerprint artifact path, optionally forbidding dataset-alias fallback."""
    normalized = dataset.lower()
    exact_candidates = _candidate_fingerprint_artifact_paths(
        normalized,
        filename=filename,
        split=split,
        modality=modality,
        model_slug=model_slug,
        output_tag=output_tag,
    )
    exact_path = next((candidate for candidate in exact_candidates if candidate.exists()), None)
    if exact_path is not None:
        return exact_path

    alias_candidates: List[Path] = []
    for dataset_alias in _dataset_result_aliases(dataset):
        if dataset_alias == normalized:
            continue
        alias_candidates.extend(
            _candidate_fingerprint_artifact_paths(
                dataset_alias,
                filename=filename,
                split=split,
                modality=modality,
                model_slug=model_slug,
                output_tag=output_tag,
            )
        )

    if not allow_alias_fallback:
        alias_path = next((candidate for candidate in alias_candidates if candidate.exists()), None)
        if alias_path is not None:
            raise FileNotFoundError(
                "Requested fingerprint artifact is missing for the selected dataset family, "
                f"but an alias artifact exists: requested={exact_candidates[0]} alias={alias_path}"
            )
        return exact_candidates[0]

    candidate_paths = exact_candidates + alias_candidates
    return next((candidate for candidate in candidate_paths if candidate.exists()), candidate_paths[0])


def _compute_fingerprint_quality_status(unknown_rate: float) -> str:
    """Map unknown-rate diagnostics into a paper-facing quality status."""
    if unknown_rate > 0.50:
        return "invalid"
    if unknown_rate > 0.10:
        return "degraded"
    return "ok"


def _count_unknown_from_confusion_matrix(confusion_matrix: Any) -> int:
    """Recover UNKNOWN detections from a serialized confusion matrix when CSVs are absent."""
    if not isinstance(confusion_matrix, dict):
        return 0
    total = 0
    for key, value in confusion_matrix.items():
        if isinstance(key, str) and key.endswith("->UNKNOWN"):
            parsed = safe_float(value)
            if parsed is not None:
                total += int(parsed)
    return total


def _compute_quality_metrics_from_saved_artifacts(
    result_dir: Path,
    *,
    fallback_total: Optional[int],
) -> Dict[str, Any]:
    """Infer fingerprint quality metrics from sibling CSV/JSONL artifacts."""
    unknown_count = 0
    total = fallback_total or 0
    rows_path = result_dir / "fingerprint_results.csv"
    if rows_path.exists():
        with open(rows_path, "r") as f:
            rows = list(csv.DictReader(f))
        total = len(rows)
        unknown_count = sum(1 for row in rows if get_loss_detected_heuristic(row) == "UNKNOWN")

    nonfinite_loss_count = 0
    details_path = result_dir / "fingerprint_details.jsonl"
    if details_path.exists():
        detail_total = 0
        with open(details_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                detail_total += 1
                aggregated = (record.get("perplexity") or {}).get("aggregated") or {}
                neutral_loss = (record.get("perplexity") or {}).get("neutral_loss")
                losses = []
                for raw in aggregated.values():
                    try:
                        losses.append(float(raw))
                    except (TypeError, ValueError):
                        losses.append(float("nan"))
                has_nonfinite = not losses or any(not math.isfinite(loss) for loss in losses)
                if neutral_loss is not None:
                    try:
                        neutral_val = float(neutral_loss)
                    except (TypeError, ValueError):
                        neutral_val = float("nan")
                    has_nonfinite = has_nonfinite or (not math.isfinite(neutral_val))
                if has_nonfinite:
                    nonfinite_loss_count += 1
        if detail_total:
            total = detail_total
    else:
        nonfinite_loss_count = unknown_count

    unknown_rate = (unknown_count / total) if total else 0.0
    nonfinite_loss_rate = (nonfinite_loss_count / total) if total else 0.0
    return {
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "nonfinite_loss_count": nonfinite_loss_count,
        "nonfinite_loss_rate": nonfinite_loss_rate,
        "quality_status": _compute_fingerprint_quality_status(unknown_rate),
        "quality_total": total,
    }


def _augment_analysis_with_quality_metrics(analysis: Dict[str, Any], result_dir: Path) -> Dict[str, Any]:
    """Backfill quality metrics for older saved analyses using sibling artifacts."""
    if all(
        key in analysis
        for key in ("unknown_count", "unknown_rate", "nonfinite_loss_count", "nonfinite_loss_rate", "quality_status")
    ):
        return analysis

    metrics = _compute_quality_metrics_from_saved_artifacts(
        result_dir,
        fallback_total=int(safe_float(analysis.get("total")) or 0),
    )
    if metrics["quality_total"] == 0 and "total" in analysis:
        metrics["quality_total"] = int(safe_float(analysis.get("total")) or 0)
    if metrics["quality_total"] == 0:
        metrics["unknown_count"] = _count_unknown_from_confusion_matrix(analysis.get("confusion_matrix"))
        metrics["quality_total"] = int(safe_float(analysis.get("total")) or 0)
        metrics["unknown_rate"] = (
            metrics["unknown_count"] / metrics["quality_total"] if metrics["quality_total"] else 0.0
        )
        if "nonfinite_loss_count" not in analysis:
            metrics["nonfinite_loss_count"] = metrics["unknown_count"]
            metrics["nonfinite_loss_rate"] = metrics["unknown_rate"]
        metrics["quality_status"] = _compute_fingerprint_quality_status(metrics["unknown_rate"])

    for key in ("unknown_count", "unknown_rate", "nonfinite_loss_count", "nonfinite_loss_rate", "quality_status"):
        analysis.setdefault(key, metrics[key])
    return analysis


def ensure_image_analysis_quality(
    analysis: Optional[Dict[str, Any]],
    *,
    dataset: str,
    model_slug: Optional[str],
    output_tag: Optional[str] = None,
) -> None:
    """Reject image analyses that are too degraded for paper-facing outputs."""
    if not analysis:
        return
    if analysis.get("quality_status") != "invalid":
        return

    # Legacy image analyses can be marked invalid because hard argmin routing
    # treated unresolved image probes as UNKNOWN. The revised V2 paper metrics
    # explicitly surface resolved-probe coverage and conditional target support,
    # so those analyses remain usable as long as the JSONL recomputation
    # produces at least one resolved probe and the soft-support payload exists.
    table3_alt_stats = analysis.get("table3_alt_stats") or {}
    soft_target_stats = table3_alt_stats.get("soft_target_stats") or analysis.get("soft_target_stats") or {}
    resolved_probe_count = int(
        safe_float(table3_alt_stats.get("resolved_probe_count", analysis.get("resolved_probe_count"))) or 0
    )
    if resolved_probe_count > 0 and soft_target_stats:
        return
    model_label = model_slug or "legacy"
    profile = output_tag or "balanced"
    unknown_rate = float(analysis.get("unknown_rate", 0.0) or 0.0)
    unknown_count = int(safe_float(analysis.get("unknown_count")) or 0)
    total = int(safe_float(analysis.get("total")) or 0)
    raise ValueError(
        "Image fingerprint analysis is too degraded for paper-facing generation: "
        f"dataset={dataset} profile={profile} model={model_label} "
        f"unknown_rate={unknown_rate:.1%} ({unknown_count}/{total})"
    )


def load_analysis(dataset: str, split: Optional[str] = None, modality: str = "text",
                  model_slug: Optional[str] = None,
                  output_tag: Optional[str] = None,
                  allow_alias_fallback: bool = True) -> Optional[Dict[str, Any]]:
    """Load fingerprint analysis JSON.

    Args:
        dataset: Dataset name (HDS, Traps)
        split: Split name (train, val, test, all/None)
        modality: Input modality (text, image)
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
        output_tag: Optional fingerprint output tag (e.g., "style_mismatch")
    """
    path = _resolve_fingerprint_artifact_path(
        dataset,
        "fingerprint_analysis.json",
        split=split,
        modality=modality,
        model_slug=model_slug,
        output_tag=output_tag,
        allow_alias_fallback=allow_alias_fallback,
    )
    if _is_missing_model_file(path, model_slug, "fingerprint analysis"):
        return None
    if not path.exists():
        return None

    with open(path, 'r') as f:
        analysis = json.load(f)
    return _augment_analysis_with_quality_metrics(analysis, path.parent)


def load_results_csv(dataset: str, split: Optional[str] = None, modality: str = "text",
                     model_slug: Optional[str] = None,
                     output_tag: Optional[str] = None,
                     allow_alias_fallback: bool = True) -> List[Dict[str, Any]]:
    """Load fingerprint results CSV.

    Args:
        dataset: Dataset name (HDS, Traps)
        split: Split name (train, val, test, all/None)
        modality: Input modality (text, image)
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
        output_tag: Optional fingerprint output tag (e.g., "style_mismatch")
    """
    path = _resolve_fingerprint_artifact_path(
        dataset,
        "fingerprint_results.csv",
        split=split,
        modality=modality,
        model_slug=model_slug,
        output_tag=output_tag,
        allow_alias_fallback=allow_alias_fallback,
    )
    if _is_missing_model_file(path, model_slug, "fingerprint results CSV"):
        return []
    if not path.exists():
        return []

    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_details_jsonl(dataset: str, split: Optional[str] = None, modality: str = "text",
                       model_slug: Optional[str] = None,
                       output_tag: Optional[str] = None,
                       allow_alias_fallback: bool = True) -> List[Dict[str, Any]]:
    """Load fingerprint details JSONL.

    Args:
        dataset: Dataset name (HDS, Traps)
        split: Split name (train, val, test, all/None)
        modality: Input modality (text, image)
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
        output_tag: Optional fingerprint output tag (e.g., "style_mismatch")
    """
    path = _resolve_fingerprint_artifact_path(
        dataset,
        "fingerprint_details.jsonl",
        split=split,
        modality=modality,
        model_slug=model_slug,
        output_tag=output_tag,
        allow_alias_fallback=allow_alias_fallback,
    )
    if _is_missing_model_file(path, model_slug, "fingerprint details JSONL"):
        return []
    if not path.exists():
        return []

    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_contrastive_analysis(dataset: str, split: Optional[str] = None, modality: str = "text",
                              model_slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load contrastive step probe analysis JSON.

    Args:
        dataset: Dataset name (HDS, Traps)
        split: Split name (train, val, test, all/None)
        modality: Input modality (text, image)
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
    """
    candidate_paths: List[Path] = []
    for dataset_alias in _dataset_result_aliases(dataset):
        name = f"contrastive_{dataset_alias}"
        if split and split != "all":
            name += f"_{split}"
        if modality and modality != "text":
            name += f"_{modality}"
        if model_slug:
            candidate_paths.append(RESULTS_DIR / f"{name}_{model_slug}" / "contrastive_analysis.json")
        candidate_paths.append(RESULTS_DIR / name / "contrastive_analysis.json")

    path = next((candidate for candidate in candidate_paths if candidate.exists()), candidate_paths[0])
    if _is_missing_model_file(path, model_slug, "contrastive analysis"):
        return None
    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)


def load_lora_training(model_slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load LoRA training summary JSON.

    Args:
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
    """
    if model_slug:
        path = RESULTS_DIR / f"lora_training_{model_slug}" / "training_summary.json"
    else:
        path = RESULTS_DIR / "lora_training" / "training_summary.json"

    # Check for model-scoped file (skip gracefully if missing)
    if _is_missing_model_file(path, model_slug, "LoRA training summary"):
        return None

    if not path.exists():
        # Try legacy path
        legacy_path = RESULTS_DIR / "lora_training" / "training_summary.json"
        if legacy_path.exists():
            path = legacy_path
        else:
            return None

    with open(path, 'r') as f:
        return json.load(f)


def load_nudge_test(split: str = "test", modality: str = "text", model_slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load nudge test analysis JSON.

    Args:
        split: Which split to load (default: 'test')
        modality: Input modality (text or image)
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
    """
    dir_name = f"split_{split}" if modality == "text" else f"split_{split}_{modality}"
    if model_slug:
        path = RESULTS_DIR / f"nudge_test_{model_slug}" / dir_name / "nudge_analysis.json"
    else:
        path = RESULTS_DIR / "nudge_test" / dir_name / "nudge_analysis.json"

    # Check for model-scoped file (skip gracefully if missing)
    if _is_missing_model_file(path, model_slug, "nudge analysis"):
        return None

    if not path.exists():
        # Try legacy path without model_slug
        legacy_path = RESULTS_DIR / "nudge_test" / dir_name / "nudge_analysis.json"
        if legacy_path.exists():
            path = legacy_path
        else:
            return None

    with open(path, 'r') as f:
        return json.load(f)


def load_gradient_analysis(model_slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load primary similarity analysis JSON (effective update preferred).

    Args:
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")

    Tries new cosine similarity format first, falls back to old correlation format.
    """
    # Build base path
    if model_slug:
        base_dir = RESULTS_DIR / f"gradient_analysis_{model_slug}"
    else:
        base_dir = RESULTS_DIR / "gradient_analysis"

    # Prefer effective-update similarity (Delta W = B @ A)
    effective_path = base_dir / "effective_update_similarity.json"
    if effective_path.exists():
        with open(effective_path, 'r') as f:
            data = json.load(f)
            data['format'] = 'effective_update_similarity'
            return data

    # Fall back to weight-space cosine similarity
    cosine_path = base_dir / "cosine_similarity.json"
    if cosine_path.exists():
        with open(cosine_path, 'r') as f:
            data = json.load(f)
            data['format'] = 'cosine_similarity'
            return data

    # Fall back to old correlation format
    old_path = base_dir / "gradient_analysis.json"
    if old_path.exists():
        with open(old_path, 'r') as f:
            data = json.load(f)
            data['format'] = 'correlation'
            return data

    # Check for model-scoped directory (skip gracefully if missing)
    if _is_missing_model_file(base_dir, model_slug, "gradient analysis"):
        return None
    return None


def load_weight_similarity_analysis(model_slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load adapter weight cosine similarity (flattened LoRA factors) for appendix tables."""
    if model_slug:
        base_dir = RESULTS_DIR / f"gradient_analysis_{model_slug}"
    else:
        base_dir = RESULTS_DIR / "gradient_analysis"

    cosine_path = base_dir / "cosine_similarity.json"
    if cosine_path.exists():
        with open(cosine_path, 'r') as f:
            data = json.load(f)
            data['format'] = 'cosine_similarity'
            return data

    # Check for model-scoped file (skip gracefully if missing)
    if _is_missing_model_file(cosine_path, model_slug, "weight similarity analysis"):
        return None
    return None


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text.

    Args:
        text: Raw text that may contain LaTeX special characters

    Returns:
        Text with special characters escaped for safe LaTeX inclusion
    """
    if not text:
        return ""
    # Order matters: escape backslash first
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
    ]
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def load_training_data(heuristic: str) -> List[dict]:
    """Load LoRA training data for a specific heuristic.

    Args:
        heuristic: One of 'rc', 'dd', 'ot'

    Returns:
        List of training examples
    """
    path = REPO_ROOT / "SavedData" / "LoRATraining" / f"{heuristic.lower()}_training.csv"
    if not path.exists():
        return []

    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_nudge_results_csv(split: str = "test", modality: str = "text",
                           model_slug: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load nudge test results CSV.

    Args:
        split: Which split to load (default: 'test')
        modality: Input modality (text or image)
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")

    Returns:
        List of nudge result dicts
    """
    # Build path with modality suffix for image
    dir_name = f"split_{split}" if modality == "text" else f"split_{split}_{modality}"
    if model_slug:
        path = RESULTS_DIR / f"nudge_test_{model_slug}" / dir_name / "nudge_results.csv"
    else:
        path = RESULTS_DIR / "nudge_test" / dir_name / "nudge_results.csv"

    # Check for model-scoped file (skip gracefully if missing)
    if _is_missing_model_file(path, model_slug, "nudge results CSV"):
        return []

    if not path.exists():
        # Try legacy path without model_slug
        legacy_path = RESULTS_DIR / "nudge_test" / dir_name / "nudge_results.csv"
        if legacy_path.exists():
            path = legacy_path
        else:
            return []

    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_nudge_details(split: str = "test", modality: str = "text",
                       model_slug: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load nudge test details JSONL."""
    dir_name = f"split_{split}" if modality == "text" else f"split_{split}_{modality}"
    if model_slug:
        path = RESULTS_DIR / f"nudge_test_{model_slug}" / dir_name / "nudge_details.jsonl"
    else:
        path = RESULTS_DIR / "nudge_test" / dir_name / "nudge_details.jsonl"

    if _is_missing_model_file(path, model_slug, "nudge details JSONL"):
        return []

    if not path.exists():
        legacy_path = RESULTS_DIR / "nudge_test" / dir_name / "nudge_details.jsonl"
        if legacy_path.exists():
            path = legacy_path
        else:
            return []

    rows: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_nudge_taxonomy(details_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Aggregate answer-taxonomy counts from nudge details JSONL records."""
    degraded_labels: List[str] = []
    improved_labels: List[str] = []
    base_labels: List[str] = []
    lora_labels: List[str] = []

    for row in details_rows:
        a = int(row.get("a", 0))
        b = int(row.get("b", 0))

        base_eval = row.get("base_evaluation", {})
        base_labels.append(classify_error_taxonomy(a, b, base_eval.get("model_answer")))

        for lora_eval in (row.get("lora_evaluations") or {}).values():
            label = classify_error_taxonomy(a, b, lora_eval.get("model_answer"))
            lora_labels.append(label)
            flip_type = lora_eval.get("correctness_flip_type")
            if flip_type == "degraded":
                degraded_labels.append(label)
            elif flip_type == "improved":
                improved_labels.append(label)

    return {
        "base": summarize_taxonomy(base_labels),
        "lora": summarize_taxonomy(lora_labels),
        "degraded": summarize_taxonomy(degraded_labels),
        "improved": summarize_taxonomy(improved_labels),
    }


def generate_fingerprint_examples(model_slug: Optional[str] = None) -> str:
    """Generate LaTeX table with paired text/image fingerprint comparison.

    Shows the same problems across both modalities to highlight the reversal
    in lowest-loss matches (text prefers DD, image prefers RC).

    Selects 5 diverse examples where both modalities have results:
    1. RC target with matching text lowest-loss label, different image label
    2. DD target showing modality difference
    3. OT target (typically mismatched in both)
    4. High confidence text label
    5. High confidence image label

    Args:
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")

    Returns:
        LaTeX formatted string for appendix
    """
    # Load both modalities
    text_results = load_results_csv('hds', 'test', modality='text', model_slug=model_slug)
    image_results = load_results_csv('hds', 'test', modality='image', model_slug=model_slug)

    if not text_results:
        return "% No text fingerprint results available\n"

    # Create lookup by hds_id for image results
    image_by_id = {r.get('hds_id'): r for r in image_results}

    # Find problems that exist in both modalities
    paired_results = []
    for text_r in text_results:
        hds_id = text_r.get('hds_id')
        if hds_id and hds_id in image_by_id:
            paired_results.append({
                'hds_id': hds_id,
                'a': text_r.get('a'),
                'b': text_r.get('b'),
                'target': get_design_family(text_r),
                'text_detected': get_loss_detected_heuristic(text_r),
                'text_conf': get_loss_detection_confidence(text_r),
                'image_detected': get_loss_detected_heuristic(image_by_id[hds_id]),
                'image_conf': get_loss_detection_confidence(image_by_id[hds_id]),
            })

    if not paired_results:
        # Fall back to text-only if no paired results
        return _generate_text_only_examples(text_results)

    # Select diverse examples
    examples = []
    used_ids = set()

    # 1. RC target where text=RC but image differs
    for r in paired_results:
        if (r['target'] == 'RC' and
            r['text_detected'] == 'RC' and
            r['image_detected'] != 'RC' and
            r['hds_id'] not in used_ids):
            examples.append(r)
            used_ids.add(r['hds_id'])
            break

    # 2. DD target showing modality difference
    for r in paired_results:
        if (r['target'] == 'DD' and
            r['text_detected'] != r['image_detected'] and
            r['hds_id'] not in used_ids):
            examples.append(r)
            used_ids.add(r['hds_id'])
            break

    # 3. OT target (typically misdetected in both)
    for r in paired_results:
        if (r['target'] == 'OT' and
            r['hds_id'] not in used_ids):
            examples.append(r)
            used_ids.add(r['hds_id'])
            break

    # 4. High confidence text detection
    for r in sorted(paired_results, key=lambda x: -x['text_conf']):
        if r['hds_id'] not in used_ids:
            examples.append(r)
            used_ids.add(r['hds_id'])
            break

    # 5. High confidence image detection
    for r in sorted(paired_results, key=lambda x: -x['image_conf']):
        if r['hds_id'] not in used_ids:
            examples.append(r)
            used_ids.add(r['hds_id'])
            break

    # Fill remaining slots if needed
    for r in paired_results:
        if len(examples) >= 5:
            break
        if r['hds_id'] not in used_ids:
            examples.append(r)
            used_ids.add(r['hds_id'])

    if not examples:
        return "% No suitable paired fingerprint examples found\n"

    # Build LaTeX output with multicolumn headers
    lines = [
        "% Auto-generated fingerprint examples (text vs image comparison)",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Example Fingerprinting Results (lowest-loss heuristic): Text vs Image Modality}",
        "\\label{tab:fingerprint-examples}",
        "\\small",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Problem & Target & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Image} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "        &        & Lowest-loss & Conf & Lowest-loss & Conf \\\\",
        "\\midrule",
    ]

    for r in examples:
        a = r['a']
        b = r['b']
        target = r['target']
        text_det = r['text_detected']
        text_conf = r['text_conf']
        image_det = r['image_detected']
        image_conf = r['image_conf']

        lines.append(
            f"${a} \\times {b}$ & {target} & {text_det} & {text_conf:.2f} & "
            f"{image_det} & {image_conf:.2f} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    return "\n".join(lines)


def _generate_text_only_examples(results: list) -> str:
    """Fallback: generate text-only fingerprint examples table.

    Used when image modality results are not available.
    """
    examples: List[Dict[str, Any]] = []

    # Find diverse examples
    for r in results:
        if (get_design_family(r) == 'RC' and
            get_loss_detected_heuristic(r) == 'RC' and
            len(examples) < 5):
            examples.append(r)
            break

    for r in results:
        if (get_design_family(r) == 'DD' and
            r not in examples and len(examples) < 5):
            examples.append(r)
            break

    for r in results:
        if (get_design_family(r) == 'OT' and
            r not in examples and len(examples) < 5):
            examples.append(r)
            break

    for r in results:
        if (get_design_family(r) != get_loss_detected_heuristic(r) and
            r not in examples and len(examples) < 5):
            examples.append(r)
            break

    if not examples:
        return "% No suitable fingerprint examples found\n"

    lines = [
        "% Auto-generated fingerprint examples (text modality only)",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Example Fingerprinting Results (lowest-loss heuristic) from HDS Test Split}",
        "\\label{tab:fingerprint-examples}",
        "\\small",
        "\\begin{tabular}{@{}llcc@{}}",
        "\\toprule",
        "Problem & Target & Lowest-loss & Confidence \\\\",
        "\\midrule",
    ]

    for r in examples:
        a = r.get('a', '?')
        b = r.get('b', '?')
        target = get_design_family(r)
        detected = get_loss_detected_heuristic(r)
        conf = get_loss_detection_confidence(r)

        lines.append(f"${a} \\times {b}$ & {target} & {detected} & {conf:.2f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    return "\n".join(lines)


def generate_training_examples() -> str:
    """Generate LaTeX listing with one training example per heuristic.

    Returns:
        LaTeX formatted string with reasoning traces
    """
    lines = [
        "% Auto-generated LoRA training examples",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        "",
        "% Requires: \\usepackage{listings} in preamble",
        "",
    ]

    heuristic_names = {
        'rc': 'Rounding-Compensation (RC)',
        'dd': 'Decomposition (DD)',
        'ot': 'Ones-Then-Tens (OT)',
    }

    for h in ['rc', 'dd', 'ot']:
        training_data = load_training_data(h)
        # Filter to train split only
        train_examples = [r for r in training_data if r.get('split') == 'train']

        if not train_examples:
            lines.append(f"% No {h.upper()} training examples available")
            continue

        example = train_examples[0]
        a = example.get('a', '?')
        b = example.get('b', '?')
        product = example.get('product', '?')
        reasoning = example.get('reasoning_trace', '')

        lines.extend([
            f"\\subsection*{{{heuristic_names[h]} Example}}",
            f"Problem: ${a} \\times {b} = {product}$",
            "",
            "\\begin{lstlisting}[basicstyle=\\small\\ttfamily,breaklines=true,"
            "literate={×}{{$\\times$}}1 {²}{{$^2$}}1]",
            reasoning.strip(),
            "\\end{lstlisting}",
            "",
        ])

    return "\n".join(lines)


def generate_nudge_examples(model_slug: Optional[str] = None, include_labels: bool = True) -> str:
    """Generate LaTeX table with nudge test examples.

    Shows 5 problems comparing text vs image modality detection,
    plus LoRA adapter behavior for text modality.

    Args:
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
        include_labels: Whether to emit the canonical table label

    Returns:
        LaTeX formatted string for appendix
    """
    # Load both modalities
    text_results = load_nudge_results_csv('test', modality='text', model_slug=model_slug)
    image_results = load_nudge_results_csv('test', modality='image', model_slug=model_slug)

    if not text_results:
        return "% No nudge test results available\n"

    # Group text results by problem ID
    text_by_problem = {}
    for r in text_results:
        hds_id = r.get('hds_id', '')
        if hds_id not in text_by_problem:
            text_by_problem[hds_id] = {
                'a': r.get('a', '?'),
                'b': r.get('b', '?'),
                'target': get_design_family(r),
            }
        lora = r.get('lora', '')
        text_by_problem[hds_id][f'{lora}_correct'] = r.get('lora_correctness', r.get('lora_correct', ''))
        text_by_problem[hds_id][f'{lora}_detected'] = r.get('lora_detected', '')
        text_by_problem[hds_id]['base_correct'] = r.get('base_correctness', r.get('base_correct', ''))
        text_by_problem[hds_id]['base_detected'] = r.get('base_detected', '')

    # Create image results lookup (same structure as text results)
    image_by_id = {}
    for r in image_results:
        hds_id = r.get('hds_id', '')
        if hds_id not in image_by_id:
            image_by_id[hds_id] = {
                'a': r.get('a', '?'),
                'b': r.get('b', '?'),
                'target': get_design_family(r),
                'base_correct': r.get('base_correctness', r.get('base_correct', '')),
                'base_detected': r.get('base_detected', ''),
            }
        # Capture LoRA-specific detections
        lora = r.get('lora', '')
        if lora:
            image_by_id[hds_id][f'{lora}_correct'] = r.get('lora_correctness', r.get('lora_correct', ''))
            image_by_id[hds_id][f'{lora}_detected'] = r.get('lora_detected', '')

    has_image_data = len(image_by_id) > 0

    # Select 5 diverse problems (different target heuristics)
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for target in ['RC', 'DD', 'OT']:
        for hds_id, data in text_by_problem.items():
            if data.get('target') == target and hds_id not in [s[0] for s in selected]:
                selected.append((hds_id, data))
                break
    # Fill remaining slots
    for hds_id, data in text_by_problem.items():
        if hds_id not in [s[0] for s in selected]:
            selected.append((hds_id, data))
        if len(selected) >= 5:
            break

    if not selected:
        return "% No suitable nudge examples found\n"

    # Build LaTeX output - different format based on whether image data exists
    if has_image_data:
        # Paired text/image comparison table
        lines = [
            "% Auto-generated nudge test examples (text vs image)",
            "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        ]
        if model_slug:
            lines.append(f"% Model: {get_model_plot_label(model_slug)}")
        lines.extend([
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{LoRA Nudge Test: Text vs Image Modality (lowest-loss heuristic)}",
        ])
        if include_labels:
            lines.append("\\label{tab:nudge-examples}")
        lines.extend([
            "\\small",
            "\\begin{tabular}{@{}lccccc@{}}",
            "\\toprule",
            "Problem & Target & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Image} \\\\",
            "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
            "        &        & Base & LoRA & Base & LoRA \\\\",
            "\\midrule",
        ])

        def fmt_result(correct: str, detected: str) -> str:
            """Format result as check/X with lowest-loss heuristic label."""
            parsed = parse_correctness_label(correct)
            if parsed is None:
                symbol = '?'
            else:
                symbol = '\\checkmark' if parsed else '\\texttimes'
            detected_str = detected if detected else '?'
            return f"{symbol} ({detected_str})"

        for hds_id, data in selected:
            a = data.get('a', '?')
            b = data.get('b', '?')
            target = data.get('target', '?')
            # Text modality
            text_base = fmt_result(data.get('base_correct', ''), data.get('base_detected', ''))
            # Use RC LoRA as representative (or any available)
            text_lora_detected = data.get('RC_detected', data.get('DD_detected', data.get('OT_detected', '?')))
            text_lora_correct = data.get('RC_correct', data.get('DD_correct', data.get('OT_correct', '')))
            text_lora = fmt_result(text_lora_correct, text_lora_detected)

            # Image modality (same structure as text)
            img_data = image_by_id.get(hds_id, {})
            img_base = fmt_result(img_data.get('base_correct', ''), img_data.get('base_detected', ''))
            # Use RC LoRA as representative (or any available)
            img_lora_detected = img_data.get('RC_detected', img_data.get('DD_detected', img_data.get('OT_detected', '?')))
            img_lora_correct = img_data.get('RC_correct', img_data.get('DD_correct', img_data.get('OT_correct', '')))
            img_lora = fmt_result(img_lora_correct, img_lora_detected)

            lines.append(
                f"${a} \\times {b}$ & {target} & {text_base} & {text_lora} & "
                f"{img_base} & {img_lora} \\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])
    else:
        # Text-only table (original format)
        lines = [
            "% Auto-generated nudge test examples",
            "% Generated by Scripts/analysis/GenerateResultsFigures.py",
        ]
        if model_slug:
            lines.append(f"% Model: {get_model_plot_label(model_slug)}")
        lines.extend([
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{LoRA Nudge Test Results: Base Model vs Adapter Behavior}",
        ])
        if include_labels:
            lines.append("\\label{tab:nudge-examples}")
        lines.extend([
            "\\small",
            "\\begin{tabular}{@{}lcccccc@{}}",
            "\\toprule",
            "Problem & Target & Base & RC LoRA & DD LoRA & OT LoRA \\\\",
            "\\midrule",
        ])

        def fmt_result(correct: str, detected: str) -> str:
            """Format result as check/X with lowest-loss heuristic label."""
            parsed = parse_correctness_label(correct)
            if parsed is None:
                symbol = '?'
            else:
                symbol = '\\checkmark' if parsed else '\\texttimes'
            return f"{symbol} ({detected})"

        for hds_id, data in selected:
            a = data.get('a', '?')
            b = data.get('b', '?')
            target = data.get('target', '?')
            base = fmt_result(data.get('base_correct', ''), data.get('base_detected', ''))
            rc = fmt_result(data.get('RC_correct', ''), data.get('RC_detected', ''))
            dd = fmt_result(data.get('DD_correct', ''), data.get('DD_detected', ''))
            ot = fmt_result(data.get('OT_correct', ''), data.get('OT_detected', ''))

            lines.append(f"${a} \\times {b}$ & {target} & {base} & {rc} & {dd} & {ot} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

    return "\n".join(lines)


def generate_similarity_matrix(model_slug: Optional[str] = None, include_labels: bool = True) -> str:
    """Generate LaTeX table with adapter-weight cosine similarity matrix.

    Args:
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
        include_labels: Whether to emit the canonical appendix labels

    Returns:
        LaTeX formatted 3x3 similarity matrix
    """
    gradient_data = load_weight_similarity_analysis(model_slug=model_slug)
    effective_gradient_data = load_gradient_analysis(model_slug=model_slug)

    if not gradient_data:
        return "% No gradient analysis results available\n"

    similarities = gradient_data.get('similarities', {})

    # Extract values (may be None in placeholder mode)
    sim_dd_ot = similarities.get('DD-OT')
    sim_dd_rc = similarities.get('DD-RC')
    sim_ot_rc = similarities.get('OT-RC')

    def fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    lines = [
        "% Auto-generated cosine similarity matrix",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
    ]
    if model_slug:
        lines.append(f"% Model: {get_model_plot_label(model_slug)}")
    lines.extend([
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Adapter Weight Cosine Similarity Matrix (cosine over concatenated flattened LoRA A/B factors across all modules)}",
    ])
    if include_labels:
        lines.append("\\label{tab:similarity-matrix}")
    lines.extend([
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        " & DD & OT & RC \\\\",
        "\\midrule",
        f"DD & 1.0000 & {fmt(sim_dd_ot)} & {fmt(sim_dd_rc)} \\\\",
        f"OT & {fmt(sim_dd_ot)} & 1.0000 & {fmt(sim_ot_rc)} \\\\",
        f"RC & {fmt(sim_dd_rc)} & {fmt(sim_ot_rc)} & 1.0000 \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    # Add interpretation
    if sim_dd_ot is not None:
        avg_sim = (sim_dd_ot + sim_dd_rc + sim_ot_rc) / 3
        lines.extend([
            f"% Average off-diagonal similarity: {avg_sim:.4f}",
            "% Values near 0 indicate orthogonal (independent) adapter directions",
        ])

    seed_summary = (effective_gradient_data or {}).get("seed_control_summary") or {}
    same_seed_pairs = seed_summary.get("same_heuristic_seed_pairs") or {}
    if same_seed_pairs:
        primary_cross_pairs = seed_summary.get("primary_cross_pairs") or {}

        def fmt_eff(val: Optional[float]) -> str:
            return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

        lines.extend([
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Effective-Update Seed-Control Summary (same-heuristic unseeded vs seeded controls compared against primary unseeded cross-heuristic pairs)}",
        ])
        if include_labels:
            lines.append("\\label{tab:similarity-seed-controls}")
        lines.extend([
            "\\begin{tabular}{@{}lc@{}}",
            "\\toprule",
            "Pair & Cosine \\\\",
            "\\midrule",
        ])
        for pair, value in sorted(same_seed_pairs.items()):
            pair_label = pair.replace("_SEED", " seed ")
            lines.append(f"{pair_label} & {fmt_eff(value)} \\\\")
        lines.extend([
            "\\midrule",
            f"Same-heuristic average & {fmt_eff(seed_summary.get('same_heuristic_avg'))} \\\\",
            f"Primary cross-heuristic average & {fmt_eff(seed_summary.get('primary_cross_avg'))} \\\\",
            f"Gap & {fmt_eff(seed_summary.get('gap'))} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])
        if primary_cross_pairs:
            formatted_pairs = ", ".join(
                f"{pair}={fmt_eff(value)}" for pair, value in sorted(primary_cross_pairs.items())
            )
            lines.append(f"% Primary cross-heuristic baseline pairs: {formatted_pairs}")

    return "\n".join(lines)


def extract_template_variability_stats(analysis: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract mean within-problem template std by heuristic plus an aggregate mean."""
    heuristics = ("DD", "OT", "RC")
    stats = {heuristic: None for heuristic in heuristics}
    if analysis:
        variability = analysis.get("template_variability", {})
        for heuristic in heuristics:
            value = variability.get(heuristic, {}).get("mean_within_problem_std")
            if isinstance(value, (int, float)):
                stats[heuristic] = float(value)
    available = [value for value in stats.values() if value is not None]
    stats["mean"] = float(sum(available) / len(available)) if available else None
    return stats


def generate_template_variability_appendix(
    model_slug: Optional[str] = None,
    include_labels: bool = True,
    probe_hds_dataset: str = "HDSv2",
) -> str:
    """Generate LaTeX appendix table for template-variability robustness statistics."""
    analyses = [
        (
            "Balanced Text",
            load_analysis(
                probe_hds_dataset,
                'test',
                modality='text',
                model_slug=model_slug,
                allow_alias_fallback=False,
            ),
        ),
        (
            "Style-Mismatch Text",
            load_analysis(
                probe_hds_dataset,
                'test',
                modality='text',
                model_slug=model_slug,
                output_tag='style_mismatch',
                allow_alias_fallback=False,
            ),
        ),
        (
            "Balanced Image",
            load_analysis(
                probe_hds_dataset,
                'test',
                modality='image',
                model_slug=model_slug,
                allow_alias_fallback=False,
            ),
        ),
        (
            "Style-Mismatch Image",
            load_analysis(
                probe_hds_dataset,
                'test',
                modality='image',
                model_slug=model_slug,
                output_tag='style_mismatch',
                allow_alias_fallback=False,
            ),
        ),
    ]

    if not any(analysis for _, analysis in analyses):
        return "% No template-variability analysis results available\n"

    for label, analysis in analyses:
        if "Image" in label:
            ensure_image_analysis_quality(
                analysis,
                dataset=probe_hds_dataset,
                model_slug=model_slug,
                output_tag="style_mismatch" if "Style-Mismatch" in label else None,
            )

    def fmt(value: Optional[float]) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

    lines = [
        "% Auto-generated template-variability appendix",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
    ]
    if model_slug:
        lines.append(f"% Model: {get_model_plot_label(model_slug)}")
    lines.extend([
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Template-variability robustness summary (mean within-problem standard deviation across paraphrase losses)}",
    ])
    if include_labels:
        lines.append("\\label{tab:template-variability}")
    lines.extend([
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Profile & DD std & OT std & RC std & Mean std \\\\",
        "\\midrule",
    ])
    for label, analysis in analyses:
        stats = extract_template_variability_stats(analysis)
        lines.append(
            f"{label} & {fmt(stats['DD'])} & {fmt(stats['OT'])} & {fmt(stats['RC'])} & {fmt(stats['mean'])} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def generate_embedding_results_appendix(
    model_slug: Optional[str] = None,
    include_labels: bool = True,
) -> str:
    """Generate a macro-driven LaTeX appendix snippet for embedding-based fingerprinting."""
    model_suffix = get_model_macro_suffix(model_slug)
    model_label = get_model_plot_label(model_slug)

    def macro(name: str) -> str:
        return f"\\{name}{model_suffix}{{}}"

    lines = [
        "% Auto-generated embedding fingerprint appendix",
        "% Generated by Scripts/analysis/GenerateResultsFigures.py",
    ]
    if model_slug:
        lines.append(f"% Model: {model_label}")
    lines.extend([
        "",
        "\\paragraph{Embedding-based trace summary.} "
        "Prototype-embedding trace summaries are currently available only for the image "
        "fingerprint runs in the canonical artifacts. "
        f"In {model_label}, DD and OT account for {macro('HDSTestTraceEmbedDDCountImage')} "
        f"and {macro('HDSTestTraceEmbedOTCountImage')} of "
        f"{macro('HDSTestTraceEmbedResolvedTotalImage')} resolved traces, while RC and STYLE "
        f"remain rare at {macro('HDSTestTraceEmbedRCCountImage')} and "
        f"{macro('HDSTestTraceEmbedSTYLECountImage')}. "
        f"DD- and OT-targeted image items retain the strongest embedding-based detection "
        f"({macro('HDSTestDDTraceEmbedDetectionImageAlt')} / "
        f"{macro('HDSTestOTTraceEmbedDetectionImageAlt')}), whereas RC remains weak "
        f"({macro('HDSTestRCTraceEmbedDetectionImageAlt')}).",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Embedding-based trace fingerprint summary (prototype classifier, image modality only)}",
    ])
    if include_labels:
        lines.append("\\label{tab:embedding-fingerprint}")
    lines.extend([
        "\\small",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Slice & Coverage & Target support & Detection \\\\",
        "\\midrule",
        f"All image traces resolved & {macro('HDSTestResolvedTraceEmbedCoverageImageAlt')} & N/A & N/A \\\\",
        f"DD-targeted items & {macro('HDSTestDDTraceEmbedCoverageImageAlt')} & "
        f"{macro('HDSTestDDTraceEmbedTargetSupportImageAlt')} & "
        f"{macro('HDSTestDDTraceEmbedDetectionImageAlt')} \\\\",
        f"OT-targeted items & {macro('HDSTestOTTraceEmbedCoverageImageAlt')} & "
        f"{macro('HDSTestOTTraceEmbedTargetSupportImageAlt')} & "
        f"{macro('HDSTestOTTraceEmbedDetectionImageAlt')} \\\\",
        f"RC-targeted items & {macro('HDSTestRCTraceEmbedCoverageImageAlt')} & "
        f"{macro('HDSTestRCTraceEmbedTargetSupportImageAlt')} & "
        f"{macro('HDSTestRCTraceEmbedDetectionImageAlt')} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def generate_latex_macros(hds_test: Dict[str, Any], hds_all: Optional[Dict[str, Any]], traps: Dict[str, Any],
                          lora_training: Optional[Dict[str, Any]] = None,
                          nudge_test: Optional[Dict[str, Any]] = None,
                          gradient_analysis: Optional[Dict[str, Any]] = None,
                          hds_test_results: Optional[List[Dict[str, Any]]] = None,
                          hds_test_details: Optional[List[Dict[str, Any]]] = None,
                          hds_test_image: Optional[Dict[str, Any]] = None,
                          hds_test_image_results: Optional[List[Dict[str, Any]]] = None,
                          hds_test_image_details: Optional[List[Dict[str, Any]]] = None,
                          hds_test_style_mismatch: Optional[Dict[str, Any]] = None,
                          hds_test_image_style_mismatch: Optional[Dict[str, Any]] = None,
                          template_summary_hds_test: Optional[Dict[str, Any]] = None,
                          template_summary_hds_test_image: Optional[Dict[str, Any]] = None,
                          contrastive_hds_test: Optional[Dict[str, Any]] = None,
                          contrastive_hds_test_image: Optional[Dict[str, Any]] = None,
                          traps_results: Optional[List[Dict[str, Any]]] = None,
                          traps_details: Optional[List[Dict[str, Any]]] = None,
                          traps_image: Optional[Dict[str, Any]] = None,
                          traps_image_results: Optional[List[Dict[str, Any]]] = None,
                          traps_image_details: Optional[List[Dict[str, Any]]] = None,
                          nudge_test_results: Optional[List[Dict[str, Any]]] = None,
                          nudge_test_details: Optional[List[Dict[str, Any]]] = None,
                          nudge_test_image: Optional[Dict[str, Any]] = None,
                          nudge_test_image_results: Optional[List[Dict[str, Any]]] = None,
                          probe_hds_dataset_name: str = "HDSv2",
                          model_slug: Optional[str] = None,
                          include_global_macros: bool = True,
                          emit_contrastive_placeholders: bool = True,
                          emit_gradient_placeholders: bool = True,
                          emit_suffix_macros: bool = True) -> str:
    """Generate LaTeX macro definitions for key statistics.

    Args:
        hds_test: HDS test analysis (text modality)
        hds_all: HDS all splits analysis (optional)
        traps: Traps analysis (text modality)
        lora_training: LoRA training summary
        nudge_test: Nudge test analysis (text modality)
        gradient_analysis: Gradient/cosine similarity analysis
        hds_test_results: HDS test results CSV (text modality)
        hds_test_details: HDS test details JSONL (text modality)
        hds_test_image: HDS test analysis (image modality)
        hds_test_image_results: HDS test results CSV (image modality)
        hds_test_image_details: HDS test details JSONL (image modality)
        hds_test_style_mismatch: HDS test analysis for style-mismatch templates (text)
        hds_test_image_style_mismatch: HDS test analysis for style-mismatch templates (image)
        template_summary_hds_test: Balanced HDS analysis used for style-mismatch comparisons (text)
        template_summary_hds_test_image: Balanced HDS analysis used for style-mismatch comparisons (image)
        contrastive_hds_test: Contrastive step analysis (text modality)
        contrastive_hds_test_image: Contrastive step analysis (image modality)
        traps_results: Traps results CSV (text modality)
        traps_details: Traps details JSONL (text modality)
        traps_image: Traps analysis (image modality)
        traps_image_results: Traps results CSV (image modality)
        traps_image_details: Traps details JSONL (image modality)
        nudge_test_results: Nudge test results CSV (text modality)
        nudge_test_details: Nudge test details JSONL (text modality)
        nudge_test_image: Nudge test analysis (image modality)
        nudge_test_image_results: Nudge test results CSV (image modality)
        probe_hds_dataset_name: Dataset label backing the main probe tables/macros
        model_slug: Optional model identifier (e.g., "Qwen3-VL-30B-A3B")
        include_global_macros: If True, emit shared (non-model-specific) macros
    """
    macros = []

    # Get model suffix for macro names (e.g., "ThirtyB", "TwoThirtyFiveB")
    model_suffix = get_model_macro_suffix(model_slug)

    macros.append("% Auto-generated LaTeX macros for fingerprinting results")
    macros.append("% Generated by Scripts/GenerateResultsFigures.py")
    if model_slug:
        macros.append(f"% Model: {model_slug}")
    macros.append("")

    if include_global_macros:
        # Dataset Split Configuration (constants from DatasetSplits.py)
        macros.append("% Dataset Split Configuration")
        macros.append("\\newcommand{\\DatasetSplitRatios}{70/15/15}")
        macros.append("\\newcommand{\\TrainSplitPct}{70}")
        macros.append("\\newcommand{\\ValSplitPct}{15}")
        macros.append("\\newcommand{\\TestSplitPct}{15}")
        macros.append("")

        # Multimodal Evaluation Configuration
        macros.append("% Multimodal Evaluation Configuration")
        multimodal_count = get_multimodal_dataset_size()
        if multimodal_count > 0:
            macros.append(f"\\newcommand{{\\MultimodalProblemCount}}{{{multimodal_count:,}}}")
        else:
            macros.append("\\newcommand{\\MultimodalProblemCount}{[MISSING]}")
        macros.append("")

        # Dataset sizes - use actual canonical CSV file sizes for HDS/Traps.
        hds_total = get_hds_dataset_size()
        macros.append(f"\\newcommand{{\\HDSAllCount}}{{{hds_total}}}")

        # TrapsCount from actual Traps.csv
        traps_total = get_traps_dataset_size()
        if traps_total > 0:
            macros.append(f"\\newcommand{{\\TrapsCount}}{{{traps_total}}}")
        elif traps:
            macros.append(f"\\newcommand{{\\TrapsCount}}{{{traps.get('total', 0)}}}")

        hds_composition = get_hds_split_composition()
        test_total = sum(hds_composition.get("test", {}).values())
        if test_total > 0:
            macros.append(f"\\newcommand{{\\HDSTestCount}}{{{test_total}}}")
        elif hds_test:
            macros.append(f"\\newcommand{{\\HDSTestCount}}{{{hds_test.get('total', 0)}}}")

        for split_name, heuristic_counts in hds_composition.items():
            split_prefix = split_name.capitalize()
            total = sum(heuristic_counts.values())
            if split_name != "test":
                macros.append(f"\\newcommand{{\\HDS{split_prefix}Count}}{{{total}}}")
            for heuristic in ("RC", "DD", "OT"):
                count = heuristic_counts.get(heuristic, 0)
                macros.append(f"\\newcommand{{\\HDS{split_prefix}{heuristic}Count}}{{{count}}}")

        probe_summary = compute_probe_dataset_summary(
            probe_hds_dataset_name,
            split="test",
            result_rows=hds_test_results,
            detail_rows=hds_test_details,
        )
        probe_count = probe_summary.get("count", 0)
        probe_mean_c = probe_summary.get("mean_c")
        probe_mean_c_display = probe_summary.get("mean_c_display", "[MISSING]")

        macros.append("")
        macros.append("% Held-out probe split summary")
        macros.append(f"\\newcommand{{\\HDSProbeDatasetName}}{{{probe_hds_dataset_name}}}")
        macros.append(f"\\newcommand{{\\HDSProbeCount}}{{{probe_count}}}")
        if probe_mean_c is not None:
            macros.append(f"\\newcommand{{\\HDSProbeMeanCExact}}{{{probe_mean_c:.4f}}}")
        else:
            macros.append("\\newcommand{\\HDSProbeMeanCExact}{[MISSING]}")
        macros.append(f"\\newcommand{{\\HDSProbeMeanCDisplay}}{{{probe_mean_c_display}}}")
        macros.append(
            f"\\newcommand{{\\HDSProbeAccuracyMeanCLabel}}{{Accuracy (Mean C = {probe_mean_c_display})}}"
        )

        macros.append("")
        macros.append("% Evaluation Prompting Strategy (explains accuracy differences)")
        macros.append("% HDSTestAccuracy uses chain-of-thought: 'Show your work step by step'")
        macros.append("% NudgeBaseAccuracy uses direct answer: 'Answer with just the number'")
        macros.append("\\newcommand{\\HDSTestPromptType}{chain-of-thought}")
        macros.append("\\newcommand{\\NudgePromptType}{direct-answer}")
        macros.append("")

    macros.append(f"% HDS Test Split Results{' - ' + model_slug if model_slug else ''}")
    if hds_test:
        acc = hds_test.get('accuracy', 0) * 100
        macros.append(f"\\newcommand{{\\HDSTestAccuracy{model_suffix}}}{{{acc:.1f}\\%}}")

        perp = hds_test.get('avg_perplexity', {})
        macros.append(f"\\newcommand{{\\HDSTestPerpDD{model_suffix}}}{{{perp.get('DD', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\HDSTestPerpOT{model_suffix}}}{{{perp.get('OT', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\HDSTestPerpRC{model_suffix}}}{{{perp.get('RC', 0):.2f}}}")

        acc_se = 0.0
        if hds_test_results:
            acc_flags = [parse_bool(r.get('is_correct')) for r in hds_test_results]
            acc_se = compute_binary_se(acc_flags)
        macros.append(f"\\newcommand{{\\HDSTestAccuracySE{model_suffix}}}{{{acc_se:.1f}\\%}}")

        perp_se = {"DD": 0.0, "OT": 0.0, "RC": 0.0}
        if hds_test_details:
            perp_values = extract_perplexity_values(hds_test_details)
            for h in ['DD', 'OT', 'RC']:
                _, se = mean_and_se(perp_values.get(h, []))
                perp_se[h] = se
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\HDSTestPerp{h}SE{model_suffix}}}{{{perp_se[h]:.2f}}}")

        # OT vs DD gap
        ot_perp = perp.get('OT', 0)
        dd_perp = perp.get('DD', 0)
        rc_perp = perp.get('RC', 0)
        # Check for infinity to prevent NaN in gap calculations
        if dd_perp > 0 and dd_perp != float('inf') and ot_perp != float('inf'):
            ot_dd_gap = ((ot_perp - dd_perp) / dd_perp) * 100
            macros.append(f"\\newcommand{{\\OTDDGapPercent{model_suffix}}}{{{ot_dd_gap:.1f}\\%}}")
        if dd_perp > 0 and dd_perp != float('inf') and rc_perp != float('inf'):
            # RC vs DD gap
            rc_dd_gap = ((rc_perp - dd_perp) / dd_perp) * 100
            macros.append(f"\\newcommand{{\\RCDDGapPercent{model_suffix}}}{{{rc_dd_gap:.1f}\\%}}")

        ot_dd_gap_se = 0.0
        rc_dd_gap_se = 0.0
        if hds_test_details:
            ot_dd_pairs = extract_loss_pairs(hds_test_details, "OT", "DD")
            rc_dd_pairs = extract_loss_pairs(hds_test_details, "RC", "DD")
            ot_dd_gap_se = percent_ratio_se(ot_dd_pairs)
            rc_dd_gap_se = percent_ratio_se(rc_dd_pairs)
        macros.append(f"\\newcommand{{\\OTDDGapPercentSE{model_suffix}}}{{{ot_dd_gap_se:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\RCDDGapPercentSE{model_suffix}}}{{{rc_dd_gap_se:.1f}\\%}}")

        det = get_family_match_rate(hds_test) * 100
        macros.append(f"\\newcommand{{\\HDSTestDetection{model_suffix}}}{{{det:.1f}\\%}}")

        # Per-family lexical detection
        by_target = get_family_breakdown(hds_test)
        for h in ['DD', 'OT', 'RC']:
            h_data = by_target.get(h, {})
            det_rate = h_data.get('family_match_rate', h_data.get('detection_rate', 0)) * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}Detection{model_suffix}}}{{{det_rate:.0f}\\%}}")
            acc_rate = h_data.get('accuracy', 0) * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}Accuracy{model_suffix}}}{{{acc_rate:.0f}\\%}}")

        det_se = 0.0
        if hds_test_results:
            det_flags = [
                get_loss_detected_heuristic(r) == get_design_family(r)
                for r in hds_test_results
            ]
            det_se = compute_binary_se(det_flags)
        macros.append(f"\\newcommand{{\\HDSTestDetectionSE{model_suffix}}}{{{det_se:.1f}\\%}}")

        if hds_test_results:
            for h in ['DD', 'OT', 'RC']:
                target_rows = [r for r in hds_test_results if get_design_family(r) == h]
                det_flags = [get_loss_detected_heuristic(r) == h for r in target_rows]
                acc_flags = [parse_bool(r.get('is_correct')) for r in target_rows]
                det_se = compute_binary_se(det_flags)
                acc_se = compute_binary_se(acc_flags)
                macros.append(f"\\newcommand{{\\HDSTest{h}DetectionSE{model_suffix}}}{{{det_se:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\HDSTest{h}AccuracySE{model_suffix}}}{{{acc_se:.1f}\\%}}")

    # Bootstrap confidence intervals (if results CSV available)
    if hds_test_results:
        macros.append("")
        macros.append(f"% Bootstrap 95% Confidence Intervals{' - ' + model_slug if model_slug else ''}")
        for h in ['DD', 'OT', 'RC']:
            rate, lower, upper = compute_detection_rate_ci(hds_test_results, h)
            # Format as "X.X% (X.X-X.X)"
            ci_str = f"{rate:.1f}\\% ({lower:.1f}--{upper:.1f})"
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionCI{model_suffix}}}{{{ci_str}}}")
            # Also provide individual bounds
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionLower{model_suffix}}}{{{lower:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionUpper{model_suffix}}}{{{upper:.1f}\\%}}")

    # Image modality results (if available)
    if hds_test_image:
        macros.append("")
        macros.append(f"% HDS Test Split Results - Image Modality{' - ' + model_slug if model_slug else ''}")
        acc = hds_test_image.get('accuracy', 0) * 100
        macros.append(f"\\newcommand{{\\HDSTestAccuracyImage{model_suffix}}}{{{acc:.1f}\\%}}")

        perp = hds_test_image.get('avg_perplexity', {})
        macros.append(f"\\newcommand{{\\HDSTestPerpDDImage{model_suffix}}}{{{perp.get('DD', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\HDSTestPerpOTImage{model_suffix}}}{{{perp.get('OT', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\HDSTestPerpRCImage{model_suffix}}}{{{perp.get('RC', 0):.2f}}}")

        acc_se = 0.0
        if hds_test_image_results:
            acc_flags = [parse_bool(r.get('is_correct')) for r in hds_test_image_results]
            acc_se = compute_binary_se(acc_flags)
        macros.append(f"\\newcommand{{\\HDSTestAccuracyImageSE{model_suffix}}}{{{acc_se:.1f}\\%}}")

        perp_se = {"DD": 0.0, "OT": 0.0, "RC": 0.0}
        if hds_test_image_details:
            perp_values = extract_perplexity_values(hds_test_image_details)
            for h in ['DD', 'OT', 'RC']:
                _, se = mean_and_se(perp_values.get(h, []))
                perp_se[h] = se
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\HDSTestPerp{h}ImageSE{model_suffix}}}{{{perp_se[h]:.2f}}}")

        det = get_family_match_rate(hds_test_image) * 100
        macros.append(f"\\newcommand{{\\HDSTestDetectionImage{model_suffix}}}{{{det:.1f}\\%}}")

        # Per-family lexical detection for image modality
        by_target = get_family_breakdown(hds_test_image)
        for h in ['DD', 'OT', 'RC']:
            h_data = by_target.get(h, {})
            det_rate = h_data.get('family_match_rate', h_data.get('detection_rate', 0)) * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImage{model_suffix}}}{{{det_rate:.0f}\\%}}")
            acc_rate = h_data.get('accuracy', 0) * 100
            macros.append(f"\\newcommand{{\\HDSTest{h}AccuracyImage{model_suffix}}}{{{acc_rate:.0f}\\%}}")

        det_se = 0.0
        if hds_test_image_results:
            det_flags = [
                get_loss_detected_heuristic(r) == get_design_family(r)
                for r in hds_test_image_results
            ]
            det_se = compute_binary_se(det_flags)
        macros.append(f"\\newcommand{{\\HDSTestDetectionImageSE{model_suffix}}}{{{det_se:.1f}\\%}}")

        if hds_test_image_results:
            for h in ['DD', 'OT', 'RC']:
                target_rows = [r for r in hds_test_image_results if get_design_family(r) == h]
                det_flags = [get_loss_detected_heuristic(r) == h for r in target_rows]
                acc_flags = [parse_bool(r.get('is_correct')) for r in target_rows]
                det_se = compute_binary_se(det_flags)
                acc_se = compute_binary_se(acc_flags)
                macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImageSE{model_suffix}}}{{{det_se:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\HDSTest{h}AccuracyImageSE{model_suffix}}}{{{acc_se:.1f}\\%}}")

        # OT vs DD gap for image modality
        ot_perp = perp.get('OT', 0)
        dd_perp = perp.get('DD', 0)
        rc_perp = perp.get('RC', 0)
        # Check for infinity to prevent NaN in gap calculations
        if dd_perp > 0 and dd_perp != float('inf') and ot_perp != float('inf'):
            ot_dd_gap = ((ot_perp - dd_perp) / dd_perp) * 100
            macros.append(f"\\newcommand{{\\OTDDGapPercentImage{model_suffix}}}{{{ot_dd_gap:.1f}\\%}}")
        if dd_perp > 0 and dd_perp != float('inf') and rc_perp != float('inf'):
            rc_dd_gap = ((rc_perp - dd_perp) / dd_perp) * 100
            macros.append(f"\\newcommand{{\\RCDDGapPercentImage{model_suffix}}}{{{rc_dd_gap:.1f}\\%}}")

        ot_dd_gap_se = 0.0
        rc_dd_gap_se = 0.0
        if hds_test_image_details:
            ot_dd_pairs = extract_loss_pairs(hds_test_image_details, "OT", "DD")
            rc_dd_pairs = extract_loss_pairs(hds_test_image_details, "RC", "DD")
            ot_dd_gap_se = percent_ratio_se(ot_dd_pairs)
            rc_dd_gap_se = percent_ratio_se(rc_dd_pairs)
        macros.append(f"\\newcommand{{\\OTDDGapPercentImageSE{model_suffix}}}{{{ot_dd_gap_se:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\RCDDGapPercentImageSE{model_suffix}}}{{{rc_dd_gap_se:.1f}\\%}}")

    # Bootstrap confidence intervals for image modality
    if hds_test_image_results:
        macros.append("")
        macros.append(f"% Bootstrap 95% Confidence Intervals - Image Modality{' - ' + model_slug if model_slug else ''}")
        for h in ['DD', 'OT', 'RC']:
            rate, lower, upper = compute_detection_rate_ci(hds_test_image_results, h)
            ci_str = f"{rate:.1f}\\% ({lower:.1f}--{upper:.1f})"
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImageCI{model_suffix}}}{{{ci_str}}}")
            # Individual bounds (matching text modality)
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImageLower{model_suffix}}}{{{lower:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\HDSTest{h}DetectionImageUpper{model_suffix}}}{{{upper:.1f}\\%}}")

    # Traps image modality results
    if traps_image:
        macros.append("")
        macros.append(f"% Traps Results - Image Modality{' - ' + model_slug if model_slug else ''}")
        acc = traps_image.get('accuracy', 0) * 100
        macros.append(f"\\newcommand{{\\TrapsAccuracyImage{model_suffix}}}{{{acc:.1f}\\%}}")

        perp = traps_image.get('avg_perplexity', {})
        macros.append(f"\\newcommand{{\\TrapsPerpDDImage{model_suffix}}}{{{perp.get('DD', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\TrapsPerpOTImage{model_suffix}}}{{{perp.get('OT', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\TrapsPerpRCImage{model_suffix}}}{{{perp.get('RC', 0):.2f}}}")

        acc_se = 0.0
        if traps_image_results:
            acc_flags = [parse_bool(r.get('is_correct')) for r in traps_image_results]
            acc_se = compute_binary_se(acc_flags)
        macros.append(f"\\newcommand{{\\TrapsAccuracyImageSE{model_suffix}}}{{{acc_se:.1f}\\%}}")

        perp_se = {"DD": 0.0, "OT": 0.0, "RC": 0.0}
        if traps_image_details:
            perp_values = extract_perplexity_values(traps_image_details)
            for h in ['DD', 'OT', 'RC']:
                _, se = mean_and_se(perp_values.get(h, []))
                perp_se[h] = se
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\TrapsPerp{h}ImageSE{model_suffix}}}{{{perp_se[h]:.2f}}}")

        by_target = get_family_breakdown(traps_image)
        for h in ['DD', 'OT', 'RC']:
            h_data = by_target.get(h, {})
            det_rate = (get_family_detection_rate(h_data) or 0.0) * 100
            macros.append(f"\\newcommand{{\\Traps{h}DetectionImage{model_suffix}}}{{{det_rate:.0f}\\%}}")

        if traps_image_results:
            for h in ['DD', 'OT', 'RC']:
                target_rows = [r for r in traps_image_results if get_design_family(r) == h]
                det_flags = [get_loss_detected_heuristic(r) == h for r in target_rows]
                det_se = compute_binary_se(det_flags)
                macros.append(f"\\newcommand{{\\Traps{h}DetectionImageSE{model_suffix}}}{{{det_se:.1f}\\%}}")

        soft_stats = get_soft_target_stats(traps_image)
        for h in ['DD', 'OT', 'RC']:
            h_stats = soft_stats.get(h, {}) or {}
            support = (h_stats.get('target_support') or 0.0) * 100
            support_se = (h_stats.get('target_support_se') or 0.0) * 100
            macros.append(f"\\newcommand{{\\Traps{h}TargetSupportImage{model_suffix}}}{{{support:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\Traps{h}TargetSupportImageSE{model_suffix}}}{{{support_se:.1f}\\%}}")

    # Modality comparison macros (text vs image)
    if hds_test and hds_test_image:
        macros.append("")
        macros.append(f"% Modality Comparison (Text vs Image){' - ' + model_slug if model_slug else ''}")
        text_acc = hds_test.get('accuracy', 0) * 100
        image_acc = hds_test_image.get('accuracy', 0) * 100
        acc_gap = text_acc - image_acc
        macros.append(f"\\newcommand{{\\ModalityAccuracyGap{model_suffix}}}{{{abs(acc_gap):.1f}\\%}}")

        # Perplexity comparison
        text_perp = hds_test.get('avg_perplexity', {})
        image_perp = hds_test_image.get('avg_perplexity', {})
        for h in ['DD', 'OT', 'RC']:
            text_val = text_perp.get(h, 0)
            image_val = image_perp.get(h, 0)
            if text_val > 0:
                perp_change = ((image_val - text_val) / text_val) * 100
                macros.append(f"\\newcommand{{\\Perp{h}ModalityChange{model_suffix}}}{{{perp_change:+.1f}\\%}}")

        perp_change_se = {"DD": 0.0, "OT": 0.0, "RC": 0.0}
        if hds_test_details and hds_test_image_details:
            text_by_id = build_perplexity_by_id(hds_test_details)
            image_by_id = build_perplexity_by_id(hds_test_image_details)
            for h in ['DD', 'OT', 'RC']:
                pairs = []
                for hds_id, text_vals in text_by_id.items():
                    image_vals = image_by_id.get(hds_id)
                    if not image_vals:
                        continue
                    pairs.append((image_vals[h], text_vals[h]))
                perp_change_se[h] = percent_ratio_se(pairs)
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\Perp{h}ModalityChangeSE{model_suffix}}}{{{perp_change_se[h]:.1f}\\%}}")

        # Cross-modal diagnostic metrics (detect if results are suspiciously similar)
        if hds_test_details and hds_test_image_details:
            text_by_id = build_perplexity_by_id(hds_test_details)
            image_by_id = build_perplexity_by_id(hds_test_image_details)

            # Pearson correlation between text/image DD perplexities
            dd_text_vals = []
            dd_image_vals = []
            neutral_deltas = []
            trace_identical_count = 0
            trace_total_count = 0

            text_details_by_id = {r.get('hds_id'): r for r in hds_test_details}
            image_details_by_id = {r.get('hds_id'): r for r in hds_test_image_details}

            for hds_id, text_perp in text_by_id.items():
                image_perp = image_by_id.get(hds_id)
                if image_perp:
                    dd_text_vals.append(text_perp.get('DD', 0))
                    dd_image_vals.append(image_perp.get('DD', 0))

                    # Neutral loss delta
                    text_detail = text_details_by_id.get(hds_id, {})
                    image_detail = image_details_by_id.get(hds_id, {})
                    text_neutral = text_detail.get('perplexity', {}).get('neutral_loss', 0)
                    image_neutral = image_detail.get('perplexity', {}).get('neutral_loss', 0)
                    if text_neutral > 0 and image_neutral > 0:
                        neutral_deltas.append(abs(image_neutral - text_neutral))

                    # Trace identity check
                    text_trace = text_detail.get('generation', {}).get('trace', '')
                    image_trace = image_detail.get('generation', {}).get('trace', '')
                    if text_trace and image_trace:
                        trace_total_count += 1
                        if text_trace == image_trace:
                            trace_identical_count += 1

            # Compute Pearson correlation for DD perplexities
            perp_corr = 0.0
            if len(dd_text_vals) > 1:
                mean_text = sum(dd_text_vals) / len(dd_text_vals)
                mean_image = sum(dd_image_vals) / len(dd_image_vals)
                cov = sum((t - mean_text) * (i - mean_image) for t, i in zip(dd_text_vals, dd_image_vals)) / len(dd_text_vals)
                std_text = (sum((t - mean_text) ** 2 for t in dd_text_vals) / len(dd_text_vals)) ** 0.5
                std_image = (sum((i - mean_image) ** 2 for i in dd_image_vals) / len(dd_image_vals)) ** 0.5
                if std_text > 0 and std_image > 0:
                    perp_corr = cov / (std_text * std_image)

            avg_neutral_delta = sum(neutral_deltas) / len(neutral_deltas) if neutral_deltas else 0.0
            trace_identical_rate = (trace_identical_count / trace_total_count * 100) if trace_total_count > 0 else 0.0

            macros.append(f"\\newcommand{{\\ModalityPerpCorrelation{model_suffix}}}{{{perp_corr:.3f}}}")
            macros.append(f"\\newcommand{{\\ModalityTraceIdenticalRate{model_suffix}}}{{{trace_identical_rate:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\ModalityNeutralDelta{model_suffix}}}{{{avg_neutral_delta:.3f}}}")

    # Error-based detection analysis (ErrorShapeParser)
    if hds_test and hds_test.get('error_detection'):
        macros.append("")
        macros.append(f"% Error-Based Detection (ErrorShapeParser){' - ' + model_slug if model_slug else ''}")
        ed = hds_test['error_detection']
        macros.append(f"\\newcommand{{\\HDSTestErrorsTotal{model_suffix}}}{{{ed.get('total_with_errors', 0)}}}")
        macros.append(f"\\newcommand{{\\HDSTestErrorsDetected{model_suffix}}}{{{ed.get('errors_with_heuristic_detected', 0)}}}")
        by_h = ed.get('by_heuristic', {})
        for h in ['DD', 'OT', 'RC']:
            h_data = by_h.get(h, {})
            count = h_data.get('count', 0)
            conf = h_data.get('avg_confidence')
            macros.append(f"\\newcommand{{\\HDSTestError{h}Count{model_suffix}}}{{{count}}}")
            if conf is not None:
                macros.append(f"\\newcommand{{\\HDSTestError{h}Conf{model_suffix}}}{{{conf:.2f}}}")

    # Trace-based detection analysis (TraceClassifier)
    if hds_test and hds_test.get('trace_detection'):
        macros.append("")
        macros.append(f"% Trace-Based Detection (TraceClassifier){' - ' + model_slug if model_slug else ''}")
        td = hds_test['trace_detection']
        macros.append(f"\\newcommand{{\\HDSTestTracesTotal{model_suffix}}}{{{td.get('total_with_traces', 0)}}}")
        macros.append(f"\\newcommand{{\\HDSTestTracesDetected{model_suffix}}}{{{td.get('traces_with_heuristic_detected', 0)}}}")
        by_h = td.get('by_heuristic', {})
        for h in ['DD', 'OT', 'RC']:
            h_data = by_h.get(h, {})
            count = h_data.get('count', 0)
            conf = h_data.get('avg_confidence')
            macros.append(f"\\newcommand{{\\HDSTestTrace{h}Count{model_suffix}}}{{{count}}}")
            if conf is not None:
                macros.append(f"\\newcommand{{\\HDSTestTrace{h}Conf{model_suffix}}}{{{conf:.2f}}}")

    if hds_test and hds_test.get('embedding_detection'):
        macros.append("")
        macros.append(f"% Trace-Embedding Detection (PrototypeEmbeddingClassifier){' - ' + model_slug if model_slug else ''}")
        ed = hds_test['embedding_detection']
        macros.append(f"\\newcommand{{\\HDSTestTraceEmbedTracesTotal{model_suffix}}}{{{ed.get('total_with_traces', 0)}}}")
        macros.append(f"\\newcommand{{\\HDSTestTraceEmbedResolvedTotal{model_suffix}}}{{{ed.get('resolved_embeddings', 0)}}}")
        by_h = ed.get('by_heuristic', {})
        for h in ['DD', 'OT', 'RC', 'STYLE']:
            h_data = by_h.get(h, {})
            count = h_data.get('count', 0)
            conf = h_data.get('avg_confidence')
            margin = h_data.get('avg_margin')
            macros.append(f"\\newcommand{{\\HDSTestTraceEmbed{h}Count{model_suffix}}}{{{count}}}")
            if conf is not None:
                macros.append(f"\\newcommand{{\\HDSTestTraceEmbed{h}Conf{model_suffix}}}{{{conf:.2f}}}")
            if margin is not None:
                macros.append(f"\\newcommand{{\\HDSTestTraceEmbed{h}Margin{model_suffix}}}{{{margin:.2f}}}")

    # Delta loss analysis (new multi-template baseline approach)
    if hds_test and hds_test.get('avg_delta_loss'):
        macros.append("")
        macros.append(f"% Delta Loss (relative to neutral baseline){' - ' + model_slug if model_slug else ''}")
        neutral = hds_test.get('avg_neutral_loss')
        if neutral is not None:
            macros.append(f"\\newcommand{{\\HDSTestNeutralLoss{model_suffix}}}{{{neutral:.4f}}}")
        delta = hds_test['avg_delta_loss']
        for h in ['DD', 'OT', 'RC']:
            d = delta.get(h)
            if d is not None:
                macros.append(f"\\newcommand{{\\HDSTestDelta{h}{model_suffix}}}{{{d:+.4f}}}")

    # Error-based detection analysis - Image Modality (ErrorShapeParser)
    if hds_test_image and hds_test_image.get('error_detection'):
        macros.append("")
        macros.append(f"% Error-Based Detection - Image Modality (ErrorShapeParser){' - ' + model_slug if model_slug else ''}")
        ed = hds_test_image['error_detection']
        macros.append(f"\\newcommand{{\\HDSTestErrorsTotalImage{model_suffix}}}{{{ed.get('total_with_errors', 0)}}}")
        macros.append(f"\\newcommand{{\\HDSTestErrorsDetectedImage{model_suffix}}}{{{ed.get('errors_with_heuristic_detected', 0)}}}")
        by_h = ed.get('by_heuristic', {})
        for h in ['DD', 'OT', 'RC']:
            h_data = by_h.get(h, {})
            count = h_data.get('count', 0)
            conf = h_data.get('avg_confidence')
            macros.append(f"\\newcommand{{\\HDSTestError{h}CountImage{model_suffix}}}{{{count}}}")
            if conf is not None:
                macros.append(f"\\newcommand{{\\HDSTestError{h}ConfImage{model_suffix}}}{{{conf:.2f}}}")

    # Trace-based detection analysis - Image Modality (TraceClassifier)
    if hds_test_image and hds_test_image.get('trace_detection'):
        macros.append("")
        macros.append(f"% Trace-Based Detection - Image Modality (TraceClassifier){' - ' + model_slug if model_slug else ''}")
        td = hds_test_image['trace_detection']
        macros.append(f"\\newcommand{{\\HDSTestTracesTotalImage{model_suffix}}}{{{td.get('total_with_traces', 0)}}}")
        macros.append(f"\\newcommand{{\\HDSTestTracesDetectedImage{model_suffix}}}{{{td.get('traces_with_heuristic_detected', 0)}}}")
        by_h = td.get('by_heuristic', {})
        for h in ['DD', 'OT', 'RC']:
            h_data = by_h.get(h, {})
            count = h_data.get('count', 0)
            conf = h_data.get('avg_confidence')
            macros.append(f"\\newcommand{{\\HDSTestTrace{h}CountImage{model_suffix}}}{{{count}}}")
            if conf is not None:
                macros.append(f"\\newcommand{{\\HDSTestTrace{h}ConfImage{model_suffix}}}{{{conf:.2f}}}")

    if hds_test_image and hds_test_image.get('embedding_detection'):
        macros.append("")
        macros.append(f"% Trace-Embedding Detection - Image Modality (PrototypeEmbeddingClassifier){' - ' + model_slug if model_slug else ''}")
        ed = hds_test_image['embedding_detection']
        macros.append(f"\\newcommand{{\\HDSTestTraceEmbedTracesTotalImage{model_suffix}}}{{{ed.get('total_with_traces', 0)}}}")
        macros.append(f"\\newcommand{{\\HDSTestTraceEmbedResolvedTotalImage{model_suffix}}}{{{ed.get('resolved_embeddings', 0)}}}")
        by_h = ed.get('by_heuristic', {})
        for h in ['DD', 'OT', 'RC', 'STYLE']:
            h_data = by_h.get(h, {})
            count = h_data.get('count', 0)
            conf = h_data.get('avg_confidence')
            margin = h_data.get('avg_margin')
            macros.append(f"\\newcommand{{\\HDSTestTraceEmbed{h}CountImage{model_suffix}}}{{{count}}}")
            if conf is not None:
                macros.append(f"\\newcommand{{\\HDSTestTraceEmbed{h}ConfImage{model_suffix}}}{{{conf:.2f}}}")
            if margin is not None:
                macros.append(f"\\newcommand{{\\HDSTestTraceEmbed{h}MarginImage{model_suffix}}}{{{margin:.2f}}}")

    # Delta loss analysis - Image Modality (multi-template baseline approach)
    if hds_test_image and hds_test_image.get('avg_delta_loss'):
        macros.append("")
        macros.append(f"% Delta Loss - Image Modality (relative to neutral baseline){' - ' + model_slug if model_slug else ''}")
        neutral = hds_test_image.get('avg_neutral_loss')
        if neutral is not None:
            macros.append(f"\\newcommand{{\\HDSTestNeutralLossImage{model_suffix}}}{{{neutral:.4f}}}")
        delta = hds_test_image['avg_delta_loss']
        for h in ['DD', 'OT', 'RC']:
            d = delta.get(h)
            if d is not None:
                macros.append(f"\\newcommand{{\\HDSTestDelta{h}Image{model_suffix}}}{{{d:+.4f}}}")

    def _fmt_rate(value: Optional[float]) -> str:
        return f"{value * 100:.1f}\\%" if value is not None else "[PENDING]"

    def _fmt_delta(value: Optional[float]) -> str:
        return f"{value:+.4f}" if value is not None else "[PENDING]"

    def _fmt_count(value: Optional[float]) -> str:
        return f"{int(value)}" if value is not None else "[PENDING]"

    def _fmt_pct_or_pending(value: Optional[float]) -> str:
        return f"{value * 100:.1f}\\%" if value is not None else "[PENDING]"

    def _fmt_std_or_pending(value: Optional[float]) -> str:
        return f"{value:.4f}" if value is not None else "[PENDING]"

    def _emit_template_profile_summary_macros(
        balanced: Optional[Dict[str, Any]],
        style_mismatch: Optional[Dict[str, Any]],
        image_tag: str,
    ) -> None:
        if not balanced and not style_mismatch:
            return
        suffix = f"{image_tag}{model_suffix}"
        label = "Image Modality" if image_tag else "Text Modality"
        macros.append("")
        macros.append(
            f"% Template profile summary - {label}{' - ' + model_slug if model_slug else ''}"
        )

        def emit(profile_prefix: str, analysis: Optional[Dict[str, Any]]) -> None:
            accuracy = analysis.get('accuracy') if analysis else None
            heuristic_match = get_family_match_rate(analysis) if analysis else None
            by_target = get_family_breakdown(analysis)
            variability = extract_template_variability_stats(analysis)
            macros.append(
                f"\\newcommand{{\\Template{profile_prefix}Accuracy{suffix}}}{{{_fmt_pct_or_pending(accuracy)}}}"
            )
            macros.append(
                f"\\newcommand{{\\Template{profile_prefix}HeuristicMatch{suffix}}}{{{_fmt_pct_or_pending(heuristic_match)}}}"
            )
            macros.append(
                f"\\newcommand{{\\Template{profile_prefix}MeanStd{suffix}}}{{{_fmt_std_or_pending(variability['mean'])}}}"
            )
            for h in ['DD', 'OT', 'RC']:
                det_rate = get_family_detection_rate(by_target.get(h))
                macros.append(
                    f"\\newcommand{{\\Template{profile_prefix}{h}Detection{suffix}}}{{{_fmt_pct_or_pending(det_rate)}}}"
                )
                macros.append(
                    f"\\newcommand{{\\Template{profile_prefix}{h}MeanStd{suffix}}}{{{_fmt_std_or_pending(variability[h])}}}"
                )

        emit("Balanced", balanced)
        emit("StyleMismatch", style_mismatch)

    def _emit_contrastive_macros(label: str, analysis: Optional[Dict[str, Any]], image_tag: str) -> None:
        suffix = f"{image_tag}{model_suffix}"
        if not analysis and not emit_contrastive_placeholders:
            return
        macros.append("")
        macros.append(f"% Contrastive Step Probe - {label}{' - ' + model_slug if model_slug else ''}")
        if not analysis:
            macros.append(f"\\newcommand{{\\ContrastiveCount{suffix}}}{{[PENDING]}}")
            macros.append(f"\\newcommand{{\\ContrastivePrefRate{suffix}}}{{[PENDING]}}")
            macros.append(f"\\newcommand{{\\ContrastivePrefRateSE{suffix}}}{{[PENDING]}}")
            macros.append(f"\\newcommand{{\\ContrastiveDelta{suffix}}}{{[PENDING]}}")
            macros.append(f"\\newcommand{{\\ContrastiveDeltaSE{suffix}}}{{[PENDING]}}")
            for h in ['DD', 'OT', 'RC']:
                macros.append(f"\\newcommand{{\\Contrastive{h}Count{suffix}}}{{[PENDING]}}")
                macros.append(f"\\newcommand{{\\Contrastive{h}PrefRate{suffix}}}{{[PENDING]}}")
                macros.append(f"\\newcommand{{\\Contrastive{h}PrefRateSE{suffix}}}{{[PENDING]}}")
                macros.append(f"\\newcommand{{\\Contrastive{h}Delta{suffix}}}{{[PENDING]}}")
                macros.append(f"\\newcommand{{\\Contrastive{h}DeltaSE{suffix}}}{{[PENDING]}}")
            return

        overall = analysis.get("overall", {})
        macros.append(f"\\newcommand{{\\ContrastiveCount{suffix}}}{{{_fmt_count(overall.get('count'))}}}")
        macros.append(f"\\newcommand{{\\ContrastivePrefRate{suffix}}}{{{_fmt_rate(overall.get('pref_rate'))}}}")
        macros.append(f"\\newcommand{{\\ContrastivePrefRateSE{suffix}}}{{{_fmt_rate(overall.get('pref_rate_se'))}}}")
        macros.append(f"\\newcommand{{\\ContrastiveDelta{suffix}}}{{{_fmt_delta(overall.get('mean_delta'))}}}")
        macros.append(f"\\newcommand{{\\ContrastiveDeltaSE{suffix}}}{{{_fmt_delta(overall.get('delta_se'))}}}")

        by_target = get_canonical_breakdown(analysis)
        for h in ['DD', 'OT', 'RC']:
            stats = by_target.get(h, {})
            macros.append(f"\\newcommand{{\\Contrastive{h}Count{suffix}}}{{{_fmt_count(stats.get('count'))}}}")
            macros.append(f"\\newcommand{{\\Contrastive{h}PrefRate{suffix}}}{{{_fmt_rate(stats.get('pref_rate'))}}}")
            macros.append(f"\\newcommand{{\\Contrastive{h}PrefRateSE{suffix}}}{{{_fmt_rate(stats.get('pref_rate_se'))}}}")
            macros.append(f"\\newcommand{{\\Contrastive{h}Delta{suffix}}}{{{_fmt_delta(stats.get('mean_delta'))}}}")
            macros.append(f"\\newcommand{{\\Contrastive{h}DeltaSE{suffix}}}{{{_fmt_delta(stats.get('delta_se'))}}}")

    _emit_contrastive_macros("Text Modality", contrastive_hds_test, "")
    _emit_contrastive_macros("Image Modality", contrastive_hds_test_image, "Image")
    _emit_template_profile_summary_macros(
        template_summary_hds_test if template_summary_hds_test is not None else hds_test,
        hds_test_style_mismatch,
        "",
    )
    _emit_template_profile_summary_macros(
        template_summary_hds_test_image if template_summary_hds_test_image is not None else hds_test_image,
        hds_test_image_style_mismatch,
        "Image",
    )

    macros.append("")
    macros.append(f"% HDS All Splits Results{' - ' + model_slug if model_slug else ''}")
    if hds_all:
        acc = hds_all.get('accuracy', 0) * 100
        macros.append(f"\\newcommand{{\\HDSAllAccuracy{model_suffix}}}{{{acc:.1f}\\%}}")

        perp = hds_all.get('avg_perplexity', {})
        macros.append(f"\\newcommand{{\\HDSAllPerpDD{model_suffix}}}{{{perp.get('DD', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\HDSAllPerpOT{model_suffix}}}{{{perp.get('OT', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\HDSAllPerpRC{model_suffix}}}{{{perp.get('RC', 0):.2f}}}")

    macros.append("")
    macros.append(f"% Traps Results{' - ' + model_slug if model_slug else ''}")
    if traps:
        acc = traps.get('accuracy', 0) * 100
        macros.append(f"\\newcommand{{\\TrapsAccuracy{model_suffix}}}{{{acc:.1f}\\%}}")

        perp = traps.get('avg_perplexity', {})
        macros.append(f"\\newcommand{{\\TrapsPerpDD{model_suffix}}}{{{perp.get('DD', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\TrapsPerpOT{model_suffix}}}{{{perp.get('OT', 0):.2f}}}")
        macros.append(f"\\newcommand{{\\TrapsPerpRC{model_suffix}}}{{{perp.get('RC', 0):.2f}}}")

        acc_se = 0.0
        if traps_results:
            acc_flags = [parse_bool(r.get('is_correct')) for r in traps_results]
            acc_se = compute_binary_se(acc_flags)
        macros.append(f"\\newcommand{{\\TrapsAccuracySE{model_suffix}}}{{{acc_se:.1f}\\%}}")

        perp_se = {"DD": 0.0, "OT": 0.0, "RC": 0.0}
        if traps_details:
            perp_values = extract_perplexity_values(traps_details)
            for h in ['DD', 'OT', 'RC']:
                _, se = mean_and_se(perp_values.get(h, []))
                perp_se[h] = se
        for h in ['DD', 'OT', 'RC']:
            macros.append(f"\\newcommand{{\\TrapsPerp{h}SE{model_suffix}}}{{{perp_se[h]:.2f}}}")

        by_target = get_family_breakdown(traps)
        for h in ['DD', 'OT', 'RC']:
            h_data = by_target.get(h, {})
            det_rate = (get_family_detection_rate(h_data) or 0.0) * 100
            macros.append(f"\\newcommand{{\\Traps{h}Detection{model_suffix}}}{{{det_rate:.0f}\\%}}")

        if traps_results:
            for h in ['DD', 'OT', 'RC']:
                target_rows = [r for r in traps_results if get_design_family(r) == h]
                det_flags = [get_loss_detected_heuristic(r) == h for r in target_rows]
                det_se = compute_binary_se(det_flags)
                macros.append(f"\\newcommand{{\\Traps{h}DetectionSE{model_suffix}}}{{{det_se:.1f}\\%}}")

        soft_stats = get_soft_target_stats(traps)
        for h in ['DD', 'OT', 'RC']:
            h_stats = soft_stats.get(h, {}) or {}
            support = (h_stats.get('target_support') or 0.0) * 100
            support_se = (h_stats.get('target_support_se') or 0.0) * 100
            macros.append(f"\\newcommand{{\\Traps{h}TargetSupport{model_suffix}}}{{{support:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\Traps{h}TargetSupportSE{model_suffix}}}{{{support_se:.1f}\\%}}")

    # LoRA Training Results
    if lora_training:
        macros.append("")
        macros.append(f"% LoRA Training Results{' - ' + model_slug if model_slug else ''}")
        heuristics = lora_training.get('heuristics', {})
        total_train = 0
        total_time = 0

        for h in ['RC', 'DD', 'OT']:
            h_data = heuristics.get(h, {})
            train_n = h_data.get('num_examples', 0)
            val_n = h_data.get('num_val_examples', 0)
            best_val_loss = h_data.get('best_val_loss', 0)
            best_epoch = h_data.get('best_epoch', 0)
            elapsed = h_data.get('elapsed_seconds', 0)

            total_train += train_n
            total_time += elapsed

            macros.append(f"\\newcommand{{\\LoRA{h}TrainN{model_suffix}}}{{{train_n}}}")
            macros.append(f"\\newcommand{{\\LoRA{h}ValN{model_suffix}}}{{{val_n}}}")
            macros.append(f"\\newcommand{{\\LoRA{h}BestValLoss{model_suffix}}}{{{best_val_loss:.3f}}}")
            macros.append(f"\\newcommand{{\\LoRA{h}BestEpoch{model_suffix}}}{{{best_epoch}}}")
            macros.append("")

        macros.append(f"\\newcommand{{\\LoRATotalTrainN{model_suffix}}}{{{total_train}}}")
        macros.append(f"\\newcommand{{\\LoRATotalTime{model_suffix}}}{{{total_time/60:.1f} min}}")

    # Nudge Test Results
    if nudge_test:
        primary_nudge_test = _filter_nudge_analysis_by_heuristics(nudge_test, PRIMARY_LORA_HEURISTICS)
        control_nudge_test = {
            heuristic: payload
            for heuristic, payload in (nudge_test or {}).items()
            if normalize_heuristic_label(heuristic) not in PRIMARY_LORA_RESULT_LABELS
        }
        primary_nudge_results = _filter_nudge_result_rows(nudge_test_results, PRIMARY_LORA_HEURISTICS)
        primary_nudge_details = _filter_nudge_detail_rows(nudge_test_details, PRIMARY_LORA_DETAIL_LABELS)
        macros.append("")
        macros.append(f"% LoRA Nudge Test Results{' - ' + model_slug if model_slug else ''}")
        # Get first heuristic to find total problems
        first_h = list(primary_nudge_test.keys())[0] if primary_nudge_test else None
        if first_h:
            n_problems = primary_nudge_test[first_h].get('total_problems', 0)
            macros.append(f"\\newcommand{{\\NudgeTestCount{model_suffix}}}{{{n_problems}}}")
            lora_count = len(primary_nudge_test)
            macros.append(f"\\newcommand{{\\NudgeLoRACount{model_suffix}}}{{{lora_count}}}")
            macros.append(
                f"\\newcommand{{\\NudgeTotalComparisons{model_suffix}}}{{{lora_count * n_problems}}}"
            )

            # Calculate base accuracy from by_target_heuristic
            by_target = primary_nudge_test[first_h].get('by_design_family', primary_nudge_test[first_h].get('by_target_heuristic', {}))
            base_correct = sum(v.get('base_correct', 0) for v in by_target.values())
            base_known_total = sum(v.get('base_known_total', v.get('total', 0)) for v in by_target.values())
            if base_known_total > 0:
                base_acc = (base_correct / base_known_total) * 100
                macros.append(f"\\newcommand{{\\NudgeBaseAccuracy{model_suffix}}}{{{base_acc:.1f}\\%}}")

            base_acc_se = 0.0
            if primary_nudge_results:
                base_by_id = {}
                for row in primary_nudge_results:
                    hds_id = row.get('hds_id')
                    if not hds_id or hds_id in base_by_id:
                        continue
                    parsed = parse_correctness_label(row.get('base_correctness', row.get('base_correct')))
                    if parsed is None:
                        continue
                    base_by_id[hds_id] = parsed
                if base_by_id:
                    base_acc_se = compute_binary_se(list(base_by_id.values()))
            macros.append(f"\\newcommand{{\\NudgeBaseAccuracySE{model_suffix}}}{{{base_acc_se:.1f}\\%}}")

        if primary_nudge_results:
            base_target_flags = []
            lora_target_flags = []
            for row in primary_nudge_results:
                target = get_design_family(row)
                base_detected = row.get('base_detected')
                lora_detected = row.get('lora_detected')
                if target and base_detected:
                    base_target_flags.append(base_detected == target)
                if target and lora_detected:
                    lora_target_flags.append(lora_detected == target)
            if base_target_flags:
                base_target_rate = sum(base_target_flags) / len(base_target_flags) * 100
                macros.append(
                    f"\\newcommand{{\\NudgeTargetDetectionBase{model_suffix}}}{{{base_target_rate:.1f}\\%}}"
                )
            if lora_target_flags:
                lora_target_rate = sum(lora_target_flags) / len(lora_target_flags) * 100
                macros.append(
                    f"\\newcommand{{\\NudgeTargetDetectionLoRA{model_suffix}}}{{{lora_target_rate:.1f}\\%}}"
                )

            base_mismatch_total = 0
            base_mismatch_dd = 0
            lora_mismatch_total = 0
            lora_mismatch_dd = 0
            for row in primary_nudge_results:
                target = get_design_family(row)
                base_detected = row.get('base_detected')
                lora_detected = row.get('lora_detected')
                if target and base_detected and base_detected != target:
                    base_mismatch_total += 1
                    if base_detected == "DD":
                        base_mismatch_dd += 1
                if target and lora_detected and lora_detected != target:
                    lora_mismatch_total += 1
                    if lora_detected == "DD":
                        lora_mismatch_dd += 1
            if base_mismatch_total > 0:
                base_mismatch_dd_rate = base_mismatch_dd / base_mismatch_total * 100
                macros.append(
                    f"\\newcommand{{\\NudgeMismatchToDDBase{model_suffix}}}{{{base_mismatch_dd_rate:.1f}\\%}}"
                )
            if lora_mismatch_total > 0:
                lora_mismatch_dd_rate = lora_mismatch_dd / lora_mismatch_total * 100
                macros.append(
                    f"\\newcommand{{\\NudgeMismatchToDDLoRA{model_suffix}}}{{{lora_mismatch_dd_rate:.1f}\\%}}"
                )

        # Total flips across heuristic LoRAs only
        total_flips = sum(h.get('flips', {}).get('total', 0) for h in primary_nudge_test.values())
        improved = sum(h.get('flips', {}).get('improved', 0) for h in primary_nudge_test.values())
        degraded = sum(h.get('flips', {}).get('degraded', 0) for h in primary_nudge_test.values())
        detection_flips = sum(h.get('detection_flips', {}).get('total', 0) for h in primary_nudge_test.values())

        macros.append(f"\\newcommand{{\\NudgeTotalFlips{model_suffix}}}{{{total_flips}}}")
        macros.append(f"\\newcommand{{\\NudgeImproved{model_suffix}}}{{{improved}}}")
        macros.append(f"\\newcommand{{\\NudgeDegraded{model_suffix}}}{{{degraded}}}")
        macros.append(f"\\newcommand{{\\NudgeDetectionFlips{model_suffix}}}{{{detection_flips}}}")

        if primary_nudge_details:
            taxonomy = analyze_nudge_taxonomy(primary_nudge_details)
            degraded_counts = taxonomy.get("degraded", {})
            macros.append(
                f"\\newcommand{{\\NudgeDegradedCarryDrop{model_suffix}}}{{{degraded_counts.get('carry_drop', 0)}}}"
            )
            macros.append(
                f"\\newcommand{{\\NudgeDegradedPartialProduct{model_suffix}}}{{{degraded_counts.get('partial_product_omission', 0)}}}"
            )
            macros.append(
                f"\\newcommand{{\\NudgeDegradedMagnitudeSlip{model_suffix}}}{{{degraded_counts.get('magnitude_slip', 0)}}}"
            )

        if control_nudge_test:
            style_control = control_nudge_test.get("STYLE") or next(iter(control_nudge_test.values()))
            control_problems = style_control.get("total_problems", 0)
            control_flips = style_control.get("flips", {})
            control_detection_flips = style_control.get("detection_flips", {})
            macros.append("")
            macros.append(f"% LoRA Control Results{' - ' + model_slug if model_slug else ''}")
            macros.append(f"\\newcommand{{\\StyleControlCount{model_suffix}}}{{{len(control_nudge_test)}}}")
            macros.append(
                f"\\newcommand{{\\StyleControlTotalComparisons{model_suffix}}}{{{len(control_nudge_test) * control_problems}}}"
            )
            macros.append(f"\\newcommand{{\\StyleControlFlips{model_suffix}}}{{{control_flips.get('total', 0)}}}")
            macros.append(f"\\newcommand{{\\StyleControlImproved{model_suffix}}}{{{control_flips.get('improved', 0)}}}")
            macros.append(f"\\newcommand{{\\StyleControlDegraded{model_suffix}}}{{{control_flips.get('degraded', 0)}}}")
            macros.append(
                f"\\newcommand{{\\StyleControlDetectionFlips{model_suffix}}}{{{control_detection_flips.get('total', 0)}}}"
            )

    # Nudge Test Results - Image Modality
    if nudge_test_image:
        primary_nudge_test_image = _filter_nudge_analysis_by_heuristics(nudge_test_image, PRIMARY_LORA_HEURISTICS)
        control_nudge_test_image = {
            heuristic: payload
            for heuristic, payload in (nudge_test_image or {}).items()
            if normalize_heuristic_label(heuristic) not in PRIMARY_LORA_RESULT_LABELS
        }
        primary_nudge_test_image_results = _filter_nudge_result_rows(nudge_test_image_results, PRIMARY_LORA_HEURISTICS)
        macros.append("")
        macros.append(f"% LoRA Nudge Test Results - Image Modality{' - ' + model_slug if model_slug else ''}")
        first_h = list(primary_nudge_test_image.keys())[0] if primary_nudge_test_image else None
        if first_h:
            n_problems = primary_nudge_test_image[first_h].get('total_problems', 0)
            macros.append(f"\\newcommand{{\\NudgeTestCountImage{model_suffix}}}{{{n_problems}}}")

            # Calculate base accuracy from by_target_heuristic
            by_target = primary_nudge_test_image[first_h].get('by_design_family', primary_nudge_test_image[first_h].get('by_target_heuristic', {}))
            base_correct = sum(v.get('base_correct', 0) for v in by_target.values())
            base_known_total = sum(v.get('base_known_total', v.get('total', 0)) for v in by_target.values())
            if base_known_total > 0:
                base_acc = (base_correct / base_known_total) * 100
                macros.append(f"\\newcommand{{\\NudgeBaseAccuracyImage{model_suffix}}}{{{base_acc:.1f}\\%}}")

            base_acc_se = 0.0
            if primary_nudge_test_image_results:
                base_by_id = {}
                for row in primary_nudge_test_image_results:
                    hds_id = row.get('hds_id')
                    if not hds_id or hds_id in base_by_id:
                        continue
                    parsed = parse_correctness_label(row.get('base_correctness', row.get('base_correct')))
                    if parsed is None:
                        continue
                    base_by_id[hds_id] = parsed
                if base_by_id:
                    base_acc_se = compute_binary_se(list(base_by_id.values()))
            macros.append(f"\\newcommand{{\\NudgeBaseAccuracyImageSE{model_suffix}}}{{{base_acc_se:.1f}\\%}}")

        if primary_nudge_test_image_results:
            base_target_flags = []
            lora_target_flags = []
            for row in primary_nudge_test_image_results:
                target = get_design_family(row)
                base_detected = row.get('base_detected')
                lora_detected = row.get('lora_detected')
                if target and base_detected:
                    base_target_flags.append(base_detected == target)
                if target and lora_detected:
                    lora_target_flags.append(lora_detected == target)
            if base_target_flags:
                base_target_rate = sum(base_target_flags) / len(base_target_flags) * 100
                macros.append(
                    f"\\newcommand{{\\NudgeTargetDetectionBaseImage{model_suffix}}}{{{base_target_rate:.1f}\\%}}"
                )
            if lora_target_flags:
                lora_target_rate = sum(lora_target_flags) / len(lora_target_flags) * 100
                macros.append(
                    f"\\newcommand{{\\NudgeTargetDetectionLoRAImage{model_suffix}}}{{{lora_target_rate:.1f}\\%}}"
                )

        # Total flips across heuristic LoRAs for image modality
        total_flips = sum(h.get('flips', {}).get('total', 0) for h in primary_nudge_test_image.values())
        improved = sum(h.get('flips', {}).get('improved', 0) for h in primary_nudge_test_image.values())
        degraded = sum(h.get('flips', {}).get('degraded', 0) for h in primary_nudge_test_image.values())
        detection_flips = sum(h.get('detection_flips', {}).get('total', 0) for h in primary_nudge_test_image.values())

        macros.append(f"\\newcommand{{\\NudgeTotalFlipsImage{model_suffix}}}{{{total_flips}}}")
        macros.append(f"\\newcommand{{\\NudgeImprovedImage{model_suffix}}}{{{improved}}}")
        macros.append(f"\\newcommand{{\\NudgeDegradedImage{model_suffix}}}{{{degraded}}}")
        macros.append(f"\\newcommand{{\\NudgeDetectionFlipsImage{model_suffix}}}{{{detection_flips}}}")

        if control_nudge_test_image:
            style_control_image = control_nudge_test_image.get("STYLE") or next(iter(control_nudge_test_image.values()))
            control_image_problems = style_control_image.get("total_problems", 0)
            control_image_flips = style_control_image.get("flips", {})
            control_image_detection_flips = style_control_image.get("detection_flips", {})
            macros.append(f"\\newcommand{{\\StyleControlCountImage{model_suffix}}}{{{len(control_nudge_test_image)}}}")
            macros.append(
                f"\\newcommand{{\\StyleControlTotalComparisonsImage{model_suffix}}}{{{len(control_nudge_test_image) * control_image_problems}}}"
            )
            macros.append(f"\\newcommand{{\\StyleControlFlipsImage{model_suffix}}}{{{control_image_flips.get('total', 0)}}}")
            macros.append(f"\\newcommand{{\\StyleControlImprovedImage{model_suffix}}}{{{control_image_flips.get('improved', 0)}}}")
            macros.append(f"\\newcommand{{\\StyleControlDegradedImage{model_suffix}}}{{{control_image_flips.get('degraded', 0)}}}")
            macros.append(
                f"\\newcommand{{\\StyleControlDetectionFlipsImage{model_suffix}}}{{{control_image_detection_flips.get('total', 0)}}}"
            )

    # Gradient Orthogonality / Cosine Similarity Results
    if gradient_analysis:
        macros.append("")

        analysis_format = gradient_analysis.get('format', 'unknown')

        if analysis_format in ('cosine_similarity', 'effective_update_similarity'):
            # Cosine similarity results (effective updates preferred)
            label = "Effective Update Cosine Similarity Results" if analysis_format == 'effective_update_similarity' else "Adapter Weight Cosine Similarity Results"
            macros.append(f"% {label}{' - ' + model_slug if model_slug else ''}")
            similarities = gradient_analysis.get('similarities', {})

            # Extract pairwise cosine similarities (may be None in placeholder mode)
            sim_dd_ot = similarities.get('DD-OT')
            sim_dd_rc = similarities.get('DD-RC')
            sim_ot_rc = similarities.get('OT-RC')

            # Handle None values (placeholder mode)
            def fmt_sim(val):
                return f"{val:.4f}" if val is not None else "N/A"

            def fmt_corr(val):
                return f"{val:.4f}" if val is not None else "N/A"

            macros.append(f"\\newcommand{{\\CosineDDOT{model_suffix}}}{{{fmt_sim(sim_dd_ot)}}}")
            macros.append(f"\\newcommand{{\\CosineDDRC{model_suffix}}}{{{fmt_sim(sim_dd_rc)}}}")
            macros.append(f"\\newcommand{{\\CosineOTRC{model_suffix}}}{{{fmt_sim(sim_ot_rc)}}}")

            # Keep old macro names for backwards compatibility (using cosine values)
            macros.append(f"\\newcommand{{\\CorrOTDD{model_suffix}}}{{{fmt_corr(sim_dd_ot)}}}")
            macros.append(f"\\newcommand{{\\CorrOTRC{model_suffix}}}{{{fmt_corr(sim_ot_rc)}}}")
            macros.append(f"\\newcommand{{\\CorrDDRC{model_suffix}}}{{{fmt_corr(sim_dd_rc)}}}")

            finite_similarities = [
                float(value)
                for value in similarities.values()
                if isinstance(value, (int, float)) and math.isfinite(value)
            ]
            if finite_similarities:
                macros.append(
                    f"\\newcommand{{\\EffectiveUpdateMinCross{model_suffix}}}{{{min(finite_similarities):.2f}}}"
                )
                macros.append(
                    f"\\newcommand{{\\EffectiveUpdateMaxCross{model_suffix}}}{{{max(finite_similarities):.2f}}}"
                )
            else:
                macros.append(f"\\newcommand{{\\EffectiveUpdateMinCross{model_suffix}}}{{N/A}}")
                macros.append(f"\\newcommand{{\\EffectiveUpdateMaxCross{model_suffix}}}{{N/A}}")

            seed_summary = gradient_analysis.get("seed_control_summary") or {}
            macros.append(
                f"\\newcommand{{\\EffectiveUpdateSameSeedAvg{model_suffix}}}{{{fmt_sim(seed_summary.get('same_heuristic_avg'))}}}"
            )
            macros.append(
                f"\\newcommand{{\\EffectiveUpdatePrimaryCrossAvg{model_suffix}}}{{{fmt_sim(seed_summary.get('primary_cross_avg'))}}}"
            )
            macros.append(
                f"\\newcommand{{\\EffectiveUpdateSeedGap{model_suffix}}}{{{fmt_sim(seed_summary.get('gap'))}}}"
            )

            # Vector statistics
            stats = gradient_analysis.get('vector_stats', {})
            for h in ['DD', 'OT', 'RC']:
                h_stats = stats.get(h, {})
                n_params = h_stats.get('num_parameters', 0)
                l2_norm = h_stats.get('l2_norm', 0)
                macros.append(f"\\newcommand{{\\Adapter{h}Params{model_suffix}}}{{{n_params:,}}}")
                macros.append(f"\\newcommand{{\\Adapter{h}Norm{model_suffix}}}{{{l2_norm:.4f}}}")

        else:
            # Old format: loss correlation (for backwards compatibility)
            macros.append(f"% Gradient Orthogonality Results (loss correlation){' - ' + model_slug if model_slug else ''}")
            corr_matrix = gradient_analysis.get('correlation_matrix', {})
            corr_ot_dd = corr_matrix.get('OT-DD', 0)
            corr_ot_rc = corr_matrix.get('OT-RC', 0)
            corr_dd_rc = corr_matrix.get('DD-RC', 0)
            macros.append(f"\\newcommand{{\\CorrOTDD{model_suffix}}}{{{corr_ot_dd:.2f}}}")
            macros.append(f"\\newcommand{{\\CorrOTRC{model_suffix}}}{{{corr_ot_rc:.2f}}}")
            macros.append(f"\\newcommand{{\\CorrDDRC{model_suffix}}}{{{corr_dd_rc:.2f}}}")
            macros.append(f"\\newcommand{{\\EffectiveUpdateMinCross{model_suffix}}}{{N/A}}")
            macros.append(f"\\newcommand{{\\EffectiveUpdateMaxCross{model_suffix}}}{{N/A}}")
            macros.append(f"\\newcommand{{\\EffectiveUpdateSameSeedAvg{model_suffix}}}{{N/A}}")
            macros.append(f"\\newcommand{{\\EffectiveUpdatePrimaryCrossAvg{model_suffix}}}{{N/A}}")
            macros.append(f"\\newcommand{{\\EffectiveUpdateSeedGap{model_suffix}}}{{N/A}}")
    elif emit_gradient_placeholders:
        # No gradient analysis available - add placeholder macros
        # These will be updated when GradientOrthogonality.py is run
        macros.append("")
        macros.append(f"% Gradient Orthogonality Results (PENDING - run GradientOrthogonality.py){' - ' + model_slug if model_slug else ''}")
        macros.append(f"\\newcommand{{\\CorrOTDD{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\CorrOTRC{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\CorrDDRC{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\EffectiveUpdateMinCross{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\EffectiveUpdateMaxCross{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\EffectiveUpdateSameSeedAvg{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\EffectiveUpdatePrimaryCrossAvg{model_suffix}}}{{[PENDING]}}")
        macros.append(f"\\newcommand{{\\EffectiveUpdateSeedGap{model_suffix}}}{{[PENDING]}}")

    if emit_suffix_macros:
        # Suffix accuracy analysis (smoking gun for columnar hypothesis)
        # These are computed dynamically when the figure is generated
        # and added to the macro file via the suffix_metrics return value
        macros.append("")
        macros.append(f"% Suffix Accuracy Analysis (columnar computation test){' - ' + model_slug if model_slug else ''}")

        # Theoretical baselines (constants used in figure and paper)
        # Random chance: 10%, 1%, 0.1%, 0.01% for k=1,2,3,4
        macros.append("% Random chance baseline for suffix matching")
        macros.append(f"\\newcommand{{\\SuffixRandomKOne{model_suffix}}}{{10\\%}}")
        macros.append(f"\\newcommand{{\\SuffixRandomKTwo{model_suffix}}}{{1\\%}}")
        macros.append(f"\\newcommand{{\\SuffixRandomKThree{model_suffix}}}{{0.1\\%}}")
        macros.append(f"\\newcommand{{\\SuffixRandomKFour{model_suffix}}}{{0.01\\%}}")
        # Columnar prediction: errors propagate left-to-right, preserving low digits
        macros.append("% Expected suffix accuracy if using columnar multiplication")
        macros.append(f"\\newcommand{{\\SuffixColumnarKOne{model_suffix}}}{{85\\%}}")
        macros.append(f"\\newcommand{{\\SuffixColumnarKTwo{model_suffix}}}{{70\\%}}")
        macros.append(f"\\newcommand{{\\SuffixColumnarKThree{model_suffix}}}{{50\\%}}")
        macros.append(f"\\newcommand{{\\SuffixColumnarKFour{model_suffix}}}{{35\\%}}")

    if hds_test_results:
        macros.append(f"% Observed suffix accuracy (text modality){' - ' + model_slug if model_slug else ''}")
        suffix_data = analyze_suffix_accuracy(hds_test_results)
        rates = suffix_data.get('all_errors', [])
        if rates and rates[0][1] > 0:
            se_values = []
            for rate, total in rates:
                if total > 0:
                    se_values.append(math.sqrt(rate * (1 - rate) / total) * 100)
                else:
                    se_values.append(0.0)
            macros.append(f"\\newcommand{{\\SuffixKOneText{model_suffix}}}{{{rates[0][0]*100:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\SuffixKOneTextSE{model_suffix}}}{{{se_values[0]:.1f}\\%}}")
            if len(rates) > 1:
                macros.append(f"\\newcommand{{\\SuffixKTwoText{model_suffix}}}{{{rates[1][0]*100:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\SuffixKTwoTextSE{model_suffix}}}{{{se_values[1]:.1f}\\%}}")
            if len(rates) > 2:
                macros.append(f"\\newcommand{{\\SuffixKThreeText{model_suffix}}}{{{rates[2][0]*100:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\SuffixKThreeTextSE{model_suffix}}}{{{se_values[2]:.1f}\\%}}")
            if len(rates) > 3:
                macros.append(f"\\newcommand{{\\SuffixKFourText{model_suffix}}}{{{rates[3][0]*100:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\SuffixKFourTextSE{model_suffix}}}{{{se_values[3]:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\SuffixErrorCountText{model_suffix}}}{{{rates[0][1]}}}")

    if hds_test_image_results:
        macros.append(f"% Observed suffix accuracy (image modality){' - ' + model_slug if model_slug else ''}")
        suffix_data_image = analyze_suffix_accuracy(hds_test_image_results)
        rates_image = suffix_data_image.get('all_errors', [])
        if rates_image and rates_image[0][1] > 0:
            se_values = []
            for rate, total in rates_image:
                if total > 0:
                    se_values.append(math.sqrt(rate * (1 - rate) / total) * 100)
                else:
                    se_values.append(0.0)
            macros.append(f"\\newcommand{{\\SuffixKOneImage{model_suffix}}}{{{rates_image[0][0]*100:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\SuffixKOneImageSE{model_suffix}}}{{{se_values[0]:.1f}\\%}}")
            if len(rates_image) > 1:
                macros.append(f"\\newcommand{{\\SuffixKTwoImage{model_suffix}}}{{{rates_image[1][0]*100:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\SuffixKTwoImageSE{model_suffix}}}{{{se_values[1]:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\SuffixErrorCountImage{model_suffix}}}{{{rates_image[0][1]}}}")

    # Modulo invariant analysis (mod 2, 5, 9, 10)
    macros.append("")
    macros.append(f"% Modulo Invariant Analysis{' - ' + model_slug if model_slug else ''}")
    macros.append("% Tests whether wrong answers preserve modular properties")

    if hds_test_results:
        mod_data = analyze_modulo_invariants(hds_test_results)
        error_rates = mod_data.get('errors', {})
        macros.append(f"% Text modality modulo preservation (wrong answers)")
        for m in [2, 5, 9, 10]:
            rate = error_rates.get(f'mod{m}', 0) * 100
            mod_label = number_to_words(m)
            macros.append(f"\\newcommand{{\\ModuloText{mod_label}{model_suffix}}}{{{rate:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\ModuloTextErrorCount{model_suffix}}}{{{mod_data.get('error_count', 0)}}}")

    if hds_test_image_results:
        mod_data_image = analyze_modulo_invariants(hds_test_image_results)
        error_rates_image = mod_data_image.get('errors', {})
        macros.append(f"% Image modality modulo preservation (wrong answers)")
        for m in [2, 5, 9, 10]:
            rate = error_rates_image.get(f'mod{m}', 0) * 100
            mod_label = number_to_words(m)
            macros.append(f"\\newcommand{{\\ModuloImage{mod_label}{model_suffix}}}{{{rate:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\ModuloImageErrorCount{model_suffix}}}{{{mod_data_image.get('error_count', 0)}}}")

    # Suffix by complexity analysis
    if hds_test_results:
        complexity_data = analyze_suffix_by_complexity(hds_test_results)
        macros.append("")
        macros.append(f"% Suffix Accuracy by Complexity{' - ' + model_slug if model_slug else ''}")
        for bin_label, rates in complexity_data.items():
            if rates and rates[0][1] > 0:
                bin_safe = macro_safe_range_label(bin_label)
                macros.append(f"\\newcommand{{\\SuffixComplexity{bin_safe}KOne{model_suffix}}}{{{rates[0][0]*100:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\SuffixComplexity{bin_safe}Count{model_suffix}}}{{{rates[0][1]}}}")

    # Suffix by detected heuristic analysis
    if hds_test_results:
        heuristic_data = analyze_suffix_by_heuristic(hds_test_results)
        macros.append("")
        macros.append(f"% Suffix Accuracy by Detected Heuristic{' - ' + model_slug if model_slug else ''}")
        for heuristic, rates in heuristic_data.items():
            if rates and rates[0][1] > 0:
                macros.append(f"\\newcommand{{\\SuffixHeuristic{heuristic}KOne{model_suffix}}}{{{rates[0][0]*100:.1f}\\%}}")
                macros.append(f"\\newcommand{{\\SuffixHeuristic{heuristic}Count{model_suffix}}}{{{rates[0][1]}}}")

    # Generate alternative Table 3 macros from JSONL (for verification/fallback)
    if hds_test_details or hds_test_image_details:
        alt_macros = generate_jsonl_alt_macros(
            hds_test_details,
            hds_test_image_details,
            model_suffix
        )
        if alt_macros.strip():
            macros.append("")
            macros.append(alt_macros)

    return "\n".join(macros)


def plot_perplexity_comparison(hds_test: dict, traps: dict, output_path: Path):
    """Create bar chart comparing forced-completion loss across heuristics and datasets."""
    fig, ax = plt.subplots(figsize=(8, 5))

    heuristics = ['DD', 'RC', 'OT']
    x = np.arange(len(heuristics))
    width = 0.35

    # Get perplexity values (use avg_perplexity key)
    hds_perp = hds_test.get('avg_perplexity', {})
    traps_perp = traps.get('avg_perplexity', {})

    hds_vals = [hds_perp.get(h, 0) for h in heuristics]
    traps_vals = [traps_perp.get(h, 0) for h in heuristics]

    bars1 = ax.bar(x - width/2, hds_vals, width, label='HDS (Test)', color='steelblue')
    bars2 = ax.bar(x + width/2, traps_vals, width, label='Traps', color='coral')

    ax.set_xlabel('Heuristic Template', fontsize=FONT_LABEL)
    ax.set_ylabel('Avg forced-completion loss (lower = deeper alignment)', fontsize=FONT_LABEL)
    ax.set_title('Forced-completion loss by heuristic template', fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Decomposition\n(DD)', 'Rounding\n(RC)', 'Columnar\n(OT)'])
    ax.legend()
    ax.set_ylim(0, max(max(hds_vals), max(traps_vals)) * 1.15)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=FONT_SMALL)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    tprint(f"Saved perplexity comparison to {output_path}")


def plot_detection_rates(hds_test: dict, traps: dict, output_path: Path):
    """Create bar chart of lexical family-match rates by design family."""
    fig, ax = plt.subplots(figsize=(8, 5))

    heuristics = ['RC', 'DD', 'OT']
    x = np.arange(len(heuristics))
    width = 0.35

    # Get lexical family-match rates
    hds_by_target = get_family_breakdown(hds_test)
    traps_by_target = get_family_breakdown(traps)

    hds_vals = [hds_by_target.get(h, {}).get('family_match_rate', hds_by_target.get(h, {}).get('detection_rate', 0)) * 100 for h in heuristics]
    traps_vals = [traps_by_target.get(h, {}).get('family_match_rate', traps_by_target.get(h, {}).get('detection_rate', 0)) * 100 for h in heuristics]

    bars1 = ax.bar(x - width/2, hds_vals, width, label='HDS (Test)', color='steelblue')
    bars2 = ax.bar(x + width/2, traps_vals, width, label='Traps', color='coral')

    ax.set_xlabel('Design Family', fontsize=FONT_LABEL)
    ax.set_ylabel('Lexical family match rate (%)', fontsize=FONT_LABEL)
    ax.set_title('Lexical family match rate by dataset', fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Rounding\n(RC)', 'Decomposition\n(DD)', 'Columnar\n(OT)'])
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=FONT_SMALL)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    tprint(f"Saved detection rates to {output_path}")


def plot_perplexity_comparison_multi(
    model_series: Sequence[Dict[str, Any]],
    output_path: Path
):
    """Create multi-panel forced-completion loss comparison (one panel per model)."""
    valid_models = [m for m in model_series if m.get('hds_test') and m.get('traps')]
    if not valid_models:
        tprint("Skipping perplexity comparison - missing data")
        return

    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    axes = np.atleast_1d(axes)

    heuristics = ['DD', 'RC', 'OT']
    x = np.arange(len(heuristics))
    width = 0.35

    max_val = 0.0
    panel_vals = []
    for model in valid_models:
        hds_perp = model['hds_test'].get('avg_perplexity', {})
        traps_perp = model['traps'].get('avg_perplexity', {})
        hds_vals = [hds_perp.get(h, 0) for h in heuristics]
        traps_vals = [traps_perp.get(h, 0) for h in heuristics]
        panel_vals.append((hds_vals, traps_vals))
        max_val = max(max_val, max(hds_vals + traps_vals))

    for ax, model, (hds_vals, traps_vals) in zip(axes, valid_models, panel_vals):
        bars1 = ax.bar(x - width/2, hds_vals, width, label='HDS (Test)', color='steelblue')
        bars2 = ax.bar(x + width/2, traps_vals, width, label='Traps', color='coral')

        ax.set_xticks(x)
        ax.set_xticklabels(['Decomposition\n(DD)', 'Rounding\n(RC)', 'Columnar\n(OT)'])
        ax.set_title(model['label'], fontsize=FONT_TITLE, fontweight='bold')
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.15)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=FONT_SMALL)

        if ax is axes[0]:
            ax.set_ylabel('Avg forced-completion loss (lower = deeper alignment)', fontsize=FONT_LABEL)
            ax.legend()

    fig.suptitle('Forced-completion loss by heuristic template', fontsize=FONT_SUPTITLE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    tprint(f"Saved perplexity comparison to {output_path}")


def plot_multimodal_perplexity_comparison(
    hds_test_text: dict,
    hds_test_image: dict,
    output_path: Path
):
    """Create grouped bar chart comparing text vs image forced-completion loss.

    Args:
        hds_test_text: HDS test analysis for text modality
        hds_test_image: HDS test analysis for image modality
        output_path: Path to save the figure
    """
    if not hds_test_text or not hds_test_image:
        tprint("Skipping multimodal perplexity comparison - missing data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    heuristics = ['DD', 'RC', 'OT']
    x = np.arange(len(heuristics))
    width = 0.35

    text_perp = hds_test_text.get('avg_perplexity', {})
    image_perp = hds_test_image.get('avg_perplexity', {})

    text_vals = [text_perp.get(h, 0) for h in heuristics]
    image_vals = [image_perp.get(h, 0) for h in heuristics]

    bars1 = ax.bar(x - width/2, text_vals, width, label='Text', color='steelblue')
    bars2 = ax.bar(x + width/2, image_vals, width, label='Image', color='seagreen')

    ax.set_xlabel('Heuristic Template', fontsize=FONT_LABEL)
    ax.set_ylabel('Avg forced-completion loss (lower = deeper alignment)', fontsize=FONT_LABEL)
    ax.set_title('Forced-completion loss by modality (HDS Test)', fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Decomposition\n(DD)', 'Rounding\n(RC)', 'Columnar\n(OT)'])
    ax.legend()
    ax.set_ylim(0, max(max(text_vals), max(image_vals)) * 1.15)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=FONT_SMALL)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    tprint(f"Saved multimodal perplexity comparison to {output_path}")


def plot_multimodal_detection_comparison(
    hds_test_text: dict,
    hds_test_image: dict,
    output_path: Path
):
    """Create grouped bar chart comparing text vs image target support rates.

    Args:
        hds_test_text: HDS test analysis for text modality
        hds_test_image: HDS test analysis for image modality
        output_path: Path to save the figure
    """
    if not hds_test_text or not hds_test_image:
        tprint("Skipping multimodal detection comparison - missing data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    heuristics = ['RC', 'DD', 'OT']
    x = np.arange(len(heuristics))
    width = 0.35

    text_vals = get_soft_target_support_series(hds_test_text, heuristics)
    image_vals = get_soft_target_support_series(hds_test_image, heuristics)

    bars1 = ax.bar(x - width/2, text_vals, width, label='Text', color='steelblue')
    bars2 = ax.bar(x + width/2, image_vals, width, label='Image', color='seagreen')

    ax.set_xlabel('Design Family', fontsize=FONT_LABEL)
    ax.set_ylabel('Target support among resolved probes (%)', fontsize=FONT_LABEL)
    ax.set_title('Target support by modality (HDS Test)', fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Rounding\n(RC)', 'Decomposition\n(DD)', 'Columnar\n(OT)'])
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=FONT_SMALL)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    tprint(f"Saved multimodal detection comparison to {output_path}")


def plot_confusion_matrix(analysis: dict, title: str, output_path: Path):
    """Create heatmap of detection confusion matrix."""
    confusion = analysis.get('confusion_matrix', {})

    targets = ['RC', 'DD', 'OT']
    detected = ['RC', 'DD', 'OT']

    matrix = np.zeros((3, 3))
    for i, t in enumerate(targets):
        for j, d in enumerate(detected):
            key = f"{t}->{d}"
            matrix[i, j] = confusion.get(key, 0)

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_norm = matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix_norm, cmap='Blues', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(detected)))
    ax.set_yticks(np.arange(len(targets)))
    ax.set_xticklabels(detected)
    ax.set_yticklabels(targets)
    ax.set_xlabel('Lowest-loss heuristic', fontsize=FONT_LABEL)
    ax.set_ylabel('Target Heuristic', fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)

    # Add text annotations
    for i in range(len(targets)):
        for j in range(len(detected)):
            val = matrix_norm[i, j]
            count = int(matrix[i, j])
            text = ax.text(j, i, f'{val:.0f}%\n({count})',
                          ha="center", va="center",
                          color="white" if val > 50 else "black", fontsize=FONT_SMALL)

    plt.colorbar(im, label='%')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    tprint(f"Saved confusion matrix to {output_path}")


def plot_heuristic_reversal(
    hds_test_text: dict,
    hds_test_image: dict,
    output_path: Path
):
    """Create Tufte-style slope graph showing cross-modal target support.

    Args:
        hds_test_text: HDS test analysis for text modality
        hds_test_image: HDS test analysis for image modality
        output_path: Path to save the figure
    """
    if not hds_test_text or not hds_test_image:
        tprint("Skipping heuristic reversal plot - missing data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Extract paper-facing target support rates from analyses
    heuristics = ['DD', 'OT', 'RC']
    labels = {
        'DD': 'Decomposition (DD)',
        'OT': 'Columnar (OT)',
        'RC': 'Rounding (RC)'
    }

    text_rates = get_soft_target_support_series(hds_test_text, heuristics)
    image_rates = get_soft_target_support_series(hds_test_image, heuristics)

    # Colors for each heuristic (colorblind-friendly)
    colors = {
        'DD': '#2ecc71',   # Green
        'OT': '#e74c3c',   # Red
        'RC': '#3498db'    # Blue
    }

    # X positions for text (0) and image (1)
    x_text, x_image = 0, 1

    # Plot connecting lines and points
    for i, h in enumerate(heuristics):
        # Draw line connecting text to image
        ax.plot([x_text, x_image], [text_rates[i], image_rates[i]],
                color=colors[h], linewidth=2.5, marker='o', markersize=12,
                label=labels[h], zorder=3)

        # Add value labels on left (text)
        ax.annotate(f'{text_rates[i]:.0f}%',
                   xy=(x_text, text_rates[i]),
                   xytext=(-35, 0),
                   textcoords='offset points',
                   ha='right', va='center',
                   fontsize=FONT_BASE, fontweight='bold',
                   color=colors[h])

        # Add value labels on right (image)
        ax.annotate(f'{image_rates[i]:.0f}%',
                   xy=(x_image, image_rates[i]),
                   xytext=(35, 0),
                   textcoords='offset points',
                   ha='left', va='center',
                   fontsize=FONT_BASE, fontweight='bold',
                   color=colors[h])

    # Style the plot (Tufte-inspired minimalism)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-5, 110)

    # X-axis labels
    ax.set_xticks([x_text, x_image])
    ax.set_xticklabels(['Text\nModality', 'Image\nModality'], fontsize=FONT_LABEL, fontweight='bold')

    # Y-axis
    ax.set_ylabel('Target support among resolved probes (%)', fontsize=FONT_LABEL)

    # Title
    ax.set_title('Cross-Modal Heuristic Analysis', fontsize=FONT_TITLE, fontweight='bold', pad=15)

    # Remove top and right spines (Tufte style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Add subtle gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # Legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=FONT_SMALL,
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved heuristic reversal plot to {output_path}")


def _plot_heuristic_reversal_panel(
    ax: Any,
    hds_test_text: dict,
    hds_test_image: dict,
    title: str,
    show_legend: bool = False
) -> None:
    heuristics = ['DD', 'OT', 'RC']
    labels = {
        'DD': 'Decomposition (DD)',
        'OT': 'Columnar (OT)',
        'RC': 'Rounding (RC)'
    }

    text_rates = get_soft_target_support_series(hds_test_text, heuristics)
    image_rates = get_soft_target_support_series(hds_test_image, heuristics)

    colors = {
        'DD': '#2ecc71',
        'OT': '#e74c3c',
        'RC': '#3498db'
    }
    linestyles = {
        'DD': '-',      # Solid line
        'OT': '--',     # Dashed line
        'RC': '-.'      # Dash-dot line
    }

    x_text, x_image = 0, 1

    for i, h in enumerate(heuristics):
        ax.plot([x_text, x_image], [text_rates[i], image_rates[i]],
                color=colors[h], linestyle=linestyles[h], linewidth=3.0,
                marker='o', markersize=12, label=labels[h], zorder=3)

        ax.annotate(f'{text_rates[i]:.0f}%',
                   xy=(x_text, text_rates[i]),
                   xytext=(-40, 0),
                   textcoords='offset points',
                   ha='right', va='center',
                   fontsize=FONT_LABEL, fontweight='bold',
                   color=colors[h])

        ax.annotate(f'{image_rates[i]:.0f}%',
                   xy=(x_image, image_rates[i]),
                   xytext=(40, 0),
                   textcoords='offset points',
                   ha='left', va='center',
                   fontsize=FONT_LABEL, fontweight='bold',
                   color=colors[h])

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-5, 110)
    ax.set_xticks([x_text, x_image])
    ax.set_xticklabels(['Text\nModality', 'Image\nModality'], fontsize=FONT_LABEL, fontweight='bold')
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

    if show_legend:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            fontsize=FONT_SMALL,
            framealpha=0.9
        )


def plot_heuristic_reversal_multi(
    model_series: Sequence[Dict[str, Any]],
    output_path: Path
):
    """Create multi-panel heuristic reversal plot (one panel per model)."""
    valid_models = [m for m in model_series if m.get('hds_test') and m.get('hds_test_image')]
    if not valid_models:
        tprint("Skipping heuristic reversal plot - missing data")
        return

    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6), sharey=True)
    axes = np.atleast_1d(axes)

    for idx, (ax, model) in enumerate(zip(axes, valid_models)):
        _plot_heuristic_reversal_panel(
            ax,
            model['hds_test'],
            model['hds_test_image'],
            model['label'],
            show_legend=(idx == 0)
        )

        if idx == 0:
            ax.set_ylabel('Target support among resolved probes (%)', fontsize=FONT_LABEL, fontweight='bold')

    fig.suptitle('Cross-Modal Heuristic Analysis', fontsize=FONT_SUPTITLE, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved heuristic reversal plot to {output_path}")


def analyze_suffix_accuracy(
    results: List[dict],
    max_k: int = 4
) -> Dict[str, List[Tuple[float, int]]]:
    """Analyze whether model answers preserve low-order digits.

    If LLMs used columnar multiplication (OT), they would compute
    digits right-to-left, meaning even wrong answers should preserve
    low-order digits. This tests that hypothesis.

    Args:
        results: List of result dicts with 'product', 'model_answer', 'is_correct'
        max_k: Maximum number of suffix digits to check (default 4)

    Returns:
        Dict with 'all_errors': [(match_rate, count) for k=1..max_k]
    """
    # Filter to incorrect answers with valid model output
    errors = []
    for r in results:
        is_correct = r.get('is_correct')
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() == 'true'
        if not is_correct and r.get('model_answer'):
            errors.append(r)

    # Compute suffix match rates for all errors
    rates_by_k = []
    for k in range(1, max_k + 1):
        matches = 0
        total = 0
        for r in errors:
            try:
                model_ans = r.get('model_answer', '')
                if model_ans is None or model_ans == '':
                    continue
                model = int(float(model_ans))
                correct = int(r.get('product', 0))
                if model > 0 and correct > 0:
                    mod = 10 ** k
                    if model % mod == correct % mod:
                        matches += 1
                    total += 1
            except (ValueError, TypeError):
                continue
        rate = matches / total if total > 0 else 0
        rates_by_k.append((rate, total))

    return {'all_errors': rates_by_k}


def compute_digit_complexity(a: int, b: int) -> int:
    """Compute digit complexity as total_digits × non_zero_digits.

    Args:
        a: First operand
        b: Second operand

    Returns:
        Complexity score
    """
    s = str(a) + str(b)
    total_digits = len(s)
    non_zero_digits = sum(1 for c in s if c != '0')
    return total_digits * non_zero_digits


def analyze_suffix_by_complexity(
    results: List[dict],
    complexity_bins: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, List[Tuple[float, int]]]:
    """Analyze suffix accuracy grouped by digit complexity.

    If errors are random, suffix accuracy should be uniform across complexity.
    If errors follow heuristic patterns, we might see complexity-dependent effects.

    Args:
        results: List of result dicts with 'a', 'b', 'product', 'model_answer', 'is_correct'
        complexity_bins: List of (min, max) tuples for complexity grouping.
                         Default: canonical hard-profile bands.

    Returns:
        Dict mapping bin label to [(match_rate, count) for k=1..4]
    """
    if complexity_bins is None:
        complexity_bins = list(CANONICAL_COMPLEXITY_BANDS)

    # Group errors by complexity
    errors_by_bin: Dict[str, List[dict]] = {f"{lo}-{hi}": [] for lo, hi in complexity_bins}

    for r in results:
        is_correct = r.get('is_correct')
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() == 'true'
        if not is_correct and r.get('model_answer'):
            try:
                a = int(r.get('a', 0))
                b = int(r.get('b', 0))
                complexity = compute_digit_complexity(a, b)

                for lo, hi in complexity_bins:
                    if lo <= complexity <= hi:
                        errors_by_bin[f"{lo}-{hi}"].append(r)
                        break
            except (ValueError, TypeError):
                continue

    # Compute suffix rates for each bin
    result = {}
    for bin_label, errors in errors_by_bin.items():
        rates_by_k = []
        for k in range(1, 5):
            matches = 0
            total = 0
            for r in errors:
                try:
                    model = int(float(r.get('model_answer', 0)))
                    correct = int(r.get('product', 0))
                    if model > 0 and correct > 0:
                        mod = 10 ** k
                        if model % mod == correct % mod:
                            matches += 1
                        total += 1
                except (ValueError, TypeError):
                    continue
            rate = matches / total if total > 0 else 0
            rates_by_k.append((rate, total))
        result[bin_label] = rates_by_k

    return result


def analyze_modulo_invariants(
    results: List[dict]
) -> ModuloInvariantAnalysis:
    """Analyze modular arithmetic invariants in model answers.

    Tests whether model answers preserve mod 2, 5, 9, 10 of true product.
    - mod 9 (casting out nines): digital root must match
    - mod 10: last digit must match (same as suffix k=1)
    - mod 2: parity must match
    - mod 5: must match (combined with mod 2 gives mod 10)

    Args:
        results: List of result dicts with 'product', 'model_answer', 'is_correct'

    Returns:
        Dict with 'errors' and 'correct' sub-dicts containing match rates for each modulus
    """
    moduli = [2, 5, 9, 10]

    def get_invariant_rates(records: List[dict]) -> Dict[str, float]:
        rates = {}
        for m in moduli:
            matches = 0
            total = 0
            for r in records:
                try:
                    model = int(float(r.get('model_answer', 0)))
                    correct = int(r.get('product', 0))
                    if model > 0 and correct > 0:
                        if model % m == correct % m:
                            matches += 1
                        total += 1
                except (ValueError, TypeError):
                    continue
            rates[f'mod{m}'] = matches / total if total > 0 else 0
            rates[f'mod{m}_count'] = total
        return rates

    # Split into errors and correct
    errors = []
    correct = []
    for r in results:
        is_correct = r.get('is_correct')
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() == 'true'
        if r.get('model_answer'):
            if is_correct:
                correct.append(r)
            else:
                errors.append(r)

    return {
        'errors': get_invariant_rates(errors),
        'correct': get_invariant_rates(correct),
        'error_count': len(errors),
        'correct_count': len(correct)
    }


def analyze_suffix_by_heuristic(
    results: List[dict]
) -> Dict[str, List[Tuple[float, int]]]:
    """Analyze suffix accuracy grouped by detected heuristic.

    Tests whether suffix accuracy differs by the heuristic the model appears to use.
    DD might show different patterns than RC due to different error propagation.

    Args:
        results: List of result dicts with 'detected_heuristic', 'product', 'model_answer', 'is_correct'

    Returns:
        Dict mapping heuristic to [(match_rate, count) for k=1..4]
    """
    # Group errors by detected heuristic
    errors_by_heuristic: Dict[str, List[dict]] = {'DD': [], 'OT': [], 'RC': [], 'UNKNOWN': []}

    for r in results:
        is_correct = r.get('is_correct')
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() == 'true'
        if not is_correct and r.get('model_answer'):
            h = get_loss_detected_heuristic(r)
            if h not in errors_by_heuristic:
                h = 'UNKNOWN'
            errors_by_heuristic[h].append(r)

    # Compute suffix rates for each heuristic
    result = {}
    for heuristic, errors in errors_by_heuristic.items():
        if not errors:
            continue
        rates_by_k = []
        for k in range(1, 5):
            matches = 0
            total = 0
            for r in errors:
                try:
                    model = int(float(r.get('model_answer', 0)))
                    correct = int(r.get('product', 0))
                    if model > 0 and correct > 0:
                        mod = 10 ** k
                        if model % mod == correct % mod:
                            matches += 1
                        total += 1
                except (ValueError, TypeError):
                    continue
            rate = matches / total if total > 0 else 0
            rates_by_k.append((rate, total))
        result[heuristic] = rates_by_k

    return result


def plot_suffix_by_complexity(
    results: List[dict],
    output_path: Path
) -> Dict[str, str]:
    """Create figure showing suffix accuracy by digit complexity.

    Args:
        results: List of result dicts
        output_path: Path to save the figure

    Returns:
        Dict of key metrics for LaTeX macros
    """
    import matplotlib.pyplot as plt

    complexity_data = analyze_suffix_by_complexity(results)

    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = {}
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    log_floor = 0.005

    ks = [1, 2, 3, 4]
    bin_labels = list(complexity_data.keys())

    for i, (bin_label, rates) in enumerate(complexity_data.items()):
        if not rates or rates[0][1] == 0:
            continue

        counts = [r[1] for r in rates]
        match_rates, zero_indices = _log_rate_series(rates, log_floor)

        ax.plot(ks, match_rates, color=colors[i % len(colors)],
                marker=markers[i % len(markers)], linewidth=2, markersize=8,
                label=f'Complexity {bin_label} (n={counts[0]})')
        if zero_indices:
            ax.scatter(
                [ks[idx] for idx in zero_indices],
                [log_floor] * len(zero_indices),
                marker='v',
                s=60,
                facecolors='none',
                edgecolors=colors[i % len(colors)],
                linewidths=1.5,
                zorder=4
            )

        # Store metrics
        metrics[f'SuffixComplexity{bin_label.replace("-", "to")}K1'] = f'{rates[0][0]*100:.1f}\\%'

    # Random chance baseline
    baseline = [10, 1, 0.1, 0.01]
    ax.plot(ks, baseline, 'k--', linewidth=2, label='Random Chance', alpha=0.7)

    ax.set_xlabel('Suffix Length (digits)', fontsize=FONT_LABEL)
    ax.set_ylabel('Suffix Match Rate (%)', fontsize=FONT_LABEL)
    ax.set_title('Suffix Accuracy by Digit Complexity\n'
                 '(Higher complexity = larger operands)', fontsize=FONT_TITLE)
    ax.set_xticks(ks)
    ax.set_xticklabels([f'Last {k}' for k in ks])
    ax.set_yscale('log')
    ax.set_ylim(log_floor, 100)
    ax.legend(loc='upper right', fontsize=FONT_SMALL)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved suffix by complexity plot to {output_path}")

    return metrics


def plot_suffix_by_complexity_multi(
    model_series: Sequence[Dict[str, Any]],
    output_path: Path
) -> None:
    """Create multi-panel suffix-by-complexity plot (one panel per model)."""
    valid_models = [m for m in model_series if m.get('hds_test_results')]
    if not valid_models:
        tprint("Skipping suffix by complexity plot - missing data")
        return

    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    axes = np.atleast_1d(axes)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    ks = [1, 2, 3, 4]
    log_floor = 0.005

    for ax, model in zip(axes, valid_models):
        complexity_data = analyze_suffix_by_complexity(model['hds_test_results'])
        for i, (bin_label, rates) in enumerate(complexity_data.items()):
            if not rates or rates[0][1] == 0:
                continue

            counts = [r[1] for r in rates]
            match_rates, zero_indices = _log_rate_series(rates, log_floor)

            ax.plot(ks, match_rates, color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], linewidth=2, markersize=7,
                    label=f'Complexity {bin_label} (n={counts[0]})')
            if zero_indices:
                ax.scatter(
                    [ks[idx] for idx in zero_indices],
                    [log_floor] * len(zero_indices),
                    marker='v',
                    s=55,
                    facecolors='none',
                    edgecolors=colors[i % len(colors)],
                    linewidths=1.5,
                    zorder=4
                )

        baseline = [10, 1, 0.1, 0.01]
        ax.plot(ks, baseline, 'k--', linewidth=2, label='Random Chance', alpha=0.7)

        ax.set_title(model['label'], fontsize=FONT_TITLE, fontweight='bold')
        ax.set_xlabel('Suffix Length (digits)', fontsize=FONT_BASE)
        ax.set_xticks(ks)
        ax.set_xticklabels([f'Last {k}' for k in ks])
        ax.set_yscale('log')
        ax.set_ylim(log_floor, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax is axes[0]:
            ax.set_ylabel('Suffix Match Rate (%)', fontsize=FONT_BASE)
            ax.legend(loc='upper right', fontsize=FONT_SMALL)

    fig.suptitle('Suffix Accuracy by Digit Complexity\n(Higher complexity = larger operands)',
                 fontsize=FONT_SUPTITLE, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved suffix by complexity plot to {output_path}")


def _style_empty_panel(
    ax: Any,
    title: str,
    message: str,
    title_size: int,
    message_size: int
) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha='center',
        va='center',
        transform=ax.transAxes,
        fontsize=message_size,
        color='#666'
    )
    ax.set_title(title, fontsize=title_size, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _log_rate_series(
    rates: List[Tuple[float, int]],
    log_floor: float
) -> Tuple[List[float], List[int]]:
    plot_rates: List[float] = []
    zero_indices: List[int] = []
    for idx, (rate, count) in enumerate(rates):
        if count == 0:
            plot_rates.append(np.nan)
            continue
        if rate <= 0:
            plot_rates.append(log_floor)
            zero_indices.append(idx)
        else:
            plot_rates.append(rate * 100)
    return plot_rates, zero_indices


def plot_modulo_invariants(
    text_results: List[dict],
    image_results: List[dict],
    output_path: Path
) -> Dict[str, str]:
    """Create figure showing modular invariant preservation.

    Args:
        text_results: Text modality results
        image_results: Image modality results
        output_path: Path to save the figure

    Returns:
        Dict of key metrics for LaTeX macros
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    metrics = {}
    moduli = ['mod2', 'mod5', 'mod9', 'mod10']
    mod_labels = ['mod 2\n(parity)', 'mod 5', 'mod 9\n(digit sum)', 'mod 10\n(last digit)']

    for ax, (results, label) in zip(axes, [(text_results, 'Text'), (image_results, 'Image')]):
        if not results:
            _style_empty_panel(
                ax,
                f'{label} Modality',
                f'No {label} data',
                FONT_TITLE,
                FONT_SMALL
            )
            continue

        mod_data = analyze_modulo_invariants(results)
        error_rates = mod_data['errors']
        correct_rates = mod_data['correct']

        if mod_data['error_count'] == 0:
            _style_empty_panel(
                ax,
                f'{label} Modality',
                'No errors to analyze',
                FONT_TITLE,
                FONT_SMALL
            )
            continue

        x = range(len(moduli))
        width = 0.35

        # Error answers
        error_vals = [error_rates.get(m, 0) * 100 for m in moduli]
        bars1 = ax.bar([i - width/2 for i in x], error_vals, width,
                      label=f'Wrong Answers (n={mod_data["error_count"]})',
                      color='#e74c3c', alpha=0.8)

        # Correct answers (should all be 100%)
        correct_vals = [correct_rates.get(m, 0) * 100 for m in moduli]
        bars2 = ax.bar([i + width/2 for i in x], correct_vals, width,
                      label=f'Correct Answers (n={mod_data["correct_count"]})',
                      color='#2ecc71', alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.0f}%',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=FONT_TINY)

        # Random chance baselines
        random_chances = [50, 20, 11.1, 10]
        for i, chance in enumerate(random_chances):
            ax.axhline(y=chance, xmin=(i/4)+0.05, xmax=((i+1)/4)-0.05,
                      color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Modulus', fontsize=FONT_LABEL)
        ax.set_title(f'{label} Modality', fontsize=FONT_TITLE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(mod_labels, fontsize=FONT_SMALL)
        ax.legend(loc='upper left', fontsize=FONT_SMALL)
        ax.set_ylim(0, 110)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Store metrics
        for m in moduli:
            rate = error_rates.get(m, 0) * 100
            metrics[f'{m.title()}{label}'] = f'{rate:.1f}\\%'

    axes[0].set_ylabel('Invariant Preservation Rate (%)', fontsize=FONT_LABEL)

    fig.suptitle('Do LLMs Preserve Modular Invariants in Wrong Answers?\n'
                 '(Correct answers always preserve; wrong answers reveal computation patterns)',
                 fontsize=FONT_TITLE, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved modulo invariants plot to {output_path}")

    return metrics


def plot_modulo_invariants_multi(
    model_series: Sequence[Dict[str, Any]],
    output_path: Path
) -> None:
    """Create multi-panel modulo invariant plot (rows=models, cols=modality)."""
    valid_models = [
        m for m in model_series
        if m.get('hds_test_results') or m.get('hds_test_image_results')
    ]
    if not valid_models:
        tprint("Skipping modulo invariants plot - missing data")
        return

    n_models = len(valid_models)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4.5 * n_models), sharey=True)
    axes = np.atleast_2d(axes)

    moduli = ['mod2', 'mod5', 'mod9', 'mod10']
    mod_labels = ['mod 2\n(parity)', 'mod 5', 'mod 9\n(digit sum)', 'mod 10\n(last digit)']

    for row, model in enumerate(valid_models):
        for col, (results, label) in enumerate([
            (model.get('hds_test_results'), 'Text'),
            (model.get('hds_test_image_results'), 'Image')
        ]):
            ax = axes[row, col]

            if not results:
                _style_empty_panel(
                    ax,
                    f'{model["label"]} - {label}',
                    f'No {label} data',
                    FONT_BASE,
                    FONT_SMALL
                )
                continue

            mod_data = analyze_modulo_invariants(results)
            error_rates = mod_data['errors']
            correct_rates = mod_data['correct']

            if mod_data['error_count'] == 0:
                _style_empty_panel(
                    ax,
                    f'{model["label"]} - {label}',
                    'No errors to analyze',
                    FONT_BASE,
                    FONT_SMALL
                )
                continue

            x = range(len(moduli))
            width = 0.35

            error_vals = [error_rates.get(m, 0) * 100 for m in moduli]
            bars1 = ax.bar([i - width/2 for i in x], error_vals, width,
                          label=f'Wrong (n={mod_data["error_count"]})',
                          color='#e74c3c', alpha=0.8)

            correct_vals = [correct_rates.get(m, 0) * 100 for m in moduli]
            bars2 = ax.bar([i + width/2 for i in x], correct_vals, width,
                          label=f'Correct (n={mod_data["correct_count"]})',
                          color='#2ecc71', alpha=0.8)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.0f}%',
                                   xy=(bar.get_x() + bar.get_width()/2, height),
                                   xytext=(0, 3), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=FONT_TINY)

            random_chances = [50, 20, 11.1, 10]
            for i, chance in enumerate(random_chances):
                ax.axhline(y=chance, xmin=(i/4)+0.05, xmax=((i+1)/4)-0.05,
                          color='gray', linestyle='--', alpha=0.5)

            ax.set_title(f'{model["label"]} - {label}', fontsize=FONT_BASE, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(mod_labels, fontsize=FONT_SMALL)
            ax.set_ylim(0, 110)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=FONT_TINY)

        axes[row, 0].set_ylabel('Invariant Preservation Rate (%)', fontsize=FONT_BASE)

    fig.suptitle('Do LLMs Preserve Modular Invariants in Wrong Answers?\n'
                 '(Correct answers always preserve; wrong answers reveal computation patterns)',
                 fontsize=FONT_TITLE, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved modulo invariants plot to {output_path}")


def plot_suffix_accuracy(
    text_results: List[dict],
    image_results: List[dict],
    output_path: Path
) -> Dict[str, str]:
    """Create figure showing suffix accuracy by modality.

    Tests whether LLMs preserve low-order digits in wrong answers.
    Columnar multiplication computes right-to-left, so should preserve
    low-order digits even when wrong. Random guessing gives 10%, 1%, 0.1%
    for k=1,2,3 digits respectively.

    Args:
        text_results: List of text modality fingerprint results
        image_results: List of image modality fingerprint results
        output_path: Path to save the figure

    Returns:
        Dict of key metrics for LaTeX macros
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    metrics = {}

    for ax, (results, label) in zip(axes,
            [(text_results, 'Text'), (image_results, 'Image')]):

        if not results:
            _style_empty_panel(
                ax,
                f'{label} Modality',
                f'No {label} data',
                FONT_TITLE,
                FONT_SMALL
            )
            continue

        suffix_data = analyze_suffix_accuracy(results)
        rates = suffix_data['all_errors']

        if not rates or rates[0][1] == 0:
            _style_empty_panel(
                ax,
                f'{label} Modality',
                'No errors to analyze',
                FONT_TITLE,
                FONT_SMALL
            )
            continue

        ks = list(range(1, len(rates) + 1))
        match_rates = [r[0] * 100 for r in rates]
        counts = [r[1] for r in rates]

        # Model's actual suffix accuracy
        bars = ax.bar(ks, match_rates, color='#3498db', alpha=0.8,
               label=f'Model Answers (n={counts[0]})', edgecolor='#2980b9', linewidth=1)

        # Add value labels on bars
        for bar, rate in zip(bars, match_rates):
            if rate > 0:
                ax.annotate(f'{rate:.1f}%',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=FONT_SMALL, fontweight='bold')

        # Random chance baseline: 10%, 1%, 0.1%, 0.01%
        baseline = [10, 1, 0.1, 0.01][:len(ks)]
        ax.plot(ks, baseline, 'r--', linewidth=2, marker='s',
                label='Random Chance', markersize=6)

        # Columnar prediction: high suffix preservation (errors propagate left-to-right)
        columnar = [85, 70, 50, 35][:len(ks)]
        ax.plot(ks, columnar, 'g:', linewidth=2, marker='^',
                label='If Columnar (OT)', markersize=6)

        ax.set_xlabel('Suffix Length (digits)', fontsize=FONT_LABEL)
        ax.set_title(f'{label} Modality', fontsize=FONT_TITLE, fontweight='bold')
        ax.set_xticks(ks)
        ax.set_xticklabels([f'Last {k}' for k in ks])
        ax.legend(loc='upper right', fontsize=FONT_SMALL)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Store metrics for LaTeX
        if rates:
            metrics[f'SuffixK1{label}'] = f'{rates[0][0]*100:.1f}\\%'
            if len(rates) > 1:
                metrics[f'SuffixK2{label}'] = f'{rates[1][0]*100:.1f}\\%'
            # Count of errors analyzed
            metrics[f'SuffixErrorCount{label}'] = str(counts[0])

    axes[0].set_ylabel('Suffix Match Rate (%)', fontsize=FONT_LABEL)

    fig.suptitle('Do LLMs Preserve Low-Order Digits in Wrong Answers?\n'
                 '(Columnar multiplication would show high suffix accuracy)',
                 fontsize=FONT_SUPTITLE, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved suffix accuracy plot to {output_path}")

    return metrics


def plot_suffix_accuracy_multi(
    model_series: Sequence[Dict[str, Any]],
    output_path: Path
) -> None:
    """Create multi-panel suffix accuracy plot (rows=models, cols=modality)."""
    valid_models = [
        m for m in model_series
        if m.get('hds_test_results') or m.get('hds_test_image_results')
    ]
    if not valid_models:
        tprint("Skipping suffix accuracy plot - missing data")
        return

    n_models = len(valid_models)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4.5 * n_models), sharey=True)
    axes = np.atleast_2d(axes)

    for row, model in enumerate(valid_models):
        for col, (results, label) in enumerate([
            (model.get('hds_test_results'), 'Text'),
            (model.get('hds_test_image_results'), 'Image')
        ]):
            ax = axes[row, col]

            if not results:
                _style_empty_panel(
                    ax,
                    f'{model["label"]} - {label}',
                    f'No {label} data',
                    FONT_BASE,
                    FONT_SMALL
                )
                continue

            suffix_data = analyze_suffix_accuracy(results)
            rates = suffix_data['all_errors']

            if not rates or rates[0][1] == 0:
                _style_empty_panel(
                    ax,
                    f'{model["label"]} - {label}',
                    'No errors to analyze',
                    FONT_BASE,
                    FONT_SMALL
                )
                continue

            ks = list(range(1, len(rates) + 1))
            match_rates = [r[0] * 100 for r in rates]
            counts = [r[1] for r in rates]

            bars = ax.bar(ks, match_rates, color='#3498db', alpha=0.8,
                          label=f'Model Answers (n={counts[0]})',
                          edgecolor='#2980b9', linewidth=1)

            for bar, rate in zip(bars, match_rates):
                if rate > 0:
                    ax.annotate(f'{rate:.1f}%',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=FONT_SMALL, fontweight='bold')

            baseline = [10, 1, 0.1, 0.01][:len(ks)]
            ax.plot(ks, baseline, 'r--', linewidth=2, marker='s',
                    label='Random Chance', markersize=5)

            columnar = [85, 70, 50, 35][:len(ks)]
            ax.plot(ks, columnar, 'g:', linewidth=2, marker='^',
                    label='If Columnar (OT)', markersize=5)

            ax.set_xlabel('Suffix Length (digits)', fontsize=FONT_BASE)
            ax.set_title(f'{model["label"]} - {label}', fontsize=FONT_BASE, fontweight='bold')
            ax.set_xticks(ks)
            ax.set_xticklabels([f'Last {k}' for k in ks])
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=FONT_TINY)

        axes[row, 0].set_ylabel('Suffix Match Rate (%)', fontsize=FONT_BASE)

    fig.suptitle('Do LLMs Preserve Low-Order Digits in Wrong Answers?\n'
                 '(Columnar multiplication would show high suffix accuracy)',
                 fontsize=FONT_SUPTITLE, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    tprint(f"Saved suffix accuracy plot to {output_path}")


def main():
    """Generate all figures and LaTeX macros."""
    # Parse command-line arguments
    args = parse_args()
    output_type = args.output_type
    probe_hds_dataset = args.probe_hds_dataset
    append_macros = args.append_macros
    requested_model = args.model
    output_path = args.output_path
    fragment_paths = [Path(path) for path in args.fragment_path]
    require_hds_all = output_type in ("all", "fingerprint-macros")
    require_fingerprint_quality = output_type in ("all", "fingerprint-figures", "fingerprint-macros")

    tprint("=" * 60)
    tprint("Generating Results Figures and LaTeX Macros")
    tprint("=" * 60)
    tprint(f"Output type: {output_type}")
    tprint(f"Probe HDS dataset: {probe_hds_dataset}")
    if requested_model:
        tprint(f"Model filter: {requested_model}")
    if append_macros:
        tprint("Append mode: enabled")
    if output_path is not None:
        tprint(f"Output path override: {output_path}")

    if output_type == "merge-macros":
        target_path = output_path or (OUTPUT_DIR / "results_macros.tex")
        tprint("\nMerging macro fragments...")
        merge_text_fragments(fragment_paths, target_path, dedupe_newcommands=True)
        return

    if output_type == "merge-nudge-appendix":
        target_path = output_path or (OUTPUT_DIR / "appendix_nudge_examples.tex")
        tprint("\nMerging nudge appendix fragments...")
        merge_nudge_appendix_fragments(fragment_paths, target_path)
        return

    if output_type == "merge-fingerprint-appendix":
        target_path = output_path or (OUTPUT_DIR / "appendix_template_variability.tex")
        tprint("\nMerging template-variability appendix fragments...")
        merge_template_variability_appendix_fragments(fragment_paths, target_path)
        return

    if output_type == "merge-gradient-appendix":
        target_path = output_path or (OUTPUT_DIR / "appendix_similarity_matrix.tex")
        tprint("\nMerging similarity appendix fragments...")
        merge_similarity_appendix_fragments(fragment_paths, target_path)
        return

    if output_type == "merge-embedding-appendix":
        target_path = output_path or (OUTPUT_DIR / "appendix_embedding_results.tex")
        tprint("\nMerging embedding appendix fragments...")
        merge_embedding_results_appendix_fragments(fragment_paths, target_path)
        return

    if output_type == "training-appendix":
        tprint("\nGenerating training appendix (no model analysis required)...")
        training_tex = generate_training_examples()
        training_path = output_path or (OUTPUT_DIR / "appendix_training_examples.tex")
        write_text_artifact(training_tex, training_path)
        tprint(f"Saved training examples to {training_path}")
        return

    existing_global_macros = False
    macro_path = output_path or (OUTPUT_DIR / "results_macros.tex")
    if append_macros and macro_path.exists():
        try:
            with open(macro_path, 'r') as f:
                existing_global_macros = "\\DatasetSplitRatios" in f.read()
        except OSError:
            existing_global_macros = False

    # Discover available model variants
    model_variants = discover_model_variants()
    if model_variants:
        tprint(f"\nDiscovered {len(model_variants)} model variant(s): {model_variants}")
    else:
        tprint("\nNo model variants found, using legacy paths")
        model_variants = [None]  # None = use legacy paths without model suffix

    # Filter to requested model if specified
    if requested_model:
        # Match against model slug (e.g., "Qwen3-VL-30B-A3B" or partial "30B")
        matching = [m for m in model_variants if m and requested_model in m]
        if matching:
            model_variants = matching
            tprint(f"Filtered to model(s): {model_variants}")
        else:
            # Try exact match with slug construction
            model_variants = [requested_model]
            tprint(f"Using specified model: {requested_model}")

    # Accumulate macros from all models
    all_macros = []
    model_plot_data: Dict[str, Dict[str, Any]] = {}
    global_macros_emitted = existing_global_macros

    # First model will also be used for figures (or the only model if legacy)
    primary_hds_test = None
    primary_hds_test_image = None
    primary_traps = None
    primary_traps_image = None
    primary_hds_test_results = None
    primary_hds_test_image_results = None
    primary_model_slug = None  # Track primary model for appendix generators

    for model_slug in model_variants:
        model_label = model_slug or "legacy"
        tprint(f"\n{'='*40}")
        tprint(f"Processing model: {model_label}")
        tprint(f"{'='*40}")

        # Load text modality analyses
        tprint("\nLoading text modality results...")
        hds_test = load_analysis(
            probe_hds_dataset,
            'test',
            modality='text',
            model_slug=model_slug,
            allow_alias_fallback=False,
        )
        template_summary_hds_test = hds_test
        hds_test_style_mismatch = load_analysis(
            probe_hds_dataset,
            'test',
            modality='text',
            model_slug=model_slug,
            output_tag='style_mismatch',
            allow_alias_fallback=False,
        )
        contrastive_hds_test = load_contrastive_analysis('hds', 'test', modality='text', model_slug=model_slug)
        hds_all = None
        if require_hds_all:
            hds_all = load_analysis(
                probe_hds_dataset,
                modality='text',
                model_slug=model_slug,
                allow_alias_fallback=False,
            )
        traps = load_analysis('traps', modality='text', model_slug=model_slug)
        lora_training = load_lora_training(model_slug=model_slug)
        nudge_test = load_nudge_test('test', modality='text', model_slug=model_slug)
        gradient_analysis = load_gradient_analysis(model_slug=model_slug)

        # Load results CSVs for bootstrap CI computation
        hds_test_results = load_results_csv(
            probe_hds_dataset,
            'test',
            modality='text',
            model_slug=model_slug,
            allow_alias_fallback=False,
        )
        hds_test_details = load_details_jsonl(
            probe_hds_dataset,
            'test',
            modality='text',
            model_slug=model_slug,
            allow_alias_fallback=False,
        )
        hds_test_table3_stats = compute_table3_stats_from_jsonl(hds_test_details) if hds_test_details else None
        hds_test_style_mismatch_details = load_details_jsonl(
            probe_hds_dataset,
            'test',
            modality='text',
            model_slug=model_slug,
            output_tag='style_mismatch',
            allow_alias_fallback=False,
        )
        hds_test_style_mismatch_table3_stats = (
            compute_table3_stats_from_jsonl(hds_test_style_mismatch_details)
            if hds_test_style_mismatch_details else None
        )
        traps_results = load_results_csv('traps', modality='text', model_slug=model_slug)
        traps_details = load_details_jsonl('traps', modality='text', model_slug=model_slug)
        traps_table3_stats = compute_table3_stats_from_jsonl(traps_details) if traps_details else None
        nudge_test_results = load_nudge_results_csv('test', modality='text', model_slug=model_slug)
        nudge_test_details = load_nudge_details('test', modality='text', model_slug=model_slug)

        # Load image modality analyses
        tprint("Loading image modality results...")
        hds_test_image = load_analysis(
            probe_hds_dataset,
            'test',
            modality='image',
            model_slug=model_slug,
            allow_alias_fallback=False,
        )
        template_summary_hds_test_image = hds_test_image
        hds_test_image_style_mismatch = load_analysis(
            probe_hds_dataset,
            'test',
            modality='image',
            model_slug=model_slug,
            output_tag='style_mismatch',
            allow_alias_fallback=False,
        )
        contrastive_hds_test_image = load_contrastive_analysis('hds', 'test', modality='image', model_slug=model_slug)
        traps_image = load_analysis('traps', modality='image', model_slug=model_slug)
        hds_test_image_results = load_results_csv(
            probe_hds_dataset,
            'test',
            modality='image',
            model_slug=model_slug,
            allow_alias_fallback=False,
        )
        hds_test_image_details = load_details_jsonl(
            probe_hds_dataset,
            'test',
            modality='image',
            model_slug=model_slug,
            allow_alias_fallback=False,
        )
        hds_test_image_table3_stats = (
            compute_table3_stats_from_jsonl(hds_test_image_details)
            if hds_test_image_details else None
        )
        hds_test_image_style_mismatch_details = load_details_jsonl(
            probe_hds_dataset,
            'test',
            modality='image',
            model_slug=model_slug,
            output_tag='style_mismatch',
            allow_alias_fallback=False,
        )
        hds_test_image_style_mismatch_table3_stats = (
            compute_table3_stats_from_jsonl(hds_test_image_style_mismatch_details)
            if hds_test_image_style_mismatch_details else None
        )
        traps_image_results = load_results_csv('traps', modality='image', model_slug=model_slug)
        traps_image_details = load_details_jsonl('traps', modality='image', model_slug=model_slug)
        traps_image_table3_stats = (
            compute_table3_stats_from_jsonl(traps_image_details)
            if traps_image_details else None
        )
        nudge_test_image = load_nudge_test('test', modality='image', model_slug=model_slug)

        if hds_test and hds_test_table3_stats:
            hds_test["table3_alt_stats"] = hds_test_table3_stats
            hds_test["soft_target_stats"] = hds_test_table3_stats.get("soft_target_stats", {})
            hds_test["embedding_soft_target_stats"] = hds_test_table3_stats.get("embedding_soft_target_stats", {})
        if hds_test_style_mismatch and hds_test_style_mismatch_table3_stats:
            hds_test_style_mismatch["table3_alt_stats"] = hds_test_style_mismatch_table3_stats
            hds_test_style_mismatch["soft_target_stats"] = hds_test_style_mismatch_table3_stats.get("soft_target_stats", {})
            hds_test_style_mismatch["embedding_soft_target_stats"] = hds_test_style_mismatch_table3_stats.get("embedding_soft_target_stats", {})
        if hds_test_image and hds_test_image_table3_stats:
            hds_test_image["table3_alt_stats"] = hds_test_image_table3_stats
            hds_test_image["soft_target_stats"] = hds_test_image_table3_stats.get("soft_target_stats", {})
            hds_test_image["embedding_soft_target_stats"] = hds_test_image_table3_stats.get("embedding_soft_target_stats", {})
        if hds_test_image_style_mismatch and hds_test_image_style_mismatch_table3_stats:
            hds_test_image_style_mismatch["table3_alt_stats"] = hds_test_image_style_mismatch_table3_stats
            hds_test_image_style_mismatch["soft_target_stats"] = hds_test_image_style_mismatch_table3_stats.get("soft_target_stats", {})
            hds_test_image_style_mismatch["embedding_soft_target_stats"] = hds_test_image_style_mismatch_table3_stats.get("embedding_soft_target_stats", {})
        if traps and traps_table3_stats:
            traps["table3_alt_stats"] = traps_table3_stats
            traps["soft_target_stats"] = traps_table3_stats.get("soft_target_stats", {})
            traps["embedding_soft_target_stats"] = traps_table3_stats.get("embedding_soft_target_stats", {})
        if traps_image and traps_image_table3_stats:
            traps_image["table3_alt_stats"] = traps_image_table3_stats
            traps_image["soft_target_stats"] = traps_image_table3_stats.get("soft_target_stats", {})
            traps_image["embedding_soft_target_stats"] = traps_image_table3_stats.get("embedding_soft_target_stats", {})

        if require_fingerprint_quality:
            ensure_image_analysis_quality(
                hds_test_image,
                dataset=probe_hds_dataset,
                model_slug=model_slug,
            )
            ensure_image_analysis_quality(
                hds_test_image_style_mismatch,
                dataset=probe_hds_dataset,
                model_slug=model_slug,
                output_tag='style_mismatch',
            )
            if output_type in ("all", "fingerprint-figures", "fingerprint-macros"):
                ensure_image_analysis_quality(
                    traps_image,
                    dataset='Traps',
                    model_slug=model_slug,
                )
        nudge_test_image_results = load_nudge_results_csv('test', modality='image', model_slug=model_slug)

        # Store first model's data for figures
        if primary_hds_test is None:
            primary_hds_test = hds_test
            primary_hds_test_image = hds_test_image
            primary_traps = traps
            primary_traps_image = traps_image
            primary_hds_test_results = hds_test_results
            primary_hds_test_image_results = hds_test_image_results
            primary_model_slug = model_slug  # Track for appendix generators

        model_key = model_slug or "legacy"
        model_plot_data[model_key] = {
            'slug': model_slug,
            'label': get_model_plot_label(model_slug),
            'sort_key': get_model_sort_key(model_slug),
            'hds_test': hds_test,
            'traps': traps,
            'hds_test_image': hds_test_image,
            'traps_image': traps_image,
            'hds_test_results': hds_test_results,
            'hds_test_image_results': hds_test_image_results
        }

        # Report status for this model
        tprint("\n--- Data availability ---")
        if not hds_test:
            tprint("  [MISSING] HDS test analysis (text)")
        else:
            tprint("  [OK] HDS test analysis (text)")

        if not contrastive_hds_test:
            tprint("  [MISSING] Contrastive step analysis (text)")
        else:
            tprint("  [OK] Contrastive step analysis (text)")

        if not hds_test_image:
            tprint("  [MISSING] HDS test analysis (image)")
        else:
            tprint("  [OK] HDS test analysis (image)")

        if not contrastive_hds_test_image:
            tprint("  [MISSING] Contrastive step analysis (image)")
        else:
            tprint("  [OK] Contrastive step analysis (image)")

        if not traps:
            tprint("  [MISSING] Traps analysis (text)")
        else:
            tprint("  [OK] Traps analysis (text)")

        if not traps_image:
            tprint("  [MISSING] Traps analysis (image)")
        else:
            tprint("  [OK] Traps analysis (image)")

        if not lora_training:
            tprint("  [MISSING] LoRA training summary")
        else:
            tprint("  [OK] LoRA training summary")

        if not nudge_test:
            tprint("  [MISSING] Nudge test analysis (text)")
        else:
            tprint("  [OK] Nudge test analysis (text)")

        if not nudge_test_image:
            tprint("  [MISSING] Nudge test analysis (image)")
        else:
            tprint("  [OK] Nudge test analysis (image)")

        if not gradient_analysis:
            tprint("  [MISSING] Gradient analysis")
        else:
            tprint("  [OK] Gradient analysis")

        if not hds_test_results:
            tprint("  [MISSING] HDS test results CSV (text) - no bootstrap CIs")
        else:
            tprint("  [OK] HDS test results CSV (text)")

        if not hds_test_image_results:
            tprint("  [MISSING] HDS test results CSV (image) - no bootstrap CIs")
        else:
            tprint("  [OK] HDS test results CSV (image)")

        # Generate LaTeX macros for this model based on output_type
        should_generate_macros = output_type == 'all' or output_type.endswith('-macros')

        if should_generate_macros:
            # Determine which macro types to generate based on output_type
            generate_fingerprint = output_type in ('all', 'fingerprint-macros')
            generate_nudge = output_type in ('all', 'nudge-macros')
            generate_gradient = output_type in ('all', 'gradient-macros')

            if generate_fingerprint or generate_nudge or generate_gradient:
                tprint(f"\nGenerating LaTeX macros for {model_label}...")

                # Emit shared globals once, then limit each append step to its macro family.
                include_global_macros = False
                if not global_macros_emitted:
                    include_global_macros = True
                    global_macros_emitted = True

                effective_hds_test = hds_test if generate_fingerprint else None
                effective_hds_all = hds_all if generate_fingerprint else None
                effective_traps = traps if generate_fingerprint else None
                effective_hds_test_results = hds_test_results if generate_fingerprint else None
                effective_hds_test_details = hds_test_details if generate_fingerprint else None
                effective_hds_test_image = hds_test_image if generate_fingerprint else None
                effective_hds_test_image_results = (
                    hds_test_image_results if generate_fingerprint else None
                )
                effective_hds_test_image_details = (
                    hds_test_image_details if generate_fingerprint else None
                )
                effective_hds_test_style_mismatch = (
                    hds_test_style_mismatch if generate_fingerprint else None
                )
                effective_hds_test_image_style_mismatch = (
                    hds_test_image_style_mismatch if generate_fingerprint else None
                )
                effective_template_summary_hds_test = (
                    template_summary_hds_test if generate_fingerprint else None
                )
                effective_template_summary_hds_test_image = (
                    template_summary_hds_test_image if generate_fingerprint else None
                )
                effective_contrastive_hds_test = (
                    contrastive_hds_test if generate_fingerprint else None
                )
                effective_contrastive_hds_test_image = (
                    contrastive_hds_test_image if generate_fingerprint else None
                )
                effective_traps_results = traps_results if generate_fingerprint else None
                effective_traps_details = traps_details if generate_fingerprint else None
                effective_traps_image = traps_image if generate_fingerprint else None
                effective_traps_image_results = (
                    traps_image_results if generate_fingerprint else None
                )
                effective_traps_image_details = (
                    traps_image_details if generate_fingerprint else None
                )

                effective_lora_training = lora_training if generate_nudge else None
                effective_nudge_test = nudge_test if generate_nudge else None
                effective_nudge_test_results = nudge_test_results if generate_nudge else None
                effective_nudge_test_details = nudge_test_details if generate_nudge else None
                effective_nudge_test_image = nudge_test_image if generate_nudge else None
                effective_nudge_test_image_results = (
                    nudge_test_image_results if generate_nudge else None
                )

                effective_gradient_analysis = gradient_analysis if generate_gradient else None

                macros = generate_latex_macros(
                    effective_hds_test,
                    effective_hds_all,
                    effective_traps,
                    effective_lora_training,
                    effective_nudge_test,
                    effective_gradient_analysis,
                    effective_hds_test_results,
                    hds_test_details=effective_hds_test_details,
                    hds_test_image=effective_hds_test_image,
                    hds_test_image_results=effective_hds_test_image_results,
                    hds_test_image_details=effective_hds_test_image_details,
                    hds_test_style_mismatch=effective_hds_test_style_mismatch,
                    hds_test_image_style_mismatch=effective_hds_test_image_style_mismatch,
                    template_summary_hds_test=effective_template_summary_hds_test,
                    template_summary_hds_test_image=effective_template_summary_hds_test_image,
                    contrastive_hds_test=effective_contrastive_hds_test,
                    contrastive_hds_test_image=effective_contrastive_hds_test_image,
                    traps_results=effective_traps_results,
                    traps_details=effective_traps_details,
                    traps_image=effective_traps_image,
                    traps_image_results=effective_traps_image_results,
                    traps_image_details=effective_traps_image_details,
                    nudge_test_results=effective_nudge_test_results,
                    nudge_test_details=effective_nudge_test_details,
                    nudge_test_image=effective_nudge_test_image,
                    nudge_test_image_results=effective_nudge_test_image_results,
                    probe_hds_dataset_name=probe_hds_dataset,
                    model_slug=model_slug,
                    include_global_macros=include_global_macros,
                    emit_contrastive_placeholders=generate_fingerprint,
                    emit_gradient_placeholders=generate_gradient,
                    emit_suffix_macros=generate_gradient
                )
                all_macros.append(macros)

    # End of model loop - write macros file if we generated any
    if all_macros:
        tprint("\n" + "=" * 60)
        tprint("Writing macros file")
        tprint("=" * 60)
        macro_content = "\n\n".join(all_macros)
        write_macros_with_lock(macro_content, append=append_macros, macro_path=macro_path)

    # Generate appendix example files (using primary model's data)
    # Only generate if output_type requests it
    should_generate_appendix = output_type == 'all' or output_type.endswith('-appendix')

    if should_generate_appendix:
        tprint("\nGenerating appendix example files...")
        if primary_model_slug:
            tprint(f"Using primary model: {primary_model_slug}")

        # 1. Fingerprint examples (part of fingerprint output, but stored separately)
        if output_type in ('all',):
            fingerprint_tex = generate_fingerprint_examples(model_slug=primary_model_slug)
            fingerprint_path = OUTPUT_DIR / "appendix_fingerprint_examples.tex"
            with open(fingerprint_path, 'w') as f:
                f.write(fingerprint_tex)
            tprint(f"Saved fingerprint examples to {fingerprint_path}")

        # 2. Training examples (uses local training data, no model_slug needed)
        if output_type in ('all', 'training-appendix'):
            training_tex = generate_training_examples()
            training_path = output_path or (OUTPUT_DIR / "appendix_training_examples.tex")
            write_text_artifact(training_tex, training_path)
            tprint(f"Saved training examples to {training_path}")

        # 3. Nudge test examples
        if output_type in ('all', 'nudge-appendix'):
            nudge_tex = generate_nudge_examples(
                model_slug=primary_model_slug,
                include_labels=should_emit_appendix_labels(output_path, "appendix_nudge_examples.tex"),
            )
            nudge_path = output_path or (OUTPUT_DIR / "appendix_nudge_examples.tex")
            write_text_artifact(nudge_tex, nudge_path)
            tprint(f"Saved nudge examples to {nudge_path}")

        # 4. Template-variability appendix
        if output_type in ('all', 'fingerprint-appendix'):
            template_tex = generate_template_variability_appendix(
                model_slug=primary_model_slug,
                include_labels=should_emit_appendix_labels(output_path, "appendix_template_variability.tex"),
                probe_hds_dataset=probe_hds_dataset,
            )
            template_path = output_path or (OUTPUT_DIR / "appendix_template_variability.tex")
            write_text_artifact(template_tex, template_path)
            tprint(f"Saved template-variability appendix to {template_path}")

        # 5. Similarity matrix
        if output_type in ('all', 'gradient-appendix'):
            similarity_tex = generate_similarity_matrix(
                model_slug=primary_model_slug,
                include_labels=should_emit_appendix_labels(output_path, "appendix_similarity_matrix.tex"),
            )
            similarity_path = output_path or (OUTPUT_DIR / "appendix_similarity_matrix.tex")
            write_text_artifact(similarity_tex, similarity_path)
            tprint(f"Saved similarity matrix to {similarity_path}")

        # 6. Embedding-based fingerprint appendix
        if output_type in ('all', 'embedding-appendix'):
            embedding_tex = generate_embedding_results_appendix(
                model_slug=primary_model_slug,
                include_labels=should_emit_appendix_labels(output_path, "appendix_embedding_results.tex"),
            )
            embedding_path = output_path or (OUTPUT_DIR / "appendix_embedding_results.tex")
            write_text_artifact(embedding_tex, embedding_path)
            tprint(f"Saved embedding appendix to {embedding_path}")

    # Generate figures
    # Only generate if output_type requests fingerprint-figures or all
    should_generate_figures = output_type in ('all', 'fingerprint-figures')
    plot_models = sorted(model_plot_data.values(), key=lambda m: m.get('sort_key', 0))

    if should_generate_figures and plot_models:
        tprint("\nGenerating figures...")

        if len(plot_models) > 1:
            plot_perplexity_comparison_multi(
                plot_models,
                OUTPUT_DIR / "perplexity_comparison.png"
            )
        elif primary_hds_test and primary_traps:
            plot_perplexity_comparison(
                primary_hds_test, primary_traps,
                OUTPUT_DIR / "perplexity_comparison.png"
            )

        if primary_hds_test and primary_traps:
            plot_detection_rates(
                primary_hds_test, primary_traps,
                OUTPUT_DIR / "detection_rates.png"
            )

            plot_confusion_matrix(
                primary_hds_test,
                "HDS (Test Split) Lowest-loss Heuristic Confusion",
                OUTPUT_DIR / "confusion_hds_test.png"
            )

            plot_confusion_matrix(
                primary_traps,
                "Traps Lowest-loss Heuristic Confusion",
                OUTPUT_DIR / "confusion_traps.png"
            )

    # Generate multimodal comparison figures (if image data available)
    if should_generate_figures and primary_hds_test_image:
        tprint("\nGenerating multimodal comparison figures...")

        plot_multimodal_perplexity_comparison(
            primary_hds_test, primary_hds_test_image,
            OUTPUT_DIR / "perplexity_comparison_multimodal.png"
        )

        plot_multimodal_detection_comparison(
            primary_hds_test, primary_hds_test_image,
            OUTPUT_DIR / "detection_rates_multimodal.png"
        )

        plot_confusion_matrix(
            primary_hds_test_image,
            "HDS (Test Split) Lowest-loss Heuristic Confusion - Image Modality",
            OUTPUT_DIR / "confusion_hds_test_image.png"
        )

        if primary_traps_image:
            plot_confusion_matrix(
                primary_traps_image,
                "Traps Lowest-loss Heuristic Confusion - Image Modality",
                OUTPUT_DIR / "confusion_traps_image.png"
            )

        if len(plot_models) > 1:
            plot_heuristic_reversal_multi(
                plot_models,
                OUTPUT_DIR / "heuristic_reversal.png"
            )

            plot_suffix_accuracy_multi(
                plot_models,
                OUTPUT_DIR / "suffix_accuracy.png"
            )

            plot_suffix_by_complexity_multi(
                plot_models,
                OUTPUT_DIR / "suffix_by_complexity.png"
            )

            plot_modulo_invariants_multi(
                plot_models,
                OUTPUT_DIR / "modulo_invariants.png"
            )
        else:
            plot_heuristic_reversal(
                primary_hds_test, primary_hds_test_image,
                OUTPUT_DIR / "heuristic_reversal.png"
            )

            plot_suffix_accuracy(
                primary_hds_test_results,
                primary_hds_test_image_results,
                OUTPUT_DIR / "suffix_accuracy.png"
            )

            if primary_hds_test_results:
                plot_suffix_by_complexity(
                    primary_hds_test_results,
                    OUTPUT_DIR / "suffix_by_complexity.png"
                )

            plot_modulo_invariants(
                primary_hds_test_results,
                primary_hds_test_image_results,
                OUTPUT_DIR / "modulo_invariants.png"
            )

    tprint("\n" + "=" * 60)
    tprint("Done!")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
