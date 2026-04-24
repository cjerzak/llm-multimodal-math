#!/usr/bin/env python3
"""
GradientOrthogonality.py

Compute cosine similarities between LoRA adapters using two metrics:
1) Flattened LoRA factor weights (A/B) in parameter space.
2) Effective update matrices (Delta W = B @ A), which are gauge-invariant.

This script:
1. Loads saved adapter weights from TrainHeuristicLoRAs.py
2. Computes cosine similarity for flattened LoRA factor weights (A/B)
3. Computes cosine similarity for effective update matrices Delta W = B @ A

This replaces the previous loss-correlation approach with actual
parameter-space analysis, and adds a gauge-invariant effective-update
metric for orthogonality claims.

Usage:
    python Scripts/GradientOrthogonality.py
    python Scripts/GradientOrthogonality.py --weights-dir /path/to/weights
"""

import os
import sys
import json
import csv
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence, Iterable
from itertools import combinations
import numpy as np

# Paths (experiments/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "SavedResults"

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))
from core.Logging import tprint
from core.TinkerStartup import create_tinker_service_client

# Default model (for backward compatibility)
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
PRIMARY_HEURISTICS = ("DD", "OT", "RC")


def get_model_paths(model_name: Optional[str] = None) -> Tuple[Path, Path, Path]:
    """Get model-specific paths for weights, output, and canonical training summary.

    Args:
        model_name: Full model name (e.g., "Qwen/Qwen3-VL-30B-A3B-Instruct").
                    If None, uses legacy paths without model suffix.

    Returns:
        Tuple of (weights_dir, output_dir, training_summary_path)
    """
    if model_name:
        model_slug = model_name.split("/")[-1].replace("-Instruct", "")
        weights_dir = RESULTS_DIR / f"gradient_analysis_{model_slug}" / "adapter_weights"
        output_dir = RESULTS_DIR / f"gradient_analysis_{model_slug}"
        training_summary_path = RESULTS_DIR / f"lora_training_{model_slug}" / "training_summary.json"
    else:
        # Legacy paths
        weights_dir = RESULTS_DIR / "gradient_analysis" / "adapter_weights"
        output_dir = RESULTS_DIR / "gradient_analysis"
        training_summary_path = RESULTS_DIR / "lora_training" / "training_summary.json"

    return weights_dir, output_dir, training_summary_path


def get_training_summary_paths(model_name: Optional[str] = None) -> List[Path]:
    """Return canonical plus seed-scoped training summaries for a model."""
    if model_name:
        model_slug = model_name.split("/")[-1].replace("-Instruct", "")
        paths = [RESULTS_DIR / f"lora_training_{model_slug}" / "training_summary.json"]
        paths.extend(sorted(RESULTS_DIR.glob(f"lora_training_seed*_{model_slug}/training_summary.json")))
        return paths
    return [RESULTS_DIR / "lora_training" / "training_summary.json"]


# Legacy paths for backward compatibility (used if --model not specified)
WEIGHTS_DIR = RESULTS_DIR / "gradient_analysis" / "adapter_weights"
OUTPUT_DIR = RESULTS_DIR / "gradient_analysis"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "lora_training" / "training_summary.json"


def _seed_suffix_from_training_summary(training_summary_path: Path, summary: Dict[str, Any]) -> str:
    """Infer the seed suffix used for adapter filenames from a training summary."""
    output_dir = summary.get("output_dir") or str(training_summary_path.parent)
    match = re.search(r"_seed(\d+)_", output_dir)
    return f"_seed{match.group(1)}" if match else ""


def download_weights_from_checkpoints(weights_dir: Path, training_summary_paths: Sequence[Path]) -> int:
    """Download adapter weights from one or more Tinker training summaries.

    Args:
        weights_dir: Directory to save downloaded weights
        training_summary_paths: Paths to training summary JSON files with checkpoint paths

    Returns:
        Number of successfully downloaded adapters
    """
    import tarfile
    import tempfile
    import urllib.request

    success_count = 0
    seen_outputs = set()
    for training_summary_path in training_summary_paths:
        if not training_summary_path.exists():
            continue

        with open(training_summary_path) as f:
            summary = json.load(f)

        seed_suffix = _seed_suffix_from_training_summary(training_summary_path, summary)

        for heuristic, data in summary.get('heuristics', {}).items():
            checkpoint_path = data.get('adapter_path')
            if not checkpoint_path:
                tprint(f"  {heuristic}{seed_suffix}: No checkpoint path in training summary")
                continue

            output_path = weights_dir / f"{heuristic.lower()}{seed_suffix}_adapter_weights.npz"
            if output_path in seen_outputs:
                continue
            seen_outputs.add(output_path)

            if output_path.exists():
                tprint(f"  {heuristic}{seed_suffix}: Weights already exist at {output_path}")
                success_count += 1
                continue

            tprint(f"  {heuristic}{seed_suffix}: Downloading from {checkpoint_path}...")
            try:
                import tinker
                service_client = create_tinker_service_client(
                    tinker_module=tinker,
                    api_key=os.getenv("TINKER_API_KEY"),
                )
                rest_client = service_client.create_rest_client()

                # Get signed download URL
                resp = rest_client.get_checkpoint_archive_url_from_tinker_path(checkpoint_path).result()

                with tempfile.TemporaryDirectory() as tmpdir:
                    # Download tar archive
                    tar_path = Path(tmpdir) / "checkpoint.tar"
                    urllib.request.urlretrieve(resp.url, tar_path)

                    # Extract
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(tmpdir)

                    # Look for adapter weights
                    weights_file = None
                    for pattern in ['adapter_model.safetensors', 'adapter_model.bin', '*.safetensors']:
                        matches = list(Path(tmpdir).rglob(pattern))
                        if matches:
                            weights_file = matches[0]
                            break

                    if weights_file is None:
                        tprint(f"    Warning: No adapter weights found in checkpoint")
                        continue

                    # Load and convert to numpy
                    if weights_file.suffix == '.safetensors':
                        from safetensors import safe_open
                        weights = {}
                        with safe_open(weights_file, framework="numpy") as f:
                            for key in f.keys():
                                weights[key] = f.get_tensor(key)
                    else:
                        import torch
                        state_dict = torch.load(weights_file, map_location='cpu')
                        weights = {k: v.numpy() for k, v in state_dict.items()}

                    # Save as npz
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(output_path, **weights)
                    total_params = sum(w.size for w in weights.values())
                    tprint(f"    Saved {len(weights)} tensors ({total_params:,} params)")
                    success_count += 1

            except Exception as e:
                tprint(f"    Failed to download: {e}")
                continue

    return success_count


def _format_keys(keys: Sequence[str]) -> str:
    if not keys:
        return "none"
    preview = ", ".join(keys[:5])
    suffix = "" if len(keys) <= 5 else f", ... (+{len(keys) - 5} more)"
    return f"[{preview}{suffix}]"


def _split_lora_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a LoRA weight key into (base, kind) if it matches A/B suffixes."""
    suffixes = [
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
        (".lora_A", "A"),
        (".lora_B", "B"),
    ]
    for suffix, kind in suffixes:
        if key.endswith(suffix):
            return key[:-len(suffix)], kind
    return None, None


def _as_expert_dim(tensor: np.ndarray) -> np.ndarray:
    """Ensure tensor has an expert dimension (E, ...)."""
    if tensor.ndim == 2:
        return tensor[None, ...]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"Unsupported LoRA tensor rank: {tensor.ndim}")


def _validate_lora_shapes(base: str, a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> None:
    """Validate LoRA A/B shapes are consistent for effective update computation."""
    if len(a_shape) not in (2, 3) or len(b_shape) not in (2, 3):
        raise ValueError(f"{base}: unexpected LoRA tensor ranks A={a_shape}, B={b_shape}")
    a_r = a_shape[-2]
    b_r = b_shape[-1]
    if a_r != b_r:
        raise ValueError(f"{base}: rank mismatch A={a_shape} vs B={b_shape}")


def _effective_update_inner(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> float:
    """Compute inner product <B1 A1, B2 A2> without materializing full matrices."""
    a1e = _as_expert_dim(a1)
    b1e = _as_expert_dim(b1)
    a2e = _as_expert_dim(a2)
    b2e = _as_expert_dim(b2)

    # Validate expert dims where both are >1
    if b1e.shape[0] != b2e.shape[0] and b1e.shape[0] != 1 and b2e.shape[0] != 1:
        raise ValueError(f"Expert dim mismatch for B: {b1e.shape} vs {b2e.shape}")
    if a1e.shape[0] != a2e.shape[0] and a1e.shape[0] != 1 and a2e.shape[0] != 1:
        raise ValueError(f"Expert dim mismatch for A: {a1e.shape} vs {a2e.shape}")

    # X_e = B1_e^T B2_e, shape (E, r, r)
    x = np.einsum("eor,eos->ers", b1e, b2e)
    # Y_e = A2_e A1_e^T, shape (E, r, r)
    y = np.einsum("eri,esi->ers", a2e, a1e)

    # trace(X_e Y_e) = sum_{i,j} X_e[i,j] * Y_e[j,i]
    return float(np.sum(x * np.swapaxes(y, 1, 2), dtype=np.float64))


def _load_adapter_files(weights_dir: Path) -> Dict[str, np.lib.npyio.NpzFile]:
    """Load adapter weight files with mmap for streaming access."""
    adapter_files: Dict[str, np.lib.npyio.NpzFile] = {}
    for npz_path in sorted(weights_dir.glob("*_adapter_weights.npz")):
        name = npz_path.stem.replace("_adapter_weights", "").upper()
        adapter_files[name] = np.load(npz_path, mmap_mode="r")
    return adapter_files


def _validate_weight_alignment(adapter_files: Dict[str, np.lib.npyio.NpzFile]) -> List[str]:
    """Ensure all adapter files have identical keys and shapes."""
    if not adapter_files:
        return []
    ref_name = next(iter(adapter_files))
    ref = adapter_files[ref_name]
    expected_keys = sorted(ref.files)
    expected_shapes = {k: tuple(ref[k].shape) for k in expected_keys}

    for name, data in adapter_files.items():
        keys = sorted(data.files)
        shapes = {k: tuple(data[k].shape) for k in keys}
        missing = [k for k in expected_keys if k not in shapes]
        extra = [k for k in keys if k not in expected_shapes]
        shape_mismatch = [
            k for k in expected_keys
            if k in shapes and shapes[k] != expected_shapes.get(k)
        ]

        if missing or extra or shape_mismatch:
            raise ValueError(
                "Adapter weight tensors are misaligned. "
                f"{name} vs {ref_name}: "
                f"missing={_format_keys(missing)} "
                f"extra={_format_keys(extra)} "
                f"shape_mismatch={_format_keys(shape_mismatch)}"
            )

    return expected_keys


def _build_lora_key_index(keys: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Build base->(A,B) key mapping for LoRA tensors."""
    index: Dict[str, Dict[str, str]] = {}
    for key in keys:
        base, kind = _split_lora_key(key)
        if base is None or kind is None:
            continue
        index.setdefault(base, {})[kind] = key

    missing = [base for base, parts in index.items() if "A" not in parts or "B" not in parts]
    if missing:
        raise ValueError(f"Missing A/B LoRA pairs for: {_format_keys(sorted(missing))}")
    return index


def _validate_lora_alignment(
    adapter_files: Dict[str, np.lib.npyio.NpzFile]
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """Validate LoRA A/B keys align across adapters and return index maps."""
    if not adapter_files:
        return [], {}, {}

    ref_name = next(iter(adapter_files))
    ref_index = _build_lora_key_index(adapter_files[ref_name].files)
    ref_shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for base, keys in ref_index.items():
        a_shape = tuple(adapter_files[ref_name][keys["A"]].shape)
        b_shape = tuple(adapter_files[ref_name][keys["B"]].shape)
        _validate_lora_shapes(base, a_shape, b_shape)
        ref_shapes[base] = (a_shape, b_shape)

    for name, data in adapter_files.items():
        index = _build_lora_key_index(data.files)
        missing = [base for base in ref_index if base not in index]
        extra = [base for base in index if base not in ref_index]
        if missing or extra:
            raise ValueError(
                "LoRA key sets are misaligned. "
                f"{name} vs {ref_name}: missing={_format_keys(missing)} extra={_format_keys(extra)}"
            )
        for base, keys in ref_index.items():
            a_shape = tuple(data[index[base]["A"]].shape)
            b_shape = tuple(data[index[base]["B"]].shape)
            _validate_lora_shapes(base, a_shape, b_shape)
            if (a_shape, b_shape) != ref_shapes[base]:
                raise ValueError(
                    f"{base}: shape mismatch {name} vs {ref_name}: "
                    f"{(a_shape, b_shape)} vs {ref_shapes[base]}"
                )

    return sorted(ref_index.keys()), {name: _build_lora_key_index(data.files) for name, data in adapter_files.items()}, ref_shapes

def load_adapter_weights(weights_dir: Path) -> Dict[str, np.ndarray]:
    """Load all adapter weight files and flatten into vectors.

    Args:
        weights_dir: Directory containing *_adapter_weights.npz files

    Returns:
        Dict mapping heuristic name (with optional seed suffix) to flattened weight vector.
        Names like "RC" for standard adapters, "RC_SEED42" for seed-controlled adapters.
    """
    weight_vectors = {}
    expected_keys: Optional[List[str]] = None
    expected_shapes: Dict[str, Tuple[int, ...]] = {}
    reference_name: Optional[str] = None

    for npz_path in weights_dir.glob("*_adapter_weights.npz"):
        # Extract heuristic name from filename
        # Standard: "rc_adapter_weights.npz" -> "RC"
        # Seed-controlled: "rc_seed42_adapter_weights.npz" -> "RC_SEED42"
        name = npz_path.stem.replace("_adapter_weights", "").upper()

        tprint(f"  Loading {npz_path.name}...")
        data = np.load(npz_path)

        keys = sorted(data.files)
        shapes = {k: tuple(data[k].shape) for k in keys}

        if expected_keys is None:
            expected_keys = keys
            expected_shapes = shapes
            reference_name = npz_path.name
        else:
            missing = [k for k in expected_keys if k not in shapes]
            extra = [k for k in keys if k not in expected_shapes]
            shape_mismatch = [
                k for k in expected_keys
                if k in shapes and shapes[k] != expected_shapes.get(k)
            ]

            if missing or extra or shape_mismatch:
                raise ValueError(
                    "Adapter weight tensors are misaligned. "
                    f"{npz_path.name} vs {reference_name}: "
                    f"missing={_format_keys(missing)} "
                    f"extra={_format_keys(extra)} "
                    f"shape_mismatch={_format_keys(shape_mismatch)}"
                )

        # Flatten all weight tensors into a single vector
        all_weights = []
        for key in expected_keys or []:
            tensor = data[key]
            all_weights.append(tensor.flatten())

        if all_weights:
            flat_vector = np.concatenate(all_weights)
            weight_vectors[name] = flat_vector
            tprint(f"    {name}: {len(data.files)} tensors, {flat_vector.shape[0]:,} parameters")
        else:
            tprint(f"    Warning: No weights found in {npz_path.name}")

    return weight_vectors


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    if len(v1) != len(v2):
        raise ValueError(
            "Adapter vectors have different lengths "
            f"({len(v1):,} vs {len(v2):,}). "
            "Check adapter weight keys and shapes for alignment."
        )

    # Handle zero vectors
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def _initialize_stream_stats() -> Dict[str, float]:
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "near_zero": 0,
    }


def _finalize_stream_stats(stats: Dict[str, float]) -> Dict[str, float]:
    count = stats["count"]
    if count == 0:
        return {
            "num_parameters": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "l2_norm": 0.0,
            "sparsity": 0.0,
        }
    mean = stats["sum"] / count
    var = max(stats["sumsq"] / count - mean ** 2, 0.0)
    return {
        "num_parameters": int(count),
        "mean": float(mean),
        "std": float(np.sqrt(var)),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
        "l2_norm": float(np.sqrt(stats["sumsq"])),
        "sparsity": float(stats["near_zero"] / count),
    }


def compute_weight_space_similarities(
    adapter_files: Dict[str, np.lib.npyio.NpzFile]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute cosine similarity of flattened LoRA factor weights (streamed)."""
    expected_keys = _validate_weight_alignment(adapter_files)
    heuristics = sorted(adapter_files.keys())

    dot_sums = {f"{h1}-{h2}": 0.0 for h1, h2 in combinations(heuristics, 2)}
    norms = {h: 0.0 for h in heuristics}
    stream_stats = {h: _initialize_stream_stats() for h in heuristics}

    for key in expected_keys:
        tensors = {}
        for h in heuristics:
            t = np.asarray(adapter_files[h][key])
            tensors[h] = t
            stats = stream_stats[h]
            stats["count"] += t.size
            stats["sum"] += float(np.sum(t, dtype=np.float64))
            stats["sumsq"] += float(np.sum(t * t, dtype=np.float64))
            stats["min"] = min(stats["min"], float(np.min(t)))
            stats["max"] = max(stats["max"], float(np.max(t)))
            stats["near_zero"] += int(np.sum(np.abs(t) < 1e-6))
            norms[h] += float(np.sum(t * t, dtype=np.float64))

        for h1, h2 in combinations(heuristics, 2):
            pair = f"{h1}-{h2}"
            dot_sums[pair] += float(np.sum(tensors[h1] * tensors[h2], dtype=np.float64))

    similarities = {}
    for h1, h2 in combinations(heuristics, 2):
        pair = f"{h1}-{h2}"
        denom = np.sqrt(norms[h1] * norms[h2])
        similarities[pair] = float(dot_sums[pair] / denom) if denom > 0 else 0.0

    vector_stats = {h: _finalize_stream_stats(stream_stats[h]) for h in heuristics}
    return similarities, vector_stats


def compute_effective_update_similarities(
    adapter_files: Dict[str, np.lib.npyio.NpzFile]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute cosine similarity of effective LoRA updates (Delta W = B @ A)."""
    base_keys, lora_index, _ = _validate_lora_alignment(adapter_files)
    heuristics = sorted(adapter_files.keys())

    dot_sums = {f"{h1}-{h2}": 0.0 for h1, h2 in combinations(heuristics, 2)}
    norms = {h: 0.0 for h in heuristics}

    for base in base_keys:
        tensors = {}
        for h in heuristics:
            keys = lora_index[h][base]
            a = np.asarray(adapter_files[h][keys["A"]])
            b = np.asarray(adapter_files[h][keys["B"]])
            tensors[h] = (a, b)

        for h in heuristics:
            a, b = tensors[h]
            norms[h] += _effective_update_inner(a, b, a, b)

        for h1, h2 in combinations(heuristics, 2):
            pair = f"{h1}-{h2}"
            a1, b1 = tensors[h1]
            a2, b2 = tensors[h2]
            dot_sums[pair] += _effective_update_inner(a1, b1, a2, b2)

    similarities = {}
    for h1, h2 in combinations(heuristics, 2):
        pair = f"{h1}-{h2}"
        denom = np.sqrt(norms[h1] * norms[h2])
        similarities[pair] = float(dot_sums[pair] / denom) if denom > 0 else 0.0

    update_norms = {h: float(np.sqrt(norms[h])) for h in heuristics}
    return similarities, update_norms


def classify_comparisons(
    similarities: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Classify pairwise comparisons as same-heuristic or cross-heuristic.

    Same-heuristic comparisons (e.g., DD_SEED42-DD_SEED123) serve as controls
    to validate that cross-heuristic orthogonality is meaningful.

    Args:
        similarities: Dict mapping pair name to cosine similarity

    Returns:
        Tuple of (same_heuristic_sims, cross_heuristic_sims)
    """
    same_heuristic = {}
    cross_heuristic = {}

    for pair, sim in similarities.items():
        h1, h2 = pair.split("-")

        # Extract base heuristic (before any _SEED suffix)
        base1 = h1.split("_SEED")[0]
        base2 = h2.split("_SEED")[0]

        if base1 == base2:
            same_heuristic[pair] = sim
        else:
            cross_heuristic[pair] = sim

    return same_heuristic, cross_heuristic


def summarize_seed_controls(similarities: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Build a canonical same-heuristic seed-control summary for primary heuristics."""
    same_seed_pairs: Dict[str, float] = {}
    for pair, sim in similarities.items():
        h1, h2 = pair.split("-")
        base1 = h1.split("_SEED")[0]
        base2 = h2.split("_SEED")[0]
        seeded1 = "_SEED" in h1
        seeded2 = "_SEED" in h2
        if base1 != base2 or base1 not in PRIMARY_HEURISTICS:
            continue
        # Keep only the direct unseeded-vs-seeded controls.
        if seeded1 == seeded2:
            continue
        same_seed_pairs[pair] = sim

    primary_cross_pairs: Dict[str, float] = {}
    for h1, h2 in combinations(PRIMARY_HEURISTICS, 2):
        forward = f"{h1}-{h2}"
        reverse = f"{h2}-{h1}"
        if forward in similarities:
            primary_cross_pairs[forward] = similarities[forward]
        elif reverse in similarities:
            primary_cross_pairs[reverse] = similarities[reverse]

    if not same_seed_pairs:
        return None

    same_avg = float(np.mean(list(same_seed_pairs.values()))) if same_seed_pairs else None
    primary_cross_avg = (
        float(np.mean(list(primary_cross_pairs.values()))) if primary_cross_pairs else None
    )
    gap = (
        float(same_avg - primary_cross_avg)
        if same_avg is not None and primary_cross_avg is not None
        else None
    )

    return {
        "same_heuristic_seed_pairs": same_seed_pairs,
        "primary_cross_pairs": primary_cross_pairs,
        "same_heuristic_avg": same_avg,
        "primary_cross_avg": primary_cross_avg,
        "gap": gap,
    }


def print_analysis(
    title: str,
    similarities: Dict[str, float],
    stats: Optional[Dict[str, Dict]] = None,
    same_heuristic_sims: Optional[Dict[str, float]] = None,
    cross_heuristic_sims: Optional[Dict[str, float]] = None
):
    """Print formatted similarity analysis."""
    tprint("\n" + "=" * 60)
    tprint(title)
    tprint("=" * 60)

    # Vector statistics
    if stats:
        tprint("\nAdapter Weight Statistics:")
        tprint("-" * 40)
        for heuristic, s in sorted(stats.items()):
            tprint(f"\n  {heuristic}:")
            tprint(f"    Parameters: {s['num_parameters']:,}")
            tprint(f"    L2 norm: {s['l2_norm']:.4f}")
            tprint(f"    Mean: {s['mean']:.6f}")
            tprint(f"    Std: {s['std']:.6f}")
            tprint(f"    Sparsity: {s['sparsity']:.1%}")

    # Same-heuristic controls (if available)
    if same_heuristic_sims:
        tprint("\n\nSAME-HEURISTIC SEED CONTROLS:")
        tprint("-" * 40)
        tprint("  (These should be ALIGNED if training is stable)")
        tprint()
        for pair, sim in sorted(same_heuristic_sims.items()):
            if abs(sim) > 0.5:
                interp = "ALIGNED (good - validates training)"
            elif abs(sim) > 0.3:
                interp = "Moderate alignment"
            else:
                interp = "WARNING: Low alignment - training instability?"
            tprint(f"  {pair}: {sim:+.4f} ({interp})")

        avg_same = np.mean(list(same_heuristic_sims.values()))
        tprint(f"\n  Average same-heuristic similarity: {avg_same:+.4f}")

    # Cross-heuristic comparisons
    cross_sims = cross_heuristic_sims or similarities
    tprint("\n\nCROSS-HEURISTIC COMPARISONS:")
    tprint("-" * 40)
    tprint("  (Values near 0 = orthogonal/independent mechanisms)")
    tprint()

    for pair, sim in sorted(cross_sims.items()):
        if abs(sim) < 0.1:
            interp = "ORTHOGONAL"
        elif abs(sim) < 0.3:
            interp = "Nearly orthogonal"
        elif abs(sim) < 0.5:
            interp = "Moderate correlation"
        elif abs(sim) < 0.7:
            interp = "Strong correlation"
        else:
            interp = "HIGHLY ALIGNED"
        tprint(f"  {pair}: {sim:+.4f} ({interp})")

    if cross_sims:
        avg_cross = np.mean(list(cross_sims.values()))
        tprint(f"\n  Average cross-heuristic similarity: {avg_cross:+.4f}")

    # Orthogonality validation summary
    if same_heuristic_sims and cross_sims:
        avg_same = np.mean(list(same_heuristic_sims.values()))
        avg_cross = np.mean(list(cross_sims.values()))
        gap = avg_same - avg_cross

        tprint("\n\nORTHOGONALITY VALIDATION:")
        tprint("-" * 40)
        tprint(f"  Same-heuristic avg:  {avg_same:+.4f}")
        tprint(f"  Cross-heuristic avg: {avg_cross:+.4f}")
        tprint(f"  Gap: {gap:+.4f}")
        if gap > 0.2:
            tprint("  VERDICT: Orthogonality claim SUPPORTED")
            tprint("           (Same-heuristic LoRAs are more aligned than cross-heuristic)")
        elif gap > 0.05:
            tprint("  VERDICT: Weak support for orthogonality")
        else:
            tprint("  VERDICT: Orthogonality claim NOT SUPPORTED")
            tprint("           (Cross-heuristic is as aligned as same-heuristic)")

    # Build similarity matrix for display
    heuristics = sorted(set(h for pair in similarities for h in pair.split("-")))
    tprint("\n\nSimilarity Matrix:")
    tprint("-" * 40)
    header = "        " + "  ".join(f"{h:>8}" for h in heuristics)
    tprint(header)

    for h1 in heuristics:
        row_vals: List[Optional[float]] = []
        for h2 in heuristics:
            if h1 == h2:
                row_vals.append(1.000)
            else:
                key = f"{min(h1,h2)}-{max(h1,h2)}"
                row_vals.append(similarities.get(key))
        formatted = []
        for v in row_vals:
            formatted.append(f"{v:>8.4f}" if v is not None else f"{'N/A':>8}")
        tprint(f"  {h1}:  " + "  ".join(formatted))


def _save_similarity_analysis(
    similarities: Dict[str, float],
    output_dir: Path,
    filename_prefix: str,
    method: str,
    description: str,
    stats: Optional[Dict[str, Dict]] = None,
    same_heuristic_sims: Optional[Dict[str, float]] = None,
    cross_heuristic_sims: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """Save similarity analysis results and CSV matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    orthogonality_validation = None
    if same_heuristic_sims and cross_heuristic_sims:
        avg_same = float(np.mean(list(same_heuristic_sims.values())))
        avg_cross = float(np.mean(list(cross_heuristic_sims.values())))
        gap = avg_same - avg_cross
        orthogonality_validation = {
            "same_heuristic_avg": avg_same,
            "cross_heuristic_avg": avg_cross,
            "gap": gap,
            "verdict": "supported" if gap > 0.2 else ("weak" if gap > 0.05 else "not_supported")
        }

    analysis: Dict[str, Any] = {
        "method": method,
        "description": description,
        "similarities": similarities,
        "same_heuristic_similarities": same_heuristic_sims or {},
        "cross_heuristic_similarities": cross_heuristic_sims or similarities,
        "orthogonality_validation": orthogonality_validation,
    }
    if stats is not None:
        analysis["vector_stats"] = stats
    if extra:
        analysis.update(extra)

    json_path = output_dir / f"{filename_prefix}_similarity.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    tprint(f"Saved {filename_prefix} similarity analysis to {json_path}")

    heuristics = sorted(set(h for pair in similarities for h in pair.split("-")))
    matrix_path = output_dir / f"{filename_prefix}_similarity_matrix.csv"
    with open(matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + heuristics)
        for h1 in heuristics:
            row: List[Union[str, float]] = [h1]
            for h2 in heuristics:
                if h1 == h2:
                    row.append(1.0)
                else:
                    key = f"{min(h1,h2)}-{max(h1,h2)}"
                    row.append(similarities.get(key, "N/A"))
            writer.writerow(row)
    tprint(f"Saved similarity matrix to {matrix_path}")


def save_placeholder_results(output_dir: Path):
    """Save placeholder results when weights are unavailable."""
    output_dir.mkdir(parents=True, exist_ok=True)

    placeholder_similarities = {
        "DD-OT": None,
        "DD-RC": None,
        "OT-RC": None,
    }

    _save_similarity_analysis(
        placeholder_similarities,
        output_dir,
        filename_prefix="cosine",
        method="placeholder",
        description="Adapter weights not available from Tinker API",
        stats={},
        extra={"note": "Weight extraction from Tinker API not supported. Rerun after API update."},
    )

    _save_similarity_analysis(
        placeholder_similarities,
        output_dir,
        filename_prefix="effective_update",
        method="placeholder",
        description="Adapter weights not available from Tinker API",
        stats={},
        extra={"note": "Weight extraction from Tinker API not supported. Rerun after API update."},
    )


def main():
    """Run gradient orthogonality analysis using adapter weights."""
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between LoRA adapter weight vectors"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Model name for model-specific paths (default: use legacy paths)"
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=None,
        help="Directory containing adapter weight files (overrides --model)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results (overrides --model)"
    )
    args = parser.parse_args()

    # Get model-specific paths (or use explicit overrides)
    default_weights, default_output, _training_summary_path = get_model_paths(args.model)
    training_summary_paths = get_training_summary_paths(args.model)
    weights_dir = args.weights_dir or default_weights
    output_dir = args.output_dir or default_output

    tprint("=" * 60)
    tprint("LoRA Similarity Analysis (Weight-Space + Effective Update)")
    tprint("=" * 60)
    if args.model:
        tprint(f"Model: {args.model}")
    tprint(f"Weights directory: {weights_dir}")
    tprint(f"Output directory: {output_dir}")
    tprint()

    # Attempt to top up weights from any canonical/seed-scoped training summaries.
    downloaded = 0
    if training_summary_paths:
        tprint("Checking training summaries for downloadable adapter weights...")
        downloaded = download_weights_from_checkpoints(weights_dir, training_summary_paths)
        if downloaded:
            tprint(f"  Downloaded or confirmed {downloaded} adapter weight files")

    # Check if weights directory exists and has weight files
    weights_available = (
        weights_dir.exists() and
        any(weights_dir.glob("*_adapter_weights.npz"))
    )

    if not weights_available:
        tprint("Adapter weights not found locally.")
        tprint(f"  Expected directory: {weights_dir}")
        tprint()
        tprint(f"Could not download enough adapter weights ({downloaded}/2 minimum needed)")
        tprint("Creating placeholder results (cosine similarities = N/A)")
        tprint()

        # Create placeholder results so pipeline can continue
        save_placeholder_results(output_dir)

        tprint("\n" + "=" * 60)
        tprint("Done! (placeholder mode)")
        tprint("=" * 60)
        sys.exit(0)  # Exit successfully so pipeline continues

    # Load adapter files
    tprint("Loading adapter weights (mmap)...")
    adapter_files = _load_adapter_files(weights_dir)

    if len(adapter_files) < 2:
        tprint(f"\nWARNING: Need at least 2 adapters for comparison, found {len(adapter_files)}")
        tprint("Creating placeholder results...")
        save_placeholder_results(output_dir)
        tprint("\n" + "=" * 60)
        tprint("Done! (placeholder mode)")
        tprint("=" * 60)
        sys.exit(0)

    tprint(f"\nLoaded {len(adapter_files)} adapter files")

    # Weight-space cosine similarity (flattened A/B factors)
    tprint("\nComputing weight-space cosine similarities (flattened A/B factors)...")
    weight_similarities, weight_stats = compute_weight_space_similarities(adapter_files)
    same_weight, cross_weight = classify_comparisons(weight_similarities)

    # Effective update cosine similarity (Delta W = B @ A)
    tprint("Computing effective-update cosine similarities (Delta W = B @ A)...")
    effective_similarities, effective_norms = compute_effective_update_similarities(adapter_files)
    same_effective, cross_effective = classify_comparisons(effective_similarities)
    seed_control_summary = summarize_seed_controls(effective_similarities)

    # Print results
    print_analysis(
        "ADAPTER WEIGHT COSINE SIMILARITY ANALYSIS (FLATTENED A/B)",
        weight_similarities,
        weight_stats,
        same_weight,
        cross_weight,
    )
    print_analysis(
        "EFFECTIVE UPDATE COSINE SIMILARITY ANALYSIS (DELTA W = B @ A)",
        effective_similarities,
        None,
        same_effective,
        cross_effective,
    )

    # Save results
    tprint("\n" + "=" * 60)
    tprint("Saving Results")
    tprint("=" * 60)
    _save_similarity_analysis(
        weight_similarities,
        output_dir,
        filename_prefix="cosine",
        method="adapter_weight_cosine_similarity",
        description="Pairwise cosine similarity between flattened LoRA adapter weights (A/B factors)",
        stats=weight_stats,
        same_heuristic_sims=same_weight,
        cross_heuristic_sims=cross_weight,
    )
    _save_similarity_analysis(
        effective_similarities,
        output_dir,
        filename_prefix="effective_update",
        method="effective_update_cosine_similarity",
        description="Pairwise cosine similarity between effective LoRA updates (Delta W = B @ A)",
        stats=None,
        same_heuristic_sims=same_effective,
        cross_heuristic_sims=cross_effective,
        extra={
            "update_norms": effective_norms,
            "seed_control_summary": seed_control_summary,
        },
    )

    tprint("\n" + "=" * 60)
    tprint("Done!")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
