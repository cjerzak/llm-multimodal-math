#!/usr/bin/env python3
"""
DatasetSplits.py

Centralized utilities for train/val/test split management across all datasets.
Ensures reproducibility, prevents data leakage, and provides consistent split logic.

Usage:
    from DatasetSplits import assign_splits, load_split, get_hds_splits, build_exclusion_set
"""

import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, DefaultDict
from collections import defaultdict

# Add Scripts to path for imports when run as __main__
_SCRIPT_DIR = Path(__file__).parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.Logging import tprint

# =============================================================================
# CONFIGURATION
# =============================================================================
SPLIT_SEED = 42  # Global seed for reproducibility
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Repository root (relative to this script)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent


# =============================================================================
# CORE SPLIT FUNCTIONS
# =============================================================================

def set_reproducible_state(seed: int = SPLIT_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)


def _deterministic_split(problem_id: str, ratios: Dict[str, float], seed: int = SPLIT_SEED) -> str:
    """
    Assign split based on hash of problem_id.

    This ensures:
    - Same problem always gets same split
    - Adding new problems doesn't change existing assignments
    - Order-independent: shuffling dataset doesn't change splits

    Args:
        problem_id: Unique identifier for the problem
        ratios: Dict with keys 'train', 'val', 'test' and float values summing to 1.0
        seed: Random seed for hash

    Returns:
        Split name ('train', 'val', or 'test')
    """
    hash_input = f"{seed}:{problem_id}".encode()
    hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
    normalized = (hash_val % 10000) / 10000  # Value in [0, 1)

    cumulative = 0.0
    for split, ratio in ratios.items():
        cumulative += ratio
        if normalized < cumulative:
            return split

    # Fallback to last split (handles floating point edge cases)
    return list(ratios.keys())[-1]


def assign_splits(
    dataset: List[Dict[str, Any]],
    ratios: Optional[Dict[str, float]] = None,
    id_key: str = "id",
    stratify_key: Optional[str] = None,
    seed: int = SPLIT_SEED
) -> List[Dict[str, Any]]:
    """
    Add 'split' column to dataset with deterministic assignment.

    Args:
        dataset: List of row dicts
        ratios: Split ratios (default: 70/15/15)
        id_key: Column name for unique ID
        stratify_key: Optional column for stratified splitting
        seed: Random seed

    Returns:
        Dataset with 'split' column added to each row
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS.copy()

    # Validate ratios sum to 1.0
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    if stratify_key:
        # Stratified splitting: maintain proportions within each stratum
        return _stratified_split(dataset, ratios, id_key, stratify_key, seed)
    else:
        # Simple deterministic splitting
        for row in dataset:
            row["split"] = _deterministic_split(row[id_key], ratios, seed)
        return dataset


def _stratified_split(
    dataset: List[Dict[str, Any]],
    ratios: Dict[str, float],
    id_key: str,
    stratify_key: str,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Stratified split maintaining proportions within each group.

    Uses deterministic assignment within each stratum.
    """
    # Group by stratify key
    groups = defaultdict(list)
    for row in dataset:
        groups[row[stratify_key]].append(row)

    # Assign splits within each group
    for group_name, group_rows in groups.items():
        for row in group_rows:
            # Include group in hash for better distribution
            composite_id = f"{group_name}:{row[id_key]}"
            row["split"] = _deterministic_split(composite_id, ratios, seed)

    return dataset


def load_split(
    csv_path: Path,
    split: str = "train",
    id_key: str = "id"
) -> List[Dict[str, Any]]:
    """
    Load only rows matching the specified split.

    Args:
        csv_path: Path to CSV file with 'split' column
        split: Which split to load ('train', 'val', 'test', or 'all')
        id_key: Unused, kept for API compatibility

    Returns:
        List of row dicts matching the split
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split == "all" or row.get("split") == split:
                rows.append(row)
    return rows


def load_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load all rows from a CSV file."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# =============================================================================
# HDS-SPECIFIC FUNCTIONS
# =============================================================================

def get_hds_splits(
    hds_rows: List[Dict[str, Any]],
    ratios: Optional[Dict[str, float]] = None,
    seed: int = SPLIT_SEED
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stratified split of HDS by target_heuristic.

    Ensures balanced representation of each heuristic (RC, DD, OT) in each split.

    Args:
        hds_rows: List of HDS row dicts with 'target_heuristic' field
        ratios: Split ratios (default: 70/15/15)
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing list of rows
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS.copy()

    # Apply stratified splits
    hds_with_splits = assign_splits(
        hds_rows,
        ratios=ratios,
        id_key="id",
        stratify_key="target_heuristic",
        seed=seed
    )

    # Group by split
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in hds_with_splits:
        splits[row["split"]].append(row)

    return splits


# =============================================================================
# DATA LEAKAGE PREVENTION
# =============================================================================

def get_problem_fingerprint(a: int, b: int) -> str:
    """
    Generate unique fingerprint for a multiplication problem.

    Treats (a, b) and (b, a) as the same problem to respect commutativity.

    Args:
        a: First operand
        b: Second operand

    Returns:
        String fingerprint "min:max"
    """
    left, right = (a, b) if a <= b else (b, a)
    return f"{left}:{right}"


def build_exclusion_set(
    test_csv_paths: List[Path],
    split_filter: str = "test"
) -> Set[str]:
    """
    Build set of problem fingerprints from test datasets.

    Use this to prevent test problems from appearing in training data.

    Args:
        test_csv_paths: Paths to CSV files containing problems to exclude
        split_filter: Only include problems with this split value
                     (use "all" to include all problems, or None for files without split column)

    Returns:
        Set of problem fingerprints to exclude from training
    """
    exclusion_set = set()

    for csv_path in test_csv_paths:
        if not csv_path.exists():
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check split filter
                if "split" in row:
                    if split_filter != "all" and row.get("split") != split_filter:
                        continue

                # Get operands (handle different column names)
                a = int(row.get("a", 0))
                b = int(row.get("b", 0))

                if a and b:
                    exclusion_set.add(get_problem_fingerprint(a, b))

    return exclusion_set


def is_excluded_problem(a: int, b: int, exclusion_set: Set[str]) -> bool:
    """
    Check if a problem appears in the exclusion set.

    Args:
        a: First operand
        b: Second operand
        exclusion_set: Set of problem fingerprints to check

    Returns:
        True if problem should be excluded
    """
    fingerprint = get_problem_fingerprint(a, b)
    return fingerprint in exclusion_set


def get_default_exclusion_set() -> Set[str]:
    """
    Build exclusion set from standard test datasets.

    Includes:
    - HDS test split
    - Traps (all problems - adversarial by design)
    - Multi-modal grid test splits

    Returns:
        Set of problem fingerprints to exclude from training
    """
    test_paths = [
        REPO_ROOT / "SavedData" / "HDS.csv",
        REPO_ROOT / "SavedData" / "Traps.csv",
        REPO_ROOT / "SavedData" / "ImageGrid.csv",
        REPO_ROOT / "SavedData" / "AudioGrid.csv",
        REPO_ROOT / "SavedData" / "TextGrid.csv",
    ]

    exclusion_set = set()

    for csv_path in test_paths:
        if not csv_path.exists():
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            for row in reader:
                # For files with split column, only exclude test split
                # For Traps, exclude all (no split column expected)
                if "split" in fieldnames:
                    if row.get("split") not in ("test", "val"):
                        continue

                # Get operands
                a = int(row.get("a", 0))
                b = int(row.get("b", 0))

                if a and b:
                    exclusion_set.add(get_problem_fingerprint(a, b))

    return exclusion_set


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_no_leakage(
    train_csv: Path,
    test_csv: Path,
    a_col: str = "a",
    b_col: str = "b"
) -> Tuple[bool, List[str]]:
    """
    Validate that no problems in train appear in test.

    Args:
        train_csv: Path to training data CSV
        test_csv: Path to test data CSV
        a_col: Column name for first operand
        b_col: Column name for second operand

    Returns:
        (is_valid, list of leaked fingerprints)
    """
    # Build test set
    test_fingerprints = set()
    with open(test_csv, 'r') as f:
        for row in csv.DictReader(f):
            a, b = int(row[a_col]), int(row[b_col])
            test_fingerprints.add(get_problem_fingerprint(a, b))

    # Check train for leaks
    leaked = []
    with open(train_csv, 'r') as f:
        for row in csv.DictReader(f):
            a, b = int(row[a_col]), int(row[b_col])
            fp = get_problem_fingerprint(a, b)
            if fp in test_fingerprints:
                leaked.append(fp)

    return len(leaked) == 0, leaked


def get_split_stats(csv_path: Path) -> Dict[str, int]:
    """
    Get count of rows in each split.

    Args:
        csv_path: Path to CSV with 'split' column

    Returns:
        Dict mapping split name to count
    """
    stats: DefaultDict[str, int] = defaultdict(int)
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            split = row.get("split", "unknown")
            stats[split] += 1
    return dict(stats)


# =============================================================================
# CLI UTILITIES
# =============================================================================

def parse_split_ratios(ratio_string: str) -> Dict[str, float]:
    """
    Parse split ratio string like "70/15/15" into dict.

    Args:
        ratio_string: Format "train/val/test" as percentages

    Returns:
        Dict with 'train', 'val', 'test' keys and float values
    """
    parts = ratio_string.split("/")
    if len(parts) == 3:
        train, val, test = [float(p) / 100 for p in parts]
        return {"train": train, "val": val, "test": test}
    elif len(parts) == 2:
        train, test = [float(p) / 100 for p in parts]
        return {"train": train, "val": 0.0, "test": test}
    else:
        raise ValueError(f"Invalid ratio format: {ratio_string}. Use 'train/val/test' like '70/15/15'")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    tprint("DatasetSplits.py - Split utility module")
    tprint("=" * 60)
    tprint()

    # Test deterministic split
    tprint("Testing deterministic split assignment:")
    test_ids = ["prob_001", "prob_002", "prob_003", "prob_004", "prob_005"]
    for pid in test_ids:
        split = _deterministic_split(pid, DEFAULT_RATIOS)
        tprint(f"  {pid} -> {split}")
    tprint()

    # Test ratio parsing
    tprint("Testing ratio parsing:")
    for ratio_str in ["70/15/15", "80/10/10", "80/20"]:
        try:
            ratios = parse_split_ratios(ratio_str)
            tprint(f"  '{ratio_str}' -> {ratios}")
        except ValueError as e:
            tprint(f"  '{ratio_str}' -> ERROR: {e}")
    tprint()

    # Test fingerprint
    tprint("Testing problem fingerprints:")
    tprint(f"  47 x 36 -> {get_problem_fingerprint(47, 36)}")
    tprint(f"  36 x 47 -> {get_problem_fingerprint(36, 47)}")
    tprint()

    tprint("Module loaded successfully!")
