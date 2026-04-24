#!/usr/bin/env python3
"""
GenerateMathHelpers.py

Shared utilities for generating paired multimodal multiplication datasets.
Used by GenerateMathImages.py, GenerateMathAudio.py, and GenerateMathText.py.
"""

import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .DatasetSplits import DEFAULT_RATIOS, SPLIT_SEED, assign_splits
from .Logging import tprint

# Paths relative to repository root (core/ -> Scripts/ -> repo root)
REPO_ROOT = Path(__file__).parent.parent.parent
SHARED_MULTIMODAL_CSV = REPO_ROOT / "SavedData" / "SharedMultimodalGrid.csv"

MULTIMODAL_DEFAULT_COUNT = 10_000
MULTIMODAL_ID_PREFIX = "mm"
MULTIMODAL_ID_WIDTH = 5
MIN_OPERAND_DIGITS = 1
MAX_OPERAND_DIGITS = 9
DEFAULT_COMPLEXITY_MIN = 10
DEFAULT_COMPLEXITY_MAX = 324
CANONICAL_TARGET_MEAN_C = 182.0
CANONICAL_COMPLEXITY_BANDS: Tuple[Tuple[int, int], ...] = (
    (10, 60),
    (61, 120),
    (121, 180),
    (181, 240),
    (241, 324),
)
CANONICAL_BAND_MIN_FRACTION = 0.10
LOCK_TIMEOUT_SECONDS = 300
LOCK_STALE_SECONDS = 900
LOCK_POLL_INTERVAL_SECONDS = 0.2


def count_nonzero_digits(n: int) -> int:
    """Count non-zero digits in an integer."""
    return sum(1 for ch in str(abs(n)) if ch != "0")


def compute_problem_stats(a: int, b: int) -> Dict[str, Any]:
    """Compute derived metadata for a multiplication problem."""
    digits_a = len(str(abs(a)))
    digits_b = len(str(abs(b)))
    digit_total = digits_a + digits_b
    nonzero_a = count_nonzero_digits(a)
    nonzero_b = count_nonzero_digits(b)
    nonzero_total = nonzero_a + nonzero_b
    complexity_c = digit_total * nonzero_total
    return {
        "digits_a": digits_a,
        "digits_b": digits_b,
        "nonzero_a": nonzero_a,
        "nonzero_b": nonzero_b,
        "digit_total": digit_total,
        "nonzero_total": nonzero_total,
        "complexity_c": complexity_c,
        "stratum_id": f"d{digit_total}_nz{nonzero_total}",
    }


def canonical_problem_key(a: int, b: int) -> Tuple[int, int]:
    """Return a commutative canonical key for a multiplication problem."""
    return (a, b) if a <= b else (b, a)


def _validate_complexity_band(complexity_min: int, complexity_max: int) -> None:
    """Validate a requested complexity band."""
    if complexity_min < 1:
        raise ValueError("complexity_min must be >= 1")
    if complexity_max < complexity_min:
        raise ValueError("complexity_max must be >= complexity_min")


def _feasible_strata() -> List[Tuple[int, int]]:
    """Return feasible (digit_total, nonzero_total) strata."""
    strata: List[Tuple[int, int]] = []
    for digit_total in range(2, (MAX_OPERAND_DIGITS * 2) + 1):
        for nonzero_total in range(2, digit_total + 1):
            strata.append((digit_total, nonzero_total))
    return strata


def canonical_band_label(lo: int, hi: int) -> str:
    """Return a stable text label for a complexity band."""
    return f"{lo}-{hi}"


def canonical_complexity_band(complexity_c: int) -> Optional[Tuple[int, int]]:
    """Return the canonical complexity band containing a complexity value."""
    for lo, hi in CANONICAL_COMPLEXITY_BANDS:
        if lo <= complexity_c <= hi:
            return (lo, hi)
    return None


def band_counts_for_complexities(values: Sequence[int]) -> Dict[str, int]:
    """Count complexity values by canonical band label."""
    counts = {canonical_band_label(lo, hi): 0 for lo, hi in CANONICAL_COMPLEXITY_BANDS}
    for value in values:
        band = canonical_complexity_band(int(value))
        if band is None:
            continue
        counts[canonical_band_label(*band)] += 1
    return counts


def _reachable_complexities(
    complexity_min: int = DEFAULT_COMPLEXITY_MIN,
    complexity_max: int = DEFAULT_COMPLEXITY_MAX,
) -> List[int]:
    """Return sorted reachable complexity values inside the requested band."""
    _validate_complexity_band(complexity_min, complexity_max)
    return sorted(
        {
            digit_total * nonzero_total
            for digit_total, nonzero_total in _feasible_strata()
            if complexity_min <= digit_total * nonzero_total <= complexity_max
        }
    )


def _partition_operands(total: int, minimum: int, maximum: int) -> List[Tuple[int, int]]:
    """Enumerate ordered two-part partitions within operand digit bounds."""
    parts = []
    for left in range(minimum, maximum + 1):
        right = total - left
        if minimum <= right <= maximum:
            parts.append((left, right))
    return parts


def _generate_number_with_nonzero_digits(
    length: int,
    nonzero_digits: int,
    rng: random.Random
) -> Tuple[int, str]:
    """Generate an integer and template with fixed length and non-zero count."""
    if length < 1:
        raise ValueError("Length must be >= 1")
    if nonzero_digits < 1 or nonzero_digits > length:
        raise ValueError("nonzero_digits must be between 1 and length")

    digits = ["0"] * length
    positions = [0]
    if nonzero_digits > 1:
        positions.extend(rng.sample(range(1, length), nonzero_digits - 1))
    for idx in positions:
        digits[idx] = str(rng.randint(1, 9))

    template = "".join("D" if d != "0" else "0" for d in digits)
    return int("".join(digits)), template


def _build_stratified_problem(
    digit_total: int,
    nonzero_total: int,
    rng: random.Random
) -> Tuple[int, int, str, str]:
    """Sample one problem from a target (digit_total, nonzero_total) stratum."""
    digit_partitions = _partition_operands(
        digit_total,
        minimum=MIN_OPERAND_DIGITS,
        maximum=MAX_OPERAND_DIGITS,
    )
    if not digit_partitions:
        raise ValueError(f"No digit partitions for total digits={digit_total}")

    feasible_partitions: List[Tuple[int, int, int, int]] = []
    for digits_a, digits_b in digit_partitions:
        for nonzero_a in range(1, digits_a + 1):
            nonzero_b = nonzero_total - nonzero_a
            if 1 <= nonzero_b <= digits_b:
                feasible_partitions.append((digits_a, digits_b, nonzero_a, nonzero_b))

    if not feasible_partitions:
        raise ValueError(
            f"No feasible partitions for digit_total={digit_total}, nonzero_total={nonzero_total}"
        )

    digits_a, digits_b, nonzero_a, nonzero_b = rng.choice(feasible_partitions)
    a, template_a = _generate_number_with_nonzero_digits(digits_a, nonzero_a, rng)
    b, template_b = _generate_number_with_nonzero_digits(digits_b, nonzero_b, rng)
    return a, b, template_a, template_b


def _allocate_even_quotas_with_capacities(
    keys: Sequence[Any],
    total_count: int,
    capacities: Dict[Any, int],
) -> Dict[Any, int]:
    """Allocate quotas as evenly as possible without exceeding each key's capacity."""
    if total_count < len(keys):
        raise ValueError(
            f"Requested count={total_count} is too small for {len(keys)} buckets."
        )

    total_capacity = sum(capacities[key] for key in keys)
    if total_count > total_capacity:
        raise ValueError(
            f"Requested count={total_count} exceeds total unique capacity={total_capacity}."
        )

    quotas: Dict[Any, int] = {key: 0 for key in keys}
    remaining = total_count
    while remaining > 0:
        open_keys = [key for key in keys if quotas[key] < capacities[key]]
        if not open_keys:
            raise RuntimeError("Ran out of feasible strata before meeting requested count.")

        fair_share = remaining // len(open_keys)
        if fair_share == 0:
            for key in open_keys:
                if remaining == 0:
                    break
                quotas[key] += 1
                remaining -= 1
            continue

        for key in open_keys:
            spare = capacities[key] - quotas[key]
            add = min(fair_share, spare)
            quotas[key] += add
            remaining -= add

    return quotas


def _allocate_even_quotas(strata: Sequence[Tuple[int, int]], total_count: int) -> Dict[Tuple[int, int], int]:
    """Allocate row quotas as evenly as possible without exceeding stratum capacity."""
    capacities = {stratum: _compute_stratum_capacity(*stratum) for stratum in strata}
    return _allocate_even_quotas_with_capacities(strata, total_count, capacities)


def _allocate_weighted_quotas_with_capacities(
    keys: Sequence[Any],
    total_count: int,
    capacities: Dict[Any, int],
    weights: Dict[Any, float],
    minimum_per_key: int = 1,
) -> Dict[Any, int]:
    """Allocate quotas with capacity caps, preserving support for every key."""
    if total_count < len(keys) * minimum_per_key:
        raise ValueError(
            f"Requested count={total_count} is too small for {len(keys)} buckets "
            f"with minimum_per_key={minimum_per_key}."
        )

    total_capacity = sum(capacities[key] for key in keys)
    if total_count > total_capacity:
        raise ValueError(
            f"Requested count={total_count} exceeds total unique capacity={total_capacity}."
        )

    quotas: Dict[Any, int] = {}
    for key in keys:
        if capacities[key] < minimum_per_key:
            raise ValueError(
                f"Bucket {key!r} has capacity={capacities[key]} < minimum_per_key={minimum_per_key}."
            )
        if weights.get(key, 0.0) <= 0.0:
            raise ValueError(f"Bucket {key!r} must have a positive weight.")
        quotas[key] = minimum_per_key

    remaining = total_count - (len(keys) * minimum_per_key)
    key_order = {key: idx for idx, key in enumerate(keys)}

    while remaining > 0:
        active_keys = [key for key in keys if quotas[key] < capacities[key]]
        if not active_keys:
            raise RuntimeError("Ran out of feasible buckets before meeting requested count.")

        total_weight = sum(weights[key] for key in active_keys)
        if total_weight <= 0.0:
            raise RuntimeError("Weighted quota allocation requires positive active weight.")

        raw_shares = {
            key: (remaining * weights[key]) / total_weight
            for key in active_keys
        }
        remainders: List[Tuple[float, float, int, Any]] = []
        assigned = 0

        for key in active_keys:
            spare = capacities[key] - quotas[key]
            floor_share = min(int(raw_shares[key]), spare)
            quotas[key] += floor_share
            assigned += floor_share
            remainders.append(
                (
                    raw_shares[key] - floor_share,
                    weights[key],
                    -key_order[key],
                    key,
                )
            )

        remaining -= assigned
        if remaining == 0:
            break

        remainders.sort(reverse=True)
        remainder_progress = 0
        for _, _, _, key in remainders:
            if remaining == 0:
                break
            if quotas[key] >= capacities[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            remainder_progress += 1

        if assigned == 0 and remainder_progress == 0:
            raise RuntimeError("Weighted allocation stalled before meeting requested count.")

    return quotas


def _count_numbers_with_stats(length: int, nonzero_digits: int) -> int:
    """Count valid positive integers with a fixed length and non-zero digit count."""
    if length < 1 or nonzero_digits < 1 or nonzero_digits > length:
        return 0
    return math.comb(length - 1, nonzero_digits - 1) * (9 ** nonzero_digits)


def _compute_stratum_capacity(digit_total: int, nonzero_total: int) -> int:
    """Count unique commutative problems available in a given stratum."""
    group_counts: Dict[Tuple[int, int], int] = {}
    for digits_a, digits_b in _partition_operands(
        digit_total,
        minimum=MIN_OPERAND_DIGITS,
        maximum=MAX_OPERAND_DIGITS,
    ):
        for nonzero_a in range(1, digits_a + 1):
            nonzero_b = nonzero_total - nonzero_a
            if 1 <= nonzero_b <= digits_b:
                group_counts[(digits_a, nonzero_a)] = _count_numbers_with_stats(digits_a, nonzero_a)
                group_counts[(digits_b, nonzero_b)] = _count_numbers_with_stats(digits_b, nonzero_b)

    ordered_groups = sorted(group_counts.keys())
    capacity = 0
    for idx, left_group in enumerate(ordered_groups):
        left_digits, left_nonzero = left_group
        left_count = group_counts[left_group]
        for right_group in ordered_groups[idx:]:
            right_digits, right_nonzero = right_group
            if left_digits + right_digits != digit_total:
                continue
            if left_nonzero + right_nonzero != nonzero_total:
                continue
            right_count = group_counts[right_group]
            if left_group == right_group:
                capacity += left_count * (left_count + 1) // 2
            else:
                capacity += left_count * right_count
    return capacity


def _complexity_weights(
    complexities: Sequence[int],
    exponent: float,
) -> Dict[int, float]:
    """Return positive exact-C weights for an allocation exponent."""
    return {complexity: max(float(complexity), 1.0) ** exponent for complexity in complexities}


def _apply_quota_increment(
    quotas: Dict[int, int],
    increment: Dict[int, int],
) -> None:
    """Accumulate quota increments into an existing quota mapping."""
    for key, value in increment.items():
        quotas[key] = quotas.get(key, 0) + value


def _allocate_exact_c_quotas(
    count: int,
    complexities: Sequence[int],
    capacities: Dict[int, int],
    target_mean: float = CANONICAL_TARGET_MEAN_C,
    minimum_per_key: int = 1,
    band_min_fraction: float = CANONICAL_BAND_MIN_FRACTION,
) -> Dict[int, int]:
    """Allocate exact-C quotas with full support, band coverage, and a target mean."""
    if not complexities:
        raise ValueError("At least one reachable complexity is required.")
    if count < len(complexities) * minimum_per_key:
        raise ValueError(
            f"Requested count={count} is too small for {len(complexities)} exact-C buckets."
        )

    active_bands = [
        ((lo, hi), [c for c in complexities if lo <= c <= hi])
        for lo, hi in CANONICAL_COMPLEXITY_BANDS
    ]
    active_bands = [(band, keys) for band, keys in active_bands if keys]

    def quotas_for_exponent(exponent: float) -> Dict[int, int]:
        quotas = {complexity: minimum_per_key for complexity in complexities}
        remaining = count - sum(quotas.values())
        weights = _complexity_weights(complexities, exponent)

        for (_, _), band_keys in active_bands:
            band_target = max(
                math.ceil(count * band_min_fraction),
                len(band_keys) * minimum_per_key,
            )
            current = sum(quotas[key] for key in band_keys)
            if current >= band_target:
                continue
            need = band_target - current
            if need > remaining:
                raise ValueError("Band minimum coverage exceeds requested total count.")
            spare_capacities = {key: capacities[key] - quotas[key] for key in band_keys}
            band_increment = _allocate_weighted_quotas_with_capacities(
                band_keys,
                need,
                spare_capacities,
                {key: weights[key] for key in band_keys},
                minimum_per_key=0,
            )
            _apply_quota_increment(quotas, band_increment)
            remaining -= need

        if remaining > 0:
            spare_capacities = {key: capacities[key] - quotas[key] for key in complexities}
            global_increment = _allocate_weighted_quotas_with_capacities(
                complexities,
                remaining,
                spare_capacities,
                weights,
                minimum_per_key=0,
            )
            _apply_quota_increment(quotas, global_increment)

        return quotas

    def quota_mean(quotas: Dict[int, int]) -> float:
        total = sum(quotas.values())
        return sum(complexity * quota for complexity, quota in quotas.items()) / total

    low_exp = -1.0
    high_exp = 8.0
    low_quotas = quotas_for_exponent(low_exp)
    high_quotas = quotas_for_exponent(high_exp)
    low_mean = quota_mean(low_quotas)
    high_mean = quota_mean(high_quotas)

    if target_mean <= low_mean:
        return low_quotas
    if target_mean >= high_mean:
        return high_quotas

    best_quotas = low_quotas
    best_gap = abs(low_mean - target_mean)
    lo = low_exp
    hi = high_exp
    for _ in range(32):
        mid = (lo + hi) / 2.0
        candidate_quotas = quotas_for_exponent(mid)
        candidate_mean = quota_mean(candidate_quotas)
        candidate_gap = abs(candidate_mean - target_mean)
        if candidate_gap < best_gap:
            best_quotas = candidate_quotas
            best_gap = candidate_gap
        if candidate_mean < target_mean:
            lo = mid
        else:
            hi = mid

    return best_quotas


def _reachable_multimodal_strata(
    complexity_min: int,
    complexity_max: int,
) -> List[Tuple[int, int]]:
    """Return feasible multimodal strata inside a requested complexity band."""
    return [
        stratum
        for stratum in _feasible_strata()
        if complexity_min <= stratum[0] * stratum[1] <= complexity_max
    ]


def _multimodal_complexity_plan(
    complexity_min: int,
    complexity_max: int,
) -> Tuple[List[int], Dict[int, List[Tuple[int, int]]]]:
    """Return reachable exact C values and their feasible strata."""
    complexities = _reachable_complexities(complexity_min, complexity_max)
    strata = _reachable_multimodal_strata(complexity_min, complexity_max)
    return complexities, {
        complexity: [stratum for stratum in strata if stratum[0] * stratum[1] == complexity]
        for complexity in complexities
    }


def _allocate_multimodal_complexity_quotas(
    count: int,
    complexity_min: int,
    complexity_max: int,
) -> Dict[int, int]:
    """Allocate exact-C quotas for the shared multimodal grid with a hard-profile target mean."""
    complexities, complexity_to_strata = _multimodal_complexity_plan(
        complexity_min,
        complexity_max,
    )
    capacities = {
        complexity: sum(_compute_stratum_capacity(*stratum) for stratum in grouped_strata)
        for complexity, grouped_strata in complexity_to_strata.items()
    }
    return _allocate_exact_c_quotas(
        count,
        complexities,
        capacities,
        target_mean=CANONICAL_TARGET_MEAN_C,
        minimum_per_key=1,
        band_min_fraction=CANONICAL_BAND_MIN_FRACTION,
    )


def generate_paired_multimodal_dataset(
    count: int = MULTIMODAL_DEFAULT_COUNT,
    include_splits: bool = True,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = SPLIT_SEED,
    complexity_min: int = DEFAULT_COMPLEXITY_MIN,
    complexity_max: int = DEFAULT_COMPLEXITY_MAX,
) -> List[Dict[str, Any]]:
    """Generate one shared paired benchmark for text, image, and audio modalities."""
    _validate_complexity_band(complexity_min, complexity_max)
    rng = random.Random(seed)
    strata = _reachable_multimodal_strata(complexity_min, complexity_max)
    if not strata:
        raise ValueError(
            f"No feasible multimodal strata in complexity band [{complexity_min}, {complexity_max}]."
        )

    complexities, complexity_to_strata = _multimodal_complexity_plan(
        complexity_min,
        complexity_max,
    )
    complexity_quotas = _allocate_multimodal_complexity_quotas(
        count,
        complexity_min,
        complexity_max,
    )

    quotas: Dict[Tuple[int, int], int] = {}
    for complexity, grouped_strata in complexity_to_strata.items():
        grouped_capacities = {
            stratum: _compute_stratum_capacity(*stratum) for stratum in grouped_strata
        }
        grouped_weights = {stratum: 1.0 for stratum in grouped_strata}
        quotas.update(
            _allocate_weighted_quotas_with_capacities(
                grouped_strata,
                complexity_quotas[complexity],
                grouped_capacities,
                grouped_weights,
                minimum_per_key=0,
            )
        )
    seen_keys = set()
    dataset: List[Dict[str, Any]] = []

    for digit_total, nonzero_total in sorted(
        strata,
        key=lambda stratum: (stratum[0] * stratum[1], stratum[0], stratum[1]),
    ):
        target = quotas[(digit_total, nonzero_total)]
        generated = 0
        attempts = 0
        max_attempts = max(5000, target * 50)

        while generated < target:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not fill stratum d={digit_total}, nz={nonzero_total} "
                    f"after {attempts} attempts"
                )
            attempts += 1

            a, b, template_a, template_b = _build_stratified_problem(
                digit_total=digit_total,
                nonzero_total=nonzero_total,
                rng=rng,
            )
            key = canonical_problem_key(a, b)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            stats = compute_problem_stats(a, b)
            row = {
                "a": a,
                "b": b,
                "a_times_b": a * b,
                "template_a": template_a,
                "template_b": template_b,
                **stats,
            }
            dataset.append(row)
            generated += 1

    shuffle_rng = random.Random(seed + 1)
    shuffle_rng.shuffle(dataset)

    for idx, row in enumerate(dataset, start=1):
        row["id"] = f"{MULTIMODAL_ID_PREFIX}_{idx:0{MULTIMODAL_ID_WIDTH}d}"

    if include_splits:
        ratios = split_ratios if split_ratios else DEFAULT_RATIOS.copy()
        dataset = assign_splits(
            dataset,
            ratios=ratios,
            id_key="id",
            stratify_key="complexity_c",
            seed=seed,
        )

    return dataset


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Load all rows from a CSV file as dictionaries."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _is_matching_saved_dataset(
    path: Path,
    count: int,
    complexity_min: int,
    complexity_max: int,
) -> bool:
    """Return True when an existing CSV already satisfies the requested size."""
    if not path.exists():
        return False
    rows = load_csv_rows(path)
    if len(rows) != count:
        return False
    try:
        complexities = {int(row["complexity_c"]) for row in rows}
    except (KeyError, TypeError, ValueError):
        return False

    expected_quotas = _allocate_multimodal_complexity_quotas(
        count,
        complexity_min,
        complexity_max,
    )
    expected_complexities = set(expected_quotas)
    if not complexities.issubset(expected_complexities):
        return False
    if complexities != expected_complexities:
        return False
    actual_quotas: Dict[int, int] = {}
    for row in rows:
        actual_quotas[int(row["complexity_c"])] = actual_quotas.get(int(row["complexity_c"]), 0) + 1
    if actual_quotas != expected_quotas:
        return False

    split_bands: Dict[str, set[str]] = {}
    for row in rows:
        split = row.get("split")
        if split:
            band = canonical_complexity_band(int(row["complexity_c"]))
            if band is None:
                continue
            split_bands.setdefault(split, set()).add(canonical_band_label(*band))
    if split_bands:
        required_splits = {"train", "val", "test"}
        if set(split_bands) != required_splits:
            return False
        required_bands = {
            canonical_band_label(lo, hi)
            for lo, hi in CANONICAL_COMPLEXITY_BANDS
            if any(lo <= complexity <= hi for complexity in expected_complexities)
        }
        if any(split_bands[split] != required_bands for split in required_splits):
            return False

    return True


def _acquire_lock(lock_path: Path) -> int:
    """Acquire an exclusive lock file, waiting for concurrent writers if needed."""
    deadline = time.time() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{os.getpid()}\n")
            return fd
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
                if age_seconds > LOCK_STALE_SECONDS:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue

            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)


def get_or_create_shared_multimodal_dataset(
    count: int = MULTIMODAL_DEFAULT_COUNT,
    include_splits: bool = True,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = SPLIT_SEED,
    force_regenerate: bool = False,
    complexity_min: int = DEFAULT_COMPLEXITY_MIN,
    complexity_max: int = DEFAULT_COMPLEXITY_MAX,
) -> List[Dict[str, Any]]:
    """Load the shared paired benchmark or generate it if missing/stale."""
    ratios = split_ratios if split_ratios else DEFAULT_RATIOS.copy()
    if not force_regenerate and _is_matching_saved_dataset(
        SHARED_MULTIMODAL_CSV,
        count,
        complexity_min,
        complexity_max,
    ):
        return load_csv_rows(SHARED_MULTIMODAL_CSV)

    lock_path = SHARED_MULTIMODAL_CSV.with_suffix(f"{SHARED_MULTIMODAL_CSV.suffix}.lock")
    _acquire_lock(lock_path)
    try:
        if not force_regenerate and _is_matching_saved_dataset(
            SHARED_MULTIMODAL_CSV,
            count,
            complexity_min,
            complexity_max,
        ):
            return load_csv_rows(SHARED_MULTIMODAL_CSV)

        dataset = generate_paired_multimodal_dataset(
            count=count,
            include_splits=include_splits,
            split_ratios=ratios,
            seed=seed,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
        )
        save_csv(dataset, SHARED_MULTIMODAL_CSV)
        return dataset
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def generate_dataset(
    count: int = MULTIMODAL_DEFAULT_COUNT,
    id_prefix: str = MULTIMODAL_ID_PREFIX,
    include_splits: bool = True,
    split_ratios: Optional[Dict[str, float]] = None,
    include_templates: bool = True,
    seed: int = SPLIT_SEED,
    complexity_min: int = DEFAULT_COMPLEXITY_MIN,
    complexity_max: int = DEFAULT_COMPLEXITY_MAX,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper for the shared paired multimodal dataset."""
    del id_prefix  # IDs are now shared across modalities by design.
    dataset = generate_paired_multimodal_dataset(
        count=count,
        include_splits=include_splits,
        split_ratios=split_ratios,
        seed=seed,
        complexity_min=complexity_min,
        complexity_max=complexity_max,
    )
    if include_templates:
        return dataset
    return [
        {k: v for k, v in row.items() if not k.startswith("template_")}
        for row in dataset
    ]


def save_csv(
    dataset: List[Dict[str, Any]],
    path: Path,
    fieldnames: Optional[List[str]] = None
) -> None:
    """Save dataset to CSV file.

    Args:
        dataset: List of problem dictionaries.
        path: Output file path.
        fieldnames: Column order. If None, inferred from first row with standard order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset:
        tprint(f"Warning: Empty dataset, skipping save to {path}")
        return

    # Infer fieldnames from dataset if not provided
    if fieldnames is None:
        # Standard field order
        standard_order = [
            "id",
            "a",
            "b",
            "a_times_b",
            "template_a",
            "template_b",
            "digits_a",
            "digits_b",
            "nonzero_a",
            "nonzero_b",
            "digit_total",
            "nonzero_total",
            "complexity_c",
            "stratum_id",
            "split",
        ]
        available_fields = set(dataset[0].keys())
        fieldnames = [f for f in standard_order if f in available_fields]
        # Add any remaining fields not in standard order
        for key in dataset[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    tprint(f"Saved {len(dataset)} rows to {path}")
