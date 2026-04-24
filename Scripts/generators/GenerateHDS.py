#!/usr/bin/env python3
"""
GenerateHDS.py

Generates the Heuristic-Disagreement Set (HDS) - a probe set
where different arithmetic heuristics predict different behaviors.

Also generates adversarial "trap" sets designed to trigger characteristic
failures for each heuristic.

Usage:
    python Scripts/GenerateHDS.py --count 99
    python Scripts/GenerateHDS.py --count 99 --split-ratios 70/15/15

Output files:
- SavedData/HDSv2.csv: Core probe set with symmetric heuristic costs
- SavedData/Trapsv2.csv: Adversarial test sets targeting each heuristic
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Set, Optional

# Paths (generators/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))

from core.DatasetSplits import assign_splits, parse_split_ratios, SPLIT_SEED
from core.GenerateMathHelpers import (
    CANONICAL_BAND_MIN_FRACTION,
    _allocate_exact_c_quotas,
    compute_problem_stats,
)
from core.Logging import tprint


@dataclass
class HDSItem:
    """A single item in the Heuristic-Disagreement Set."""
    id: str
    a: int
    b: int
    product: int
    category: str  # e.g., "near_base", "zero_factor", "carry_heavy"
    notes: str
    digit_total: int
    nonzero_total: int
    complexity_c: int
    design_family: str = ""  # The construction family used to generate the item
    canonical_target_heuristic: str = ""  # The cost-model winner for the item
    canonical_target_margin: float = 0.0
    ot_cost: float = 0.0
    dd_cost: float = 0.0
    rc_cost: float = 0.0
    heuristic_definition_version: str = ""
    target_heuristic: str = ""  # Legacy alias kept for backward-compatible loaders
    ot_score: float = 0.0  # Legacy field kept for backward-compatible loaders
    dd_score: float = 0.0  # Legacy field kept for backward-compatible loaders
    rc_score: float = 0.0  # Legacy field kept for backward-compatible loaders
    split: str = ""  # train/val/test


# HDS generation configuration defaults
BASE_RC_BASES = [25, 50, 100, 200, 250, 500]
DEFAULT_MIN_VAL = 10
DEFAULT_MAX_VAL = 999_999_999
DEFAULT_MIN_DIGITS = 2
DEFAULT_MAX_DIGITS = 9
DEFAULT_RC_OFFSET = 10
DEFAULT_MIN_SCORE_GAP = 0.1
DEFAULT_TARGET_TWO_DIGIT_RATIO = 0.1
DEFAULT_COMPLEXITY_MIN = 12
DEFAULT_COMPLEXITY_MAX = 324
HDS_PROFILE_TARGET_MEAN_C = 190.0
DEFAULT_MIN_COST_MARGIN = 0.75
DEFAULT_CURATED_COST_MARGIN = 0.5
HEURISTIC_DEFINITION_VERSION_V2 = "cost_model_v2"


@dataclass(frozen=True)
class HDSScoringConfig:
    """Scoring knobs for heuristic applicability."""
    name: str
    min_score_gap: float
    curated_min_score_gap: float
    rc_both_near_dist: int
    rc_both_near_score: float
    dd_base: float
    dd_zero_bonus: float
    dd_double_zero_bonus: float
    dd_25_bonus: float
    dd_5_bonus: float
    dd_easy_five_bonus: float
    ot_base: float
    ot_unstructured_score: float
    ot_carry_threshold: float
    ot_carry_score: float
    ot_structured_cap: float
    ot_structured_rc_threshold: float
    ot_structured_dd_threshold: float


BASE_SCORING_CONFIG = HDSScoringConfig(
    name="base",
    min_score_gap=DEFAULT_MIN_SCORE_GAP,
    curated_min_score_gap=DEFAULT_MIN_SCORE_GAP,
    rc_both_near_dist=2,
    rc_both_near_score=0.55,
    dd_base=0.3,
    dd_zero_bonus=0.5,
    dd_double_zero_bonus=0.2,
    dd_25_bonus=0.2,
    dd_5_bonus=0.1,
    dd_easy_five_bonus=0.0,
    ot_base=0.45,
    ot_unstructured_score=0.8,
    ot_carry_threshold=0.5,
    ot_carry_score=0.7,
    ot_structured_cap=1.0,
    ot_structured_rc_threshold=1.1,
    ot_structured_dd_threshold=1.1
)

TUNED_SCORING_CONFIG = HDSScoringConfig(
    name="tuned_v1",
    min_score_gap=DEFAULT_MIN_SCORE_GAP,
    curated_min_score_gap=0.08,
    rc_both_near_dist=2,
    rc_both_near_score=0.7,
    dd_base=0.3,
    dd_zero_bonus=0.5,
    dd_double_zero_bonus=0.2,
    dd_25_bonus=0.35,
    dd_5_bonus=0.2,
    dd_easy_five_bonus=0.15,
    ot_base=0.45,
    ot_unstructured_score=0.8,
    ot_carry_threshold=0.5,
    ot_carry_score=0.7,
    ot_structured_cap=0.55,
    ot_structured_rc_threshold=0.7,
    ot_structured_dd_threshold=0.6
)

SCORING_CANDIDATES = [BASE_SCORING_CONFIG, TUNED_SCORING_CONFIG]


@dataclass
class HDSConfig:
    """Configuration for HDS/Traps generation."""
    min_val: int
    max_val: int
    min_digits: int
    max_digits: int
    rc_bases: List[int]
    rc_offset: int
    digit_mix: Dict[int, float]
    target_two_digit_ratio: float
    complexity_min: int
    complexity_max: int
    scoring: HDSScoringConfig


@dataclass(frozen=True)
class CuratedSpec:
    """Curated item with intended heuristic target."""
    a: int
    b: int
    expected_heuristic: str
    category: str
    notes: str


@dataclass(frozen=True)
class CuratedMismatch:
    """Mismatch between curated intent and scoring."""
    a: int
    b: int
    expected: str
    scored: Optional[str]
    scores: Dict[str, float]
    category: str


@dataclass
class CuratedAudit:
    """Audit result for curated items under a scoring config."""
    scoring_name: str
    attempts: int
    overrides_used: int
    mismatches: List[CuratedMismatch]


def digit_length(n: int) -> int:
    """Return digit length of a positive integer."""
    return len(str(abs(n)))


def digit_bounds(digits: int, min_val: int, max_val: int) -> Optional[Tuple[int, int]]:
    """Return inclusive bounds for a given digit length, clipped to [min_val, max_val]."""
    if digits <= 0:
        return None
    low = max(min_val, 10 ** (digits - 1))
    high = min(max_val, (10 ** digits) - 1)
    if low > high:
        return None
    return low, high


def in_digit_range(n: int, config: HDSConfig) -> bool:
    """Check whether n is within the configured digit range."""
    d = digit_length(n)
    return config.min_digits <= d <= config.max_digits


def random_number_with_digits(
    rng: random.Random,
    digits: int,
    min_val: int,
    max_val: int
) -> Optional[int]:
    """Sample a random integer with a specific digit length inside bounds."""
    bounds = digit_bounds(digits, min_val, max_val)
    if bounds is None:
        return None
    low, high = bounds
    return rng.randint(low, high)


def random_number_with_last_digit(
    rng: random.Random,
    digits: int,
    min_val: int,
    max_val: int,
    last_digit: int
) -> Optional[int]:
    """Sample a random integer with a specific digit length and last digit."""
    bounds = digit_bounds(digits, min_val, max_val)
    if bounds is None:
        return None
    low, high = bounds
    start = low + ((last_digit - low) % 10)
    if start > high:
        return None
    count = ((high - start) // 10) + 1
    return start + 10 * rng.randrange(count)


def random_multiple_with_digits(
    rng: random.Random,
    digits: int,
    min_val: int,
    max_val: int,
    multiple: int,
) -> Optional[int]:
    """Sample a multiple with a specific digit length inside bounds."""
    bounds = digit_bounds(digits, min_val, max_val)
    if bounds is None:
        return None
    low, high = bounds
    start = low + ((multiple - (low % multiple)) % multiple)
    if start > high:
        return None
    count = ((high - start) // multiple) + 1
    return start + multiple * rng.randrange(count)


def normalized_rc_distance(distance: int, base: int) -> float:
    """Normalize absolute base distance by the base's scale."""
    scale = 10 ** max(digit_length(base) - 2, 0)
    return distance / scale


def near_base_offsets(base: int, rc_offset: int) -> List[int]:
    """Return offset magnitudes that stay near a base across scales."""
    digits = digit_length(base)
    scale = 10 ** max(digits - 2, 0)
    offsets = {0}
    for step in range(1, rc_offset + 1):
        offsets.add(step * scale)
    repeated_width = max(1, digits - 2)
    for digit in (1, 3, 5, 7, 9):
        repeated = int(str(digit) * repeated_width)
        if repeated <= rc_offset * scale:
            offsets.add(repeated)
    return sorted(offsets)


def random_far_from_rc_base(
    rng: random.Random,
    digits: int,
    config: HDSConfig,
    min_distance: Optional[int] = None,
) -> Optional[int]:
    """Sample a number far from any RC base and without simple DD cues."""
    distance = config.rc_offset if min_distance is None else min_distance
    for _ in range(100):
        n = random_number_with_digits(rng, digits, config.min_val, config.max_val)
        if n is None:
            return None
        nearest_base, nearest_distance = nearest_base_info(n, config.rc_bases)
        if normalized_rc_distance(nearest_distance, nearest_base) <= distance:
            continue
        if n % 5 == 0:
            continue
        return n
    return None


def normalize_digit_mix(
    digit_mix: Dict[int, float],
    valid_digits: List[int]
) -> Dict[int, float]:
    """Normalize a digit mix dict and filter to valid digits."""
    filtered = {d: w for d, w in digit_mix.items() if d in valid_digits and w > 0}
    if not filtered:
        raise ValueError("Digit mix has no valid weights for the configured digit range.")
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("Digit mix weights must sum to a positive value.")
    return {d: w / total for d, w in filtered.items()}


def parse_digit_mix(value: str) -> Dict[int, float]:
    """Parse digit mix string like '2:0.1,3:0.9' or '2=0.1,3=0.9'."""
    mix: Dict[int, float] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, val = part.split(":", 1)
        elif "=" in part:
            key, val = part.split("=", 1)
        else:
            raise ValueError(f"Invalid digit-mix entry '{part}'. Use 'digits:weight'.")
        mix[int(key.strip())] = float(val.strip())
    if not mix:
        raise ValueError("Digit mix string is empty.")
    return mix


def build_default_digit_mix(
    valid_digits: List[int],
    target_two_digit_ratio: float
) -> Dict[int, float]:
    """Build a default digit mix that preserves legacy 2/3-digit ratio."""
    if len(valid_digits) == 1:
        return {valid_digits[0]: 1.0}
    if valid_digits == [2, 3]:
        two_ratio = max(0.0, min(1.0, target_two_digit_ratio))
        return {2: two_ratio, 3: 1.0 - two_ratio}
    # Uniform mix across available digits
    weight = 1.0 / len(valid_digits)
    return {d: weight for d in valid_digits}


def get_curated_specs() -> List[CuratedSpec]:
    """Return the curated (hand-crafted) HDS items with expected targets."""
    specs: List[CuratedSpec] = []

    rc_problems = [
        # Near 50
        (49, 51, "near_50_symmetric", "49×51 = 50²-1 = 2499"),
        (48, 52, "near_50_2off", "Both 2 away from 50"),
        (47, 53, "near_50_3off", "Control: 3 away from 50"),
        # Near 100
        (99, 101, "near_100_symmetric", "99×101 = 100²-1 = 9999"),
        (98, 102, "near_100_2off", "Both 2 away from 100"),
        (97, 103, "near_100_3off", "Control: 3 away from 100"),
        # Near 200
        (199, 201, "near_200_symmetric", "199×201 = 200²-1"),
        (198, 202, "near_200_2off", "Both 2 away from 200"),
        # Mixed near-base
        (49, 102, "near_50_and_100", "One near 50, one near 100"),
        (51, 98, "near_50_and_100_v2", "Variant"),
    ]
    for a, b, category, notes in rc_problems:
        specs.append(CuratedSpec(a=a, b=b, expected_heuristic="RC", category=category, notes=notes))

    dd_problems = [
        # Trailing zeros
        (47, 60, "zero_factor", "47×60 = 47×6×10"),
        (73, 40, "zero_factor", "73×40 = 73×4×10"),
        (58, 30, "zero_factor", "58×30 = 58×3×10"),
        (62, 50, "zero_factor_50", "62×50 = 62×5×10"),
        (84, 20, "zero_factor", "84×20 = 84×2×10"),
        # Double zeros
        (37, 100, "hundred_factor", "37×100 = 3700"),
        (64, 200, "two_hundred_factor", "64×200"),
        # Multiple of 25
        (48, 25, "quarter_hundred", "48×25 = 48×100÷4 = 1200"),
        (76, 25, "quarter_hundred", "76×25 = 76×100÷4 = 1900"),
        # Clean tens decomposition
        (45, 36, "clean_decomp", "45×36 = (40+5)×36"),
        (55, 44, "clean_decomp", "55×44 = 55×(40+4)"),
    ]
    for a, b, category, notes in dd_problems:
        specs.append(CuratedSpec(a=a, b=b, expected_heuristic="DD", category=category, notes=notes))

    ot_problems = [
        # Heavy carries
        (87, 96, "carry_heavy", "Multiple carries expected"),
        (79, 68, "carry_heavy", "7×8=56, 9×6=54, etc."),
        (58, 76, "carry_heavy", "8×6=48, 5×7+4=39"),
        (89, 47, "carry_heavy", "9×7=63, 8×4+6=38"),
        (67, 83, "carry_heavy", "7×3=21, 6×8+2=50"),
        (78, 94, "carry_heavy", "8×4=32, 7×9+3=66"),
        (69, 57, "carry_heavy", "9×7=63, 6×5+6=36"),
        # No special structure
        (43, 67, "generic", "No special pattern"),
        (56, 38, "generic", "No special pattern"),
        (72, 49, "generic", "49 near 50 but 72 isn't"),
    ]
    for a, b, category, notes in ot_problems:
        specs.append(CuratedSpec(a=a, b=b, expected_heuristic="OT", category=category, notes=notes))

    perturbation_pairs = [
        # Near-base perturbation
        (46, 54, "RC", "perturb_base", "Near 50 - should favor RC"),
        (44, 56, "OT", "perturb_control", "Control: farther from base"),
        (45, 55, "RC", "perturb_base", "Balanced around 50"),
        # Zero perturbation
        (73, 60, "DD", "perturb_zero", "Has zero - should favor DD"),
        (73, 59, "OT", "perturb_control", "Control: no zero"),
        (73, 61, "OT", "perturb_control", "Control: no zero"),
    ]
    for a, b, expected, category, notes in perturbation_pairs:
        specs.append(CuratedSpec(a=a, b=b, expected_heuristic=expected, category=category, notes=notes))

    return specs


def choose_digit_length(rng: random.Random, digit_mix: Dict[int, float]) -> int:
    """Sample a digit length from a normalized digit mix."""
    roll = rng.random()
    cumulative = 0.0
    for d in sorted(digit_mix):
        cumulative += digit_mix[d]
        if roll <= cumulative:
            return d
    return sorted(digit_mix)[-1]


def scale_rc_bases(
    base_bases: List[int],
    min_digits: int,
    max_digits: int,
    min_val: int,
    max_val: int
) -> List[int]:
    """Scale RC bases across digit lengths and clip to range."""
    scaled: Set[int] = set()
    for digits in range(min_digits, max_digits + 1):
        scale = 10 ** max(digits - 2, 0)
        for base in base_bases:
            val = base * scale
            if min_val <= val <= max_val:
                scaled.add(val)
    return sorted(scaled)


def canonical_pair(a: int, b: int) -> Tuple[int, int]:
    """Canonicalize a pair to handle commutativity."""
    return (a, b) if a <= b else (b, a)


def nearest_base_info(n: int, rc_bases: List[int]) -> Tuple[int, int]:
    """Return (nearest_base, distance) for RC scoring."""
    base = min(rc_bases, key=lambda c: abs(n - c))
    return base, abs(n - base)


def select_target(scores: Dict[str, float], min_gap: float = DEFAULT_MIN_SCORE_GAP) -> Optional[str]:
    """Select target heuristic if the top score is separated by a margin."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        return None
    top, runner_up = ranked[0], ranked[1]
    if top[1] - runner_up[1] < min_gap:
        return None
    return top[0].upper()


def trailing_zero_count(n: int) -> int:
    """Return the number of trailing zeros in a positive integer."""
    count = 0
    while n and n % 10 == 0:
        count += 1
        n //= 10
    return count


def place_value_terms(n: int) -> List[int]:
    """Return non-zero place-value terms for a positive integer."""
    terms: List[int] = []
    for power, digit_char in enumerate(reversed(str(abs(n)))):
        digit = int(digit_char)
        if digit:
            terms.append(digit * (10 ** power))
    return terms


def nonzero_term_count(n: int) -> int:
    """Return the number of non-zero place-value terms."""
    return len(place_value_terms(n))


def count_carry_pairs(a: int, b: int) -> int:
    """Count digit-pair products that induce a carry in schoolbook multiplication."""
    digits_a = [int(d) for d in str(abs(a))]
    digits_b = [int(d) for d in str(abs(b))]
    return sum(1 for da in digits_a for db in digits_b if da * db >= 10)


def dd_orientation_cost(factor: int, other: int) -> float:
    """Estimate decomposition cost when expanding `factor` against `other`."""
    factor_terms = max(nonzero_term_count(factor), 1)
    generic_cost = factor_terms * digit_length(other) + max(factor_terms - 1, 0)

    special_costs = [generic_cost]

    trailing_zeros = trailing_zero_count(factor)
    if trailing_zeros:
        stripped = factor // (10 ** trailing_zeros)
        stripped_terms = max(nonzero_term_count(stripped), 1)
        special_costs.append(stripped_terms * digit_length(other) + max(stripped_terms - 1, 0) + 0.25)

    if factor % 25 == 0:
        reduced = factor // 25
        reduced_terms = max(nonzero_term_count(reduced), 1)
        special_costs.append(reduced_terms * digit_length(other) + max(reduced_terms - 1, 0) + 1.0)
    elif factor % 5 == 0:
        reduced = factor // 5
        reduced_terms = max(nonzero_term_count(reduced), 1)
        special_costs.append(reduced_terms * digit_length(other) + max(reduced_terms - 1, 0) + 1.5)

    return min(special_costs)


def compute_ot_cost(a: int, b: int) -> float:
    """Estimate schoolbook column-multiplication cost."""
    digits_a = digit_length(a)
    digits_b = digit_length(b)
    total_pairs = max(digits_a * digits_b, 1)
    carry_penalty = count_carry_pairs(a, b) / total_pairs
    return (digits_a * digits_b) + (0.25 * carry_penalty)


def compute_dd_cost(a: int, b: int) -> float:
    """Estimate one-sided decomposition cost."""
    return min(dd_orientation_cost(a, b), dd_orientation_cost(b, a))


def compute_rc_cost(a: int, b: int, rc_bases: List[int]) -> float:
    """Estimate round-and-correct cost using the best shared round base."""
    best_cost = math.inf
    for base in rc_bases:
        offset_a = a - base
        offset_b = b - base
        distance_penalty = (
            normalized_rc_distance(abs(offset_a), base)
            + normalized_rc_distance(abs(offset_b), base)
        ) / 5.0
        correction_cost = 1.0
        if offset_a + offset_b != 0:
            correction_cost += digit_length(abs(offset_a + offset_b))
        if offset_a != 0 and offset_b != 0:
            correction_cost += digit_length(abs(offset_a)) * digit_length(abs(offset_b))
        best_cost = min(best_cost, correction_cost + distance_penalty)
    return best_cost


def compute_heuristic_costs(a: int, b: int, config: HDSConfig) -> Dict[str, float]:
    """Compute symmetric cost estimates for OT/DD/RC."""
    return {
        "ot": compute_ot_cost(a, b),
        "dd": compute_dd_cost(a, b),
        "rc": compute_rc_cost(a, b, config.rc_bases),
    }


def select_target_from_costs(
    costs: Dict[str, float],
    min_margin: float = DEFAULT_MIN_COST_MARGIN,
) -> Tuple[Optional[str], float]:
    """Select the minimum-cost heuristic if it is separated by a clear margin."""
    ranked = sorted(costs.items(), key=lambda kv: kv[1])
    if len(ranked) < 2:
        return None, 0.0
    best, runner_up = ranked[0], ranked[1]
    margin = runner_up[1] - best[1]
    if margin < min_margin:
        return None, margin
    return best[0].upper(), margin


def max_pair_digits(a: int, b: int) -> int:
    """Return the max digit length across a pair."""
    return max(digit_length(a), digit_length(b))


def complexity_in_band(a: int, b: int, config: HDSConfig) -> bool:
    """Check whether a problem falls within the configured complexity band."""
    stats = compute_problem_stats(a, b)
    return config.complexity_min <= stats["complexity_c"] <= config.complexity_max


def select_complexity_weighted(
    candidates: List[HDSItem],
    count: int,
    rng: random.Random,
) -> List[HDSItem]:
    """Select items without replacement, weighted toward higher exact C values."""
    if count > len(candidates):
        raise ValueError(f"Need {count} candidates, only {len(candidates)} available.")
    keyed = []
    for item in candidates:
        weight = max(float(item.complexity_c), 1.0)
        key = math.log(rng.random()) / weight
        keyed.append((key, item))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keyed[:count]]


def select_hard_profile_candidates(
    candidates: List[HDSItem],
    count: int,
    rng: random.Random,
) -> List[HDSItem]:
    """Select candidates to match the canonical hard complexity profile."""
    if count > len(candidates):
        raise ValueError(f"Need {count} candidates, only {len(candidates)} available.")

    by_complexity: Dict[int, List[HDSItem]] = {}
    for item in candidates:
        by_complexity.setdefault(item.complexity_c, []).append(item)

    complexities = sorted(by_complexity)
    capacities = {complexity: len(items) for complexity, items in by_complexity.items()}
    quotas = _allocate_exact_c_quotas(
        count,
        complexities,
        capacities,
        target_mean=HDS_PROFILE_TARGET_MEAN_C,
        minimum_per_key=0,
        band_min_fraction=CANONICAL_BAND_MIN_FRACTION,
    )

    selected: List[HDSItem] = []
    for complexity in complexities:
        quota = quotas.get(complexity, 0)
        if quota <= 0:
            continue
        bucket = list(by_complexity[complexity])
        rng.shuffle(bucket)
        selected.extend(bucket[:quota])

    if len(selected) != count:
        raise ValueError(f"Selected {len(selected)} HDS candidates, expected {count}.")
    return selected


def select_length_balanced(
    candidates: List[HDSItem],
    count: int,
    rng: random.Random,
    digit_mix: Dict[int, float]
) -> List[HDSItem]:
    """Select candidates with a target mix of digit lengths."""
    buckets: Dict[int, List[HDSItem]] = {d: [] for d in digit_mix}
    for item in candidates:
        digits = max_pair_digits(item.a, item.b)
        if digits in buckets:
            buckets[digits].append(item)

    for items in buckets.values():
        rng.shuffle(items)

    raw_targets = {d: count * digit_mix[d] for d in digit_mix}
    targets = {d: int(round(raw_targets[d])) for d in digit_mix}
    diff = count - sum(targets.values())
    if diff != 0:
        frac_order = sorted(
            digit_mix,
            key=lambda d: raw_targets[d] - int(raw_targets[d]),
            reverse=(diff > 0)
        )
        for d in frac_order:
            if diff == 0:
                break
            if diff < 0 and targets[d] == 0:
                continue
            targets[d] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    selected: List[HDSItem] = []
    for d, items in buckets.items():
        take = min(targets.get(d, 0), len(items))
        selected.extend(items[:take])

    if len(selected) < count:
        remainder = []
        for d, items in buckets.items():
            remainder.extend(items[targets.get(d, 0):])
        rng.shuffle(remainder)
        selected.extend(remainder[:count - len(selected)])

    return selected[:count]


def compute_heuristic_scores(a: int, b: int, config: HDSConfig) -> Dict[str, float]:
    """
    Compute applicability scores for each heuristic.

    Higher score = this heuristic is more applicable/natural for the problem.
    """
    scoring = config.scoring
    # RC score: proximity to a shared round base (or one near-base operand)
    base_a, dist_a = nearest_base_info(a, config.rc_bases)
    base_b, dist_b = nearest_base_info(b, config.rc_bases)
    scaled_dist_a = normalized_rc_distance(dist_a, base_a)
    scaled_dist_b = normalized_rc_distance(dist_b, base_b)
    same_base = base_a == base_b
    max_dist = max(scaled_dist_a, scaled_dist_b)
    min_dist = min(scaled_dist_a, scaled_dist_b)

    if same_base and max_dist == 0:
        rc_score = 1.0
    elif same_base and max_dist <= 2:
        rc_score = 0.9
    elif same_base and max_dist <= 5:
        rc_score = 0.75
    elif min_dist <= 2:
        rc_score = 0.55
    elif min_dist <= 5:
        rc_score = 0.35
    else:
        rc_score = 0.1

    if scaled_dist_a <= scoring.rc_both_near_dist and scaled_dist_b <= scoring.rc_both_near_dist:
        rc_score = max(rc_score, scoring.rc_both_near_score)

    # DD score: based on presence of zeros / clean decomposition
    dd_score = scoring.dd_base
    if a % 10 == 0 or b % 10 == 0:
        dd_score += scoring.dd_zero_bonus
    if a % 100 == 0 or b % 100 == 0:
        dd_score += scoring.dd_double_zero_bonus
    # Multiples of 25 are particularly easy to decompose (quarter-hundred trick)
    if a % 25 == 0 or b % 25 == 0:
        dd_score += scoring.dd_25_bonus
    # Also good if one factor is a multiple of 5
    if a % 5 == 0 or b % 5 == 0:
        dd_score += scoring.dd_5_bonus
    if a % 10 == 5 or b % 10 == 5:
        dd_score += scoring.dd_easy_five_bonus
    dd_score = min(1.0, dd_score)

    # OT score: always applicable, but favored when others aren't
    # Also higher for problems with many carries
    ot_score = scoring.ot_base

    # OT is more "natural" when problem doesn't favor DD or RC
    if rc_score < 0.5 and dd_score < 0.5:
        ot_score = scoring.ot_unstructured_score

    # Count expected carries via digit-pair products
    digits_a = [int(d) for d in str(a)]
    digits_b = [int(d) for d in str(b)]
    carry_pairs = sum(1 for da in digits_a for db in digits_b if da * db >= 10)
    total_pairs = len(digits_a) * len(digits_b)
    carry_ratio = carry_pairs / total_pairs if total_pairs else 0.0
    if carry_ratio >= scoring.ot_carry_threshold:
        ot_score = max(ot_score, scoring.ot_carry_score)
    if rc_score >= scoring.ot_structured_rc_threshold or dd_score >= scoring.ot_structured_dd_threshold:
        ot_score = min(ot_score, scoring.ot_structured_cap)

    return {"ot": ot_score, "dd": dd_score, "rc": rc_score}


def build_hds_item(
    a: int,
    b: int,
    category: str,
    notes: str,
    config: HDSConfig,
    min_gap_override: Optional[float] = None,
    design_family: str = "",
) -> Optional[HDSItem]:
    """Build an HDS item with symmetric costs and a clear canonical target heuristic."""
    if not (config.min_val <= a <= config.max_val and config.min_val <= b <= config.max_val):
        return None
    if not (in_digit_range(a, config) and in_digit_range(b, config)):
        return None
    stats = compute_problem_stats(a, b)
    if not (config.complexity_min <= stats["complexity_c"] <= config.complexity_max):
        return None
    costs = compute_heuristic_costs(a, b, config)
    min_gap = DEFAULT_MIN_COST_MARGIN if min_gap_override is None else min_gap_override
    target, margin = select_target_from_costs(costs, min_margin=min_gap)
    if not target:
        return None
    return HDSItem(
        id="",
        a=a,
        b=b,
        product=a * b,
        category=category,
        notes=notes,
        digit_total=int(stats["digit_total"]),
        nonzero_total=int(stats["nonzero_total"]),
        complexity_c=int(stats["complexity_c"]),
        design_family=design_family or target,
        canonical_target_heuristic=target,
        canonical_target_margin=margin,
        ot_cost=costs["ot"],
        dd_cost=costs["dd"],
        rc_cost=costs["rc"],
        heuristic_definition_version=HEURISTIC_DEFINITION_VERSION_V2,
        target_heuristic=target,
    )


def build_curated_items(
    curated_specs: List[CuratedSpec],
    config: HDSConfig,
    allow_override: bool
) -> Tuple[List[HDSItem], List[CuratedMismatch], int]:
    """Build curated items and report mismatches between intent and scoring."""
    items: List[HDSItem] = []
    mismatches: List[CuratedMismatch] = []
    overrides = 0
    seen_pairs: Set[Tuple[int, int]] = set()

    for spec in curated_specs:
        pair = canonical_pair(spec.a, spec.b)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if not (config.min_val <= spec.a <= config.max_val and config.min_val <= spec.b <= config.max_val):
            mismatches.append(CuratedMismatch(
                a=spec.a,
                b=spec.b,
                expected=spec.expected_heuristic,
                scored=None,
                scores={},
                category=spec.category
            ))
            continue
        if not (in_digit_range(spec.a, config) and in_digit_range(spec.b, config)):
            mismatches.append(CuratedMismatch(
                a=spec.a,
                b=spec.b,
                expected=spec.expected_heuristic,
                scored=None,
                scores={},
                category=spec.category
            ))
            continue
        if not complexity_in_band(spec.a, spec.b, config):
            mismatches.append(CuratedMismatch(
                a=spec.a,
                b=spec.b,
                expected=spec.expected_heuristic,
                scored=None,
                scores={},
                category=spec.category
            ))
            continue

        stats = compute_problem_stats(spec.a, spec.b)
        costs = compute_heuristic_costs(spec.a, spec.b, config)
        target, _ = select_target_from_costs(costs, min_margin=DEFAULT_CURATED_COST_MARGIN)
        if target != spec.expected_heuristic:
            mismatches.append(CuratedMismatch(
                a=spec.a,
                b=spec.b,
                expected=spec.expected_heuristic,
                scored=target,
                scores=costs,
                category=spec.category
            ))
            if not allow_override:
                continue
            scored_label = target if target is not None else "NONE"
            notes = (
                f"{spec.notes} | curated_override expected {spec.expected_heuristic}, "
                f"scored {scored_label}"
            )
            target = spec.expected_heuristic
            overrides += 1
        else:
            notes = spec.notes

        items.append(HDSItem(
            id="",
            a=spec.a,
            b=spec.b,
            product=spec.a * spec.b,
            category=spec.category,
            notes=notes,
            digit_total=int(stats["digit_total"]),
            nonzero_total=int(stats["nonzero_total"]),
            complexity_c=int(stats["complexity_c"]),
            design_family=spec.expected_heuristic,
            canonical_target_heuristic=target,
            canonical_target_margin=0.0,
            ot_cost=costs["ot"],
            dd_cost=costs["dd"],
            rc_cost=costs["rc"],
            heuristic_definition_version=HEURISTIC_DEFINITION_VERSION_V2,
            target_heuristic=target,
        ))

    return items, mismatches, overrides


def prepare_curated_items(
    config: HDSConfig,
    curated_specs: List[CuratedSpec]
) -> Tuple[HDSConfig, List[HDSItem], CuratedAudit]:
    """Audit curated items under the symmetric cost model without forcing labels."""
    items, mismatches, overrides = build_curated_items(
        curated_specs,
        config,
        allow_override=False
    )
    audit = CuratedAudit(
        scoring_name=HEURISTIC_DEFINITION_VERSION_V2,
        attempts=1,
        overrides_used=overrides,
        mismatches=mismatches,
    )
    return config, items, audit


def print_curated_audit(audit: CuratedAudit, total_curated: int):
    """Print curated scoring audit summary."""
    tprint(f"  Curated scoring: {audit.scoring_name} (attempts: {audit.attempts})")
    if audit.overrides_used:
        tprint(f"  Curated overrides applied: {audit.overrides_used}/{total_curated}")
        for mismatch in audit.mismatches:
            scored = mismatch.scored if mismatch.scored is not None else "NONE"
            tprint(
                f"    {mismatch.a}x{mismatch.b}: expected {mismatch.expected}, "
                f"scored {scored}, scores {mismatch.scores}"
            )
    elif audit.mismatches:
        tprint(f"  Curated mismatches remaining: {len(audit.mismatches)}/{total_curated}")


def generate_hds(config: HDSConfig) -> List[HDSItem]:
    """Generate the core Heuristic-Disagreement Set."""
    curated_specs = get_curated_specs()
    config, items, audit = prepare_curated_items(config, curated_specs)
    for idx, item in enumerate(items):
        item.id = f"hds_{idx:03d}"
    print_curated_audit(audit, len(curated_specs))
    return items


def generate_additional_rc_problems(
    count: int,
    existing: Set[Tuple[int, int]],
    seed: int,
    config: HDSConfig,
    pool_target: Optional[int] = None,
) -> List[HDSItem]:
    """
    Generate additional RC-favored problems (near round bases).

    Returns list of HDSItem.
    """
    rng = random.Random(seed)
    candidates: List[HDSItem] = []
    seen: Set[Tuple[int, int]] = set()

    # Symmetric near-base pairs
    for base in config.rc_bases:
        offsets = near_base_offsets(base, config.rc_offset)
        signed_offsets = sorted({0, *offsets, *(-offset for offset in offsets)})
        for offset1 in signed_offsets:
            for offset2 in signed_offsets:
                a = base + offset1
                b = base + offset2
                if not (config.min_val <= a <= config.max_val and config.min_val <= b <= config.max_val):
                    continue
                pair = canonical_pair(a, b)
                if pair in existing or pair in seen:
                    continue
                item = build_hds_item(
                    a,
                    b,
                    f"near_{base}",
                    f"Generated: {a} near {base}, {b} near {base}",
                    config,
                    design_family="RC",
                )
                if not item or item.canonical_target_heuristic != "RC":
                    continue
                candidates.append(item)
                seen.add(pair)

    # Mixed near-base pairs (one operand near base, one far)
    bases_by_digits: Dict[int, List[int]] = {d: [] for d in config.digit_mix}
    for base in config.rc_bases:
        d = digit_length(base)
        if d in bases_by_digits:
            bases_by_digits[d].append(base)
    required_pool = max(count, pool_target or count)
    attempts = 0
    max_attempts = required_pool * 10000
    while len(candidates) < required_pool and attempts < max_attempts:
        attempts += 1
        base_digits = choose_digit_length(rng, config.digit_mix)
        base_pool = bases_by_digits.get(base_digits) or config.rc_bases
        if not base_pool:
            continue
        base = rng.choice(base_pool)
        offset = rng.randint(-config.rc_offset, config.rc_offset)
        a = base + offset
        if not (config.min_val <= a <= config.max_val) or not in_digit_range(a, config):
            continue
        far_digits = choose_digit_length(rng, config.digit_mix)
        b = random_far_from_rc_base(rng, far_digits, config)
        if b is None:
            continue
        pair = canonical_pair(a, b)
        if pair in existing or pair in seen:
            continue
        item = build_hds_item(
            a,
            b,
            f"near_{base}_mixed",
            f"Generated: {a} near {base}, {b} far from base",
            config,
            design_family="RC",
        )
        if not item or item.canonical_target_heuristic != "RC":
            continue
        candidates.append(item)
        seen.add(pair)

    if len(candidates) < count:
        raise ValueError(f"RC candidate pool too small ({len(candidates)} < {count}).")
    return candidates


def generate_additional_dd_problems(
    count: int,
    existing: Set[Tuple[int, int]],
    seed: int,
    config: HDSConfig,
    pool_target: Optional[int] = None,
) -> List[HDSItem]:
    """
    Generate additional DD-favored problems (clean decomposition).

    Returns list of HDSItem.
    """
    rng = random.Random(seed + 1)  # Different seed for variety
    candidates: List[HDSItem] = []
    seen: Set[Tuple[int, int]] = set()

    patterns = ["trailing_zero", "double_zero", "multiple_of_25", "clean_tens"]

    required_pool = max(count, pool_target or count)
    attempts = 0
    max_attempts = required_pool * 200
    while len(candidates) < required_pool and attempts < max_attempts:
        attempts += 1
        pattern = rng.choice(patterns)
        digits_a = choose_digit_length(rng, config.digit_mix)
        digits_b = choose_digit_length(rng, config.digit_mix)

        if pattern == "trailing_zero":
            a = random_number_with_digits(rng, digits_a, config.min_val, config.max_val)
            b = random_multiple_with_digits(rng, digits_b, config.min_val, config.max_val, multiple=10)
            category = "zero_factor"
            notes = f"Generated: {a}*{b} = {a}*{b//10}*10"

        elif pattern == "double_zero":
            a = random_number_with_digits(rng, digits_a, config.min_val, config.max_val)
            b = random_multiple_with_digits(rng, digits_b, config.min_val, config.max_val, multiple=100)
            category = "hundred_factor"
            notes = f"Generated: {a}*{b}"

        elif pattern == "multiple_of_25":
            a = random_number_with_digits(rng, digits_a, config.min_val, config.max_val)
            b = random_multiple_with_digits(rng, digits_b, config.min_val, config.max_val, multiple=25)
            category = "quarter_hundred"
            notes = f"Generated: {a}*{b} uses quarter-hundred decomposition"

        else:  # clean_tens
            a = random_number_with_last_digit(
                rng,
                digits_a,
                config.min_val,
                config.max_val,
                last_digit=5
            )
            b = random_number_with_digits(rng, digits_b, config.min_val, config.max_val)
            category = "clean_decomp"
            notes = f"Generated: {a} = {(a//10)*10}+{a%10}"

        if a is None or b is None:
            continue
        if not (config.min_val <= a <= config.max_val and config.min_val <= b <= config.max_val):
            continue

        pair = canonical_pair(a, b)
        if pair in existing or pair in seen:
            continue

        item = build_hds_item(a, b, category, notes, config, design_family="DD")
        if not item or item.canonical_target_heuristic != "DD":
            continue

        candidates.append(item)
        seen.add(pair)

    if len(candidates) < count:
        raise ValueError(f"DD candidate pool too small ({len(candidates)} < {count}).")
    return candidates


def generate_additional_ot_problems(
    count: int,
    existing: Set[Tuple[int, int]],
    seed: int,
    config: HDSConfig,
    pool_target: Optional[int] = None,
) -> List[HDSItem]:
    """
    Generate additional OT-favored problems (carry-heavy, no special structure).

    Returns list of HDSItem.
    """
    rng = random.Random(seed + 2)  # Different seed for variety
    candidates: List[HDSItem] = []
    seen: Set[Tuple[int, int]] = set()

    # High-digit pairs that cause carries
    high_digits = [6, 7, 8, 9]

    def random_high_digit_number(length: int) -> int:
        digits = [rng.choice(high_digits) for _ in range(length)]
        return int("".join(str(d) for d in digits))

    required_pool = max(count, pool_target or count)
    attempts = 0
    max_attempts = required_pool * 200
    while len(candidates) < required_pool and attempts < max_attempts:
        attempts += 1
        pattern = rng.choice(["carry_heavy", "generic"])

        if pattern == "carry_heavy":
            digits_a = choose_digit_length(rng, config.digit_mix)
            digits_b = choose_digit_length(rng, config.digit_mix)
            a = random_high_digit_number(digits_a)
            b = random_high_digit_number(digits_b)
            category = "carry_heavy"
            notes = "Generated: multiple carries expected"

        else:  # generic
            digits_a = choose_digit_length(rng, config.digit_mix)
            digits_b = choose_digit_length(rng, config.digit_mix)
            a = random_number_with_digits(rng, digits_a, config.min_val, config.max_val)
            b = random_number_with_digits(rng, digits_b, config.min_val, config.max_val)
            # Avoid numbers near special bases or obvious DD cues
            if a is None or b is None:
                continue
            base_a, dist_a = nearest_base_info(a, config.rc_bases)
            base_b, dist_b = nearest_base_info(b, config.rc_bases)
            if normalized_rc_distance(dist_a, base_a) <= 3 or normalized_rc_distance(dist_b, base_b) <= 3:
                continue
            if a % 5 == 0 or b % 5 == 0:
                continue
            category = "generic"
            notes = "Generated: no special pattern"

        if not (config.min_val <= a <= config.max_val and config.min_val <= b <= config.max_val):
            continue

        pair = canonical_pair(a, b)
        if pair in existing or pair in seen:
            continue

        item = build_hds_item(a, b, category, notes, config, design_family="OT")
        if not item or item.canonical_target_heuristic != "OT":
            continue

        candidates.append(item)
        seen.add(pair)

    if len(candidates) < count:
        raise ValueError(f"OT candidate pool too small ({len(candidates)} < {count}).")
    return candidates


def generate_scaled_hds(
    target_count: int = 99,
    seed: int = SPLIT_SEED,
    config: Optional[HDSConfig] = None
) -> List[HDSItem]:
    """
    Generate HDS scaled to target count with balanced heuristic distribution.

    Args:
        target_count: Total number of HDS items (will be split ~equally across heuristics)
        seed: Random seed for reproducibility

    Returns:
        List of HDSItem with approximately target_count items
    """
    if config is None:
        raise ValueError("HDSConfig is required for scaled generation.")

    # Target per heuristic (divide equally, distribute remainder)
    base_target = target_count // 3
    remainder = target_count % 3
    heuristic_order = ["RC", "DD", "OT"]
    target_per_heuristic = {
        h: base_target + (1 if i < remainder else 0)
        for i, h in enumerate(heuristic_order)
    }

    rng_offsets = {"RC": 0, "DD": 1, "OT": 2}
    selected_items: List[HDSItem] = []
    existing_pairs: Set[Tuple[int, int]] = set()
    generators = {
        "RC": generate_additional_rc_problems,
        "DD": generate_additional_dd_problems,
        "OT": generate_additional_ot_problems,
    }

    for heuristic in heuristic_order:
        target = target_per_heuristic[heuristic]
        desired_pool_size = max(target * 4, target + 400)
        pool = generators[heuristic](
            target,
            existing_pairs,
            seed,
            config,
            pool_target=desired_pool_size,
        )
        if len(pool) < target:
            raise ValueError(
                f"{heuristic} pool too small after weighted expansion ({len(pool)} < {target})."
            )
        heuristic_rng = random.Random(seed + 10 + rng_offsets[heuristic])
        selected = select_hard_profile_candidates(pool, target, heuristic_rng)
        selected_items.extend(selected)
        existing_pairs.update(canonical_pair(item.a, item.b) for item in selected)

    if len(selected_items) != target_count:
        raise ValueError(f"Generated {len(selected_items)} HDS items, expected {target_count}.")

    selected_items.sort(
        key=lambda item: (
            item.canonical_target_heuristic,
            item.design_family,
            item.complexity_c,
            item.a,
            item.b,
        )
    )
    for idx, item in enumerate(selected_items):
        item.id = f"hds_{idx:03d}"

    return selected_items


@dataclass
class TrapItem:
    """An adversarial trap problem."""
    id: str
    a: int
    b: int
    product: int
    trap_type: str  # "carry_bomb", "missing_term", "anti_round"
    target_heuristic: str  # Heuristic this trap targets
    expected_error_type: str  # What kind of error we expect
    notes: str
    digit_total: int
    nonzero_total: int
    complexity_c: int
    design_family: str = ""
    canonical_target_heuristic: str = ""
    heuristic_definition_version: str = ""


def generate_traps(
    exclude_pairs: Optional[Set[Tuple[int, int]]] = None,
    seed: int = SPLIT_SEED,
    config: Optional[HDSConfig] = None
) -> List[TrapItem]:
    """Generate adversarial trap sets, disjoint from HDS."""
    if config is None:
        raise ValueError("HDSConfig is required for trap generation.")
    rng = random.Random(seed + 3)
    traps: List[TrapItem] = []
    used_pairs: Set[Tuple[int, int]] = set(exclude_pairs or set())
    trap_id = 0

    target_counts = {
        "carry_bomb": 8,
        "missing_term": 8,
        "anti_round": 8,
        "sign_trap": 6
    }
    counts = {k: 0 for k in target_counts}

    def add_trap(a: int, b: int, trap_type: str, target: str, expected_error: str, notes: str) -> bool:
        nonlocal trap_id
        if not (config.min_val <= a <= config.max_val and config.min_val <= b <= config.max_val):
            return False
        if not (in_digit_range(a, config) and in_digit_range(b, config)):
            return False
        stats = compute_problem_stats(a, b)
        if not (config.complexity_min <= stats["complexity_c"] <= config.complexity_max):
            return False
        pair = canonical_pair(a, b)
        if pair in used_pairs:
            return False
        used_pairs.add(pair)
        traps.append(TrapItem(
            id=f"trap_{trap_id:03d}",
            a=a, b=b, product=a*b,
            trap_type=trap_type,
            target_heuristic=target,
            expected_error_type=expected_error,
            notes=notes,
            digit_total=int(stats["digit_total"]),
            nonzero_total=int(stats["nonzero_total"]),
            complexity_c=int(stats["complexity_c"]),
            design_family=target,
            canonical_target_heuristic=target,
            heuristic_definition_version=HEURISTIC_DEFINITION_VERSION_V2,
        ))
        trap_id += 1
        counts[trap_type] += 1
        return True

    # OT-traps: Cascading carries (targets OT weakness)
    carry_bombs = [
        (99, 99, "Low-band dense carry chain"),
        (989, 979, "Three-digit carry cascade"),
        (8999, 7888, "Four-digit repeated carry pattern"),
        (98999, 97888, "Five-digit dense carries"),
        (899999, 788888, "Six-digit carry-heavy mix"),
        (9899999, 9788888, "Seven-digit carry cascade"),
        (89999999, 78888888, "Eight-digit dense carries"),
        (989999999, 878888888, "Nine-digit maximal carry pressure"),
    ]
    for a, b, notes in carry_bombs:
        if counts["carry_bomb"] >= target_counts["carry_bomb"]:
            break
        add_trap(a, b, "carry_bomb", "OT", "off_by_power_of_10", notes)

    # DD-traps: 4-term expansion with easy-to-drop term (targets DD weakness)
    missing_term_traps = [
        (67, 43, "Low-band four-term expansion"),
        (876, 543, "Three-digit expansion with similar partial products"),
        (7654, 4321, "Four-digit four-term decomposition"),
        (87654, 54321, "Five-digit dropped-term risk"),
        (765432, 432167, "Six-digit multi-term expansion"),
        (8765432, 5432167, "Seven-digit similar cross-terms"),
        (76543218, 43216754, "Eight-digit dropped-term temptation"),
        (876543219, 543216789, "Nine-digit full expansion trap"),
    ]
    for a, b, notes in missing_term_traps:
        if counts["missing_term"] >= target_counts["missing_term"]:
            break
        add_trap(a, b, "missing_term", "DD", "off_by_partial_product", notes)

    # RC-traps: Anti-round (far from bases, targets RC weakness)
    anti_round_traps = [
        (43, 32, "Low-band awkward pair"),
        (764, 583, "Three-digit no-base control"),
        (4876, 6132, "Four-digit anti-round pair"),
        (76438, 58321, "Five-digit awkward compensation"),
        (487631, 613247, "Six-digit far-from-base mix"),
        (7643821, 5832167, "Seven-digit anti-round control"),
        (48763124, 61324789, "Eight-digit no-clean-base pair"),
        (764382159, 583216497, "Nine-digit anti-round extreme"),
    ]
    for a, b, notes in anti_round_traps:
        if counts["anti_round"] >= target_counts["anti_round"]:
            break
        add_trap(a, b, "anti_round", "RC", "inefficient_or_wrong_base", notes)

    # RC sign traps: Near-base but asymmetric (compensation sign errors)
    sign_traps = [
        (49, 52, "Low-band opposite-sign compensation"),
        (101, 98, "Near 100 with opposite offsets"),
        (1001, 998, "Four-digit sign compensation"),
        (10001, 9998, "Five-digit asymmetric near-base"),
        (1000001, 999998, "Seven-digit compensation sign trap"),
        (109999999, 89000001, "Nine-digit high-complexity sign compensation"),
    ]
    for a, b, notes in sign_traps:
        if counts["sign_trap"] >= target_counts["sign_trap"]:
            break
        add_trap(a, b, "sign_trap", "RC", "wrong_compensation_sign", notes)

    # Fallback generators for any shortfalls (ensure disjointness)
    max_attempts = 5000

    def random_high_digit_number(length: int) -> int:
        digits = [rng.choice([7, 8, 9]) for _ in range(length)]
        return int("".join(str(d) for d in digits))

    attempts = 0
    while counts["carry_bomb"] < target_counts["carry_bomb"] and attempts < max_attempts:
        attempts += 1
        digits_a = choose_digit_length(rng, config.digit_mix)
        digits_b = choose_digit_length(rng, config.digit_mix)
        a = random_high_digit_number(digits_a)
        b = random_high_digit_number(digits_b)
        add_trap(a, b, "carry_bomb", "OT", "off_by_power_of_10", "Generated: carry-heavy digits")

    attempts = 0
    while counts["missing_term"] < target_counts["missing_term"] and attempts < max_attempts:
        attempts += 1
        digits_a = choose_digit_length(rng, config.digit_mix)
        digits_b = choose_digit_length(rng, config.digit_mix)
        a = random_number_with_digits(rng, digits_a, config.min_val, config.max_val)
        b = random_number_with_digits(rng, digits_b, config.min_val, config.max_val)
        if a is None or b is None:
            continue
        if a % 10 == 0 or b % 10 == 0:
            continue
        add_trap(a, b, "missing_term", "DD", "off_by_partial_product", "Generated: multi-term expansion")

    attempts = 0
    while counts["anti_round"] < target_counts["anti_round"] and attempts < max_attempts:
        attempts += 1
        digits_a = choose_digit_length(rng, config.digit_mix)
        digits_b = choose_digit_length(rng, config.digit_mix)
        a = random_number_with_digits(rng, digits_a, config.min_val, config.max_val)
        b = random_number_with_digits(rng, digits_b, config.min_val, config.max_val)
        if a is None or b is None:
            continue
        base_a, dist_a = nearest_base_info(a, config.rc_bases)
        base_b, dist_b = nearest_base_info(b, config.rc_bases)
        if (
            normalized_rc_distance(dist_a, base_a) <= config.rc_offset
            or normalized_rc_distance(dist_b, base_b) <= config.rc_offset
        ):
            continue
        if a % 5 == 0 or b % 5 == 0:
            continue
        add_trap(a, b, "anti_round", "RC", "inefficient_or_wrong_base", "Generated: far from base")

    attempts = 0
    while counts["sign_trap"] < target_counts["sign_trap"] and attempts < max_attempts:
        attempts += 1
        base = rng.choice(config.rc_bases)
        off1 = rng.randint(1, config.rc_offset)
        off2 = rng.randint(1, config.rc_offset)
        a = base + off1
        b = base - off2
        add_trap(a, b, "sign_trap", "RC", "wrong_compensation_sign", "Generated: asymmetric near-base")

    for trap_type, target_count in target_counts.items():
        if counts[trap_type] < target_count:
            raise ValueError(f"Trap generation shortfall for {trap_type}: {counts[trap_type]} < {target_count}")

    return traps


def save_hds(items: List[HDSItem], path: Path):
    """Save HDS to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'a', 'b', 'product',
            'design_family', 'canonical_target_heuristic', 'canonical_target_margin',
            'ot_cost', 'dd_cost', 'rc_cost', 'heuristic_definition_version',
            'target_heuristic', 'ot_score', 'dd_score', 'rc_score',
            'category', 'notes', 'digit_total', 'nonzero_total', 'complexity_c', 'split'
        ])
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))

    tprint(f"Saved {len(items)} HDS items to {path}")


def save_traps(traps: List[TrapItem], path: Path):
    """Save traps to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'a', 'b', 'product', 'trap_type',
            'design_family', 'canonical_target_heuristic', 'heuristic_definition_version',
            'target_heuristic', 'expected_error_type', 'notes',
            'digit_total', 'nonzero_total', 'complexity_c'
        ])
        writer.writeheader()
        for trap in traps:
            writer.writerow(asdict(trap))

    tprint(f"Saved {len(traps)} trap items to {path}")


def print_summary(hds: List[HDSItem], traps: Optional[List[TrapItem]] = None):
    """Print summary statistics."""
    tprint()
    tprint("=" * 60)
    tprint("HDS Summary")
    tprint("=" * 60)

    # Count by canonical target heuristic
    by_heuristic: Dict[str, int] = {}
    for item in hds:
        h = item.canonical_target_heuristic
        by_heuristic[h] = by_heuristic.get(h, 0) + 1

    tprint(f"Total HDS items: {len(hds)}")
    for h, count in sorted(by_heuristic.items()):
        tprint(f"  {h}: {count} items")

    # Count by category
    by_category: Dict[str, int] = {}
    for item in hds:
        c = item.category
        by_category[c] = by_category.get(c, 0) + 1

    tprint(f"\nBy category:")
    for c, count in sorted(by_category.items(), key=lambda x: -x[1])[:5]:
        tprint(f"  {c}: {count}")

    tprint()
    tprint("=" * 60)
    tprint("Traps Summary")
    tprint("=" * 60)

    traps = traps or []
    by_trap_type: Dict[str, int] = {}
    for trap in traps:
        t = trap.trap_type
        by_trap_type[t] = by_trap_type.get(t, 0) + 1

    tprint(f"Total trap items: {len(traps)}")
    for t, count in sorted(by_trap_type.items()):
        tprint(f"  {t}: {count} items")


def main():
    """Generate HDS and Traps datasets."""
    parser = argparse.ArgumentParser(description="Generate HDS and Traps datasets")
    parser.add_argument("--count", type=int, default=99,
                        help="Target number of HDS problems (default: 99)")
    parser.add_argument("--split-ratios", type=str, default="70/15/15",
                        help="Train/val/test split ratios (default: 70/15/15)")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED,
                        help=f"Random seed for reproducibility (default: {SPLIT_SEED})")
    parser.add_argument("--dataset-name", type=str, default="HDSv2",
                        help="Output dataset name (default: HDSv2)")
    parser.add_argument("--min-val", type=int, default=DEFAULT_MIN_VAL,
                        help=f"Minimum operand value (default: {DEFAULT_MIN_VAL})")
    parser.add_argument("--max-val", type=int, default=DEFAULT_MAX_VAL,
                        help=f"Maximum operand value (default: {DEFAULT_MAX_VAL})")
    parser.add_argument("--min-digits", type=int, default=DEFAULT_MIN_DIGITS,
                        help=f"Minimum digits per operand (default: {DEFAULT_MIN_DIGITS})")
    parser.add_argument("--max-digits", type=int, default=DEFAULT_MAX_DIGITS,
                        help=f"Maximum digits per operand (default: {DEFAULT_MAX_DIGITS})")
    parser.add_argument("--rc-bases", type=str, default="",
                        help="Comma-separated RC bases (overrides scaled defaults)")
    parser.add_argument("--rc-offset", type=int, default=DEFAULT_RC_OFFSET,
                        help=f"RC near-base offset (default: {DEFAULT_RC_OFFSET})")
    parser.add_argument("--digit-mix", type=str, default="",
                        help="Digit-length mix (e.g., '2:0.2,3:0.8')")
    parser.add_argument("--target-two-digit-ratio", type=float, default=DEFAULT_TARGET_TWO_DIGIT_RATIO,
                        help="Legacy 2-digit ratio when digits are 2-3 (default: 0.1)")
    parser.add_argument("--complexity-min", type=int, default=DEFAULT_COMPLEXITY_MIN,
                        help=f"Minimum digit complexity C to include (default: {DEFAULT_COMPLEXITY_MIN})")
    parser.add_argument("--complexity-max", type=int, default=DEFAULT_COMPLEXITY_MAX,
                        help=f"Maximum digit complexity C to include (default: {DEFAULT_COMPLEXITY_MAX})")
    parser.add_argument("--original-only", action="store_true",
                        help="Only generate original 37 hand-crafted HDS problems")
    parser.add_argument("--skip-traps", action="store_true",
                        help="Skip trap generation for custom high-difficulty probe sets")
    args = parser.parse_args()

    # Parse split ratios
    split_ratios = parse_split_ratios(args.split_ratios)

    tprint("=" * 60)
    tprint("Generating Heuristic-Disagreement Set (HDS)")
    tprint("=" * 60)
    tprint(f"  Target count: {args.count if not args.original_only else 'original-only'}")
    tprint(f"  Split ratios: {args.split_ratios}")
    tprint(f"  Seed: {args.seed}")
    tprint(f"  Range: [{args.min_val}, {args.max_val}], digits {args.min_digits}-{args.max_digits}")
    tprint(f"  Complexity band: [{args.complexity_min}, {args.complexity_max}]")
    tprint()

    if args.min_digits > args.max_digits:
        raise ValueError("min-digits must be <= max-digits.")
    if args.complexity_min < 1:
        raise ValueError("complexity-min must be >= 1.")
    if args.complexity_max < args.complexity_min:
        raise ValueError("complexity-max must be >= complexity-min.")

    valid_digits = [
        d for d in range(args.min_digits, args.max_digits + 1)
        if digit_bounds(d, args.min_val, args.max_val) is not None
    ]
    if not valid_digits:
        raise ValueError("No valid digit lengths in the provided min/max range.")

    if args.rc_bases:
        rc_bases = [int(x.strip()) for x in args.rc_bases.split(",") if x.strip()]
        rc_bases = [
            b for b in rc_bases
            if args.min_val <= b <= args.max_val
            and args.min_digits <= digit_length(b) <= args.max_digits
        ]
    else:
        rc_bases = scale_rc_bases(
            BASE_RC_BASES,
            args.min_digits,
            args.max_digits,
            args.min_val,
            args.max_val
        )
    if not rc_bases:
        raise ValueError("No RC bases available in range; provide --rc-bases.")

    if args.digit_mix:
        digit_mix = normalize_digit_mix(parse_digit_mix(args.digit_mix), valid_digits)
    else:
        digit_mix = normalize_digit_mix(
            build_default_digit_mix(valid_digits, args.target_two_digit_ratio),
            valid_digits
        )

    config = HDSConfig(
        min_val=args.min_val,
        max_val=args.max_val,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        rc_bases=rc_bases,
        rc_offset=args.rc_offset,
        digit_mix=digit_mix,
        target_two_digit_ratio=args.target_two_digit_ratio,
        complexity_min=args.complexity_min,
        complexity_max=args.complexity_max,
        scoring=TUNED_SCORING_CONFIG
    )

    # Generate HDS
    if args.original_only:
        hds = generate_hds(config)
    else:
        hds = generate_scaled_hds(target_count=args.count, seed=args.seed, config=config)

    # Apply stratified splits by canonical target heuristic
    hds_dicts = [asdict(item) for item in hds]
    hds_dicts = assign_splits(
        hds_dicts,
        ratios=split_ratios,
        id_key="id",
        stratify_key="canonical_target_heuristic",
        seed=args.seed
    )

    # Convert back to HDSItem
    hds = [HDSItem(**d) for d in hds_dicts]

    # Print split statistics
    split_counts = {}
    for item in hds:
        split = item.split
        split_counts[split] = split_counts.get(split, 0) + 1
    tprint(f"  Splits: {split_counts}")
    tprint()

    traps: List[TrapItem] = []
    if not args.skip_traps:
        # Generate traps (always test-only, no split column), disjoint from HDS
        hds_pairs = {canonical_pair(item.a, item.b) for item in hds}
        traps = generate_traps(exclude_pairs=hds_pairs, seed=args.seed, config=config)

    # Save
    dataset_name = args.dataset_name
    if dataset_name == "HDS":
        hds_path = REPO_ROOT / "SavedData" / "HDS.csv"
        traps_path = REPO_ROOT / "SavedData" / "Traps.csv"
    elif dataset_name == "HDSv2":
        hds_path = REPO_ROOT / "SavedData" / "HDSv2.csv"
        traps_path = REPO_ROOT / "SavedData" / "Trapsv2.csv"
    else:
        hds_path = REPO_ROOT / "SavedData" / f"{dataset_name}.csv"
        traps_path = REPO_ROOT / "SavedData" / f"Traps_{dataset_name}.csv"

    save_hds(hds, hds_path)
    if traps:
        save_traps(traps, traps_path)
    elif args.skip_traps:
        tprint(f"Skipped trap generation for {dataset_name}")

    # Summary
    print_summary(hds, traps)

    tprint()
    tprint("=" * 60)
    tprint("Done!")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
