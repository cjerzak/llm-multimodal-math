#!/usr/bin/env python3
"""
NudgeTaxonomy.py

Shared helpers for classifying arithmetic errors in base-vs-LoRA comparisons.
"""

from itertools import combinations
from typing import Dict, Iterable, List, Optional


def place_value_parts(n: int) -> List[int]:
    """Return the non-zero place-value parts of an integer."""
    digits = list(reversed(str(abs(n))))
    parts: List[int] = []
    for idx, ch in enumerate(digits):
        digit = int(ch)
        if digit:
            parts.append(digit * (10 ** idx))
    return parts


def simulate_no_carry_product(a: int, b: int) -> int:
    """Multiply two integers with column sums but without cross-column carries."""
    digits_a = [int(ch) for ch in reversed(str(abs(a)))]
    digits_b = [int(ch) for ch in reversed(str(abs(b)))]
    columns = [0] * (len(digits_a) + len(digits_b))

    for i, da in enumerate(digits_a):
        for j, db in enumerate(digits_b):
            columns[i + j] += da * db

    while len(columns) > 1 and columns[-1] == 0:
        columns.pop()

    return int("".join(str(value % 10) for value in reversed(columns)))


def _is_power_of_ten_ratio(x: int, y: int) -> bool:
    """Return True if x and y differ only by a power-of-ten factor."""
    if x <= 0 or y <= 0:
        return False
    smaller, larger = sorted((x, y))
    while larger > smaller and larger % 10 == 0:
        larger //= 10
    return larger == smaller


def _subset_sum_match(target: int, values: Iterable[int]) -> bool:
    """Return True if target equals a proper non-empty subset sum."""
    values_list = list(values)
    if len(values_list) < 2:
        return False
    full_sum = sum(values_list)
    for r in range(1, len(values_list)):
        for combo in combinations(values_list, r):
            if sum(combo) == target and sum(combo) != full_sum:
                return True
    return False


def classify_error_taxonomy(a: int, b: int, answer: Optional[int]) -> str:
    """Classify an arithmetic answer into a coarse error taxonomy."""
    if answer is None:
        return "unknown"

    product = a * b
    if answer == product:
        return "correct"

    if _is_power_of_ten_ratio(answer, product):
        return "magnitude_slip"

    partial_products_a = [part * b for part in place_value_parts(a)]
    partial_products_b = [part * a for part in place_value_parts(b)]
    if (
        _subset_sum_match(answer, partial_products_a)
        or _subset_sum_match(answer, partial_products_b)
        or answer in partial_products_a
        or answer in partial_products_b
    ):
        return "partial_product_omission"

    if answer == simulate_no_carry_product(a, b):
        return "carry_drop"

    if len(str(abs(answer))) == len(str(product)) and sorted(str(abs(answer))) == sorted(str(product)):
        return "digit_permutation"

    if abs(answer - product) <= max(9, product // 100):
        return "near_miss"

    return "other"


def summarize_taxonomy(labels: Iterable[str]) -> Dict[str, int]:
    """Count taxonomy labels."""
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
