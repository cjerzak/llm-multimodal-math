from __future__ import annotations

from collections import Counter

import pytest

from core.GenerateMathHelpers import (
    CANONICAL_COMPLEXITY_BANDS,
    CANONICAL_TARGET_MEAN_C,
    DEFAULT_COMPLEXITY_MAX as MULTIMODAL_COMPLEXITY_MAX,
    DEFAULT_COMPLEXITY_MIN as MULTIMODAL_COMPLEXITY_MIN,
    _reachable_complexities,
    band_counts_for_complexities,
    compute_problem_stats,
    generate_paired_multimodal_dataset,
)
from generators.GenerateHDS import (
    BASE_RC_BASES,
    DEFAULT_COMPLEXITY_MAX as HDS_COMPLEXITY_MAX,
    DEFAULT_COMPLEXITY_MIN as HDS_COMPLEXITY_MIN,
    DEFAULT_MAX_DIGITS,
    DEFAULT_MAX_VAL,
    DEFAULT_MIN_DIGITS,
    DEFAULT_MIN_VAL,
    DEFAULT_RC_OFFSET,
    DEFAULT_TARGET_TWO_DIGIT_RATIO,
    HDSConfig,
    TUNED_SCORING_CONFIG,
    build_default_digit_mix,
    canonical_pair,
    digit_bounds,
    generate_scaled_hds,
    generate_traps,
    normalize_digit_mix,
    scale_rc_bases,
)


def _build_default_hds_config() -> HDSConfig:
    valid_digits = [
        d
        for d in range(DEFAULT_MIN_DIGITS, DEFAULT_MAX_DIGITS + 1)
        if digit_bounds(d, DEFAULT_MIN_VAL, DEFAULT_MAX_VAL) is not None
    ]
    digit_mix = normalize_digit_mix(
        build_default_digit_mix(valid_digits, DEFAULT_TARGET_TWO_DIGIT_RATIO),
        valid_digits,
    )
    return HDSConfig(
        min_val=DEFAULT_MIN_VAL,
        max_val=DEFAULT_MAX_VAL,
        min_digits=DEFAULT_MIN_DIGITS,
        max_digits=DEFAULT_MAX_DIGITS,
        rc_bases=scale_rc_bases(
            BASE_RC_BASES,
            DEFAULT_MIN_DIGITS,
            DEFAULT_MAX_DIGITS,
            DEFAULT_MIN_VAL,
            DEFAULT_MAX_VAL,
        ),
        rc_offset=DEFAULT_RC_OFFSET,
        digit_mix=digit_mix,
        target_two_digit_ratio=DEFAULT_TARGET_TWO_DIGIT_RATIO,
        complexity_min=HDS_COMPLEXITY_MIN,
        complexity_max=HDS_COMPLEXITY_MAX,
        scoring=TUNED_SCORING_CONFIG,
    )


def test_multimodal_default_profile_spans_exact_c_and_all_bands() -> None:
    dataset = generate_paired_multimodal_dataset(
        count=10_000,
        seed=42,
        complexity_min=MULTIMODAL_COMPLEXITY_MIN,
        complexity_max=MULTIMODAL_COMPLEXITY_MAX,
    )
    expected_complexities = set(
        _reachable_complexities(MULTIMODAL_COMPLEXITY_MIN, MULTIMODAL_COMPLEXITY_MAX)
    )

    assert len(dataset) == 10_000
    assert {int(row["complexity_c"]) for row in dataset} == expected_complexities
    assert all(
        MULTIMODAL_COMPLEXITY_MIN <= int(row["complexity_c"]) <= MULTIMODAL_COMPLEXITY_MAX
        for row in dataset
    )
    mean_complexity = sum(int(row["complexity_c"]) for row in dataset) / len(dataset)
    assert mean_complexity == pytest.approx(CANONICAL_TARGET_MEAN_C, abs=0.01)

    band_counts = band_counts_for_complexities([int(row["complexity_c"]) for row in dataset])
    assert all(count >= 1000 for count in band_counts.values())

    for split in ("train", "val", "test"):
        split_band_counts = band_counts_for_complexities(
            [int(row["complexity_c"]) for row in dataset if row["split"] == split]
        )
        assert all(split_band_counts[f"{lo}-{hi}"] > 0 for lo, hi in CANONICAL_COMPLEXITY_BANDS)


def test_multimodal_default_profile_is_hard_skewed_by_exact_c() -> None:
    dataset = generate_paired_multimodal_dataset(
        count=10_000,
        seed=42,
        complexity_min=MULTIMODAL_COMPLEXITY_MIN,
        complexity_max=MULTIMODAL_COMPLEXITY_MAX,
    )
    counts = Counter(int(row["complexity_c"]) for row in dataset)

    assert counts[324] > counts[240] > counts[180] > counts[120] > counts[60] > counts[10]


def test_hds_defaults_keep_heuristics_balanced_under_hard_profile() -> None:
    items = generate_scaled_hds(target_count=1000, seed=42, config=_build_default_hds_config())
    heuristic_counts = Counter(item.canonical_target_heuristic for item in items)

    assert heuristic_counts == Counter({"RC": 334, "DD": 333, "OT": 333})
    assert all(HDS_COMPLEXITY_MIN <= item.complexity_c <= HDS_COMPLEXITY_MAX for item in items)
    mean_complexity = sum(item.complexity_c for item in items) / len(items)
    assert 181.0 <= mean_complexity <= 184.0
    assert min(item.complexity_c for item in items) <= 20
    assert max(item.complexity_c for item in items) >= 300

    band_counts = band_counts_for_complexities([item.complexity_c for item in items])
    assert all(count >= 100 for count in band_counts.values())

    for item in items:
        stats = compute_problem_stats(item.a, item.b)
        assert item.digit_total == stats["digit_total"]
        assert item.nonzero_total == stats["nonzero_total"]
        assert item.complexity_c == stats["complexity_c"]
        assert item.design_family == item.canonical_target_heuristic
        assert item.target_heuristic == item.design_family
        assert item.heuristic_definition_version == "cost_model_v2"
        assert item.canonical_target_margin > 0
        assert item.ot_cost > 0
        assert item.dd_cost > 0
        assert item.rc_cost > 0


def test_traps_defaults_span_low_mid_high_per_family_and_include_complexity_metadata() -> None:
    config = _build_default_hds_config()
    hds_items = generate_scaled_hds(target_count=99, seed=42, config=config)
    traps = generate_traps(
        exclude_pairs={canonical_pair(item.a, item.b) for item in hds_items},
        seed=42,
        config=config,
    )

    assert len(traps) == 30
    assert Counter(trap.trap_type for trap in traps) == Counter(
        {
            "carry_bomb": 8,
            "missing_term": 8,
            "anti_round": 8,
            "sign_trap": 6,
        }
    )
    assert all(HDS_COMPLEXITY_MIN <= trap.complexity_c <= HDS_COMPLEXITY_MAX for trap in traps)

    def bucket(complexity_c: int) -> str:
        if complexity_c <= 60:
            return "low"
        if complexity_c <= 180:
            return "mid"
        return "high"

    bucketed = {
        trap_type: {bucket(trap.complexity_c) for trap in traps if trap.trap_type == trap_type}
        for trap_type in {"carry_bomb", "missing_term", "anti_round", "sign_trap"}
    }
    assert all(levels == {"low", "mid", "high"} for levels in bucketed.values())

    for trap in traps:
        stats = compute_problem_stats(trap.a, trap.b)
        assert trap.digit_total == stats["digit_total"]
        assert trap.nonzero_total == stats["nonzero_total"]
        assert trap.complexity_c == stats["complexity_c"]
        assert trap.design_family == trap.target_heuristic
        assert trap.canonical_target_heuristic == trap.target_heuristic
        assert trap.heuristic_definition_version == "cost_model_v2"
