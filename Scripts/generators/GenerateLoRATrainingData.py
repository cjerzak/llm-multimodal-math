#!/usr/bin/env python3
"""
GenerateLoRATrainingData.py

Generate synthetic reasoning traces for LoRA heuristic training.
Creates examples per heuristic (RC, DD, OT, STYLE) with step-by-step solutions.

IMPORTANT: This script now EXCLUDES all test problems (HDS, Traps, and test
splits from other datasets) to prevent data leakage.

Usage:
    python Scripts/GenerateLoRATrainingData.py --count 100
    python Scripts/GenerateLoRATrainingData.py --count 100 --exclude-test-problems
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from enum import Enum

# Paths (generators/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))

from core.DatasetSplits import (
    get_default_exclusion_set, is_excluded_problem,
    get_problem_fingerprint, assign_splits, parse_split_ratios, SPLIT_SEED
)
from core.Logging import tprint
OUTPUT_DIR = REPO_ROOT / "SavedData" / "LoRATraining"
HDS_PATH = REPO_ROOT / "SavedData" / "HDS.csv"
TRAPS_PATH = REPO_ROOT / "SavedData" / "Traps.csv"


class Heuristic(Enum):
    RC = "rounding_compensation"
    DD = "decomposition"
    OT = "ones_then_tens"
    STYLE = "style_control"


@dataclass
class TrainingExample:
    """A single training example with prompt and reasoning trace."""
    id: str
    a: int
    b: int
    product: int
    heuristic: str
    prompt: str
    reasoning_trace: str
    full_text: str  # prompt + reasoning for training


def nearest_base(n: int, bases: List[int] = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]) -> int:
    """Find the nearest round base to a number."""
    return min(bases, key=lambda b: abs(n - b))


def generate_rc_trace(a: int, b: int) -> str:
    """Generate Rounding-Compensation reasoning trace.

    RC is best when numbers are close to round bases (50, 100, etc.)
    Uses identities like (n-k)(n+k) = n^2 - k^2
    """
    product = a * b
    base_a = nearest_base(a)
    base_b = nearest_base(b)
    diff_a = a - base_a
    diff_b = b - base_b

    # Check for symmetric case (a+k)(a-k) = a^2 - k^2
    if base_a == base_b and diff_a == -diff_b:
        base = base_a
        k = abs(diff_a)
        base_squared = base * base
        k_squared = k * k
        # Determine which is larger to get correct signs
        sign_a = "+" if diff_a > 0 else "-"
        sign_b = "+" if diff_b > 0 else "-"
        trace = f"""What is {a} × {b}?
Let me use the difference of squares identity.
{a} = {base} {sign_a} {k} and {b} = {base} {sign_b} {k}.
So {a} × {b} = ({base} {sign_a} {k})({base} {sign_b} {k}) = {base}² - {k}².
{base}² = {base_squared}.
{k}² = {k_squared}.
{base_squared} - {k_squared} = {product}.
Answer: {product}"""
    else:
        # General rounding approach
        base_product = base_a * base_b
        # Compute adjustment using distributive property
        # a*b = (base_a + diff_a)(base_b + diff_b)
        #     = base_a*base_b + base_a*diff_b + diff_a*base_b + diff_a*diff_b
        term1 = base_a * base_b
        term2 = base_a * diff_b
        term3 = diff_a * base_b
        term4 = diff_a * diff_b

        trace = f"""What is {a} × {b}?
Let me round to convenient bases and adjust.
{a} is close to {base_a} (difference: {diff_a:+d}).
{b} is close to {base_b} (difference: {diff_b:+d}).
Start with {base_a} × {base_b} = {term1}.
Adjustment for {a}: {base_a} × {diff_b:+d} = {term2:+d}.
Adjustment for {b}: {diff_a:+d} × {base_b} = {term3:+d}.
Cross term: {diff_a:+d} × {diff_b:+d} = {term4:+d}.
Total: {term1} + ({term2:+d}) + ({term3:+d}) + ({term4:+d}) = {product}.
Answer: {product}"""

    return trace


def generate_dd_trace(a: int, b: int) -> str:
    """Generate Decomposition (Distributive Property) reasoning trace.

    DD decomposes numbers into tens and ones, then uses distributive property.
    """
    product = a * b
    a_tens = (a // 10) * 10
    a_ones = a % 10
    b_tens = (b // 10) * 10
    b_ones = b % 10

    # Decompose the larger number
    if a >= b:
        # Decompose a: a*b = (a_tens + a_ones)*b = a_tens*b + a_ones*b
        part1 = a_tens * b
        part2 = a_ones * b
        trace = f"""What is {a} × {b}?
Let me decompose {a} into {a_tens} + {a_ones}.
First compute {a_tens} × {b}:
{a_tens} × {b} = {part1}.
Then compute {a_ones} × {b}:
{a_ones} × {b} = {part2}.
Now sum the partial products:
{part1} + {part2} = {product}.
Answer: {product}"""
    else:
        # Decompose b: a*b = a*(b_tens + b_ones) = a*b_tens + a*b_ones
        part1 = a * b_tens
        part2 = a * b_ones
        trace = f"""What is {a} × {b}?
Let me decompose {b} into {b_tens} + {b_ones}.
First compute {a} × {b_tens}:
{a} × {b_tens} = {part1}.
Then compute {a} × {b_ones}:
{a} × {b_ones} = {part2}.
Now sum the partial products:
{part1} + {part2} = {product}.
Answer: {product}"""

    return trace


def generate_ot_trace(a: int, b: int) -> str:
    """Generate Ones-Then-Tens (Columnar Multiplication) reasoning trace.

    OT does digit-by-digit multiplication like traditional column multiplication.
    """
    product = a * b

    # Get digits
    a_ones = a % 10
    a_tens = (a // 10) % 10
    a_hundreds = (a // 100) % 10
    b_ones = b % 10
    b_tens = (b // 10) % 10
    b_hundreds = (b // 100) % 10

    trace_lines = [f"What is {a} × {b}?", "Let me use column multiplication step by step."]

    # For 2-digit × 2-digit case (most common)
    if a < 100 and b < 100:
        # First partial product: a × b_ones
        p1 = a * b_ones
        p1_ones = p1 % 10
        p1_rest = p1 // 10

        trace_lines.append(f"Step 1: Multiply {a} by ones digit {b_ones}:")
        trace_lines.append(f"  {a_ones} × {b_ones} = {a_ones * b_ones}, write {(a_ones * b_ones) % 10}, carry {(a_ones * b_ones) // 10}.")
        trace_lines.append(f"  {a_tens} × {b_ones} = {a_tens * b_ones}, plus carry = {a_tens * b_ones + (a_ones * b_ones) // 10}.")
        trace_lines.append(f"  First partial product: {p1}.")

        # Second partial product: a × b_tens (shifted)
        p2 = a * b_tens

        trace_lines.append(f"Step 2: Multiply {a} by tens digit {b_tens}:")
        trace_lines.append(f"  {a_ones} × {b_tens} = {a_ones * b_tens}, write {(a_ones * b_tens) % 10}, carry {(a_ones * b_tens) // 10}.")
        trace_lines.append(f"  {a_tens} × {b_tens} = {a_tens * b_tens}, plus carry = {a_tens * b_tens + (a_ones * b_tens) // 10}.")
        trace_lines.append(f"  Second partial product: {p2} (shifted by 10 = {p2 * 10}).")

        trace_lines.append(f"Step 3: Add partial products:")
        trace_lines.append(f"  {p1} + {p2 * 10} = {product}.")
        trace_lines.append(f"Answer: {product}")
    else:
        # Simpler approach for larger numbers
        # Just show the basic columnar steps
        p1 = a * b_ones
        p2 = a * (b_tens * 10) if b >= 10 else 0
        p3 = a * (b_hundreds * 100) if b >= 100 else 0

        trace_lines.append(f"Step 1: {a} × {b_ones} = {p1}")
        if b >= 10:
            trace_lines.append(f"Step 2: {a} × {b_tens}0 = {p2}")
        if b >= 100:
            trace_lines.append(f"Step 3: {a} × {b_hundreds}00 = {p3}")
        trace_lines.append(f"Sum: {p1} + {p2} + {p3} = {product}.")
        trace_lines.append(f"Answer: {product}")

    return "\n".join(trace_lines)


def generate_style_trace(a: int, b: int) -> str:
    """Generate a generic reasoning-trace style control with no heuristic content."""
    product = a * b
    return f"""What is {a} × {b}?
Let me organize the work clearly.
Step 1: Keep the problem in standard form: {a} × {b}.
Step 2: Compute the exact product carefully.
Step 3: The exact product is {product}.
Answer: {product}"""


def generate_rc_problems(n: int) -> List[Tuple[int, int]]:
    """Generate n problems well-suited for RC heuristic (near round bases)."""
    problems = []
    bases = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
    far_from_base = [
        x for x in range(10, 1000)
        if min(abs(x - b) for b in bases) > 5 and x % 5 != 0
    ]

    for _ in range(n):
        base = random.choice(bases)
        offset = random.randint(1, 5)
        sign = random.choice([-1, 1])

        # Create near-base pair
        if random.random() < 0.4:
            # Symmetric: (base-k)(base+k)
            a = base - offset
            b = base + offset
        elif random.random() < 0.7:
            # Both near same base
            a = base + sign * offset
            b = base + random.choice([-1, 1]) * random.randint(1, 5)
        else:
            # One near base, one far
            a = base + sign * offset
            b = random.choice(far_from_base) if far_from_base else base + offset

        problems.append((a, b))

    return problems


def generate_dd_problems(n: int) -> List[Tuple[int, int]]:
    """Generate n problems well-suited for DD heuristic (clean decomposition)."""
    problems = []

    for _ in range(n):
        # Favor numbers with trailing zeros or round factors
        if random.random() < 0.4:
            # One factor has trailing zeros
            a = random.randint(10, 99)
            b = random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100, 200])
        elif random.random() < 0.3:
            # Clean tens + small ones
            a = random.randint(2, 9) * 10 + random.randint(0, 5)
            b = random.randint(2, 9) * 10 + random.randint(0, 5)
        else:
            # General 2-digit numbers
            a = random.randint(11, 99)
            b = random.randint(11, 99)

        problems.append((a, b))

    return problems


def generate_ot_problems(n: int) -> List[Tuple[int, int]]:
    """Generate n problems for OT heuristic (carry-heavy multiplication)."""
    problems = []

    for _ in range(n):
        if random.random() < 0.6:
            # Carry-heavy numbers with high digits (2-3 digits)
            length = random.choice([2, 3])
            a = int("".join(str(random.randint(6, 9)) for _ in range(length)))
            b = int("".join(str(random.randint(6, 9)) for _ in range(length)))
        else:
            # Generic multi-digit numbers
            a = random.randint(32, 999)
            b = random.randint(32, 999)

        problems.append((a, b))

    return problems


def generate_style_problems(n: int) -> List[Tuple[int, int]]:
    """Generate a mixed problem pool for the style-only control adapter."""
    pools = [
        generate_rc_problems(max(1, n // 3 + 1)),
        generate_dd_problems(max(1, n // 3 + 1)),
        generate_ot_problems(max(1, n // 3 + 1)),
    ]
    mixed = [pair for pool in pools for pair in pool]
    random.shuffle(mixed)
    return mixed[:n]


def generate_training_data(
    heuristic: Heuristic,
    target_count: int = 100,
    exclusion_set: Optional[Set[str]] = None,
    seed: int = SPLIT_SEED
) -> List[TrainingExample]:
    """Generate training examples for a specific heuristic.

    IMPORTANT: This function NO LONGER uses HDS problems to prevent data leakage.
    All problems are freshly generated and checked against the exclusion set.

    Args:
        heuristic: Which heuristic to generate for (RC, DD, OT, STYLE)
        target_count: Number of examples to generate
        exclusion_set: Set of problem fingerprints to exclude (test data)
        seed: Random seed for reproducibility
    """
    heuristic_seed_offset = {"RC": 0, "DD": 1, "OT": 2, "STYLE": 3}[heuristic.name]
    random.seed(seed + heuristic_seed_offset)  # Unique seed per heuristic

    examples: List[TrainingExample] = []
    seen_pairs: Set[Tuple[int, int]] = set()

    if exclusion_set is None:
        exclusion_set = set()

    # Generate ALL problems freshly (NO HDS seeding to prevent leakage)
    max_attempts = target_count * 50

    if heuristic == Heuristic.RC:
        problem_generator = generate_rc_problems
    elif heuristic == Heuristic.DD:
        problem_generator = generate_dd_problems
    elif heuristic == Heuristic.OT:
        problem_generator = generate_ot_problems
    else:
        problem_generator = generate_style_problems

    # Generate more than needed and filter
    candidate_problems = problem_generator(max_attempts)

    for a, b in candidate_problems:
        if len(examples) >= target_count:
            break

        # Skip duplicates
        if (a, b) in seen_pairs or (b, a) in seen_pairs:
            continue

        # Skip excluded problems (test data)
        if is_excluded_problem(a, b, exclusion_set):
            continue

        seen_pairs.add((a, b))
        product = a * b

        if heuristic == Heuristic.RC:
            trace = generate_rc_trace(a, b)
        elif heuristic == Heuristic.DD:
            trace = generate_dd_trace(a, b)
        elif heuristic == Heuristic.OT:
            trace = generate_ot_trace(a, b)
        else:
            trace = generate_style_trace(a, b)

        prompt = f"What is {a} × {b}?"
        examples.append(TrainingExample(
            id=f"{heuristic.name.lower()}_gen_{len(examples):04d}",
            a=a, b=b, product=product,
            heuristic=heuristic.name,
            prompt=prompt,
            reasoning_trace=trace,
            full_text=trace
        ))

    if len(examples) < target_count:
        tprint(f"  Warning: Only generated {len(examples)}/{target_count} examples "
              f"(exclusions may have filtered too many)")

    return examples


def save_training_data(
    examples: List[TrainingExample],
    heuristic: Heuristic,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = SPLIT_SEED
) -> Path:
    """Save training examples to CSV file with train/val splits.

    Note: No test split - test data comes from HDS/Traps which are excluded.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{heuristic.name.lower()}_training.csv"

    # Default to 85/15 train/val (no test - that's HDS/Traps)
    if split_ratios is None:
        split_ratios = {"train": 0.85, "val": 0.15, "test": 0.0}

    # Convert to dicts for split assignment
    example_dicts = [
        {
            "id": ex.id,
            "a": ex.a,
            "b": ex.b,
            "product": ex.product,
            "heuristic": ex.heuristic,
            "prompt": ex.prompt,
            "reasoning_trace": ex.reasoning_trace,
            "full_text": ex.full_text
        }
        for ex in examples
    ]

    # Assign splits (train/val only)
    example_dicts = assign_splits(example_dicts, ratios=split_ratios, id_key="id", seed=seed)

    # Remove any "test" split (shouldn't happen with 0% test, but just in case)
    for ex in example_dicts:
        if ex.get("split") == "test":
            ex["split"] = "val"  # Reassign to val if somehow assigned to test

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "a", "b", "product", "heuristic", "prompt", "reasoning_trace", "full_text", "split"
        ])
        writer.writeheader()
        writer.writerows(example_dicts)

    # Print split statistics
    split_counts: Dict[str, int] = {}
    for ex in example_dicts:
        split = ex.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    tprint(f"    Splits: {split_counts}")

    return output_path


def main():
    """Generate training data for heuristic adapters and the style control."""
    parser = argparse.ArgumentParser(description="Generate LoRA training data")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of examples per heuristic (default: 100)")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED,
                        help=f"Random seed for reproducibility (default: {SPLIT_SEED})")
    parser.add_argument("--exclude-test-problems", action="store_true", default=True,
                        help="Exclude HDS/Traps/test problems from training (default: True)")
    parser.add_argument("--no-exclude", action="store_true",
                        help="Disable exclusion (not recommended - causes data leakage)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--no-style-control", action="store_true",
                        help="Do not generate the style-only control dataset")
    args = parser.parse_args()

    tprint("=" * 60)
    tprint("Generating LoRA Training Data")
    tprint("=" * 60)
    tprint(f"  Count per heuristic: {args.count}")
    tprint(f"  Seed: {args.seed}")
    tprint(f"  Val ratio: {args.val_ratio}")

    # Build exclusion set to prevent data leakage
    if args.no_exclude:
        tprint("  Exclusion: DISABLED (WARNING: may cause data leakage!)")
        exclusion_set = set()
    else:
        tprint("  Building exclusion set from HDS/Traps/test splits...")
        exclusion_set = get_default_exclusion_set()
        tprint(f"  Excluded {len(exclusion_set)} problem fingerprints")

    # Split ratios for train/val (no test - that's HDS/Traps)
    split_ratios = {"train": 1.0 - args.val_ratio, "val": args.val_ratio, "test": 0.0}
    tprint(f"  Train/val split: {split_ratios['train']:.0%}/{split_ratios['val']:.0%}")
    tprint()

    heuristics_to_generate = [Heuristic.RC, Heuristic.DD, Heuristic.OT]
    if not args.no_style_control:
        heuristics_to_generate.append(Heuristic.STYLE)

    for heuristic in heuristics_to_generate:
        tprint(f"Generating {heuristic.name} training data...")

        examples = generate_training_data(
            heuristic,
            target_count=args.count,
            exclusion_set=exclusion_set,
            seed=args.seed
        )

        output_path = save_training_data(
            examples,
            heuristic,
            split_ratios=split_ratios,
            seed=args.seed
        )

        tprint(f"  Generated {len(examples)} examples")
        tprint(f"  Saved to: {output_path}")

        # Print sample
        if examples:
            tprint(f"\n  Sample {heuristic.name} trace:")
            tprint("-" * 40)
            tprint(examples[0].full_text[:500] + "..." if len(examples[0].full_text) > 500 else examples[0].full_text)
            tprint("-" * 40)
        tprint()

    tprint("=" * 60)
    tprint("Done! Training data saved to SavedData/LoRATraining/")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
