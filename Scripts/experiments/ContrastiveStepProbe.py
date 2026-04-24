#!/usr/bin/env python3
"""
ContrastiveStepProbe.py

Run contrastive correct-vs-incorrect step probes on HDS/Traps.

This script:
1. Loads a dataset split (default: HDS test)
2. Scores correct vs plausible-incorrect heuristic steps
3. Saves per-problem details and aggregated analysis
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import asyncio
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Paths (experiments/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))

from core.DatasetSplits import get_hds_splits
from core.Logging import tprint
from core.TinkerClient import (
    VisionTinkerClient,
    DEFAULT_MODEL_NAME,
    get_contrastive_template_profile,
    set_contrastive_template_mode,
    set_contrastive_template_profile,
    get_contrastive_template_mode,
    get_contrastive_template_seed,
)


@dataclass
class HDSRow:
    """A row from HDS.csv or Traps.csv."""
    id: str
    a: int
    b: int
    product: int
    category: str
    notes: str
    target_heuristic: str
    design_family: str = ""
    canonical_target_heuristic: str = ""
    ot_score: float = 0.0
    dd_score: float = 0.0
    rc_score: float = 0.0
    split: str = ""


@dataclass
class ContrastiveResult:
    """Result of contrastive probing for a single problem."""
    hds_id: str
    a: int
    b: int
    product: int
    target_heuristic: str
    correct_losses: Dict[str, float]
    incorrect_losses: Dict[str, float]
    delta_losses: Dict[str, float]
    prefers_correct: Dict[str, Optional[bool]]
    per_template_losses: Dict[str, Dict[str, Any]]
    design_family: str = ""
    canonical_target_heuristic: str = ""


def load_hds(path: Path) -> List[HDSRow]:
    """Load HDS or Traps from CSV (auto-detects format)."""
    rows: List[HDSRow] = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'canonical_target_heuristic' in row:
                rows.append(HDSRow(
                    id=row['id'],
                    a=int(row['a']),
                    b=int(row['b']),
                    product=int(row['product']),
                    category=row.get('category') or row.get('trap_type', 'unknown'),
                    notes=row.get('notes', ''),
                    target_heuristic=row.get('canonical_target_heuristic') or row.get('target_heuristic', ''),
                    design_family=row.get('design_family', ''),
                    canonical_target_heuristic=row.get('canonical_target_heuristic') or row.get('target_heuristic', ''),
                    ot_score=float(row.get('ot_score', 0.0) or 0.0),
                    dd_score=float(row.get('dd_score', 0.0) or 0.0),
                    rc_score=float(row.get('rc_score', 0.0) or 0.0),
                    split=row.get('split', '')
                ))
            elif 'ot_score' in row:
                target = row['target_heuristic']
                rows.append(HDSRow(
                    id=row['id'],
                    a=int(row['a']),
                    b=int(row['b']),
                    product=int(row['product']),
                    category=row['category'],
                    notes=row['notes'],
                    target_heuristic=target,
                    design_family=target,
                    canonical_target_heuristic=target,
                    ot_score=float(row['ot_score']),
                    dd_score=float(row['dd_score']),
                    rc_score=float(row['rc_score']),
                    split=row.get('split', '')
                ))
            else:
                target = row.get('canonical_target_heuristic') or row['target_heuristic']
                rows.append(HDSRow(
                    id=row['id'],
                    a=int(row['a']),
                    b=int(row['b']),
                    product=int(row['product']),
                    category=row.get('trap_type', 'unknown'),
                    notes=row.get('notes', ''),
                    target_heuristic=target,
                    design_family=row.get('design_family', target),
                    canonical_target_heuristic=target,
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
        hds_dicts = [
            {
                "id": r.id,
                "a": r.a,
                "b": r.b,
                "product": r.product,
                "target_heuristic": r.target_heuristic,
                "ot_score": r.ot_score,
                "dd_score": r.dd_score,
                "rc_score": r.rc_score,
                "category": r.category,
                "notes": r.notes,
            }
            for r in all_rows
        ]
        splits = get_hds_splits(hds_dicts)
        return [HDSRow(**d) for d in splits.get(split, [])]

    return [r for r in all_rows if getattr(r, 'split', split) == split]


def _is_finite(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_and_se(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    if len(values) < 2:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(var / len(values))
    return (mean, se)


def binary_se(flags: List[int]) -> float:
    n = len(flags)
    if n == 0:
        return 0.0
    p = sum(flags) / n
    return math.sqrt(p * (1 - p) / n)


def summarize_pairs(pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
    correct_vals: List[float] = []
    incorrect_vals: List[float] = []
    delta_vals: List[float] = []
    pref_flags: List[int] = []

    for correct, incorrect in pairs:
        if not (_is_finite(correct) and _is_finite(incorrect)):
            continue
        correct_vals.append(correct)
        incorrect_vals.append(incorrect)
        delta_vals.append(incorrect - correct)
        pref_flags.append(1 if correct < incorrect else 0)

    mean_correct, se_correct = mean_and_se(correct_vals)
    mean_incorrect, se_incorrect = mean_and_se(incorrect_vals)
    mean_delta, se_delta = mean_and_se(delta_vals)
    pref_rate = sum(pref_flags) / len(pref_flags) if pref_flags else 0.0
    pref_rate_se = binary_se(pref_flags)

    return {
        "count": len(delta_vals),
        "mean_correct_loss": mean_correct,
        "mean_correct_loss_se": se_correct,
        "mean_incorrect_loss": mean_incorrect,
        "mean_incorrect_loss_se": se_incorrect,
        "mean_delta": mean_delta,
        "delta_se": se_delta,
        "pref_rate": pref_rate,
        "pref_rate_se": pref_rate_se,
    }


class ContrastiveProbeRunner:
    """Run contrastive probing for text or image modality."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        modality: str,
        images_dir: Optional[Path] = None
    ) -> None:
        self.modality = modality
        self.client = VisionTinkerClient(
            model_name=model_name,
            api_key=api_key,
            verbose=True
        )
        self.images_dir = images_dir or (REPO_ROOT / "SavedData" / "HDSImages")
        self.traps_images_dir = REPO_ROOT / "SavedData" / "TrapsImages"

    def _get_image_path(self, hds_id: str) -> Path:
        hds_path = self.images_dir / f"{hds_id}.png"
        if hds_path.exists():
            return hds_path
        traps_path = self.traps_images_dir / f"{hds_id}.png"
        if traps_path.exists():
            return traps_path
        return hds_path

    def _empty_probe(self) -> Dict[str, Any]:
        losses = {"OT": float('inf'), "DD": float('inf'), "RC": float('inf')}
        return {
            "correct_losses": losses.copy(),
            "incorrect_losses": losses.copy(),
            "delta_losses": losses.copy(),
            "per_template_losses": {}
        }

    def probe_row(self, row: HDSRow) -> ContrastiveResult:
        if self.modality == "text":
            probe = self.client.compute_contrastive_step_losses(row.a, row.b)
        else:
            image_path = self._get_image_path(row.id)
            if not image_path.exists():
                tprint(f"    Warning: Image not found: {image_path}")
                probe = self._empty_probe()
            else:
                probe = self.client.compute_contrastive_step_losses_with_image_batched(
                    image_path, row.a, row.b
                )

        correct_losses = probe.get("correct_losses", {})
        incorrect_losses = probe.get("incorrect_losses", {})
        delta_losses = probe.get("delta_losses", {})
        per_template_losses = probe.get("per_template_losses", {})

        prefers_correct: Dict[str, Optional[bool]] = {}
        for h in ("OT", "DD", "RC"):
            correct = correct_losses.get(h)
            incorrect = incorrect_losses.get(h)
            if _is_finite(correct) and _is_finite(incorrect):
                prefers_correct[h] = correct < incorrect
            else:
                prefers_correct[h] = None

        return ContrastiveResult(
            hds_id=row.id,
            a=row.a,
            b=row.b,
            product=row.product,
            target_heuristic=row.target_heuristic,
            design_family=row.design_family or row.target_heuristic,
            canonical_target_heuristic=row.canonical_target_heuristic or row.target_heuristic,
            correct_losses=correct_losses,
            incorrect_losses=incorrect_losses,
            delta_losses=delta_losses,
            prefers_correct=prefers_correct,
            per_template_losses=per_template_losses
        )


def analyze_results(results: List[ContrastiveResult]) -> Dict[str, Any]:
    heuristics = ("OT", "DD", "RC")

    by_heuristic: Dict[str, Dict[str, Any]] = {}
    by_target: Dict[str, Dict[str, Any]] = {}

    for h in heuristics:
        pairs = []
        for r in results:
            pairs.append((r.correct_losses.get(h), r.incorrect_losses.get(h)))
        by_heuristic[h] = summarize_pairs(pairs)

        target_pairs = []
        for r in results:
            if (r.canonical_target_heuristic or r.target_heuristic) == h:
                target_pairs.append((r.correct_losses.get(h), r.incorrect_losses.get(h)))
        by_target[h] = summarize_pairs(target_pairs)

    overall_pairs = []
    for r in results:
        target = r.canonical_target_heuristic or r.target_heuristic
        if target in heuristics:
            overall_pairs.append((r.correct_losses.get(target), r.incorrect_losses.get(target)))
    overall = summarize_pairs(overall_pairs)

    return {
        "total": len(results),
        "probe_kind": "contrastive_step",
        "by_heuristic": by_heuristic,
        "by_canonical_target_heuristic": by_target,
        "overall": overall
    }


def save_results(results: List[ContrastiveResult], analysis: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "contrastive_results.csv"
    fieldnames = [
        "hds_id", "a", "b", "product", "target_heuristic",
        "design_family", "canonical_target_heuristic",
        "ot_correct_loss", "ot_incorrect_loss", "ot_delta_loss", "ot_prefers_correct",
        "dd_correct_loss", "dd_incorrect_loss", "dd_delta_loss", "dd_prefers_correct",
        "rc_correct_loss", "rc_incorrect_loss", "rc_delta_loss", "rc_prefers_correct",
    ]

    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "hds_id": r.hds_id,
                "a": r.a,
                "b": r.b,
                "product": r.product,
                "target_heuristic": r.target_heuristic,
                "design_family": r.design_family or r.target_heuristic,
                "canonical_target_heuristic": r.canonical_target_heuristic or r.target_heuristic,
                "ot_correct_loss": r.correct_losses.get("OT"),
                "ot_incorrect_loss": r.incorrect_losses.get("OT"),
                "ot_delta_loss": r.delta_losses.get("OT"),
                "ot_prefers_correct": r.prefers_correct.get("OT"),
                "dd_correct_loss": r.correct_losses.get("DD"),
                "dd_incorrect_loss": r.incorrect_losses.get("DD"),
                "dd_delta_loss": r.delta_losses.get("DD"),
                "dd_prefers_correct": r.prefers_correct.get("DD"),
                "rc_correct_loss": r.correct_losses.get("RC"),
                "rc_incorrect_loss": r.incorrect_losses.get("RC"),
                "rc_delta_loss": r.delta_losses.get("RC"),
                "rc_prefers_correct": r.prefers_correct.get("RC"),
            }
            writer.writerow(row)

    tprint(f"Saved results to {csv_path}")

    json_path = output_dir / "contrastive_analysis.json"
    with json_path.open('w') as f:
        json.dump(analysis, f, indent=2)
    tprint(f"Saved analysis to {json_path}")


def save_details(results: List[ContrastiveResult], output_dir: Path) -> None:
    from datetime import datetime, timezone

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "contrastive_details.jsonl"

    with jsonl_path.open('w') as f:
        for r in results:
            record = {
                "hds_id": r.hds_id,
                "a": r.a,
                "b": r.b,
                "product": r.product,
                "target_heuristic": r.target_heuristic,
                "design_family": r.design_family or r.target_heuristic,
                "canonical_target_heuristic": r.canonical_target_heuristic or r.target_heuristic,
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "contrastive": {
                    "correct_losses": r.correct_losses,
                    "incorrect_losses": r.incorrect_losses,
                    "delta_losses": r.delta_losses,
                    "prefers_correct": r.prefers_correct,
                    "templates": r.per_template_losses,
                },
            }
            f.write(json.dumps(record) + "\n")

    tprint(f"Saved detailed results to {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contrastive correct-vs-incorrect step probes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="HDSv2", help="Dataset name (default: HDSv2)")
    parser.add_argument("--csv", type=str, help="Optional CSV path (overrides --dataset)")
    parser.add_argument("--split", type=str, default="test", help="Split to use (train/val/test/all)")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "image"], help="Input modality")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of problems")
    parser.add_argument("--images-dir", type=str, help="Override image directory (HDSImages)")
    parser.add_argument("--template-mode", type=str, default="multi",
                        choices=["single", "multi"],
                        help="Contrastive template mode: single or multi (default: multi)")
    parser.add_argument("--template-seed", type=int, default=None,
                        help="Seed for single-template selection (default: 0)")
    parser.add_argument("--template-profile", type=str, default=None,
                        choices=["standard", "harder"],
                        help="Contrastive template profile (default: standard)")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Optional suffix for the SavedResults directory")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent probes (default: 4)")

    args = parser.parse_args()

    if args.template_mode is not None or args.template_seed is not None:
        mode = args.template_mode or get_contrastive_template_mode()
        set_contrastive_template_mode(mode, seed=args.template_seed)
    if args.template_profile is not None:
        set_contrastive_template_profile(args.template_profile)

    if args.csv:
        dataset_path = Path(args.csv)
        dataset_name = dataset_path.stem
    else:
        dataset_name = args.dataset
        dataset_path = REPO_ROOT / "SavedData" / f"{dataset_name}.csv"

    tprint("=" * 60)
    tprint(f"Contrastive Step Probing on {dataset_name}")
    tprint("=" * 60)
    tprint(f"Model: {args.model}")
    tprint(f"Modality: {args.modality}")
    tprint(f"Split: {args.split}")
    tprint(f"Template mode: {get_contrastive_template_mode()} (seed={get_contrastive_template_seed()})")
    tprint(f"Template profile: {get_contrastive_template_profile()}")
    tprint()

    tprint(f"Loading dataset from {dataset_path}...")
    all_rows = load_hds(dataset_path)
    tprint(f"Loaded {len(all_rows)} total problems")

    rows = select_rows_for_split(all_rows, args.split, dataset_name)
    if args.split == "all":
        tprint(f"Using all problems: {len(rows)}")
    else:
        tprint(f"Using {args.split} split: {len(rows)} problems")

    if args.limit is not None:
        rows = rows[:args.limit]
        tprint(f"Limiting to first {len(rows)} problems")

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not found in environment")

    images_dir = Path(args.images_dir) if args.images_dir else None
    runner = ContrastiveProbeRunner(
        api_key=api_key,
        model_name=args.model,
        modality=args.modality,
        images_dir=images_dir
    )

    tprint()
    tprint("Running contrastive probes...")
    tprint("-" * 40)

    start_time = time.time()
    results: List[ContrastiveResult] = []

    async def _run_probes():
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        results_local: List[Optional[ContrastiveResult]] = [None] * len(rows)

        async def _one(idx: int, row: HDSRow):
            async with semaphore:
                res = await asyncio.to_thread(runner.probe_row, row)
                results_local[idx] = res
                if (idx + 1) % max(1, len(rows) // 10 or 1) == 0:
                    tprint(f"  {idx+1}/{len(rows)}")

        await asyncio.gather(*(_one(i, r) for i, r in enumerate(rows)))
        return [r for r in results_local if r is not None]

    results = asyncio.run(_run_probes())
    elapsed = time.time() - start_time

    if rows:
        tprint(f"Completed in {elapsed:.1f}s ({elapsed/len(rows):.2f}s per problem)")

    analysis = analyze_results(results)
    analysis["model"] = args.model
    analysis["dataset"] = dataset_name
    analysis["split"] = args.split
    analysis["modality"] = args.modality
    analysis["template_mode"] = get_contrastive_template_mode()
    analysis["template_profile"] = get_contrastive_template_profile()
    analysis["output_tag"] = args.output_tag
    analysis["elapsed_seconds"] = elapsed
    analysis["use_real_api"] = True

    model_slug = args.model.split("/")[-1].replace("-Instruct", "")
    split_suffix = f"_{args.split}" if args.split and args.split != "all" else ""
    modality_suffix = "" if args.modality == "text" else f"_{args.modality}"
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    output_dir = REPO_ROOT / "SavedResults" / (
        f"contrastive_{dataset_name.lower()}{split_suffix}{modality_suffix}{tag_suffix}_{model_slug}"
    )

    save_results(results, analysis, output_dir)
    save_details(results, output_dir)


if __name__ == "__main__":
    main()
