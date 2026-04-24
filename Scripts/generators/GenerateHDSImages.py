#!/usr/bin/env python3
"""
GenerateHDSImages.py

Generates LaTeX-rendered PNG images for HDS and Traps datasets.
These images enable multimodal fingerprinting experiments comparing text vs image inputs.

Usage:
    python Scripts/generators/GenerateHDSImages.py
    python Scripts/generators/GenerateHDSImages.py --dataset HDS --split test
    python Scripts/generators/GenerateHDSImages.py --dataset Traps
    python Scripts/generators/GenerateHDSImages.py --max-workers 8
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

# Add Scripts to path for imports
sys.path.insert(0, str(SCRIPTS_DIR))

from core.Logging import tprint

# Import image generation utilities from GenerateMathImages
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    tprint("Warning: pdf2image not installed. Run: pip install pdf2image")

# Dataset paths
DATASETS = {
    "HDS": {
        "csv": REPO_ROOT / "SavedData" / "HDS.csv",
        "output_dir": REPO_ROOT / "SavedData" / "HDSImages",
    },
    "HDSv2": {
        "csv": REPO_ROOT / "SavedData" / "HDSv2.csv",
        "output_dir": REPO_ROOT / "SavedData" / "HDSv2Images",
    },
    "Traps": {
        "csv": REPO_ROOT / "SavedData" / "Traps.csv",
        "output_dir": REPO_ROOT / "SavedData" / "TrapsImages",
    },
    "Trapsv2": {
        "csv": REPO_ROOT / "SavedData" / "Trapsv2.csv",
        "output_dir": REPO_ROOT / "SavedData" / "Trapsv2Images",
    },
}


def generate_latex(a: int, b: int) -> str:
    """Generate LaTeX document for a multiplication problem."""
    return rf"""\documentclass{{standalone}}
\usepackage{{amsmath}}
\begin{{document}}
\huge ${a} \times {b} = \text{{?}}$
\end{{document}}
"""


def compile_latex_to_png(latex_content: str, output_path: Path, dpi: int = 300) -> bool:
    """
    Compile LaTeX to PNG image.

    Args:
        latex_content: LaTeX document content.
        output_path: Path for output PNG file.
        dpi: Resolution for PNG output.

    Returns:
        True if successful, False otherwise.
    """
    if not PDF2IMAGE_AVAILABLE:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_path = tmpdir_path / "problem.tex"
        pdf_path = tmpdir_path / "problem.pdf"

        # Write LaTeX file
        tex_path.write_text(latex_content)

        # Compile to PDF
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(tmpdir_path), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if not pdf_path.exists():
                return False

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

        # Convert PDF to PNG
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            if images:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                images[0].save(output_path, "PNG")
                return True
            return False
        except Exception:
            return False


def load_problems(csv_path: Path, split: Optional[str] = None) -> List[Dict]:
    """
    Load problems from CSV.

    Args:
        csv_path: Path to CSV file
        split: Optional split filter ('train', 'val', 'test', or None for all)

    Returns:
        List of problem dictionaries
    """
    if not csv_path.exists():
        tprint(f"Error: CSV not found at {csv_path}")
        tprint("Run: python Scripts/generators/GenerateHDS.py first")
        sys.exit(1)

    problems = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_val = row.get('split')
            if split is None or split_val is None or split_val == split:
                problems.append({
                    'id': row['id'],
                    'a': int(row['a']),
                    'b': int(row['b']),
                    'split': row.get('split', 'all')
                })

    return problems


def generate_image(problem: Dict, output_dir: Path, force: bool = False) -> tuple:
    """
    Generate a single problem image.

    Args:
        problem: Problem dictionary with id, a, b
        output_dir: Directory for output images

    Returns:
        Tuple of (problem_id, success_bool, status)
    """
    output_path = output_dir / f"{problem['id']}.png"

    # Skip if already exists (unless forcing regeneration)
    if output_path.exists() and not force:
        return (problem['id'], True, "skipped")

    latex_content = generate_latex(problem['a'], problem['b'])
    success = compile_latex_to_png(latex_content, output_path)

    return (problem['id'], success, "generated" if success else "failed")


def check_dependencies() -> bool:
    """Check for required dependencies."""
    all_ok = True

    # Check pdflatex
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, timeout=5)
        tprint("pdflatex: OK")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tprint("pdflatex: NOT FOUND - Install a TeX distribution (e.g., MacTeX, TeX Live)")
        all_ok = False

    # Check pdf2image
    if PDF2IMAGE_AVAILABLE:
        tprint("pdf2image: OK")
    else:
        tprint("pdf2image: NOT FOUND - Run: pip install pdf2image")
        all_ok = False

    return all_ok


def resolve_dataset_paths(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    """Resolve CSV and output paths from CLI arguments."""
    if args.csv:
        csv_path = Path(args.csv)
        output_dir = Path(args.output_dir) if args.output_dir else (
            REPO_ROOT / "SavedData" / f"{csv_path.stem}Images"
        )
        label = csv_path.stem
        return csv_path, output_dir, label

    dataset = args.dataset
    if dataset not in DATASETS:
        tprint(f"Error: Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}")
        sys.exit(1)
    csv_path = DATASETS[dataset]["csv"]
    output_dir = Path(args.output_dir) if args.output_dir else DATASETS[dataset]["output_dir"]
    return csv_path, output_dir, dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX-rendered images for HDS/Traps problems"
    )
    parser.add_argument(
        "--dataset",
        choices=["HDS", "HDSv2", "Traps", "Trapsv2"],
        default="HDSv2",
        help="Dataset to generate images for (default: HDSv2)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Custom CSV path (overrides --dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for images (default: SavedData/<name>Images)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to generate images for (default: all, only applies to HDS)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for image generation (default: 4)"
    )
    regen_group = parser.add_mutually_exclusive_group()
    regen_group.add_argument(
        "--force",
        action="store_true",
        help="Regenerate images even if files already exist (default)"
    )
    regen_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        deps_ok = check_dependencies()
        sys.exit(0 if deps_ok else 1)

    if not check_dependencies():
        tprint("\nDependency check failed. Cannot generate images.")
        sys.exit(1)

    csv_path, output_dir, dataset_label = resolve_dataset_paths(args)
    force = args.force or not args.skip_existing

    # Load problems
    split_filter = None if args.split == "all" else args.split
    problems = load_problems(csv_path, split_filter)

    if not problems:
        tprint(f"No problems found for dataset: {dataset_label}, split: {args.split}")
        sys.exit(1)

    split_info = f", split: {args.split}" if args.split != "all" else ""
    tprint(f"\nGenerating images for {len(problems)} {dataset_label} problems{split_info}")
    tprint(f"Output directory: {output_dir}")

    # Create output directory (clear existing when forcing regeneration)
    if force and output_dir.exists():
        tprint(f"Clearing existing images in {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images in parallel
    generated = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(generate_image, p, output_dir, force): p for p in problems}

        for i, future in enumerate(as_completed(futures), 1):
            problem_id, success, status = future.result()

            if status == "skipped":
                skipped += 1
            elif status == "generated":
                generated += 1
            else:
                failed += 1
                tprint(f"  Failed: {problem_id}")

            # Progress update every 50 items
            if i % 50 == 0 or i == len(problems):
                tprint(f"  Progress: {i}/{len(problems)} (generated: {generated}, skipped: {skipped}, failed: {failed})")

    tprint(f"\nComplete!")
    tprint(f"  Generated: {generated}")
    tprint(f"  Skipped (existing): {skipped}")
    tprint(f"  Failed: {failed}")
    tprint(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
