#!/usr/bin/env python3
"""
MultiModalGenerator.py

Base class for multi-modal dataset generation (images, audio, text).

Provides:
- Common CLI argument parsing (--count, --split-ratios, --seed, --max-workers)
- Dataset generation with splits
- Progress tracking with parallel execution
- Completion statistics

Usage:
    Subclass MultiModalGenerator and implement:
    - generate_item(row, output_path) -> bool
    - check_dependencies() -> bool (optional)
    - Define class attributes: MODALITY_NAME, OUTPUT_DIR, FILE_EXTENSION, etc.
"""

import argparse
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .GenerateMathHelpers import (
    MULTIMODAL_DEFAULT_COUNT,
    DEFAULT_COMPLEXITY_MAX,
    DEFAULT_COMPLEXITY_MIN,
    REPO_ROOT,
    get_or_create_shared_multimodal_dataset,
    save_csv,
)
from .DatasetSplits import parse_split_ratios, SPLIT_SEED
from .Logging import tprint


class MultiModalGenerator(ABC):
    """
    Base class for multi-modal dataset generators.

    Subclasses must define:
    - MODALITY_NAME: str (e.g., "Image", "Audio", "Text")
    - CSV_PATH: Path to output CSV
    - OUTPUT_DIR: Path to output directory for media files
    - FILE_EXTENSION: str (e.g., "png", "mp3", "txt")
    - ID_PREFIX: str (e.g., "img", "aud", "txt")
    - SKIP_FLAG: str (e.g., "skip-images", "skip-audio")

    Subclasses must implement:
    - generate_item(row, output_path) -> bool
    """

    # Override these in subclasses
    MODALITY_NAME: str = "Multi-Modal"
    CSV_PATH: Path = REPO_ROOT / "SavedData" / "Dataset.csv"
    OUTPUT_DIR: Path = REPO_ROOT / "SavedData" / "Output"
    FILE_EXTENSION: str = "dat"
    ID_PREFIX: str = "data"
    SKIP_FLAG: str = "skip-files"

    def __init__(self):
        """Initialize the generator."""
        self.args = None
        self.dataset = None

    @abstractmethod
    def generate_item(self, row: Dict, output_path: Path) -> bool:
        """
        Generate a single item (image, audio, or text file).

        Args:
            row: Dataset row with 'id', 'a', 'b', 'a_times_b', etc.
            output_path: Path where the file should be saved

        Returns:
            True if successful, False otherwise
        """
        pass

    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.

        Override in subclasses to check modality-specific dependencies.

        Returns:
            True if all dependencies are available, False otherwise.
        """
        return True

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description=f"Generate multiplication problem {self.MODALITY_NAME.lower()}s"
        )
        parser.add_argument(
            "--count", type=int, default=MULTIMODAL_DEFAULT_COUNT,
            help=f"Number of paired problems to generate (default: {MULTIMODAL_DEFAULT_COUNT})"
        )
        parser.add_argument(
            "--split-ratios", type=str, default="70/15/15",
            help="Train/val/test split ratios (default: 70/15/15)"
        )
        parser.add_argument(
            "--complexity-min", type=int, default=DEFAULT_COMPLEXITY_MIN,
            help=f"Minimum digit complexity C to export (default: {DEFAULT_COMPLEXITY_MIN})"
        )
        parser.add_argument(
            "--complexity-max", type=int, default=DEFAULT_COMPLEXITY_MAX,
            help=f"Maximum digit complexity C to export (default: {DEFAULT_COMPLEXITY_MAX})"
        )
        parser.add_argument(
            "--seed", type=int, default=SPLIT_SEED,
            help=f"Random seed for reproducibility (default: {SPLIT_SEED})"
        )
        parser.add_argument(
            f"--{self.SKIP_FLAG}", action="store_true",
            dest="skip_files",
            help=f"Only generate CSV, skip {self.MODALITY_NAME.lower()} generation"
        )
        parser.add_argument(
            "--max-workers", type=int, default=1,
            help="Number of parallel workers (default: 1, sequential)"
        )
        parser.add_argument(
            "--force-regenerate-grid", action="store_true",
            help="Regenerate the shared paired multimodal CSV even if it already exists"
        )

        self.args = parser.parse_args(args)
        return self.args

    def print_banner(self):
        """Print the startup banner."""
        tprint("=" * 60)
        tprint(f"GenerateMath{self.MODALITY_NAME}s - Multi-Modal Math Dataset")
        tprint("=" * 60)
        tprint(f"  Count: {self.args.count}")
        tprint(f"  Split ratios: {self.args.split_ratios}")
        tprint(f"  Complexity band: [{self.args.complexity_min}, {self.args.complexity_max}]")
        tprint(f"  Seed: {self.args.seed}")
        if self.args.max_workers > 1:
            tprint(f"  Parallel workers: {self.args.max_workers}")
        tprint()

    def generate_csv(self) -> List[Dict]:
        """Generate the dataset and save CSV."""
        tprint("Generating shared paired multimodal dataset...")
        split_ratios = parse_split_ratios(self.args.split_ratios)

        self.dataset = get_or_create_shared_multimodal_dataset(
            count=self.args.count,
            include_splits=True,
            split_ratios=split_ratios,
            seed=self.args.seed,
            force_regenerate=self.args.force_regenerate_grid,
            complexity_min=self.args.complexity_min,
            complexity_max=self.args.complexity_max,
        )

        # Save modality-specific CSV as a copy of the shared paired grid.
        save_csv(self.dataset, self.CSV_PATH)

        # Print split statistics
        split_counts = self._count_splits()
        tprint(f"  Splits: {split_counts}")
        tprint()

        return self.dataset

    def _count_splits(self) -> Dict[str, int]:
        """Count items per split."""
        split_counts: Dict[str, int] = {}
        for row in self.dataset:
            split = row.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1
        return split_counts

    def _generate_single(self, row: Dict) -> bool:
        """Generate a single item with error handling."""
        item_id = row["id"]
        output_path = self.OUTPUT_DIR / f"{item_id}.{self.FILE_EXTENSION}"

        try:
            return self.generate_item(row, output_path)
        except Exception as e:
            tprint(f"  Error generating {item_id}: {e}")
            return False

    def generate_files(self) -> tuple:
        """
        Generate all media files from the dataset.

        Returns:
            (success_count, fail_count) tuple
        """
        tprint(f"Generating {self.MODALITY_NAME.lower()} files...")

        total = len(self.dataset)
        progress_interval = max(1, total // 20)  # Report every 5%

        if self.args.max_workers <= 1:
            # Sequential processing
            return self._generate_sequential(progress_interval)
        else:
            # Parallel processing
            return self._generate_parallel(progress_interval)

    def _generate_sequential(self, progress_interval: int) -> tuple:
        """Generate files sequentially."""
        success_count = 0
        fail_count = 0
        total = len(self.dataset)

        for i, row in enumerate(self.dataset):
            if self._generate_single(row):
                success_count += 1
            else:
                fail_count += 1
                tprint(f"  Failed: {row['id']} ({row['a']} x {row['b']})")

            if (i + 1) % progress_interval == 0:
                tprint(f"  Progress: {i + 1}/{total} files generated")

        return success_count, fail_count

    def _generate_parallel(self, progress_interval: int) -> tuple:
        """Generate files in parallel."""
        success_count = 0
        fail_count = 0
        total = len(self.dataset)
        completed = 0

        tprint(f"  Using {self.args.max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._generate_single, row): row
                for row in self.dataset
            }

            for future in as_completed(futures):
                row = futures[future]
                completed += 1

                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                        tprint(f"  Failed: {row['id']} ({row['a']} x {row['b']})")
                except Exception as e:
                    fail_count += 1
                    tprint(f"  Error: {row['id']}: {e}")

                if completed % progress_interval == 0:
                    tprint(f"  Progress: {completed}/{total} files generated")

        return success_count, fail_count

    def clean_outputs(self):
        """Remove existing output files to ensure fresh generation."""
        removed_any = False

        if self.CSV_PATH.exists():
            self.CSV_PATH.unlink()
            removed_any = True

        if self.OUTPUT_DIR.exists():
            if self.OUTPUT_DIR.is_dir():
                shutil.rmtree(self.OUTPUT_DIR)
            else:
                self.OUTPUT_DIR.unlink()
            removed_any = True

        if removed_any:
            tprint(f"Cleared existing {self.MODALITY_NAME.lower()} outputs")

    def print_completion(self, success_count: int, fail_count: int):
        """Print completion summary."""
        tprint()
        tprint("=" * 60)
        tprint(f"Complete! Generated {success_count} {self.MODALITY_NAME.lower()} files, {fail_count} failures")
        tprint(f"CSV: {self.CSV_PATH}")
        tprint(f"{self.MODALITY_NAME}s: {self.OUTPUT_DIR}")
        tprint("=" * 60)

    def run(self, args: Optional[List[str]] = None):
        """
        Main entry point for the generator.

        Args:
            args: Optional list of command-line arguments (for testing)
        """
        # Parse arguments
        self.parse_args(args)

        # Print banner
        self.print_banner()

        # Check dependencies
        if not self.args.skip_files:
            tprint("Checking dependencies...")
            if not self.check_dependencies():
                tprint("\nSome dependencies are missing. Please install them and try again.")
                return
            tprint()

            self.clean_outputs()

        # Generate dataset and CSV
        self.generate_csv()

        # Skip file generation if requested
        if self.args.skip_files:
            tprint(f"Skipping {self.MODALITY_NAME.lower()} generation (--{self.SKIP_FLAG})")
            return

        # Generate files
        success_count, fail_count = self.generate_files()

        # Print completion
        self.print_completion(success_count, fail_count)
