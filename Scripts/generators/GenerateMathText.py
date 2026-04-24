#!/usr/bin/env python3
"""
GenerateMathText.py

Generates multiplication problem dataset with text-only prompts.
Creates a CSV dataset and individual text files for multi-modal LLM testing.

Usage:
    python Scripts/GenerateMathText.py --count 1000
    python Scripts/GenerateMathText.py --count 500 --split-ratios 80/10/10
"""

import sys
from pathlib import Path
from typing import Dict

# Paths (generators/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent

# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))

from core.GenerateMathHelpers import REPO_ROOT
from core.MultiModalGenerator import MultiModalGenerator
from core.Logging import tprint


def generate_text_prompt(a: int, b: int) -> str:
    """Generate natural language text prompt for a multiplication problem."""
    return f"What is {a} times {b}?"


def save_text_file(text: str, output_path: Path) -> bool:
    """
    Save text prompt to file.

    Args:
        text: Text content to save.
        output_path: Path for output text file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        return True
    except Exception:
        return False


class TextGenerator(MultiModalGenerator):
    """Generator for text-only multiplication problem prompts."""

    MODALITY_NAME = "Text"
    CSV_PATH = REPO_ROOT / "SavedData" / "TextGrid.csv"
    OUTPUT_DIR = REPO_ROOT / "SavedData" / "TextFiles"
    FILE_EXTENSION = "txt"
    ID_PREFIX = "txt"
    SKIP_FLAG = "skip-text-files"

    def generate_item(self, row: Dict, output_path: Path) -> bool:
        """Generate a text file for a multiplication problem."""
        text_content = generate_text_prompt(row["a"], row["b"])
        return save_text_file(text_content, output_path)


def main():
    """Main entry point."""
    generator = TextGenerator()
    generator.run()


if __name__ == "__main__":
    main()
