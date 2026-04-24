#!/usr/bin/env python3
"""
GenerateMathImages.py

Generates multiplication problem dataset with LaTeX-rendered images.
Creates a CSV dataset and PNG images for multi-modal LLM testing.

Usage:
    python Scripts/GenerateMathImages.py --count 1000
    python Scripts/GenerateMathImages.py --count 500 --split-ratios 80/10/10
    python Scripts/GenerateMathImages.py --count 1000 --max-workers 8
"""

import subprocess
import sys
import tempfile
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

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    tprint("Warning: pdf2image not installed. Run: pip install pdf2image")


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


class ImageGenerator(MultiModalGenerator):
    """Generator for LaTeX-rendered multiplication problem images."""

    MODALITY_NAME = "Image"
    CSV_PATH = REPO_ROOT / "SavedData" / "ImageGrid.csv"
    OUTPUT_DIR = REPO_ROOT / "SavedData" / "Images"
    FILE_EXTENSION = "png"
    ID_PREFIX = "img"
    SKIP_FLAG = "skip-images"

    def generate_item(self, row: Dict, output_path: Path) -> bool:
        """Generate a PNG image for a multiplication problem."""
        latex_content = generate_latex(row["a"], row["b"])
        return compile_latex_to_png(latex_content, output_path)

    def check_dependencies(self) -> bool:
        """Check for pdflatex and pdf2image."""
        all_ok = True

        # Check pdflatex
        try:
            subprocess.run(["pdflatex", "--version"], capture_output=True, timeout=5)
            tprint("pdflatex: OK")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tprint("pdflatex: NOT FOUND - Please install a TeX distribution (e.g., MacTeX, TeX Live)")
            all_ok = False

        # Check pdf2image
        if PDF2IMAGE_AVAILABLE:
            tprint("pdf2image: OK")
        else:
            tprint("pdf2image: NOT FOUND - Run: pip install pdf2image")
            all_ok = False

        return all_ok


def main():
    """Main entry point."""
    generator = ImageGenerator()
    generator.run()


if __name__ == "__main__":
    main()
