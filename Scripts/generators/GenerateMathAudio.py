#!/usr/bin/env python3
"""
GenerateMathAudio.py

Generates multiplication problem dataset with gTTS-rendered audio files.
Creates a CSV dataset and MP3 files for multi-modal LLM testing.

Usage:
    python Scripts/GenerateMathAudio.py --count 1000
    python Scripts/GenerateMathAudio.py --count 500 --split-ratios 80/10/10
    python Scripts/GenerateMathAudio.py --count 1000 --max-workers 16
"""

import subprocess
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

# Try to import TTS libraries
GTTS_AVAILABLE = False
MACOS_SAY_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    pass

# Check for macOS 'say' command (offline fallback)
try:
    result = subprocess.run(["which", "say"], capture_output=True, timeout=5)
    MACOS_SAY_AVAILABLE = result.returncode == 0
except Exception:
    pass

# Check for ffmpeg (for AIFF→MP3 conversion)
FFMPEG_AVAILABLE = False
try:
    result = subprocess.run(["which", "ffmpeg"], capture_output=True, timeout=5)
    FFMPEG_AVAILABLE = result.returncode == 0
except Exception:
    pass

if not GTTS_AVAILABLE and not MACOS_SAY_AVAILABLE:
    tprint("Warning: No TTS available. Install gTTS (pip install gTTS) or use macOS.")


def generate_audio_gtts(text: str, output_path: Path) -> bool:
    """Generate audio using gTTS (online, Google TTS)."""
    if not GTTS_AVAILABLE:
        return False
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(str(output_path))
        # Verify file is valid (gTTS sometimes creates tiny corrupt files on rate limit)
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True
        # Delete corrupt file
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception:
        # Clean up any partial file
        if output_path.exists():
            output_path.unlink()
        return False


def convert_aiff_to_mp3(aiff_path: Path, mp3_path: Path) -> bool:
    """Convert AIFF to MP3 using ffmpeg, then delete original."""
    if not FFMPEG_AVAILABLE:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff_path), "-q:a", "2", str(mp3_path)],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0 and mp3_path.exists() and mp3_path.stat().st_size > 1000:
            aiff_path.unlink()  # Delete original AIFF
            return True
        return False
    except Exception:
        return False


def generate_audio_macos_say(text: str, output_path: Path) -> bool:
    """Generate audio using macOS 'say', convert AIFF→MP3 if ffmpeg available."""
    if not MACOS_SAY_AVAILABLE:
        return False

    # Generate to temp AIFF file (macOS say only outputs AIFF natively)
    aiff_path = output_path.with_suffix('.aiff')

    try:
        result = subprocess.run(
            ["say", "-o", str(aiff_path), text],
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0 or not aiff_path.exists() or aiff_path.stat().st_size < 1000:
            if aiff_path.exists():
                aiff_path.unlink()
            return False

        # Convert AIFF to MP3 if ffmpeg is available
        if FFMPEG_AVAILABLE and output_path.suffix.lower() == '.mp3':
            if convert_aiff_to_mp3(aiff_path, output_path):
                return True
            # Conversion failed, clean up
            if aiff_path.exists():
                aiff_path.unlink()
            return False

        # No ffmpeg: only allow AIFF outputs to avoid mislabeled files
        if output_path.suffix.lower() == '.mp3':
            if aiff_path.exists():
                aiff_path.unlink()
            return False

        if aiff_path != output_path:
            aiff_path.rename(output_path)
        return output_path.exists() and output_path.stat().st_size > 1000

    except Exception:
        if aiff_path.exists():
            aiff_path.unlink()
        return False


def generate_audio(a: int, b: int, output_path: Path) -> bool:
    """
    Generate audio file for a multiplication problem.

    Tries gTTS first (online), falls back to macOS 'say' (offline) if that fails.

    Args:
        a: First operand.
        b: Second operand.
        output_path: Path for output file.

    Returns:
        True if successful, False otherwise.
    """
    if not GTTS_AVAILABLE and not MACOS_SAY_AVAILABLE:
        return False

    text = f"{a} times {b} equals what?"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try gTTS first (better quality, but online)
    if GTTS_AVAILABLE:
        if generate_audio_gtts(text, output_path):
            return True

    # Fall back to macOS say (offline, works when rate-limited)
    if MACOS_SAY_AVAILABLE:
        if generate_audio_macos_say(text, output_path):
            return True

    return False


class AudioGenerator(MultiModalGenerator):
    """Generator for TTS-rendered multiplication problem audio files."""

    MODALITY_NAME = "Audio"
    CSV_PATH = REPO_ROOT / "SavedData" / "AudioGrid.csv"
    OUTPUT_DIR = REPO_ROOT / "SavedData" / "AudioFiles"
    FILE_EXTENSION = "mp3"  # gTTS outputs mp3 directly; macOS say → aiff → ffmpeg → mp3
    ID_PREFIX = "aud"
    SKIP_FLAG = "skip-audio"

    def generate_item(self, row: Dict, output_path: Path) -> bool:
        """Generate an MP3 audio file for a multiplication problem."""
        return generate_audio(row["a"], row["b"], output_path)

    def check_dependencies(self) -> bool:
        """Check for TTS libraries and ffmpeg."""
        has_any = False

        if GTTS_AVAILABLE:
            tprint("gTTS: OK (online TTS, outputs MP3)")
            has_any = True
        else:
            tprint("gTTS: NOT FOUND (rate-limited or not installed)")

        if MACOS_SAY_AVAILABLE:
            tprint("macOS say: OK (offline fallback, outputs AIFF)")
            has_any = True
        else:
            tprint("macOS say: NOT FOUND (not on macOS)")

        if FFMPEG_AVAILABLE:
            tprint("ffmpeg: OK (AIFF→MP3 conversion)")
        else:
            tprint("ffmpeg: NOT FOUND (macOS say will output AIFF instead of MP3)")
            if MACOS_SAY_AVAILABLE and not GTTS_AVAILABLE:
                self.FILE_EXTENSION = "aiff"
                tprint("Using .aiff output extension for macOS 'say' (no ffmpeg available).")

        if not has_any:
            tprint("\nNo TTS available. Install gTTS (pip install gTTS) or use macOS.")
            return False

        return True


def main():
    """Main entry point."""
    generator = AudioGenerator()
    generator.run()


if __name__ == "__main__":
    main()
