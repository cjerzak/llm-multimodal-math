"""Logging utilities with timestamps."""
from datetime import datetime
from pathlib import Path

# Repository root for Logs folder
_SCRIPT_DIR = Path(__file__).parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
REPO_ROOT = _SCRIPTS_DIR.parent
LOGS_DIR = REPO_ROOT / "Logs"


def tprint(*args, **kwargs):
    """Print with timestamp prefix.

    Works exactly like print() but prepends [HH:MM:SS] timestamp.

    Examples:
        tprint("Starting process")  # [14:32:05] Starting process
        tprint("Value:", 42)        # [14:32:05] Value: 42
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    kwargs.setdefault("flush", True)
    print(f"[{timestamp}]", *args, **kwargs)


def log_detail(filename: str, *args, **kwargs):
    """Write detailed info to Logs/{filename} instead of stdout.

    Use this for verbose per-call information that would clutter the main log.
    Details are preserved for debugging but don't spam stdout.

    Args:
        filename: Log filename (e.g., "image_generation.log")
        *args: Arguments to log (same as print)
        **kwargs: Keyword arguments (passed to print, except 'file')

    Examples:
        log_detail("api_calls.log", "Request:", data)
        log_detail("image_generation.log", f"{path}: {text[:200]}")
    """
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / filename

    # Remove 'file' from kwargs if present (we're setting it ourselves)
    kwargs.pop('file', None)

    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_path, 'a') as f:
        print(f"[{timestamp}]", *args, file=f, **kwargs)
