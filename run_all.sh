#!/usr/bin/env bash
# run_all.sh - convenience entrypoint for the multimodal multiplication pipeline
#
# Usage:
#   ./run_all.sh <machine> <phase> [dry-run] [--resume] [--skip-lora-training] [--model-size 30b|235b]
#
# Examples:
#   ./run_all.sh M4 all
#   ./run_all.sh M4 data
#   ./run_all.sh Studio experiment
#   ./run_all.sh M4 all 1
#   ./run_all.sh M4 all 0 --resume
#   ./run_all.sh M4 all 0 --skip-lora-training
#   ./run_all.sh M4 all 0 --model-size 30b

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load API keys from .env (silently, to avoid exposing secrets)
if [ -f .env ]; then
    set +x
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Machine tag (first positional arg)
export MachineTag=${1:-default}
export Phase=${2:-all}
export DryRun=${3:-0}

# Check for --skip-lora-training flag (can be any positional arg)
export SkipLoraTraining=0
export ResumePipeline=0
for arg in "$@"; do
    if [[ "$arg" == "--skip-lora-training" ]]; then
        export SkipLoraTraining=1
    elif [[ "$arg" == "--resume" ]]; then
        export ResumePipeline=1
    fi
done

# Check for --model-size flag (30b or 235b)
export ModelSize=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--model-size" ]]; then
        export ModelSize="$arg"
    fi
    prev_arg="$arg"
done

# Python command (prefer tm_env conda environment, allow override via $PYTHON)
DEFAULT_TM_ENV_MINICONDA="${HOME}/miniconda3/envs/tm_env/bin/python"
DEFAULT_TM_ENV_MINIFORGE="${HOME}/miniforge3/envs/tm_env/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
    echo "Using PYTHON override: $PYTHON"
elif [[ -x "$DEFAULT_TM_ENV_MINICONDA" ]]; then
    PYTHON="$DEFAULT_TM_ENV_MINICONDA"
elif [[ -x "$DEFAULT_TM_ENV_MINIFORGE" ]]; then
    PYTHON="$DEFAULT_TM_ENV_MINIFORGE"
    echo "WARNING: tm_env not found at $DEFAULT_TM_ENV_MINICONDA; using $PYTHON instead"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    echo "WARNING: tm_env not found at $DEFAULT_TM_ENV_MINICONDA or $DEFAULT_TM_ENV_MINIFORGE; falling back to python3 at $PYTHON"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
    echo "WARNING: tm_env not found and python3 missing; falling back to python at $PYTHON"
else
    echo "ERROR: No Python interpreter found. Create tm_env or set PYTHON to an interpreter." >&2
    exit 1
fi
export PYTHON

cleanup_sleep_guards() {
    if [[ -n "${PMSET_NOIDLE_PID:-}" ]]; then
        kill "$PMSET_NOIDLE_PID" >/dev/null 2>&1 || true
        wait "$PMSET_NOIDLE_PID" 2>/dev/null || true
    fi
    if command -v pkill >/dev/null 2>&1; then
        pkill caffeinate >/dev/null 2>&1 || true
    fi
}
trap cleanup_sleep_guards EXIT

# Kill old caffeinate instances (macOS sleep prevention)
if command -v pkill >/dev/null 2>&1; then
    pkill caffeinate >/dev/null 2>&1 || true
fi

# Start caffeinate (no sleeping during long runs) when available.
if command -v caffeinate >/dev/null 2>&1; then
    nohup caffeinate -i -u -m >/dev/null 2>&1 &
fi

# Start persistent noidle assertion on macOS when available.
if command -v pmset >/dev/null 2>&1; then
    pmset noidle >/dev/null 2>&1 & PMSET_NOIDLE_PID=$!
fi

CPU_COUNT=$("$PYTHON" -c 'import os; print(os.cpu_count() or 1)')

echo "═══ SYSTEM RESOURCES ═══"
echo "CPUs: $CPU_COUNT"

echo ""
echo "═══ CONFIGURATION ═══"
echo "Machine: $MachineTag"
echo "Phase: $Phase"
echo "DryRun: $DryRun"
if [[ "$SkipLoraTraining" == "1" ]]; then
    echo "Skip LoRA Training: yes"
fi
if [[ "$ResumePipeline" == "1" ]]; then
    echo "Resume: yes"
fi
if [[ -n "$ModelSize" ]]; then
    echo "Model Size: $ModelSize only"
fi
echo ""

CMD=("$PYTHON" run_all.py --run-pipeline --machine "$MachineTag" --phase "$Phase")

if [[ "$DryRun" == "1" ]]; then
    CMD+=(--dry-run)
fi

if [[ "$ResumePipeline" == "1" ]]; then
    CMD+=(--resume)
fi

if [[ "$SkipLoraTraining" == "1" ]]; then
    CMD+=(--skip-lora-training)
fi

if [[ -n "$ModelSize" ]]; then
    CMD+=(--model-size "$ModelSize")
fi

"${CMD[@]}"
