#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f Scripts/experiments/BaselineFingerprint.py ]] || {
  echo "Run this from the repository root."
  exit 1
}

set -a
[[ -f .env ]] && source .env
set +a

: "${TINKER_API_KEY:?TINKER_API_KEY must be set}"

PY="${PY:-python}"
PROBE_DATASET="${PROBE_DATASET:-HDSv2}"
RUN_TRAPS="${RUN_TRAPS:-0}"
RUN_APPENDIX="${RUN_APPENDIX:-0}"
RUN_FIGURES="${RUN_FIGURES:-1}"

# Use local HF cache; set HF_HUB_OFFLINE=0 if you need network fetches.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

JOB_PIDS=()
JOB_LABELS=()

run_fp() {
  local hf_model="$1"
  local dataset="$2"
  local split="$3"
  local profile="$4"

  local args=(
    Scripts/experiments/BaselineFingerprint.py
    --model "$hf_model"
    --dataset "$dataset"
    --split "$split"
    --modality image
    --batch-size 30
    --async
    --concurrency 4
    --template-mode multi
    --template-profile "$profile"
  )

  [[ "$profile" != "balanced" ]] && args+=(--output-tag "$profile")
  "$PY" "${args[@]}"
}

launch_fp_job() {
  local label="$1"
  shift

  echo "Launching: $label"
  run_fp "$@" &
  JOB_PIDS+=("$!")
  JOB_LABELS+=("$label")
}

cleanup_background_jobs() {
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

wait_for_jobs_fail_fast() {
  local pending=()
  local next_pending=()
  local idx
  local pid
  local label
  local status
  local progressed

  for ((idx = 0; idx < ${#JOB_PIDS[@]}; idx++)); do
    pending+=("$idx")
  done

  while [[ ${#pending[@]} -gt 0 ]]; do
    next_pending=()
    progressed=0

    for idx in "${pending[@]}"; do
      pid="${JOB_PIDS[$idx]}"
      label="${JOB_LABELS[$idx]}"

      if kill -0 "$pid" 2>/dev/null; then
        next_pending+=("$idx")
        continue
      fi

      progressed=1
      if wait "$pid"; then
        echo "Completed: $label"
      else
        status=$?
        echo "Failed: $label (exit $status)" >&2
        cleanup_background_jobs
        for pid in "${JOB_PIDS[@]}"; do
          wait "$pid" 2>/dev/null || true
        done
        return "$status"
      fi
    done

    pending=()
    for idx in "${next_pending[@]:-}"; do
      [[ -n "${idx:-}" ]] && pending+=("$idx")
    done
    if [[ ${#pending[@]} -gt 0 && "$progressed" -eq 0 ]]; then
      sleep 5
    fi
  done
}

run_macros() {
  local slug="$1"
  local fragment="$2"

  "$PY" Scripts/analysis/GenerateResultsFigures.py \
    --model "$slug" \
    --output-type fingerprint-macros \
    --probe-hds-dataset "$PROBE_DATASET" \
    --output-path "$fragment"
}

run_appendix() {
  local slug="$1"
  local fragment="$2"

  "$PY" Scripts/analysis/GenerateResultsFigures.py \
    --model "$slug" \
    --output-type fingerprint-appendix \
    --probe-hds-dataset "$PROBE_DATASET" \
    --output-path "$fragment"
}

trap cleanup_background_jobs EXIT INT TERM

models=(
  "Qwen/Qwen3-VL-30B-A3B-Instruct|Qwen3-VL-30B-A3B|PaperTexFolder/Figures/results_macros.30b.fingerprint.tex|PaperTexFolder/Figures/appendix_template_variability.30b.tex"
  "Qwen/Qwen3-VL-235B-A22B-Instruct|Qwen3-VL-235B-A22B|PaperTexFolder/Figures/results_macros.235b.fingerprint.tex|PaperTexFolder/Figures/appendix_template_variability.235b.tex"
)

for spec in "${models[@]}"; do
  IFS='|' read -r hf_model slug macro_fragment appendix_fragment <<< "$spec"
  launch_fp_job "$slug image balanced" "$hf_model" "$PROBE_DATASET" test balanced
  launch_fp_job "$slug image style_mismatch" "$hf_model" "$PROBE_DATASET" test style_mismatch
done

wait_for_jobs_fail_fast
JOB_PIDS=()
JOB_LABELS=()

for spec in "${models[@]}"; do
  IFS='|' read -r hf_model slug macro_fragment appendix_fragment <<< "$spec"

  if [[ "$RUN_TRAPS" == "1" ]]; then
    run_fp "$hf_model" Trapsv2 all balanced
  fi

  run_macros "$slug" "$macro_fragment"

  if [[ "$RUN_APPENDIX" == "1" ]]; then
    run_appendix "$slug" "$appendix_fragment"
  fi
done

"$PY" Scripts/analysis/GenerateResultsFigures.py \
  --output-type merge-macros \
  --output-path PaperTexFolder/Figures/results_macros.tex \
  --fragment-path PaperTexFolder/Figures/results_macros.30b.fingerprint.tex \
  --fragment-path PaperTexFolder/Figures/results_macros.30b.nudge.tex \
  --fragment-path PaperTexFolder/Figures/results_macros.30b.gradient.tex \
  --fragment-path PaperTexFolder/Figures/results_macros.235b.fingerprint.tex \
  --fragment-path PaperTexFolder/Figures/results_macros.235b.nudge.tex \
  --fragment-path PaperTexFolder/Figures/results_macros.235b.gradient.tex

if [[ "$RUN_FIGURES" == "1" ]]; then
  "$PY" Scripts/analysis/GenerateResultsFigures.py \
    --model Qwen3-VL-30B-A3B \
    --output-type fingerprint-figures \
    --probe-hds-dataset "$PROBE_DATASET"
fi

if [[ "$RUN_APPENDIX" == "1" ]]; then
  "$PY" Scripts/analysis/GenerateResultsFigures.py \
    --output-type merge-fingerprint-appendix \
    --output-path PaperTexFolder/Figures/appendix_template_variability.tex \
    --fragment-path PaperTexFolder/Figures/appendix_template_variability.30b.tex \
    --fragment-path PaperTexFolder/Figures/appendix_template_variability.235b.tex
fi
