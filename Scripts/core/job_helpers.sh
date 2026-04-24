#!/usr/bin/env bash
# job_helpers.sh - Lock-based job coordination for parallel pipeline execution
#
# This script provides helper functions for coordinating parallel jobs using
# sentinel files. Jobs wait only for their specific dependencies rather than
# waiting for entire phases to complete.
#
# Usage:
#   source Scripts/core/job_helpers.sh
#   init_locks
#   run_job "job_id" "dep1,dep2" command arg1 arg2
#
# Lock directory structure:
#   ./Tmp/locks/
#   ├── *.done     # Completion sentinels (created on success)
#   └── FAILED     # Global failure flag (created on any error)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOCK_DIR="${LOCK_DIR:-./Tmp/locks}"
FAILED_FLAG="$LOCK_DIR/FAILED"
POLL_INTERVAL="${POLL_INTERVAL:-2}"  # Seconds between dependency checks

# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize lock directory (call once at pipeline start)
init_locks() {
    rm -rf "$LOCK_DIR"
    mkdir -p "$LOCK_DIR"
    echo "[INIT] Lock directory initialized: $LOCK_DIR"
}

# Check if any job has failed
check_failed() {
    [[ -f "$FAILED_FLAG" ]]
}

# Signal global failure (stops all waiting jobs)
signal_failure() {
    touch "$FAILED_FLAG"
    echo "[FAIL] Global failure flag set"
}

# Mark a job as complete
mark_done() {
    local job_id="$1"
    touch "$LOCK_DIR/${job_id}.done"
}

# Check if a specific job is complete
is_done() {
    local job_id="$1"
    [[ -f "$LOCK_DIR/${job_id}.done" ]]
}

# Wait for multiple dependencies to complete
# Returns 0 if all deps complete, 1 if failure detected
wait_for() {
    local deps=("$@")

    for dep in "${deps[@]}"; do
        local lockfile="$LOCK_DIR/${dep}.done"
        while [[ ! -f "$lockfile" ]]; do
            if check_failed; then
                return 1
            fi
            sleep "$POLL_INTERVAL"
        done
    done
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN JOB WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

# Run a job with dependency coordination
# Usage: run_job <job_id> <deps_csv> <command> [args...]
#
# Arguments:
#   job_id   - Unique identifier for this job (used for lock file)
#   deps_csv - Comma-separated list of job_ids this depends on, or "none"
#   command  - The command to run
#   args...  - Arguments to pass to the command
#
# Behavior:
#   1. Check if pipeline has already failed (skip if so)
#   2. Wait for all dependencies to complete
#   3. Run the command
#   4. On success: mark job done
#   5. On failure: signal global failure
run_job() {
    local job_id="$1"
    local deps_str="$2"
    shift 2

    # Timestamp for logging
    local start_time
    start_time=$(date "+%H:%M:%S")

    # Check if already failed before starting
    if check_failed; then
        echo "[$start_time] [$job_id] SKIPPED - pipeline already failed"
        return 1
    fi

    # Wait for dependencies
    if [[ -n "$deps_str" && "$deps_str" != "none" ]]; then
        IFS=',' read -ra deps <<< "$deps_str"
        echo "[$start_time] [$job_id] Waiting for: ${deps[*]}"

        if ! wait_for "${deps[@]}"; then
            echo "[$(date "+%H:%M:%S")] [$job_id] ABORTED - dependency failed"
            return 1
        fi

        echo "[$(date "+%H:%M:%S")] [$job_id] Dependencies ready"
    fi

    # Add random jitter (0-5 seconds) to avoid race conditions in concurrent initialization
    # This prevents threading deadlocks when multiple Tinker clients start simultaneously
    local jitter_secs=$(( RANDOM % 6 ))
    if [[ $jitter_secs -gt 0 ]]; then
        echo "[$(date "+%H:%M:%S")] [$job_id] Jitter delay: ${jitter_secs}s"
        sleep "$jitter_secs"
    fi

    # Run the actual command
    echo "[$(date "+%H:%M:%S")] [$job_id] STARTING..."

    if "$@"; then
        mark_done "$job_id"
        echo "[$(date "+%H:%M:%S")] [$job_id] COMPLETE"
        return 0
    else
        local exit_code=$?
        echo "[$(date "+%H:%M:%S")] [$job_id] FAILED (exit code: $exit_code)"
        signal_failure
        return $exit_code
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Parse a job specification string "job_id|deps|command args..."
# and execute it with run_job
# Usage: exec_job_spec "job_id|deps|command args..."
exec_job_spec() {
    local spec="$1"

    # Parse the pipe-delimited spec
    local job_id deps cmd
    IFS='|' read -r job_id deps cmd <<< "$spec"

    # Execute using run_job
    # Note: cmd is passed as a single string and needs eval for proper expansion
    run_job "$job_id" "$deps" bash -c "$cmd"
}

# List all completed jobs
list_completed() {
    if [[ -d "$LOCK_DIR" ]]; then
        find "$LOCK_DIR" -name "*.done" -exec basename {} .done \; | sort
    fi
}

# List all pending jobs (from a job list)
# Usage: list_pending "job1|deps1|cmd1" "job2|deps2|cmd2" ...
list_pending() {
    for spec in "$@"; do
        local job_id
        IFS='|' read -r job_id _ _ <<< "$spec"
        if ! is_done "$job_id"; then
            echo "$job_id"
        fi
    done
}

# Print pipeline status
print_status() {
    echo ""
    echo "═══ PIPELINE STATUS ═══"
    echo "Lock directory: $LOCK_DIR"

    if check_failed; then
        echo "Status: FAILED"
    else
        local completed
        completed=$(find "$LOCK_DIR" -name "*.done" 2>/dev/null | wc -l | tr -d ' ')
        echo "Status: RUNNING"
        echo "Completed jobs: $completed"
    fi

    if [[ -d "$LOCK_DIR" ]]; then
        echo ""
        echo "Completed:"
        list_completed | sed 's/^/  /'
    fi
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# DRY RUN SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

# Dry-run version of run_job (prints what would happen)
run_job_dry() {
    local job_id="$1"
    local deps_str="$2"
    shift 2

    echo "[DRY-RUN] $job_id"
    if [[ -n "$deps_str" && "$deps_str" != "none" ]]; then
        echo "  Depends on: $deps_str"
    else
        echo "  Depends on: (none - starts immediately)"
    fi
    echo "  Command: $*"
    echo ""
}
