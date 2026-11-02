#!/usr/bin/env bash
: "${STEPOPD_ROOT:?STEPOPD_ROOT must point to the data and output root}"
[[ "$CODE_ROOT" == /* && -d "$CODE_ROOT/src/stepopd" && -d "$CODE_ROOT/verl" ]] || {
    printf 'CODE_ROOT must be an absolute project path containing src/stepopd and verl: %s\n' "$CODE_ROOT" >&2; exit 2;
}
cd "$CODE_ROOT"
CODE_ROOT="$(pwd -P)"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/verl"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
tensorboard_prefix="${TENSORBOARD_DIR:-}"
export -n tensorboard_prefix
DRY_RUN=0
ACTIVE_PID=''

cleanup_run() {
    local rc=$? stage_rc=0
    trap - EXIT
    trap '' INT TERM
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" || stage_rc=$?
        if (( rc == 0 && stage_rc != 0 )); then rc=$stage_rc; fi
        ACTIVE_PID=''
    fi
    exit "$rc"
}

start_run() {
    METHOD="$1"
    local run_id="$(date +%Y%m%d-%H%M%S)-$$" category
    category="$(printf '%s' "$METHOD" | tr '[:upper:]' '[:lower:]')"
    RUN_DIR="$STEPOPD_ROOT/outputs/$category/suites/$run_id"
    started_at=$(date +%s)
    (( DRY_RUN == 0 )) || return 0
    mkdir -p "$RUN_DIR"
    trap cleanup_run EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

begin_experiment() {
    local config="$1" name environment elapsed eta='estimated after the first experiment finishes'
    name="$(basename "$config" .yaml)"
    environment="$(basename "$(dirname "$(dirname "$config")")")"
    EXPERIMENT_DIR="$RUN_DIR/${environment}_$name"
    if [[ -n "$tensorboard_prefix" ]]; then
        export TENSORBOARD_DIR="${tensorboard_prefix%/}/$METHOD/${environment}_$name"
    else
        export TENSORBOARD_DIR="$EXPERIMENT_DIR/tensorboard"
    fi
    elapsed=$(( $(date +%s) - started_at ))
    if (( index )); then eta="$(( elapsed * (total - index) / index / 60 )) min (rough estimate)"; fi
    printf '\n[%d/%d] %s/%s_%s | elapsed %ds | remaining ETA %s\n' "$((index + 1))" "$total" "$METHOD" "$environment" "$name" "$elapsed" "$eta"
}

run_stage() {
    local log="$1" rc=0
    shift
    printf 'Run:'; printf ' %q' "$@"; printf '\n'
    (( DRY_RUN == 0 )) || return 0
    python -m stepopd.runtime.stage "$log" "$@" &
    ACTIVE_PID=$!
    wait "$ACTIVE_PID" || rc=$?
    ACTIVE_PID=''
    return "$rc"
}
