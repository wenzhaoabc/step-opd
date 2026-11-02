#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
[[ "$CODE_ROOT" == /* && -f "$CODE_ROOT/src/stepopd/runtime/scripts/train_common.sh" ]] || {
    printf 'Invalid CODE_ROOT (project absolute path required): %s\n' "$CODE_ROOT" >&2; exit 2;
}
source "$CODE_ROOT/src/stepopd/runtime/scripts/train_common.sh"
if [[ $# == 1 && "$1" == --dry-run ]]; then
    DRY_RUN=1
elif (( $# )); then
    printf 'Usage: bash sft_baseline_suite.sh [--dry-run]\n' >&2; exit 2
fi

queue=(
    src/stepopd/sft/alfworld/scripts/sft_30b_to_1p7b.yaml
    src/stepopd/sft/alfworld/scripts/sft_30b_to_4b.yaml
    src/stepopd/sft/scienceworld/scripts/sft_30b_to_1p7b.yaml
    src/stepopd/sft/scienceworld/scripts/sft_30b_to_4b.yaml
    src/stepopd/sft/search_r1/scripts/sft_30b_to_1p7b.yaml
    src/stepopd/sft/search_r1/scripts/sft_30b_to_4b.yaml
)
if [[ -n "${TRAIN_QUEUE:-}" ]]; then read -r -a queue <<< "$TRAIN_QUEUE"; fi
read -r -a overrides <<< "${TRAIN_OVERRIDES:-}"
(( ${#queue[@]} )) || { printf 'Experiment list is empty\n' >&2; exit 2; }
for index in "${!queue[@]}"; do
    config="${queue[$index]}"
    [[ -f "$config" ]] || { printf 'Experiment YAML not found: %s\n' "$config" >&2; exit 2; }
    queue[$index]="$(cd "$(dirname "$config")" && pwd -P)/$(basename "$config")"
    [[ "${queue[$index]}" == "$CODE_ROOT/src/stepopd/sft/"*/scripts/*.yaml ]] || {
        printf 'Use experiment configs from the sft package: %s\n' "$config" >&2; exit 2;
    }
done

start_run SFT
total=${#queue[@]}
printf '\n[Collection & export] Collecting toward 1.2x the required successful samples, max 3 attempts per task; both students share the success data.\n'
run_stage "$RUN_DIR/teacher_collection.log" python -m stepopd.sft.collect \
    --configs "${queue[@]}" --overrides ${overrides[@]+"${overrides[@]}"}
started_at=$(date +%s)
for index in "${!queue[@]}"; do
    config="${queue[$index]}"
    begin_experiment "$config"
    run_stage "$EXPERIMENT_DIR/train.log" python -m stepopd.sft.train \
        --config-path "$(dirname "$config")" --config-name "$(basename "$config" .yaml)" \
        ${overrides[@]+"${overrides[@]}"}
    run_stage "$EXPERIMENT_DIR/eval.log" python -m stepopd.sft.evaluate \
        --config "$config" --overrides ${overrides[@]+"${overrides[@]}"}
done
if (( DRY_RUN )); then
    printf '\nDry run complete, %d experiments total; no tasks were launched.\n' "$total"
else
    printf '\nAll %d experiments finished, logs: %s\n' "$total" "$RUN_DIR"
fi
