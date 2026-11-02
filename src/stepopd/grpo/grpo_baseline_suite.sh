#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
[[ "$CODE_ROOT" == /* && -f "$CODE_ROOT/src/stepopd/runtime/scripts/train_common.sh" ]] || {
    printf 'Invalid CODE_ROOT (project absolute path required): %s\n' "$CODE_ROOT" >&2; exit 2;
}
source "$CODE_ROOT/src/stepopd/runtime/scripts/train_common.sh"
(( $# == 0 )) || { printf 'Use TRAIN_QUEUE to select experiments and TRAIN_OVERRIDES for parameter overrides\n' >&2; exit 2; }

queue=(
    src/stepopd/grpo/alfworld/scripts/grpo_qwen3_1p7b.yaml
    src/stepopd/grpo/alfworld/scripts/grpo_qwen3_4b.yaml
    src/stepopd/grpo/alfworld/scripts/grpo_qwen3_8b.yaml
    src/stepopd/grpo/scienceworld/scripts/grpo_qwen3_1p7b.yaml
    src/stepopd/grpo/scienceworld/scripts/grpo_qwen3_4b.yaml
    src/stepopd/grpo/scienceworld/scripts/grpo_qwen3_8b.yaml
    src/stepopd/grpo/search_r1/scripts/grpo_qwen3_1p7b.yaml
    src/stepopd/grpo/search_r1/scripts/grpo_qwen3_4b.yaml
    src/stepopd/grpo/search_r1/scripts/grpo_qwen3_8b.yaml
)
if [[ -n "${TRAIN_QUEUE:-}" ]]; then read -r -a queue <<< "$TRAIN_QUEUE"; fi
read -r -a overrides <<< "${TRAIN_OVERRIDES:-}"
(( ${#queue[@]} )) || { printf 'Experiment list is empty\n' >&2; exit 2; }
for index in "${!queue[@]}"; do
    config="${queue[$index]}"
    [[ -f "$config" ]] || { printf 'Experiment YAML not found: %s\n' "$config" >&2; exit 2; }
    queue[$index]="$(cd "$(dirname "$config")" && pwd -P)/$(basename "$config")"
    [[ "${queue[$index]}" == "$CODE_ROOT/src/stepopd/grpo/"*/scripts/*.yaml ]] || {
        printf 'Use experiment configs from the grpo package: %s\n' "$config" >&2; exit 2;
    }
done

start_run GRPO
total=${#queue[@]}
for index in "${!queue[@]}"; do
    config="${queue[$index]}"
    begin_experiment "$config"
    run_stage "$EXPERIMENT_DIR/train.log" python -m stepopd.opd.train \
        --config-path "$(dirname "$config")" --config-name "$(basename "$config" .yaml)" \
        ${overrides[@]+"${overrides[@]}"}
    run_stage "$EXPERIMENT_DIR/eval.log" bash "$CODE_ROOT/src/stepopd/eval/final_eval.sh" \
        --config "$config" --overrides ${overrides[@]+"${overrides[@]}"}
done
printf '\nAll %d experiments finished, logs: %s\n' "$total" "$RUN_DIR"
