#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
[[ "$CODE_ROOT" == /* && -f "$CODE_ROOT/configs/env/container.sh" && -f "$CODE_ROOT/src/stepopd/runtime/scripts/train_common.sh" ]] || {
    printf 'Invalid CODE_ROOT (project absolute path required): %s\n' "$CODE_ROOT" >&2; exit 2;
}
cd "$CODE_ROOT"
CODE_ROOT="$(pwd -P)"
export CODE_ROOT

environment=all
model=all
smoke=0
dry_run=0
rerun=0
selected=0
while (( $# )); do
    case "$1" in
        --environment=*) environment="${1#--environment=}"; selected=1 ;;
        --model=*) model="${1#--model=}"; selected=1 ;;
        --smoke) smoke=1; selected=1 ;;
        --rerun) rerun=1 ;;
        --dry-run) dry_run=1 ;;
        --) shift; break ;;
        *) printf 'Unknown argument: %s; use TRAIN_QUEUE for a custom queue\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
case "$environment" in all|alfworld|scienceworld|search_r1) ;; *) printf 'Unsupported environment: %s\n' "$environment" >&2; exit 2 ;; esac
case "$model" in all|1p7b|4b) ;; *) printf 'Unsupported model: %s\n' "$model" >&2; exit 2 ;; esac
if (( smoke )) && [[ "$model" == 4b ]]; then
    printf 'smoke uses 1.7B\n' >&2; exit 2
fi
environments=(alfworld scienceworld search_r1)
models=(1p7b 4b)
[[ "$environment" == all ]] || environments=("$environment")
[[ "$model" == all ]] || models=("$model")
(( smoke == 0 )) || models=(1p7b)
queue=()
for environment in "${environments[@]}"; do
    for model in "${models[@]}"; do
        name="opsd_qwen3_$model"
        (( smoke == 0 )) || name=smoke
        queue+=("src/stepopd/opsd/$environment/scripts/$name.yaml")
    done
done
if [[ -n "${TRAIN_QUEUE:-}" ]]; then
    (( selected == 0 )) || { printf 'TRAIN_QUEUE cannot be combined with --environment/--model/--smoke\n' >&2; exit 2; }
    read -r -a queue <<< "$TRAIN_QUEUE"
fi
read -r -a overrides <<< "${TRAIN_OVERRIDES:-}"
overrides+=("$@")
for override in ${overrides[@]+"${overrides[@]}"}; do
    case "$override" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--version|--multirun|-m|--resolve|--info|\
        --config-path|--config-path=*|-cp|-cp=*|--config-name|--config-name=*|-cn|-cn=*|\
        --shell-completion|-sc|--package|-p)
            printf 'Training overrides must not include Hydra control arguments: %s\n' "$override" >&2; exit 2 ;;
    esac
done
(( ${#queue[@]} )) || { printf 'Experiment list is empty\n' >&2; exit 2; }
for index in "${!queue[@]}"; do
    config="${queue[$index]}"
    [[ -f "$config" ]] || { printf 'Experiment YAML not found: %s\n' "$config" >&2; exit 2; }
    queue[$index]="$(cd "$(dirname "$config")" && pwd -P)/$(basename "$config")"
    [[ "${queue[$index]}" == "$CODE_ROOT/src/stepopd/opsd/"*/scripts/*.yaml ]] || {
        printf 'Use experiment configs from the opsd package: %s\n' "$config" >&2; exit 2;
    }
done

if (( dry_run == 0 )); then
    source "$CODE_ROOT/configs/env/container.sh"
else
    printf 'Base environment: source %q\n' "$CODE_ROOT/configs/env/container.sh"
fi
source "$CODE_ROOT/src/stepopd/runtime/scripts/train_common.sh"
DRY_RUN=$dry_run
start_run OPSD
total=${#queue[@]}

suite_complete() {
    local phase="$1" rc=0
    if (( rerun )); then
        (( dry_run == 0 )) || printf '  --rerun given, ignoring completion marker (will run)\n'
        return 1
    fi
    if (( dry_run )); then
        printf 'Check:'; printf ' %q' python -m stepopd.opsd.suite_state check \
            --config "$config" --phase "$phase" ${overrides[@]+"${overrides[@]}"}; printf '\n'
    fi
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/verl" \
        python -m stepopd.opsd.suite_state check --config "$config" --phase "$phase" \
        ${overrides[@]+"${overrides[@]}"} || rc=$?
    if (( rc == 0 )); then
        if (( dry_run == 0 )); then
            printf 'Already complete, skipping %s: %s\n' "$phase" "$config"
        else
            printf '  Already complete (will skip)\n'
        fi
        return 0
    fi
    if (( dry_run )); then
        printf '  Not complete (will run)\n'
    elif (( rc != 1 )); then
        printf 'Completion state undetermined (rc=%d), treating as incomplete: %s %s\n' "$rc" "$config" "$phase" >&2
    fi
    return 1
}

executed=0
skipped=0
mark_failed=0
mark_rc=0
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
for phase in train eval; do
    printf '\nPhase: %s (%d experiments)\n' "$phase" "$total"
    for index in "${!queue[@]}"; do
        config="${queue[$index]}"
        begin_experiment "$config"
        if suite_complete "$phase"; then
            skipped=$(( skipped + 1 ))
            continue
        fi
        if [[ "$phase" == train ]]; then
            run_stage "$EXPERIMENT_DIR/train.log" python -m stepopd.opsd.train \
                --config-path "$(dirname "$config")" --config-name "$(basename "$config" .yaml)" \
                ${overrides[@]+"${overrides[@]}"}
        else
            run_stage "$EXPERIMENT_DIR/eval.log" bash "$CODE_ROOT/src/stepopd/eval/final_eval.sh" \
                --config "$config" --overrides ${overrides[@]+"${overrides[@]}"}
        fi
        executed=$(( executed + 1 ))
        if (( dry_run == 0 )); then
            PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/verl" \
                python -m stepopd.opsd.suite_state mark --config "$config" --phase "$phase" \
                ${overrides[@]+"${overrides[@]}"} || {
                    mark_rc=$?
                    printf 'Warning: failed to write completion marker for %s %s (rc=%d); this phase will rerun next time\n' \
                        "$config" "$phase" "$mark_rc" >&2
                    mark_failed=1
                }
        fi
    done
done
printf '\nTotal %d experiment phases: executed %d, skipped %d; logs: %s\n' \
    "$(( total * 2 ))" "$executed" "$skipped" "$RUN_DIR"
if (( mark_failed )); then
    printf 'Completion marker writes failed for some phases (see warnings above); those phases will rerun next time\n' >&2
    exit 1
fi
