#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${STEPOPD_ROOT:?final_eval requires STEPOPD_ROOT pointing to the data/output mount root}"
export PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"

exec python -m stepopd.eval.final_eval "$@"
