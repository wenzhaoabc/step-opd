#!/usr/bin/env bash
set -euo pipefail
export CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
exec bash "$CODE_ROOT/src/stepopd/opsd/opsd_baseline_suite.sh" "$@"
