_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STEPOPD_ROOT:=$(cd "$_script_dir/../.." && pwd)}"
export STEPOPD_ROOT

command -v python >/dev/null 2>&1 || { echo "[container.sh] Error: python not found" >&2; return 1 2>/dev/null || exit 1; }
export PYTHONPATH="$STEPOPD_ROOT/src:$STEPOPD_ROOT/verl"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1

export QWEN3_TOKENIZER="${QWEN3_TOKENIZER:-$STEPOPD_ROOT/../models/Qwen3-8B}"

export SEARCH_SOURCE_DIR="${SEARCH_SOURCE_DIR:-$STEPOPD_ROOT/envs/search_r1/source}"

if [ -n "${BM25_JAVA_HOME:-}" ]; then
    export BM25_JAVA_HOME
else
    echo "[container.sh] Hint: BM25_JAVA_HOME (Java 21) is not set; the Search-R1 BM25 precheck will fail" >&2
fi

cd "$STEPOPD_ROOT"
