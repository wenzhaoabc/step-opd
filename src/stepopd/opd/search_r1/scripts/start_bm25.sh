#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

: "${STEPOPD_ROOT:?STEPOPD_ROOT must point to the persistent project root}"
SEARCH_INDEX="$STEPOPD_ROOT/envs/search_r1/wiki-18-bm25-index/bm25"
SEARCH_CORPUS="$STEPOPD_ROOT/envs/search_r1/wiki-18-corpus/wiki-18.jsonl"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/verl"
export CUDA_VISIBLE_DEVICES=""
export BM25_JAVA_HOME="/opt/java21"

test -d "$SEARCH_INDEX"
test -f "$SEARCH_CORPUS"
exec python -m stepopd.opd.search_r1.bm25 --index-path "$SEARCH_INDEX" --corpus-path "$SEARCH_CORPUS" \
    --host 127.0.0.1 --port 8000 "$@"
