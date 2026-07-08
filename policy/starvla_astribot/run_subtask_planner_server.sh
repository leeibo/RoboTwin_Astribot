#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARGVLA_ROOT="${STARGVLA_REPO_ROOT:-/data/lmz/code/starVLA-A}"
PYTHON_BIN="${STARGVLA_PYTHON:-conda run -n starVLA python}"
MODEL_PATH="${SUBTASK_PLANNER_MODEL_PATH:-${STARGVLA_ROOT}/playground/Pretrained_models/Qwen3-VL-30B-A3B-Instruct}"
HOST="${SUBTASK_PLANNER_HOST:-127.0.0.1}"
PORT="${SUBTASK_PLANNER_PORT:-7991}"

cd "${SCRIPT_DIR}"
${PYTHON_BIN} subtask_planner_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --model-path "${MODEL_PATH}" \
    "$@"
