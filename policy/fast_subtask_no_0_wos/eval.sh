#!/usr/bin/env bash
set -euo pipefail

policy_name="$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
task_name=${1:?task_name}
task_config=${2:?task_config}
ckpt_setting=${3:-${policy_name}}
seed=${4:-0}
gpu_id=${5:-0}
port=${6:-7980}
python_bin=${ROBOTWIN_PYTHON:-${ROBOTWIN_PY:-python}}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (env side): ${gpu_id}\033[0m"
echo -e "\033[33mStarVLA policy server port: ${port}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

PYTHONWARNINGS=ignore::UserWarning "${python_bin}" script/eval_policy.py \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --port "${port}"

