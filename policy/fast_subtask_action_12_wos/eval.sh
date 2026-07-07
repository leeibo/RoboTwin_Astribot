#!/bin/bash
set -euo pipefail

policy_name=fast_subtask_action_12_wos
task_name=${1}
task_config=${2}
ckpt_setting=${3:-fast_subtask_action_12_wos}
seed=${4:-0}
gpu_id=${5:-0}
port=${6:-7901}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (env side): ${gpu_id}\033[0m"
echo -e "\033[33mStarVLA policy server port: ${port}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --port ${port}
