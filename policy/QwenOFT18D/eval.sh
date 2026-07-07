#!/usr/bin/env bash
set -euo pipefail


task_list2=(
beat_block_hammer_rotate_view
blocks_ranking_rgb_rotate_view
blocks_ranking_size_rotate_view
click_alarmclock_rotate_view
click_bell_rotate_view
move_pillbottle_pad_rotate_view
move_stapler_pad_rotate_view
open_laptop_rotate_view
place_a2b_left_rotate_view
place_a2b_right_rotate_view
place_container_plate_rotate_view
place_empty_cup_rotate_view
place_fan_rotate_view
place_mouse_pad_rotate_view
place_object_scale_rotate_view
place_object_stand_rotate_view
place_shoe_rotate_view
press_stapler_rotate_view
shake_bottle_horizontally_rotate_view
shake_bottle_rotate_view
stack_blocks_three_rotate_view
stack_blocks_two_rotate_view
stamp_seal_rotate_view
turn_switch_rotate_view

)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_name=QwenOFT18D
task_config="${1:-demo_randomized_easy_ep200_r5}"
ckpt_setting="${2:-qwengroot18d}"
seed=0
gpu_id=0
test_num="${QWENGROOT18D_TEST_NUM:-10}"
python_bin="${ROBOTWIN_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mQwenOFT18D server: ${QWENGROOT18D_HOST:-127.0.0.1}:${QWENGROOT18D_PORT:-5702}\033[0m"
echo -e "\033[33mtest_num: ${test_num}\033[0m"

cd "${REPO_ROOT}"

for task_name in "${task_list2[@]}"; do
    PYTHONWARNINGS=ignore::UserWarning "${python_bin}" script/eval_policy.py \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --test_num "${test_num}" \
    --policy_name "${policy_name}"
done


