#!/bin/bash

set -u

task_config=${1:-demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer}
gpu_id=${2:-0}
continue_on_error=${CONTINUE_ON_ERROR:-1}
task_timeout_seconds=${TASK_TIMEOUT_SECONDS:-1800}

task_list=(
    beat_block_hammer_rotate_view
    blocks_ranking_rgb_fan_double
    blocks_ranking_rgb_rotate_view
    blocks_ranking_size_fan_double
    blocks_ranking_size_rotate_view
    click_alarmclock_rotate_view
    click_bell_rotate_view
    move_pillbottle_pad_rotate_view
    move_stapler_pad_rotate_view
    open_laptop_rotate_view
    place_a2b_left_rotate_view
    place_a2b_right_rotate_view
    place_cans_plasticbox_rotate_view
    place_container_plate_rotate_view
    place_empty_cup_rotate_view
    place_fan_rotate_view
    place_mouse_pad_rotate_view
    place_object_basket_fan_double
    place_object_scale_rotate_view
    place_object_stand_rotate_view
    place_shoe_rotate_view
    press_stapler_rotate_view
    put_block_breadbasket_fan_double
    put_block_on_upper_easy
    put_block_on_upper_hard
    put_block_plasticbox_fan_double
    put_block_skillet_fan_double
    search_object
    shake_bottle_horizontally_rotate_view
    shake_bottle_rotate_view
    stack_blocks_three_rotate_view
    stack_blocks_two_rotate_view
    stamp_seal_rotate_view
    turn_switch_rotate_view
)

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

failed_tasks=()

for task_name in "${task_list[@]}"; do
    echo "Collecting ${task_name} with ${task_config} on GPU ${gpu_id}"
    PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1 \
    timeout "${task_timeout_seconds}" python script/collect_data.py "${task_name}" "${task_config}"
    collect_status=$?

    rm -rf data/"${task_name}"/"${task_config}"*/.cache
    rm -rf data_view/"${task_name}"/"${task_config}"*/.cache

    if [[ ${collect_status} -ne 0 ]]; then
        echo "[failed] ${task_name} ${task_config} exit=${collect_status}"
        failed_tasks+=("${task_name}:${collect_status}")
        if [[ "${continue_on_error}" != "1" ]]; then
            exit ${collect_status}
        fi
    else
        echo "[ok] ${task_name}"
    fi
done

if [[ ${#failed_tasks[@]} -gt 0 ]]; then
    echo "DYJ 34-task collection finished with failures:"
    for item in "${failed_tasks[@]}"; do
        echo "  ${item}"
    done
    exit 1
fi

echo "DYJ 34-task collection finished successfully."
