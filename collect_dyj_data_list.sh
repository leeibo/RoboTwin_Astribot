#!/bin/bash

set -u

task_config=${1:-auto}
gpu_id=${2:-0}

fan_config=${DYJ_FAN_CONFIG:-demo_randomized}
fan_double_config=${DYJ_FAN_DOUBLE_CONFIG:-${fan_config}}
continue_on_error=${CONTINUE_ON_ERROR:-1}

if [[ "${task_config}" == "-h" || "${task_config}" == "--help" ]]; then
    echo "Usage: bash collect_dyj_data_list.sh [task_config|auto] [gpu_id]"
    echo ""
    echo "Default:"
    echo "  bash collect_dyj_data_list.sh"
    echo "  Uses the same demo_randomized config for fan and fan_double tasks."
    echo ""
    echo "Override the shared config:"
    echo "  DYJ_FAN_CONFIG=demo_clean bash collect_dyj_data_list.sh auto 0"
    echo ""
    echo "Optional legacy per-group override:"
    echo "  DYJ_FAN_CONFIG=demo_clean DYJ_FAN_DOUBLE_CONFIG=demo_clean_double bash collect_dyj_data_list.sh auto 0"
    echo ""
    echo "Force one config for every task:"
    echo "  bash collect_dyj_data_list.sh demo_clean_fan_double_test 0"
    exit 0
fi

if [[ "${task_config}" != "auto" ]]; then
    fan_config=${task_config}
    fan_double_config=${task_config}
fi

search_task_list=(
    search_object
    search_object_left
    search_object_right
)

fan_double_task_list=(
    put_block_on
    put_block_on_upper_easy
    put_block_on_upper_hard
    put_block_on_lower_easy
    put_block_on_lower_hard
    blocks_ranking_rgb_fan_double
    blocks_ranking_size_fan_double
    place_object_basket_fan_double
)

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

failed_tasks=()

run_collect() {
    local task_name=$1
    local config_name=$2

    echo "Collecting data for ${task_name} with ${config_name} on GPU ${gpu_id}"
    PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "${task_name}" "${config_name}"
    local collect_status=$?

    rm -rf data/"${task_name}"/"${config_name}"*/.cache

    if [[ ${collect_status} -ne 0 ]]; then
        echo "[failed] ${task_name} ${config_name} exit=${collect_status}"
        failed_tasks+=("${task_name}:${config_name}:${collect_status}")
        if [[ "${continue_on_error}" != "1" ]]; then
            exit ${collect_status}
        fi
    else
        echo "[ok] ${task_name} ${config_name}"
    fi
}

echo "DYJ collection started"
echo "  fan_config=${fan_config}"
echo "  fan_double_config=${fan_double_config}"
echo "  gpu_id=${gpu_id}"
echo "  continue_on_error=${continue_on_error}"

for task_name in "${search_task_list[@]}"; do
    run_collect "${task_name}" "${fan_config}"
done

for task_name in "${fan_double_task_list[@]}"; do
    run_collect "${task_name}" "${fan_double_config}"
done

if [[ ${#failed_tasks[@]} -gt 0 ]]; then
    echo "DYJ collection finished with failures:"
    for item in "${failed_tasks[@]}"; do
        echo "  ${item}"
    done
    exit 1
fi

echo "DYJ collection finished successfully."
