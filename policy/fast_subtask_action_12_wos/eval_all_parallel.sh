#!/bin/bash
set -u -o pipefail

POLICY_NAME=fast_subtask_action_12_wos
TASK_CONFIG=${1:-info_gathering_demo}
CKPT_SETTING=${2:-fast_subtask_action_12_wos}
SEED=${3:-0}
PORT=${4:-7901}

# GPU 0 is reserved for the StarVLA policy server.
# Override if needed: GPU_LIST="1 2 3" bash policy/fast_subtask_action_12_wos/eval_all_parallel.sh
GPU_LIST=${GPU_LIST:-"1 2 3 4 5 6"}
HISTORY_FRAMES=${HISTORY_FRAMES:-12}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

TASKS=(
    # Single-layer rotate-view tasks.
    beat_block_hammer_rotate_view
    blocks_ranking_rgb_rotate_view
    blocks_ranking_size_rotate_view
    click_bell_rotate_view
    move_pillbottle_pad_rotate_view
    move_stapler_pad_rotate_view
    place_a2b_left_rotate_view
    place_a2b_right_rotate_view
    place_cans_plasticbox_rotate_view
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
    stack_blocks_two_rotate_view
    stamp_seal_rotate_view
    turn_switch_rotate_view

    # Double / fan-double / upper-layer tasks.
    blocks_ranking_rgb_fan_double
    blocks_ranking_size_fan_double
    place_object_basket_fan_double
    put_block_on_upper_easy
    put_block_on_upper_hard

    # Info-gathering tasks.
    count_target_press_button
    count_random_object_press_button
    count_color_kinds_press_button
    check_block_color
    check_cola_date
    check_cola_color
    rank_backside_rgb_blocks
    match_backside_two_blocks
)

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROBOTWIN_ROOT}/policy/${POLICY_NAME}/parallel_eval_logs/${RUN_ID}"
TASK_LOG_ROOT="${RUN_ROOT}/tasks"
mkdir -p "${TASK_LOG_ROOT}"

NEXT_INDEX_FILE="${RUN_ROOT}/next_index.txt"
QUEUE_LOCK="${RUN_ROOT}/queue.lock"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
SUMMARY_LOCK="${RUN_ROOT}/summary.lock"
STOP_FILE="${RUN_ROOT}/stop"

printf '0\n' > "${NEXT_INDEX_FILE}"
printf 'idx\ttask\tgpu\tstatus\tstart_time\tend_time\tseconds\tlog_path\n' > "${SUMMARY_FILE}"

echo "[parallel-eval] policy=${POLICY_NAME}"
echo "[parallel-eval] task_config=${TASK_CONFIG}, ckpt_setting=${CKPT_SETTING}, seed=${SEED}, port=${PORT}"
echo "[parallel-eval] gpus=${GPU_LIST} (GPU 0 reserved for server)"
echo "[parallel-eval] total_tasks=${#TASKS[@]}, run_root=${RUN_ROOT}"

get_next_task() {
    local idx
    {
        flock -x 200
        idx="$(cat "${NEXT_INDEX_FILE}")"
        if (( idx >= ${#TASKS[@]} )); then
            return 1
        fi
        printf '%s\n' "$((idx + 1))" > "${NEXT_INDEX_FILE}"
        printf '%s\t%s\n' "${idx}" "${TASKS[$idx]}"
    } 200>"${QUEUE_LOCK}"
}

append_summary() {
    local idx="$1"
    local task="$2"
    local gpu="$3"
    local status="$4"
    local start_time="$5"
    local end_time="$6"
    local seconds="$7"
    local log_path="$8"
    {
        flock -x 201
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${idx}" "${task}" "${gpu}" "${status}" \
            "${start_time}" "${end_time}" "${seconds}" "${log_path}" >> "${SUMMARY_FILE}"
    } 201>"${SUMMARY_LOCK}"
}

run_eval_task() {
    local gpu="$1"
    local idx="$2"
    local task="$3"
    local task_dir="${TASK_LOG_ROOT}/$(printf '%02d' "$((idx + 1))")_${task}"
    local stdout_log="${task_dir}/stdout.log"
    local request_log="${task_dir}/requests.jsonl"
    local image_dir="${task_dir}/images"
    local start_time
    local end_time
    local start_seconds
    local end_seconds
    local elapsed_seconds
    local status

    mkdir -p "${image_dir}"

    start_time="$(date '+%Y-%m-%d %H:%M:%S')"
    start_seconds="$(date +%s)"
    echo "[gpu ${gpu}] start $((idx + 1))/${#TASKS[@]} ${task}"

    local cmd=(
        python script/eval_policy.py
        --config "policy/${POLICY_NAME}/deploy_policy.yml"
        --overrides
        --task_name "${task}"
        --task_config "${TASK_CONFIG}"
        --ckpt_setting "${CKPT_SETTING}"
        --seed "${SEED}"
        --policy_name "${POLICY_NAME}"
        --port "${PORT}"
        --history_frames "${HISTORY_FRAMES}"
        --request_log_path "${request_log}"
        --request_image_dir "${image_dir}"
    )

    if [[ "${TASK_TIMEOUT_SECONDS}" != "0" ]]; then
        cmd=(timeout "${TASK_TIMEOUT_SECONDS}" "${cmd[@]}")
    fi

    if CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONWARNINGS=ignore::UserWarning \
        PYTHONUNBUFFERED=1 \
        "${cmd[@]}" > "${stdout_log}" 2>&1; then
        status=0
    else
        status=$?
    fi

    end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    end_seconds="$(date +%s)"
    elapsed_seconds="$((end_seconds - start_seconds))"
    append_summary "${idx}" "${task}" "${gpu}" "${status}" "${start_time}" "${end_time}" "${elapsed_seconds}" "${stdout_log}"

    if [[ "${status}" -eq 0 ]]; then
        echo "[gpu ${gpu}] ok ${task} (${elapsed_seconds}s)"
    else
        echo "[gpu ${gpu}] failed ${task} exit=${status} (${elapsed_seconds}s); log=${stdout_log}"
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
            touch "${STOP_FILE}"
        fi
    fi
}

worker() {
    local gpu="$1"
    local item
    local idx
    local task

    while [[ ! -f "${STOP_FILE}" ]]; do
        if ! item="$(get_next_task)"; then
            break
        fi
        idx="${item%%$'\t'*}"
        task="${item#*$'\t'}"
        run_eval_task "${gpu}" "${idx}" "${task}"
    done
    echo "[gpu ${gpu}] worker exit"
}

pids=()
for gpu in ${GPU_LIST}; do
    worker "${gpu}" &
    pids+=("$!")
done

trap 'echo "[parallel-eval] interrupted; stopping workers"; touch "${STOP_FILE}"; kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null; exit 130' INT TERM

overall_status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        overall_status=1
    fi
done

failed_count="$(awk -F '\t' 'NR > 1 && $4 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
done_count="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"

echo "[parallel-eval] finished ${done_count}/${#TASKS[@]} tasks"
echo "[parallel-eval] summary=${SUMMARY_FILE}"

if [[ "${failed_count}" != "0" ]]; then
    echo "[parallel-eval] failed_tasks=${failed_count}"
    awk -F '\t' 'NR > 1 && $4 != 0 {printf "  idx=%s task=%s gpu=%s status=%s log=%s\n", $1, $2, $3, $4, $8}' "${SUMMARY_FILE}"
    exit 1
fi

exit "${overall_status}"
