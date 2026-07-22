#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/eval_seed_whitelist_randomized.sh"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
PI05_CKPT_KEY=${PI05_CKPT_KEY:-pi05_astribot_global_step_100000}
EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}
PI05_MAX_ACTIONS_PER_CALL=${PI05_MAX_ACTIONS_PER_CALL:-16}
PI05_ACTION_SEMANTICS=${PI05_ACTION_SEMANTICS:-absolute}
PI05_LOG_REQUEST_DEBUG=${PI05_LOG_REQUEST_DEBUG:-False}
PI05_LOG_CHUNK_TIMING=${PI05_LOG_CHUNK_TIMING:-True}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-False}
TMUX_MONITOR=${TMUX_MONITOR:-0}
TASK_LIMIT=${TASK_LIMIT:-0}
DRY_RUN=${DRY_RUN:-0}

CLEAN_GPU_LIST=${CLEAN_GPU_LIST:-"0 1 2 3 4"}
RANDOMIZED_GPU_LIST=${RANDOMIZED_GPU_LIST:-"5 6 7 8"}
CLEAN_RUN_ID=${CLEAN_RUN_ID:-pi05_clean_100k_50ep16a_${STAMP}}
RANDOMIZED_RUN_ID=${RANDOMIZED_RUN_ID:-pi05_randomized_100k_50ep16a_${STAMP}}

RUN_ROOT_BASE="${SCRIPT_DIR}/runs/eval_seed_whitelist_randomized"
NOHUP_LOG_ROOT="${SCRIPT_DIR}/nohup_logs"
mkdir -p "${RUN_ROOT_BASE}" "${NOHUP_LOG_ROOT}"

launch_one() {
    local __pid_var="$1"
    local label="$2"
    local run_id="$3"
    local gpu_list="$4"
    local task_config="$5"
    local log_path="${NOHUP_LOG_ROOT}/${run_id}.log"

    echo "[pi05-full] starting ${label}"
    echo "[pi05-full] run_id=${run_id}"
    echo "[pi05-full] gpus=${gpu_list}"
    echo "[pi05-full] task_config=${task_config}"
    echo "[pi05-full] log=${log_path}"
    echo "[pi05-full] result_root=${RUN_ROOT_BASE}/${run_id}"

    cd "${ROBOTWIN_ROOT}"
    nohup setsid env \
        RUN_ID="${run_id}" \
        GPU_LIST="${gpu_list}" \
        TASK_CONFIG="${task_config}" \
        PI05_CKPT_KEY="${PI05_CKPT_KEY}" \
        EVAL_TEST_NUM="${EVAL_TEST_NUM}" \
        TASK_LIMIT="${TASK_LIMIT}" \
        PI05_MAX_ACTIONS_PER_CALL="${PI05_MAX_ACTIONS_PER_CALL}" \
        PI05_ACTION_SEMANTICS="${PI05_ACTION_SEMANTICS}" \
        PI05_LOG_REQUEST_DEBUG="${PI05_LOG_REQUEST_DEBUG}" \
        PI05_LOG_CHUNK_TIMING="${PI05_LOG_CHUNK_TIMING}" \
        ROBOTWIN_EVAL_VIDEO_LOG="${ROBOTWIN_EVAL_VIDEO_LOG}" \
        TMUX_MONITOR="${TMUX_MONITOR}" \
        DRY_RUN="${DRY_RUN}" \
        bash "${LAUNCHER}" \
        < /dev/null > "${log_path}" 2>&1 &
    printf -v "${__pid_var}" '%s' "$!"
    cd - >/dev/null
}

launch_one clean_pid clean "${CLEAN_RUN_ID}" "${CLEAN_GPU_LIST}" info_gathering_demo
launch_one randomized_pid randomized "${RANDOMIZED_RUN_ID}" "${RANDOMIZED_GPU_LIST}" info_gathering_randomized

cat <<EOF
[pi05-full] launched
clean_pid=${clean_pid}
randomized_pid=${randomized_pid}

clean_run=${RUN_ROOT_BASE}/${CLEAN_RUN_ID}
randomized_run=${RUN_ROOT_BASE}/${RANDOMIZED_RUN_ID}

clean_log=${NOHUP_LOG_ROOT}/${CLEAN_RUN_ID}.log
randomized_log=${NOHUP_LOG_ROOT}/${RANDOMIZED_RUN_ID}.log

tail -f ${NOHUP_LOG_ROOT}/${CLEAN_RUN_ID}.log
tail -f ${NOHUP_LOG_ROOT}/${RANDOMIZED_RUN_ID}.log
EOF
