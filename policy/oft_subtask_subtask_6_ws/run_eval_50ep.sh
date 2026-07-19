#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="oft_subtask_subtask_6_ws"

GPU_COUNT=${GPU_COUNT:-4}
CANDIDATE_GPUS=${CANDIDATE_GPUS:-"0 1 2 3 4 5"}
GPU_CHECK_INTERVAL=${GPU_CHECK_INTERVAL:-60}
GPU_STABLE_CHECKS=${GPU_STABLE_CHECKS:-2}
QWEN_HEALTH_URL=${QWEN_HEALTH_URL:-"http://127.0.0.1:8000/v1/models"}

if [[ ! "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_COUNT must be a positive integer, got: ${GPU_COUNT}" >&2
    exit 2
fi
if [[ ! "${GPU_CHECK_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_CHECK_INTERVAL must be a positive integer, got: ${GPU_CHECK_INTERVAL}" >&2
    exit 2
fi
if [[ ! "${GPU_STABLE_CHECKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_STABLE_CHECKS must be a positive integer, got: ${GPU_STABLE_CHECKS}" >&2
    exit 2
fi

read -r -a candidates <<< "${CANDIDATE_GPUS//,/ }"
if (( ${#candidates[@]} < GPU_COUNT )); then
    echo "CANDIDATE_GPUS has fewer than GPU_COUNT entries." >&2
    exit 2
fi

available_gpus() {
    local index uuid
    declare -A occupied=()
    while IFS=',' read -r uuid; do
        uuid="${uuid//[[:space:]]/}"
        if [[ "${uuid}" == GPU-* ]]; then
            occupied["${uuid}"]=1
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)

    declare -A candidate_set=()
    for index in "${candidates[@]}"; do
        candidate_set["${index}"]=1
    done
    while IFS=',' read -r index uuid; do
        index="${index//[[:space:]]/}"
        uuid="${uuid//[[:space:]]/}"
        if [[ -n "${candidate_set[${index}]:-}" && -z "${occupied[${uuid}]:-}" ]]; then
            printf '%s\n' "${index}"
        fi
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
}

if [[ -n "${GPU_LIST:-}" ]]; then
    selected="${GPU_LIST//,/ }"
else
    stable_count=0
    previous_selection=""
    while true; do
        mapfile -t free_gpus < <(available_gpus)
        selected="${free_gpus[*]:0:GPU_COUNT}"
        if (( ${#free_gpus[@]} >= GPU_COUNT )); then
            if [[ "${selected}" == "${previous_selection}" ]]; then
                stable_count=$((stable_count + 1))
            else
                previous_selection="${selected}"
                stable_count=1
            fi
            echo "[eval-waiter] free GPUs: ${free_gpus[*]}; selected: ${selected}; stable ${stable_count}/${GPU_STABLE_CHECKS}"
            if (( stable_count >= GPU_STABLE_CHECKS )); then
                if curl -fsS --max-time 5 "${QWEN_HEALTH_URL}" >/dev/null; then
                    break
                fi
                echo "[eval-waiter] Qwen endpoint is unavailable: ${QWEN_HEALTH_URL}"
                stable_count=0
            fi
        else
            stable_count=0
            previous_selection=""
            echo "[eval-waiter] waiting for ${GPU_COUNT} free GPUs; currently free: ${free_gpus[*]:-(none)}"
        fi
        sleep "${GPU_CHECK_INTERVAL}"
    done
fi

echo "[eval-waiter] launching ${MODEL_NAME} on GPUs: ${selected}"
cd "${ROBOTWIN_ROOT}"
exec env \
    GPU_LIST="${selected}" \
    EVAL_TEST_NUM=50 \
    EARLY_STOP_ZERO_EPISODES=0 \
    CONTINUE_ON_ERROR=1 \
    TMUX_MONITOR=1 \
    bash eval_seed_whitelist_randomized.sh "${MODEL_NAME}"
