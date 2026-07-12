#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash eval_seed_whitelist_randomized.sh MODEL_NAME

Environment variables:
  GPU_LIST                  GPU ids, comma- or space-separated (default: all GPUs)
  SERVER_READY_TIMEOUT      Seconds to wait for each server (default: 900)
  TASK_TIMEOUT_SECONDS      Per-task timeout; 0 disables it (default: 0)
  CONTINUE_ON_ERROR         Continue after an eval failure (default: 1)
  SERVER_START_STAGGER_SEC  Delay between worker launches (default: 1)
  ROBOTWIN_PYTHON           RoboTwin environment Python (default: conda env RoboTwin)
  TMUX_MONITOR              Open one tmux window per GPU (default: 1)
  DRY_RUN                   Validate inputs without starting servers (default: 0)
EOF
}

MODEL_NAME=${1:-}
if [[ -z "${MODEL_NAME}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid model name: ${MODEL_NAME}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${SCRIPT_DIR}"
STARVLA_ROOT="/private/zjb/workspace/starVLA-A"
RESULT_ROOT="${STARVLA_ROOT}/results/${MODEL_NAME}"
CKPT_MAPPING="${STARVLA_ROOT}/ckpt_mapping.yaml"
MANAGED_RUN_ROOT="${STARVLA_ROOT}/examples/RoboTwin_Astribot/train_files/managed_runs/${MODEL_NAME}"
SERVER_SCRIPT="${MANAGED_RUN_ROOT}/run_policy_server.sh"
POLICY_ROOT="${ROBOTWIN_ROOT}/policy/${MODEL_NAME}"
EVAL_SCRIPT="${POLICY_ROOT}/eval.sh"
DEPLOY_CONFIG="${POLICY_ROOT}/deploy_policy.yml"
WHITELIST="${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml"
TASK_CONFIG="info_gathering_randomized"
EVAL_SEED_LIST_ROOT="${ROBOTWIN_ROOT}/eval_seed_lists"

SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SERVER_START_STAGGER_SEC=${SERVER_START_STAGGER_SEC:-1}
TMUX_MONITOR=${TMUX_MONITOR:-1}
DRY_RUN=${DRY_RUN:-0}

required_commands=(awk find flock jq nvidia-smi sed setsid sort ss tail tee timeout)
for command_name in "${required_commands[@]}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command was not found: ${command_name}" >&2
        exit 1
    fi
done
if [[ "${TMUX_MONITOR}" == "1" ]] && ! command -v tmux >/dev/null 2>&1; then
    echo "tmux was not found; set TMUX_MONITOR=0 to run without live windows." >&2
    exit 1
fi

for path in "${RESULT_ROOT}" "${MANAGED_RUN_ROOT}" "${POLICY_ROOT}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Required directory does not exist: ${path}" >&2
        exit 1
    fi
done
for path in "${CKPT_MAPPING}" "${SERVER_SCRIPT}" "${EVAL_SCRIPT}" "${DEPLOY_CONFIG}" "${WHITELIST}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file does not exist: ${path}" >&2
        exit 1
    fi
done

if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found; set ROBOTWIN_PYTHON explicitly." >&2
        exit 1
    fi
    ROBOTWIN_ENV_ROOT="$(conda env list 2>/dev/null | awk '$1 == "RoboTwin" {print $NF; exit}')"
    if [[ -n "${ROBOTWIN_ENV_ROOT}" ]]; then
        ROBOTWIN_PYTHON="${ROBOTWIN_ENV_ROOT}/bin/python"
    fi
fi
if [[ -z "${ROBOTWIN_PYTHON:-}" || ! -x "${ROBOTWIN_PYTHON}" ]]; then
    echo "Could not resolve the RoboTwin conda Python. Set ROBOTWIN_PYTHON explicitly." >&2
    exit 1
fi
if ! "${ROBOTWIN_PYTHON}" -c 'import numpy, sapien, yaml' >/dev/null 2>&1; then
    echo "RoboTwin Python failed to import numpy, sapien, or yaml: ${ROBOTWIN_PYTHON}" >&2
    exit 1
fi

if ! CHECKPOINT_STEP="$("${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${MODEL_NAME}" <<'PY'
import sys
from pathlib import Path

import yaml

mapping_path = Path(sys.argv[1])
model_name = sys.argv[2]
payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
checkpoints = payload.get("checkpoints") or {}
unavailable = payload.get("unavailable") or {}

if model_name in unavailable:
    print(
        f"No checkpoint is available for {model_name!r} according to {mapping_path}.",
        file=sys.stderr,
    )
    raise SystemExit(1)
if model_name not in checkpoints:
    print(
        f"No checkpoint mapping exists for {model_name!r} in {mapping_path}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

step = checkpoints[model_name]
if isinstance(step, bool) or not str(step).isdigit() or int(step) <= 0:
    print(
        f"Invalid checkpoint step for {model_name!r} in {mapping_path}: {step!r}.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(int(step))
PY
)"; then
    exit 1
fi
CHECKPOINT_PATH="${RESULT_ROOT}/checkpoints/steps_${CHECKPOINT_STEP}_pytorch_model.pt"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Mapped checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ -z "${GPU_LIST:-}" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi was not found; set GPU_LIST explicitly." >&2
        exit 1
    fi
    GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
    echo "GPU_LIST is empty." >&2
    exit 1
fi

declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "GPU ids must be non-negative integers, got: ${gpu}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "Duplicate GPU id: ${gpu}" >&2
        exit 2
    fi
    SEEN_GPUS[${gpu}]=1
    port=$((19000 + gpu))
    if (( port > 65535 )); then
        echo "GPU ${gpu} produces invalid port ${port}." >&2
        exit 2
    fi
done

mapfile -t TASKS < <(sed -n 's/^[[:space:]]*-[[:space:]]*//p' "${WHITELIST}")
if (( ${#TASKS[@]} == 0 )); then
    echo "No tasks found in ${WHITELIST}." >&2
    exit 1
fi
for task in "${TASKS[@]}"; do
    seed_list_path="${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}/${task}.json"
    if [[ ! -f "${seed_list_path}" ]]; then
        echo "Randomized eval seed list does not exist: ${seed_list_path}" >&2
        exit 1
    fi
    seed_count="$(jq -r 'if (.entries | type) == "array" then (.entries | length) else (.seeds | length) end' "${seed_list_path}")"
    if [[ ! "${seed_count}" =~ ^[0-9]+$ ]] || (( seed_count < 100 )); then
        echo "Randomized eval seed list needs at least 100 seeds: ${seed_list_path} (found ${seed_count})" >&2
        exit 1
    fi
done

write_markdown_report() {
    local tmp_path="${REPORT_FILE}.tmp"
    local completed_count failed_count status_text success_text result_link
    completed_count="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    failed_count="$(awk -F '\t' 'NR > 1 && $5 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"

    {
        printf '# RoboTwin Eval Report: %s\n\n' "${MODEL_NAME}"
        printf -- '- Checkpoint: `%s`\n' "${CHECKPOINT_PATH}"
        printf -- '- Checkpoint mapping: `%s` (step `%s`)\n' "${CKPT_MAPPING}" "${CHECKPOINT_STEP}"
        printf -- '- Task whitelist: `%s`\n' "${WHITELIST}"
        printf -- '- Task config: `%s`\n' "${TASK_CONFIG}"
        printf -- '- Eval seed lists: `%s/%s`\n' "${EVAL_SEED_LIST_ROOT}" "${TASK_CONFIG}"
        printf -- '- RoboTwin Python: `%s`\n' "${ROBOTWIN_PYTHON}"
        printf -- '- Eval result root: `%s`\n' "${EVAL_RESULT_ROOT}"
        printf -- '- GPUs: `%s`\n' "${GPUS[*]}"
        if [[ "${TMUX_MONITOR}" == "1" ]]; then
            printf -- '- Tmux session: `%s`\n' "${TMUX_SESSION}"
        fi
        printf -- '- Progress: `%s/%s`\n' "${completed_count}" "${#TASKS[@]}"
        printf -- '- Failed: `%s`\n\n' "${failed_count}"
        printf '| # | Task | GPU | Port | Status | Success Rate | Seconds | Log | Result |\n'
        printf '|---:|---|---:|---:|---|---:|---:|---|---|\n'

        while IFS=$'\t' read -r idx task gpu port status start_time end_time seconds log_path success_rate result_path; do
            if [[ "${status}" == "0" ]]; then
                status_text="success"
            else
                status_text="failed (${status})"
            fi
            if [[ "${success_rate}" == "-" ]]; then
                success_text="-"
            else
                success_text="$(awk -v rate="${success_rate}" 'BEGIN {printf "%.1f%%", rate * 100}')"
            fi
            if [[ "${result_path}" == "-" ]]; then
                result_link="-"
            else
                result_link="[result](<${result_path}>)"
            fi
            printf '| %s | %s | %s | %s | %s | %s | %s | [log](<%s>) | %s |\n' \
                "$((idx + 1))" "${task}" "${gpu}" "${port}" "${status_text}" \
                "${success_text}" "${seconds}" "${log_path}" "${result_link}"
        done < <(tail -n +2 "${SUMMARY_FILE}" | sort -t $'\t' -k1,1n)
    } > "${tmp_path}"
    mv "${tmp_path}" "${REPORT_FILE}"
}

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${ROBOTWIN_ROOT}/logs/eval_seed_whitelist_randomized/${MODEL_NAME}/${RUN_ID}"
TASK_LOG_ROOT="${RUN_ROOT}/tasks"
SERVER_LOG_ROOT="${RUN_ROOT}/servers"
GPU_CONSOLE_ROOT="${RUN_ROOT}/gpu_consoles"
EVAL_RESULT_ROOT="${RUN_ROOT}/eval_result"
TMUX_SESSION="robotwin_randomized_${MODEL_NAME}_${RUN_ID}"
mkdir -p "${TASK_LOG_ROOT}" "${SERVER_LOG_ROOT}" "${GPU_CONSOLE_ROOT}" "${EVAL_RESULT_ROOT}"
WRITE_TEST_PATH="${EVAL_RESULT_ROOT}/.write_test"
if ! printf 'ok\n' > "${WRITE_TEST_PATH}"; then
    echo "Eval result directory is not writable: ${EVAL_RESULT_ROOT}" >&2
    exit 1
fi
rm -f "${WRITE_TEST_PATH}"

NEXT_INDEX_FILE="${RUN_ROOT}/next_index.txt"
QUEUE_LOCK="${RUN_ROOT}/queue.lock"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
SUMMARY_LOCK="${RUN_ROOT}/summary.lock"
SERVER_SUMMARY_FILE="${RUN_ROOT}/servers.tsv"
SERVER_SUMMARY_LOCK="${RUN_ROOT}/servers.lock"
REPORT_FILE="${RUN_ROOT}/report.md"
STOP_FILE="${RUN_ROOT}/stop"

printf '0\n' > "${NEXT_INDEX_FILE}"
printf 'idx\ttask\tgpu\tport\tstatus\tstart_time\tend_time\tseconds\tlog_path\tsuccess_rate\tresult_path\n' > "${SUMMARY_FILE}"
printf 'gpu\tport\tstatus\tpid\tlog_path\n' > "${SERVER_SUMMARY_FILE}"
write_markdown_report

echo "[eval-launcher] model=${MODEL_NAME}"
echo "[eval-launcher] checkpoint=${CHECKPOINT_PATH}"
echo "[eval-launcher] checkpoint_mapping=${CKPT_MAPPING} step=${CHECKPOINT_STEP}"
echo "[eval-launcher] robotwin_python=${ROBOTWIN_PYTHON}"
echo "[eval-launcher] eval_result_root=${EVAL_RESULT_ROOT}"
echo "[eval-launcher] task_config=${TASK_CONFIG}"
echo "[eval-launcher] eval_seed_lists=${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}"
echo "[eval-launcher] tasks=${#TASKS[@]} whitelist=${WHITELIST}"
echo "[eval-launcher] gpus=${GPUS[*]} ports=19000+gpu_id"
echo "[eval-launcher] run_root=${RUN_ROOT}"

port_is_listening() {
    local port="$1"
    [[ -n "$(ss -H -ltn "sport = :${port}" 2>/dev/null)" ]]
}

for gpu in "${GPUS[@]}"; do
    port=$((19000 + gpu))
    if port_is_listening "${port}"; then
        echo "Port ${port} for GPU ${gpu} is already in use." >&2
        exit 1
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    for gpu in "${GPUS[@]}"; do
        echo "[dry-run] gpu=${gpu} server_port=$((19000 + gpu))"
    done
    echo "[dry-run] validation passed; no server or eval process was started"
    exit 0
fi

setup_tmux_monitor() {
    local gpu console_log monitor_command
    local first_window=1

    for gpu in "${GPUS[@]}"; do
        console_log="${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
        : > "${console_log}"
        printf '[gpu %s] waiting for policy server and eval task...\n' "${gpu}" >> "${console_log}"
        monitor_command="tail -n +1 -F '${console_log}'"
        if (( first_window == 1 )); then
            tmux new-session -d -s "${TMUX_SESSION}" -n "gpu_${gpu}" "${monitor_command}"
            first_window=0
        else
            tmux new-window -d -t "${TMUX_SESSION}" -n "gpu_${gpu}" "${monitor_command}"
        fi
    done
    tmux set-option -t "${TMUX_SESSION}" remain-on-exit on >/dev/null
    tmux select-window -t "${TMUX_SESSION}:gpu_${GPUS[0]}"
    echo "[eval-launcher] tmux_session=${TMUX_SESSION}"
    echo "[eval-launcher] attach: tmux attach -t ${TMUX_SESSION}"
    echo "[eval-launcher] inside tmux: tmux switch-client -t ${TMUX_SESSION}"
}

if [[ "${TMUX_MONITOR}" == "1" ]]; then
    setup_tmux_monitor
else
    for gpu in "${GPUS[@]}"; do
        : > "${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
    done
fi

wait_for_server() {
    local pid="$1"
    local port="$2"
    local timeout_seconds="$3"
    local deadline=$((SECONDS + timeout_seconds))

    while (( SECONDS < deadline )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 1
        fi
        if port_is_listening "${port}"; then
            sleep 1
            kill -0 "${pid}" 2>/dev/null && return 0
            return 1
        fi
        sleep 2
    done
    return 1
}

get_next_task() {
    local idx
    exec 200>"${QUEUE_LOCK}"
    flock -x 200
    idx="$(<"${NEXT_INDEX_FILE}")"
    if (( idx >= ${#TASKS[@]} )); then
        return 1
    fi
    printf '%s\n' "$((idx + 1))" > "${NEXT_INDEX_FILE}"
    printf '%s\t%s\n' "${idx}" "${TASKS[${idx}]}"
}

append_summary() {
    local idx="$1" task="$2" gpu="$3" port="$4" status="$5"
    local start_time="$6" end_time="$7" seconds="$8" log_path="$9"
    local success_rate="${10}" result_path="${11}"
    {
        flock -x 201
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${idx}" "${task}" "${gpu}" "${port}" "${status}" \
            "${start_time}" "${end_time}" "${seconds}" "${log_path}" \
            "${success_rate}" "${result_path}" >> "${SUMMARY_FILE}"
        write_markdown_report
    } 201>"${SUMMARY_LOCK}"
}

append_server_summary() {
    local gpu="$1" port="$2" status="$3" pid="$4" log_path="$5"
    {
        flock -x 202
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${gpu}" "${port}" "${status}" "${pid}" "${log_path}" >> "${SERVER_SUMMARY_FILE}"
    } 202>"${SERVER_SUMMARY_LOCK}"
}

worker() {
    local gpu="$1"
    local port=$((19000 + gpu))
    local server_log="${SERVER_LOG_ROOT}/gpu_${gpu}_port_${port}.log"
    local gpu_console_log="${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
    local server_pid=""
    local eval_pid=""

    cleanup_worker() {
        if [[ -n "${eval_pid}" ]] && kill -0 "${eval_pid}" 2>/dev/null; then
            kill -TERM -- "-${eval_pid}" 2>/dev/null || true
            wait "${eval_pid}" 2>/dev/null || true
        fi
        if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
            kill -TERM -- "-${server_pid}" 2>/dev/null || true
            wait "${server_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_worker EXIT
    trap 'exit 130' INT TERM

    if port_is_listening "${port}"; then
        echo "[gpu ${gpu}] port ${port} is already in use; refusing to reuse an unknown server" >&2
        append_server_summary "${gpu}" "${port}" "port_in_use" "-" "${server_log}"
        return 1
    fi

    echo "[gpu ${gpu}] starting server on port ${port}"
    printf '\n===== server start: gpu=%s port=%s =====\n' "${gpu}" "${port}" >> "${gpu_console_log}"
    setsid env \
        STARGVLA_REPO_ROOT="${STARVLA_ROOT}" \
        RUN_OUTPUT="${RESULT_ROOT}" \
        POLICY_CKPT_PATH="${CHECKPOINT_PATH}" \
        POLICY_PORT="${port}" \
        POLICY_GPU_ID="${gpu}" \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        IDLE_TIMEOUT=-1 \
        bash "${SERVER_SCRIPT}" > "${server_log}" 2>&1 &
    server_pid=$!

    if ! wait_for_server "${server_pid}" "${port}" "${SERVER_READY_TIMEOUT}"; then
        echo "[gpu ${gpu}] server failed to become ready; log=${server_log}" >&2
        append_server_summary "${gpu}" "${port}" "not_ready" "${server_pid}" "${server_log}"
        return 1
    fi
    append_server_summary "${gpu}" "${port}" "ready" "${server_pid}" "${server_log}"
    echo "[gpu ${gpu}] server ready pid=${server_pid} port=${port}"
    printf '===== server ready: pid=%s =====\n' "${server_pid}" >> "${gpu_console_log}"

    local item idx task task_dir stdout_log start_time end_time
    local start_seconds end_seconds elapsed_seconds status result_path success_rate
    while [[ ! -f "${STOP_FILE}" ]]; do
        if ! item="$(get_next_task)"; then
            break
        fi
        idx="${item%%$'\t'*}"
        task="${item#*$'\t'}"
        task_dir="${TASK_LOG_ROOT}/$(printf '%02d' "$((idx + 1))")_${task}"
        stdout_log="${task_dir}/stdout.log"
        mkdir -p "${task_dir}"

        start_time="$(date '+%Y-%m-%d %H:%M:%S')"
        start_seconds="$(date +%s)"
        echo "[gpu ${gpu}] start $((idx + 1))/${#TASKS[@]} ${task}"
        printf '\n===== task %s/%s start: %s =====\n' \
            "$((idx + 1))" "${#TASKS[@]}" "${task}" >> "${gpu_console_log}"

        local eval_cmd=(bash "${EVAL_SCRIPT}" "${task}" "${gpu}" "${port}" "${TASK_CONFIG}")
        if [[ "${TASK_TIMEOUT_SECONDS}" != "0" ]]; then
            eval_cmd=(timeout --signal=TERM --kill-after=30 "${TASK_TIMEOUT_SECONDS}" "${eval_cmd[@]}")
        fi

        : > "${stdout_log}"
        setsid bash -o pipefail -c '
            task_log=$1
            gpu_log=$2
            shift 2
            "$@" 2>&1 | tee -a "$task_log" "$gpu_log" >/dev/null
        ' _ "${stdout_log}" "${gpu_console_log}" \
            env \
                STARGVLA_REPO_ROOT="${STARVLA_ROOT}" \
                STARGVLA_HOST=127.0.0.1 \
                STARGVLA_PORT="${port}" \
                ROBOTWIN_USE_EVAL_SEED_LIST=1 \
                ROBOTWIN_EVAL_SEED_LIST_PATH="${EVAL_SEED_LIST_ROOT}" \
                ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
                ROBOTWIN_PY="${ROBOTWIN_PYTHON}" \
                ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}" \
                ROBOTWIN_REQUEST_LOG_PATH="${task_dir}/requests.jsonl" \
                ROBOTWIN_REQUEST_IMAGE_DIR= \
                PYTHONUNBUFFERED=1 \
                "${eval_cmd[@]}" &
        eval_pid=$!
        if wait "${eval_pid}"; then
            status=0
        else
            status=$?
        fi
        eval_pid=""

        end_time="$(date '+%Y-%m-%d %H:%M:%S')"
        end_seconds="$(date +%s)"
        elapsed_seconds=$((end_seconds - start_seconds))
        result_path="$(sed -n 's/^Data has been saved to //p' "${stdout_log}" | tail -n 1)"
        success_rate="-"
        if [[ -n "${result_path}" ]]; then
            if [[ "${result_path}" != /* ]]; then
                result_path="${ROBOTWIN_ROOT}/${result_path}"
            fi
            if [[ -f "${result_path}" ]]; then
                success_rate="$(awk '/^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {value=$1} END {if (value != "") print value}' "${result_path}")"
                success_rate=${success_rate:--}
            else
                result_path="-"
            fi
        else
            result_path="-"
        fi
        append_summary "${idx}" "${task}" "${gpu}" "${port}" "${status}" \
            "${start_time}" "${end_time}" "${elapsed_seconds}" "${stdout_log}" \
            "${success_rate}" "${result_path}"
        printf '===== task end: %s exit=%s seconds=%s =====\n' \
            "${task}" "${status}" "${elapsed_seconds}" >> "${gpu_console_log}"

        if (( status == 0 )); then
            echo "[gpu ${gpu}] ok ${task} (${elapsed_seconds}s)"
        else
            echo "[gpu ${gpu}] failed ${task} exit=${status} (${elapsed_seconds}s); log=${stdout_log}" >&2
            if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
                touch "${STOP_FILE}"
            fi
        fi

        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "[gpu ${gpu}] server exited unexpectedly; log=${server_log}" >&2
            return 1
        fi
    done
    echo "[gpu ${gpu}] worker finished"
    printf '\n===== gpu %s queue finished =====\n' "${gpu}" >> "${gpu_console_log}"
}

worker_pids=()
cleanup_main() {
    touch "${STOP_FILE}"
    if (( ${#worker_pids[@]} > 0 )); then
        kill -TERM "${worker_pids[@]}" 2>/dev/null || true
        wait "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap 'cleanup_main; exit 130' INT TERM

for gpu in "${GPUS[@]}"; do
    worker "${gpu}" &
    worker_pids+=("$!")
    if [[ "${SERVER_START_STAGGER_SEC}" != "0" ]]; then
        sleep "${SERVER_START_STAGGER_SEC}"
    fi
done

worker_failure=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        worker_failure=1
    fi
done

done_count="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
failed_count="$(awk -F '\t' 'NR > 1 && $5 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"

echo "[eval-launcher] finished=${done_count}/${#TASKS[@]} failed=${failed_count}"
echo "[eval-launcher] summary=${SUMMARY_FILE}"
echo "[eval-launcher] servers=${SERVER_SUMMARY_FILE}"
echo "[eval-launcher] report=${REPORT_FILE}"
if [[ "${TMUX_MONITOR}" == "1" ]]; then
    echo "[eval-launcher] tmux_session=${TMUX_SESSION}"
    echo "[eval-launcher] close tmux: tmux kill-session -t ${TMUX_SESSION}"
fi

if (( done_count != ${#TASKS[@]} || failed_count != 0 || worker_failure != 0 )); then
    exit 1
fi
