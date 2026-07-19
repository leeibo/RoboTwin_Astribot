#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash eval_seed_whitelist_randomized.sh MODEL_NAME

Environment variables:
  GPU_LIST                  GPU ids, comma- or space-separated (default: all GPUs)
  TASK_LIST                 Optional comma- or space-separated whitelist subset
  SERVER_READY_TIMEOUT      Seconds to wait for each server (default: 900)
  TASK_TIMEOUT_SECONDS      Per-task timeout; 0 disables it (default: 0)
  TASK_LIMIT                Only run the first N whitelisted tasks; 0 means all (default: 0)
  EVAL_TEST_NUM             Episodes per task (default: 100)
  EARLY_STOP_ZERO_EPISODES  Stop after this many episodes if successes stay at 0
                            and assume the final rate is 0 (default: 40; 0 disables)
  CONTINUE_ON_ERROR         Continue after an eval failure (default: 1)
  SERVER_START_STAGGER_SEC  Delay between worker launches (default: 1)
  SERVER_POST_READY_SETTLE_SEC
                             Delay after all servers are ready (default: 10)
  EVAL_WORKER_START_STAGGER_SEC
                             Delay between first eval starts (default: 0)
  RESUME_FROM_RUN           Prior run root; successful tasks are imported and skipped
  PLANNER_MODEL_NAME        Paired planner checkpoint name (default: planner_oft_planner)
  STARVLA_ROOT              starVLA-A repository (default: sibling of RoboTwin_Astribot)
  STARVLA_PYTHON            starVLA environment Python (default: conda env starVLA)
  STARGVLA_BASE_VLM         Local action-token base VLM (default: under starVLA-A/playground)
  STARGVLA_FAST_TOKENIZER   Local physical-intelligence/fast tokenizer directory
  ROBOTWIN_PYTHON           RoboTwin environment Python (default: conda env RoboTwin)
  SKIP_PYTHON_IMPORT_CHECK  Skip the startup-only numpy/sapien/yaml probe (default: 0)
  TMUX_MONITOR              Open one tmux window per GPU (default: 1)
  DRY_RUN                   Validate inputs without starting servers (default: 0)
EOF
}

REQUESTED_MODEL_NAME=${1:-}
if [[ -z "${REQUESTED_MODEL_NAME}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${REQUESTED_MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid model name: ${REQUESTED_MODEL_NAME}" >&2
    exit 2
fi

MODEL_NAME="${REQUESTED_MODEL_NAME}"
JOINT_PLANNER=0
if [[ "${MODEL_NAME}" == "planner_oft_planner" ]]; then
    MODEL_NAME=planner_oft
fi
if [[ "${MODEL_NAME}" == "planner_oft" ]]; then
    JOINT_PLANNER=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${SCRIPT_DIR}"
STARVLA_ROOT="${STARVLA_ROOT:-${SCRIPT_DIR}/../starVLA-A}"
if [[ -d "${STARVLA_ROOT}" ]]; then
    STARVLA_ROOT="$(cd "${STARVLA_ROOT}" && pwd)"
fi
STARGVLA_FAST_TOKENIZER="${STARGVLA_FAST_TOKENIZER:-${STARVLA_ROOT}/playground/Pretrained_models/fast}"
RESULT_ROOT="${STARVLA_ROOT}/results/${MODEL_NAME}"
RESULT_CONFIG="${RESULT_ROOT}/config.yaml"
CKPT_MAPPING="${STARVLA_ROOT}/ckpt_mapping.yaml"
MANAGED_RUN_ROOT="${STARVLA_ROOT}/examples/RoboTwin_Astribot/train_files/managed_runs/${MODEL_NAME}"
SERVER_SCRIPT="${MANAGED_RUN_ROOT}/run_policy_server.sh"
POLICY_ROOT="${ROBOTWIN_ROOT}/policy/${MODEL_NAME}"
EVAL_SCRIPT="${POLICY_ROOT}/eval.sh"
DEPLOY_CONFIG="${POLICY_ROOT}/deploy_policy.yml"
WHITELIST="${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml"
STEP_LIMITS="${ROBOTWIN_ROOT}/task_config/_eval_step_limit.yml"
TASK_CONFIG="info_gathering_randomized"
EVAL_SEED_LIST_ROOT="${ROBOTWIN_ROOT}/eval_seed_lists"
PLANNER_MODEL_NAME=${PLANNER_MODEL_NAME:-planner_oft_planner}
PLANNER_RESULT_ROOT="${STARVLA_ROOT}/results/${PLANNER_MODEL_NAME}"
PLANNER_SERVER_SCRIPT="${POLICY_ROOT}/run_planner_server.sh"

SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
TASK_LIMIT=${TASK_LIMIT:-0}
EVAL_TEST_NUM=${EVAL_TEST_NUM:-100}
EARLY_STOP_ZERO_EPISODES=${EARLY_STOP_ZERO_EPISODES:-40}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SERVER_START_STAGGER_SEC=${SERVER_START_STAGGER_SEC:-1}
SERVER_POST_READY_SETTLE_SEC=${SERVER_POST_READY_SETTLE_SEC:-10}
EVAL_WORKER_START_STAGGER_SEC=${EVAL_WORKER_START_STAGGER_SEC:-0}
TMUX_MONITOR=${TMUX_MONITOR:-1}
DRY_RUN=${DRY_RUN:-0}
RESUME_FROM_RUN=${RESUME_FROM_RUN:-}
SKIP_PYTHON_IMPORT_CHECK=${SKIP_PYTHON_IMPORT_CHECK:-0}

if [[ ! "${EARLY_STOP_ZERO_EPISODES}" =~ ^[0-9]+$ ]]; then
    echo "EARLY_STOP_ZERO_EPISODES must be a non-negative integer, got: ${EARLY_STOP_ZERO_EPISODES}" >&2
    exit 2
fi
if [[ ! "${EVAL_TEST_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_TEST_NUM must be a positive integer, got: ${EVAL_TEST_NUM}" >&2
    exit 2
fi
if [[ ! "${SERVER_POST_READY_SETTLE_SEC}" =~ ^[0-9]+$ ]]; then
    echo "SERVER_POST_READY_SETTLE_SEC must be a non-negative integer, got: ${SERVER_POST_READY_SETTLE_SEC}" >&2
    exit 2
fi
if [[ ! "${EVAL_WORKER_START_STAGGER_SEC}" =~ ^[0-9]+$ ]]; then
    echo "EVAL_WORKER_START_STAGGER_SEC must be a non-negative integer, got: ${EVAL_WORKER_START_STAGGER_SEC}" >&2
    exit 2
fi

required_commands=(awk find flock nvidia-smi sed setsid sort ss tail tee timeout)
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
for path in "${RESULT_CONFIG}" "${CKPT_MAPPING}" "${SERVER_SCRIPT}" "${EVAL_SCRIPT}" \
    "${DEPLOY_CONFIG}" "${WHITELIST}" "${STEP_LIMITS}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file does not exist: ${path}" >&2
        exit 1
    fi
done
if [[ "${JOINT_PLANNER}" == "1" ]]; then
    for path in "${PLANNER_RESULT_ROOT}" "${PLANNER_SERVER_SCRIPT}"; do
        if [[ ! -e "${path}" ]]; then
            echo "Required planner path does not exist: ${path}" >&2
            exit 1
        fi
    done
fi

if [[ -z "${STARGVLA_BASE_VLM:-}" ]]; then
    CONFIGURED_BASE_VLM="$(
        awk '
            $1 == "base_vlm:" {
                value = $0
                sub(/^[[:space:]]*base_vlm:[[:space:]]*/, "", value)
                gsub(/^["'\'' ]+|["'\'' ]+$/, "", value)
                print value
                exit
            }
        ' "${RESULT_CONFIG}"
    )"
    if [[ -z "${CONFIGURED_BASE_VLM}" ]]; then
        echo "Could not resolve framework.qwenvl.base_vlm from ${RESULT_CONFIG}." >&2
        echo "Set STARGVLA_BASE_VLM explicitly." >&2
        exit 1
    fi
    if [[ "${CONFIGURED_BASE_VLM}" == /* ]]; then
        STARGVLA_BASE_VLM="${CONFIGURED_BASE_VLM}"
    elif [[ "${CONFIGURED_BASE_VLM}" == ./* ]]; then
        STARGVLA_BASE_VLM="${STARVLA_ROOT}/${CONFIGURED_BASE_VLM#./}"
    else
        echo "Configured base VLM is not a local path: ${CONFIGURED_BASE_VLM}" >&2
        echo "Set STARGVLA_BASE_VLM to its local directory." >&2
        exit 1
    fi
fi
if [[ ! -d "${STARGVLA_BASE_VLM}" ]]; then
    echo "Required base VLM does not exist: ${STARGVLA_BASE_VLM}" >&2
    echo "Configured by ${RESULT_CONFIG}; set STARGVLA_BASE_VLM to override it." >&2
    exit 1
fi
for path in \
    "${STARGVLA_FAST_TOKENIZER}/processing_action_tokenizer.py" \
    "${STARGVLA_FAST_TOKENIZER}/tokenizer.json" \
    "${STARGVLA_FAST_TOKENIZER}/processor_config.json"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required FAST tokenizer file does not exist: ${path}" >&2
        echo "Set STARGVLA_FAST_TOKENIZER to a complete physical-intelligence/fast directory." >&2
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
ROBOTWIN_ENV_ROOT="$(cd "$(dirname "${ROBOTWIN_PYTHON}")/.." && pwd)"
ROBOTWIN_CUROBO_SRC="${ROBOTWIN_CUROBO_SRC:-${ROBOTWIN_ROOT}/envs/curobo/src}"
ROBOTWIN_CUDA_HOME="${ROBOTWIN_CUDA_HOME:-${ROBOTWIN_ENV_ROOT}}"
if [[ ! -f "${ROBOTWIN_CUROBO_SRC}/curobo/types/math.py" ]]; then
    echo "NVIDIA cuRobo source was not found: ${ROBOTWIN_CUROBO_SRC}" >&2
    exit 1
fi
if [[ ! -x "${ROBOTWIN_CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA compiler was not found: ${ROBOTWIN_CUDA_HOME}/bin/nvcc" >&2
    exit 1
fi
if [[ "${SKIP_PYTHON_IMPORT_CHECK}" != "1" ]] \
    && ! "${ROBOTWIN_PYTHON}" -c 'import numpy, sapien, yaml' >/dev/null 2>&1; then
    echo "RoboTwin Python failed to import numpy, sapien, or yaml: ${ROBOTWIN_PYTHON}" >&2
    exit 1
fi

if [[ -z "${STARVLA_PYTHON:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found; set STARVLA_PYTHON explicitly." >&2
        exit 1
    fi
    STARVLA_ENV_ROOT="$(conda env list 2>/dev/null | awk '$1 == "starVLA" {print $NF; exit}')"
    if [[ -n "${STARVLA_ENV_ROOT}" ]]; then
        STARVLA_PYTHON="${STARVLA_ENV_ROOT}/bin/python"
    fi
fi
if [[ -z "${STARVLA_PYTHON:-}" || ! -x "${STARVLA_PYTHON}" ]]; then
    echo "Could not resolve the starVLA conda Python. Set STARVLA_PYTHON explicitly." >&2
    exit 1
fi
if ! "${STARVLA_PYTHON}" -c 'import torch, yaml' >/dev/null 2>&1; then
    echo "starVLA Python failed to import torch or yaml: ${STARVLA_PYTHON}" >&2
    exit 1
fi

resolve_checkpoint_step() {
    "${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "$1" <<'PY'
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
}

if ! CHECKPOINT_STEP="$(resolve_checkpoint_step "${MODEL_NAME}")"; then
    exit 1
fi
CHECKPOINT_PATH="${RESULT_ROOT}/checkpoints/steps_${CHECKPOINT_STEP}_pytorch_model.pt"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Mapped checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
PLANNER_CHECKPOINT_STEP=""
PLANNER_CHECKPOINT_PATH=""
if [[ "${JOINT_PLANNER}" == "1" ]]; then
    if ! PLANNER_CHECKPOINT_STEP="$(resolve_checkpoint_step "${PLANNER_MODEL_NAME}")"; then
        exit 1
    fi
    PLANNER_CHECKPOINT_PATH="${PLANNER_RESULT_ROOT}/checkpoints/steps_${PLANNER_CHECKPOINT_STEP}_pytorch_model.pt"
    if [[ ! -f "${PLANNER_CHECKPOINT_PATH}" ]]; then
        echo "Mapped planner checkpoint does not exist: ${PLANNER_CHECKPOINT_PATH}" >&2
        exit 1
    fi
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
    planner_port=$((20000 + gpu))
    if (( port > 65535 || planner_port > 65535 )); then
        echo "GPU ${gpu} produces an invalid server port." >&2
        exit 2
    fi
done

mapfile -t WHITELIST_TASKS < <(sed -n 's/^[[:space:]]*-[[:space:]]*//p' "${WHITELIST}")
if (( ${#WHITELIST_TASKS[@]} == 0 )); then
    echo "No tasks found in ${WHITELIST}." >&2
    exit 1
fi
if [[ -n "${TASK_LIST:-}" ]]; then
    normalized_task_list="${TASK_LIST//,/ }"
    read -r -a TASKS <<< "${normalized_task_list}"
    declare -A ALLOWED_TASKS=()
    for task in "${WHITELIST_TASKS[@]}"; do
        ALLOWED_TASKS["${task}"]=1
    done
    for task in "${TASKS[@]}"; do
        if [[ -z "${ALLOWED_TASKS[${task}]:-}" ]]; then
            echo "TASK_LIST contains a task outside ${WHITELIST}: ${task}" >&2
            exit 2
        fi
    done
else
    TASKS=("${WHITELIST_TASKS[@]}")
fi
if [[ ! "${TASK_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "TASK_LIMIT must be a non-negative integer, got: ${TASK_LIMIT}" >&2
    exit 2
fi
if (( TASK_LIMIT > 0 && TASK_LIMIT < ${#TASKS[@]} )); then
    TASKS=("${TASKS[@]:0:TASK_LIMIT}")
fi
if (( ${#TASKS[@]} == 0 )); then
    echo "TASK_LIST is empty." >&2
    exit 2
fi
if ! "${ROBOTWIN_PYTHON}" - "${STEP_LIMITS}" "${TASKS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
tasks = sys.argv[2:]
try:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"Failed to parse eval step limits {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(payload, dict):
    print(f"Eval step limits must be a mapping: {path}", file=sys.stderr)
    raise SystemExit(1)

missing = [task for task in tasks if task not in payload]
invalid = {
    task: payload.get(task)
    for task in tasks
    if task in payload
    and (
        isinstance(payload[task], bool)
        or not isinstance(payload[task], int)
        or payload[task] <= 0
    )
}
if missing:
    print(f"Tasks missing from {path}: {', '.join(missing)}", file=sys.stderr)
if invalid:
    print(f"Invalid eval step limits in {path}: {invalid}", file=sys.stderr)
if missing or invalid:
    raise SystemExit(1)

print(f"[eval-launcher] validated eval step limits for {len(tasks)} tasks: {path}")
PY
then
    exit 1
fi
for task in "${TASKS[@]}"; do
    seed_list_path="${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}/${task}.json"
    if [[ ! -f "${seed_list_path}" ]]; then
        echo "Randomized eval seed list does not exist: ${seed_list_path}" >&2
        exit 1
    fi
    seed_count="$("${ROBOTWIN_PYTHON}" - "${seed_list_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    payload = json.load(file)
entries = payload.get("entries")
print(len(entries if isinstance(entries, list) else payload.get("seeds", [])))
PY
)"
    if [[ ! "${seed_count}" =~ ^[0-9]+$ ]] || (( seed_count < EVAL_TEST_NUM )); then
        echo "Randomized eval seed list needs at least ${EVAL_TEST_NUM} seeds: ${seed_list_path} (found ${seed_count})" >&2
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
        printf -- '- Action checkpoint: `%s`\n' "${CHECKPOINT_PATH}"
        printf -- '- Action checkpoint mapping: `%s` (step `%s`)\n' "${CKPT_MAPPING}" "${CHECKPOINT_STEP}"
        if [[ "${JOINT_PLANNER}" == "1" ]]; then
            printf -- '- Planner checkpoint: `%s`\n' "${PLANNER_CHECKPOINT_PATH}"
            printf -- '- Planner checkpoint mapping: `%s` (step `%s`)\n' \
                "${CKPT_MAPPING}" "${PLANNER_CHECKPOINT_STEP}"
        fi
        printf -- '- Task whitelist: `%s`\n' "${WHITELIST}"
        printf -- '- Eval step limits: `%s`\n' "${STEP_LIMITS}"
        printf -- '- Task config: `%s`\n' "${TASK_CONFIG}"
        printf -- '- Eval seed lists: `%s/%s`\n' "${EVAL_SEED_LIST_ROOT}" "${TASK_CONFIG}"
        printf -- '- Episodes per task: `%s`\n' "${EVAL_TEST_NUM}"
        printf -- '- RoboTwin Python: `%s`\n' "${ROBOTWIN_PYTHON}"
        printf -- '- starVLA Python: `%s`\n' "${STARVLA_PYTHON}"
        printf -- '- starVLA base VLM: `%s`\n' "${STARGVLA_BASE_VLM}"
        printf -- '- FAST tokenizer: `%s`\n' "${STARGVLA_FAST_TOKENIZER}"
        printf -- '- Eval result root: `%s`\n' "${EVAL_RESULT_ROOT}"
        if [[ -n "${RESUME_FROM_RUN}" ]]; then
            printf -- '- Resumed from: `%s` (successful tasks skipped)\n' "${RESUME_FROM_RUN}"
        fi
        printf -- '- GPUs: `%s`\n' "${GPUS[*]}"
        printf -- '- Zero-success early stop: `%s` completed episodes (`0` disables)\n' \
            "${EARLY_STOP_ZERO_EPISODES}"
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
TORCH_EXTENSIONS_DIR="${ROBOTWIN_TORCH_EXTENSIONS_DIR:-${ROBOTWIN_ROOT}/logs/torch_extensions}"
TASK_LOG_ROOT="${RUN_ROOT}/tasks"
SERVER_LOG_ROOT="${RUN_ROOT}/servers"
GPU_CONSOLE_ROOT="${RUN_ROOT}/gpu_consoles"
EVAL_RESULT_ROOT="${RUN_ROOT}/eval_result"
TMUX_SESSION="robotwin_randomized_${MODEL_NAME}_${RUN_ID}"
mkdir -p "${TASK_LOG_ROOT}" "${SERVER_LOG_ROOT}" "${GPU_CONSOLE_ROOT}" "${EVAL_RESULT_ROOT}" "${TORCH_EXTENSIONS_DIR}"
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
if [[ -n "${RESUME_FROM_RUN}" ]]; then
    RESUME_FROM_RUN="$(cd "${RESUME_FROM_RUN}" && pwd)"
    RESUME_SUMMARY="${RESUME_FROM_RUN}/summary.tsv"
    if [[ ! -f "${RESUME_SUMMARY}" ]]; then
        echo "Resume summary does not exist: ${RESUME_SUMMARY}" >&2
        exit 1
    fi
    for idx in "${!TASKS[@]}"; do
        task="${TASKS[${idx}]}"
        resume_row="$(awk -F '\t' -v task="${task}" \
            'NR > 1 && $2 == task && $5 == 0 {print; exit}' "${RESUME_SUMMARY}")"
        if [[ -n "${resume_row}" ]]; then
            printf '%s\t%s\n' "${idx}" "${resume_row#*$'\t'}" >> "${SUMMARY_FILE}"
        fi
    done
fi
printf 'role\tgpu\tport\tstatus\tpid\tlog_path\n' > "${SERVER_SUMMARY_FILE}"
write_markdown_report

echo "[eval-launcher] model=${MODEL_NAME} requested_model=${REQUESTED_MODEL_NAME}"
echo "[eval-launcher] action_checkpoint=${CHECKPOINT_PATH}"
echo "[eval-launcher] action_checkpoint_mapping=${CKPT_MAPPING} step=${CHECKPOINT_STEP}"
if [[ "${JOINT_PLANNER}" == "1" ]]; then
    echo "[eval-launcher] planner_model=${PLANNER_MODEL_NAME}"
    echo "[eval-launcher] planner_checkpoint=${PLANNER_CHECKPOINT_PATH}"
    echo "[eval-launcher] planner_checkpoint_mapping=${CKPT_MAPPING} step=${PLANNER_CHECKPOINT_STEP}"
fi
echo "[eval-launcher] robotwin_python=${ROBOTWIN_PYTHON}"
echo "[eval-launcher] starvla_python=${STARVLA_PYTHON}"
echo "[eval-launcher] starvla_base_vlm=${STARGVLA_BASE_VLM}"
echo "[eval-launcher] fast_tokenizer=${STARGVLA_FAST_TOKENIZER}"
echo "[eval-launcher] eval_result_root=${EVAL_RESULT_ROOT}"
echo "[eval-launcher] test_episodes_per_new_task=${EVAL_TEST_NUM}"
if [[ -n "${RESUME_FROM_RUN}" ]]; then
    echo "[eval-launcher] resume_from_run=${RESUME_FROM_RUN} successful_tasks_skipped=1"
fi
echo "[eval-launcher] task_config=${TASK_CONFIG}"
echo "[eval-launcher] eval_seed_lists=${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}"
echo "[eval-launcher] eval_test_num=${EVAL_TEST_NUM}"
echo "[eval-launcher] tasks=${#TASKS[@]} whitelist=${WHITELIST}"
echo "[eval-launcher] eval_step_limits=${STEP_LIMITS}"
echo "[eval-launcher] gpus=${GPUS[*]} ports=19000+gpu_id"
echo "[eval-launcher] early_stop_zero_episodes=${EARLY_STOP_ZERO_EPISODES}"
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
    planner_port=$((20000 + gpu))
    if [[ "${JOINT_PLANNER}" == "1" ]] && port_is_listening "${planner_port}"; then
        echo "Planner port ${planner_port} for GPU ${gpu} is already in use." >&2
        exit 1
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    for gpu in "${GPUS[@]}"; do
        echo "[dry-run] gpu=${gpu} action_server_port=$((19000 + gpu)) planner_server_port=$((20000 + gpu))"
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
    local idx task
    exec 200>"${QUEUE_LOCK}"
    flock -x 200
    while true; do
        idx="$(<"${NEXT_INDEX_FILE}")"
        if (( idx >= ${#TASKS[@]} )); then
            return 1
        fi
        printf '%s\n' "$((idx + 1))" > "${NEXT_INDEX_FILE}"
        task="${TASKS[${idx}]}"
        if awk -F '\t' -v idx="${idx}" -v task="${task}" \
            '$1 == idx && $2 == task && $5 == 0 {found=1} END {exit !found}' \
            "${SUMMARY_FILE}"; then
            printf '[queue] skip completed %s/%s %s\n' "$((idx + 1))" "${#TASKS[@]}" "${task}" >&2
            continue
        fi
        printf '%s\t%s\n' "${idx}" "${task}"
        return 0
    done
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
    local role="$1" gpu="$2" port="$3" status="$4" pid="$5" log_path="$6"
    {
        flock -x 202
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${role}" "${gpu}" "${port}" "${status}" "${pid}" "${log_path}" >> "${SERVER_SUMMARY_FILE}"
    } 202>"${SERVER_SUMMARY_LOCK}"
}

monitor_zero_success() {
    local eval_pid="$1" stdout_log="$2" gpu_console_log="$3" marker_path="$4"
    local line successes completed

    while IFS= read -r line; do
        if [[ "${line}" =~ Success\ rate:\ ([0-9]+)/([0-9]+) ]]; then
            successes="${BASH_REMATCH[1]}"
            completed="${BASH_REMATCH[2]}"
            if (( completed >= EARLY_STOP_ZERO_EPISODES && successes == 0 )); then
                : > "${marker_path}"
                printf '[eval-launcher] early stop: 0 successes after %s episodes; assuming the remaining episodes fail and final success rate is 0\n' \
                    "${completed}" | tee -a "${stdout_log}" "${gpu_console_log}" >/dev/null
                kill -TERM -- "-${eval_pid}" 2>/dev/null || true
                return 0
            fi
        fi
    done < <(
        tail --pid="${eval_pid}" -n +1 -F "${stdout_log}" 2>/dev/null \
            | sed -u -E $'s/\033\\[[0-9;]*[[:alpha:]]//g' \
            | sed -u -n '/Success rate:/p'
    )
}

worker() {
    local gpu="$1"
    local worker_index="$2"
    local port=$((19000 + gpu))
    local planner_port=$((20000 + gpu))
    local server_log="${SERVER_LOG_ROOT}/gpu_${gpu}_port_${port}.log"
    local planner_server_log="${SERVER_LOG_ROOT}/gpu_${gpu}_planner_port_${planner_port}.log"
    local gpu_console_log="${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
    local server_pid=""
    local planner_server_pid=""
    local eval_pid=""
    local early_stop_monitor_pid=""

    cleanup_worker() {
        if [[ -n "${early_stop_monitor_pid}" ]] && kill -0 "${early_stop_monitor_pid}" 2>/dev/null; then
            kill -TERM "${early_stop_monitor_pid}" 2>/dev/null || true
            wait "${early_stop_monitor_pid}" 2>/dev/null || true
        fi
        if [[ -n "${eval_pid}" ]] && kill -0 "${eval_pid}" 2>/dev/null; then
            kill -TERM -- "-${eval_pid}" 2>/dev/null || true
            wait "${eval_pid}" 2>/dev/null || true
        fi
        if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
            kill -TERM -- "-${server_pid}" 2>/dev/null || true
            wait "${server_pid}" 2>/dev/null || true
        fi
        if [[ -n "${planner_server_pid}" ]] && kill -0 "${planner_server_pid}" 2>/dev/null; then
            kill -TERM -- "-${planner_server_pid}" 2>/dev/null || true
            wait "${planner_server_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_worker EXIT
    trap 'exit 130' INT TERM

    if port_is_listening "${port}"; then
        echo "[gpu ${gpu}] port ${port} is already in use; refusing to reuse an unknown server" >&2
        append_server_summary action "${gpu}" "${port}" "port_in_use" "-" "${server_log}"
        return 1
    fi
    if [[ "${JOINT_PLANNER}" == "1" ]] && port_is_listening "${planner_port}"; then
        echo "[gpu ${gpu}] planner port ${planner_port} is already in use; refusing to reuse an unknown server" >&2
        append_server_summary planner "${gpu}" "${planner_port}" "port_in_use" "-" "${planner_server_log}"
        return 1
    fi

    echo "[gpu ${gpu}] starting server on port ${port}"
    printf '\n===== server start: gpu=%s port=%s =====\n' "${gpu}" "${port}" >> "${gpu_console_log}"
    setsid env \
        STARGVLA_REPO_ROOT="${STARVLA_ROOT}" \
        STARGVLA_BASE_VLM="${STARGVLA_BASE_VLM}" \
        STARGVLA_FAST_TOKENIZER="${STARGVLA_FAST_TOKENIZER}" \
        STARVLA_PYTHON="${STARVLA_PYTHON}" \
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
        append_server_summary action "${gpu}" "${port}" "not_ready" "${server_pid}" "${server_log}"
        return 1
    fi
    append_server_summary action "${gpu}" "${port}" "ready" "${server_pid}" "${server_log}"
    echo "[gpu ${gpu}] server ready pid=${server_pid} port=${port}"
    printf '===== server ready: pid=%s =====\n' "${server_pid}" >> "${gpu_console_log}"

    if [[ "${JOINT_PLANNER}" == "1" ]]; then
        echo "[gpu ${gpu}] starting planner server on port ${planner_port}"
        printf '\n===== planner server start: gpu=%s port=%s =====\n' \
            "${gpu}" "${planner_port}" >> "${gpu_console_log}"
        setsid env \
            STARGVLA_REPO_ROOT="${STARVLA_ROOT}" \
            STARGVLA_BASE_VLM="${STARGVLA_BASE_VLM}" \
            STARVLA_PYTHON="${STARVLA_PYTHON}" \
            PLANNER_CKPT_PATH="${PLANNER_CHECKPOINT_PATH}" \
            PLANNER_PORT="${planner_port}" \
            PLANNER_GPU_ID="${gpu}" \
            CUDA_VISIBLE_DEVICES="${gpu}" \
            IDLE_TIMEOUT=-1 \
            bash "${PLANNER_SERVER_SCRIPT}" > "${planner_server_log}" 2>&1 &
        planner_server_pid=$!
        if ! wait_for_server "${planner_server_pid}" "${planner_port}" "${SERVER_READY_TIMEOUT}"; then
            echo "[gpu ${gpu}] planner server failed to become ready; log=${planner_server_log}" >&2
            append_server_summary planner "${gpu}" "${planner_port}" "not_ready" \
                "${planner_server_pid}" "${planner_server_log}"
            return 1
        fi
        append_server_summary planner "${gpu}" "${planner_port}" "ready" \
            "${planner_server_pid}" "${planner_server_log}"
        echo "[gpu ${gpu}] planner server ready pid=${planner_server_pid} port=${planner_port}"
        printf '===== planner server ready: pid=%s =====\n' "${planner_server_pid}" >> "${gpu_console_log}"
    fi

    local all_servers_ready peer_gpu barrier_deadline
    barrier_deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    while true; do
        all_servers_ready=1
        for peer_gpu in "${GPUS[@]}"; do
            if ! port_is_listening "$((19000 + peer_gpu))" \
                || { [[ "${JOINT_PLANNER}" == "1" ]] \
                    && ! port_is_listening "$((20000 + peer_gpu))"; }; then
                all_servers_ready=0
                break
            fi
        done
        if (( all_servers_ready )); then
            break
        fi
        if (( SECONDS >= barrier_deadline )); then
            echo "[gpu ${gpu}] timed out waiting for all policy servers" >&2
            return 1
        fi
        sleep 2
    done
    if (( SERVER_POST_READY_SETTLE_SEC > 0 )); then
        sleep "${SERVER_POST_READY_SETTLE_SEC}"
    fi
    echo "[gpu ${gpu}] all policy servers ready"
    local eval_start_delay=$((worker_index * EVAL_WORKER_START_STAGGER_SEC))
    if (( eval_start_delay > 0 )); then
        echo "[gpu ${gpu}] delaying first eval start by ${eval_start_delay}s"
        sleep "${eval_start_delay}"
    fi

    local item idx task task_dir stdout_log start_time end_time early_stop_marker
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

        local eval_cmd=(
            bash "${EVAL_SCRIPT}" "${task}" "${gpu}" "${port}" "${TASK_CONFIG}"
            "${MODEL_NAME}" 0 "${EVAL_TEST_NUM}"
        )
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
                PLANNER_OFT_HOST=127.0.0.1 \
                PLANNER_OFT_PORT="${planner_port}" \
                ROBOTWIN_USE_EVAL_SEED_LIST=1 \
                ROBOTWIN_EVAL_SEED_LIST_PATH="${EVAL_SEED_LIST_ROOT}" \
                ROBOTWIN_EVAL_TEST_NUM="${EVAL_TEST_NUM}" \
                ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
                ROBOTWIN_PY="${ROBOTWIN_PYTHON}" \
                ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}" \
                ROBOTWIN_REQUEST_LOG_PATH="${task_dir}/requests.jsonl" \
                ROBOTWIN_REQUEST_IMAGE_DIR= \
                CUDA_HOME="${ROBOTWIN_CUDA_HOME}" \
                TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
                PYTHONPATH="${ROBOTWIN_CUROBO_SRC}:${PYTHONPATH:-}" \
                PYTHONUNBUFFERED=1 \
                "${eval_cmd[@]}" &
        eval_pid=$!
        early_stop_marker="${task_dir}/early_stop_zero_success"
        rm -f "${early_stop_marker}"
        if (( EARLY_STOP_ZERO_EPISODES > 0 )); then
            monitor_zero_success \
                "${eval_pid}" "${stdout_log}" "${gpu_console_log}" "${early_stop_marker}" &
            early_stop_monitor_pid=$!
        fi
        if wait "${eval_pid}"; then
            status=0
        else
            status=$?
        fi
        if [[ -n "${early_stop_monitor_pid}" ]]; then
            kill -TERM "${early_stop_monitor_pid}" 2>/dev/null || true
            wait "${early_stop_monitor_pid}" 2>/dev/null || true
            early_stop_monitor_pid=""
        fi
        eval_pid=""

        if [[ -f "${early_stop_marker}" ]]; then
            status=0
        fi

        end_time="$(date '+%Y-%m-%d %H:%M:%S')"
        end_seconds="$(date +%s)"
        elapsed_seconds=$((end_seconds - start_seconds))
        if [[ -f "${early_stop_marker}" ]]; then
            result_path="${task_dir}/early_stop_assumed_result.txt"
            printf '0\n' > "${result_path}"
            success_rate=0
        else
            result_path="$(sed -n 's/^Data has been saved to //p' "${stdout_log}" | tail -n 1)"
            success_rate="-"
        fi
        if [[ "${success_rate}" == "-" && -n "${result_path}" ]]; then
            if [[ "${result_path}" != /* ]]; then
                result_path="${ROBOTWIN_ROOT}/${result_path}"
            fi
            if [[ -f "${result_path}" ]]; then
                success_rate="$(awk '/^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {value=$1} END {if (value != "") print value}' "${result_path}")"
                success_rate=${success_rate:--}
            else
                result_path="-"
            fi
        elif [[ -z "${result_path}" ]]; then
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
        if [[ "${JOINT_PLANNER}" == "1" ]] && ! kill -0 "${planner_server_pid}" 2>/dev/null; then
            echo "[gpu ${gpu}] planner server exited unexpectedly; log=${planner_server_log}" >&2
            return 1
        fi
    done
    echo "[gpu ${gpu}] worker finished"
    printf '\n===== gpu %s queue finished =====\n' "${gpu}" >> "${gpu_console_log}"
    cleanup_worker
    trap - EXIT
}

worker_pids=()
worker_index=0
cleanup_main() {
    touch "${STOP_FILE}"
    if (( ${#worker_pids[@]} > 0 )); then
        kill -TERM "${worker_pids[@]}" 2>/dev/null || true
        wait "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap 'cleanup_main; exit 130' INT TERM

for gpu in "${GPUS[@]}"; do
    worker "${gpu}" "${worker_index}" &
    worker_pids+=("$!")
    worker_index=$((worker_index + 1))
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
