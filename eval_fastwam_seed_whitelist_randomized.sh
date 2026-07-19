#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash eval_fastwam_seed_whitelist_randomized.sh

Runs FastWAM on the randomized RoboTwin seed whitelist using one task worker
per GPU. Each task is a separate Python process, so FastWAM is reloaded when a
worker takes its next task.

Environment variables:
  GPU_LIST                  GPU ids, comma- or space-separated. By default,
                            select GPUs with at least GPU_MIN_FREE_MIB free.
  GPU_MIN_FREE_MIB          Minimum free memory for auto-selection (default: 30000)
  TASK_LIST                 Optional comma- or space-separated whitelist subset
  EVAL_TEST_NUM             Episodes per task (default: 100)
  NUM_INFERENCE_STEPS       FastWAM denoising steps (default: 10)
  REPLAN_STEPS              Actions executed before replanning (default: 24)
  SAVE_PREDICTED_VISUALS    Save infer_joint GIF and PNG frames (default: 0)
  PREDICTED_GIF_FPS         Predicted GIF playback rate (default: 4)
  TASK_TIMEOUT_SECONDS      Per-task timeout; 0 disables it (default: 0)
  CONTINUE_ON_ERROR         Continue after a task failure (default: 1)
  WORKER_START_STAGGER_SEC  Delay between worker starts (default: 15)
  IMPORT_WARMUP_RETRIES     Full dependency import attempts before eval (default: 10)
  IMPORT_WARMUP_DELAY_SEC   Delay between import attempts (default: 2)
  USE_LOCAL_RUNTIME         Run from a prepared node-local copy (default: 0)
  LOCAL_RUNTIME_ROOT        Prepared local copy (default: /opt/zjb_fastwam_runtime)
  FASTWAM_EVAL_OUTPUT_BASE  Persistent run output directory
  TMUX_MONITOR              Create one live-log tmux window per GPU (default: 1)
  DRY_RUN                   Validate only; do not start evaluation (default: 0)
  RUN_ID                    Override the generated run id
  FASTWAM_LOCK_ROOT         Local lock directory (default: /tmp/fastwam_eval_locks/RUN_ID)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_LOCAL_RUNTIME=${USE_LOCAL_RUNTIME:-0}
LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT:-/opt/zjb_fastwam_runtime}
if [[ "${USE_LOCAL_RUNTIME}" != "0" && "${USE_LOCAL_RUNTIME}" != "1" ]]; then
    echo "USE_LOCAL_RUNTIME must be 0 or 1, got: ${USE_LOCAL_RUNTIME}" >&2
    exit 2
fi
if [[ "${USE_LOCAL_RUNTIME}" == "1" && "${FASTWAM_LOCAL_ACTIVE:-0}" != "1" ]]; then
    LOCAL_ROBOTWIN_ROOT="${LOCAL_RUNTIME_ROOT}/RoboTwin_Astribot"
    LOCAL_FASTWAM_ROOT="${LOCAL_RUNTIME_ROOT}/FastWAM"
    LOCAL_FASTWAM_PYTHON="${LOCAL_RUNTIME_ROOT}/env/bin/python"
    LOCAL_EVAL_SCRIPT="${LOCAL_ROBOTWIN_ROOT}/eval_fastwam_seed_whitelist_randomized.sh"
    for local_path in "${LOCAL_EVAL_SCRIPT}" "${LOCAL_FASTWAM_PYTHON}" \
        "${LOCAL_FASTWAM_ROOT}/third_party/curobo/src" "${LOCAL_ROBOTWIN_ROOT}/assets"; do
        if [[ ! -e "${local_path}" ]]; then
            echo "Local runtime is incomplete: ${local_path}" >&2
            echo "Run: bash ${SCRIPT_DIR}/prepare_fastwam_local_runtime.sh" >&2
            exit 1
        fi
    done
    echo "[fastwam-eval] switching to local runtime: ${LOCAL_RUNTIME_ROOT}"
    exec env \
        FASTWAM_LOCAL_ACTIVE=1 \
        USE_LOCAL_RUNTIME=0 \
        FASTWAM_ROOT="${LOCAL_FASTWAM_ROOT}" \
        FASTWAM_PYTHON="${LOCAL_FASTWAM_PYTHON}" \
        FASTWAM_EVAL_OUTPUT_BASE="${FASTWAM_EVAL_OUTPUT_BASE:-${SCRIPT_DIR}/logs/eval_fastwam_seed_whitelist_randomized}" \
        bash "${LOCAL_EVAL_SCRIPT}"
fi

ROBOTWIN_ROOT="${SCRIPT_DIR}"
cd "${ROBOTWIN_ROOT}"
FASTWAM_ROOT=${FASTWAM_ROOT:-/private/zjb/workspace/FastWAM}
FASTWAM_PYTHON=${FASTWAM_PYTHON:-/private/zjb/conda_envs/fastwam/bin/python}
FASTWAM_ENV_ROOT="$(dirname "$(dirname "${FASTWAM_PYTHON}")")"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${FASTWAM_ROOT}/checkpoints/robotwin-ckpt/fast_wam/step_008995.pt}
DATASET_STATS_PATH=${DATASET_STATS_PATH:-${FASTWAM_ROOT}/checkpoints/robotwin-ckpt/fast_wam/dataset_stats.json}
SIM_CFG_PATH=${SIM_CFG_PATH:-${FASTWAM_ROOT}/configs/sim_robotwin.yaml}
SIM_TASK=${SIM_TASK:-astribot_uncond_1cam_384_1e-4}
POLICY_CONFIG="${ROBOTWIN_ROOT}/policy/fastwam_policy/deploy_policy.yml"
WHITELIST="${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml"
STEP_LIMITS="${ROBOTWIN_ROOT}/task_config/_eval_step_limit.yml"
TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
EVAL_SEED_LIST_ROOT="${ROBOTWIN_ROOT}/eval_seed_lists"
FFMPEG_SOURCE=${FFMPEG_SOURCE:-${FASTWAM_ENV_ROOT}/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
FFMPEG_BIN_DIR="/tmp/fastwam-ffmpeg-bin"

EVAL_TEST_NUM=${EVAL_TEST_NUM:-100}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
REPLAN_STEPS=${REPLAN_STEPS:-24}
SAVE_PREDICTED_VISUALS=${SAVE_PREDICTED_VISUALS:-0}
PREDICTED_GIF_FPS=${PREDICTED_GIF_FPS:-4}
GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-30000}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
WORKER_START_STAGGER_SEC=${WORKER_START_STAGGER_SEC:-15}
IMPORT_WARMUP_RETRIES=${IMPORT_WARMUP_RETRIES:-10}
IMPORT_WARMUP_DELAY_SEC=${IMPORT_WARMUP_DELAY_SEC:-2}
TMUX_MONITOR=${TMUX_MONITOR:-1}
DRY_RUN=${DRY_RUN:-0}

for value_name in EVAL_TEST_NUM NUM_INFERENCE_STEPS REPLAN_STEPS PREDICTED_GIF_FPS \
    GPU_MIN_FREE_MIB IMPORT_WARMUP_RETRIES; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
if [[ "${SAVE_PREDICTED_VISUALS}" != "0" && "${SAVE_PREDICTED_VISUALS}" != "1" ]]; then
    echo "SAVE_PREDICTED_VISUALS must be 0 or 1, got: ${SAVE_PREDICTED_VISUALS}" >&2
    exit 2
fi
if [[ "${SAVE_PREDICTED_VISUALS}" == "1" ]]; then
    SAVE_PREDICTED_VISUALS_ARG=true
else
    SAVE_PREDICTED_VISUALS_ARG=false
fi
for value_name in TASK_TIMEOUT_SECONDS WORKER_START_STAGGER_SEC IMPORT_WARMUP_DELAY_SEC; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done

required_commands=(awk flock jq nvidia-smi sed setsid sort tail tee timeout)
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

for path in "${FASTWAM_PYTHON}" "${CHECKPOINT_PATH}" "${DATASET_STATS_PATH}" \
    "${SIM_CFG_PATH}" "${POLICY_CONFIG}" "${WHITELIST}" "${STEP_LIMITS}" "${FFMPEG_SOURCE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done
if [[ ! -x "${FASTWAM_PYTHON}" ]]; then
    echo "FastWAM Python is not executable: ${FASTWAM_PYTHON}" >&2
    exit 1
fi

export NV_PREFIX=${NV_PREFIX:-/private/zjb/nvidia-vulkan-570.172.08}
export LD_LIBRARY_PATH="${NV_PREFIX}/root/usr/lib/x86_64-linux-gnu:${FASTWAM_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NV_PREFIX}/nvidia_icd.local.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NV_PREFIX}/root/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export PATH="${FFMPEG_BIN_DIR}:${FASTWAM_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${FASTWAM_ROOT}/third_party/curobo/src:${FASTWAM_ROOT}/src:${FASTWAM_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONUNBUFFERED=1

mkdir -p "${FFMPEG_BIN_DIR}"
ln -sfn "${FFMPEG_SOURCE}" "${FFMPEG_BIN_DIR}/ffmpeg"

warmup_imports() {
    local attempt
    for ((attempt = 1; attempt <= IMPORT_WARMUP_RETRIES; attempt++)); do
        echo "[fastwam-eval] dependency import warmup ${attempt}/${IMPORT_WARMUP_RETRIES}"
        if "${FASTWAM_PYTHON}" - <<'PY'
import hydra
import numpy
import omegaconf
import open3d
import sapien
import torch
import yaml
from scipy import linalg, ndimage, optimize, spatial, special, stats
from sklearn import decomposition, linear_model, metrics, neighbors, preprocessing, svm
from curobo.geom.sdf.world_mesh import WorldMeshCollision
PY
        then
            echo "[fastwam-eval] dependency import warmup passed"
            return 0
        fi
        if (( attempt < IMPORT_WARMUP_RETRIES && IMPORT_WARMUP_DELAY_SEC > 0 )); then
            sleep "${IMPORT_WARMUP_DELAY_SEC}"
        fi
    done
    return 1
}

if ! warmup_imports; then
    echo "FastWAM Python failed dependency import warmup after ${IMPORT_WARMUP_RETRIES} attempts: ${FASTWAM_PYTHON}" >&2
    exit 1
fi

if [[ -z "${GPU_LIST:-}" ]]; then
    GPU_LIST="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F ',' -v minimum="${GPU_MIN_FREE_MIB}" '$2 + 0 >= minimum {gsub(/[[:space:]]/, "", $1); print $1}' \
        | tr '\n' ' ')"
fi
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs selected. Set GPU_LIST explicitly or lower GPU_MIN_FREE_MIB." >&2
    exit 1
fi

declare -A SEEN_GPUS=()
declare -A GPU_FREE_MIB=()
while IFS=',' read -r gpu_index gpu_free; do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_free="${gpu_free//[[:space:]]/}"
    GPU_FREE_MIB["${gpu_index}"]="${gpu_free}"
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -z "${GPU_FREE_MIB[${gpu}]:-}" ]]; then
        echo "Invalid GPU id: ${gpu}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "Duplicate GPU id: ${gpu}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
    if (( GPU_FREE_MIB[${gpu}] < GPU_MIN_FREE_MIB )); then
        echo "GPU ${gpu} has only ${GPU_FREE_MIB[${gpu}]} MiB free; need ${GPU_MIN_FREE_MIB} MiB." >&2
        exit 1
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

for task in "${TASKS[@]}"; do
    seed_list_path="${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}/${task}.json"
    if [[ ! -f "${seed_list_path}" ]]; then
        echo "Eval seed list does not exist: ${seed_list_path}" >&2
        exit 1
    fi
    seed_count="$(jq -r 'if type == "array" then length elif (.entries | type) == "array" then (.entries | length) elif (.seeds | type) == "array" then (.seeds | length) else 0 end' "${seed_list_path}")"
    if [[ ! "${seed_count}" =~ ^[0-9]+$ ]] || (( seed_count < EVAL_TEST_NUM )); then
        echo "Need ${EVAL_TEST_NUM} seeds: ${seed_list_path} (found ${seed_count})" >&2
        exit 1
    fi
done

# eval_policy.py uses ckpt_setting both as a load path and as a result-directory
# component. A repo-local symlink keeps the latter bounded under the run root.
CHECKPOINT_ALIAS=".fastwam_eval_step_008995.pt"
ln -sfn "${CHECKPOINT_PATH}" "${ROBOTWIN_ROOT}/${CHECKPOINT_ALIAS}"

RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)_$$"}
FASTWAM_EVAL_OUTPUT_BASE=${FASTWAM_EVAL_OUTPUT_BASE:-${ROBOTWIN_ROOT}/logs/eval_fastwam_seed_whitelist_randomized}
RUN_ROOT="${FASTWAM_EVAL_OUTPUT_BASE}/${RUN_ID}"
LOCAL_LOCK_ROOT=${FASTWAM_LOCK_ROOT:-"/tmp/fastwam_eval_locks/${RUN_ID}"}
TASK_LOG_ROOT="${RUN_ROOT}/tasks"
GPU_CONSOLE_ROOT="${RUN_ROOT}/gpu_consoles"
EVAL_RESULT_ROOT="${RUN_ROOT}/eval_result"
NEXT_INDEX_FILE="${RUN_ROOT}/next_index.txt"
QUEUE_LOCK="${LOCAL_LOCK_ROOT}/queue.lock"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
SUMMARY_LOCK="${LOCAL_LOCK_ROOT}/summary.lock"
REPORT_FILE="${RUN_ROOT}/report.md"
STOP_FILE="${RUN_ROOT}/stop"
TMUX_SESSION="fastwam_eval_${RUN_ID}"

mkdir -p "${TASK_LOG_ROOT}" "${GPU_CONSOLE_ROOT}" "${EVAL_RESULT_ROOT}" "${LOCAL_LOCK_ROOT}"
printf '0\n' > "${NEXT_INDEX_FILE}"
printf 'idx\ttask\tgpu\tstatus\tstart_time\tend_time\tseconds\tlog_path\tsuccess_rate\tresult_path\n' > "${SUMMARY_FILE}"

write_markdown_report() {
    local tmp_path="${REPORT_FILE}.tmp"
    local completed_count failed_count status_text success_text result_link
    completed_count="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    failed_count="$(awk -F '\t' 'NR > 1 && $4 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    {
        printf '# FastWAM RoboTwin Randomized Evaluation\n\n'
        printf -- '- Checkpoint: `%s`\n' "${CHECKPOINT_PATH}"
        printf -- '- Task config: `%s`\n' "${TASK_CONFIG}"
        printf -- '- Episodes per task: `%s`\n' "${EVAL_TEST_NUM}"
        printf -- '- Inference steps: `%s`\n' "${NUM_INFERENCE_STEPS}"
        printf -- '- Replan steps: `%s`\n' "${REPLAN_STEPS}"
        printf -- '- Predicted GIF/frames: `%s`\n' "${SAVE_PREDICTED_VISUALS}"
        printf -- '- GPUs: `%s`\n' "${GPUS[*]}"
        printf -- '- Progress: `%s/%s`\n' "${completed_count}" "${#TASKS[@]}"
        printf -- '- Failed tasks: `%s`\n' "${failed_count}"
        if [[ "${TMUX_MONITOR}" == "1" ]]; then
            printf -- '- Tmux session: `%s`\n' "${TMUX_SESSION}"
        fi
        printf '\n| # | Task | GPU | Status | Success Rate | Seconds | Log | Result |\n'
        printf '|---:|---|---:|---|---:|---:|---|---|\n'
        while IFS=$'\t' read -r idx task gpu status start_time end_time seconds log_path success_rate result_path; do
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
            printf '| %s | %s | %s | %s | %s | %s | [log](<%s>) | %s |\n' \
                "$((idx + 1))" "${task}" "${gpu}" "${status_text}" "${success_text}" \
                "${seconds}" "${log_path}" "${result_link}"
        done < <(tail -n +2 "${SUMMARY_FILE}" | sort -t $'\t' -k1,1n)
    } > "${tmp_path}"
    mv "${tmp_path}" "${REPORT_FILE}"
}

write_markdown_report

echo "[fastwam-eval] checkpoint=${CHECKPOINT_PATH}"
echo "[fastwam-eval] task_config=${TASK_CONFIG} tasks=${#TASKS[@]} episodes_per_task=${EVAL_TEST_NUM}"
echo "[fastwam-eval] inference_steps=${NUM_INFERENCE_STEPS} replan_steps=${REPLAN_STEPS}"
echo "[fastwam-eval] save_predicted_visuals=${SAVE_PREDICTED_VISUALS} gif_fps=${PREDICTED_GIF_FPS}"
echo "[fastwam-eval] gpus=${GPUS[*]}"
echo "[fastwam-eval] run_root=${RUN_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] validation passed; no evaluation process was started"
    exit 0
fi

setup_tmux_monitor() {
    local gpu console_log monitor_command first_window=1
    for gpu in "${GPUS[@]}"; do
        console_log="${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
        : > "${console_log}"
        printf '[gpu %s] waiting for an evaluation task...\n' "${gpu}" >> "${console_log}"
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
    echo "[fastwam-eval] monitor: tmux attach -t ${TMUX_SESSION}"
}

if [[ "${TMUX_MONITOR}" == "1" ]]; then
    setup_tmux_monitor
else
    for gpu in "${GPUS[@]}"; do
        : > "${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
    done
fi

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
    local idx="$1" task="$2" gpu="$3" status="$4" start_time="$5"
    local end_time="$6" seconds="$7" log_path="$8" success_rate="$9" result_path="${10}"
    {
        flock -x 201
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${idx}" "${task}" "${gpu}" "${status}" "${start_time}" "${end_time}" \
            "${seconds}" "${log_path}" "${success_rate}" "${result_path}" >> "${SUMMARY_FILE}"
        write_markdown_report
    } 201>"${SUMMARY_LOCK}"
}

worker() {
    local gpu="$1"
    local gpu_console_log="${GPU_CONSOLE_ROOT}/gpu_${gpu}.log"
    local eval_pid=""

    cleanup_worker() {
        if [[ -n "${eval_pid}" ]] && kill -0 "${eval_pid}" 2>/dev/null; then
            kill -TERM -- "-${eval_pid}" 2>/dev/null || true
            wait "${eval_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_worker EXIT
    trap 'exit 130' INT TERM

    local item idx task task_dir stdout_log seed_list_path start_time end_time
    local start_seconds end_seconds elapsed_seconds status result_path success_rate
    while [[ ! -f "${STOP_FILE}" ]]; do
        if ! item="$(get_next_task)"; then
            break
        fi
        idx="${item%%$'\t'*}"
        task="${item#*$'\t'}"
        task_dir="${TASK_LOG_ROOT}/$(printf '%02d' "$((idx + 1))")_${task}"
        stdout_log="${task_dir}/stdout.log"
        seed_list_path="${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}/${task}.json"
        mkdir -p "${task_dir}"
        : > "${stdout_log}"

        start_time="$(date '+%Y-%m-%d %H:%M:%S')"
        start_seconds="$(date +%s)"
        echo "[gpu ${gpu}] start $((idx + 1))/${#TASKS[@]} ${task}"
        printf '\n===== task %s/%s start: %s =====\n' \
            "$((idx + 1))" "${#TASKS[@]}" "${task}" >> "${gpu_console_log}"

        local base_cmd=(
            "${FASTWAM_PYTHON}" -u script/eval_policy.py
            --config "${POLICY_CONFIG}"
            --overrides
            --task_name "${task}"
            --task_config "${TASK_CONFIG}"
            --ckpt_setting "${CHECKPOINT_ALIAS}"
            --seed 0
            --policy_name fastwam_policy
            --instruction_type unseen
            --test_num "${EVAL_TEST_NUM}"
            --eval_seed_list_path "${seed_list_path}"
            --sim_cfg_path "${SIM_CFG_PATH}"
            --sim_task "${SIM_TASK}"
            --mixed_precision bf16
            --device cuda
            --dataset_stats_path "${DATASET_STATS_PATH}"
            --replan_steps "${REPLAN_STEPS}"
            --num_inference_steps "${NUM_INFERENCE_STEPS}"
            --save_predicted_visuals "${SAVE_PREDICTED_VISUALS_ARG}"
            --predicted_visuals_dir "${task_dir}/predictions"
            --predicted_gif_fps "${PREDICTED_GIF_FPS}"
        )
        local eval_cmd=("${base_cmd[@]}")
        if [[ "${TASK_TIMEOUT_SECONDS}" != "0" ]]; then
            eval_cmd=(timeout --signal=TERM --kill-after=30 "${TASK_TIMEOUT_SECONDS}" "${base_cmd[@]}")
        fi

        setsid bash -o pipefail -c '
            task_log=$1
            gpu_log=$2
            shift 2
            "$@" 2>&1 | tee -a "$task_log" "$gpu_log" >/dev/null
        ' _ "${stdout_log}" "${gpu_console_log}" \
            env \
                CUDA_VISIBLE_DEVICES="${gpu}" \
                ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}" \
                ROBOTWIN_SKIP_RENDER_TEST=1 \
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
        append_summary "${idx}" "${task}" "${gpu}" "${status}" "${start_time}" \
            "${end_time}" "${elapsed_seconds}" "${stdout_log}" "${success_rate}" "${result_path}"
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

worker_index=0
for gpu in "${GPUS[@]}"; do
    worker "${gpu}" &
    worker_pids+=("$!")
    worker_index=$((worker_index + 1))
    if (( worker_index < ${#GPUS[@]} && WORKER_START_STAGGER_SEC > 0 )); then
        sleep "${WORKER_START_STAGGER_SEC}"
    fi
done

worker_failure=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        worker_failure=1
    fi
done

done_count="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
failed_count="$(awk -F '\t' 'NR > 1 && $4 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
echo "[fastwam-eval] finished=${done_count}/${#TASKS[@]} failed=${failed_count}"
echo "[fastwam-eval] summary=${SUMMARY_FILE}"
echo "[fastwam-eval] report=${REPORT_FILE}"
if [[ "${TMUX_MONITOR}" == "1" ]]; then
    echo "[fastwam-eval] tmux_session=${TMUX_SESSION}"
fi

if (( done_count != ${#TASKS[@]} || failed_count != 0 || worker_failure != 0 )); then
    exit 1
fi
