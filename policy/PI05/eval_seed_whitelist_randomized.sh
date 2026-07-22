#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash policy/PI05/eval_seed_whitelist_randomized.sh

Environment variables:
  GPU_LIST                    GPU ids, comma- or space-separated (default: all GPUs)
  PI05_CKPT_KEY               Checkpoint key (default: pi05_astribot_global_step_100000)
  TASK_CONFIG                 RoboTwin task config (default: info_gathering_randomized)
  WHITELIST                   Task whitelist YAML (default: task_config/eval_seed_task_whitelist.yml)
  EVAL_SEED_LIST_ROOT         Eval seed-list root (default: eval_seed_lists)
  EVAL_TEST_NUM               Episodes per task (default: 50)
  TASK_LIMIT                  Only run the first N tasks; 0 means all (default: 0)
  PI05_MAX_ACTIONS_PER_CALL   Actions executed per inference request (default: 16)
  PI05_ACTION_HORIZON         Maximum accepted action horizon (default: deploy config)
  PI05_NUM_STEPS              Model denoising steps, not executed actions (default: 5)
  PI05_ACTION_SEMANTICS       absolute or delta_to_abs (default: absolute)
  PI05_PORT_BASE              Per-GPU port is base + physical GPU id (default: 5702)
  SERVER_READY_TIMEOUT        Server startup timeout in seconds (default: 900)
  PI05_REQUEST_TIMEOUT        Per-request timeout in seconds (default: 180)
  TASK_TIMEOUT_SECONDS        Per-task timeout; 0 disables it (default: 0)
  CONTINUE_ON_ERROR           Continue after an eval failure (default: 1)
  SERVER_START_STAGGER_SEC    Delay between worker launches (default: 1)
  RESUME_FROM_RUN             Prior run root; successful tasks are skipped
  TMUX_MONITOR                Open one tmux window per GPU (default: 1)
  DRY_RUN                     Validate without starting servers or tasks (default: 0)

Runtime path overrides:
  ROBOTWIN_PYTHON, PI05_PYTHON, PI05_RLINF_ROOT, OPENPI_CLIENT_SRC,
  ROBOTWIN_CLIENT_DEPS, CKPT_MAPPING
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
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PI05_CKPT_KEY=${PI05_CKPT_KEY:-pi05_astribot_global_step_100000}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy_policy.yml}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
EVAL_SEED_LIST_ROOT=${EVAL_SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}

ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
PI05_PYTHON=${PI05_PYTHON:-/root/autodl-tmp/PI05_eval/PI05/env/rlinf-pi05/bin/python}
PI05_RLINF_ROOT=${PI05_RLINF_ROOT:-/root/autodl-tmp/PI05_eval/PI05/source/RLinf}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-${ROBOTWIN_ROOT}/policy/pi05/packages/openpi-client/src}
ROBOTWIN_CLIENT_DEPS=${ROBOTWIN_CLIENT_DEPS:-${SCRIPT_DIR}/env/robotwin_client_deps}

EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}
TASK_LIMIT=${TASK_LIMIT:-0}
PI05_NUM_STEPS=${PI05_NUM_STEPS:-5}
PI05_ACTION_SEMANTICS=${PI05_ACTION_SEMANTICS:-absolute}
PI05_PORT_BASE=${PI05_PORT_BASE:-5702}
PI05_REQUEST_TIMEOUT=${PI05_REQUEST_TIMEOUT:-180}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SERVER_START_STAGGER_SEC=${SERVER_START_STAGGER_SEC:-1}
TMUX_MONITOR=${TMUX_MONITOR:-1}
DRY_RUN=${DRY_RUN:-0}
RESUME_FROM_RUN=${RESUME_FROM_RUN:-}
SEED=${SEED:-0}

OPENPI_TORCH_COMPILE=${OPENPI_TORCH_COMPILE:-0}
OPENPI_TORCH_COMPILE_MODE=${OPENPI_TORCH_COMPILE_MODE:-max-autotune}
ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR=${ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR:-rt}
ROBOTWIN_SAPIEN_RT_SAMPLES=${ROBOTWIN_SAPIEN_RT_SAMPLES:-32}
ROBOTWIN_SAPIEN_RT_PATH_DEPTH=${ROBOTWIN_SAPIEN_RT_PATH_DEPTH:-8}
ROBOTWIN_SAPIEN_RT_DENOISER=${ROBOTWIN_SAPIEN_RT_DENOISER:-none}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-False}
PI05_LOG_REQUEST_DEBUG=${PI05_LOG_REQUEST_DEBUG:-False}
PI05_LOG_CHUNK_TIMING=${PI05_LOG_CHUNK_TIMING:-True}

required_commands=(awk flock nvidia-smi sed setsid sort tail tee timeout)
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

for path in "${SCRIPT_DIR}" "${ROBOTWIN_ROOT}" "${PI05_RLINF_ROOT}" \
    "${OPENPI_CLIENT_SRC}" "${ROBOTWIN_CLIENT_DEPS}" "${EVAL_SEED_LIST_ROOT}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Required directory does not exist: ${path}" >&2
        exit 1
    fi
done
for path in "${CKPT_MAPPING}" "${DEPLOY_CONFIG}" "${WHITELIST}" \
    "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file does not exist: ${path}" >&2
        exit 1
    fi
done
if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
    echo "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${PI05_PYTHON}" ]]; then
    echo "PI05_PYTHON is not executable: ${PI05_PYTHON}" >&2
    exit 1
fi
if ! PYTHONPATH="${ROBOTWIN_CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" \
    "${ROBOTWIN_PYTHON}" -c \
    'import msgpack, numpy, requests, sapien, websockets, yaml; import openpi_client.msgpack_numpy' \
    >/dev/null 2>&1; then
    echo "RoboTwin Python failed to import simulator or PI05 client dependencies." >&2
    exit 1
fi
if ! PYTHONPATH="${PI05_RLINF_ROOT}:${PYTHONPATH:-}" \
    "${PI05_PYTHON}" -c 'import openpi, rlinf, torch, tyro, websockets' \
    >/dev/null 2>&1; then
    echo "PI05 Python failed to import server dependencies." >&2
    exit 1
fi

for value_name in EVAL_TEST_NUM PI05_NUM_STEPS PI05_PORT_BASE SERVER_READY_TIMEOUT \
    PI05_REQUEST_TIMEOUT; do
    value=${!value_name}
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
for value_name in TASK_LIMIT TASK_TIMEOUT_SECONDS SERVER_START_STAGGER_SEC; do
    value=${!value_name}
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done
if [[ "${PI05_ACTION_SEMANTICS}" != "absolute" && "${PI05_ACTION_SEMANTICS}" != "delta_to_abs" ]]; then
    echo "PI05_ACTION_SEMANTICS must be absolute or delta_to_abs." >&2
    exit 2
fi

read_deploy_int() {
    local key="$1"
    "${ROBOTWIN_PYTHON}" - "${DEPLOY_CONFIG}" "${key}" <<'PY'
import sys
from pathlib import Path

import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
value = payload.get(sys.argv[2])
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise SystemExit(f"{sys.argv[2]} must be a positive integer in {sys.argv[1]}")
print(value)
PY
}

PI05_ACTION_HORIZON=${PI05_ACTION_HORIZON:-$(read_deploy_int action_horizon)}
PI05_MAX_ACTIONS_PER_CALL=${PI05_MAX_ACTIONS_PER_CALL:-16}
for value_name in PI05_ACTION_HORIZON PI05_MAX_ACTIONS_PER_CALL; do
    value=${!value_name}
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
if (( PI05_MAX_ACTIONS_PER_CALL > PI05_ACTION_HORIZON )); then
    echo "PI05_MAX_ACTIONS_PER_CALL cannot exceed PI05_ACTION_HORIZON." >&2
    exit 2
fi

if [[ -z "${GPU_LIST:-}" ]]; then
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
    port=$((PI05_PORT_BASE + gpu))
    if (( port > 65535 )); then
        echo "GPU ${gpu} produces invalid port ${port}." >&2
        exit 2
    fi
done

mapfile -t TASKS < <("${ROBOTWIN_PYTHON}" - "${WHITELIST}" <<'PY'
import sys
from pathlib import Path

import yaml

items = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(items, list):
    raise SystemExit("whitelist must contain a YAML list")
for item in items:
    if not isinstance(item, str) or not item:
        raise SystemExit(f"invalid whitelist item: {item!r}")
    print(item)
PY
)
if (( ${#TASKS[@]} == 0 )); then
    echo "No tasks found in ${WHITELIST}." >&2
    exit 1
fi
if (( TASK_LIMIT > 0 && TASK_LIMIT < ${#TASKS[@]} )); then
    TASKS=("${TASKS[@]:0:TASK_LIMIT}")
fi

"${ROBOTWIN_PYTHON}" - "${EVAL_SEED_LIST_ROOT}" "${TASK_CONFIG}" "${EVAL_TEST_NUM}" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task_config = sys.argv[2]
required_count = int(sys.argv[3])
errors = []
for task in sys.argv[4:]:
    path = root / task_config / f"{task}.json"
    if not path.is_file():
        errors.append(f"missing: {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("entries", payload.get("seeds", []))
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    if not isinstance(entries, list) or len(entries) < required_count:
        errors.append(f"needs {required_count} seeds: {path} (found {len(entries) if isinstance(entries, list) else 0})")
if errors:
    raise SystemExit("\n".join(errors))
print(f"seed lists validated: tasks={len(sys.argv[4:])} episodes_per_task={required_count}")
PY

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
RUN_ROOT="${SCRIPT_DIR}/runs/eval_seed_whitelist_randomized/${RUN_ID}"
TASK_LOG_ROOT="${RUN_ROOT}/tasks"
SERVER_LOG_ROOT="${RUN_ROOT}/servers"
GPU_CONSOLE_ROOT="${RUN_ROOT}/gpu_consoles"
EVAL_RESULT_ROOT="${RUN_ROOT}/eval_result"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
SUMMARY_LOCK="${RUN_ROOT}/summary.lock"
SERVER_SUMMARY_FILE="${RUN_ROOT}/servers.tsv"
SERVER_SUMMARY_LOCK="${RUN_ROOT}/servers.lock"
REPORT_FILE="${RUN_ROOT}/report.md"
MANIFEST_FILE="${RUN_ROOT}/manifest.yaml"
RESOLVED_JSON="${RUN_ROOT}/resolved_checkpoint.json"
NEXT_INDEX_FILE="${RUN_ROOT}/next_index.txt"
QUEUE_LOCK="${RUN_ROOT}/queue.lock"
STOP_FILE="${RUN_ROOT}/stop"
TMUX_SESSION="pi05_randomized_${RUN_ID}"
mkdir -p "${TASK_LOG_ROOT}" "${SERVER_LOG_ROOT}" "${GPU_CONSOLE_ROOT}" "${EVAL_RESULT_ROOT}"

mapfile -t CHECKPOINT_FIELDS < <("${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${PI05_CKPT_KEY}" "${RESOLVED_JSON}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

mapping_path = Path(sys.argv[1])
key = sys.argv[2]
payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
checkpoints = payload.get("checkpoints") or {}
unavailable = payload.get("unavailable") or {}
if key in unavailable:
    raise SystemExit(f"checkpoint is unavailable: {key}: {unavailable[key]}")
if key not in checkpoints:
    raise SystemExit(f"checkpoint key not found in {mapping_path}: {key}")
record = dict(checkpoints[key])
checkpoint_dir = Path(record["checkpoint_dir"])
if not checkpoint_dir.is_dir():
    raise SystemExit(f"checkpoint directory does not exist: {checkpoint_dir}")
for relative_path in record.get("required_files", []):
    path = checkpoint_dir / relative_path
    if not path.is_file():
        raise SystemExit(f"missing checkpoint file: {path}")
Path(sys.argv[3]).write_text(json.dumps(record, indent=2), encoding="utf-8")
print(checkpoint_dir)
print(record["openpi_config_name"])
print(record.get("step", ""))
print(record.get("stage", ""))
PY
)
CHECKPOINT_DIR=${CHECKPOINT_FIELDS[0]}
OPENPI_CONFIG_NAME=${CHECKPOINT_FIELDS[1]}
CHECKPOINT_STEP=${CHECKPOINT_FIELDS[2]}
CHECKPOINT_STAGE=${CHECKPOINT_FIELDS[3]}

cat > "${MANIFEST_FILE}" <<EOF
run_id: ${RUN_ID}
baseline: PI05
checkpoint_key: ${PI05_CKPT_KEY}
checkpoint_dir: ${CHECKPOINT_DIR}
checkpoint_step: ${CHECKPOINT_STEP}
checkpoint_stage: ${CHECKPOINT_STAGE}
openpi_config_name: ${OPENPI_CONFIG_NAME}
task_config: ${TASK_CONFIG}
whitelist: ${WHITELIST}
eval_seed_list_root: ${EVAL_SEED_LIST_ROOT}
eval_test_num: ${EVAL_TEST_NUM}
task_limit: ${TASK_LIMIT}
gpus: "${GPUS[*]}"
port_base: ${PI05_PORT_BASE}
pi05_num_steps: ${PI05_NUM_STEPS}
pi05_action_horizon: ${PI05_ACTION_HORIZON}
pi05_max_actions_per_call: ${PI05_MAX_ACTIONS_PER_CALL}
pi05_action_semantics: ${PI05_ACTION_SEMANTICS}
pi05_request_timeout: ${PI05_REQUEST_TIMEOUT}
task_timeout_seconds: ${TASK_TIMEOUT_SECONDS}
robotwin_python: ${ROBOTWIN_PYTHON}
pi05_python: ${PI05_PYTHON}
pi05_rlinf_root: ${PI05_RLINF_ROOT}
openpi_client_src: ${OPENPI_CLIENT_SRC}
run_root: ${RUN_ROOT}
EOF

printf '0\n' > "${NEXT_INDEX_FILE}"
printf 'idx\ttask\tgpu\tport\tstatus\tstart_time\tend_time\tseconds\tlog_path\tsuccess_rate\tresult_path\n' > "${SUMMARY_FILE}"
if [[ -n "${RESUME_FROM_RUN}" ]]; then
    RESUME_FROM_RUN="$(cd "${RESUME_FROM_RUN}" && pwd)"
    RESUME_SUMMARY="${RESUME_FROM_RUN}/summary.tsv"
    if [[ ! -f "${RESUME_SUMMARY}" ]]; then
        echo "Resume summary does not exist: ${RESUME_SUMMARY}" >&2
        exit 1
    fi
    awk -F '\t' 'NR > 1 && $5 == 0' "${RESUME_SUMMARY}" >> "${SUMMARY_FILE}"
fi
printf 'role\tgpu\tport\tstatus\tpid\tlog_path\n' > "${SERVER_SUMMARY_FILE}"

write_markdown_report() {
    local tmp_path="${REPORT_FILE}.tmp"
    local completed_count failed_count status_text success_text result_link
    completed_count="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    failed_count="$(awk -F '\t' 'NR > 1 && $5 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    {
        printf '# PI05 Randomized Seed Eval Report\n\n'
        printf -- '- Checkpoint: `%s` (`%s`, step `%s`)\n' "${PI05_CKPT_KEY}" "${CHECKPOINT_STAGE}" "${CHECKPOINT_STEP}"
        printf -- '- Checkpoint dir: `%s`\n' "${CHECKPOINT_DIR}"
        printf -- '- Task config: `%s`\n' "${TASK_CONFIG}"
        printf -- '- Task whitelist: `%s`\n' "${WHITELIST}"
        printf -- '- Eval seed lists: `%s/%s`\n' "${EVAL_SEED_LIST_ROOT}" "${TASK_CONFIG}"
        printf -- '- Episodes per task: `%s`\n' "${EVAL_TEST_NUM}"
        printf -- '- GPUs: `%s`\n' "${GPUS[*]}"
        printf -- '- Denoising steps: `%s`\n' "${PI05_NUM_STEPS}"
        printf -- '- Executed actions per request: `%s` (horizon `%s`)\n' \
            "${PI05_MAX_ACTIONS_PER_CALL}" "${PI05_ACTION_HORIZON}"
        if [[ -n "${RESUME_FROM_RUN}" ]]; then
            printf -- '- Resumed from: `%s`\n' "${RESUME_FROM_RUN}"
        fi
        if [[ "${TMUX_MONITOR}" == "1" ]]; then
            printf -- '- Tmux session: `%s`\n' "${TMUX_SESSION}"
        fi
        printf -- '- Progress: `%s/%s`, failed: `%s`\n\n' \
            "${completed_count}" "${#TASKS[@]}" "${failed_count}"
        printf '| # | Task | GPU | Port | Status | Success Rate | Seconds | Log | Result |\n'
        printf '|---:|---|---:|---:|---|---:|---:|---|---|\n'
        while IFS=$'\t' read -r idx task gpu port status start_time end_time seconds log_path success_rate result_path; do
            if [[ "${status}" == "0" ]]; then
                status_text=success
            else
                status_text="failed (${status})"
            fi
            if [[ "${success_rate}" == "-" ]]; then
                success_text=-
            else
                success_text="$(awk -v rate="${success_rate}" 'BEGIN {printf "%.1f%%", rate * 100}')"
            fi
            if [[ "${result_path}" == "-" ]]; then
                result_link=-
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

write_markdown_report

echo "[pi05-launcher] checkpoint=${PI05_CKPT_KEY} dir=${CHECKPOINT_DIR}"
echo "[pi05-launcher] task_config=${TASK_CONFIG} tasks=${#TASKS[@]} episodes=${EVAL_TEST_NUM}"
echo "[pi05-launcher] seed_lists=${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}"
echo "[pi05-launcher] gpus=${GPUS[*]} ports=${PI05_PORT_BASE}+gpu_id"
echo "[pi05-launcher] denoise_steps=${PI05_NUM_STEPS} actions_per_request=${PI05_MAX_ACTIONS_PER_CALL}"
echo "[pi05-launcher] run_root=${RUN_ROOT}"

port_is_listening() {
    local port="$1"
    "${ROBOTWIN_PYTHON}" - "${port}" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.5):
        pass
except OSError:
    raise SystemExit(1)
PY
}

for gpu in "${GPUS[@]}"; do
    port=$((PI05_PORT_BASE + gpu))
    if port_is_listening "${port}"; then
        echo "Port ${port} for GPU ${gpu} is already in use." >&2
        exit 1
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    for gpu in "${GPUS[@]}"; do
        echo "[dry-run] gpu=${gpu} server_port=$((PI05_PORT_BASE + gpu)) device=cuda:0"
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
        printf '[gpu %s] waiting for PI05 server and eval task...\n' "${gpu}" >> "${console_log}"
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
    echo "[pi05-launcher] tmux_session=${TMUX_SESSION}"
    echo "[pi05-launcher] attach: tmux attach -t ${TMUX_SESSION}"
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
    local server_log="$3"
    "${ROBOTWIN_PYTHON}" - "${pid}" "${port}" "${SERVER_READY_TIMEOUT}" "${server_log}" <<'PY'
import os
import sys
import time
from pathlib import Path

import requests

pid, port, timeout = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
server_log = Path(sys.argv[4])
url = f"http://127.0.0.1:{port}/healthz"
start = time.time()
last_error = None
while time.time() - start < timeout:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        tail = "\n".join(server_log.read_text(errors="replace").splitlines()[-80:]) if server_log.exists() else ""
        raise SystemExit(f"server exited before readiness\n{tail}")
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print(f"server ready: {url}")
            raise SystemExit(0)
        last_error = f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
raise SystemExit(f"server not ready after {timeout}s: {last_error}")
PY
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
        if awk -F '\t' -v task="${task}" '$2 == task && $5 == 0 {found=1} END {exit !found}' "${SUMMARY_FILE}"; then
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
            "${idx}" "${task}" "${gpu}" "${port}" "${status}" "${start_time}" \
            "${end_time}" "${seconds}" "${log_path}" "${success_rate}" "${result_path}" \
            >> "${SUMMARY_FILE}"
        write_markdown_report
    } 201>"${SUMMARY_LOCK}"
}

append_server_summary() {
    local gpu="$1" port="$2" status="$3" pid="$4" log_path="$5"
    {
        flock -x 202
        printf 'policy\t%s\t%s\t%s\t%s\t%s\n' \
            "${gpu}" "${port}" "${status}" "${pid}" "${log_path}" >> "${SERVER_SUMMARY_FILE}"
    } 202>"${SERVER_SUMMARY_LOCK}"
}

worker() {
    local gpu="$1"
    local port=$((PI05_PORT_BASE + gpu))
    local server_log="${SERVER_LOG_ROOT}/pi05_gpu${gpu}_port${port}.log"
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

    echo "[gpu ${gpu}] starting PI05 server on port ${port}"
    printf '\n===== server start: gpu=%s port=%s =====\n' "${gpu}" "${port}" >> "${gpu_console_log}"
    (
        cd "${PI05_RLINF_ROOT}"
        export PYTHONPATH="${PI05_RLINF_ROOT}:${PYTHONPATH:-}"
        export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/PI05_eval/PI05/cache/openpi}"
        export HF_HOME="${HF_HOME:-/root/autodl-tmp/PI05_eval/PI05/cache/huggingface}"
        export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
        export OPENPI_TORCH_COMPILE OPENPI_TORCH_COMPILE_MODE
        mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"
        exec setsid env CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PI05_PYTHON}" -m toolkits.standalone_eval_scripts.openpi.serve_policy \
            --config-name "${OPENPI_CONFIG_NAME}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --host 127.0.0.1 \
            --port "${port}" \
            --device cuda:0 \
            --num-steps "${PI05_NUM_STEPS}"
    ) > "${server_log}" 2>&1 &
    server_pid=$!

    if ! wait_for_server "${server_pid}" "${port}" "${server_log}"; then
        echo "[gpu ${gpu}] server failed to become ready; log=${server_log}" >&2
        append_server_summary "${gpu}" "${port}" not_ready "${server_pid}" "${server_log}"
        return 1
    fi
    append_server_summary "${gpu}" "${port}" ready "${server_pid}" "${server_log}"
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
        task_dir="${TASK_LOG_ROOT}/$(printf '%03d' "$((idx + 1))")_${task}"
        stdout_log="${task_dir}/stdout.log"
        mkdir -p "${task_dir}"

        start_time="$(date -Is)"
        start_seconds="$(date +%s)"
        echo "[gpu ${gpu}] start $((idx + 1))/${#TASKS[@]} ${task}"
        printf '\n===== task %s/%s start: %s =====\n' \
            "$((idx + 1))" "${#TASKS[@]}" "${task}" >> "${gpu_console_log}"

        local eval_cmd=(
            "${ROBOTWIN_PYTHON}" script/eval_policy.py
            --config policy/PI05/deploy_policy.yml
            --overrides
            --task_name "${task}"
            --task_config "${TASK_CONFIG}"
            --ckpt_setting "${PI05_CKPT_KEY}"
            --seed "${SEED}"
            --test_num "${EVAL_TEST_NUM}"
            --policy_name PI05
            --host 127.0.0.1
            --port "${port}"
            --request_timeout "${PI05_REQUEST_TIMEOUT}"
            --openpi_client_src "${OPENPI_CLIENT_SRC}"
            --action_horizon "${PI05_ACTION_HORIZON}"
            --max_actions_per_call "${PI05_MAX_ACTIONS_PER_CALL}"
            --action_semantics "${PI05_ACTION_SEMANTICS}"
            --log_request_debug "${PI05_LOG_REQUEST_DEBUG}"
            --log_chunk_timing "${PI05_LOG_CHUNK_TIMING}"
            --temp_root "${task_dir}/pi05_debug"
            --eval_video_log "${ROBOTWIN_EVAL_VIDEO_LOG}"
        )
        if (( TASK_TIMEOUT_SECONDS > 0 )); then
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
                CUDA_VISIBLE_DEVICES="${gpu}" \
                PI05_HOST=127.0.0.1 \
                PI05_PORT="${port}" \
                PI05_ACTION_SEMANTICS="${PI05_ACTION_SEMANTICS}" \
                OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC}" \
                ROBOTWIN_USE_EVAL_SEED_LIST=1 \
                ROBOTWIN_EVAL_SEED_LIST_PATH="${EVAL_SEED_LIST_ROOT}" \
                ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}" \
                ROBOTWIN_SAPIEN_RENDER_DEVICE=cuda:0 \
                ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR="${ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR}" \
                ROBOTWIN_SAPIEN_RT_SAMPLES="${ROBOTWIN_SAPIEN_RT_SAMPLES}" \
                ROBOTWIN_SAPIEN_RT_PATH_DEPTH="${ROBOTWIN_SAPIEN_RT_PATH_DEPTH}" \
                ROBOTWIN_SAPIEN_RT_DENOISER="${ROBOTWIN_SAPIEN_RT_DENOISER}" \
                PYTHONPATH="${ROBOTWIN_CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" \
                PYTHONUNBUFFERED=1 \
                "${eval_cmd[@]}" &
        eval_pid=$!
        if wait "${eval_pid}"; then
            status=0
        else
            status=$?
        fi
        eval_pid=""

        end_time="$(date -Is)"
        end_seconds="$(date +%s)"
        elapsed_seconds=$((end_seconds - start_seconds))
        result_path="$(sed -n 's/^Data has been saved to //p' "${stdout_log}" | tail -n 1)"
        success_rate=-
        if [[ -n "${result_path}" && -f "${result_path}" ]]; then
            success_rate="$(awk 'NF && $1 ~ /^[-+0-9.]+$/ {v=$1} END {if (v == "") print "-"; else print v}' "${result_path}")"
        else
            result_path=-
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
    cleanup_worker
    trap - EXIT
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
    if (( SERVER_START_STAGGER_SEC > 0 )); then
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
echo "[pi05-launcher] finished=${done_count}/${#TASKS[@]} failed=${failed_count}"
echo "[pi05-launcher] summary=${SUMMARY_FILE}"
echo "[pi05-launcher] servers=${SERVER_SUMMARY_FILE}"
echo "[pi05-launcher] report=${REPORT_FILE}"
if [[ "${TMUX_MONITOR}" == "1" ]]; then
    echo "[pi05-launcher] tmux_session=${TMUX_SESSION}"
    echo "[pi05-launcher] close tmux: tmux kill-session -t ${TMUX_SESSION}"
fi

if (( done_count != ${#TASKS[@]} || failed_count != 0 || worker_failure != 0 )); then
    exit 1
fi
