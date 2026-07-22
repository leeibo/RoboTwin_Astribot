#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE=${MODE:-smoke}
PI05_CKPT_KEY=${PI05_CKPT_KEY:-pi05_astribot_global_step_100000}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
PI05_PYTHON=${PI05_PYTHON:-/root/autodl-tmp/PI05_eval/PI05/env/rlinf-pi05/bin/python}
PI05_RLINF_ROOT=${PI05_RLINF_ROOT:-/root/autodl-tmp/PI05_eval/PI05/source/RLinf}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-/root/autodl-tmp/RoboTwin_Astribot/policy/pi05/packages/openpi-client/src}
ROBOTWIN_CLIENT_DEPS=${ROBOTWIN_CLIENT_DEPS:-${SCRIPT_DIR}/env/robotwin_client_deps}
TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
SEED_LIST_ROOT=${SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}
SEED=${SEED:-0}
GPU_ID=${GPU_ID:-0}
PI05_ISOLATE_GPU=${PI05_ISOLATE_GPU:-1}
if [[ "${PI05_ISOLATE_GPU}" != "0" && "${PI05_ISOLATE_GPU}" != "false" && "${PI05_ISOLATE_GPU}" != "False" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    fi
    PI05_DEVICE=${PI05_DEVICE:-cuda:0}
    ROBOTWIN_SAPIEN_RENDER_DEVICE=${ROBOTWIN_SAPIEN_RENDER_DEVICE:-cuda:0}
else
    PI05_DEVICE=${PI05_DEVICE:-cuda:${GPU_ID}}
    ROBOTWIN_SAPIEN_RENDER_DEVICE=${ROBOTWIN_SAPIEN_RENDER_DEVICE:-}
fi
PI05_HOST=${PI05_HOST:-127.0.0.1}
PI05_PORT=${PI05_PORT:-$((5702 + GPU_ID))}
PI05_NUM_STEPS=${PI05_NUM_STEPS:-5}
PI05_REQUEST_TIMEOUT=${PI05_REQUEST_TIMEOUT:-180}
PI05_ACTION_SEMANTICS=${PI05_ACTION_SEMANTICS:-absolute}
OPENPI_TORCH_COMPILE=${OPENPI_TORCH_COMPILE:-0}
OPENPI_TORCH_COMPILE_MODE=${OPENPI_TORCH_COMPILE_MODE:-max-autotune}
ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR=${ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR:-rt}
ROBOTWIN_SAPIEN_RT_SAMPLES=${ROBOTWIN_SAPIEN_RT_SAMPLES:-32}
ROBOTWIN_SAPIEN_RT_PATH_DEPTH=${ROBOTWIN_SAPIEN_RT_PATH_DEPTH:-8}
ROBOTWIN_SAPIEN_RT_DENOISER=${ROBOTWIN_SAPIEN_RT_DENOISER:-none}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-False}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
DRY_RUN=${DRY_RUN:-0}
KEEP_SERVER=${KEEP_SERVER:-0}
PI05_SERVER_MANAGED=${PI05_SERVER_MANAGED:-1}
case "${MODE}" in
    smoke) EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}; TASK_LIMIT=${TASK_LIMIT:-1}; TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-300} ;;
    formal) EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}; TASK_LIMIT=${TASK_LIMIT:-0}; TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-1800} ;;
    custom) EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}; TASK_LIMIT=${TASK_LIMIT:-1}; TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-900} ;;
    *) echo "Invalid MODE=${MODE}; use smoke, formal, or custom" >&2; exit 2 ;;
esac
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${MODE}_$$}
RUN_ROOT="${SCRIPT_DIR}/runs/eval/${RUN_ID}"
TASK_LOG_ROOT="${RUN_ROOT}/tasks"
SERVER_LOG_ROOT="${RUN_ROOT}/servers"
EVAL_RESULT_ROOT="${RUN_ROOT}/eval_result"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
MANIFEST_FILE="${RUN_ROOT}/manifest.yaml"
REPORT_FILE="${RUN_ROOT}/report.md"
RESOLVED_JSON="${RUN_ROOT}/resolved_checkpoint.json"
mkdir -p "${TASK_LOG_ROOT}" "${SERVER_LOG_ROOT}" "${EVAL_RESULT_ROOT}"
[[ -f "${WHITELIST}" ]] || { echo "WHITELIST does not exist: ${WHITELIST}" >&2; exit 1; }
if [[ -n "${ROBOTWIN_EVAL_SEED_LIST_PATH:-}" && ! -e "${ROBOTWIN_EVAL_SEED_LIST_PATH}" ]]; then
    echo "ROBOTWIN_EVAL_SEED_LIST_PATH does not exist: ${ROBOTWIN_EVAL_SEED_LIST_PATH}" >&2
    exit 1
fi
SERVER_PID=""
kill_tree() {
    local pid="$1"
    local child
    [[ -n "${pid}" ]] || return 0
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        kill_tree "${child}"
    done
    kill "${pid}" >/dev/null 2>&1 || true
}
cleanup() {
    if [[ -n "${SERVER_PID}" && "${KEEP_SERVER}" != "1" ]]; then
        kill_tree "${SERVER_PID}"
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM
[[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}" >&2; exit 1; }
if [[ "${DRY_RUN}" != "1" && "${PI05_SERVER_MANAGED}" == "1" && ! -x "${PI05_PYTHON}" ]]; then echo "PI05_PYTHON is not executable: ${PI05_PYTHON}" >&2; echo "Run policy/PI05/check.sh or set PI05_PYTHON." >&2; exit 1; fi
"${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${PI05_CKPT_KEY}" "${RESOLVED_JSON}" <<'PYMAP'
import json, sys, yaml
from pathlib import Path
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
record = dict((payload.get('checkpoints') or {})[sys.argv[2]])
for rel in record.get('required_files', []):
    path = Path(record['checkpoint_dir']) / rel
    if not path.exists(): raise SystemExit(f'missing checkpoint file: {path}')
Path(sys.argv[3]).write_text(json.dumps(record), encoding='utf-8')
PYMAP
read_field() { "${ROBOTWIN_PYTHON}" - "${RESOLVED_JSON}" "$1" <<'PYREAD'
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(record.get(sys.argv[2], ''))
PYREAD
}
CHECKPOINT_DIR="$(read_field checkpoint_dir)"
OPENPI_CONFIG_NAME="$(read_field openpi_config_name)"
CHECKPOINT_STEP="$(read_field step)"
CHECKPOINT_STAGE="$(read_field stage)"
mapfile -t ALL_TASKS < <("${ROBOTWIN_PYTHON}" - "${WHITELIST}" <<'PYTASK'
import sys, yaml
from pathlib import Path
items = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
for item in items: print(item)
PYTASK
)
if (( TASK_LIMIT > 0 && TASK_LIMIT < ${#ALL_TASKS[@]} )); then TASKS=("${ALL_TASKS[@]:0:${TASK_LIMIT}}"); else TASKS=("${ALL_TASKS[@]}"); fi
if (( ${#TASKS[@]} == 0 )); then
    echo "No tasks resolved from WHITELIST=${WHITELIST}" >&2
    exit 1
fi
cat > "${MANIFEST_FILE}" <<EOF
run_id: ${RUN_ID}
mode: ${MODE}
baseline: PI05
checkpoint_key: ${PI05_CKPT_KEY}
checkpoint_dir: ${CHECKPOINT_DIR}
checkpoint_step: ${CHECKPOINT_STEP}
checkpoint_stage: ${CHECKPOINT_STAGE}
openpi_config_name: ${OPENPI_CONFIG_NAME}
robotwin_root: ${ROBOTWIN_ROOT}
pi05_rlinf_root: ${PI05_RLINF_ROOT}
task_config: ${TASK_CONFIG}
whitelist: ${WHITELIST}
seed_list_root: ${SEED_LIST_ROOT}
eval_test_num: ${EVAL_TEST_NUM}
task_limit: ${TASK_LIMIT}
gpu_id: ${GPU_ID}
pi05_isolate_gpu: ${PI05_ISOLATE_GPU}
cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}
robotwin_sapien_render_device: ${ROBOTWIN_SAPIEN_RENDER_DEVICE:-}
pi05_host: ${PI05_HOST}
pi05_port: ${PI05_PORT}
pi05_device: ${PI05_DEVICE}
pi05_num_steps: ${PI05_NUM_STEPS}
pi05_request_timeout: ${PI05_REQUEST_TIMEOUT}
pi05_action_semantics: ${PI05_ACTION_SEMANTICS}
openpi_torch_compile: ${OPENPI_TORCH_COMPILE}
openpi_torch_compile_mode: ${OPENPI_TORCH_COMPILE_MODE}
task_timeout_seconds: ${TASK_TIMEOUT_SECONDS}
robotwin_sapien_camera_shader_dir: ${ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR}
robotwin_sapien_rt_samples: ${ROBOTWIN_SAPIEN_RT_SAMPLES}
robotwin_sapien_rt_path_depth: ${ROBOTWIN_SAPIEN_RT_PATH_DEPTH}
robotwin_sapien_rt_denoiser: ${ROBOTWIN_SAPIEN_RT_DENOISER}
robotwin_eval_video_log: ${ROBOTWIN_EVAL_VIDEO_LOG}
robotwin_python: ${ROBOTWIN_PYTHON}
pi05_python: ${PI05_PYTHON}
openpi_client_src: ${OPENPI_CLIENT_SRC}
robotwin_client_deps: ${ROBOTWIN_CLIENT_DEPS}
run_root: ${RUN_ROOT}
EOF
printf 'idx\ttask\tgpu\tport\tstatus\tstart_time\tend_time\tseconds\tlog_path\tsuccess_rate\tresult_path\n' > "${SUMMARY_FILE}"
write_report() {
    local completed failed
    completed="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    failed="$(awk -F '\t' 'NR > 1 && $5 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    {
        printf '# PI05 Eval Report\n\n'
        printf -- '- Run: `%s`\n- Mode: `%s`\n- Checkpoint: `%s` (`%s`, step `%s`)\n' "${RUN_ID}" "${MODE}" "${PI05_CKPT_KEY}" "${CHECKPOINT_STAGE}" "${CHECKPOINT_STEP}"
        printf -- '- Checkpoint dir: `%s`\n- Task config: `%s`\n- Episodes per task: `%s`\n' "${CHECKPOINT_DIR}" "${TASK_CONFIG}" "${EVAL_TEST_NUM}"
        printf -- '- Server: `%s:%s` on `%s`\n- Run root: `%s`\n- Progress: `%s/%s`, failed: `%s`\n\n' "${PI05_HOST}" "${PI05_PORT}" "${PI05_DEVICE}" "${RUN_ROOT}" "${completed}" "${#TASKS[@]}" "${failed}"
        printf '| # | Task | Status | Success Rate | Seconds | Log | Result |\n|---:|---|---:|---:|---:|---|---|\n'
        tail -n +2 "${SUMMARY_FILE}" | while IFS=$'\t' read -r idx task gpu port status start end seconds log_path success_rate result_path; do
            [[ "${result_path}" == "-" ]] && result_link="-" || result_link="[result](<${result_path}>)"
            printf '| %s | %s | %s | %s | %s | [log](<%s>) | %s |\n' "$((idx + 1))" "${task}" "${status}" "${success_rate}" "${seconds}" "${log_path}" "${result_link}"
        done
    } > "${REPORT_FILE}"
}
write_report
if [[ "${DRY_RUN}" == "1" ]]; then echo "DRY_RUN=1; wrote manifest: ${MANIFEST_FILE}"; exit 0; fi
if [[ "${PI05_SERVER_MANAGED}" == "1" ]]; then
    SERVER_LOG="${SERVER_LOG_ROOT}/pi05_gpu${GPU_ID}_port${PI05_PORT}.log"
    echo "Starting PI05 server; log=${SERVER_LOG}"
    (
        cd "${PI05_RLINF_ROOT}"
        export PYTHONPATH="${PI05_RLINF_ROOT}:${PYTHONPATH:-}"
        export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/PI05_eval/PI05/cache/openpi}"
        export HF_HOME="${HF_HOME:-/root/autodl-tmp/PI05_eval/PI05/cache/huggingface}"
        export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
        export OPENPI_TORCH_COMPILE OPENPI_TORCH_COMPILE_MODE
        mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"
        exec "${PI05_PYTHON}" -m toolkits.standalone_eval_scripts.openpi.serve_policy \
            --config-name "${OPENPI_CONFIG_NAME}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --host "${PI05_HOST}" \
            --port "${PI05_PORT}" \
            --device "${PI05_DEVICE}" \
            --num-steps "${PI05_NUM_STEPS}"
    ) > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
fi
"${ROBOTWIN_PYTHON}" - "${PI05_HOST}" "${PI05_PORT}" "${SERVER_READY_TIMEOUT}" "${SERVER_PID:-}" "${SERVER_LOG:-}" <<'PYREADY'
import os
import sys
import time
from pathlib import Path

import requests

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
server_pid = int(sys.argv[4]) if sys.argv[4] else None
server_log = Path(sys.argv[5]) if sys.argv[5] else None
url = f'http://{host}:{port}/healthz'
start = time.time()
last = None

def server_is_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            fields = stat_path.read_text().split()
            if len(fields) >= 3 and fields[2] == "Z":
                return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

while time.time() - start < timeout:
    if server_pid is not None and not server_is_alive(server_pid):
        detail = ""
        if server_log and server_log.exists():
            lines = server_log.read_text(errors="replace").splitlines()[-80:]
            detail_lines = ["", "--- PI05 server log tail ---", *lines]
            detail = "\n".join(detail_lines)
        raise SystemExit(f'PI05 server exited before readiness check passed: pid={server_pid}{detail}')
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            print(f'server ready: {url}')
            raise SystemExit(0)
        last = f'status {r.status_code}'
    except Exception as exc:
        last = f'{type(exc).__name__}: {exc}'
    time.sleep(2)
raise SystemExit(f'server not ready after {timeout}s: {last}')
PYREADY
cd "${ROBOTWIN_ROOT}"
export OPENPI_CLIENT_SRC PI05_HOST PI05_PORT PI05_ACTION_SEMANTICS
export ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}"
export ROBOTWIN_SAPIEN_RENDER_DEVICE
export ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR ROBOTWIN_SAPIEN_RT_SAMPLES ROBOTWIN_SAPIEN_RT_PATH_DEPTH ROBOTWIN_SAPIEN_RT_DENOISER
export PYTHONPATH="${ROBOTWIN_CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}"
for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_slug="$(printf '%03d_%s' "${idx}" "${task}")"
    task_log_dir="${TASK_LOG_ROOT}/${task_slug}"
    mkdir -p "${task_log_dir}"
    log_path="${task_log_dir}/stdout.log"
    start_time="$(date -Is)"; start_sec="$(date +%s)"; status=0
    echo "[PI05 eval] task=${task} log=${log_path}"
    cmd=("${ROBOTWIN_PYTHON}" script/eval_policy.py --config policy/PI05/deploy_policy.yml --overrides --task_name "${task}" --task_config "${TASK_CONFIG}" --ckpt_setting "${PI05_CKPT_KEY}" --seed "${SEED}" --test_num "${EVAL_TEST_NUM}" --policy_name PI05 --host "${PI05_HOST}" --port "${PI05_PORT}" --request_timeout "${PI05_REQUEST_TIMEOUT}" --openpi_client_src "${OPENPI_CLIENT_SRC}" --action_semantics "${PI05_ACTION_SEMANTICS}" --eval_video_log "${ROBOTWIN_EVAL_VIDEO_LOG}")
    if (( TASK_TIMEOUT_SECONDS > 0 )); then timeout --kill-after=30s "${TASK_TIMEOUT_SECONDS}" "${cmd[@]}" > "${log_path}" 2>&1 || status=$?; else "${cmd[@]}" > "${log_path}" 2>&1 || status=$?; fi
    end_time="$(date -Is)"; end_sec="$(date +%s)"; seconds=$((end_sec - start_sec))
    result_path="$(awk '/Data has been saved to / {sub(/^.*Data has been saved to /, ""); p=$0} END {print p}' "${log_path}")"
    success_rate="-"
    if [[ -n "${result_path}" && -f "${result_path}" ]]; then success_rate="$(awk 'NF && $1 ~ /^[-+0-9.]+$/ {v=$1} END {if (v == "") print "-"; else print v}' "${result_path}")"; else result_path="-"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${idx}" "${task}" "${GPU_ID}" "${PI05_PORT}" "${status}" "${start_time}" "${end_time}" "${seconds}" "${log_path}" "${success_rate}" "${result_path}" >> "${SUMMARY_FILE}"
    write_report
    if [[ "${status}" != "0" && "${CONTINUE_ON_ERROR}" != "1" ]]; then echo "Task failed and CONTINUE_ON_ERROR=0: ${task}" >&2; exit "${status}"; fi
done
echo "Run complete: ${RUN_ROOT}"
echo "Report: ${REPORT_FILE}"
