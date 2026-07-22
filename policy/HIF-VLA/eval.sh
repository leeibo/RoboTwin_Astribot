#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE=${MODE:-smoke}
HIFVLA_CKPT_KEY=${HIFVLA_CKPT_KEY:-hifvla_astribot35_150k}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy_policy.yml}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
HIFVLA_PYTHON=${HIFVLA_PYTHON:-/root/autodl-tmp/HIF-VLA_eval/HIF-VLA/env/hifvla/bin/python}
TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
EVAL_SEED_LIST_ROOT=${EVAL_SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}
TASK_NAME=${TASK_NAME:-}
SEED=${SEED:-0}
GPU_ID=${GPU_ID:-0}
HIFVLA_ISOLATE_GPU=${HIFVLA_ISOLATE_GPU:-1}
HIFVLA_HOST=${HIFVLA_HOST:-127.0.0.1}
HIFVLA_PORT=${HIFVLA_PORT:-$((5802 + GPU_ID))}
HIFVLA_REQUEST_TIMEOUT=${HIFVLA_REQUEST_TIMEOUT:-300}
HIFVLA_ACTION_HORIZON=${HIFVLA_ACTION_HORIZON:-8}
HIFVLA_MAX_ACTIONS_PER_CALL=${HIFVLA_MAX_ACTIONS_PER_CALL:-8}
HIFVLA_HISTORY_LENGTH=${HIFVLA_HISTORY_LENGTH:-8}
HIFVLA_ACTION_SEMANTICS=${HIFVLA_ACTION_SEMANTICS:-absolute}
HIFVLA_LOG_REQUEST_DEBUG=${HIFVLA_LOG_REQUEST_DEBUG:-True}
HIFVLA_LOG_CHUNK_TIMING=${HIFVLA_LOG_CHUNK_TIMING:-True}
HIFVLA_SERVER_MANAGED=${HIFVLA_SERVER_MANAGED:-1}
HIFVLA_MOCK_SERVER=${HIFVLA_MOCK_SERVER:-0}
HIFVLA_SERVER_PYTHON=${HIFVLA_SERVER_PYTHON:-}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
DRY_RUN=${DRY_RUN:-0}
KEEP_SERVER=${KEEP_SERVER:-0}

ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR=${ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR:-rt}
ROBOTWIN_SAPIEN_RT_SAMPLES=${ROBOTWIN_SAPIEN_RT_SAMPLES:-32}
ROBOTWIN_SAPIEN_RT_PATH_DEPTH=${ROBOTWIN_SAPIEN_RT_PATH_DEPTH:-8}
ROBOTWIN_SAPIEN_RT_DENOISER=${ROBOTWIN_SAPIEN_RT_DENOISER:-none}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-False}

case "${MODE}" in
    smoke)
        EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}
        TASK_LIMIT=${TASK_LIMIT:-1}
        TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-600}
        ;;
    formal)
        EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}
        TASK_LIMIT=${TASK_LIMIT:-0}
        TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
        ;;
    custom)
        EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}
        TASK_LIMIT=${TASK_LIMIT:-1}
        TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
        ;;
    *)
        echo "MODE must be smoke, formal, or custom: ${MODE}" >&2
        exit 2
        ;;
esac

for value_name in GPU_ID HIFVLA_PORT HIFVLA_ACTION_HORIZON HIFVLA_MAX_ACTIONS_PER_CALL     HIFVLA_HISTORY_LENGTH SERVER_READY_TIMEOUT EVAL_TEST_NUM; do
    value=${!value_name}
    [[ "${value}" =~ ^[1-9][0-9]*$ || ("${value_name}" == "GPU_ID" && "${value}" == "0") ]] || {
        echo "${value_name} must be a positive integer (GPU_ID may be zero): ${value}" >&2
        exit 2
    }
done
[[ "${TASK_LIMIT}" =~ ^[0-9]+$ ]] || { echo "TASK_LIMIT must be non-negative" >&2; exit 2; }
[[ "${TASK_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || { echo "TASK_TIMEOUT_SECONDS must be non-negative" >&2; exit 2; }
[[ "${HIFVLA_ACTION_HORIZON}" == "8" ]] || { echo "HIFVLA_ACTION_HORIZON must be 8" >&2; exit 2; }
[[ "${HIFVLA_HISTORY_LENGTH}" == "8" ]] || { echo "HIFVLA_HISTORY_LENGTH must be 8" >&2; exit 2; }
(( HIFVLA_MAX_ACTIONS_PER_CALL <= HIFVLA_ACTION_HORIZON )) || {
    echo "HIFVLA_MAX_ACTIONS_PER_CALL cannot exceed HIFVLA_ACTION_HORIZON" >&2
    exit 2
}
[[ "${HIFVLA_ACTION_SEMANTICS}" == "absolute" || "${HIFVLA_ACTION_SEMANTICS}" == "delta_to_abs" ]] || {
    echo "HIFVLA_ACTION_SEMANTICS must be absolute or delta_to_abs" >&2
    exit 2
}

if [[ "${HIFVLA_ISOLATE_GPU}" != "0" && "${HIFVLA_ISOLATE_GPU}" != "false" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_ID}}
    HIFVLA_DEVICE=${HIFVLA_DEVICE:-cuda:0}
    ROBOTWIN_SAPIEN_RENDER_DEVICE=${ROBOTWIN_SAPIEN_RENDER_DEVICE:-cuda:0}
else
    HIFVLA_DEVICE=${HIFVLA_DEVICE:-cuda:${GPU_ID}}
    ROBOTWIN_SAPIEN_RENDER_DEVICE=${ROBOTWIN_SAPIEN_RENDER_DEVICE:-}
fi

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

for path in "${CKPT_MAPPING}" "${DEPLOY_CONFIG}" "${WHITELIST}"     "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"; do
    [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 1; }
done
[[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}" >&2; exit 1; }

mapfile -t CHECKPOINT_FIELDS < <("${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${HIFVLA_CKPT_KEY}" "${RESOLVED_JSON}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

mapping_path, key, output_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
if key in (payload.get("unavailable") or {}):
    raise SystemExit(f"checkpoint unavailable: {key}: {payload['unavailable'][key]}")
record = dict((payload.get("checkpoints") or {})[key])
checkpoint_dir = Path(record["checkpoint_dir"])
base_checkpoint = Path(record["base_checkpoint"])
hifvla_root = Path(record["hifvla_root"])
for path in (checkpoint_dir, base_checkpoint, hifvla_root):
    if not path.is_dir():
        raise SystemExit(f"required directory does not exist: {path}")
for relative in record.get("required_files", []):
    path = checkpoint_dir / relative
    if not path.is_file():
        raise SystemExit(f"missing checkpoint file: {path}")
for relative in record.get("required_base_files", []):
    path = base_checkpoint / relative
    if not path.is_file():
        raise SystemExit(f"missing base model file: {path}")
output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
for field in ("checkpoint_dir", "base_checkpoint", "hifvla_root", "unnorm_key", "step", "stage"):
    print(record.get(field, ""))
PY
)
CHECKPOINT_DIR=${CHECKPOINT_FIELDS[0]}
BASE_CHECKPOINT=${CHECKPOINT_FIELDS[1]}
HIFVLA_ROOT=${CHECKPOINT_FIELDS[2]}
UNNORM_KEY=${CHECKPOINT_FIELDS[3]}
CHECKPOINT_STEP=${CHECKPOINT_FIELDS[4]}
CHECKPOINT_STAGE=${CHECKPOINT_FIELDS[5]}

if [[ -n "${TASK_NAME}" ]]; then
    TASKS=("${TASK_NAME}")
else
    mapfile -t ALL_TASKS < <("${ROBOTWIN_PYTHON}" - "${WHITELIST}" <<'PY'
import sys
from pathlib import Path
import yaml
items = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(items, list):
    raise SystemExit("whitelist must contain a list")
for item in items:
    if not isinstance(item, str) or not item:
        raise SystemExit(f"invalid task: {item!r}")
    print(item)
PY
)
    if (( TASK_LIMIT > 0 && TASK_LIMIT < ${#ALL_TASKS[@]} )); then
        TASKS=("${ALL_TASKS[@]:0:TASK_LIMIT}")
    else
        TASKS=("${ALL_TASKS[@]}")
    fi
fi
(( ${#TASKS[@]} > 0 )) || { echo "No tasks selected" >&2; exit 1; }

"${ROBOTWIN_PYTHON}" - "${EVAL_SEED_LIST_ROOT}" "${TASK_CONFIG}" "${EVAL_TEST_NUM}" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path
root, config, need = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
errors = []
for task in sys.argv[4:]:
    path = root / config / f"{task}.json"
    if not path.is_file():
        errors.append(f"missing: {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload.get("seeds", [])) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or len(entries) < need:
        errors.append(f"needs {need} seeds: {path}")
if errors:
    raise SystemExit("\n".join(errors))
PY

cat > "${MANIFEST_FILE}" <<EOF
run_id: ${RUN_ID}
mode: ${MODE}
baseline: HIF-VLA
checkpoint_key: ${HIFVLA_CKPT_KEY}
checkpoint_dir: ${CHECKPOINT_DIR}
base_checkpoint: ${BASE_CHECKPOINT}
checkpoint_step: ${CHECKPOINT_STEP}
checkpoint_stage: ${CHECKPOINT_STAGE}
hifvla_root: ${HIFVLA_ROOT}
unnorm_key: ${UNNORM_KEY}
task_config: ${TASK_CONFIG}
whitelist: ${WHITELIST}
eval_seed_list_root: ${EVAL_SEED_LIST_ROOT}
eval_test_num: ${EVAL_TEST_NUM}
task_limit: ${TASK_LIMIT}
gpu_id: ${GPU_ID}
cuda_visible_devices: "${CUDA_VISIBLE_DEVICES:-}"
hifvla_host: ${HIFVLA_HOST}
hifvla_port: ${HIFVLA_PORT}
hifvla_device: ${HIFVLA_DEVICE}
hifvla_action_horizon: ${HIFVLA_ACTION_HORIZON}
hifvla_max_actions_per_call: ${HIFVLA_MAX_ACTIONS_PER_CALL}
hifvla_history_length: ${HIFVLA_HISTORY_LENGTH}
hifvla_action_semantics: ${HIFVLA_ACTION_SEMANTICS}
hifvla_request_timeout: ${HIFVLA_REQUEST_TIMEOUT}
hifvla_mock_server: ${HIFVLA_MOCK_SERVER}
task_timeout_seconds: ${TASK_TIMEOUT_SECONDS}
robotwin_python: ${ROBOTWIN_PYTHON}
hifvla_python: ${HIFVLA_PYTHON}
run_root: ${RUN_ROOT}
EOF

printf 'idx\ttask\tgpu\tport\tstatus\tstart_time\tend_time\tseconds\tlog_path\tsuccess_rate\tresult_path\n' > "${SUMMARY_FILE}"

write_report() {
    local completed failed
    completed="$(awk 'NR > 1 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    failed="$(awk -F '\t' 'NR > 1 && $5 != 0 {n++} END {print n + 0}' "${SUMMARY_FILE}")"
    {
        printf '# HIF-VLA Eval Report\n\n'
        printf -- '- Run: `%s`\n- Mode: `%s`\n' "${RUN_ID}" "${MODE}"
        printf -- '- Checkpoint: `%s` (`%s`, step `%s`)\n' "${HIFVLA_CKPT_KEY}" "${CHECKPOINT_STAGE}" "${CHECKPOINT_STEP}"
        printf -- '- Checkpoint dir: `%s`\n- Base: `%s`\n' "${CHECKPOINT_DIR}" "${BASE_CHECKPOINT}"
        printf -- '- Task config: `%s`\n- Episodes per task: `%s`\n' "${TASK_CONFIG}" "${EVAL_TEST_NUM}"
        printf -- '- Server: `%s:%s` on `%s`\n' "${HIFVLA_HOST}" "${HIFVLA_PORT}" "${HIFVLA_DEVICE}"
        printf -- '- Run root: `%s`\n- Progress: `%s/%s`, failed: `%s`\n\n' "${RUN_ROOT}" "${completed}" "${#TASKS[@]}" "${failed}"
        printf '| # | Task | Status | Success Rate | Seconds | Log | Result |\n'
        printf '|---:|---|---:|---:|---:|---|---|\n'
        tail -n +2 "${SUMMARY_FILE}" | while IFS=$'\t' read -r idx task gpu port status start end seconds log_path rate result_path; do
            [[ "${result_path}" == "-" ]] && result_link="-" || result_link="[result](<${result_path}>)"
            printf '| %s | %s | %s | %s | %s | [log](<%s>) | %s |\n'                 "$((idx + 1))" "${task}" "${status}" "${rate}" "${seconds}" "${log_path}" "${result_link}"
        done
    } > "${REPORT_FILE}"
}
write_report

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] HIF-VLA validation passed; tasks=${#TASKS[@]} run_root=${RUN_ROOT}"
    exit 0
fi

if [[ "${HIFVLA_SERVER_MANAGED}" == "1" ]]; then
    if [[ -z "${HIFVLA_SERVER_PYTHON}" ]]; then
        if [[ "${HIFVLA_MOCK_SERVER}" == "1" ]]; then
            HIFVLA_SERVER_PYTHON="${ROBOTWIN_PYTHON}"
        else
            HIFVLA_SERVER_PYTHON="${HIFVLA_PYTHON}"
        fi
    fi
    [[ -x "${HIFVLA_SERVER_PYTHON}" ]] || {
        echo "HIF-VLA server Python is not executable: ${HIFVLA_SERVER_PYTHON}" >&2
        echo "Run policy/HIF-VLA/install_env.sh or set HIFVLA_SERVER_PYTHON." >&2
        exit 1
    }
fi

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" && "${KEEP_SERVER}" != "1" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${HIFVLA_SERVER_MANAGED}" == "1" ]]; then
    SERVER_LOG="${SERVER_LOG_ROOT}/hifvla_gpu${GPU_ID}_port${HIFVLA_PORT}.log"
    server_cmd=(
        "${HIFVLA_SERVER_PYTHON}" "${SCRIPT_DIR}/serve_policy.py"
        --platform astribot
        --base-checkpoint "${BASE_CHECKPOINT}"
        --checkpoint-dir "${CHECKPOINT_DIR}"
        --hifvla-root "${HIFVLA_ROOT}"
        --unnorm-key "${UNNORM_KEY}"
        --device "${HIFVLA_DEVICE}"
        --history-length "${HIFVLA_HISTORY_LENGTH}"
        --action-horizon "${HIFVLA_ACTION_HORIZON}"
        --action-dim 18
        --host "${HIFVLA_HOST}"
        --port "${HIFVLA_PORT}"
    )
    [[ "${HIFVLA_MOCK_SERVER}" == "1" ]] && server_cmd+=(--mock)
    echo "Starting HIF-VLA server; log=${SERVER_LOG}"
    setsid env         PYTHONPATH="${HIFVLA_ROOT}:${PYTHONPATH:-}"         PYTHONUNBUFFERED=1         "${server_cmd[@]}" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
fi

"${ROBOTWIN_PYTHON}" - "${HIFVLA_HOST}" "${HIFVLA_PORT}" "${SERVER_READY_TIMEOUT}" "${SERVER_PID:-}" "${SERVER_LOG:-}" <<'PY'
import os
import sys
import time
from pathlib import Path
import requests

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
pid = int(sys.argv[4]) if sys.argv[4] else None
log = Path(sys.argv[5]) if sys.argv[5] else None
url = f"http://{host}:{port}/healthz"
start, last = time.time(), None
while time.time() - start < timeout:
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:]) if log and log.exists() else ""
            raise SystemExit(f"server exited before readiness\n{tail}")
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200 and response.json().get("ready") is True:
            print(f"server ready: {url}")
            raise SystemExit(0)
        last = f"HTTP {response.status_code}: {response.text[:500]}"
    except requests.RequestException as exc:
        last = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
raise SystemExit(f"server not ready after {timeout}s: {last}")
PY

cd "${ROBOTWIN_ROOT}"
export HIFVLA_HOST HIFVLA_PORT HIFVLA_ACTION_SEMANTICS
export ROBOTWIN_USE_EVAL_SEED_LIST=1
export ROBOTWIN_EVAL_SEED_LIST_PATH="${EVAL_SEED_LIST_ROOT}"
export ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}"
export ROBOTWIN_SAPIEN_RENDER_DEVICE
export ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR ROBOTWIN_SAPIEN_RT_SAMPLES
export ROBOTWIN_SAPIEN_RT_PATH_DEPTH ROBOTWIN_SAPIEN_RT_DENOISER

for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_dir="${TASK_LOG_ROOT}/$(printf '%03d_%s' "$((idx + 1))" "${task}")"
    mkdir -p "${task_dir}"
    log_path="${task_dir}/stdout.log"
    start_time="$(date -Is)"
    start_seconds="$(date +%s)"
    status=0
    cmd=(
        "${ROBOTWIN_PYTHON}" script/eval_policy.py
        --config policy/HIF-VLA/deploy_policy.yml
        --overrides
        --task_name "${task}"
        --task_config "${TASK_CONFIG}"
        --ckpt_setting "${HIFVLA_CKPT_KEY}"
        --seed "${SEED}"
        --test_num "${EVAL_TEST_NUM}"
        --policy_name HIF-VLA
        --host "${HIFVLA_HOST}"
        --port "${HIFVLA_PORT}"
        --request_timeout "${HIFVLA_REQUEST_TIMEOUT}"
        --action_horizon "${HIFVLA_ACTION_HORIZON}"
        --max_actions_per_call "${HIFVLA_MAX_ACTIONS_PER_CALL}"
        --action_semantics "${HIFVLA_ACTION_SEMANTICS}"
        --history_length "${HIFVLA_HISTORY_LENGTH}"
        --log_request_debug "${HIFVLA_LOG_REQUEST_DEBUG}"
        --log_chunk_timing "${HIFVLA_LOG_CHUNK_TIMING}"
        --temp_root "${task_dir}/hifvla_debug"
        --eval_video_log "${ROBOTWIN_EVAL_VIDEO_LOG}"
    )
    echo "[HIF-VLA eval] task=${task} log=${log_path}"
    if (( TASK_TIMEOUT_SECONDS > 0 )); then
        timeout --signal=TERM --kill-after=30 "${TASK_TIMEOUT_SECONDS}" "${cmd[@]}" > "${log_path}" 2>&1 || status=$?
    else
        "${cmd[@]}" > "${log_path}" 2>&1 || status=$?
    fi
    end_time="$(date -Is)"
    elapsed=$(( $(date +%s) - start_seconds ))
    result_path="$(sed -n 's/^Data has been saved to //p' "${log_path}" | tail -n 1)"
    success_rate=-
    if [[ -n "${result_path}" && -f "${result_path}" ]]; then
        success_rate="$(awk 'NF && $1 ~ /^[-+0-9.]+$/ {v=$1} END {print v == "" ? "-" : v}' "${result_path}")"
    else
        result_path=-
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'         "${idx}" "${task}" "${GPU_ID}" "${HIFVLA_PORT}" "${status}"         "${start_time}" "${end_time}" "${elapsed}" "${log_path}"         "${success_rate}" "${result_path}" >> "${SUMMARY_FILE}"
    write_report
    if [[ "${status}" != "0" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit "${status}"
    fi
done

echo "Run complete: ${RUN_ROOT}"
echo "Report: ${REPORT_FILE}"
