#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE=${MODE:-smoke}
ENVIRONMENT_TYPE=${ENVIRONMENT_TYPE:-clean}
MEMER_CKPT_KEY=${MEMER_CKPT_KEY:-memer_astribot_step_4500_18000}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy_policy.yml}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
HIGH_PYTHON=${HIGH_PYTHON:-/root/autodl-tmp/MEMER_eval/MemER/env/qwen3vl/bin/python}
HIGH_PROCESSOR_DIR=${HIGH_PROCESSOR_DIR:-}
LOW_PYTHON=${LOW_PYTHON:-/root/autodl-tmp/MEMER_eval/MemER/env/rlinf-pi05/bin/python}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-/root/autodl-tmp/MEMER_eval/MemER/source/low_level/openpi/packages/openpi-client/src}
ROBOTWIN_CLIENT_DEPS=${ROBOTWIN_CLIENT_DEPS:-${SCRIPT_DIR}/env/robotwin_client_deps}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
SEED_LIST_ROOT=${SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}
TASK_NAME=${TASK_NAME:-}
TASK_CONFIG=${TASK_CONFIG:-}
SEED=${SEED:-0}
HIGH_GPU=${HIGH_GPU:-}
LOW_GPU=${LOW_GPU:-}
HIGH_HOST=${HIGH_HOST:-127.0.0.1}
HIGH_PORT=${HIGH_PORT:-}
LOW_HOST=${LOW_HOST:-127.0.0.1}
LOW_PORT=${LOW_PORT:-}
HIGH_DEVICE=${HIGH_DEVICE:-cuda:0}
LOW_DEVICE=${LOW_DEVICE:-cuda:0}
WORKER_ID=${WORKER_ID:-worker-0}
HIGH_SERVER_MANAGED=${HIGH_SERVER_MANAGED:-1}
LOW_SERVER_MANAGED=${LOW_SERVER_MANAGED:-1}
HIGH_MOCK=${HIGH_MOCK:-0}
LOW_MOCK=${LOW_MOCK:-0}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-300}
ACTION_HORIZON=${ACTION_HORIZON:-50}
LOW_LEVEL_EXECUTION_HORIZON=${LOW_LEVEL_EXECUTION_HORIZON:-5}
HIGH_LEVEL_REPLAN_INTERVAL=${HIGH_LEVEL_REPLAN_INTERVAL:-5}
RECENT_FRAMES=${RECENT_FRAMES:-8}
MEMORY_FRAMES=${MEMORY_FRAMES:-8}
RECENT_FRAME_INTERVAL=${RECENT_FRAME_INTERVAL:-5}
KEYFRAME_MERGE_DISTANCE=${KEYFRAME_MERGE_DISTANCE:-5}
HIGH_MAX_NEW_TOKENS=${HIGH_MAX_NEW_TOKENS:-128}
HIGH_MAX_SESSIONS=${HIGH_MAX_SESSIONS:-32}
HIGH_ATTN_IMPLEMENTATION=${HIGH_ATTN_IMPLEMENTATION:-flash_attention_2}
VLM_LOG_IMAGES=${VLM_LOG_IMAGES:-1}
LOW_NUM_STEPS=${LOW_NUM_STEPS:-5}
ACTION_SEMANTICS=${ACTION_SEMANTICS:-absolute}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
DRY_RUN=${DRY_RUN:-0}
KEEP_SERVER=${KEEP_SERVER:-0}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-}
TASK_TIMEOUT_SECONDS_PER_EPISODE=${TASK_TIMEOUT_SECONDS_PER_EPISODE:-360}
TASK_LIMIT=${TASK_LIMIT:-}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-False}
ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR=${ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR:-rt}
ROBOTWIN_SAPIEN_RT_SAMPLES=${ROBOTWIN_SAPIEN_RT_SAMPLES:-32}
ROBOTWIN_SAPIEN_RT_PATH_DEPTH=${ROBOTWIN_SAPIEN_RT_PATH_DEPTH:-8}
ROBOTWIN_SAPIEN_RT_DENOISER=${ROBOTWIN_SAPIEN_RT_DENOISER:-none}

case "${ENVIRONMENT_TYPE}" in
  clean)
    TASK_CONFIG=${TASK_CONFIG:-info_gathering_demo}
    HIGH_GPU=${HIGH_GPU:-0}
    LOW_GPU=${LOW_GPU:-1}
    HIGH_PORT=${HIGH_PORT:-5901}
    LOW_PORT=${LOW_PORT:-5902}
    ;;
  randomized)
    TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
    HIGH_GPU=${HIGH_GPU:-5}
    LOW_GPU=${LOW_GPU:-6}
    HIGH_PORT=${HIGH_PORT:-5951}
    LOW_PORT=${LOW_PORT:-5960}
    ;;
  *) echo "ENVIRONMENT_TYPE must be clean or randomized" >&2; exit 2 ;;
esac
case "${MODE}" in
  smoke)
    EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}
    TASK_LIMIT=${TASK_LIMIT:-1}
    if [[ -z "${TASK_TIMEOUT_SECONDS}" ]]; then
      [[ "${EVAL_TEST_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_TEST_NUM must be positive: ${EVAL_TEST_NUM}" >&2; exit 2; }
      [[ "${TASK_TIMEOUT_SECONDS_PER_EPISODE}" =~ ^[1-9][0-9]*$ ]] || {
        echo "TASK_TIMEOUT_SECONDS_PER_EPISODE must be positive: ${TASK_TIMEOUT_SECONDS_PER_EPISODE}" >&2
        exit 2
      }
      TASK_TIMEOUT_SECONDS=$((EVAL_TEST_NUM * TASK_TIMEOUT_SECONDS_PER_EPISODE))
      (( TASK_TIMEOUT_SECONDS >= 900 )) || TASK_TIMEOUT_SECONDS=900
    fi
    ;;
  formal) EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}; TASK_LIMIT=${TASK_LIMIT:-0}; TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0} ;;
  custom) EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}; TASK_LIMIT=${TASK_LIMIT:-1}; TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0} ;;
  *) echo "MODE must be smoke, formal, or custom" >&2; exit 2 ;;
esac
[[ "${TASK_CONFIG}" == "info_gathering_demo" && "${ENVIRONMENT_TYPE}" == "clean" || \
   "${TASK_CONFIG}" == "info_gathering_randomized" && "${ENVIRONMENT_TYPE}" == "randomized" ]] || {
  echo "TASK_CONFIG=${TASK_CONFIG} does not match ENVIRONMENT_TYPE=${ENVIRONMENT_TYPE}" >&2; exit 2;
}
for name in HIGH_PORT LOW_PORT ACTION_HORIZON LOW_LEVEL_EXECUTION_HORIZON HIGH_LEVEL_REPLAN_INTERVAL \
  RECENT_FRAMES MEMORY_FRAMES RECENT_FRAME_INTERVAL HIGH_MAX_NEW_TOKENS EVAL_TEST_NUM \
  TASK_TIMEOUT_SECONDS_PER_EPISODE; do
  value=${!name}; [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be positive: ${value}" >&2; exit 2; }
done
[[ "${TASK_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || { echo "TASK_TIMEOUT_SECONDS must be non-negative: ${TASK_TIMEOUT_SECONDS}" >&2; exit 2; }
(( LOW_LEVEL_EXECUTION_HORIZON <= ACTION_HORIZON )) || { echo "execution horizon exceeds action horizon" >&2; exit 2; }
[[ "${ACTION_HORIZON}" == 50 ]] || { echo "ACTION_HORIZON must be 50" >&2; exit 2; }
[[ "${VLM_LOG_IMAGES}" == 0 || "${VLM_LOG_IMAGES}" == 1 ]] || { echo "VLM_LOG_IMAGES must be 0 or 1" >&2; exit 2; }

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${ENVIRONMENT_TYPE}_${MODE}_$$}
RUN_ROOT="${SCRIPT_DIR}/runs/eval/${RUN_ID}"
mkdir -p "${RUN_ROOT}/tasks" "${RUN_ROOT}/servers" "${RUN_ROOT}/eval_result"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
MANIFEST_FILE="${RUN_ROOT}/manifest.yaml"
REPORT_FILE="${RUN_ROOT}/report.md"
RESOLVED_JSON="${RUN_ROOT}/resolved_checkpoint.json"

for path in "${CKPT_MAPPING}" "${DEPLOY_CONFIG}" "${WHITELIST}" "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"; do
  [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 1; }
done
[[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}" >&2; exit 1; }

if [[ "${DRY_RUN}" != 1 ]]; then
  "${ROBOTWIN_PYTHON}" - "${HIGH_SERVER_MANAGED}" "${HIGH_HOST}" "${HIGH_PORT}" \
    "${LOW_SERVER_MANAGED}" "${LOW_HOST}" "${LOW_PORT}" <<'PY'
import socket, sys

sockets = []
specs = (
    ("high", sys.argv[1], sys.argv[2], int(sys.argv[3])),
    ("low", sys.argv[4], sys.argv[5], int(sys.argv[6])),
)
try:
    for label, managed, host, port in specs:
        if managed != "1":
            continue
        sock = socket.socket()
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise SystemExit(
                f"{label} managed-server port is unavailable: {host}:{port}: {exc}. "
                "Choose another port or stop the owning evaluation."
            ) from exc
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
PY
fi

mapfile -t FIELDS < <("${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${MEMER_CKPT_KEY}" "${RESOLVED_JSON}" <<'PY'
import json, sys, yaml
from pathlib import Path
mapping, key, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
data = yaml.safe_load(mapping.read_text(encoding="utf-8")) or {}
record = dict((data.get("checkpoints") or {})[key])
for group, root_key in (("required_high_files", "high_checkpoint_dir"), ("required_low_files", "low_checkpoint_dir")):
    root = Path(record[root_key])
    if not root.is_dir(): raise SystemExit(f"missing checkpoint dir: {root}")
    for rel in record.get(group, []):
        if not (root / rel).is_file(): raise SystemExit(f"missing checkpoint file: {root / rel}")
output.write_text(json.dumps(record, indent=2), encoding="utf-8")
for key in ("high_checkpoint_dir", "low_checkpoint_dir", "rlinf_root", "openpi_config_name", "high_step", "low_step"):
    print(record[key])
PY
)
HIGH_CHECKPOINT=${FIELDS[0]}
HIGH_PROCESSOR_DIR=${HIGH_PROCESSOR_DIR:-${HIGH_CHECKPOINT}}
LOW_CHECKPOINT=${FIELDS[1]}
RLINF_ROOT=${FIELDS[2]}
OPENPI_CONFIG_NAME=${FIELDS[3]}
HIGH_STEP=${FIELDS[4]}
LOW_STEP=${FIELDS[5]}

if [[ "${HIGH_SERVER_MANAGED}" == 1 && "${HIGH_MOCK}" != 1 ]]; then
  for name in preprocessor_config.json video_preprocessor_config.json tokenizer_config.json; do
    [[ -f "${HIGH_PROCESSOR_DIR}/${name}" ]] || {
      echo "missing Qwen3-VL processor file: ${HIGH_PROCESSOR_DIR}/${name}" >&2
      echo "Copy the processor files from the high-level training output directory," >&2
      echo "or set HIGH_PROCESSOR_DIR=/path/to/high-level-training-output." >&2
      exit 1
    }
  done
fi

if [[ -n "${TASK_NAME}" ]]; then
  TASKS=("${TASK_NAME}")
else
  mapfile -t TASKS < <("${ROBOTWIN_PYTHON}" - "${WHITELIST}" <<'PY'
import sys, yaml
from pathlib import Path
items = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(items, list): raise SystemExit("whitelist must be a list")
for item in items: print(item)
PY
  )
  if (( TASK_LIMIT > 0 && TASK_LIMIT < ${#TASKS[@]} )); then TASKS=("${TASKS[@]:0:TASK_LIMIT}"); fi
fi
"${ROBOTWIN_PYTHON}" - "${SEED_LIST_ROOT}" "${TASK_CONFIG}" "${EVAL_TEST_NUM}" "${TASKS[@]}" <<'PY'
import json, sys
from pathlib import Path
root, config, need = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for task in sys.argv[4:]:
    path = root / config / f"{task}.json"
    if not path.is_file(): raise SystemExit(f"missing seed list: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("entries", value.get("seeds", [])) if isinstance(value, dict) else value
    if not isinstance(rows, list) or len(rows) < need: raise SystemExit(f"seed list has fewer than {need}: {path}")
PY

cat > "${MANIFEST_FILE}" <<EOF
run_id: ${RUN_ID}
baseline: MemER
mode: ${MODE}
environment_type: ${ENVIRONMENT_TYPE}
task_config: ${TASK_CONFIG}
checkpoint_key: ${MEMER_CKPT_KEY}
high_checkpoint: ${HIGH_CHECKPOINT}
high_processor_dir: ${HIGH_PROCESSOR_DIR}
low_checkpoint: ${LOW_CHECKPOINT}
high_step: ${HIGH_STEP}
low_step: ${LOW_STEP}
high_gpu: ${HIGH_GPU}
low_gpu: ${LOW_GPU}
high_server: ${HIGH_HOST}:${HIGH_PORT}
low_server: ${LOW_HOST}:${LOW_PORT}
worker_id: ${WORKER_ID}
action_horizon: ${ACTION_HORIZON}
low_level_execution_horizon: ${LOW_LEVEL_EXECUTION_HORIZON}
high_level_replan_interval: ${HIGH_LEVEL_REPLAN_INTERVAL}
recent_frame_interval: ${RECENT_FRAME_INTERVAL}
recent_frames: ${RECENT_FRAMES}
memory_frames: ${MEMORY_FRAMES}
keyframe_merge_distance: ${KEYFRAME_MERGE_DISTANCE}
vlm_log_dir: ${RUN_ROOT}/vlm_logs
vlm_log_images: ${VLM_LOG_IMAGES}
eval_test_num: ${EVAL_TEST_NUM}
task_timeout_seconds: ${TASK_TIMEOUT_SECONDS}
task_timeout_seconds_per_episode: ${TASK_TIMEOUT_SECONDS_PER_EPISODE}
run_root: ${RUN_ROOT}
EOF
printf 'idx\ttask\tstatus\tseconds\tlog_path\tsuccess_rate\tresult_path\n' > "${SUMMARY_FILE}"

write_report() {
  local completed failed
  completed=$(awk 'NR>1 {n++} END {print n+0}' "${SUMMARY_FILE}")
  failed=$(awk -F '\t' 'NR>1 && $3 != 0 {n++} END {print n+0}' "${SUMMARY_FILE}")
  {
    printf '# MemER Eval Report\n\n- Run: `%s`\n- Environment: `%s`\n- Episodes/task: `%s`\n' "${RUN_ID}" "${ENVIRONMENT_TYPE}" "${EVAL_TEST_NUM}"
    printf -- '- Checkpoints: high `%s`, low `%s`\n- Progress: `%s/%s`, failed launchers: `%s`\n\n' "${HIGH_STEP}" "${LOW_STEP}" "${completed}" "${#TASKS[@]}" "${failed}"
    printf '| # | Task | Status | Success Rate | Seconds | Log | Result |\n|---:|---|---:|---:|---:|---|---|\n'
    tail -n +2 "${SUMMARY_FILE}" | while IFS=$'\t' read -r idx task status seconds log rate result; do
      [[ "${result}" == "-" ]] && result_link=- || result_link="[result](<${result}>)"
      printf '| %s | %s | %s | %s | %s | [log](<%s>) | %s |\n' "$((idx+1))" "${task}" "${status}" "${rate}" "${seconds}" "${log}" "${result_link}"
    done
  } > "${REPORT_FILE}"
}
write_report
if [[ "${DRY_RUN}" == 1 ]]; then echo "MemER dry-run passed: ${RUN_ROOT}"; exit 0; fi

if [[ "${HIGH_SERVER_MANAGED}" == 1 && ! -x "${HIGH_PYTHON}" ]]; then echo "HIGH_PYTHON missing; run install_env.sh: ${HIGH_PYTHON}" >&2; exit 1; fi
if [[ "${LOW_SERVER_MANAGED}" == 1 && ! -x "${LOW_PYTHON}" ]]; then echo "LOW_PYTHON missing; run install_env.sh: ${LOW_PYTHON}" >&2; exit 1; fi

PIDS=()
ACTIVE_TASK_PID=""
HIGH_SERVER_PID=0
LOW_SERVER_PID=0

terminate_group() {
  local pid=${1:-} attempt
  [[ -n "${pid}" && "${pid}" != 0 ]] || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || return 0
  for attempt in {1..50}; do
    kill -0 -- "-${pid}" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  terminate_group "${ACTIVE_TASK_PID}"
  [[ -n "${ACTIVE_TASK_PID}" ]] && wait "${ACTIVE_TASK_PID}" 2>/dev/null || true
  if [[ "${KEEP_SERVER}" != 1 ]]; then
    for pid in "${PIDS[@]:-}"; do terminate_group "${pid}"; done
    for pid in "${PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

HIGH_SERVER_TOKEN="${RUN_ID}:high:${RANDOM}${RANDOM}"
LOW_SERVER_TOKEN="${RUN_ID}:low:${RANDOM}${RANDOM}"

if [[ "${HIGH_SERVER_MANAGED}" == 1 ]]; then
  high_cmd=("${HIGH_PYTHON}" "${SCRIPT_DIR}/serve_high_policy.py" --checkpoint "${HIGH_CHECKPOINT}" --processor-dir "${HIGH_PROCESSOR_DIR}" --host "${HIGH_HOST}" --port "${HIGH_PORT}" --device "${HIGH_DEVICE}" --recent-frames "${RECENT_FRAMES}" --memory-frames "${MEMORY_FRAMES}" --recent-frame-interval "${RECENT_FRAME_INTERVAL}" --merge-distance "${KEYFRAME_MERGE_DISTANCE}" --max-new-tokens "${HIGH_MAX_NEW_TOKENS}" --max-sessions "${HIGH_MAX_SESSIONS}" --attn-implementation "${HIGH_ATTN_IMPLEMENTATION}" --server-token "${HIGH_SERVER_TOKEN}" --vlm-log-dir "${RUN_ROOT}/vlm_logs")
  [[ "${VLM_LOG_IMAGES}" == 0 ]] && high_cmd+=(--no-vlm-log-images)
  [[ "${HIGH_MOCK}" == 1 ]] && high_cmd+=(--mock)
  setsid env CUDA_VISIBLE_DEVICES="${HIGH_GPU}" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${high_cmd[@]}" > "${RUN_ROOT}/servers/high.log" 2>&1 &
  HIGH_SERVER_PID=$!
  PIDS+=("${HIGH_SERVER_PID}")
fi
if [[ "${LOW_SERVER_MANAGED}" == 1 ]]; then
  low_cmd=("${LOW_PYTHON}" "${SCRIPT_DIR}/serve_low_policy.py" --checkpoint "${LOW_CHECKPOINT}" --rlinf-root "${RLINF_ROOT}" --openpi-client-src "${OPENPI_CLIENT_SRC}" --config-name "${OPENPI_CONFIG_NAME}" --host "${LOW_HOST}" --port "${LOW_PORT}" --device "${LOW_DEVICE}" --num-steps "${LOW_NUM_STEPS}" --action-horizon "${ACTION_HORIZON}" --server-token "${LOW_SERVER_TOKEN}")
  [[ "${LOW_MOCK}" == 1 ]] && low_cmd+=(--mock)
  setsid env CUDA_VISIBLE_DEVICES="${LOW_GPU}" PYTHONNOUSERSITE=1 PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 "${low_cmd[@]}" > "${RUN_ROOT}/servers/low.log" 2>&1 &
  LOW_SERVER_PID=$!
  PIDS+=("${LOW_SERVER_PID}")
fi

"${ROBOTWIN_PYTHON}" - "${HIGH_HOST}" "${HIGH_PORT}" "${LOW_HOST}" "${LOW_PORT}" \
  "${SERVER_READY_TIMEOUT}" "${HIGH_SERVER_MANAGED}" "${HIGH_SERVER_PID}" "${HIGH_SERVER_TOKEN}" \
  "${LOW_SERVER_MANAGED}" "${LOW_SERVER_PID}" "${LOW_SERVER_TOKEN}" <<'PY'
import os, sys, time, requests

high_host, high_port = sys.argv[1], int(sys.argv[2])
low_host, low_port, timeout = sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
specs = {
    "high": {
        "url": f"http://{high_host}:{high_port}/healthz", "managed": sys.argv[6] == "1",
        "pid": int(sys.argv[7]), "token": sys.argv[8],
    },
    "low": {
        "url": f"http://{low_host}:{low_port}/healthz", "managed": sys.argv[9] == "1",
        "pid": int(sys.argv[10]), "token": sys.argv[11],
    },
}
deadline = time.monotonic() + timeout
pending = set(specs)
while pending and time.monotonic() < deadline:
    for role in list(pending):
        spec = specs[role]
        if spec["managed"] and (spec["pid"] <= 0 or not os.path.exists(f"/proc/{spec['pid']}")):
            raise SystemExit(
                f"managed {role} server exited before readiness; see servers/{role}.log"
            )
        try:
            response = requests.get(spec["url"], timeout=3)
            if response.status_code != 200:
                continue
            if spec["managed"] and response.json().get("server_token") != spec["token"]:
                continue
            pending.remove(role)
        except (requests.RequestException, ValueError):
            pass
    if pending: time.sleep(2)
if pending:
    raise SystemExit(f"servers not ready: {sorted(pending)}")
PY

cd "${ROBOTWIN_ROOT}"
export CUDA_VISIBLE_DEVICES="${LOW_GPU}"
export ROBOTWIN_SAPIEN_RENDER_DEVICE=cuda:0
export ROBOTWIN_SAPIEN_CAMERA_SHADER_DIR ROBOTWIN_SAPIEN_RT_SAMPLES ROBOTWIN_SAPIEN_RT_PATH_DEPTH ROBOTWIN_SAPIEN_RT_DENOISER
export ROBOTWIN_EVAL_RESULT_ROOT="${RUN_ROOT}/eval_result"
export MEMER_HIGH_HOST="${HIGH_HOST}" MEMER_HIGH_PORT="${HIGH_PORT}" MEMER_LOW_HOST="${LOW_HOST}" MEMER_LOW_PORT="${LOW_PORT}"
export MEMER_ENVIRONMENT_TYPE="${ENVIRONMENT_TYPE}" MEMER_WORKER_ID="${WORKER_ID}"
export PYTHONPATH="${ROBOTWIN_CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}"

for idx in "${!TASKS[@]}"; do
  task=${TASKS[$idx]}; log="${RUN_ROOT}/tasks/$(printf '%03d_%s' "${idx}" "${task}").log"; start=$(date +%s); status=0
  export ROBOTWIN_EVAL_SEED_LIST_PATH="${SEED_LIST_ROOT}/${TASK_CONFIG}/${task}.json"
  cmd=("${ROBOTWIN_PYTHON}" script/eval_policy.py --config policy/MemER/deploy_policy.yml --overrides --task_name "${task}" --task_config "${TASK_CONFIG}" --ckpt_setting "${MEMER_CKPT_KEY}" --seed "${SEED}" --test_num "${EVAL_TEST_NUM}" --policy_name MemER --high_host "${HIGH_HOST}" --high_port "${HIGH_PORT}" --low_host "${LOW_HOST}" --low_port "${LOW_PORT}" --request_timeout "${REQUEST_TIMEOUT}" --openpi_client_src "${OPENPI_CLIENT_SRC}" --action_horizon "${ACTION_HORIZON}" --low_level_execution_horizon "${LOW_LEVEL_EXECUTION_HORIZON}" --high_level_replan_interval "${HIGH_LEVEL_REPLAN_INTERVAL}" --action_semantics "${ACTION_SEMANTICS}" --environment_type "${ENVIRONMENT_TYPE}" --worker_id "${WORKER_ID}" --eval_video_log "${ROBOTWIN_EVAL_VIDEO_LOG}")
  if (( TASK_TIMEOUT_SECONDS > 0 )); then
    setsid timeout --kill-after=30s "${TASK_TIMEOUT_SECONDS}" "${cmd[@]}" > "${log}" 2>&1 &
  else
    setsid "${cmd[@]}" > "${log}" 2>&1 &
  fi
  ACTIVE_TASK_PID=$!
  wait "${ACTIVE_TASK_PID}" || status=$?
  ACTIVE_TASK_PID=""
  if [[ "${status}" == 124 ]]; then
    printf '[MemER] task timed out after %s seconds (%s requested episodes); set TASK_TIMEOUT_SECONDS to override.\n' \
      "${TASK_TIMEOUT_SECONDS}" "${EVAL_TEST_NUM}" >> "${log}"
  fi
  seconds=$(($(date +%s)-start)); result=$(awk '/Data has been saved to / {sub(/^.*Data has been saved to /, ""); p=$0} END {print p}' "${log}")
  rate=-; if [[ -n "${result}" && -f "${result}" ]]; then rate=$(awk 'NF && $1 ~ /^[-+0-9.]+$/ {v=$1} END {print v==""?"-":v}' "${result}"); else result=-; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${idx}" "${task}" "${status}" "${seconds}" "${log}" "${rate}" "${result}" >> "${SUMMARY_FILE}"
  write_report
  [[ "${status}" == 0 || "${CONTINUE_ON_ERROR}" == 1 ]] || exit "${status}"
done
echo "MemER evaluation complete: ${RUN_ROOT}"
