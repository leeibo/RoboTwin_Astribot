#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENVIRONMENT_TYPE=${ENVIRONMENT_TYPE:-clean}
MODE=${MODE:-formal}
MEMER_CKPT_KEY=${MEMER_CKPT_KEY:-memer_astribot_step_4500_18000}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
HIGH_PYTHON=${HIGH_PYTHON:-/root/autodl-tmp/MEMER_eval/MemER/env/qwen3vl/bin/python}
HIGH_PROCESSOR_DIR=${HIGH_PROCESSOR_DIR:-}
LOW_PYTHON=${LOW_PYTHON:-/root/autodl-tmp/MEMER_eval/MemER/env/rlinf-pi05/bin/python}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-/root/autodl-tmp/MEMER_eval/MemER/source/low_level/openpi/packages/openpi-client/src}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
SEED_LIST_ROOT=${SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}
EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}
TASK_LIMIT=${TASK_LIMIT:-0}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
RESUME_FROM_RUN=${RESUME_FROM_RUN:-}
DRY_RUN=${DRY_RUN:-0}
TMUX_MONITOR=${TMUX_MONITOR:-0}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
HIGH_MOCK=${HIGH_MOCK:-0}
LOW_MOCK=${LOW_MOCK:-0}
HIGH_HOST=${HIGH_HOST:-127.0.0.1}
LOW_HOST=${LOW_HOST:-127.0.0.1}
LOW_LEVEL_EXECUTION_HORIZON=${LOW_LEVEL_EXECUTION_HORIZON:-5}
HIGH_LEVEL_REPLAN_INTERVAL=${HIGH_LEVEL_REPLAN_INTERVAL:-5}
RECENT_FRAME_INTERVAL=${RECENT_FRAME_INTERVAL:-5}
RECENT_FRAMES=${RECENT_FRAMES:-8}
MEMORY_FRAMES=${MEMORY_FRAMES:-8}
KEYFRAME_MERGE_DISTANCE=${KEYFRAME_MERGE_DISTANCE:-5}
HIGH_MAX_NEW_TOKENS=${HIGH_MAX_NEW_TOKENS:-128}
HIGH_MAX_SESSIONS=${HIGH_MAX_SESSIONS:-32}
VLM_LOG_IMAGES=${VLM_LOG_IMAGES:-1}
LOW_NUM_STEPS=${LOW_NUM_STEPS:-5}

case "${ENVIRONMENT_TYPE}" in
  clean)
    TASK_CONFIG=${TASK_CONFIG:-info_gathering_demo}
    HIGH_GPU=${HIGH_GPU:-0}
    LOW_GPUS=${LOW_GPUS:-1,2,3,4}
    HIGH_PORT=${HIGH_PORT:-5901}
    LOW_PORT_BASE=${LOW_PORT_BASE:-5910}
    ;;
  randomized)
    TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
    HIGH_GPU=${HIGH_GPU:-5}
    LOW_GPUS=${LOW_GPUS:-6,7,8}
    HIGH_PORT=${HIGH_PORT:-5951}
    LOW_PORT_BASE=${LOW_PORT_BASE:-5960}
    ;;
  *) echo "ENVIRONMENT_TYPE must be clean or randomized" >&2; exit 2 ;;
esac
IFS=',' read -r -a GPU_ARRAY <<< "${LOW_GPUS}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "LOW_GPUS is empty" >&2; exit 2; }
for gpu in "${GPU_ARRAY[@]}"; do [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid LOW_GPUS entry: ${gpu}" >&2; exit 2; }; done
[[ "${HIGH_GPU}" =~ ^[0-9]+$ ]] || { echo "invalid HIGH_GPU: ${HIGH_GPU}" >&2; exit 2; }
[[ "${TASK_LIMIT}" =~ ^[0-9]+$ && "${EVAL_TEST_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid task/episode limits" >&2; exit 2; }
[[ "${VLM_LOG_IMAGES}" == 0 || "${VLM_LOG_IMAGES}" == 1 ]] || { echo "VLM_LOG_IMAGES must be 0 or 1" >&2; exit 2; }
if [[ "${TMUX_MONITOR}" == 1 ]] && ! command -v tmux >/dev/null 2>&1; then
  echo "TMUX_MONITOR=1 but tmux is not installed" >&2
  exit 1
fi

if [[ -n "${RESUME_FROM_RUN}" ]]; then
  RUN_ROOT=$(realpath "${RESUME_FROM_RUN}")
  [[ -d "${RUN_ROOT}" ]] || { echo "RESUME_FROM_RUN does not exist: ${RUN_ROOT}" >&2; exit 1; }
  RUN_ID=$(basename "${RUN_ROOT}")
else
  RUN_ID=${RUN_ID:-memer_${ENVIRONMENT_TYPE}_$(date +%Y%m%d_%H%M%S)_$$}
  RUN_ROOT="${SCRIPT_DIR}/runs/eval_seed_whitelist_randomized/${RUN_ID}"
  mkdir -p "${RUN_ROOT}/tasks" "${RUN_ROOT}/servers"
fi
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
MANIFEST_FILE="${RUN_ROOT}/manifest.yaml"
REPORT_FILE="${RUN_ROOT}/report.md"
WORK_PLAN="${RUN_ROOT}/work_plan.tsv"
NEXT_FILE="${RUN_ROOT}/next_index.txt"
QUEUE_LOCK="${RUN_ROOT}/queue.lock"
SUMMARY_LOCK="${RUN_ROOT}/summary.lock"

if [[ -n "${RESUME_FROM_RUN}" ]]; then
  [[ -f "${MANIFEST_FILE}" ]] || { echo "resume manifest missing: ${MANIFEST_FILE}" >&2; exit 1; }
  "${ROBOTWIN_PYTHON}" - "${MANIFEST_FILE}" "${ENVIRONMENT_TYPE}" "${TASK_CONFIG}" \
    "${MEMER_CKPT_KEY}" "${EVAL_TEST_NUM}" "${LOW_LEVEL_EXECUTION_HORIZON}" \
    "${HIGH_LEVEL_REPLAN_INTERVAL}" "${RECENT_FRAME_INTERVAL}" "${TASK_LIMIT}" <<'PY'
import sys, yaml
from pathlib import Path
manifest = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
expected = {
    "environment_type": sys.argv[2], "task_config": sys.argv[3],
    "checkpoint_key": sys.argv[4], "episodes_per_task": int(sys.argv[5]),
    "low_level_execution_horizon": int(sys.argv[6]),
    "high_level_replan_interval": int(sys.argv[7]),
    "recent_frame_interval": int(sys.argv[8]),
    "task_limit": int(sys.argv[9]),
}
errors = [f"{key}: old={manifest.get(key)!r}, requested={value!r}" for key, value in expected.items() if manifest.get(key) != value]
if errors: raise SystemExit("resume configuration mismatch:\n" + "\n".join(errors))
PY
fi

mapfile -t FIELDS < <("${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${MEMER_CKPT_KEY}" <<'PY'
import sys, yaml
from pathlib import Path
record = (yaml.safe_load(Path(sys.argv[1]).read_text()) or {})["checkpoints"][sys.argv[2]]
for group, root_key in (("required_high_files", "high_checkpoint_dir"), ("required_low_files", "low_checkpoint_dir")):
    root = Path(record[root_key])
    for rel in record[group]:
        if not (root / rel).is_file(): raise SystemExit(f"missing: {root / rel}")
for key in ("high_checkpoint_dir", "low_checkpoint_dir", "rlinf_root", "openpi_config_name"):
    print(record[key])
PY
)
HIGH_CHECKPOINT=${FIELDS[0]}; LOW_CHECKPOINT=${FIELDS[1]}; RLINF_ROOT=${FIELDS[2]}; OPENPI_CONFIG_NAME=${FIELDS[3]}
HIGH_PROCESSOR_DIR=${HIGH_PROCESSOR_DIR:-${HIGH_CHECKPOINT}}
if [[ "${HIGH_MOCK}" != 1 ]]; then
  for name in preprocessor_config.json video_preprocessor_config.json tokenizer_config.json; do
    [[ -f "${HIGH_PROCESSOR_DIR}/${name}" ]] || {
      echo "missing Qwen3-VL processor file: ${HIGH_PROCESSOR_DIR}/${name}" >&2
      echo "Set HIGH_PROCESSOR_DIR=/path/to/high-level-training-output." >&2
      exit 1
    }
  done
fi
mapfile -t ALL_TASKS < <("${ROBOTWIN_PYTHON}" - "${WHITELIST}" <<'PY'
import sys, yaml
from pathlib import Path
for item in yaml.safe_load(Path(sys.argv[1]).read_text()): print(item)
PY
)
if (( TASK_LIMIT > 0 && TASK_LIMIT < ${#ALL_TASKS[@]} )); then ALL_TASKS=("${ALL_TASKS[@]:0:TASK_LIMIT}"); fi

declare -A DONE=()
if [[ -n "${RESUME_FROM_RUN}" && -f "${SUMMARY_FILE}" ]]; then
  while IFS=$'\t' read -r idx task worker gpu port status seconds log child_run; do
    [[ "${idx}" == idx ]] && continue
    [[ "${status}" == 0 ]] && DONE["${task}"]=1
  done < "${SUMMARY_FILE}"
fi
PENDING=()
for task in "${ALL_TASKS[@]}"; do [[ -n "${DONE[$task]:-}" ]] || PENDING+=("${task}"); done
printf 'idx\ttask\n' > "${WORK_PLAN}"
for idx in "${!PENDING[@]}"; do printf '%s\t%s\n' "${idx}" "${PENDING[$idx]}" >> "${WORK_PLAN}"; done
if [[ ! -f "${SUMMARY_FILE}" ]]; then printf 'idx\ttask\tworker\tgpu\tport\tstatus\tseconds\tlog_path\tchild_run\n' > "${SUMMARY_FILE}"; fi
printf '0\n' > "${NEXT_FILE}"

if [[ -z "${RESUME_FROM_RUN}" ]]; then
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
high_gpu: ${HIGH_GPU}
low_gpus: ${LOW_GPUS}
high_server: ${HIGH_HOST}:${HIGH_PORT}
low_port_base: ${LOW_PORT_BASE}
workers: ${#GPU_ARRAY[@]}
episodes_per_task: ${EVAL_TEST_NUM}
task_limit: ${TASK_LIMIT}
low_level_execution_horizon: ${LOW_LEVEL_EXECUTION_HORIZON}
high_level_replan_interval: ${HIGH_LEVEL_REPLAN_INTERVAL}
recent_frame_interval: ${RECENT_FRAME_INTERVAL}
vlm_log_dir: ${RUN_ROOT}/vlm_logs
vlm_log_images: ${VLM_LOG_IMAGES}
resume_from_run: ${RESUME_FROM_RUN:-null}
run_root: ${RUN_ROOT}
EOF
fi

write_report() {
  local completed failed
  completed=$(awk -F '\t' 'NR>1 {seen[$2]=1} END {for (task in seen) n++; print n+0}' "${SUMMARY_FILE}")
  failed=$(awk -F '\t' 'NR>1 {status[$2]=$6} END {for (task in status) if (status[task] != 0) n++; print n+0}' "${SUMMARY_FILE}")
  {
    printf '# MemER %s Full Evaluation\n\n' "${ENVIRONMENT_TYPE}"
    printf -- '- Run: `%s`\n- Workers: `%s` (`%s`)\n- Episodes/task: `%s`\n' "${RUN_ID}" "${#GPU_ARRAY[@]}" "${LOW_GPUS}" "${EVAL_TEST_NUM}"
    printf -- '- Completed rows: `%s`, launcher failures: `%s`, pending at start: `%s`\n\n' "${completed}" "${failed}" "${#PENDING[@]}"
    printf '| Task | Worker | GPU | Status | Seconds | Log | Child run |\n|---|---:|---:|---:|---:|---|---|\n'
    tail -n +2 "${SUMMARY_FILE}" | while IFS=$'\t' read -r idx task worker gpu port status seconds log child; do
      printf '| %s | %s | %s | %s | %s | [log](<%s>) | `%s` |\n' "${task}" "${worker}" "${gpu}" "${status}" "${seconds}" "${log}" "${child}"
    done
  } > "${REPORT_FILE}"
}
write_report

TMUX_SESSION="memer_${ENVIRONMENT_TYPE}_${RUN_ID//[^a-zA-Z0-9_]/_}"
if [[ "${DRY_RUN}" == 1 ]]; then
  MODE=custom TASK_LIMIT=1 EVAL_TEST_NUM=1 TASK_NAME="${ALL_TASKS[0]}" ENVIRONMENT_TYPE="${ENVIRONMENT_TYPE}" \
    TASK_CONFIG="${TASK_CONFIG}" HIGH_GPU="${HIGH_GPU}" LOW_GPU="${GPU_ARRAY[0]}" HIGH_PORT="${HIGH_PORT}" LOW_PORT="${LOW_PORT_BASE}" \
    HIGH_PROCESSOR_DIR="${HIGH_PROCESSOR_DIR}" DRY_RUN=1 bash "${SCRIPT_DIR}/eval.sh"
  echo "Full-run dry-run passed: ${RUN_ROOT}; pending=${#PENDING[@]} workers=${#GPU_ARRAY[@]}"
  exit 0
fi
if (( ${#PENDING[@]} == 0 )); then
  echo "Nothing to resume; all selected tasks already have status=0: ${RUN_ROOT}"
  exit 0
fi

"${ROBOTWIN_PYTHON}" - "${HIGH_HOST}" "${HIGH_PORT}" "${LOW_HOST}" "${LOW_PORT_BASE}" \
  "${#GPU_ARRAY[@]}" <<'PY'
import socket, sys

high_host, high_port = sys.argv[1], int(sys.argv[2])
low_host, low_base, workers = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
specs = [("high", high_host, high_port)]
specs.extend((f"low-worker-{idx}", low_host, low_base + idx) for idx in range(workers))
sockets = []
try:
    for label, host, port in specs:
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

if [[ "${TMUX_MONITOR}" == 1 ]]; then
  tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && tmux kill-session -t "${TMUX_SESSION}"
  tmux new-session -d -s "${TMUX_SESSION}" -n progress \
    "watch -n 2 'tail -n 40 \"${SUMMARY_FILE}\"; echo; tail -n 30 \"${REPORT_FILE}\"'"
  tmux set-option -t "${TMUX_SESSION}" remain-on-exit on >/dev/null
  echo "Monitor: tmux attach -t ${TMUX_SESSION}"
fi
[[ -x "${HIGH_PYTHON}" ]] || { echo "HIGH_PYTHON missing: ${HIGH_PYTHON}" >&2; exit 1; }
[[ -x "${LOW_PYTHON}" ]] || { echo "LOW_PYTHON missing: ${LOW_PYTHON}" >&2; exit 1; }

PIDS=()
WORKER_PIDS=()
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
kill_descendants() {
  local parent=$1 child
  for child in $(pgrep -P "${parent}" 2>/dev/null || true); do
    kill_descendants "${child}"
  done
  kill "${parent}" 2>/dev/null || true
}
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do [[ -n "${pid}" ]] && kill_descendants "${pid}"; done
  for pid in "${PIDS[@]:-}"; do terminate_group "${pid}"; done
  for pid in "${WORKER_PIDS[@]:-}"; do [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

HIGH_SERVER_TOKEN="${RUN_ID}:high:${RANDOM}${RANDOM}"
high_cmd=("${HIGH_PYTHON}" "${SCRIPT_DIR}/serve_high_policy.py" --checkpoint "${HIGH_CHECKPOINT}" --processor-dir "${HIGH_PROCESSOR_DIR}" --host "${HIGH_HOST}" --port "${HIGH_PORT}" --device cuda:0 --recent-frames "${RECENT_FRAMES}" --memory-frames "${MEMORY_FRAMES}" --recent-frame-interval "${RECENT_FRAME_INTERVAL}" --merge-distance "${KEYFRAME_MERGE_DISTANCE}" --max-new-tokens "${HIGH_MAX_NEW_TOKENS}" --max-sessions "${HIGH_MAX_SESSIONS}" --server-token "${HIGH_SERVER_TOKEN}" --vlm-log-dir "${RUN_ROOT}/vlm_logs")
[[ "${VLM_LOG_IMAGES}" == 0 ]] && high_cmd+=(--no-vlm-log-images)
[[ "${HIGH_MOCK}" == 1 ]] && high_cmd+=(--mock)
setsid env CUDA_VISIBLE_DEVICES="${HIGH_GPU}" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${high_cmd[@]}" > "${RUN_ROOT}/servers/high.log" 2>&1 &
HIGH_SERVER_PID=$!
PIDS+=("${HIGH_SERVER_PID}")

LOW_PORTS=()
LOW_PIDS=()
LOW_TOKENS=()
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}; port=$((LOW_PORT_BASE + worker)); LOW_PORTS+=("${port}")
  token="${RUN_ID}:low-${worker}:${RANDOM}${RANDOM}"
  LOW_TOKENS+=("${token}")
  low_cmd=("${LOW_PYTHON}" "${SCRIPT_DIR}/serve_low_policy.py" --checkpoint "${LOW_CHECKPOINT}" --rlinf-root "${RLINF_ROOT}" --openpi-client-src "${OPENPI_CLIENT_SRC}" --config-name "${OPENPI_CONFIG_NAME}" --host "${LOW_HOST}" --port "${port}" --device cuda:0 --num-steps "${LOW_NUM_STEPS}" --action-horizon 50 --server-token "${token}")
  [[ "${LOW_MOCK}" == 1 ]] && low_cmd+=(--mock)
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 "${low_cmd[@]}" > "${RUN_ROOT}/servers/low_worker${worker}_gpu${gpu}.log" 2>&1 &
  pid=$!
  LOW_PIDS+=("${pid}")
  PIDS+=("${pid}")
done

READY_ARGS=("${HIGH_HOST}" "${HIGH_PORT}" "${HIGH_SERVER_PID}" "${HIGH_SERVER_TOKEN}" \
  "${LOW_HOST}" "${SERVER_READY_TIMEOUT}" "${#LOW_PORTS[@]}")
for worker in "${!LOW_PORTS[@]}"; do
  READY_ARGS+=("${LOW_PORTS[$worker]}" "${LOW_PIDS[$worker]}" "${LOW_TOKENS[$worker]}")
done
"${ROBOTWIN_PYTHON}" - "${READY_ARGS[@]}" <<'PY'
import os, sys, time, requests

hh, hp, high_pid, high_token = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
lh, timeout, count = sys.argv[5], float(sys.argv[6]), int(sys.argv[7])
specs = {
    "high": {"url": f"http://{hh}:{hp}/healthz", "pid": high_pid, "token": high_token},
}
offset = 8
for idx in range(count):
    port, pid, token = int(sys.argv[offset]), int(sys.argv[offset + 1]), sys.argv[offset + 2]
    specs[f"low-worker-{idx}"] = {
        "url": f"http://{lh}:{port}/healthz", "pid": pid, "token": token,
    }
    offset += 3
deadline = time.monotonic() + timeout
pending = set(specs)
while pending and time.monotonic() < deadline:
    for role in list(pending):
        spec = specs[role]
        if not os.path.exists(f"/proc/{spec['pid']}"):
            raise SystemExit(f"managed {role} server exited before readiness; see servers logs")
        try:
            response = requests.get(spec["url"], timeout=3)
            if response.status_code == 200 and response.json().get("server_token") == spec["token"]:
                pending.remove(role)
        except (requests.RequestException, ValueError):
            pass
    if pending: time.sleep(2)
if pending:
    raise SystemExit(f"servers not ready: {sorted(pending)}")
PY

claim_task() {
  local next
  exec 9>"${QUEUE_LOCK}"; flock 9
  next=$(<"${NEXT_FILE}")
  if (( next >= ${#PENDING[@]} )); then flock -u 9; return 1; fi
  CLAIM_INDEX=${next}; CLAIM_TASK=${PENDING[$next]}
  printf '%s\n' "$((next+1))" > "${NEXT_FILE}"
  flock -u 9
}

run_worker() {
  local worker=$1 gpu=$2 port=$3 task idx log start status seconds child
  while claim_task; do
    idx=${CLAIM_INDEX}; task=${CLAIM_TASK}; log="${RUN_ROOT}/tasks/$(printf '%03d_%s' "${idx}" "${task}")_worker${worker}.log"; start=$(date +%s); status=0
    child="${RUN_ID}_${idx}_w${worker}"
    env MODE=custom ENVIRONMENT_TYPE="${ENVIRONMENT_TYPE}" TASK_CONFIG="${TASK_CONFIG}" TASK_NAME="${task}" \
      EVAL_TEST_NUM="${EVAL_TEST_NUM}" TASK_LIMIT=1 TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS}" CONTINUE_ON_ERROR=0 \
      RUN_ID="${child}" HIGH_SERVER_MANAGED=0 LOW_SERVER_MANAGED=0 HIGH_HOST="${HIGH_HOST}" HIGH_PORT="${HIGH_PORT}" \
      LOW_HOST="${LOW_HOST}" LOW_PORT="${port}" HIGH_GPU="${HIGH_GPU}" LOW_GPU="${gpu}" WORKER_ID="${ENVIRONMENT_TYPE}-worker-${worker}" \
      LOW_LEVEL_EXECUTION_HORIZON="${LOW_LEVEL_EXECUTION_HORIZON}" HIGH_LEVEL_REPLAN_INTERVAL="${HIGH_LEVEL_REPLAN_INTERVAL}" \
      RECENT_FRAME_INTERVAL="${RECENT_FRAME_INTERVAL}" bash "${SCRIPT_DIR}/eval.sh" > "${log}" 2>&1 || status=$?
    seconds=$(($(date +%s)-start))
    exec 8>"${SUMMARY_LOCK}"; flock 8
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${idx}" "${task}" "${worker}" "${gpu}" "${port}" "${status}" "${seconds}" "${log}" "${child}" >> "${SUMMARY_FILE}"
    write_report
    flock -u 8
    if [[ "${status}" != 0 && "${CONTINUE_ON_ERROR}" != 1 ]]; then return "${status}"; fi
  done
}

for worker in "${!GPU_ARRAY[@]}"; do run_worker "${worker}" "${GPU_ARRAY[$worker]}" "${LOW_PORTS[$worker]}" & WORKER_PIDS+=("$!"); done
worker_status=0
for pid in "${WORKER_PIDS[@]}"; do wait "${pid}" || worker_status=$?; done
echo "MemER ${ENVIRONMENT_TYPE} full evaluation complete: ${RUN_ROOT}"
latest_completed=$(awk -F '\t' 'NR>1 {seen[$2]=1} END {for (task in seen) n++; print n+0}' "${SUMMARY_FILE}")
latest_failed=$(awk -F '\t' 'NR>1 {status[$2]=$6} END {for (task in status) if (status[task] != 0) n++; print n+0}' "${SUMMARY_FILE}")
if (( worker_status != 0 || latest_completed != ${#ALL_TASKS[@]} || latest_failed != 0 )); then
  echo "Incomplete/failed: completed=${latest_completed}/${#ALL_TASKS[@]} failed=${latest_failed} worker_status=${worker_status}" >&2
  exit 1
fi
