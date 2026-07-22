#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-0}
EVAL_TEST_NUM=${EVAL_TEST_NUM:-50}
TASK_LIMIT=${TASK_LIMIT:-0}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
TMUX_MONITOR=${TMUX_MONITOR:-0}

clean_run="memer_clean_${RUN_TAG}"
randomized_run="memer_randomized_${RUN_TAG}"
common=(MODE=formal DRY_RUN="${DRY_RUN}" EVAL_TEST_NUM="${EVAL_TEST_NUM}" TASK_LIMIT="${TASK_LIMIT}" TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS}" CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" TMUX_MONITOR="${TMUX_MONITOR}")

PIDS=()
kill_descendants() {
  local parent=$1 child
  for child in $(pgrep -P "${parent}" 2>/dev/null || true); do
    kill_descendants "${child}"
  done
  kill -TERM "${parent}" 2>/dev/null || true
}
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do [[ -n "${pid}" ]] && kill_descendants "${pid}"; done
  for pid in "${PIDS[@]:-}"; do [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true; done
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

env "${common[@]}" ENVIRONMENT_TYPE=clean RUN_ID="${clean_run}" HIGH_GPU=0 LOW_GPUS=1,2,3,4 HIGH_PORT=5901 LOW_PORT_BASE=5910 \
  bash "${SCRIPT_DIR}/eval_seed_whitelist_randomized.sh" &
clean_pid=$!
PIDS+=("${clean_pid}")
env "${common[@]}" ENVIRONMENT_TYPE=randomized RUN_ID="${randomized_run}" HIGH_GPU=5 LOW_GPUS=6,7,8 HIGH_PORT=5951 LOW_PORT_BASE=5960 \
  bash "${SCRIPT_DIR}/eval_seed_whitelist_randomized.sh" &
randomized_pid=$!
PIDS+=("${randomized_pid}")

status=0
wait "${clean_pid}" || status=$?
wait "${randomized_pid}" || status=$?
echo "Clean run: ${SCRIPT_DIR}/runs/eval_seed_whitelist_randomized/${clean_run}"
echo "Randomized run: ${SCRIPT_DIR}/runs/eval_seed_whitelist_randomized/${randomized_run}"
exit "${status}"
