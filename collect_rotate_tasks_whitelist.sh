#!/usr/bin/env bash
set -u
set -o pipefail

TASK_CONFIG="${1:-demo_clean}"
GPU_ID="${2:-0}"
WHITELIST_FILE="${WHITELIST_FILE:-task_config/rotate_task_whitelist.yml}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

./script/.update_path.sh > /dev/null 2>&1 || true

REPORT_DIR="data/collection_reports"
REPORT_PREFIX="collect_rotate_tasks_whitelist__${TASK_CONFIG}"
FAILED_TASKS_FILE="${REPORT_DIR}/${REPORT_PREFIX}__failed_tasks.txt"
SUMMARY_FILE="${REPORT_DIR}/${REPORT_PREFIX}__summary.txt"

if [[ ! -f "${WHITELIST_FILE}" ]]; then
  echo "[Error] whitelist file not found: ${WHITELIST_FILE}"
  exit 1
fi

ALL_ROTATE_TASKS=()
for f in envs/*_rotate_view.py; do
  if [[ ! -f "${f}" ]]; then
    continue
  fi
  task_name="$(basename "${f}" .py)"
  ALL_ROTATE_TASKS+=("${task_name}")
done

IFS=$'\n' ALL_ROTATE_TASKS=($(printf '%s\n' "${ALL_ROTATE_TASKS[@]}" | sort))
unset IFS

declare -A AVAILABLE_MAP
for task_name in "${ALL_ROTATE_TASKS[@]}"; do
  AVAILABLE_MAP["${task_name}"]=1
done

WHITELIST_TASKS_RAW="$(
python - <<'PY' "${WHITELIST_FILE}"
import sys
from pathlib import Path

import yaml

whitelist_path = Path(sys.argv[1])
data = yaml.safe_load(whitelist_path.read_text(encoding="utf-8"))
if data is None:
    data = []

if isinstance(data, dict):
    for key in ("tasks", "include", "task_list", "whitelist_tasks", "selected_tasks"):
        if key in data:
            data = data[key]
            break

if not isinstance(data, list):
    raise SystemExit(
        f"unsupported whitelist format in {whitelist_path}. "
        "Expected a YAML list or a dict containing one of: tasks/include/task_list/whitelist_tasks/selected_tasks"
    )

seen = set()
for item in data:
    if item is None:
        continue
    task_name = str(item).strip()
    if not task_name or task_name in seen:
        continue
    seen.add(task_name)
    print(task_name)
PY
)"
whitelist_status=$?
if [[ ${whitelist_status} -ne 0 ]]; then
  exit ${whitelist_status}
fi

SELECTED_TASKS=()
if [[ -n "${WHITELIST_TASKS_RAW}" ]]; then
  mapfile -t SELECTED_TASKS <<< "${WHITELIST_TASKS_RAW}"
fi

if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then
  echo "[Error] whitelist resolved to 0 tasks: ${WHITELIST_FILE}"
  exit 1
fi

INVALID_TASKS=()
for task_name in "${SELECTED_TASKS[@]}"; do
  if [[ "${task_name}" != *_rotate_view ]]; then
    INVALID_TASKS+=("${task_name} (not a rotate_view task)")
    continue
  fi
  if [[ -z "${AVAILABLE_MAP[${task_name}]:-}" ]]; then
    INVALID_TASKS+=("${task_name} (not found in envs/*_rotate_view.py)")
  fi
done

if [[ ${#INVALID_TASKS[@]} -gt 0 ]]; then
  echo "[Error] invalid whitelist entries detected in ${WHITELIST_FILE}:"
  for task_name in "${INVALID_TASKS[@]}"; do
    echo "  - ${task_name}"
  done
  exit 1
fi

echo "[Info] scope=rotate_view_whitelist"
echo "[Info] discovered=${#ALL_ROTATE_TASKS[@]} | whitelisted=${#SELECTED_TASKS[@]} | to_collect=${#SELECTED_TASKS[@]}"
echo "[Info] whitelist_file=${WHITELIST_FILE}"
echo "[Info] task_config=${TASK_CONFIG} | gpu_id=${GPU_ID} | continue_on_error=${CONTINUE_ON_ERROR} | dry_run=${DRY_RUN}"
for i in "${!SELECTED_TASKS[@]}"; do
  printf '  %03d. %s\n' "$((i + 1))" "${SELECTED_TASKS[$i]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

SUCCESS_TASKS=()
FAILED_TASKS=()

write_reports() {
  mkdir -p "${REPORT_DIR}"

  : > "${FAILED_TASKS_FILE}"
  for failed_task in "${FAILED_TASKS[@]}"; do
    printf '%s\n' "${failed_task}" >> "${FAILED_TASKS_FILE}"
  done

  {
    echo "task_config=${TASK_CONFIG}"
    echo "gpu_id=${GPU_ID}"
    echo "whitelist_file=${WHITELIST_FILE}"
    echo "continue_on_error=${CONTINUE_ON_ERROR}"
    echo "succeeded=${#SUCCESS_TASKS[@]}"
    echo "failed=${#FAILED_TASKS[@]}"
    echo "failed_tasks_file=${FAILED_TASKS_FILE}"
    echo "per_task_failure_report=./data/<task_name>/*/collection_failure.json"
    echo "selected_tasks:"
    for selected_task in "${SELECTED_TASKS[@]}"; do
      echo "  ${selected_task}"
    done
    echo "failed_tasks:"
    for failed_task in "${FAILED_TASKS[@]}"; do
      echo "  ${failed_task}"
    done
  } > "${SUMMARY_FILE}"
}

write_reports

for task_name in "${SELECTED_TASKS[@]}"; do
  echo "[Collect] ${task_name} | config=${TASK_CONFIG} | gpu=${GPU_ID}"
  if bash collect_data.sh "${task_name}" "${TASK_CONFIG}" "${GPU_ID}"; then
    SUCCESS_TASKS+=("${task_name}")
    write_reports
  else
    FAILED_TASKS+=("${task_name}")
    echo "[Error] task failed: ${task_name}"
    write_reports
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      break
    fi
  fi
done

echo
echo "[Summary]"
echo "  succeeded: ${#SUCCESS_TASKS[@]}"
echo "  failed:    ${#FAILED_TASKS[@]}"
echo "  failed_task_list: ${FAILED_TASKS_FILE}"
echo "  summary_file:     ${SUMMARY_FILE}"
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
  echo "  failed_tasks:"
  for task_name in "${FAILED_TASKS[@]}"; do
    echo "    - ${task_name}"
  done
  exit 1
fi

exit 0
