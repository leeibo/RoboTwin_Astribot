#!/usr/bin/env bash
set -u

TASK_CONFIG="${1:-demo_clean}"
GPU_ID="${2:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

./script/.update_path.sh > /dev/null 2>&1 || true

REPORT_DIR="data/collection_reports"
REPORT_PREFIX="collect_rotate_tasks_except_fixing__${TASK_CONFIG}"
FAILED_TASKS_FILE="${REPORT_DIR}/${REPORT_PREFIX}__failed_tasks.txt"
SUMMARY_FILE="${REPORT_DIR}/${REPORT_PREFIX}__summary.txt"

# Copied from task_config/fixing_task.yml
EXCLUDED_TASKS=(
  dump_bin_bigbin_rotate_view
  adjust_bottle_rotate_view
  grab_roller_rotate_view
  handover_block_rotate_view
  handover_mic_rotate_view
  hanging_mug_rotate_view
  lift_pot_rotate_view
  move_can_pot_rotate_view
  move_playingcard_away_rotate_view
  open_microwave_rotate_view
  pick_diverse_bottles_rotate_view
  pick_dual_bottles_rotate_view
  place_bread_skillet_rotate_view
  place_can_basket_rotate_view
  place_dual_shoes_rotate_view
  place_object_basket_rotate_view
  place_phone_stand_rotate_view
  put_bottles_dustbin_rotate_view
  put_object_cabinet_rotate_view
  rotate_qrcode_rotate_view
  scan_object_rotate_view
  stack_bowls_three_rotate_view
  stack_bowls_two_rotate_view
  place_bread_basket_rotate_view 
  # move_stapler_pad_rotate_view
  # place_a2b_left_rotate_view
  # place_a2b_right_rotate_view
)

declare -A EXCLUDED_MAP
for task in "${EXCLUDED_TASKS[@]}"; do
  EXCLUDED_MAP["${task}"]=1
done

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

SELECTED_TASKS=()
for task_name in "${ALL_ROTATE_TASKS[@]}"; do
  if [[ -n "${EXCLUDED_MAP[${task_name}]:-}" ]]; then
    continue
  fi
  SELECTED_TASKS+=("${task_name}")
done

echo "[Info] scope=rotate_view"
echo "[Info] discovered=${#ALL_ROTATE_TASKS[@]} | excluded=${#EXCLUDED_TASKS[@]} | to_collect=${#SELECTED_TASKS[@]}"
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
