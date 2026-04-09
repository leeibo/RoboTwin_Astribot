#!/usr/bin/env bash
set -u
set -o pipefail

# Example:
#   WHITELIST_FILE=task_config/rotate_task_whitelist.yml \
#   bash collect_rotate_tasks_multi_difficulty_whitelist.sh demo_clean 0

BASE_CONFIG="${1:-demo_clean}"
GPU_ID="${2:-0}"
WHITELIST_FILE="${WHITELIST_FILE:-task_config/rotate_task_whitelist.yml}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_GENERATED_CONFIG="${KEEP_GENERATED_CONFIG:-1}"
DIFFICULTY_PRESETS="${DIFFICULTY_PRESETS:-easy:150:1.0,medium:200:1.0,hard:240:1.0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

./script/.update_path.sh > /dev/null 2>&1 || true

BASE_CONFIG_PATH="task_config/${BASE_CONFIG}.yml"
if [[ ! -f "${BASE_CONFIG_PATH}" ]]; then
  echo "[Error] base config not found: ${BASE_CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${WHITELIST_FILE}" ]]; then
  echo "[Error] whitelist file not found: ${WHITELIST_FILE}"
  exit 1
fi

IFS=',' read -r -a PRESET_ITEMS <<< "${DIFFICULTY_PRESETS}"
if [[ ${#PRESET_ITEMS[@]} -eq 0 ]]; then
  echo "[Error] empty DIFFICULTY_PRESETS"
  exit 1
fi

GENERATED_CONFIGS=()
PRESET_DESCS=()

for item in "${PRESET_ITEMS[@]}"; do
  IFS=':' read -r label fan_angle shared_ratio <<< "${item}"
  if [[ -z "${label:-}" || -z "${fan_angle:-}" || -z "${shared_ratio:-}" ]]; then
    echo "[Error] invalid preset item: '${item}'"
    echo "        expected format: label:fan_angle_deg:shared_ratio"
    exit 1
  fi

  fan_tag="${fan_angle//./p}"
  ratio_tag="${shared_ratio//./p}"
  config_name="${BASE_CONFIG}__${label}_fan${fan_tag}_r${ratio_tag}"
  config_path="task_config/${config_name}.yml"

  python - <<'PY' "${BASE_CONFIG_PATH}" "${config_path}" "${label}" "${fan_angle}" "${shared_ratio}"
import sys
import yaml

src, dst, label, fan_angle, shared_ratio = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

cfg["fan_angle_deg"] = float(fan_angle)
cfg["rotate_theta_shared_ratio"] = float(shared_ratio)
cfg["difficulty_tag"] = f"{label}_fan{int(round(float(fan_angle)))}"

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  GENERATED_CONFIGS+=("${config_name}")
  PRESET_DESCS+=("${label} (fan=${fan_angle}, ratio=${shared_ratio}) -> ${config_name}")
done

echo "[Info] base_config=${BASE_CONFIG} | gpu_id=${GPU_ID} | continue_on_error=${CONTINUE_ON_ERROR} | dry_run=${DRY_RUN}"
echo "[Info] whitelist_file=${WHITELIST_FILE}"
echo "[Info] difficulty_presets=${DIFFICULTY_PRESETS}"
echo "[Info] generated_configs=${#GENERATED_CONFIGS[@]}"
for i in "${!PRESET_DESCS[@]}"; do
  printf '  %03d. %s\n' "$((i + 1))" "${PRESET_DESCS[$i]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  for config_name in "${GENERATED_CONFIGS[@]}"; do
    rm -f "task_config/${config_name}.yml"
  done
  echo "[DryRun] skip collection"
  exit 0
fi

SUCCESS_CONFIGS=()
FAILED_CONFIGS=()

for config_name in "${GENERATED_CONFIGS[@]}"; do
  echo
  echo "[Difficulty] collecting with config=${config_name}"
  if WHITELIST_FILE="${WHITELIST_FILE}" CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" DRY_RUN="${DRY_RUN}" \
      bash collect_rotate_tasks_whitelist.sh "${config_name}" "${GPU_ID}"; then
    SUCCESS_CONFIGS+=("${config_name}")
  else
    FAILED_CONFIGS+=("${config_name}")
    echo "[Error] difficulty collection failed: ${config_name}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      break
    fi
  fi
done

if [[ "${KEEP_GENERATED_CONFIG}" != "1" ]]; then
  for config_name in "${GENERATED_CONFIGS[@]}"; do
    rm -f "task_config/${config_name}.yml"
  done
fi

echo
echo "[Summary]"
echo "  difficulty_succeeded: ${#SUCCESS_CONFIGS[@]}"
echo "  difficulty_failed:    ${#FAILED_CONFIGS[@]}"
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
  echo "  failed_difficulties:"
  for config_name in "${FAILED_CONFIGS[@]}"; do
    echo "    - ${config_name}"
  done
  exit 1
fi

exit 0
