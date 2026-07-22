#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
RUN_ROOT="${SCRIPT_DIR}/runs/check/${RUN_ID}"
mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/check.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

ALLOW_MISSING_ENV=${ALLOW_MISSING_ENV:-0}
HIFVLA_CKPT_KEY=${HIFVLA_CKPT_KEY:-hifvla_astribot35_150k}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
HIFVLA_PYTHON=${HIFVLA_PYTHON:-/root/autodl-tmp/HIF-VLA_eval/HIF-VLA/env/hifvla/bin/python}
TASK_CONFIG=${TASK_CONFIG:-info_gathering_randomized}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
EVAL_SEED_LIST_ROOT=${EVAL_SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}
EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}

failures=0
warnings=0
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }
require_file() { [[ -f "$1" ]] && ok "file exists: $1" || fail "missing file: $1"; }
require_dir() { [[ -d "$1" ]] && ok "dir exists: $1" || fail "missing dir: $1"; }

printf '# HIF-VLA eval check\nrun_root: %s\ncheckpoint_key: %s\n\n' "${RUN_ROOT}" "${HIFVLA_CKPT_KEY}"

for path in     "${SCRIPT_DIR}/__init__.py"     "${SCRIPT_DIR}/deploy_policy.py"     "${SCRIPT_DIR}/deploy_policy.yml"     "${SCRIPT_DIR}/serve_policy.py"     "${SCRIPT_DIR}/ckpt_mapping.yaml"     "${SCRIPT_DIR}/eval.sh"     "${SCRIPT_DIR}/eval_seed_whitelist_randomized.sh"     "${SCRIPT_DIR}/install_env.sh"     "${SCRIPT_DIR}/README.md"     "${WHITELIST}"     "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"; do
    require_file "${path}"
done
require_dir "${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}"

if [[ ! "${EVAL_TEST_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    fail "EVAL_TEST_NUM must be a positive integer: ${EVAL_TEST_NUM}"
fi

if [[ -x "${ROBOTWIN_PYTHON}" ]]; then
    ok "ROBOTWIN_PYTHON executable: ${ROBOTWIN_PYTHON}"
    if "${ROBOTWIN_PYTHON}" - <<'PY'
import importlib
for name in ("numpy", "yaml", "PIL", "requests", "sapien"):
    importlib.import_module(name)
PY
    then
        ok "RoboTwin client imports pass"
    else
        fail "RoboTwin client imports failed"
    fi
    if "${ROBOTWIN_PYTHON}" -m py_compile         "${SCRIPT_DIR}/__init__.py"         "${SCRIPT_DIR}/deploy_policy.py"         "${SCRIPT_DIR}/serve_policy.py"; then
        ok "HIF-VLA Python files compile"
    else
        fail "HIF-VLA Python compilation failed"
    fi
else
    fail "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}"
fi

RESOLVED_JSON="${RUN_ROOT}/resolved_checkpoint.json"
if [[ -x "${ROBOTWIN_PYTHON}" && -f "${CKPT_MAPPING}" ]]; then
    if "${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${HIFVLA_CKPT_KEY}" "${RESOLVED_JSON}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

mapping_path, key, output_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
if key in (payload.get("unavailable") or {}):
    raise SystemExit(f"checkpoint unavailable: {key}: {payload['unavailable'][key]}")
record = dict((payload.get("checkpoints") or {})[key])
required = {
    "checkpoint_dir", "base_checkpoint", "hifvla_root", "unnorm_key",
    "step", "stage", "history_length", "action_dim", "action_horizon",
}
missing = sorted(required - set(record))
if missing:
    raise SystemExit(f"mapping record misses fields: {missing}")
output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
PY
    then
        ok "checkpoint mapping resolves"
    else
        fail "checkpoint mapping failed for ${HIFVLA_CKPT_KEY}"
    fi
fi

read_record_field() {
    "${ROBOTWIN_PYTHON}" - "${RESOLVED_JSON}" "$1" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[sys.argv[2]])
PY
}

if [[ -f "${RESOLVED_JSON}" ]]; then
    CHECKPOINT_DIR="$(read_record_field checkpoint_dir)"
    BASE_CHECKPOINT="$(read_record_field base_checkpoint)"
    HIFVLA_ROOT="$(read_record_field hifvla_root)"
    UNNORM_KEY="$(read_record_field unnorm_key)"
    require_dir "${CHECKPOINT_DIR}"
    require_dir "${BASE_CHECKPOINT}"
    require_dir "${HIFVLA_ROOT}"

    if "${ROBOTWIN_PYTHON}" "${SCRIPT_DIR}/serve_policy.py"         --platform astribot         --base-checkpoint "${BASE_CHECKPOINT}"         --checkpoint-dir "${CHECKPOINT_DIR}"         --hifvla-root "${HIFVLA_ROOT}"         --unnorm-key "${UNNORM_KEY}"         --dry-run > "${RUN_ROOT}/server_dry_run.json"; then
        ok "base model, checkpoint components, shards, and stats pass"
    else
        fail "server dry-run validation failed"
    fi

    if "${ROBOTWIN_PYTHON}" - "${RESOLVED_JSON}" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record["action_dim"] != 18:
    raise SystemExit("action_dim must be 18")
if record["action_horizon"] != 8:
    raise SystemExit("action_horizon must be 8")
if record["history_length"] != 8:
    raise SystemExit("history_length must be 8")
PY
    then
        ok "mapping dimensions are Astribot-compatible"
    else
        fail "mapping dimensions are invalid"
    fi
fi

if [[ -x "${HIFVLA_PYTHON}" ]]; then
    ok "HIFVLA_PYTHON executable: ${HIFVLA_PYTHON}"
    if PYTHONPATH="${HIFVLA_ROOT:-}:${PYTHONPATH:-}" "${HIFVLA_PYTHON}" - astribot <<'PY'
import importlib
modules = [
    "torch", "torchvision", "transformers", "peft", "draccus", "fastapi",
    "uvicorn", "tensorflow", "cv2", "av", "json_numpy",
]
for name in modules:
    importlib.import_module(name)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
if (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) != (18, 18, 8):
    raise SystemExit(
        f"constants mismatch: action={ACTION_DIM}, proprio={PROPRIO_DIM}, chunk={NUM_ACTIONS_CHUNK}"
    )
from experiments.robot.openvla_utils import (
    get_action_head,
    get_hismotion_encoder,
    get_motion_manager,
    get_proprio_projector,
    get_vla_action,
)
from experiments.robot.robot_utils import extract_motion_vectors_from_images
PY
    then
        ok "HIF-VLA server imports and Astribot constants pass"
    elif [[ "${ALLOW_MISSING_ENV}" == "1" ]]; then
        warn "HIF-VLA environment exists but imports/constants failed"
    else
        fail "HIF-VLA server imports or Astribot constants failed"
    fi
else
    if [[ "${ALLOW_MISSING_ENV}" == "1" ]]; then
        warn "HIFVLA_PYTHON is not executable: ${HIFVLA_PYTHON}"
    else
        fail "HIFVLA_PYTHON is not executable: ${HIFVLA_PYTHON}"
    fi
fi

if [[ -x "${ROBOTWIN_PYTHON}" && -f "${WHITELIST}" && -d "${EVAL_SEED_LIST_ROOT}/${TASK_CONFIG}" ]]; then
    if "${ROBOTWIN_PYTHON}" - "${WHITELIST}" "${EVAL_SEED_LIST_ROOT}" "${TASK_CONFIG}" "${EVAL_TEST_NUM}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

tasks = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
root, task_config, need = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
if not isinstance(tasks, list) or not tasks or not all(isinstance(x, str) and x for x in tasks):
    raise SystemExit("whitelist must be a non-empty list of task names")
errors = []
for task in tasks:
    path = root / task_config / f"{task}.json"
    if not path.is_file():
        errors.append(f"missing: {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload.get("seeds", [])) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or len(entries) < need:
        count = len(entries) if isinstance(entries, list) else 0
        errors.append(f"needs {need} seeds: {path} (found {count})")
if errors:
    raise SystemExit("\n".join(errors))
print(f"tasks={len(tasks)} episodes_per_task={need}")
PY
    then
        ok "task whitelist and seed lists pass"
    else
        fail "task whitelist or seed lists invalid"
    fi
fi

if [[ -x "${ROBOTWIN_PYTHON}" && -f "${SCRIPT_DIR}/deploy_policy.yml" ]]; then
    if "${ROBOTWIN_PYTHON}" - "${SCRIPT_DIR}/deploy_policy.yml" <<'PY'
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
expected = {
    "policy_name": "HIF-VLA",
    "action_dim": 18,
    "action_horizon": 8,
    "max_actions_per_call": 8,
    "history_length": 8,
    "action_semantics": "absolute",
}
for key, value in expected.items():
    if cfg.get(key) != value:
        raise SystemExit(f"{key} must be {value!r}, got {cfg.get(key)!r}")
PY
    then
        ok "deploy policy contract passes"
    else
        fail "deploy policy contract failed"
    fi
fi

printf '\nsummary: failures=%s warnings=%s log=%s\n' "${failures}" "${warnings}" "${LOG_FILE}"
(( failures == 0 ))
