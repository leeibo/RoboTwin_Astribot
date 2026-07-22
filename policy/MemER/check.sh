#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEMER_ROOT=${MEMER_ROOT:-/root/autodl-tmp/MEMER_eval/MemER}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
HIGH_PYTHON=${HIGH_PYTHON:-${MEMER_ROOT}/env/qwen3vl/bin/python}
HIGH_PROCESSOR_DIR=${HIGH_PROCESSOR_DIR:-}
LOW_PYTHON=${LOW_PYTHON:-${MEMER_ROOT}/env/rlinf-pi05/bin/python}
CLIENT_DEPS=${CLIENT_DEPS:-${SCRIPT_DIR}/env/robotwin_client_deps}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-${MEMER_ROOT}/source/low_level/openpi/packages/openpi-client/src}
CKPT_MAPPING=${CKPT_MAPPING:-${SCRIPT_DIR}/ckpt_mapping.yaml}
MEMER_CKPT_KEY=${MEMER_CKPT_KEY:-memer_astribot_step_4500_18000}
WHITELIST=${WHITELIST:-${ROBOTWIN_ROOT}/task_config/eval_seed_task_whitelist.yml}
SEED_LIST_ROOT=${SEED_LIST_ROOT:-${ROBOTWIN_ROOT}/eval_seed_lists}
EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}
ALLOW_MISSING_ENV=${ALLOW_MISSING_ENV:-0}
CHECK_PORTS=${CHECK_PORTS:-1}
HIGH_PORT=${HIGH_PORT:-5901}
LOW_PORTS=${LOW_PORTS:-5902,5910,5911,5912,5913,5951,5960,5961,5962}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
RUN_ROOT="${SCRIPT_DIR}/runs/check/${RUN_ID}"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/check.log") 2>&1
failures=0; warnings=0
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings+1)); }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures+1)); }
file() { [[ -f "$1" ]] && ok "file: $1" || fail "missing file: $1"; }
dir() { [[ -d "$1" ]] && ok "directory: $1" || fail "missing directory: $1"; }

for path in __init__.py ckpt_mapping.yaml deploy_policy.yml deploy_policy.py serve_high_policy.py serve_low_policy.py \
  install_env.sh check.sh eval.sh eval_seed_whitelist_randomized.sh eval_all.sh README.md tests/test_memer_eval.py; do file "${SCRIPT_DIR}/${path}"; done
file "${WHITELIST}"
dir "${SEED_LIST_ROOT}/info_gathering_demo"
dir "${SEED_LIST_ROOT}/info_gathering_randomized"

if [[ -x "${ROBOTWIN_PYTHON}" ]]; then
  ok "RoboTwin Python: ${ROBOTWIN_PYTHON}"
  PYTHONPATH="${CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" -m py_compile \
    "${SCRIPT_DIR}/__init__.py" "${SCRIPT_DIR}/deploy_policy.py" "${SCRIPT_DIR}/serve_high_policy.py" "${SCRIPT_DIR}/serve_low_policy.py" \
    && ok "Python compilation" || fail "Python compilation"
  if PYTHONPATH="${CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" - <<'PY'
import numpy, requests, yaml, PIL, sapien, websockets, msgpack
from openpi_client import msgpack_numpy
PY
  then
    ok "RoboTwin and websocket client imports"
  elif [[ "${ALLOW_MISSING_ENV}" == 1 ]]; then
    warn "RoboTwin websocket client dependencies are not installed"
  else
    fail "RoboTwin/websocket client imports"
  fi
else
  fail "ROBOTWIN_PYTHON not executable: ${ROBOTWIN_PYTHON}"
fi

"${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${MEMER_CKPT_KEY}" "${RUN_ROOT}/resolved_checkpoint.json" <<'PY' \
  && ok "checkpoint mapping, weights, stats, and 18/50 contract" || fail "checkpoint contract"
import json, sys, yaml
from pathlib import Path
record = (yaml.safe_load(Path(sys.argv[1]).read_text()) or {})["checkpoints"][sys.argv[2]]
assert record["action_dim"] == 18 and record["action_horizon"] == 50
for group, root_key in (
    ("required_high_files", "high_checkpoint_dir"),
    ("required_high_processor_files", "high_checkpoint_dir"),
    ("required_low_files", "low_checkpoint_dir"),
):
    root = Path(record[root_key]); assert root.is_dir(), root
    for rel in record[group]: assert (root / rel).is_file(), root / rel
processor_path = Path(record["high_checkpoint_dir"]) / "preprocessor_config.json"
processor = json.loads(processor_path.read_text())
size = processor.get("size") or {}
min_pixels = processor.get("min_pixels", size.get("shortest_edge"))
max_pixels = processor.get("max_pixels", size.get("longest_edge"))
assert (min_pixels, max_pixels) == (50176, 115200), (min_pixels, max_pixels)
stats_path = Path(record["low_checkpoint_dir"]) / "robotwin_astribot_pi05_subtask/norm_stats.json"
stats = json.loads(stats_path.read_text())["norm_stats"]
for field in ("state", "actions"):
    for stat in ("mean", "std", "q01", "q99"): assert len(stats[field][stat]) == 32, (field, stat)
Path(sys.argv[3]).write_text(json.dumps(record, indent=2))
PY

if [[ -f "${RUN_ROOT}/resolved_checkpoint.json" ]]; then
  HIGH_CHECKPOINT=$("${ROBOTWIN_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["high_checkpoint_dir"])' "${RUN_ROOT}/resolved_checkpoint.json")
  HIGH_PROCESSOR_DIR=${HIGH_PROCESSOR_DIR:-${HIGH_CHECKPOINT}}
  LOW_CHECKPOINT=$("${ROBOTWIN_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["low_checkpoint_dir"])' "${RUN_ROOT}/resolved_checkpoint.json")
  RLINF_ROOT=$("${ROBOTWIN_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rlinf_root"])' "${RUN_ROOT}/resolved_checkpoint.json")
  "${ROBOTWIN_PYTHON}" "${SCRIPT_DIR}/serve_high_policy.py" --checkpoint "${HIGH_CHECKPOINT}" \
    --processor-dir "${HIGH_PROCESSOR_DIR}" --dry-run \
    > "${RUN_ROOT}/high_server_dry_run.json" && ok "high server dry-run" || fail "high server dry-run"
  "${ROBOTWIN_PYTHON}" "${SCRIPT_DIR}/serve_low_policy.py" --checkpoint "${LOW_CHECKPOINT}" \
    --rlinf-root "${RLINF_ROOT}" --openpi-client-src "${OPENPI_CLIENT_SRC}" --dry-run \
    > "${RUN_ROOT}/low_server_dry_run.json" && ok "low server dry-run" || fail "low server dry-run"
fi

if [[ -x "${HIGH_PYTHON}" ]]; then
  "${HIGH_PYTHON}" - <<'PY' && ok "high environment transformers 4.57.0" || fail "high environment"
import torch, transformers
assert transformers.__version__ == "4.57.0"
from transformers import Qwen3VLForConditionalGeneration
PY
else
  [[ "${ALLOW_MISSING_ENV}" == 1 ]] && warn "HIGH_PYTHON missing: ${HIGH_PYTHON}" || fail "HIGH_PYTHON missing: ${HIGH_PYTHON}"
fi
if [[ -x "${LOW_PYTHON}" ]]; then
  PYTHONPATH="${MEMER_ROOT}/source/low_level/RLinf" "${LOW_PYTHON}" - <<'PY' \
    && ok "low environment transformers 4.53.2 and OpenPI patch" || fail "low environment"
import transformers
assert transformers.__version__ == "4.53.2"
import msgpack, websockets
from rlinf.models.embodiment.openpi import _transformers_replace_is_installed
assert _transformers_replace_is_installed()
PY
else
  [[ "${ALLOW_MISSING_ENV}" == 1 ]] && warn "LOW_PYTHON missing: ${LOW_PYTHON}" || fail "LOW_PYTHON missing: ${LOW_PYTHON}"
fi

for config in info_gathering_demo info_gathering_randomized; do
  "${ROBOTWIN_PYTHON}" - "${WHITELIST}" "${SEED_LIST_ROOT}" "${config}" "${EVAL_TEST_NUM}" <<'PY' \
    && ok "whitelist seeds: ${config}" || fail "whitelist seeds: ${config}"
import json, sys, yaml
from pathlib import Path
tasks = yaml.safe_load(Path(sys.argv[1]).read_text()); root = Path(sys.argv[2]); config = sys.argv[3]; need = int(sys.argv[4])
assert len(tasks) == 35, len(tasks)
for task in tasks:
    value = json.loads((root / config / f"{task}.json").read_text())
    rows = value.get("entries", value.get("seeds", [])) if isinstance(value, dict) else value
    assert len(rows) >= need, (task, len(rows), need)
PY
done

"${ROBOTWIN_PYTHON}" - "${SCRIPT_DIR}/deploy_policy.yml" <<'PY' \
  && ok "deploy timing/image/action contract" || fail "deploy contract"
import sys, yaml
from pathlib import Path
config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
expected = {
    "policy_name": "MemER", "action_dim": 18, "action_horizon": 50,
    "low_level_execution_horizon": 5, "high_level_replan_interval": 5,
    "replan_on_episode_start": True, "high_image_width": 320,
    "high_image_height": 180, "action_semantics": "absolute",
}
for key, value in expected.items():
    if config.get(key) != value: raise SystemExit(f"{key}: {config.get(key)!r} != {value!r}")
PY

bash -n "${SCRIPT_DIR}/install_env.sh" "${SCRIPT_DIR}/check.sh" "${SCRIPT_DIR}/eval.sh" \
  "${SCRIPT_DIR}/eval_seed_whitelist_randomized.sh" "${SCRIPT_DIR}/eval_all.sh" \
  && ok "shell syntax" || fail "shell syntax"

if [[ "${CHECK_PORTS}" == 1 ]]; then
  "${ROBOTWIN_PYTHON}" - "${HIGH_PORT}" "${LOW_PORTS}" <<'PY' && ok "configured ports are free" || fail "configured ports are occupied"
import socket, sys
ports = [int(sys.argv[1]), *(int(x) for x in sys.argv[2].split(","))]
for port in ports:
    sock = socket.socket(); sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: sock.bind(("127.0.0.1", port))
    except OSError as exc: raise SystemExit(f"port {port}: {exc}")
    finally: sock.close()
PY
fi

PYTHONPATH="${CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" -m unittest discover \
  -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v && ok "mock/unit tests" || fail "mock/unit tests"

printf '\nsummary: failures=%s warnings=%s log=%s\n' "${failures}" "${warnings}" "${RUN_ROOT}/check.log"
(( failures == 0 ))
