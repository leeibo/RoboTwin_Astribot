#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${SCRIPT_DIR}/runs/check/${RUN_ID}"
mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/check.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
ALLOW_MISSING_ENV=${ALLOW_MISSING_ENV:-0}
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
EVAL_TEST_NUM=${EVAL_TEST_NUM:-1}
failures=0
warnings=0
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }
require_file() { [[ -f "$1" ]] && ok "file exists: $1" || fail "missing file: $1"; }
require_dir() { [[ -d "$1" ]] && ok "dir exists: $1" || fail "missing dir: $1"; }
printf '# PI05 eval check\nrun_root: %s\ncheckpoint_key: %s\n\n' "${RUN_ROOT}" "${PI05_CKPT_KEY}"
require_dir "${ROBOTWIN_ROOT}"
require_dir "${SCRIPT_DIR}"
require_file "${SCRIPT_DIR}/deploy_policy.py"
require_file "${SCRIPT_DIR}/deploy_policy.yml"
require_file "${SCRIPT_DIR}/ckpt_mapping.yaml"
require_file "${SCRIPT_DIR}/eval.sh"
require_file "${SCRIPT_DIR}/eval_seed_whitelist_randomized.sh"
require_file "${SCRIPT_DIR}/install_env.sh"
require_file "${WHITELIST}"
require_dir "${SEED_LIST_ROOT}/${TASK_CONFIG}"
require_dir "${ROBOTWIN_CLIENT_DEPS}"
require_dir "${PI05_RLINF_ROOT}"
require_dir "${OPENPI_CLIENT_SRC}"
require_file "${OPENPI_CLIENT_SRC}/openpi_client/websocket_client_policy.py"
require_file "${OPENPI_CLIENT_SRC}/openpi_client/msgpack_numpy.py"
if [[ -x "${ROBOTWIN_PYTHON}" ]]; then
    ok "ROBOTWIN_PYTHON executable: ${ROBOTWIN_PYTHON}"
    if PYTHONPATH="${ROBOTWIN_CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" - <<'PYCHECK'
import importlib
mods = ["numpy", "yaml", "PIL", "sapien", "requests"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
PYCHECK
    then ok "ROBOTWIN_PYTHON core imports pass"; else fail "ROBOTWIN_PYTHON missing core imports"; fi
    if PYTHONPATH="${ROBOTWIN_CLIENT_DEPS}:${OPENPI_CLIENT_SRC}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" - <<'PYCHECK'
import importlib
mods = ["msgpack", "websockets", "openpi_client.msgpack_numpy", "openpi_client.websocket_client_policy"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
PYCHECK
    then ok "ROBOTWIN_PYTHON websocket client imports pass"; else
        if [[ "${ALLOW_MISSING_ENV}" == "1" ]]; then warn "ROBOTWIN_PYTHON missing websocket client deps"; else fail "ROBOTWIN_PYTHON missing websocket client deps"; fi
    fi
else
    fail "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}"
fi
if [[ -x "${PI05_PYTHON}" ]]; then
    ok "PI05_PYTHON executable: ${PI05_PYTHON}"
    if PYTHONPATH="${PI05_RLINF_ROOT}:${PYTHONPATH:-}" "${PI05_PYTHON}" - <<'PYCHECK'
import importlib
mods = ["torch", "tyro", "openpi", "rlinf", "websockets"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
PYCHECK
    then ok "PI05_PYTHON server imports pass"; else
        if [[ "${ALLOW_MISSING_ENV}" == "1" ]]; then warn "PI05_PYTHON missing server deps"; else fail "PI05_PYTHON missing server deps"; fi
    fi
else
    if [[ "${ALLOW_MISSING_ENV}" == "1" ]]; then warn "PI05_PYTHON is not executable: ${PI05_PYTHON}"; else fail "PI05_PYTHON is not executable: ${PI05_PYTHON}"; fi
fi
if [[ -x "${ROBOTWIN_PYTHON}" ]]; then
    tmp_json="${RUN_ROOT}/resolved_checkpoint.json"
    if "${ROBOTWIN_PYTHON}" - "${CKPT_MAPPING}" "${PI05_CKPT_KEY}" "${tmp_json}" <<'PYMAP'
import json, sys, yaml
from pathlib import Path
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
record = dict((payload.get('checkpoints') or {})[sys.argv[2]])
Path(sys.argv[3]).write_text(json.dumps(record), encoding='utf-8')
PYMAP
    then
        ok "checkpoint mapping resolves"
        checkpoint_dir="$(${ROBOTWIN_PYTHON} - "${tmp_json}" <<'PYREAD'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['checkpoint_dir'])
PYREAD
)"
        require_dir "${checkpoint_dir}"
        while IFS= read -r rel; do [[ -z "${rel}" ]] || require_file "${checkpoint_dir}/${rel}"; done < <(${ROBOTWIN_PYTHON} - "${tmp_json}" <<'PYREAD'
import json, sys
from pathlib import Path
for item in json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('required_files', []): print(item)
PYREAD
)
    else
        fail "checkpoint mapping failed for ${PI05_CKPT_KEY}"
    fi
fi
if [[ -f "${WHITELIST}" && -d "${SEED_LIST_ROOT}/${TASK_CONFIG}" && -x "${ROBOTWIN_PYTHON}" ]]; then
    if "${ROBOTWIN_PYTHON}" - "${WHITELIST}" "${SEED_LIST_ROOT}" "${TASK_CONFIG}" "${EVAL_TEST_NUM}" <<'PYSEED'
import json, sys, yaml
from pathlib import Path
items = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
root = Path(sys.argv[2]); cfg = sys.argv[3]; need = int(sys.argv[4])
if not isinstance(items, list) or not items: raise SystemExit('whitelist invalid')
errors = []
for task in items:
    path = root / cfg / f'{task}.json'
    if not path.exists(): errors.append(str(path)); continue
    payload = json.loads(path.read_text(encoding='utf-8'))
    entries = payload.get('entries', payload.get('seeds', [])) if isinstance(payload, dict) else payload
    if len(entries) < need: errors.append(f'{path} has {len(entries)} < {need}')
if errors: raise SystemExit('\n'.join(errors))
print(f'tasks={len(items)} seed_lists_ok_for={need}')
PYSEED
    then ok "task whitelist and seed lists pass"; else fail "task whitelist or seed lists invalid"; fi
fi
printf '\nsummary: failures=%s warnings=%s log=%s\n' "${failures}" "${warnings}" "${LOG_FILE}"
(( failures == 0 ))
