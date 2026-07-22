#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMER_ROOT=${MEMER_ROOT:-/root/autodl-tmp/MEMER_eval/MemER}
HIGH_ENV=${HIGH_ENV:-${MEMER_ROOT}/env/qwen3vl}
LOW_ENV=${LOW_ENV:-${MEMER_ROOT}/env/rlinf-pi05}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
CLIENT_DEPS=${CLIENT_DEPS:-${SCRIPT_DIR}/env/robotwin_client_deps}
PYTHON_VERSION=${PYTHON_VERSION:-3.11.14}
ENV_TOOL=${ENV_TOOL:-uv}
INSTALL_HIGH=${INSTALL_HIGH:-1}
INSTALL_LOW=${INSTALL_LOW:-1}
INSTALL_CLIENT=${INSTALL_CLIENT:-1}
OVERWRITE_HIGH=${OVERWRITE_HIGH:-0}
DRY_RUN=${DRY_RUN:-0}
USE_CN_MIRRORS=${USE_CN_MIRRORS:-1}
MEMER_PIP_INDEX_URL=${MEMER_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}
MEMER_HF_ENDPOINT=${MEMER_HF_ENDPOINT:-https://hf-mirror.com}
MEMER_GITHUB_PREFIX=${MEMER_GITHUB_PREFIX:-https://ghfast.top/}
MEMER_PYTHON_MIRROR=${MEMER_PYTHON_MIRROR:-https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download}
MEMER_PYTHON_INSTALL_DIR=${MEMER_PYTHON_INSTALL_DIR:-${MEMER_ROOT}/cache/uv/python}
REUSE_PI05_LOW_ENV=${REUSE_PI05_LOW_ENV:-auto}
PI05_LOW_ENV=${PI05_LOW_ENV:-/root/autodl-tmp/PI05_eval/PI05/env/rlinf-pi05}
LOW_UV_CACHE_DIR=${LOW_UV_CACHE_DIR:-/root/autodl-tmp/PI05_eval/PI05/cache/uv}

if [[ "${USE_CN_MIRRORS}" == 1 ]]; then
  export UV_DEFAULT_INDEX="${MEMER_PIP_INDEX_URL}"
  export PIP_INDEX_URL="${MEMER_PIP_INDEX_URL}"
  export HF_ENDPOINT="${MEMER_HF_ENDPOINT}"
  export UV_PYTHON_INSTALL_MIRROR="${MEMER_PYTHON_MIRROR}"
  ACTIVE_GITHUB_PREFIX=${MEMER_GITHUB_PREFIX}
  FLASH_ATTN_URL="${ACTIVE_GITHUB_PREFIX}https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
else
  export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.org/simple}
  export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
  export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
  export UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR:-https://github.com/astral-sh/python-build-standalone/releases/download}
  ACTIVE_GITHUB_PREFIX=
  FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-${MEMER_ROOT}/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-${MEMER_PYTHON_INSTALL_DIR}}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${MEMER_ROOT}/cache/pip}
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120}
export PIP_RETRIES=${PIP_RETRIES:-10}
export PIP_DISABLE_PIP_VERSION_CHECK=${PIP_DISABLE_PIP_VERSION_CHECK:-1}
export HF_HOME=${HF_HOME:-${MEMER_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

usage() {
  cat <<'EOF'
Usage: bash policy/MemER/install_env.sh

Builds two isolated environments because Qwen3-VL requires transformers 4.57.0
while RLinf/OpenPI requires its patched transformers 4.53.2.

Key variables:
  HIGH_ENV, LOW_ENV             target environment prefixes
  INSTALL_HIGH/LOW/CLIENT       enable each installation stage (default 1)
  OVERWRITE_HIGH                rebuild an existing high environment
  DRY_RUN                       print commands only
  USE_CN_MIRRORS                use domestic mirrors (default 1)
  MEMER_PIP_INDEX_URL           PyPI mirror (default Aliyun)
  MEMER_HF_ENDPOINT             Hugging Face mirror (default hf-mirror.com)
  MEMER_PYTHON_MIRROR           uv managed-Python mirror (default ghfast)
  MEMER_GITHUB_PREFIX           GitHub release prefix (default ghfast)
  REUSE_PI05_LOW_ENV            auto, 1, or 0 (default auto)
  PI05_LOW_ENV                  compatible PI05 environment used in reuse mode
  LOW_UV_CACHE_DIR              cache for standalone low install
  LOW_SETUP_ARGS                extra arguments passed to RLinf installer
EOF
}
[[ "${1:-}" != -h && "${1:-}" != --help ]] || { usage; exit 0; }
(( $# == 0 )) || { usage >&2; exit 2; }
[[ -d "${MEMER_ROOT}" ]] || { echo "MEMER_ROOT missing: ${MEMER_ROOT}" >&2; exit 1; }
[[ "${USE_CN_MIRRORS}" == 0 || "${USE_CN_MIRRORS}" == 1 ]] || { echo "USE_CN_MIRRORS must be 0 or 1" >&2; exit 2; }
[[ "${REUSE_PI05_LOW_ENV}" == auto || "${REUSE_PI05_LOW_ENV}" == 0 || "${REUSE_PI05_LOW_ENV}" == 1 ]] || {
  echo "REUSE_PI05_LOW_ENV must be auto, 0, or 1" >&2
  exit 2
}

printf 'mirror mode: %s\n' "${USE_CN_MIRRORS}"
printf 'pip/uv index: %s\n' "${UV_DEFAULT_INDEX}"
printf 'uv Python mirror: %s\n' "${UV_PYTHON_INSTALL_MIRROR:-official GitHub}"
printf 'Hugging Face endpoint: %s\n' "${HF_ENDPOINT}"
printf 'GitHub release prefix: %s\n' "${ACTIVE_GITHUB_PREFIX:-none}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "${DRY_RUN}" == 1 ]] || "$@"
}

low_env_compatible() {
  local candidate=$1
  [[ -x "${candidate}/bin/python" ]] || return 1
  PYTHONNOUSERSITE=1 PYTHONPATH="${MEMER_ROOT}/source/low_level/RLinf" \
    "${candidate}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
import transformers

assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
assert transformers.__version__ == "4.53.2"
from rlinf.models.embodiment.openpi import _transformers_replace_is_installed
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

assert _transformers_replace_is_installed()
config = get_openpi_config("pi05_astribot_subtask", model_path="unused", assets_dir="/tmp/assets")
assert config.model.action_horizon == 50
assert config.data.repo_id == "robotwin_astribot_pi05_subtask"
PY
}

if [[ "${INSTALL_HIGH}" == 1 ]]; then
  command -v "${ENV_TOOL}" >/dev/null 2>&1 || { echo "${ENV_TOOL} is required for the high environment" >&2; exit 1; }
  [[ "${ENV_TOOL}" == uv ]] || { echo "ENV_TOOL currently supports only uv" >&2; exit 2; }
  python_install_cmd=("${ENV_TOOL}" python install --install-dir "${UV_PYTHON_INSTALL_DIR}")
  [[ -n "${UV_PYTHON_INSTALL_MIRROR}" ]] && python_install_cmd+=(--mirror "${UV_PYTHON_INSTALL_MIRROR}")
  python_install_cmd+=("${PYTHON_VERSION}")
  run "${python_install_cmd[@]}"
  if [[ "${OVERWRITE_HIGH}" == 1 && -e "${HIGH_ENV}" ]]; then
    run "${ENV_TOOL}" venv --clear --python "${PYTHON_VERSION}" "${HIGH_ENV}"
  elif [[ -e "${HIGH_ENV}" && ! -x "${HIGH_ENV}/bin/python" ]]; then
    run "${ENV_TOOL}" venv --clear --python "${PYTHON_VERSION}" "${HIGH_ENV}"
  elif [[ ! -x "${HIGH_ENV}/bin/python" ]]; then
    run "${ENV_TOOL}" venv --python "${PYTHON_VERSION}" "${HIGH_ENV}"
  fi
  run "${ENV_TOOL}" pip install --python "${HIGH_ENV}/bin/python" \
    torch==2.6.0 torchvision==0.21.0 triton==3.2.0 transformers==4.57.0 \
    accelerate==1.7.0 peft==0.17.1 torchcodec==0.2.0 wandb==0.28.0 setuptools==83.0.0 \
    "flash-attn @ ${FLASH_ATTN_URL}"
  run "${ENV_TOOL}" pip install --python "${HIGH_ENV}/bin/python" -e "${MEMER_ROOT}/source/qwen3vl/qwen-vl-utils"
fi

if [[ "${INSTALL_LOW}" == 1 ]]; then
  if low_env_compatible "${LOW_ENV}"; then
    echo "Compatible MemER low environment already exists: ${LOW_ENV}"
  elif [[ "${REUSE_PI05_LOW_ENV}" != 0 ]] && low_env_compatible "${PI05_LOW_ENV}"; then
    echo "Reusing compatible PI05 low environment: ${PI05_LOW_ENV}"
    run rm -rf "${LOW_ENV}"
    run ln -s "${PI05_LOW_ENV}" "${LOW_ENV}"
  elif [[ "${REUSE_PI05_LOW_ENV}" == 1 ]]; then
    echo "REUSE_PI05_LOW_ENV=1 but the environment is incompatible: ${PI05_LOW_ENV}" >&2
    exit 1
  elif [[ "${DRY_RUN}" == 1 ]]; then
    printf '+ UV_CACHE_DIR=%q LOW_LEVEL_ENV=%q bash %q' \
      "${LOW_UV_CACHE_DIR}" "${LOW_ENV}" "${MEMER_ROOT}/setup_low_level_env.sh"
    [[ "${USE_CN_MIRRORS}" == 1 ]] && printf ' --use-mirror'
    printf ' %s\n' "${LOW_SETUP_ARGS:-}"
  else
    read -r -a low_args <<< "${LOW_SETUP_ARGS:-}"
    [[ "${USE_CN_MIRRORS}" == 1 ]] && low_args=(--use-mirror "${low_args[@]}")
    UV_CACHE_DIR="${LOW_UV_CACHE_DIR}" LOW_LEVEL_ENV="${LOW_ENV}" \
      bash "${MEMER_ROOT}/setup_low_level_env.sh" "${low_args[@]}"
  fi
fi

if [[ "${INSTALL_CLIENT}" == 1 ]]; then
  [[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "ROBOTWIN_PYTHON missing: ${ROBOTWIN_PYTHON}" >&2; exit 1; }
  run mkdir -p "${CLIENT_DEPS}"
  run "${ROBOTWIN_PYTHON}" -m pip install --upgrade --target "${CLIENT_DEPS}" "websockets>=14,<17" "msgpack>=1.0,<2"
fi

if [[ "${DRY_RUN}" == 1 ]]; then echo "Environment installation dry-run complete."; exit 0; fi
if [[ "${INSTALL_HIGH}" == 1 ]]; then
"${HIGH_ENV}/bin/python" - <<'PY'
import torch, transformers
assert transformers.__version__ == "4.57.0", transformers.__version__
from transformers import Qwen3VLForConditionalGeneration
print("high:", torch.__version__, transformers.__version__, Qwen3VLForConditionalGeneration.__name__)
PY
fi
if [[ "${INSTALL_LOW}" == 1 ]]; then
PYTHONPATH="${MEMER_ROOT}/source/low_level/RLinf" "${LOW_ENV}/bin/python" - <<'PY'
import transformers
assert transformers.__version__ == "4.53.2", transformers.__version__
from rlinf.models.embodiment.openpi import _transformers_replace_is_installed
assert _transformers_replace_is_installed()
print("low: transformers", transformers.__version__, "OpenPI patch OK")
PY
fi
echo "MemER environments ready. Run: bash ${SCRIPT_DIR}/check.sh"
printf '\nHigh-policy shell:\n'
printf '  source %q\n' "${HIGH_ENV}/bin/activate"
printf '  export HIGH_PYTHON=%q\n' "${HIGH_ENV}/bin/python"
printf '  export PYTHONNOUSERSITE=1 HF_ENDPOINT=%q HF_HOME=%q TRANSFORMERS_CACHE=%q\n' \
  "${HF_ENDPOINT}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"
printf '\nLow-policy shell:\n'
printf '  source %q\n' "${LOW_ENV}/bin/activate"
printf '  export LOW_PYTHON=%q\n' "${LOW_ENV}/bin/python"
printf '  export PYTHONNOUSERSITE=1 PYTHONPATH=%q\n' "${MEMER_ROOT}/source/low_level/RLinf"
printf '  export OPENPI_DATA_HOME=%q HF_HOME=%q TRANSFORMERS_CACHE=%q\n' \
  "${MEMER_ROOT}/cache/openpi" "${HF_HOME}" "${TRANSFORMERS_CACHE}"
printf '\nEvaluation launcher shell (recommended; no venv activation required):\n'
printf '  export MEMER_ROOT=%q\n' "${MEMER_ROOT}"
printf '  export HIGH_PYTHON=%q\n' "${HIGH_ENV}/bin/python"
printf '  export LOW_PYTHON=%q\n' "${LOW_ENV}/bin/python"
printf '  export ROBOTWIN_PYTHON=%q\n' "${ROBOTWIN_PYTHON}"
printf '  export OPENPI_CLIENT_SRC=%q\n' "${MEMER_ROOT}/source/low_level/openpi/packages/openpi-client/src"
printf '  export PYTHONNOUSERSITE=1 HF_ENDPOINT=%q\n' "${HF_ENDPOINT}"
