#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/root/autodl-tmp/conda_env/RoboTwin/bin/python}
CLIENT_DEPS=${CLIENT_DEPS:-${SCRIPT_DIR}/env/robotwin_client_deps}
DRY_RUN=${DRY_RUN:-0}
USE_CN_MIRRORS=${USE_CN_MIRRORS:-1}
PI05_PIP_INDEX_URL=${PI05_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}

usage() {
    cat <<'EOF'
Usage: bash policy/PI05/install_env.sh

Installs the lightweight websocket dependencies used by the RoboTwin client.
The PI05/RLinf model environment is managed by the external PI05 project.

Environment variables:
  ROBOTWIN_PYTHON    RoboTwin interpreter
  CLIENT_DEPS        Target directory (default: policy/PI05/env/robotwin_client_deps)
  DRY_RUN            Print commands without changing files (default: 0)
  USE_CN_MIRRORS     Use the configured mirror (default: 1)
  PI05_PIP_INDEX_URL Python package index (default: Aliyun)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
(( $# == 0 )) || { usage >&2; exit 2; }
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "${USE_CN_MIRRORS}" == 0 || "${USE_CN_MIRRORS}" == 1 ]] || {
    echo "USE_CN_MIRRORS must be 0 or 1" >&2
    exit 2
}
[[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}" >&2; exit 1; }

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    [[ "${DRY_RUN}" == 1 ]] || "$@"
}

pip_args=(--upgrade --target "${CLIENT_DEPS}" "websockets>=14,<17" "msgpack>=1.0,<2")
if [[ "${USE_CN_MIRRORS}" == 1 ]]; then
    pip_args+=(--index-url "${PI05_PIP_INDEX_URL}")
fi

run mkdir -p "${CLIENT_DEPS}"
run "${ROBOTWIN_PYTHON}" -m pip install "${pip_args[@]}"

if [[ "${DRY_RUN}" == 1 ]]; then
    echo "PI05 client dependency dry-run complete."
    exit 0
fi

PYTHONPATH="${CLIENT_DEPS}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" - <<'PY'
import msgpack
import websockets

print("PI05 RoboTwin client dependencies: OK")
PY

echo "Client dependencies ready: ${CLIENT_DEPS}"
echo "Run: bash ${SCRIPT_DIR}/check.sh"
