#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash prepare_fastwam_local_runtime.sh

Builds a node-local FastWAM evaluation runtime. The conda environment, source
trees, and RoboTwin assets are copied locally; checkpoints remain on /private.

Environment variables:
  LOCAL_RUNTIME_ROOT       Destination root (default: /opt/zjb_fastwam_runtime)
  SOURCE_FASTWAM_ROOT      FastWAM source root
  SOURCE_FASTWAM_ENV       FastWAM conda environment
  SOURCE_ASSETS_ROOT       RoboTwin assets source
  CONDA_BIN                Conda executable (default: /opt/conda/bin/conda)
  LOCAL_ENV_CLONE_RETRIES  Conda clone attempts (default: 3)
  LOCAL_SYNC_RETRIES       Rsync attempts for transient EPC errors (default: 5)
  FORCE_LOCAL_ENV_REBUILD  Recreate the local conda environment (default: 0)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

SOURCE_ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FASTWAM_ROOT=${SOURCE_FASTWAM_ROOT:-/private/zjb/workspace/FastWAM}
SOURCE_FASTWAM_ENV=${SOURCE_FASTWAM_ENV:-/private/zjb/conda_envs/fastwam}
SOURCE_ASSETS_ROOT=${SOURCE_ASSETS_ROOT:-"$(readlink -f "${SOURCE_ROBOTWIN_ROOT}/assets")"}
LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT:-/opt/zjb_fastwam_runtime}
CONDA_BIN=${CONDA_BIN:-/opt/conda/bin/conda}
LOCAL_ENV_CLONE_RETRIES=${LOCAL_ENV_CLONE_RETRIES:-3}
LOCAL_SYNC_RETRIES=${LOCAL_SYNC_RETRIES:-5}
FORCE_LOCAL_ENV_REBUILD=${FORCE_LOCAL_ENV_REBUILD:-0}

LOCAL_ENV="${LOCAL_RUNTIME_ROOT}/env"
LOCAL_FASTWAM_ROOT="${LOCAL_RUNTIME_ROOT}/FastWAM"
LOCAL_ROBOTWIN_ROOT="${LOCAL_RUNTIME_ROOT}/RoboTwin_Astribot"
LOCAL_ASSETS_ROOT="${LOCAL_RUNTIME_ROOT}/assets"

if [[ "${LOCAL_RUNTIME_ROOT}" != /* || "${LOCAL_RUNTIME_ROOT}" == "/" ]]; then
    echo "LOCAL_RUNTIME_ROOT must be an absolute non-root path: ${LOCAL_RUNTIME_ROOT}" >&2
    exit 2
fi
for value_name in LOCAL_ENV_CLONE_RETRIES LOCAL_SYNC_RETRIES; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer: ${value}" >&2
        exit 2
    fi
done
for value_name in FORCE_LOCAL_ENV_REBUILD; do
    value="${!value_name}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${value_name} must be 0 or 1, got: ${value}" >&2
        exit 2
    fi
done

for command_path in "${CONDA_BIN}" "${SOURCE_FASTWAM_ENV}/bin/python"; do
    if [[ ! -x "${command_path}" ]]; then
        echo "Required executable does not exist: ${command_path}" >&2
        exit 1
    fi
done
for command_name in rsync readlink; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command was not found: ${command_name}" >&2
        exit 1
    fi
done
for source_path in "${SOURCE_FASTWAM_ROOT}" "${SOURCE_ROBOTWIN_ROOT}" "${SOURCE_ASSETS_ROOT}"; do
    if [[ ! -d "${source_path}" ]]; then
        echo "Required source directory does not exist: ${source_path}" >&2
        exit 1
    fi
done

mkdir -p "${LOCAL_RUNTIME_ROOT}"

rsync_with_retries() {
    local label="$1" attempt status
    shift
    for ((attempt = 1; attempt <= LOCAL_SYNC_RETRIES; attempt++)); do
        echo "[local-runtime] ${label} rsync ${attempt}/${LOCAL_SYNC_RETRIES}"
        if rsync "$@"; then
            return 0
        else
            status=$?
        fi
        echo "[local-runtime] ${label} rsync failed with exit ${status}" >&2
        if (( attempt < LOCAL_SYNC_RETRIES )); then
            sleep 2
        fi
    done
    return "${status}"
}

if [[ "${FORCE_LOCAL_ENV_REBUILD}" == "1" && -e "${LOCAL_ENV}" ]]; then
    case "${LOCAL_ENV}" in
        "${LOCAL_RUNTIME_ROOT}"/*) rm -rf -- "${LOCAL_ENV}" ;;
        *) echo "Refusing to remove unsafe local env path: ${LOCAL_ENV}" >&2; exit 1 ;;
    esac
fi

if [[ ! -x "${LOCAL_ENV}/bin/python" || ! -f "${LOCAL_ENV}/conda-meta/history" ]]; then
    clone_ok=0
    for ((attempt = 1; attempt <= LOCAL_ENV_CLONE_RETRIES; attempt++)); do
        echo "[local-runtime] conda clone ${attempt}/${LOCAL_ENV_CLONE_RETRIES}: ${LOCAL_ENV}"
        if [[ -e "${LOCAL_ENV}" ]]; then
            case "${LOCAL_ENV}" in
                "${LOCAL_RUNTIME_ROOT}"/*) rm -rf -- "${LOCAL_ENV}" ;;
                *) echo "Refusing to remove unsafe local env path: ${LOCAL_ENV}" >&2; exit 1 ;;
            esac
        fi
        if "${CONDA_BIN}" create -y --copy -p "${LOCAL_ENV}" --clone "${SOURCE_FASTWAM_ENV}"; then
            clone_ok=1
            break
        fi
    done
    if [[ "${clone_ok}" != "1" ]]; then
        echo "Failed to clone the FastWAM conda environment." >&2
        exit 1
    fi
else
    echo "[local-runtime] reusing local conda environment: ${LOCAL_ENV}"
fi

SITE_PACKAGES="${LOCAL_ENV}/lib/python3.10/site-packages"
printf '%s\n' "${LOCAL_FASTWAM_ROOT}/src" \
    > "${SITE_PACKAGES}/__editable__.fastwam-0.1.0.pth"
printf '%s\n' "${LOCAL_FASTWAM_ROOT}/third_party/curobo/src" \
    > "${SITE_PACKAGES}/__editable__.nvidia_curobo-0.0.0.pth"

echo "[local-runtime] syncing FastWAM source"
mkdir -p "${LOCAL_FASTWAM_ROOT}"
rsync_with_retries "FastWAM source" -a --delete --delete-excluded --info=stats2 \
    --exclude='.git/' \
    --exclude='.cache/' \
    --exclude='.tmp/' \
    --exclude='checkpoints' \
    --exclude='ckpt/' \
    --exclude='evaluate_results/' \
    --exclude='train_files/managed_runs/' \
    "${SOURCE_FASTWAM_ROOT}/" "${LOCAL_FASTWAM_ROOT}/"
ln -sfn "${SOURCE_FASTWAM_ROOT}/checkpoints" "${LOCAL_FASTWAM_ROOT}/checkpoints"

echo "[local-runtime] syncing RoboTwin source"
mkdir -p "${LOCAL_ROBOTWIN_ROOT}"
rsync_with_retries "RoboTwin source" -a --delete --delete-excluded --info=stats2 \
    --exclude='.git/' \
    --exclude='.agents/' \
    --exclude='.cache/' \
    --exclude='.progress/' \
    --exclude='assets' \
    --exclude='debug_data/' \
    --exclude='eval_result/' \
    --exclude='logs/' \
    --exclude='script/calibration/live_frame_records.jsonl' \
    "${SOURCE_ROBOTWIN_ROOT}/" "${LOCAL_ROBOTWIN_ROOT}/"
ln -sfn "${LOCAL_FASTWAM_ROOT}/experiments/robotwin/fastwam_policy" \
    "${LOCAL_ROBOTWIN_ROOT}/policy/fastwam_policy"

echo "[local-runtime] syncing RoboTwin assets"
mkdir -p "${LOCAL_ASSETS_ROOT}"
rsync_with_retries "RoboTwin assets" -a --delete --info=stats2 \
    "${SOURCE_ASSETS_ROOT}/" "${LOCAL_ASSETS_ROOT}/"
ln -sfn "${LOCAL_ASSETS_ROOT}" "${LOCAL_ROBOTWIN_ROOT}/assets"
mkdir -p "${LOCAL_ROBOTWIN_ROOT}/script/calibration"

echo "[local-runtime] validating local imports"
env \
    PYTHONPATH="${LOCAL_FASTWAM_ROOT}/third_party/curobo/src:${LOCAL_FASTWAM_ROOT}/src:${LOCAL_FASTWAM_ROOT}" \
    "${LOCAL_ENV}/bin/python" -B - <<'PY'
import open3d
import scipy.stats
import sklearn.preprocessing
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from fastwam.models.wan22.mot import MoT

print("local_runtime_imports_ok")
PY

printf '%s\n' \
    "source_robotwin=${SOURCE_ROBOTWIN_ROOT}" \
    "source_fastwam=${SOURCE_FASTWAM_ROOT}" \
    "source_env=${SOURCE_FASTWAM_ENV}" \
    "source_assets=${SOURCE_ASSETS_ROOT}" \
    "prepared_at=$(date '+%Y-%m-%d %H:%M:%S')" \
    > "${LOCAL_RUNTIME_ROOT}/runtime_manifest.txt"

echo "[local-runtime] ready: ${LOCAL_RUNTIME_ROOT}"
echo "[local-runtime] run with: USE_LOCAL_RUNTIME=1 bash ${SOURCE_ROBOTWIN_ROOT}/eval_fastwam_seed_whitelist_randomized.sh"
