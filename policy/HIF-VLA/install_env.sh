#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIFVLA_PROJECT=${HIFVLA_PROJECT:-/root/autodl-tmp/HIF-VLA_eval/HIF-VLA}
HIFVLA_ROOT=${HIFVLA_ROOT:-${HIFVLA_PROJECT}/source/HiF-VLA}
HIFVLA_ENV=${HIFVLA_ENV:-${HIFVLA_PROJECT}/env/hifvla}
HIFVLA_PYTHON=${HIFVLA_PYTHON:-${HIFVLA_ENV}/bin/python}
ENV_TOOL=${ENV_TOOL:-auto}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
DRY_RUN=${DRY_RUN:-0}
SKIP_CREATE=${SKIP_CREATE:-0}
SKIP_TORCH=${SKIP_TORCH:-0}
SKIP_TENSORFLOW=${SKIP_TENSORFLOW:-0}
USE_CN_MIRRORS=${USE_CN_MIRRORS:-1}
HIFVLA_PIP_INDEX_URL=${HIFVLA_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}
HIFVLA_PYTORCH_WHEEL_ROOT=${HIFVLA_PYTORCH_WHEEL_ROOT:-${HIFVLA_PYTORCH_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu121}}
HIFVLA_HF_ENDPOINT=${HIFVLA_HF_ENDPOINT:-https://hf-mirror.com}
HIFVLA_CONDA_CHANNEL=${HIFVLA_CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}

if [[ "${USE_CN_MIRRORS}" == "1" ]]; then
    export PIP_INDEX_URL="${HIFVLA_PIP_INDEX_URL}"
    export UV_DEFAULT_INDEX="${HIFVLA_PIP_INDEX_URL}"
    export HF_ENDPOINT="${HIFVLA_HF_ENDPOINT}"
    PYTORCH_SOURCE="${HIFVLA_PYTORCH_WHEEL_ROOT}"
else
    export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
    export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-${PIP_INDEX_URL}}
    export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
    PYTORCH_SOURCE=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
fi
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120}
export PIP_RETRIES=${PIP_RETRIES:-10}
export PIP_DISABLE_PIP_VERSION_CHECK=${PIP_DISABLE_PIP_VERSION_CHECK:-1}

export HF_HOME=${HF_HOME:-${HIFVLA_PROJECT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-${HIFVLA_PROJECT}/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${HIFVLA_PROJECT}/cache/pip}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${HIFVLA_PROJECT}/cache/matplotlib}
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}"

usage() {
    cat <<'EOF'
Usage: bash policy/HIF-VLA/install_env.sh

Environment variables:
  HIFVLA_PROJECT     Wrapper root (default: /root/autodl-tmp/HIF-VLA_eval/HIF-VLA)
  HIFVLA_ROOT        HiF-VLA source root
  HIFVLA_ENV         Prefix environment path
  ENV_TOOL           auto, uv, or conda
  PYTHON_VERSION     Environment Python version (default: 3.11)
  DRY_RUN            Print commands without executing (default: 0)
  SKIP_CREATE        Reuse an existing environment (default: 0)
  SKIP_TORCH         Do not install pinned PyTorch wheels (default: 0)
  SKIP_TENSORFLOW    Do not install TensorFlow/RLDS dependencies (default: 0)
  USE_CN_MIRRORS     Use domestic mirrors for pip/uv/conda/HF/PyTorch (default: 1)
  HIFVLA_PIP_INDEX_URL
                      PyPI index (default: Aliyun)
  HIFVLA_PYTORCH_WHEEL_ROOT
                      PyTorch CUDA 12.1 wheel directory (default: Aliyun)
  HIFVLA_HF_ENDPOINT Hugging Face endpoint (default: hf-mirror.com)
  HIFVLA_CONDA_CHANNEL
                      Conda main channel (default: TUNA)
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
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "${USE_CN_MIRRORS}" == "0" || "${USE_CN_MIRRORS}" == "1" ]] || {
    echo "USE_CN_MIRRORS must be 0 or 1" >&2
    exit 2
}
[[ -d "${HIFVLA_ROOT}" ]] || { echo "HIFVLA_ROOT does not exist: ${HIFVLA_ROOT}" >&2; exit 1; }

printf 'mirror mode: %s\n' "${USE_CN_MIRRORS}"
printf 'pip/uv index: %s\n' "${PIP_INDEX_URL}"
printf 'PyTorch source: %s\n' "${PYTORCH_SOURCE}"
printf 'Hugging Face endpoint: %s\n' "${HF_ENDPOINT}"
if [[ "${USE_CN_MIRRORS}" == "1" ]]; then
    printf 'Conda channel: %s\n' "${HIFVLA_CONDA_CHANNEL}"
fi

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

if [[ "${ENV_TOOL}" == "auto" ]]; then
    if command -v uv >/dev/null 2>&1; then
        ENV_TOOL=uv
    elif command -v conda >/dev/null 2>&1; then
        ENV_TOOL=conda
    else
        echo "Neither uv nor conda is available." >&2
        exit 1
    fi
fi

if [[ "${SKIP_CREATE}" != "1" && ! -x "${HIFVLA_PYTHON}" ]]; then
    case "${ENV_TOOL}" in
        uv) run uv venv --python "${PYTHON_VERSION}" "${HIFVLA_ENV}" ;;
        conda)
            if [[ "${USE_CN_MIRRORS}" == "1" ]]; then
                run conda create -y -p "${HIFVLA_ENV}" --override-channels \
                    -c "${HIFVLA_CONDA_CHANNEL}" "python=${PYTHON_VERSION}"
            else
                run conda create -y -p "${HIFVLA_ENV}" "python=${PYTHON_VERSION}"
            fi
            ;;
        *) echo "ENV_TOOL must be auto, uv, or conda" >&2; exit 2 ;;
    esac
fi

if [[ "${DRY_RUN}" != "1" && ! -x "${HIFVLA_PYTHON}" ]]; then
    echo "Environment Python was not created: ${HIFVLA_PYTHON}" >&2
    exit 1
fi

run "${HIFVLA_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ "${SKIP_TORCH}" != "1" ]]; then
    if [[ "${USE_CN_MIRRORS}" == "1" ]]; then
        if [[ -x "${HIFVLA_PYTHON}" ]]; then
            PYTHON_TAG="$("${HIFVLA_PYTHON}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
        else
            PYTHON_TAG="cp${PYTHON_VERSION/./}"
        fi
        case "$(uname -s)-$(uname -m)" in
            Linux-x86_64) WHEEL_PLATFORM=linux_x86_64 ;;
            *)
                echo "Aliyun PyTorch wheel mode currently supports Linux x86_64 only." >&2
                echo "Set USE_CN_MIRRORS=0 or SKIP_TORCH=1 on this platform." >&2
                exit 1
                ;;
        esac
        WHEEL_ROOT="${HIFVLA_PYTORCH_WHEEL_ROOT%/}"
        # Pin the large CUDA runtime wheels to Aliyun too. Otherwise pip resolves
        # them through the general PyPI index, which is often much slower.
        run "${HIFVLA_PYTHON}" -m pip install \
            "${WHEEL_ROOT}/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_nvjitlink_cu12-12.1.105-py3-none-manylinux1_x86_64.whl" \
            "${WHEEL_ROOT}/nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl"
        run "${HIFVLA_PYTHON}" -m pip install \
            "${WHEEL_ROOT}/torch-2.2.0%2Bcu121-${PYTHON_TAG}-${PYTHON_TAG}-${WHEEL_PLATFORM}.whl" \
            "${WHEEL_ROOT}/torchvision-0.17.0%2Bcu121-${PYTHON_TAG}-${PYTHON_TAG}-${WHEEL_PLATFORM}.whl" \
            "${WHEEL_ROOT}/torchaudio-2.2.0%2Bcu121-${PYTHON_TAG}-${PYTHON_TAG}-${WHEEL_PLATFORM}.whl"
    else
        run "${HIFVLA_PYTHON}" -m pip install \
            torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
            --index-url "${PYTORCH_SOURCE}"
    fi
fi

run "${HIFVLA_PYTHON}" -m pip install \
    "accelerate>=0.25.0" draccus==0.8.0 einops huggingface_hub \
    json-numpy jsonlines matplotlib peft==0.11.1 protobuf==4.25.3 rich \
    sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 wandb==0.16.6 \
    diffusers imageio imageio-ffmpeg uvicorn fastapi numpy==1.26.4 \
    pyarrow pillow tqdm av opencv-python ijson requests

run "${HIFVLA_PYTHON}" -m pip install \
    "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"

if [[ "${SKIP_TENSORFLOW}" != "1" ]]; then
    run "${HIFVLA_PYTHON}" -m pip install \
        tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow-metadata==1.15.0 \
        tensorflow_graphics==2021.12.3 \
        "dlimp @ git+https://github.com/moojink/dlimp_openvla"
fi

run "${HIFVLA_PYTHON}" -m pip install -e "${HIFVLA_ROOT}" --no-deps

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry-run complete. No environment or package was changed."
    exit 0
fi

PYTHONPATH="${HIFVLA_ROOT}:${PYTHONPATH:-}" "${HIFVLA_PYTHON}" - astribot <<'PY'
import importlib
for name in (
    "torch", "torchvision", "transformers", "peft", "draccus", "fastapi",
    "uvicorn", "tensorflow", "cv2", "av", "json_numpy",
):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
assert (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) == (18, 18, 8)
from experiments.robot.openvla_utils import get_vla_action
from experiments.robot.robot_utils import extract_motion_vectors_from_images
print("HIF-VLA evaluation imports: OK")
PY

echo "Environment ready: ${HIFVLA_ENV}"
echo "Run: ALLOW_MISSING_ENV=0 HIFVLA_PYTHON=${HIFVLA_PYTHON} bash ${SCRIPT_DIR}/check.sh"
