#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/conda_envs/lerobot}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-a445d9c9da6bea99a8972daa4fe1fdd053d711d2}"

if [ ! -x "${PYTHON_BIN}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10 pip
fi

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot@${LEROBOT_COMMIT}" \
  h5py opencv-python opencv-python-headless tqdm tyro

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip install \
  typing_extensions diffusers einops omegaconf termcolor wandb filelock networkx \
  sympy==1.13.1 mpmath docstring-parser typeguard pandas httpx anyio httpcore h11 \
  click hf-xet markdown-it-py pygments mdurl gitpython platformdirs protobuf \
  sentry-sdk annotated-types gitdb smmap psutil urllib3 python-dateutil tzdata \
  exceptiongroup regex safetensors markupsafe werkzeug \
  nvidia-cudnn-cu12==9.1.0.70 nvidia-cusparselt-cu12==0.6.2 nvidia-nvjitlink-cu12==12.4.127

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip install "datasets==3.2.0"
PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip check
