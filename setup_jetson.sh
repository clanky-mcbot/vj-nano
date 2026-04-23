#!/usr/bin/env bash
# setup_jetson.sh — one-shot bootstrap for vj-nano on a Jetson Nano 4GB
#                   running JetPack 4.6.4 (Ubuntu 18.04 / Python 3.6).
#
# What it does:
#   1. Installs apt packages (system deps for audio, gstreamer, panda3d).
#   2. Creates a venv at ./.venv with --system-site-packages so we inherit
#      the JetPack-shipped OpenCV (CUDA-enabled) and, later, TensorRT bindings.
#   3. Upgrades pip/setuptools/wheel inside the venv (JetPack ships pip 9).
#   4. Installs the package + dev deps via pip.
#   5. Runs pytest as a smoke test.
#
# Idempotent: re-running is safe. Add --force to rebuild the venv from scratch.

set -euo pipefail

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2 ; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "== [1/5] apt packages =="
sudo apt-get update -qq
# - libsndfile1: soundfile backend
# - libportaudio2: sounddevice backend
# - python3-venv: pip-in-venv support
# - gstreamer*: webcam capture (already on JetPack but we list for clarity)
# - libfreetype6-dev, libgl1-mesa-dev: panda3d build deps
sudo apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    python3-venv \
    python3-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libopenal1

echo "== [2/5] venv =="
if [ "$FORCE" = "1" ] && [ -d .venv ]; then
    echo "  --force: removing existing .venv"
    rm -rf .venv
fi
if [ ! -d .venv ]; then
    # --system-site-packages so we keep the JetPack-provided cv2 (CUDA-enabled)
    # and can later import tensorrt/pycuda without rebuilding them in the venv.
    python3 -m venv --system-site-packages .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "== [3/5] bootstrap pip =="
# JetPack ships pip 9.0.1 which is too old to resolve modern wheels.
# Upgrade pip itself first, then the rest of the toolchain.
python -m pip install --upgrade 'pip<22' 'setuptools<60' 'wheel' -q

echo "== [4/5] install package =="
# We skip the [dev] extra here because panda3d needs a Nano-specific install
# (there's no arm64 aarch64 wheel on PyPI). That step happens later.
# opencv is already provided by system-site-packages.
pip install -q -e .
pip install -q pytest matplotlib

echo "== [5/5] smoke test =="
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import soundfile; print('soundfile', soundfile.__version__)"
python -c "import cv2; print('cv2', cv2.__version__, '(from system site-packages)')" \
    || echo "  (cv2 not visible — check --system-site-packages)"
python -m pytest tests/ -q

echo
echo "=== ✓ bootstrap complete ==="
echo "  activate with:  source .venv/bin/activate"
echo "  run tests:      pytest"
