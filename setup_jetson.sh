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
#
# NOTE: we tolerate apt errors because many JetPack 4.6.4 systems have a
# pre-existing broken nvidia-l4t-bootloader post-install (harmless, only
# affects bootloader firmware updates we don't want anyway). We verify our
# actual target packages installed via dpkg -l afterwards.
APT_PKGS=(
    libsndfile1
    libportaudio2
    python3-venv
    python3-dev
    libfreetype6-dev
    libgl1-mesa-dev
    libopenal1
)
sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]}" || true
# Verify everything we actually needed is installed.
MISSING=()
for pkg in "${APT_PKGS[@]}"; do
    if ! dpkg -s "$pkg" 2>/dev/null | grep -q '^Status: install ok installed'; then
        MISSING+=("$pkg")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  ✗ failed to install: ${MISSING[*]}" >&2
    exit 1
fi
echo "  ✓ all required apt packages installed"

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
# Force-reinstall numpy/scipy into the venv: with --system-site-packages,
# the ancient JetPack-shipped versions (numpy 1.13, scipy 0.19) satisfy
# our >= constraints and pip skips them. --ignore-installed isolates the
# venv copy from system site-packages for these critical deps.
# Pin numpy to 1.19.5 (last cp36 aarch64 wheel) and scipy to 1.5.x.
# OpenCV stays inherited from system site-packages (CUDA-enabled JetPack build).
# Panda3D also needs special handling (no PyPI aarch64 wheel) — TODO phase 2.
pip install -q --ignore-installed --no-deps 'numpy==1.19.5' 'scipy==1.5.4'
pip install -q -e .
pip install -q pytest matplotlib

# Bake the OpenBLAS workaround into the venv's activate script so every
# subsequent `source .venv/bin/activate` sets it automatically.
# The Nano's Cortex-A57 needs OPENBLAS_CORETYPE=ARMV8 or numpy 1.19 wheels
# crash with `Illegal instruction` on import.
if ! grep -q OPENBLAS_CORETYPE .venv/bin/activate; then
    cat >> .venv/bin/activate <<'EOF'

# --- vj-nano: Jetson Nano OpenBLAS fix ---
# The PyPI numpy/scipy aarch64 wheels assume ARMv8-A+SVE; Nano's Cortex-A57
# doesn't have those, so pin OpenBLAS to the base instruction set.
export OPENBLAS_CORETYPE=ARMV8
EOF
fi
# Also export for the rest of this script so [5/5] tests actually pass.
export OPENBLAS_CORETYPE=ARMV8

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
