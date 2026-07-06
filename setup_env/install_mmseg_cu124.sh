#!/bin/bash
# Installs mmsegmentation + its CUDA 12.4-matched deps into $VIRTUAL_ENV
# (/app/segenv, set by the Dockerfile RUN that calls this script).
#
# Key fix vs an earlier version of this script: mmcv is pinned to an EXACT
# version (==2.1.0), not a range (>=2.1.0). With a range, `mim` resolves to
# the latest version satisfying it, which may not have a prebuilt wheel for
# this cu124/torch2.4.1 combo — silently falling back to a source compile
# that fails without a full devel toolchain. Pinning the exact version
# known to have a matching wheel avoids that entirely.
set -euo pipefail

# --- VIRTUAL ENVIRONMENT CHECK ---
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No Python virtual environment detected."
    echo "Please activate your virtual environment before running this script."
    exit 1
fi

pip_install() {
    pip install --proxy="${http_proxy:-}" "$@"
}

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

echo "Starting installation of Python libraries..."

# 1. PyTorch + CUDA libs FIRST — mmcv's build/wheel-selection depends on
# torch already being present.
cuda_install_libraries=(
    "torch==2.4.1+cu124"
    "torchaudio==2.4.1+cu124"
    "torchvision==0.19.1+cu124"
)
echo "Installing PyTorch CUDA libraries..."
pip_install "${cuda_install_libraries[@]}" --index-url "${TORCH_INDEX_URL}" --extra-index-url "https://pypi.org/simple"

# 2. MIM (OpenMMLab's installer)
echo "Installing openmim..."
pip_install -U openmim

# 3. MMEngine + exact-pinned MMCV (see comment at top re: why exact, not range)
echo "Using mim to install MMEngine and compatible MMCV..."
mim install mmengine
mim install mmcv==2.1.0

# 4. mmsegmentation itself — baked into the image (not cloned at trigger
# time). NOTE: added back in here — make sure this isn't a duplicate of a
# clone step you already have elsewhere.
echo "Cloning and installing mmsegmentation..."
git clone --branch main --depth 1 \
    https://github.com/open-mmlab/mmsegmentation.git /app/mmsegmentation
pip_install -e /app/mmsegmentation

# 5. Remaining default libraries
default_install_libraries=(
    "opencv-python==4.11.0.86"
    "numpy==1.26.4"
    "tqdm==4.65.0"
    "pillow==11.3.0"
    "regex==2025.9.18"
    "ftfy==6.3.1"
    "onnxruntime-gpu"
    "tensorboard"
    "prettytable"
    "scipy"
)
# NOTE: colcon-common-extensions (ROS2 build tool) removed — unrelated to
# mmsegmentation/PyTorch. Add it back if this container genuinely needs ROS2.
echo "Installing remaining default libraries..."
pip_install "${default_install_libraries[@]}"

echo "All specified libraries have been successfully installed."
