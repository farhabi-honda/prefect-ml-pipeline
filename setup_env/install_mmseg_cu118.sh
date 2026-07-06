#!/bin/bash

# --- VIRTUAL ENVIRONMENT CHECK ---
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No Python virtual environment detected."
    echo "Please activate your virtual environment before running this script."
    exit 1
fi


pip_install() {
    pip install --proxy="$http_proxy" "$@"
}


# PyTorch CUDA index URL
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

echo "Starting installation of Python libraries..."

# 1. Install PyTorch and related CUDA libraries FIRST
# These must be installed first to properly set up the environment for MMCV.
cuda_install_libraries=(
    "torch==2.4.1+cu118"
    "torchaudio==2.4.1+cu118"
    "torchvision==0.19.1+cu118"
)
echo "Installing PyTorch CUDA libraries..."
pip_install "${cuda_install_libraries[@]}" --index-url "${TORCH_INDEX_URL}" --extra-index-url "https://pypi.org/simple"

if [ $? -ne 0 ]; then
    echo "Error: Failed to install one or more PyTorch CUDA libraries. Aborting."
    exit 1
fi

# 2. Install MIM (MMLab Installation Manager)
echo "Installing openmim..."
pip_install -U openmim

# 3. Use MIM to install MMEngine and the compatible MMCV
# MMCV >= 2.0.0 is required for MMSegmentation v1.x
echo "Using mim to install MMEngine and compatible MMCV..."
mim install mmengine
mim install "mmcv==2.1.0"

if [ $? -ne 0 ]; then
    echo "Error: Failed to install MMCV or MMEngine. Aborting."
    exit 1
fi

# 4. Install MMSegmentation and other default libraries
# NOTE: MMSegmentation is moved here, after its core dependencies.
default_install_libraries=(
    "opencv-python==4.11.0.86"
    "numpy==1.26.4"
    "tqdm==4.65.0"
    "mmsegmentation==1.2.2"
    "pillow==11.3.0"
    "regex==2025.9.18"
    "ftfy==6.3.1"
    "colcon-common-extensions==0.3.0"
    "pydantic"
    "future"
    "tensorboard"
)

echo "Installing MMSegmentation and other default libraries..."
pip_install "${default_install_libraries[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to install MMSegmentation or other default libraries. Aborting."
    exit 1
fi

echo "All specified libraries have been successfully installed. 🎉"