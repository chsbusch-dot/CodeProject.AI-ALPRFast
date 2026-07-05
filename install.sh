#!/bin/bash

# Installation script for the ALPRFast module.
# Called from the module directory via:  bash ../../CodeProject.AI-Server/src/setup.sh

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../CodeProject.AI-Server/src/setup.sh"
    echo
    exit 1
fi

mkdir -p "test"

# OpenCV needs these on bare-metal Linux (already present in most Docker images).
if [ "$moduleInstallErrors" = "" ] && [ "$inDocker" != true ] && [ "$os" = "linux" ]; then
    installAptPackages "libgl1-mesa-glx libglib2.0-0"
fi

# fast-alpr downloads its ONNX models (YOLOv9 detector + CCT OCR) from HuggingFace on
# first use and caches them under ~/.cache — no explicit model download step needed.

# Download a test image for the self-test.
if [ "$moduleInstallErrors" = "" ]; then
    getFromServer "test/" "license_plate_test.jpg" "test" "Downloading test image..."
fi
