#!/bin/bash

# Define a list of required packages
REQUIRED_PKG=("flask" "gradio" "grpcio-tools" "aiohttp" "djitellopy" "openai" "opencv-python" "numpy" "pillow" "filterpy" "matplotlib" "torch")

# Function to check and install package
check_and_install() {
    package=$1
    if ! pip3 list | grep -F $package > /dev/null; then
        echo "Package $package is not installed. Installing..."
        pip3 install $package
    else
        echo "Package $package is already installed."
    fi
}

# Iterate over required packages and check each one
for pkg in "${REQUIRED_PKG[@]}"; do
    check_and_install $pkg
done

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "WARNNING: OPENAI_API_KEY is not set"
fi