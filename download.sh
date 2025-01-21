#!/bin/bash

# Set Hugging Face endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Check if a model name argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name> [local_directory]"
    echo "Example: $0 OpenGVLab/InternVL2-8B"
    exit 1
fi

# Define model name and local directory
model_name=$1
local_dir=${2:-"./pretrained/$(basename $model_name)"}

# Ensure the local directory exists
mkdir -p "$local_dir"

# Get the download command passed in --exclude '*.safetensors'
# Set the download command
download_command="huggingface-cli download --resume-download $model_name --local-dir $local_dir"

# Configure retry mechanism
max_retries=5
retry_count=0

echo "Starting download for model: $model_name"
echo "Saving to directory: $local_dir"

# Download loop with retries
while true; do
    # Execute the download command
    eval $download_command

    # Check the exit status of the command
    if [ $? -eq 0 ]; then
        echo "Download completed successfully for model: $model_name"
        break
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "Maximum retry limit reached, download failed for model: $model_name"
            exit 1
        fi
        echo "Download failed, retrying in 5 seconds (Attempt $retry_count/$max_retries)"
        sleep 5
    fi
done