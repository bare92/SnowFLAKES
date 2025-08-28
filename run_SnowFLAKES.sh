#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate conda environment
echo "Activating conda environment 'snowmap'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snowmap

# Define paths
SCRIPT_PATH="./main_SnowFLAKES.py"
CONFIG_PATH="./input_json/maipo_S2.json"  # <-- Change this to your actual JSON path

# Run the script
echo "Running SnowFLAKES classification..."
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "SnowFLAKES classification completed."

