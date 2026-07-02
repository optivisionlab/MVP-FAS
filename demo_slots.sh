#!/bin/bash

# Demo script to visualize slot attention maps for images listed in a CSV file

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Check if CSV file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file>"
    echo "CSV file should contain image paths (one per line or in a column)"
    exit 1
fi

CSV_FILE="$1"

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

# Configuration
PYTHON="/data/miniconda3/envs/torch128/bin/python3"
# WEIGHTS="/data/fas/solution/MVP-FAS/runs/clip17/train_4/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"
WEIGHTS="/data/fas/solution/MVP-FAS/runs/clip14/train_7/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"
SAVE_PATH="runs/visualize_slots"
GPU_ID=0

# Create output directory if it doesn't exist
mkdir -p "$SAVE_PATH"

# Counter for processed images
count=0
total=$(wc -l < "$CSV_FILE")

echo "Processing $total images from: $CSV_FILE"
echo "Output directory: $SAVE_PATH"
echo "---"

# Read CSV file line by line
while IFS=, read -r image_path || [ -n "$image_path" ]; do
    # Skip empty lines
    [ -z "$image_path" ] && continue

    # Trim whitespace
    image_path=$(echo "$image_path" | xargs)

    # Skip if not a valid path or if file doesn't exist
    if [ ! -f "$image_path" ]; then
        echo "Warning: Image not found, skipping: $image_path"
        continue
    fi

    count=$((count + 1))
    echo "[$count/$total] Processing: $image_path"

    # Run visualization
    $PYTHON visualization/visualize_slots.py \
        --model MVP_FAS \
        --backbone "ViT-B/16" \
        --weights "$WEIGHTS" \
        --image "$image_path" \
        --save_path "$SAVE_PATH" \
        --gpu_id $GPU_ID \
        --prompts \
            "spoof face" \
            "printed photo face" \
            "replay attack face" \
            "mask face" \
            "live face" \
            "real face" \
            "baseline"

    if [ $? -ne 0 ]; then
        echo "Error processing: $image_path"
    fi

done < "$CSV_FILE"

echo "---"
echo "Done! Processed $count images."
echo "Check output in: $SAVE_PATH"
