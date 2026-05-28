#!/bin/bash

# Demo script to visualize slot attention maps

# Configuration
PYTHON="/data/miniconda3/envs/torch128/bin/python3"
WEIGHTS="/data/huyvq/fas/MVP_FAS_ViT-B-16_best_ckpt_250508.pt"
IMAGE="/data/fas/solution/MVP-FAS/runs/visualize_demo/004411_434__1773376341177-liveness-1.jpg"  # Change this to your image path /data/fas/solution/MVP-FAS/runs/visualize_demo/a1.jpg
SAVE_PATH="runs/visualize_slots"
GPU_ID=0

echo "=== Single Image Slot Visualization ==="
$PYTHON visualization/visualize_slots.py \
    --model MVP_FAS \
    --backbone "ViT-B/16" \
    --weights "$WEIGHTS" \
    --image "$IMAGE" \
    --save_path "$SAVE_PATH" \
    --gpu_id $GPU_ID \
    --prompts "spoof face" "attack face" "fake face" "real face" "baseline"

echo "Done! Check output in: $SAVE_PATH"
