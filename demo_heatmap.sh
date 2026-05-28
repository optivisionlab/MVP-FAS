#!/bin/bash

# Demo script to visualize heatmap for a single image

# Configuration
PYTHON="/data/miniconda3/envs/torch128/bin/python3"
WEIGHTS="/data/huyvq/fas/MVP_FAS_ViT-B-16_best_ckpt_250508.pt"
IMAGE="/data/fas/solution/MVP-FAS/runs/visualize_demo/004411_434__1773376341177-liveness-1.jpg"  # Change this to your image path /data/fas/solution/MVP-FAS/runs/visualize_demo/a1.jpg
SAVE_PATH="runs/visualize_demo"
METHOD="gradcam"  # or "attention" gradcam
GPU_ID=0

# Run visualization
$PYTHON visualization/visualize_heatmap.py \
    --model MVP_FAS \
    --backbone "ViT-B/16" \
    --weights "$WEIGHTS" \
    --image "$IMAGE" \
    --save_path "$SAVE_PATH" \
    --method "$METHOD" \
    --gpu_id $GPU_ID

echo "Done! Check output in: $SAVE_PATH"
