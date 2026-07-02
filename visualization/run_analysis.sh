#!/bin/bash

# Script để chạy phân tích data distribution

echo "=========================================="
echo "Data Distribution Analysis"
echo "=========================================="

# Activate environment if needed
# source /path/to/your/venv/bin/activate

# Set output directory
OUTPUT_DIR="/data/fas/solution/MVP-FAS/visualization/analysis_output"

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Sample size: 500 images per dataset"
echo ""

# Run analysis
python /data/fas/solution/MVP-FAS/visualization/analyze_data_distribution.py \
    --output_dir $OUTPUT_DIR \
    --sample_size 500

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report: cat $OUTPUT_DIR/ANALYSIS_REPORT.md"
echo "View images: ls $OUTPUT_DIR/*.png"
