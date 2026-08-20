#!/bin/bash

MODE="yolo"
# INPUT_DIR="/u01/vision/data/vft_data_level2/20260814/silicone"
OUTPUT_DIR="/u01/vision/data/OMCI"
YOLO_MODEL="/u01/quanlm/ibeta/weight/yolov8n-face.pt"
INSIGHTFACE_MODEL="buffalo_l"
INPUT_DIR_IMAGE="/data/fas" # path thư mục gốc
CSV_FILE="/data/fas/csv/p1/full-level-data-p1-flip.csv"
CSV_COLUMN="path"


CTX_ID=0
DET_SIZE=640
DET_THRESH=0.5 # 0.3
FACE_SIZE=224


if [ "${MODE}" = "yolo" ]; then
    python dataset/align.py \
        --mode yolo \
        ${INPUT_DIR:+--input-dir "${INPUT_DIR}"} \
        ${INPUT_DIR_IMAGE:+--input-dir-image "${INPUT_DIR_IMAGE}"} \
        ${CSV_FILE:+--csv-file "${CSV_FILE}"} \
        ${CSV_COLUMN:+--csv-column "${CSV_COLUMN}"} \
        --output-dir "${OUTPUT_DIR}" \
        --yolo-model "${YOLO_MODEL}" \
        --det-size "${DET_SIZE}" \
        --det-thresh "${DET_THRESH}" \
        --face-size "${FACE_SIZE}"

elif [ "${MODE}" = "insightface" ]; then
    python dataset/align.py \
        --mode insightface \
        ${INPUT_DIR:+--input-dir "${INPUT_DIR}"} \
        ${INPUT_DIR_IMAGE:+--input-dir-image "${INPUT_DIR_IMAGE}"} \
        ${CSV_FILE:+--csv-file "${CSV_FILE}"} \
        ${CSV_COLUMN:+--csv-column "${CSV_COLUMN}"} \
        --output-dir "${OUTPUT_DIR}" \
        --insightface-model "${INSIGHTFACE_MODEL}" \
        --ctx-id "${CTX_ID}" \
        --det-size "${DET_SIZE}" \
        --det-thresh "${DET_THRESH}" \
        --face-size "${FACE_SIZE}"

else
    echo "Invalid MODE: ${MODE}"
    exit 1
fi