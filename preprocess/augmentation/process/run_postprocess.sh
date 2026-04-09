#!/bin/bash

set -euo pipefail

# Text file containing image paths (one per line)
TXT_FILE="preprocess/augmentation/process/image_paths.txt"

# Python preprocessing script
PYTHON_SCRIPT="preprocess/augmentation/process/preprocess_augmented.py"

# Log directory
LOGDIR="logs/aug_process_cohort_a"
mkdir -p "$LOGDIR"

OUT_LOG="${LOGDIR}/process.out"
ERR_LOG="${LOGDIR}/process.err"

# Optional: activate environment
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate QC_open

# Load all paths into an array
mapfile -t image_paths < "$TXT_FILE"
total=${#image_paths[@]}

if [ "$total" -eq 0 ]; then
    echo "Error: $TXT_FILE is empty or invalid."
    exit 1
fi

echo "Total images: $total"
echo "Output log: $OUT_LOG"
echo "Error log: $ERR_LOG"

# Run preprocessing
python "$PYTHON_SCRIPT" "${image_paths[@]}" >"$OUT_LOG" 2>"$ERR_LOG"

echo "All processing completed."