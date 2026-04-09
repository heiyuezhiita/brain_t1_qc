#!/bin/bash
set -euo pipefail

# Config
TXT_FILE="preprocess/preprocessing/image_paths.txt"
PYTHON_SCRIPT="preprocess/preprocessing/preprocess.py"
OUTPUT_ROOT="data/images/preprocessed/original/cohort_a"

# Logs
LOGDIR="logs/prep/cohort_a"
mkdir -p "$LOGDIR"
OUT_LOG="$LOGDIR/prep.out"
ERR_LOG="$LOGDIR/prep.err"


# Read input paths
mapfile -t image_paths < "$TXT_FILE"
total=${#image_paths[@]}

if [ "$total" -eq 0 ]; then
    echo "Error: $TXT_FILE is empty or contains no valid paths."
    exit 1
fi

echo "Total inputs: $total"
echo "Output root: $OUTPUT_ROOT"
echo "Stdout log: $OUT_LOG"
echo "Stderr log: $ERR_LOG"

# Run preprocessing
python "$PYTHON_SCRIPT" \
    --inputs "${image_paths[@]}" \
    --output_root "$OUTPUT_ROOT" \
    >"$OUT_LOG" 2>"$ERR_LOG"

echo "Preprocessing finished."