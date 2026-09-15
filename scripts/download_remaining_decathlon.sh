#!/bin/bash
set -e

DATA_DIR="/Users/stnava/data/decathlon"
TASKS=(
    "Task01_BrainTumour"
    "Task06_Lung"
    "Task08_HepaticVessel"
    "Task07_Pancreas"
    "Task03_Liver"
)

BASE_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com"

for TASK in "${TASKS[@]}"; do
    if [ -d "$DATA_DIR/$TASK" ] && [ -f "$DATA_DIR/$TASK/dataset.json" ]; then
        echo "=== $TASK already exists, skipping ==="
        continue
    fi

    TAR_FILE="$DATA_DIR/${TASK}.tar"
    URL="$BASE_URL/${TASK}.tar"

    echo "=== Downloading $TASK from $URL ==="
    /opt/homebrew/bin/aria2c --file-allocation=none -x 16 -s 16 -d "$DATA_DIR" "$URL"

    echo "=== Unpacking $TAR_FILE ==="
    tar -xf "$TAR_FILE" -C "$DATA_DIR"
    rm -f "$TAR_FILE"
    echo "=== Finished $TASK ==="
done

echo "=== All Decathlon tasks downloaded and unpacked successfully! ==="
