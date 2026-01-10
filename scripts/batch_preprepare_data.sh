#!/bin/bash

# Base directory for data (absolute path)
DATA_DIR="$(pwd)/data"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR"
    exit 1
fi

# Find all episode0.hdf5 files and process them
find "$DATA_DIR" -name "episode0.hdf5" | sort | while read -r file; do
    echo "=================================================="
    echo "Processing: $file"
    echo "=================================================="
    
    python scripts/preprepare_data.py "$file"
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed $file"
    else
        echo "Failed to process $file"
    fi
    echo ""
done

echo "Batch processing complete."
