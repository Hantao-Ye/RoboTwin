#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Arguments
TASK_NAME=${1}
TASK_CONFIG=${2}
GPU_ID=${3:-0} # Default to GPU 0 if not provided
EPISODE_NUM=${4} # Optional episode number override
START_SEED=${5} # Optional start seed override

# Validation
if [ -z "$TASK_NAME" ] || [ -z "$TASK_CONFIG" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <task_name> <task_config> [gpu_id] [episode_num] [start_seed]"
    exit 1
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Determine execution command (use python)
CMD="python"

# Construct command arguments
PY_ARGS="src/robotwin/script/collect_data.py \"$TASK_NAME\" \"$TASK_CONFIG\""
if [ ! -z "$EPISODE_NUM" ]; then
    PY_ARGS="$PY_ARGS --episode_num $EPISODE_NUM"
fi
if [ ! -z "$START_SEED" ]; then
    PY_ARGS="$PY_ARGS --start_seed $START_SEED"
fi

# Run data collection
# PYTHONWARNINGS=ignore::UserWarning suppresses specific warnings for cleaner output
PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning \
eval $CMD $PY_ARGS

# Clean up cache
# Note: collect_data.py strips .yml extension for the output directory
CONFIG_DIR_NAME=$(basename "$TASK_CONFIG" .yml)
CACHE_DIR="data/${TASK_NAME}/${CONFIG_DIR_NAME}/.cache"

if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
fi
