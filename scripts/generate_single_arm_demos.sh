#!/bin/bash

# Trap SIGINT (Ctrl+C) to exit the script immediately
trap "echo 'Script interrupted by user'; exit 1" INT

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source env.sh automatically if it exists
if [ -f "$SCRIPT_DIR/../env.sh" ]; then
    source "$SCRIPT_DIR/../env.sh"
fi

CONFIG="single_arm_data_gen"
GPU_ID=0
EPISODE_NUM=${1} # Optional argument for number of episodes
START_SEED=${2} # Optional argument for start seed

# Path to the tasks file (relative to the script location)
TASKS_FILE="$SCRIPT_DIR/../docs/SINGLE_ARM_TASKS.md"

# Check if SINGLE_ARM_TASKS.md exists
if [ ! -f "$TASKS_FILE" ]; then
    echo "Error: $TASKS_FILE not found!"
    exit 1
fi

# Read tasks from SINGLE_ARM_TASKS.md
# Extract lines starting with "- ", remove the prefix and the ".py" extension
TASKS=$(grep "^- " "$TASKS_FILE" | sed 's/- //; s/.py//')

# Count total tasks
TOTAL_TASKS=$(echo "$TASKS" | wc -w)
CURRENT_TASK=1

echo "Starting data generation for single-arm tasks..."
echo "Total tasks to process: $TOTAL_TASKS"
echo "Configuration: $CONFIG"
echo "GPU ID: $GPU_ID"
echo "----------------------------------------"

for task in $TASKS; do
    echo "[${CURRENT_TASK}/${TOTAL_TASKS}] Processing task: $task"
    # Call collect_data.sh located in the same directory as this script
    "$SCRIPT_DIR/collect_data.sh" "$task" "$CONFIG" "$GPU_ID" "$EPISODE_NUM" "$START_SEED"
    echo "Finished task: $task"
    echo "----------------------------------------"
    ((CURRENT_TASK++))
done

echo "All tasks completed."
