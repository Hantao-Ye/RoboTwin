#!/bin/bash

# Trap SIGINT (Ctrl+C) to exit the script immediately
trap "echo 'Script interrupted by user'; exit 1" INT

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

CONFIG="single_arm_data_gen"
GPU_ID=0
EPISODE_NUM=${1} # Optional argument for number of episodes

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

echo "Starting data generation for single-arm tasks..."
echo "Configuration: $CONFIG"
echo "GPU ID: $GPU_ID"
echo "----------------------------------------"

for task in $TASKS; do
    echo "Processing task: $task"
    # Call collect_data.sh located in the same directory as this script
    "$SCRIPT_DIR/collect_data.sh" "$task" "$CONFIG" "$GPU_ID" "$EPISODE_NUM"
    echo "Finished task: $task"
    echo "----------------------------------------"
done

echo "All tasks completed."
