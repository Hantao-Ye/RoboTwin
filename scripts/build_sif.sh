#!/bin/bash
set -e

# Define temporary directories in current workspace to avoid /tmp quota issues
WORK_DIR=$(pwd)
mkdir -p "$WORK_DIR/singularity_build/cache"
mkdir -p "$WORK_DIR/singularity_build/tmp"

export SINGULARITY_CACHEDIR="$WORK_DIR/singularity_build/cache"
export SINGULARITY_TMPDIR="$WORK_DIR/singularity_build/tmp"

echo "Building Singularity image using workspace directory: $WORK_DIR/singularity_build"
echo "Cache Dir: $SINGULARITY_CACHEDIR"
echo "Tmp Dir: $SINGULARITY_TMPDIR"

# Check if robotwin.sif exists and remove it to avoid issues
if [ -f "robotwin.sif" ]; then
    echo "Removing existing robotwin.sif..."
    rm robotwin.sif
fi

echo "Starting build..."
singularity build --fakeroot robotwin.sif Singularity

echo "Build complete."
echo "You can remove the temporary directory with: rm -rf $WORK_DIR/singularity_build"
