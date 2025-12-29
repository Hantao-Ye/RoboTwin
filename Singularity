Bootstrap: docker
From: nvidia/cuda:12.6.0-devel-ubuntu24.04

%labels
    Author RoboTwin Team
    Version 0.1.0

%post
    # Prevent interactive prompts during apt install
    export DEBIAN_FRONTEND=noninteractive

    # Install system dependencies
    apt-get update && apt-get install -y \
        software-properties-common \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libxi6 \
        git \
        wget \
        curl \
        unzip \
        ffmpeg

    # Install Python 3 (Ubuntu 24.04 comes with Python 3.12)
    apt-get update && apt-get install -y \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        python3-setuptools

    # Set python3 as default python
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

    # Allow system-wide package installation (PEP 668)
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this section
    export PATH="/root/.local/bin:$PATH"

    # Create working directory
    mkdir -p /app

%files
    pyproject.toml /app/
    README.md /app/
    LICENSE /app/
    src /app/
    scripts /app/scripts
    docs /app/docs

%post
    # Install python dependencies
    cd /app
    export PATH="/root/.local/bin:$PATH"
    # Use --no-build-isolation as requested
    # Install torch first as it is a build dependency for pytorch3d
    # Install Cython as it is a build dependency for toppra (via mplib)
    # Install hatchling as it is the build backend for robotwin
    # Set TORCH_CUDA_ARCH_LIST to avoid build errors when torch can't detect the GPU during build
    export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
    uv pip install --system --no-build-isolation torch torchvision Cython hatchling
    uv pip install --system --no-build-isolation .
    
    # Create data directory
    mkdir -p /app/data

    # Download assets during build time so they are baked into the image
    # This avoids runtime download issues on read-only HPC systems
    export PYTHONPATH=/app/src
    export ROBOTWIN_DOWNLOAD_TEXTURES=false
    echo "Downloading assets during build..."
    python3 -c "from robotwin.assets._download import download_assets; download_assets()"

%environment
    export PYTHONPATH=/app/src
    export ROBOTWIN_DOWNLOAD_TEXTURES=false
    # Set default assets path to a writable location if not overridden
    # In Singularity, /app is read-only, so users MUST bind mount or set this var
    # We default to the internal path, but it will fail if not writable.

%runscript
    #!/bin/bash
    # This script runs when the container is executed
    
    # Check if we are in a read-only environment and ASSETS_PATH is not set/writable
    # We rely on the python script to handle the download logic, but we need to ensure
    # it writes to a valid location.
    
    echo "Singularity container started."
    
    if [ -z "$ROBOTWIN_ASSETS_PATH" ]; then
        echo "WARNING: ROBOTWIN_ASSETS_PATH is not set."
        echo "If the container is read-only, asset download will fail."
        echo "Please run with: --bind /path/to/assets:/app/src/robotwin/assets"
        echo "OR set env var: --env ROBOTWIN_ASSETS_PATH=/path/to/writable/assets"
    fi

    echo "Checking and downloading assets..."
    python -c "from robotwin.assets._download import download_assets; download_assets()"
    
    # Execute the command passed to the container
    exec "$@"

%help
    This is the Singularity container for RoboTwin.
    
    Usage:
        singularity run --nv --bind ./data:/app/data robotwin.sif ./scripts/generate_single_arm_demos.sh
        
    Assets:
        The container needs a writable location for assets.
        Option 1 (Bind mount internal path):
            mkdir -p assets
            singularity run --nv \
                --bind ./data:/app/data \
                --bind ./assets:/app/src/robotwin/assets \
                robotwin.sif ...
                
        Option 2 (Env var):
            singularity run --nv \
                --env ROBOTWIN_ASSETS_PATH=/path/to/local/assets \
                --bind ./data:/app/data \
                robotwin.sif ...
