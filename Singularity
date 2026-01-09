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
        libglvnd0 \
        libglx0 \
        libegl1 \
        libx11-6 \
        git \
        wget \
        curl \
        unzip \
        ffmpeg \
        build-essential \
        ninja-build

    # Ensure EGL vendor directory exists for sapien/vulkan tricks
    mkdir -p /usr/share/glvnd/egl_vendor.d

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
    src /app/src
    scripts /app/scripts
    docs /app/docs

%post
    # Setup project
    cd /app
    export PATH="/root/.local/bin:$PATH"

    # Create virtual environment with Python 3.11 (Matching distrobox-setup.sh)
    # Ubuntu 24.04 defaults to 3.12, so we use uv to manage the python version.
    echo "Creating Python 3.11 environment..."
    uv venv .venv --python 3.11
    . .venv/bin/activate

    # Set build variables
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.7 8.9 9.0"
    export FORCE_CUDA=1
    
    # Install project dependencies
    # (pytorch3d and curobo are intentionally excluded from pyproject.toml)
    echo "Installing standard dependencies..."
    uv pip install .

    # --- Manual Build Step for ABI-Sensitive Packages ---
    # Matches distrobox-setup.sh logic explicitly
    
    export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
    export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
    
    echo "Installing pytorch3d with ABI=0 from main branch..."
    uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

    echo "Installing nvidia-curobo with ABI=0..."
    uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/curobo.git"
    
    # Create data directory
    mkdir -p /app/data

    # Download assets during build time
    export PYTHONPATH=/app/src
    export ROBOTWIN_DOWNLOAD_TEXTURES=false
    echo "Downloading assets during build..."
    python -c "from robotwin.assets._download import download_assets; download_assets()"
    
    # Configure ur5-wsg-bimanual-static (copied from distrobox setup)
    EMBODIMENT_DIR="src/robotwin/assets/embodiments"
    if [ ! -d "$EMBODIMENT_DIR/ur5-wsg-bimanual-static" ]; then
        echo "Creating ur5-wsg-bimanual-static configuration..."
        cp -r "$EMBODIMENT_DIR/ur5-wsg" "$EMBODIMENT_DIR/ur5-wsg-bimanual-static"
        
        CONFIG_FILE="$EMBODIMENT_DIR/ur5-wsg-bimanual-static/config.yml"
        sed -i '/static_camera_list:/,$d' "$CONFIG_FILE"
        
        cat >> "$CONFIG_FILE" <<EOF
static_camera_list: 
- name: cam_left
  type: D415
  position:
  - -0.5
  - 0
  - 1.2
  look_at:
  - 0
  - 0
  - 0.74
- name: cam_right
  type: D415
  position:
  - 0.5
  - 0
  - 1.2
  look_at:
  - 0
  - 0
  - 0.74
EOF
    fi

%environment
    export VIRTUAL_ENV="/app/.venv"
    export PATH="/app/.venv/bin:$PATH"
    export PYTHONPATH="/app/src:$PYTHONPATH"
    export ROBOTWIN_DOWNLOAD_TEXTURES=false

%runscript
    #!/bin/bash
    
    # Ensure environment is active
    if [ -f "/app/.venv/bin/activate" ]; then
        source /app/.venv/bin/activate
    fi
    
    # Execute the command passed to the container
    exec "$@"


%help
    This is the Singularity container for RoboTwin.
    
    Usage:
        # Running single-arm task data generation
        # Usage: ./scripts/generate_single_arm_demos.sh [episode_num] [start_seed]
        singularity run --nv --bind ./data:/app/data robotwin.sif ./scripts/generate_single_arm_demos.sh 10 0
        
        # Collecting data for a specific task manually
        # Usage: ./scripts/collect_data.sh <task_name> <task_config> <gpu_id> [episode_num] [start_seed]
        singularity run --nv --bind ./data:/app/data robotwin.sif ./scripts/collect_data.sh adjust_bottle single_arm_data_gen 0 10 0

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
