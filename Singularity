Bootstrap: docker
From: nvidia/cuda:12.8.0-devel-ubuntu24.04

%labels
    Author RoboTwin Team
    Version 0.1.0

%post
    # Prevent interactive prompts during apt install
    export DEBIAN_FRONTEND=noninteractive

    # Install system dependencies
    # Matching distrobox-setup.sh + container specific libs
    apt-get update && apt-get install -y \
        software-properties-common \
        unzip \
        ffmpeg \
        libavcodec-extra \
        libx264-dev \
        x264 \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libxi6 \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        python3-setuptools \
        build-essential \
        ninja-build \
        git \
        wget \
        curl \
        libglvnd0 \
        libglx0 \
        libegl1 \
        libx11-6

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
    uv.lock /app/
    README.md /app/
    LICENSE /app/
    src /app/src
    scripts /app/scripts
    docs /app/docs

%post
    # Setup project
    cd /app
    export PATH="/root/.local/bin:$PATH"

    # Use uv sync to setup the environment (Matching distrobox-setup.sh)
    echo "Syncing project with uv..."
    # Set build variables
    # Keeping the broad list for container portability, unlike distrobox which targets local hardware
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.7 8.9 9.0 10.0"
    export FORCE_CUDA=1

    # Using uv sync with Python 3.11
    uv sync --python 3.11
    . .venv/bin/activate

    # --- Manual Build Step for ABI-Sensitive Packages ---
    # Matches distrobox-setup.sh logic explicitly

    echo "Reinstalling pytorch3d to ensure ABI compatibility..."
    uv pip uninstall pytorch3d || true
    
    echo "Checking PyTorch ABI compatibility..."
    # Ask the installed PyTorch which ABI it was built with (True=1, False=0)
    TORCH_ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
    echo "Detected PyTorch ABI: $TORCH_ABI"
    
    export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$TORCH_ABI"
    export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$TORCH_ABI"
    
    echo "Installing pytorch3d with ABI=$TORCH_ABI..."
    # Using @stable tag as in distrobox-setup.sh
    uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

    echo "Reinstalling nvidia-curobo to ensure ABI compatibility..."
    uv pip uninstall nvidia-curobo || true
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
        # Removing exisiting config
        sed -i '/static_camera_list:/,$d' "$CONFIG_FILE"
        
        cat >> "$CONFIG_FILE" <<EOF
static_camera_list: 
- name: cam_left
  type: D415
  position:
  - -0.6
  - 0
  - 1.3
  look_at:
  - 0
  - 0
  - 0.74
- name: cam_right
  type: D415
  position:
  - 0.6
  - 0
  - 1.3
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
