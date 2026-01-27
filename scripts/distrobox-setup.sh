#!/bin/bash
set -e

echo "Setting up RoboTwin development environment..."

# Prevent interactive prompts during apt install
export DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
echo "Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
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
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools

# Ensure uv is available (should be installed by box-init)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Configure CUDA environment if not already set
if [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda" ]; then
    echo "Configuring CUDA environment..."
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
fi

echo "Installing Python dependencies..."
# Navigate to project root (parent of scripts directory)
cd "$(dirname "$0")/.."

# Set TORCH_CUDA_ARCH_LIST for your GPU (adjust as needed)
if [ -z "$TORCH_CUDA_ARCH_LIST" ] && command -v nvidia-smi >/dev/null 2>&1; then
    # Query compute capability (e.g., "8.9"), take the first GPU found
    DETECTED_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [ -n "$DETECTED_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$DETECTED_ARCH"
    fi
fi

# Create and activate virtual environment
echo "Creating virtual environment (.venv)..."
rm -rf .venv
# uv cache clean --force

# Use uv sync to setup the environment
echo "Syncing project with uv..."
FORCE_CUDA=1 uv sync --python 3.11

source .venv/bin/activate

# FORCE REINSTALL PYTORCH3D with correct ABI
# uv sync build isolation often ignores global CFLAGS/CXXFLAGS, leading to ABI mismatch
echo "Reinstalling pytorch3d to ensure ABI compatibility..."
uv pip uninstall pytorch3d || true
export FORCE_CUDA=1

# Ensure numpy is <2.0.0 BEFORE compiling anything to avoid ABI mismatch
echo "Downgrading numpy to <2.0.0 for compatibility..."
uv pip install --no-build-isolation "numpy<2.0.0"

echo "Checking PyTorch ABI compatibility..."
# Ask the installed PyTorch which ABI it was built with (True=1, False=0)
TORCH_ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
echo "Detected PyTorch ABI: $TORCH_ABI"

# Export the correctly detected ABI flags for subsequent compiles
export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$TORCH_ABI"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$TORCH_ABI"

uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Manual installation of torch-geometric stack to fix ABI and version issues
# torch-cluster needs to be built from git for torch>=2.9 compatibility
# numpy must be kept <2.0.0 to avoid segfaults in sapien/curobo
echo "Reinstalling torch-geometric stack with correct ABI..."
uv pip uninstall torch-cluster torch-scatter torch-sparse || true
uv pip install --no-build-isolation --no-cache-dir --upgrade torch-scatter torch-sparse "numpy<2.0.0"
uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/rusty1s/pytorch_cluster.git" "numpy<2.0.0"

# Reinstall curobo to ensure ABI compatibility and CUDA compilation
echo "Reinstalling nvidia-curobo to ensure ABI compatibility..."
uv pip uninstall nvidia-curobo || true
uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/curobo.git" "numpy<2.0.0"

# Set up environment variables
export ROBOTWIN_DOWNLOAD_TEXTURES=false

echo "Creating data directory..."
mkdir -p data

echo "Downloading assets..."
python3 -c "from robotwin.assets._download import download_assets; download_assets()"

# Restore ur5-wsg-bimanual-static config if it doesn't exist (assuming it's a copy/modification of ur5-wsg)
if [ ! -d "src/robotwin/assets/embodiments/ur5-wsg-bimanual-static" ]; then
    echo "Creating ur5-wsg-bimanual-static from ur5-wsg..."
    cp -r src/robotwin/assets/embodiments/ur5-wsg src/robotwin/assets/embodiments/ur5-wsg-bimanual-static
    
    # Modify config.yml: Replace static_camera_list with bimanual configuration
    CONFIG_FILE="src/robotwin/assets/embodiments/ur5-wsg-bimanual-static/config.yml"
    
    # Remove the existing static_camera_list and everything after it
    sed -i '/static_camera_list:/,$d' "$CONFIG_FILE"
    
    # Append the new camera configuration
    cat >> "$CONFIG_FILE" <<EOF
static_camera_list: 
- name: cam_left
  type: D415
  position:
  - -0.3
  - 0
  - 1.3
  look_at:
  - 0.2
  - 0
  - 0.74
- name: cam_right
  type: D415
  position:
  - 0.6
  - 0
  - 1.3
  look_at:
  - -0.2
  - 0
  - 0.74
EOF
    echo "ur5-wsg-bimanual-static created successfully."
fi

# Generate a project-specific environment script instead of modifying global shell config
echo "Creating env.sh for easy activation..."
cat > env.sh <<'EOF'
#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export ROBOTWIN_DOWNLOAD_TEXTURES=false

# Source the venv using the safe variable
source "$PROJECT_ROOT/.venv/bin/activate"
EOF

echo "Setup complete!"
echo "To start working, run: source env.sh"