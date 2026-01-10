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
    git

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
# Common values: 8.0 (A100), 8.6 (RTX 30 series), 8.9 (RTX 40 series), 9.0 (H100), 10.0 (RTX 50 series / Blackwell)
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.7 8.9 9.0 10.0"

# Fix C++ ABI compatibility (PyTorch wheels use old ABI=0, system GCC uses new ABI=1)
# PyTorch 2.9+ likely uses New ABI (1)
export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"

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
export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Reinstall curobo to ensure ABI compatibility and CUDA compilation
echo "Reinstalling nvidia-curobo to ensure ABI compatibility..."
uv pip uninstall nvidia-curobo || true
uv pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/curobo.git"

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
    echo "ur5-wsg-bimanual-static created successfully."
fi

# Generate a project-specific environment script instead of modifying global shell config
echo "Creating env.sh for easy activation..."
cat > env.sh <<EOF
#!/bin/bash
export PYTHONPATH="$(pwd)/src:\$PYTHONPATH"
export ROBOTWIN_DOWNLOAD_TEXTURES=false
source "$(pwd)/.venv/bin/activate"
EOF

echo "Setup complete!"
echo "To start working, run: source env.sh"