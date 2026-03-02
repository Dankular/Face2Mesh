#!/bin/bash
# Quick setup script for Vast.ai instances

set -e

echo "================================"
echo "Face2Mesh Vast.ai Quick Setup"
echo "================================"

# Update system
echo "[1/6] Updating system..."
apt-get update -qq

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[2/6] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
else
    echo "[2/6] Docker already installed"
fi

# Install nvidia-docker if not present
if ! command -v nvidia-docker &> /dev/null; then
    echo "[3/6] Installing NVIDIA Docker runtime..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update -qq
    apt-get install -y nvidia-docker2
    systemctl restart docker
else
    echo "[3/6] NVIDIA Docker already installed"
fi

# Clone repository
echo "[4/6] Cloning Face2Mesh repository..."
cd /workspace
if [ ! -d "Face2Mesh" ]; then
    git clone https://github.com/Dankular/Face2Mesh.git
else
    cd Face2Mesh && git pull && cd ..
fi

cd Face2Mesh

# Build Docker image (with models baked in)
echo "[5/6] Building Docker image with models (this will take 1-2 hours and download 77GB)..."
docker build -t face2mesh:latest .

# Test the installation
echo "[6/6] Testing installation..."
docker run --rm --gpus all face2mesh:latest --help

echo ""
echo "================================"
echo "✓ Setup Complete!"
echo "================================"
echo ""
echo "Usage:"
echo "  1. Place your face image in: /workspace/Face2Mesh/input/face.jpg"
echo "  2. Run: docker run --rm --gpus all -v /workspace/Face2Mesh/input:/workspace/input -v /workspace/Face2Mesh/output:/workspace/output face2mesh:latest --input /workspace/input/face.jpg --output /workspace/output --device cuda"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up"
echo ""
echo "Models are cached inside the Docker image (~77GB)"
echo "First run will be fast since models are pre-downloaded"
echo ""
