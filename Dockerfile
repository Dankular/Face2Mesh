FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /workspace/Face2Mesh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Pre-download models (this is the key - baked into the image)
# Models will be in /root/.cache/face_models
RUN python download_models_direct.py || echo "Some models may need manual download"

# Create output directory
RUN mkdir -p /workspace/output

# Expose port for potential web interface
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "pipeline_complete.py"]
CMD ["--help"]
