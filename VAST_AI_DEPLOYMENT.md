# Vast.ai Deployment Guide - Face2Mesh

## Option 1: Pre-built Docker Image (FASTEST - Recommended)

### Step 1: Pull Pre-built Image from Docker Hub
```bash
# On your Vast.ai instance
docker pull dankular/face2mesh:latest
```

### Step 2: Run Immediately
```bash
# Create input/output directories
mkdir -p /workspace/input /workspace/output

# Copy your face image to input
cp your_face.jpg /workspace/input/face.jpg

# Run pipeline (models already included in image)
docker run --rm --gpus all \
  -v /workspace/input:/workspace/input \
  -v /workspace/output:/workspace/output \
  dankular/face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

**Time to first run**: ~5 minutes (just image pull)

---

## Option 2: Build Image on Vast.ai (Slower but Customizable)

### Step 1: Rent Vast.ai Instance

**Recommended specs**:
- GPU: RTX 4090 (24GB VRAM) or RTX 3090
- RAM: 64+ GB
- Disk: 150 GB (77GB models + 50GB working space)
- CUDA: 11.8+

**Cheapest option**: ~$0.30/hour for RTX 3090

### Step 2: SSH into Instance
```bash
ssh -p <port> root@<vast-instance-ip> -L 8080:localhost:8080
```

### Step 3: Run Setup Script
```bash
# Download and run automated setup
wget https://raw.githubusercontent.com/Dankular/Face2Mesh/master/vast_ai_setup.sh
chmod +x vast_ai_setup.sh
./vast_ai_setup.sh
```

**What this does**:
1. Installs Docker + NVIDIA runtime
2. Clones repository
3. Builds Docker image with ALL models baked in (77GB)
4. Tests installation

**Time**: 1-2 hours (one-time, models cached in image)

### Step 4: Run Pipeline
```bash
cd /workspace/Face2Mesh

# Place your face image
cp /path/to/your/face.jpg input/face.jpg

# Run with docker-compose
docker-compose up

# OR run directly
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input \
  -v $(pwd)/output:/workspace/output \
  face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

---

## Option 3: Manual Setup (Most Control)

### Step 1: Prepare Instance
```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Python 3.11
apt-get install -y python3.11 python3-pip git wget

# Install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Clone & Install
```bash
cd /workspace
git clone https://github.com/Dankular/Face2Mesh.git
cd Face2Mesh

pip install -r requirements.txt
```

### Step 3: Download Models
```bash
# This downloads 77GB to ~/.cache/face_models
python download_models_direct.py
```

### Step 4: Run Pipeline
```bash
python pipeline_complete.py \
  --input your_face.jpg \
  --output ./results \
  --device cuda
```

---

## Docker Image Structure

The pre-built image includes:

```
dankular/face2mesh:latest
├── /workspace/Face2Mesh/        # Code
│   ├── pipeline_complete.py
│   ├── download_models_direct.py
│   └── requirements.txt
├── /root/.cache/face_models/    # Pre-downloaded models (77GB)
│   ├── qwen/                    (54 GB)
│   ├── marigold/                (15 GB)
│   ├── iclight/                 (4.9 GB)
│   ├── depth_anything_v2/       (1.3 GB)
│   ├── bisenet/                 (1.1 GB)
│   └── emoca/                   (812 MB)
└── /root/.insightface/
    └── models/buffalo_l/        (280 MB - ArcFace)
```

**Total image size**: ~80-85 GB (includes CUDA, system libs, models)

---

## Building Your Own Docker Image

### Local Build (for customization)
```bash
# Clone repo
git clone https://github.com/Dankular/Face2Mesh.git
cd Face2Mesh

# Build image (downloads 77GB of models during build)
docker build -t face2mesh:latest .

# Tag for Docker Hub
docker tag face2mesh:latest yourusername/face2mesh:latest

# Push to Docker Hub
docker push yourusername/face2mesh:latest
```

**Build time**: 1-2 hours (downloads all models)

### On Vast.ai (direct build)
```bash
cd /workspace
git clone https://github.com/Dankular/Face2Mesh.git
cd Face2Mesh

# Build with NVIDIA runtime
docker build -t face2mesh:latest .

# Run
docker run --rm --gpus all face2mesh:latest --help
```

---

## Vast.ai Instance Selection

### Recommended Filters:

**GPU**:
- RTX 4090 (best performance, ~25-40 min/avatar)
- RTX 3090 (good value, ~40-60 min/avatar)
- A100 (overkill but fastest, ~20-30 min/avatar)

**VRAM**: 24+ GB

**RAM**: 64+ GB

**Disk**: 150+ GB

**CUDA**: 11.8 or 12.x

**Docker**: Must support nvidia-docker

### Cost Estimates:
- RTX 4090: $0.40-0.80/hour
- RTX 3090: $0.25-0.50/hour
- A100 (40GB): $1.00-2.00/hour

**Per avatar cost** (RTX 4090):
- Setup: $0 (one-time, image is pre-built)
- Processing: ~$0.40 (40 min @ $0.60/hr)
- Total: **~$0.40 per avatar** (after first run)

---

## Optimizations for Vast.ai

### 1. Persistent Disk Template
Save the built Docker image as a Vast.ai template:
1. Build image once on a cheap instance
2. Create template from instance
3. Future instances start with models pre-loaded

### 2. Multi-GPU Support
```bash
# Use multiple GPUs (if instance has them)
docker run --rm --gpus '"device=0,1"' \
  -v /workspace/input:/workspace/input \
  -v /workspace/output:/workspace/output \
  face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

### 3. Batch Processing
```bash
# Process multiple faces
for img in /workspace/input/*.jpg; do
  docker run --rm --gpus all \
    -v /workspace/input:/workspace/input \
    -v /workspace/output:/workspace/output \
    face2mesh:latest \
    --input /workspace/input/$(basename $img) \
    --output /workspace/output/$(basename $img .jpg) \
    --device cuda
done
```

---

## Monitoring & Logs

### Check GPU Usage
```bash
# On Vast.ai instance
nvidia-smi -l 1

# Inside Docker container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### View Pipeline Logs
```bash
# Follow logs in real-time
docker logs -f <container-id>

# Or run in foreground (remove --rm)
docker run --gpus all \
  -v /workspace/input:/workspace/input \
  -v /workspace/output:/workspace/output \
  face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

---

## Troubleshooting

### Issue: Docker image pull is slow
**Solution**: Use Vast.ai instance with fast network (check "Network Speed" filter)

### Issue: Out of disk space during build
**Solution**: Rent instance with 200+ GB disk

### Issue: CUDA out of memory
**Solution**: 
- Use instance with 24+ GB VRAM
- Reduce batch size in pipeline (edit pipeline_complete.py)

### Issue: Models not loading
**Solution**: 
- Check `/root/.cache/face_models/` exists in container
- Verify models downloaded during build (check Docker build logs)

---

## Quick Start Commands Summary

### Fastest (Pre-built Image):
```bash
# 1. Pull image (5 min)
docker pull dankular/face2mesh:latest

# 2. Run (25-40 min on RTX 4090)
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input \
  -v $(pwd)/output:/workspace/output \
  dankular/face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

### Build Your Own:
```bash
# 1. Clone repo
git clone https://github.com/Dankular/Face2Mesh.git
cd Face2Mesh

# 2. Build (1-2 hours, one-time)
docker build -t face2mesh:latest .

# 3. Run
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input \
  -v $(pwd)/output:/workspace/output \
  face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

---

## Next Steps

1. **Rent Vast.ai instance** (RTX 4090, 64GB RAM, 150GB disk)
2. **Pull pre-built image** or run setup script
3. **Upload your face image**
4. **Run pipeline** (25-40 min)
5. **Download results** (3D mesh, textures, exports)

**Total cost**: ~$0.40-0.80 per avatar (after initial setup)

**Setup time**: 5 min (pre-built) or 2 hours (build from scratch)

**Processing time**: 25-40 min per avatar (RTX 4090)
