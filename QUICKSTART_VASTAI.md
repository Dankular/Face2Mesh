# 🚀 Vast.ai Quick Start - Face2Mesh

## Fastest Way to Deploy (5 minutes)

### 1. Rent GPU Instance
**Specs**: RTX 4090, 64GB RAM, 150GB disk, CUDA 11.8+

**Cost**: ~$0.40-0.80/hour

### 2. SSH In
```bash
ssh -p <port> root@<ip>
```

### 3. Run One Command
```bash
curl -s https://raw.githubusercontent.com/Dankular/Face2Mesh/master/vast_ai_setup.sh | bash
```

**Done!** Models are downloaded and baked into Docker image.

---

## Process Your Face (30-40 min on RTX 4090)

```bash
cd /workspace/Face2Mesh

# Add your image
cp your_face.jpg input/face.jpg

# Run pipeline
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input \
  -v $(pwd)/output:/workspace/output \
  face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

**Results in**: `/workspace/Face2Mesh/output/`

---

## Cost Breakdown

| Step | Time | Cost (RTX 4090 @ $0.60/hr) |
|------|------|----------------------------|
| Setup (one-time) | 1-2 hrs | $0.60-1.20 |
| Per avatar | 30-40 min | $0.30-0.40 |

**After setup**: ~$0.35 per avatar

---

## What You Get

```
output/
├── stage1_identity/
│   ├── arcface_embedding.npy      # 512-dim identity
│   └── face_segmentation.png      # Parsed regions
├── stage2_geometry/
│   ├── depth_maps/                # Multi-view depth
│   └── mesh_raw.ply               # Initial 3D mesh
├── stage3_multiview/
│   ├── view_00_front.png          # 8-24 AI renders
│   ├── view_01_right.png
│   └── ...
├── stage4_textures/
│   ├── albedo_4k.png              # PBR textures
│   ├── normal_4k.png
│   ├── roughness_4k.png
│   └── displacement_4k.exr
├── stage5_rig/
│   └── blendshapes/               # 50+ shapes
├── final_export/
│   ├── avatar.fbx                 # Unreal/Unity
│   ├── avatar.glb                 # Web/Three.js
│   └── avatar.usd                 # Pixar/film
```

---

## Alternative: Pull Pre-built Image

If you have your own GPU setup:

```bash
docker pull dankular/face2mesh:latest

docker run --rm --gpus all \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  dankular/face2mesh:latest \
  --input /workspace/input/face.jpg \
  --output /workspace/output \
  --device cuda
```

**Image size**: 80-85 GB (includes all models)

---

## Support

- Full guide: `VAST_AI_DEPLOYMENT.md`
- Issues: https://github.com/Dankular/Face2Mesh/issues
- Code: https://github.com/Dankular/Face2Mesh
