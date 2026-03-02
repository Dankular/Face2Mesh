# Face2Mesh - Complete Film-Quality Face-to-3D Pipeline

Single photograph → Production-ready 3D head avatar with pore-level detail, anatomically correct features, full PBR materials, and animation capability.

**Built from:** `Complete_Pipeline_Walkthrough.md` (film industry standard, 25-40 min on RTX 4090)

---

## 🎯 What This Does

Input: **One face photo** (512×512 minimum, both eyes visible)

Output: **Film-quality 3D avatar** with:
- ✅ FLAME topology mesh (50+ blendshapes)
- ✅ 4K PBR texture set (albedo, normal, roughness, specular, displacement)
- ✅ Pore-level microdetail (wrinkles, skin grain)
- ✅ Layered eyes (sclera, iris, cornea with IOR 1.376)
- ✅ Teeth, tongue, inner mouth cavity
- ✅ Eyelashes, eyebrows, inner ear detail
- ✅ Hair (strands, cards, or volume)
- ✅ Audio/video-driven animation
- ✅ ACEScg color space (film standard)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models (~77 GB)
```bash
python download_models_direct.py
```

Downloads:
- Qwen-Image-Edit-2511 (54 GB) - Multiview generation
- Marigold (15 GB) - Depth/normals/intrinsics
- IC-Light (4.9 GB) - Relighting
- Depth Anything V2 (1.3 GB) - Depth estimation
- BiSeNet (1.1 GB) - Face parsing
- ArcFace (280 MB) - Identity extraction
- EMOCA (812 MB) - FLAME parameters

**Total: 77 GB** (cached in `~/.cache/face_models/`)

### 3. Run Complete Pipeline
```bash
python pipeline_complete.py --input your_face.jpg --output ./results
```

**Processing time:**
- CPU: ~2-3 hours first run
- RTX 4090: ~25-40 minutes

---

## 📋 Pipeline Stages

### Stage 1: Extract Identity & Parameters (~10 sec)
- **ArcFace**: 512-dim identity embedding for verification
- **EMOCA**: FLAME shape (β, 300-dim) + expression (ψ, 50-dim)
- **BiSeNet**: Per-pixel face segmentation (skin, hair, eyes, lips, etc.)

### Stage 2: Reconstruct 3D Geometry (2-5 min)
- **Depth Anything V2**: Monocular depth estimation
- **TSDF Fusion**: Multi-view depth integration → watertight mesh
- **Mesh Cleanup**: Remove artifacts, Laplacian smoothing, decimation

### Stage 3: Generate Texture Views (2-4 min)
- **Qwen-Image-Edit-2511**: 24 identity-consistent views
- **Multi-Angles LoRA**: Camera-controlled rendering
- **Identity Verification**: ArcFace similarity check (>0.6 threshold)

### Stage 4: Assemble Textures & Materials (5-10 min)
- **Retopology**: Shrinkwrap to FLAME topology
- **UV Unwrap**: FLAME standard UV layout
- **Texture Baking**: Project 24 views → UV space
- **PBR Generation**:
  - **Albedo**: De-lit diffuse (Marigold intrinsic decomposition)
  - **Normal**: Camera-aware surface detail (Marigold)
  - **Roughness**: Skin variation by region
  - **Displacement**: TSDF vs FLAME difference
- **Microdetail**: Pore-level displacement maps

### Stage 5: Rig & Blendshapes (~30 sec)
- **FLAME Rig**: 5 joints (neck, jaw, 2× eye, head)
- **50+ Blendshapes**: Expression shapes from FLAME
- **Corrective Shapes**: Handle complex deformations

### Stage 6: Anatomical Details (~1 min)
- **Layered Eyes**: Sclera, iris (from photo), pupil, cornea
- **Teeth**: 168 triangles, SSS material
- **Tongue**: 500-1000 tris, 6-8 ARKit blendshapes
- **Eyelashes/Eyebrows**: Hair cards or parametric models
- **Inner Ear**: Anatomical template

### Stage 7: Hair Reconstruction (5-10 min)
- Options: Hair strands / hair cards / TSDF volume

### Stage 8: Lighting & Materials (1-2 min)
- **IC-Light**: Scene-specific relighting
- **ACEScg**: Film-standard color space

### Stage 9: Animation (5-15 sec/frame)
- Audio-driven: Wav2Lip / Audio2Face-3D
- Video-driven: EMOTE / DiffPoseTalk

### Stage 10: Validation & Export
- Formats: USD, FBX, GLTF/GLB, Alembic
- Quality checklist validation

---

## 📁 Project Structure

```
Face2Mesh/
├── pipeline_complete.py          # Main 10-stage pipeline (600 lines)
├── download_models_direct.py     # Model downloader
├── requirements.txt              # Python dependencies
├── test_face.jpg                 # Example input
├── Complete_Pipeline_Walkthrough.md  # Reference spec (463 lines)
├── MODELS_READY.md               # Downloaded models status
└── output_complete_run/          # Generated outputs
    ├── stage1_identity/
    ├── stage2_geometry/
    ├── stage3_multiview/
    ├── stage4_textures/
    ├── stage5_rig/
    ├── stage6_details/
    ├── stage7_hair/
    ├── stage8_lighting/
    ├── stage9_animation/
    └── final_export/
```

---

## 🔧 Requirements

### Hardware
- **Minimum**: 16 GB RAM, 80 GB disk space
- **Recommended**: NVIDIA RTX 4090 (24 GB VRAM), 64 GB RAM, 500 GB SSD
- **Alternative**: RTX 3090/4080/A100

### Software
- Python 3.11+ (Python 3.14 has Open3D incompatibility)
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Dependencies
```
torch>=2.0.0
diffusers>=0.30.0
transformers>=4.40.0
insightface>=0.7.3
trimesh>=4.0.0
pymeshlab>=2022.2
xatlas>=0.0.11
numpy, pillow, scipy, scikit-image
```

See `requirements.txt` for complete list.

---

## 🎨 Advanced Usage

### Custom Camera Configurations
```python
from pipeline_complete import CompleteFaceTo3DPipeline

pipeline = CompleteFaceTo3DPipeline(device="cuda")

# 8 views (standard)
results = pipeline.run_complete_pipeline("face.jpg")

# 24 views (high quality)
# Edit camera_config in pipeline_complete.py
```

### Individual Stage Execution
```python
# Run only specific stages
stage1_data = pipeline.stage1_extract_identity("face.jpg")
stage2_data = pipeline.stage2_reconstruct_geometry("face.jpg", stage1_data)
# ... etc
```

### Export Formats
```python
# USD (Pixar standard)
# FBX (Autodesk/Unity/Unreal)
# GLTF/GLB (Web/compressed)
# Alembic (VFX cache)
```

---

## 📊 Model Details

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| Qwen-Image-Edit-2511 | 54 GB | Multiview generation | HuggingFace |
| Marigold | 15 GB | Depth/normal/intrinsic | prs-eth |
| IC-Light | 4.9 GB | Relighting | lllyasviel |
| Depth Anything V2 | 1.3 GB | Depth estimation | depth-anything |
| BiSeNet | 1.1 GB | Face parsing | jonathandinu |
| EMOCA | 812 MB | FLAME parameters | Manual download |
| ArcFace | 280 MB | Identity verification | insightface |

**Total**: 77 GB (auto-cached)

---

## 🐛 Troubleshooting

### Open3D Python 3.14 Incompatibility
```bash
# Solution 1: Downgrade Python
conda create -n face2mesh python=3.11
conda activate face2mesh
pip install -r requirements.txt

# Solution 2: Use PyTorch3D
pip install pytorch3d
```

### Out of Memory (GPU)
```bash
# Reduce multiview count (edit pipeline_complete.py)
num_views = 8  # instead of 24

# Use CPU fallback
python pipeline_complete.py --device cpu
```

### Missing FLAME Model
FLAME requires registration at https://flame.is.tue.mpg.de/
Download and place in: `~/.cache/face_models/flame/`

---

## 🎓 Research Papers

This pipeline implements methods from:

- **FLAME** (2023): Parametric head model
- **ArcFace** (2019): Face recognition embedding
- **BiSeNet** (2018): Real-time semantic segmentation
- **Depth Anything V2** (2024): Robust monocular depth
- **Qwen-Image-Edit** (2024): Multi-modal image editing
- **Marigold** (2023): Affine-invariant depth
- **IC-Light** (2024): Text-guided relighting
- **EMOCA** (2021): Emotion-conditioned FLAME

See `Complete_Pipeline_Walkthrough.md` for full citations.

---

## 📝 License

Research and educational use only. Commercial use requires separate licensing for:
- FLAME model (MPI-IS)
- Individual model licenses (see HuggingFace repos)

---

## 🙏 Acknowledgments

Built on research from:
- Max Planck Institute (FLAME)
- ETH Zurich (Marigold)
- Alibaba/Qwen Team (Qwen-Image-Edit)
- InsightFace (ArcFace)
- And many others (see `Complete_Pipeline_Walkthrough.md`)

---

## 🔗 Links

- **Documentation**: `Complete_Pipeline_Walkthrough.md`
- **Model Status**: `MODELS_READY.md`
- **Implementation**: `IMPLEMENTATION_STATUS.md`
- **GitHub**: https://github.com/Dankular/Face2Mesh

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- See `Complete_Pipeline_Walkthrough.md` for technical details
- Check `MODELS_READY.md` for model download status

---

**Status**: ✅ All 10 stages implemented, 77 GB models downloaded, ready for testing

**Ready to convert your face photo to a film-quality 3D avatar!**
