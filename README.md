# Face2Mesh — Complete Face-to-3D Pipeline

Converts a single face photo into an animatable 3D head asset (GLB/FBX/USD).

## Pipeline Overview

```
Input: single face image (JPG/PNG)
  │
  ▼
Stage 1  — Identity extraction        ArcFace embedding + BiSeNet segmentation
  │
  ▼
Stage 2  — Multi-view + Gaussians     FaceLift → gaussians.ply + 6 consistent views
  │
  ▼
Stage 3  — Mesh extraction            TSDF fusion of Gaussian depth renders → mesh
  │
  ▼
Stage 4  — Texture baking             Project views onto UV-unwrapped mesh
  │
  ▼
Stage 5  — Rig & blendshapes          FLAME topology fit + ARKit blendshapes
  │
  ▼
Stage 6  — Eyes / teeth / tongue      Eyeball mesh + teeth shell geometry
  │
  ▼
Stage 7  — Texture enhancement        Qwen Multi-Angles for occluded UV regions
  │
  ▼
Stage 8  — Lighting & materials       IC-Light PBR relighting
  │
  ▼
Stage 9  — Animation                  Audio/video puppet
  │
  ▼
Stage 10 — Validation & export        FBX / GLB / USD
```

---

## Stage 2 — FaceLift (Multi-view + Gaussian Splatting)

**Repo:** [weijielyu/FaceLift](https://github.com/weijielyu/FaceLift)
**Paper:** ICCV 2025 — *FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads*

### What it does

FaceLift takes a single face photo and outputs:

- **`gaussians.ply`** — a 3D Gaussian Splatting representation of the full head
- **`multiview.png`** — 6 geometrically-consistent views (0°, 60°, 120°, 180°, 240°, 300° azimuth)
- **`turntable.mp4`** — 360° video flyaround

The 6 views are generated jointly through a shared multi-view diffusion denoising process, meaning all views share latent information at every diffusion step — guaranteeing 3D-consistent texture and geometry. This is fundamentally different from generating views independently.

### Architecture

```
Input face
    │
    ▼
StableUnCLIP image encoder
    │  (CLIP image embedding)
    ▼
Multi-view diffusion (MVDiffusion)
    │  Joint denoising across 6 camera views
    │  Cross-view attention at every step
    ▼
6 consistent RGBA views at known azimuths
    │
    ▼
GS-LRM (Gaussian Splatting Large Reconstruction Model)
    │  Feed-forward, no per-scene optimization
    ▼
gaussians.ply  (3DGS representation)
```

### Installation

```bash
# Clone FaceLift
git clone https://github.com/weijielyu/FaceLift /FaceLift
cd /FaceLift

# Install dependencies (CUDA 12.4, PyTorch 2.4)
bash setup_env.sh

# Model checkpoints download automatically from HuggingFace (wlyu/OpenFaceLift)
# on first inference run. To pre-download manually:
#   checkpoints/mvdiffusion/pipeckpts/   — multi-view diffusion
#   checkpoints/gslrm/ckpt_*.pt         — GS-LRM reconstruction
```

### Usage via Face2Mesh

FaceLift is called automatically as **Stage 2** of the pipeline. To run Stage 2 standalone:

```bash
cd /Face2Mesh
python pipeline_complete.py --input /path/to/face.jpg --stage 2
```

Direct FaceLift CLI:

```bash
cd /FaceLift
python inference.py \
    --input_dir /path/to/input_dir/ \
    --output_dir /path/to/output_dir/ \
    --auto_crop True \
    --seed 4 \
    --guidance_scale_2D 3.0 \
    --step_2D 50
```

### VRAM Notes

| VRAM     | Strategy |
|----------|----------|
| ≥ 16 GB  | FP16, xformers memory-efficient attention |
| 12 GB    | FP16 + sequential CPU offload (`FACELIFT_CPU_OFFLOAD=1`) |
| < 12 GB  | Not recommended |

For 12 GB cards (e.g. RTX 3060) FaceLift is configured to run with sequential CPU offload. Inference takes **3–8 minutes** on an RTX 3060.

---

## Stage 3 — Gaussian → Mesh (TSDF Fusion)

Converts `gaussians.ply` to a watertight triangle mesh:

1. Render depth maps from each of the 6 known camera positions by projecting Gaussian centres
2. Fuse depth maps using **TSDF volumetric fusion** (Open3D `ScalableTSDFVolume`)
3. Extract surface with **Marching Cubes**
4. Falls back to Poisson mesh from point cloud if Open3D is unavailable

---

## Stage 7 — Qwen Multi-Angles (Texture Enhancement)

**Model:** `unsloth/Qwen-Image-Edit-2511-GGUF` — `qwen-image-edit-2511-Q4_0.gguf`
**HuggingFace:** https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF

### What it does

After geometry is locked from FaceLift + TSDF, Qwen generates high-quality texture images for UV regions that have **weak coverage** from the original single image:

| Region | Prompt target |
|--------|--------------|
| `back_head` | Occipital / hair / back of skull |
| `under_chin` | Under chin, neck, jaw underside |
| `left_ear` | Left ear close-up, helix, canal |
| `right_ear` | Right ear close-up, helix, canal |
| `top_head` | Top-down view, hair parting |

### Why Qwen is used here (not Stage 2)

Qwen Multi-Angles generates views **independently** with no cross-view latent communication during diffusion. This means it cannot guarantee geometric consistency between views — the same ear might look different from two angles. This makes it unsuitable for **geometry generation**.

However, once the mesh geometry is fixed, painting UV texture patches is a per-region task. **Cross-view consistency is not required** — each Qwen output gets projected onto its corresponding UV region independently. This is where Qwen excels.

**Summary:**
- FaceLift = geometry stage → *must* be cross-view consistent
- Qwen = texture-only stage → independent views are fine

### VRAM Notes

Loaded as **GGUF Q4_0** via `llama-cpp-python`. Runs on **consumer GPUs with as little as 8 GB VRAM** — no BitsAndBytes/CUDA issues. Model file `qwen-image-edit-2511-Q4_0.gguf` from [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF).

---

## Quick Start

```bash
# Full pipeline
python pipeline_complete.py --input /path/to/face.jpg --output ./output

# Single stage
python pipeline_complete.py --input /path/to/face.jpg --stage 1   # Identity
python pipeline_complete.py --input /path/to/face.jpg --stage 2   # FaceLift
```

## Requirements

```
torch >= 2.4
diffusers >= 0.30
transformers >= 4.44
insightface
open3d
trimesh
plyfile
bitsandbytes    # for 4-bit Qwen
facenet-pytorch # for FaceLift face detection
rembg           # for FaceLift background removal
xformers
```

Install:
```bash
pip install -r requirements.txt
# FaceLift Gaussian rasterizer (CUDA extension, compiles on install):
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

## Directory Structure

```
/Face2Mesh/
├── pipeline_complete.py      — Full 10-stage pipeline runner
├── face2mesh_facelift.py     — FaceLift + GaussianToMesh + QwenTextureEnhancer
├── __init__.py               — Original Face2Mesh modules
├── requirements.txt
└── output/                   — Pipeline outputs

/FaceLift/                    — FaceLift repo (cloned separately)
├── inference.py              — Called by Stage 2 via subprocess
├── checkpoints/              — Auto-downloaded from HuggingFace (wlyu/OpenFaceLift)
├── mvdiffusion/              — Multi-view diffusion module
└── gslrm/                    — Gaussian Splatting LRM module
```

---

## Known Issues & Lessons Learned

Problems hit across development sessions. Documented so they are not repeated on fresh setup.

| # | Problem | Fix / Note |
|---|---------|------------|
| 1 | **PyTorch version conflicts** | Pin exact torch+cuda combo. On Vast.ai, match the instance CUDA version. |
| 2 | **BitsAndBytes / CUDA incompatibility** | Removed entirely. Qwen now runs as GGUF Q4_0 via `llama-cpp-python`. |
| 3 | **CUDA extension compile failures** | `diff-gaussian-rasterization` compiles at install time. CUDA toolkit must match PyTorch CUDA build exactly. |
| 4 | **FLAME model behind registration wall** | Must manually register at https://flame.is.tue.mpg.de/ and place `generic_model.pkl` at `~/.cache/face_models/flame/`. Stages 4 & 5 fall back to xatlas silently without it. |
| 5 | **Gaussian Splatting misused for mesh** | gsplat is a rendering format, not a mesh source. Mesh must come from TSDF/Marching Cubes. |
| 6 | **VRAM too low for NF4 Qwen** | Switched to GGUF Q4_0 — runs on 8 GB VRAM. No BitsAndBytes needed. |
| 7 | **3D backbone undecided** | TRELLIS / Hunyuan3D-2 / TripoSG are all in requirements. Pick one and commit before next session. |
| 8 | **HuggingFace downloads timeout mid-run** | Pre-download all models to cache before running pipeline. |
| 9 | **77 GB Docker image impractical** | Use setup scripts + volume mounts instead of baking models into image. |
| 10 | **README and code stage numbers drifted** | Keep README and `pipeline_complete.py` stage numbers in sync at every commit. |
| 11 | **SSH heredoc + backtick expansion** | Backticks in plink heredocs get shell-expanded. Write scripts to a local file, upload with pscp, then run. |
| 12 | **Git HTTPS push needs token** | Inject token into remote URL: `git remote set-url origin https://<token>@github.com/...` |
| 13 | **xformers version pinning** | Must exactly match PyTorch build or silently falls back to slow attention. |
| 14 | **Open3D not available on Python 3.12+** | **Use Python 3.11.** Open3D has no 3.12/3.14 wheel. TSDF fusion breaks silently. |
| 15 | **Vast.ai instance ephemerality** | All installs lost on stop. Keep a setup script that restores env + re-downloads models from persistent volume. |
| 16 | **Test/duplicate files accumulate** | No `test_*.py`, `*_new.py`, `output_*/`, or `__pycache__` in commits. Covered by .gitignore. |
