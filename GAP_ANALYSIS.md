# Gap Analysis — Current Code vs Complete_Pipeline_Walkthrough.md
*Updated: March 2026 — post pipeline implementation*

## Status Overview

All 10 stages now have code (`stages/stage1-10_*.py`). The gap has shifted from
**"nothing implemented"** to **"code exists but models not downloaded / environment not configured"**.
This document tracks what still needs to happen on a fresh server.

---

## Stage-by-Stage

### STAGE 1 — Identity Extraction
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 1.1 ArcFace (antelopev2) | 512-dim identity embedding | ✅ stage1_identity.py | ❌ Not downloaded (~1 GB) | Download on setup |
| 1.2 MICA | FLAME β shape params | ✅ stage1_identity.py | ❌ Not downloaded (~2 GB) | Download on setup |
| 1.3 SMIRK | FLAME expression ψ params | ✅ stage1_identity.py | ❌ Not downloaded (~2 GB) | Download on setup |
| 1.4 BiSeNet | Per-pixel face segmentation | ✅ stage1_identity.py | ❌ Not downloaded (~1 GB) | Download on setup |

**Gap:** Code complete. All 4 models need downloading on fresh server.

---

### STAGE 2 — 3D Geometry Reconstruction
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 2.1 FaceLift | 3D Gaussians from photo | ✅ face2mesh_facelift.py | ❌ Not downloaded (~12 GB) | Download on setup |
| 2.2 Multi-view depth render | 72-96 depth maps from Gaussians | ✅ face2mesh_facelift.py | N/A (uses Gaussians) | — |
| 2.3 Depth Anything V2 | Refine sparse depths | ✅ pipeline_complete.py | ❌ Not downloaded (~4 GB) | Download on setup |
| 2.4 BiSeNet masking | Filter non-head depth | ✅ (from Stage 1) | (shared with Stage 1) | — |
| 2.5 TSDF fusion | Watertight mesh via Open3D | ✅ face2mesh_facelift.py | N/A (Open3D lib) | **Python must be 3.11** — Open3D not available on 3.12/3.14 |
| 2.6 Mesh cleanup | Decimation, hole fill | ✅ face2mesh_facelift.py | N/A (trimesh) | — |

**Gap:** Python version critical — use 3.11. FaceLift checkpoint download on setup.

---

### STAGE 3 — High-Detail Texture Views (Qwen)
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 3.1 Qwen + MultiAngles LoRA | 24 views, camera-controlled | ✅ stages/stage3_multiview.py | ❌ GGUF file not downloaded | `qwen-image-edit-2511-Q4_0.gguf` from unsloth/Qwen-Image-Edit-2511-GGUF |
| 3.2 24-view generation | 24 angles at 3 elevation rings | ✅ stage3_multiview.py | (same as above) | — |
| 3.3 AnyPose correction | Optional pose correction pass | ❌ Not integrated | ❌ | Low priority |
| 3.4 ArcFace identity gate | Reject CSIM < 0.6 views | ✅ stage3_multiview.py | (shared Stage 1) | — |

**Gap:** GGUF model download. BitsAndBytes/NF4 approach removed — use GGUF only.

---

### STAGE 4 — Mesh, Textures & PBR Materials
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 4a FLAME retopology | Shrinkwrap onto MICA FLAME mesh | ✅ stages/stage4_textures.py | **FLAME registration required** | Register at flame.is.tue.mpg.de — place `generic_model.pkl` at `~/.cache/face_models/flame/` |
| 4b UV unwrap | FLAME UV layout or xatlas | ✅ stage4_textures.py | xatlas (pip) | — |
| 4c Texture baking | Multi-view projection to UV | ✅ stage4_textures.py | N/A | — |
| 4d PBR maps | AlbedoMM / DSINE / Marigold | ✅ stage4_textures.py | ❌ Not downloaded (~7 GB total) | Download on setup |
| 4e HRN pore detail | Deformation + displacement maps | ✅ stage4_textures.py | ❌ Not downloaded (~8 GB) | Download on setup |
| 4f Dynamic microstructure | Strain tensor shader | ✅ stage4_textures.py | N/A (shader) | Renderer-side setup |

**Gap:** FLAME model needs manual registration + download. PBR models need downloading.

---

### STAGE 5 — Rig & Blendshapes
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 5.1 FLAME rig transfer | LBS weights for 5 joints | ✅ stages/stage5_rig.py | FLAME pkl (same as Stage 4) | Shared dependency |
| 5.2 50+ blendshapes | Expression deltas from FLAME | ✅ stage5_rig.py | FLAME pkl | Shared dependency |
| 5.3 Corrective blendshapes | Compound-pose corrections | ✅ stage5_rig.py | N/A | — |
| 5.4 Skinning verification | Extreme pose artifact check | ✅ stage5_rig.py | N/A | — |

**Gap:** Entirely dependent on FLAME pkl. No additional downloads needed beyond Stage 4.

---

### STAGE 6 — Eyes, Teeth, Tongue, Detail Geometry
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 6a 4-layer eyes | Sclera/iris/cornea/pupil meshes | ✅ stages/stage6_detail.py | N/A (procedural) | — |
| 6b Teeth (168 tris) | Jaw-rigged teeth mesh | ✅ stage6_detail.py | N/A (procedural) | — |
| 6c Tongue (500-1000 tris) | 6-8 blendshapes + rig | ✅ stage6_detail.py | N/A (procedural) | — |
| 6d Inner mouth cavity | Cavity mesh behind teeth | ✅ stage6_detail.py | N/A (procedural) | — |
| 6e Eyelashes/eyebrows | Hair cards from BiSeNet mask | ✅ stage6_detail.py | N/A (procedural) | — |
| 6f Inner ear template | Template shrinkwrap fit | ✅ stage6_detail.py | N/A (procedural) | — |

**Gap:** None blocking. All procedural geometry. Depends on BiSeNet mask from Stage 1.

---

### STAGE 7 — Hair Reconstruction
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| Option A NeuralHaircut | Strand-based hair curves | ✅ stages/stage7_hair.py | ❌ Not downloaded (~10 GB) | Low priority — use Option C for fresh start |
| Option B Hair cards | Polygon strip hairstyle | ✅ stage7_hair.py | N/A | — |
| Option C TSDF volume | Textured solid hair volume | ✅ stage7_hair.py | N/A (from Stage 2) | Recommended default |

**Gap:** NeuralHaircut optional. Default to TSDF volume hair.

---

### STAGE 8 — Lighting & Materials
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 8a DiffusionLight | HDRI environment estimation | ✅ stages/stage8_lighting.py | ❌ Not downloaded (~8 GB) | Download on setup (or skip for basic output) |
| 8b IC-Light | Relight for target scene | ✅ stage8_lighting.py | ❌ Not downloaded (~8 GB) | Download on setup |
| 8c ACEScg conversion | sRGB → linear ACEScg | ✅ stage8_lighting.py | N/A | — |
| 8d Renderer translation | Arnold/RenderMan/Cycles/Unreal | ✅ stage8_lighting.py | N/A | Renderer-specific |

**Gap:** DiffusionLight and IC-Light need downloading. Both can be skipped for basic output.

---

### STAGE 9 — Animation
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 9a EMOTE / DiffPoseTalk | Audio-driven FLAME params | ✅ stages/stage9_animation.py | ❌ Not downloaded (~4 GB) | Download on setup |
| 9b SMIRK video-driven | Per-frame expression transfer | (shared with Stage 1) | (shared) | — |
| 9c Audio2Face-3D v3.0 | Full face + tongue + eyes | ✅ stage9_animation.py | ❌ Not downloaded (~4 GB) | Nvidia download required |
| 9d Temporal smoothing | Per-parameter filtering | ✅ stage9_animation.py | N/A | — |

**Gap:** Animation models need downloading. Pipeline works without them (outputs static avatar).

---

### STAGE 10 — Validation & Export
| Step | Walkthrough Requirement | Code | Model | Gap |
|------|------------------------|------|-------|-----|
| 10a ArcFace CSIM | Identity similarity check | ✅ stages/stage10_validation.py | (shared Stage 1) | — |
| 10b LPIPS / SSIM | Perceptual quality metrics | ✅ stage10_validation.py | pyiqa (pip) | — |
| 10c Geometric validation | Watertight, no intersections | ✅ stage10_validation.py | trimesh (pip) | — |
| 10d Animation smoke test | 10-sec render + frame check | ✅ stage10_validation.py | N/A | — |

**Gap:** None blocking. All pip-installable.

---

## Setup Checklist for Fresh Server

### Must-Have (pipeline fails without these)
- [ ] Python **3.11** — required for Open3D
- [ ] CUDA 12.1 + PyTorch 2.4 (match exactly)
- [ ] `pip install -r requirements.txt`
- [ ] `pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization` (CUDA compile)
- [ ] Clone FaceLift: `git clone https://github.com/weijielyu/FaceLift /FaceLift && bash /FaceLift/setup_env.sh`
- [ ] Download `qwen-image-edit-2511-Q4_0.gguf` from [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)
- [ ] Register at https://flame.is.tue.mpg.de/ → place `generic_model.pkl` at `~/.cache/face_models/flame/`
- [ ] ArcFace antelopev2 (~1 GB), MICA (~2 GB), SMIRK (~2 GB), BiSeNet (~1 GB)

### Should-Have (quality drops without these)
- [ ] Depth Anything V2 (~4 GB)
- [ ] DSINE (~3 GB), Marigold (~4 GB), AlbedoMM (~1 GB)
- [ ] HRN (~8 GB)

### Nice-to-Have (can defer)
- [ ] DiffusionLight (~8 GB), IC-Light (~8 GB)
- [ ] NeuralHaircut (~10 GB)
- [ ] EMOTE / DiffPoseTalk (~4 GB), Audio2Face-3D (~4 GB)

---

## Known Problems (do not repeat)

See README.md → *Known Issues & Lessons Learned* for full list.

Critical ones for setup:
1. **Python must be 3.11** — Open3D breaks on 3.12+
2. **Pin PyTorch + CUDA together** — version mismatch causes silent failures
3. **No BitsAndBytes** — use GGUF Q4_0 via llama-cpp-python for Qwen
4. **FLAME registration is manual** — no automated download, plan for it upfront
5. **Vast.ai is ephemeral** — run setup script on every new instance, store models on persistent volume
