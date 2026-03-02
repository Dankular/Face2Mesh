# Film-Quality Avatar Pipeline — Complete Step-by-Step Walkthrough
## Single Photo to Puppeted 3D Head — Every Step, Every Instruction

---

## What You Are Building

You will take one photograph of a person and produce a film-quality 3D head avatar with pore-level detail, anatomically correct eyes, teeth, tongue, eyelashes, eyebrows, full PBR materials, and the ability to be puppeted from audio or video. The entire process runs in 25-40 minutes on a single RTX 4090.

**Input:** One photo (minimum 512×512 face region, both eyes visible, reasonable lighting).
**Output:** Rigged FLAME-topology mesh with 50+ blendshapes, 4K PBR texture set in ACEScg, hair, and a validation report.

---

## Prerequisites

**Hardware:** NVIDIA RTX 4090 (24GB) or A100 (40/80GB). 64GB+ RAM. 500GB SSD for models.

**Software:** PyTorch, Open3D, trimesh, xatlas. All models available via HuggingFace/GitHub.

**Models to download before starting:**

| Model | Source | VRAM |
|-------|--------|------|
| ArcFace (antelopev2) | insightface | ~1 GB |
| MICA | zielon/mica | ~2 GB |
| SMIRK | georgeretsi/smirk | ~2 GB |
| BiSeNet | face-parsing PyTorch | ~1 GB |
| FaceLift | weijielyu/FaceLift | ~12 GB |
| Depth Anything V3 | depth-anything-v3 | ~4 GB |
| Qwen-Image-Edit-2511 | Qwen/Qwen-Image-Edit-2511 | ~16 GB |
| MultiAngles LoRA | fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA | (loaded with Qwen) |
| Lightning LoRA | lightx2v/Qwen-Image-Edit-2511-Lightning | (loaded with Qwen) |
| HRN | youngLBW/HRN | ~8 GB |
| DSINE | DSINE | ~3 GB |
| Marigold IID | Marigold | ~4 GB |
| NeuralHaircut | KAIST NeuralHaircut | ~10 GB |
| IC-Light | IC-Light | ~8 GB |
| DiffusionLight | DiffusionLight | ~8 GB |
| EMOTE or DiffPoseTalk | MPI-IS / DiffPoseTalk | ~4 GB |
| NVIDIA Audio2Face-3D v3.0 | nvidia/Audio2Face-3D-v3.0 | ~4 GB |

---

## STAGE 1 — Extract Identity & Parameters (~10 seconds)

**Purpose:** Pull all parametric data from the input photo. This data feeds every downstream stage.

**Step 1.1 — Run ArcFace**
Feed the input photo through ArcFace (insightface antelopev2). Extract the 512-dimensional identity embedding vector. Save it — you will use this repeatedly to verify that generated views still look like the same person.

**Step 1.2 — Run MICA**
Feed the input photo through MICA. It outputs FLAME shape parameters (β, 300 dimensions) representing the person's face geometry in metric scale. This becomes your canonical FLAME mesh — the retopology target that everything conforms to. Save the FLAME mesh as `mica_shape.obj`.

**Step 1.3 — Run SMIRK**
Feed the input photo through SMIRK. It outputs FLAME expression parameters (ψ, 50 dimensions) plus jaw pose. If the person is smiling or making an expression in the photo, these parameters tell you what that expression is, so you can later generate a neutral version or transfer the expression to animation.

**Step 1.4 — Run BiSeNet**
Feed the input photo through BiSeNet face parsing. It outputs a per-pixel segmentation mask labelling skin, hair, left eye, right eye, eyebrows, upper lip, lower lip, inner mouth, nose, neck, background, and other regions. Save this mask — you will use it for texture region blending, depth filtering, eyelash/eyebrow placement, and compositing.

**You now have:** ArcFace embedding, FLAME shape mesh, FLAME expression params, face segmentation mask.

---

## STAGE 2 — Reconstruct 3D Geometry (2-5 minutes)

**Purpose:** Generate the actual 3D head mesh. This is the geometric foundation everything builds on.

**Step 2.1 — Run FaceLift**
Feed the input photo into FaceLift (weijielyu/FaceLift). It runs a multi-view diffusion model followed by GS-LRM to produce a 3D Gaussian Splat of the full head, including hallucinated back-of-head geometry. The output is a set of 3D Gaussians (position, colour, scale, rotation, opacity per splat).

**Step 2.2 — Render multi-view depth maps**
Set up 72-96 virtual cameras on an orbital ring around the Gaussian splat. Render an RGB image and a depth map from each camera. These cameras should cover the full sphere: eye-level ring, elevated ring, low ring, top-down, under-chin.

**Step 2.3 — Refine depth with Depth Anything V3**
For each rendered RGB image, run Depth Anything V3 to get a monocular depth estimate. Scale-align it to match the Gaussian-derived depth (affine alignment). The monocular depth fills holes at hair boundaries, thin ear structures, and anywhere the Gaussians are sparse.

**Step 2.4 — Apply BiSeNet masking**
Run BiSeNet on each rendered view. Zero out depth values in non-head regions (clothing, necklaces, background bleed). This prevents non-face geometry from entering the mesh.

**Step 2.5 — TSDF fusion**
Feed all refined, masked depth maps into Open3D's TSDF volume integration. Use voxel_length of 1-2mm for face detail, sdf_trunc of approximately 5× the voxel length. Run marching cubes to extract a watertight surface mesh.

**Step 2.6 — Clean up the mesh**
Remove small disconnected components (floating artifacts). Apply light Laplacian smoothing. Decimate with quadric edge collapse to your target polycount (50-200k triangles). Fill any remaining holes and repair normals using trimesh.

**You now have:** A raw head mesh (`.obj`) — geometrically accurate but with arbitrary topology and no UV map.

---

## STAGE 3 — Generate High-Detail Texture Views (2-4 minutes)

**Purpose:** Create 24 high-quality, identity-consistent renders of the face from controlled angles. These replace FaceLift's Gaussian colours as the texture source.

**Step 3.1 — Set up Qwen with LoRAs**
Load Qwen-Image-Edit-2511 as the base model. Attach the MultiAngles LoRA (fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) for camera control. Attach the Lightning LoRA (lightx2v/Qwen-Image-Edit-2511-Lightning) for 4-step fast inference.

**Step 3.2 — Generate 24 views**
For each of 24 target viewpoints, run Qwen with the input photo and a prompt specifying the camera angle:

- Eye-level ring (8 views): front, front-right, right, back-right, back, back-left, left, front-left. Prompt: `<sks> [direction] view eye-level shot close-up`
- Elevated ring (8 views): same 8 azimuths at high angle. Prompt: `<sks> [direction] view high-angle shot close-up`
- Low ring (4 views, front hemisphere): front, front-right, front-left, right. Prompt: `<sks> [direction] view low-angle shot close-up`
- Top-down (2 views): front and back elevated. Prompt: `<sks> [direction] view elevated shot medium shot`
- Under-chin (2 views): front and back low. Prompt: `<sks> [direction] view low-angle shot close-up`

**Step 3.3 — [Optional] AnyPose correction**
If any views show expression drift (person smiling when they should be neutral, or head tilted wrong), run a second pass with the AnyPose LoRA (lilylilith/AnyPose). Feed it the MultiAngles output as image 1 and the corresponding FaceLift Gaussian render as image 2 (pose reference). Use both base and helper LoRA at 0.7 strength.

**Step 3.4 — Verify identity**
For each of the 24 generated views, extract the ArcFace embedding and compute cosine similarity against the source photo embedding from Step 1.1. Reject any view scoring below 0.6. Regenerate rejected views with a different random seed. Select the best of N candidates per angle.

**You now have:** 24 high-quality, identity-verified face renders at known camera angles.

---

## STAGE 4 — Assemble Mesh, Textures & Materials (5-10 minutes)

**Purpose:** Convert the raw mesh into an animation-ready, textured, PBR-complete asset with pore-level detail.

### 4a. Retopologise to FLAME

**Step 4a.1** — Take the MICA FLAME mesh from Step 1.2 as your template.
**Step 4a.2** — Subdivide it 1-2 times to increase resolution.
**Step 4a.3** — Shrinkwrap-project every vertex onto the nearest point on the TSDF mesh surface.
**Step 4a.4** — Apply light Laplacian smoothing to fix projection artifacts at sharp curvature.

The result is a mesh where every vertex has known semantic meaning (vertex 1423 is always the nose tip). This is essential for blendshapes and rigging.

**Step 4a.5 — Handle the neck boundary.** The FLAME mesh ends at the neck base. Add 3-5 extra edge loops below FLAME's boundary, projecting downward following the neck cylinder. Smooth-blend the skinning weights from FLAME's neck joint. Alternatively, create a simple collar mesh that overlaps and blends with the neck boundary.

### 4b. UV Unwrap

If using FLAME topology (recommended), use FLAME's standard UV layout that ships with the model. All textures will be interchangeable with any FLAME-based asset. If using custom topology, run xatlas automatic UV parameterisation.

### 4c. Bake Textures from Qwen Views

**Step 4c.1** — For each face (triangle) on the retopologised mesh, determine which of the 24 Qwen views sees it most frontally (highest dot product between face normal and view direction).
**Step 4c.2** — Project the pixel colour from the best view onto UV space.
**Step 4c.3** — Where a face is visible from multiple views, use cosine-weighted blending.
**Step 4c.4** — Apply region-aware blending: use Qwen views exclusively for face skin (highest detail), blend Qwen with FaceLift renders for hair, fall back to FaceLift for neck/ears where Qwen coverage is thin.
**Step 4c.5** — Inpaint any remaining gaps with nearest-neighbour or diffusion inpainting.

### 4d. Generate PBR Material Maps

From the baked diffuse texture, generate the full PBR set:

**Albedo** — Run AlbedoMM or Marigold IID to produce a de-lit diffuse colour map (shadows and highlights removed).
**Normal map** — Run DSINE, which is camera-intrinsics-aware and produces crisp normals at occlusion boundaries (nose edge, jawline).
**Roughness** — Run Marigold IID intrinsic image decomposition. Skin roughness varies by region (forehead oilier/smoother, cheeks drier/rougher).
**Specular** — From Marigold IID or set manually per region.
**SSS / Translucency** — Estimate from the BiSeNet skin mask combined with depth. Ears, nostrils, and thin skin around eyes need higher translucency values.
**Displacement** — Compute the difference between the TSDF mesh surface and the FLAME shrinkwrap. This captures geometric detail that the retopology smoothed out.

### 4e. Extract Pore-Level Microdetail

This is what separates film quality from game quality. Three frequency bands, matching Hollywood light stage methodology:

**Step 4e.1 — Run HRN (mid + high frequency)**
Feed the original reference photo and 2-4 of the best frontal Qwen views into HRN (youngLBW/HRN). It uses geometry disentanglement with two pix2pix networks to produce:
- `deformation_map.exr` — UV-space vertex offsets for mid-frequency detail (wrinkles, nasolabial folds, dimples). These are expression-dependent.
- `displacement_map.exr` — Scalar displacement for high-frequency detail (pores, fine lines, skin grain). These are static.

If using multi-view mode (MV-HRN), feed multiple Qwen renders for view-consistent detail maps.

**Step 4e.2 — Run UHR facial texture reconstruction**
For 4K pore-level detail beyond what HRN provides, run the UHR approach (Huang et al. 2025). It uses divide-and-conquer with hash encoding on UV coordinates. Outputs:
- `micronormal_map_4k.exr` — Surface orientation detail at individual pore scale
- `microspecular_map_4k.exr` — Per-pore specular variation (some pores are oilier)

**Step 4e.3 — Procedural fill for occluded regions**
For areas not visible in the source photo (back of ears, under chin, scalp behind hairline): sample pore statistics from visible regions (density, size, orientation), then synthesise matching microdetail using Gabor noise or patch-based texture synthesis. Blend at boundaries with the data-driven detail.

**Step 4e.4 — Build expression-dependent displacement library**
For each of 10-15 key expressions (neutral, smile, frown, brow-raise, brow-furrow, jaw-open, eye-squeeze, lip-pucker, nose-wrinkle, surprise):
1. Generate a Qwen view showing that expression using SMIRK expression params
2. Run HRN on each expression view to produce `deformation_map_[expression].exr`

At runtime, blend these maps matching FLAME blendshape weights:
```
active_deformation = deformation_neutral
for each active blendshape bs_i with weight w_i:
    delta = deformation_map[bs_i] - deformation_neutral
    active_deformation += w_i * delta
```

### 4f. Set Up Dynamic Skin Microstructure

Static pore maps look correct at rest but wrong during animation. When skin stretches (smiling), pores flatten and skin gets shinier. When skin compresses (frowning), pores deepen and skin gets rougher.

**Step 4f.1** — For each animated frame, compute per-vertex strain from the deformation gradient (compare vertex positions to neutral rest pose). Calculate local stretch ratio per triangle.

**Step 4f.2** — Map to UV space as a tension map: red channel = stretch magnitude, green = U direction, blue = V direction.

**Step 4f.3** — Apply anisotropic convolution to the displacement map at render time:
- Stretch regions: Gaussian blur along stretch direction (smooths pores)
- Compression regions: sharpen along compression direction (deepens pores)

**Step 4f.4** — Simultaneously modulate roughness: decrease for stretched skin (shinier when taut), increase for compressed skin (rougher when wrinkled).

This runs as a GPU shader pass. The tension map can be precomputed per blendshape and blended at runtime, or computed dynamically from vertex positions.

**You now have:** Animation-ready mesh with FLAME topology, UV-mapped, full PBR texture set, displacement maps at three frequency bands, micronormal maps, and dynamic microstructure setup.

---

## STAGE 5 — Rig & Generate Blendshapes (~30 seconds)

**Purpose:** Make the mesh puppetable.

**Step 5.1 — Transfer FLAME rig**
FLAME provides linear blend skinning weights for 5 joints (neck, jaw, 2× eyeball, head). Transfer them directly to your retopologised mesh — the vertex correspondence is already established.

**Step 5.2 — Generate 50+ blendshapes**
FLAME provides 50 expression blendshapes plus jaw pose. For each blendshape:
1. Apply it to the canonical FLAME mesh to get the deformed shape
2. Shrinkwrap the deformed shape onto the corresponding deformed TSDF geometry
3. Store the vertex delta (deformed minus neutral)

This gives you blendshapes that deform with the actual face detail, not just the parametric model.

**Step 5.3 — Add corrective blendshapes**
For problematic combinations (smile + blink causing cheek-eyelid intersection, extreme jaw open + lip pucker), sculpt or procedurally generate corrective shapes. These activate only when multiple blendshapes fire simultaneously.

**Step 5.4 — Verify skinning**
Test weight painting around jaw, neck, and eye boundaries. Drive the rig to extreme poses and check for artifacts before proceeding.

**You now have:** Rigged mesh with 50+ blendshapes, ready for animation.

---

## STAGE 6 — Build Eyes, Teeth, Tongue & Detail Geometry (~1 minute)

**Purpose:** FLAME has placeholder eye and mouth regions. Film requires proper anatomy.

### 6a. Assemble Layered Eyes

Build each eye from 4 nested meshes:

**Sclera** — Sphere (~12mm radius, slightly oblate). Material: off-white diffuse with blue/pink variation, SSS diffusion profile (~2mm scatter radius), blood vessel normal map, roughness 0.3-0.5.

**Iris disc** — Concave disc inset ~0.5mm behind the cornea surface. Extract iris texture from the source photo: detect iris via BiSeNet, unwrap to polar coordinates, inpaint specular highlights and eyelid occlusions, generate a radial fibre normal map from luminance gradient. Material: roughness 0.4-0.6.

**Pupil** — Opening in the iris disc centre. Controllable dilation by scaling the iris inner radius.

**Cornea** — Convex dome extending ~2mm beyond the sclera at centre. Material: near-perfect transmission, IOR 1.376 refraction, specular ~0.9, roughness ~0.02. Add a limbal darkening ring at the cornea-sclera boundary.

**Set up gaze control:** Add a look-at constraint target decoupled from head pose for director control. Generate procedural micro-saccades (2-5 Hz, <1 degree) for life. Coordinate blink timing with gaze direction (eyes look down slightly during blinks).

**Add tear film:** Place a thin mesh strip along the lower eyelid margin with high specular to simulate the tear meniscus caustic.

### 6b. Insert Teeth

Use the GaussianAvatars approach: 168 triangles for upper and lower teeth. Rig upper teeth to the head joint, lower teeth to the jaw joint. Material: slightly translucent SSS, subtle yellowing variation, wet specular.

### 6c. Build Tongue (Required for Speech)

Create a ~500-1000 triangle tongue mesh positioned behind the lower teeth. Rig with 3-4 bones (root on jaw joint, mid, tip) for curl/extend/lateral movement. Build 6-8 blendshapes mapped to ARKit shapes: tongueOut, tongueUp, tongueDown, tongueLeft, tongueRight, tongueCurl, tongueWide, tongueNarrow. Material: wet, pink, SSS-heavy, subtle papillae bump texture.

For animation, NVIDIA Audio2Face-3D v3.0 outputs tongue motion natively. For EMOTE/DiffPoseTalk, map jaw-open + viseme class to tongue blendshapes via a lookup table.

### 6d. Build Inner Mouth Cavity

Create a cavity mesh from lip inner edge to throat, behind teeth and tongue. Texture with dark red/pink gradient, wet specular. Add SSS on gums (pinker than skin). Uvula geometry optional.

### 6e. Add Eyelashes & Eyebrows

**Eyelashes:** Place 4-6 hair cards per eye (upper and lower) along the eyelid margin using BiSeNet eye segmentation as a guide. Alternatively use the Kerbiriou parametric eyelash model (CGF 2024) for data-driven generation with semantic control. Rig to eyelid blendshapes so lashes follow during blinks. Material: dark, slightly translucent, anisotropic specular (hair shader).

**Eyebrows:** Place 8-12 hair cards per brow following natural growth direction (medial: upward, mid-brow: lateral, tail: downward/lateral). Use the EMS model (Li et al. 2023) for single-view eyebrow reconstruction, or manually place cards using the BiSeNet brow mask from the source photo. Material: hair shader matching scalp hair but finer.

### 6f. Fit Inner Ear Template

The TSDF mesh produces a smooth blob in the ear cavity. Take an anatomical ear template (~2000-5000 tris) from the Universal Head 3DMM (steliosploumpis/Universal_Head_3DMM) and shrinkwrap/deform it into the TSDF ear region. Scale and position to match the ear outline in the source photo and Qwen side-view renders. Texture from Qwen side views. Ear material uses slightly higher SSS than face skin (light shines through ears).

**You now have:** Complete head assembly with layered eyes, teeth, tongue, inner mouth, eyelashes, eyebrows, and detailed ears.

---

## STAGE 7 — Reconstruct Hair (5-10 minutes)

**Purpose:** Add hair to the head.

**Option A — NeuralHaircut (film quality):** Run NeuralHaircut on the reference photo. Produces actual hair strand curves that render with hair-specific shaders. Best for hero shots. Slow, high VRAM.

**Option B — Hair cards (real-time):** Extract the hair region from BiSeNet segmentation. Generate textured polygon strips approximating the hairstyle. Much faster, suitable for game engines.

**Option C — TSDF hair volume (quick):** The TSDF mesh already includes a solid hair volume. Texture it from the Qwen views. No strand detail but trivial to integrate.

**Recommendation:** Start with Option C for rapid iteration. Add NeuralHaircut strands for final delivery.

---

## STAGE 8 — Set Up Lighting & Materials (1-2 minutes)

### 8a. Estimate Environment Lighting

Run DiffusionLight on the original reference photo. It estimates an HDRI environment map of the lighting the face was captured under. Use this as a reference for matching your target scene.

### 8b. Relight for Target Scene

Run IC-Light to relight the face texture under your target scene's lighting conditions. This generates texture variants lit from different directions.

### 8c. Convert Colour Space (Critical for Production)

All diffusion model outputs (Qwen, HRN, UHR) are in sRGB. Film production works in linear ACEScg.

**Albedo/Diffuse maps:** Apply inverse sRGB OETF, then convert primaries to ACEScg.
**Normal/Displacement/Roughness:** Verify these are linear data — if they look too contrasty, apply inverse sRGB.
**SSS profiles:** These behave differently per renderer. Calibrate per target.

### 8d. Translate Materials for Your Renderer

| Target | Shader | Key Adjustments |
|--------|--------|----------------|
| Arnold | aiStandardSurface | SSS uses randomwalk_v2, map translucency radius per region |
| RenderMan | PxrSurface | Separate diffuse/specular lobes, subsurface uses burley |
| Cycles | Principled BSDF | SSS scale needs manual tuning, displacement requires adaptive subdivision |
| Unreal | Subsurface Profile | Bake SSS into profile asset, separate specular from base colour |

Maintain all textures in ACEScg linear EXR as the canonical format. Export renderer-specific sets via conversion script.

---

## STAGE 9 — Animate & Puppet (Real-time to minutes)

### 9a. Audio-Driven Animation

Feed speech audio into one of:

| Model | Best For | Output |
|-------|----------|--------|
| EMOTE | Emotional speech | FLAME expression + jaw params |
| DiffPoseTalk | Stylistic diversity | FLAME expression + head pose |
| FaceFormer | Solid lip-sync | FLAME mesh vertices |
| Audio2Face-3D v3.0 | Full face + tongue + eyes | Skin deformation + tongue + jaw + eyeball motion, ARKit blendshape weights |

Audio2Face-3D is the most complete option — it drives tongue and eyes natively alongside face skin.

### 9b. Video-Driven Animation

Feed actor video into:

| Model | Best For | Output |
|-------|----------|--------|
| SMIRK | Real-time transfer | FLAME expression per frame |
| EMOCA | Emotional accuracy | FLAME + emotion params |
| LivePortrait | Quick 2D preview | 2D reanimated image |

### 9c. Manual / Keyframe

Directly manipulate FLAME blendshape weights via animation curves in Blender/Maya. Most control, most labour.

### 9d. Apply Temporal Smoothing (Mandatory)

Every animation driver needs temporal filtering to remove jitter.

**Per-parameter tuning:**
- Jaw pose: light smoothing (speech needs sharp transitions for consonants)
- Expression blendshapes: moderate smoothing (preserve emotion transitions)
- Head pose: heavier smoothing (natural head movement is slow, jitter very visible)
- Eye gaze: minimal on saccades (they are naturally fast), heavy on fixation periods

Use an exponential moving average for real-time, Savitzky-Golay for offline, or Kalman filter for best real-time quality.

**Recommended workflow for film:** EMOTE/DiffPoseTalk for initial animation from script audio, refine with SMIRK from actor performance video, apply temporal smoothing, then hand-correct remaining issues.

---

## STAGE 10 — Validate Quality (~30 seconds)

**Purpose:** Automated check before delivery. Catches cascading errors.

### 10a. Identity Verification

Render the final rigged avatar from 8 viewpoints (matching Stage 3 angles). Extract ArcFace embeddings, compare to source photo. Mean cosine similarity must exceed 0.65 across all views.

### 10b. Perceptual Quality

| Metric | Tool | Target |
|--------|------|--------|
| LPIPS (frontal) | pyiqa | < 0.25 |
| LPIPS (side) | pyiqa | < 0.35 |
| ArcFace CSIM (mean) | insightface | > 0.65 |
| SSIM (frontal) | pyiqa | > 0.75 |

Note: Per CVPR 2025 avatar challenge findings, these metrics catch errors but don't guarantee subjective realism. Final artistic review remains essential.

### 10c. Geometric Validation

- Mesh watertightness (trimesh)
- No self-intersections in neutral pose or any blendshape at 100%
- Jaw open + tongue out: verify no clipping through teeth

### 10d. Animation Smoke Test

Drive the avatar with a standard 10-second speech clip. Render at 30fps. Automated check: compute per-frame FLAME parameter variance, flag frames where delta exceeds 2 standard deviations. Visual inspect for jaw jitter, expression pops, eye drift, neck seam visibility.

**Output:** Quality report (JSON) with pass/fail per metric and flagged frames for manual review.

---

## Summary — The Complete Flow

```
PHOTO IN
    │
    ├─> [1] ArcFace → identity embedding
    ├─> [1] MICA → FLAME shape mesh
    ├─> [1] SMIRK → expression params
    ├─> [1] BiSeNet → segmentation mask
    │
    ├─> [2] FaceLift → 3D Gaussians
    ├─> [2] Render 72-96 depth maps
    ├─> [2] Depth Anything V3 → refine depths
    ├─> [2] BiSeNet mask → filter non-head
    ├─> [2] TSDF fusion → raw mesh
    ├─> [2] Cleanup → decimated mesh
    │
    ├─> [3] Qwen + MultiAngles LoRA → 24 views
    ├─> [3] ArcFace gate → reject bad views
    │
    ├─> [4a] Shrinkwrap onto FLAME → retopo mesh
    ├─> [4b] FLAME UV layout
    ├─> [4c] Bake Qwen views → diffuse texture
    ├─> [4d] AlbedoMM/DSINE/Marigold → PBR maps
    ├─> [4e] HRN → deformation + displacement maps
    ├─> [4e] UHR → 4K micronormal + microspecular
    ├─> [4e] Expression displacement library (10-15 maps)
    ├─> [4f] Strain tensor → dynamic microstructure shader
    │
    ├─> [5] FLAME rig transfer → 5 joints
    ├─> [5] 50+ blendshapes + correctives
    │
    ├─> [6a] 4-layer eyes (sclera/iris/cornea/pupil)
    ├─> [6b] Teeth (168 tris, jaw-rigged)
    ├─> [6c] Tongue (500-1000 tris, 6-8 blendshapes)
    ├─> [6d] Inner mouth cavity
    ├─> [6e] Eyelash + eyebrow hair cards
    ├─> [6f] Inner ear template fit
    │
    ├─> [7] NeuralHaircut / hair cards / TSDF volume
    │
    ├─> [8a] DiffusionLight → HDRI estimate
    ├─> [8b] IC-Light → relight for target scene
    ├─> [8c] sRGB → ACEScg conversion
    ├─> [8d] Renderer-specific material export
    │
    ├─> [9] EMOTE/DiffPoseTalk/SMIRK/Audio2Face → animation
    ├─> [9d] Temporal smoothing (per-parameter)
    │
    └─> [10] ArcFace + LPIPS + geometric + smoke test
            │
            ▼
      VALIDATED AVATAR OUT
      (mesh + rig + textures + hair + report)
```

**Total time: 25-40 minutes. One photo in, film-ready puppet out.**

---

*Document version: 2.0 | Pipeline: 10-stage with validation | March 2026*
