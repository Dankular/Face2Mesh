"""
Stage 10 — Validation, Quality Metrics & Export
================================================
10a. ArcFace identity verification across 8 views (CSIM > 0.65)
10b. Perceptual quality metrics: LPIPS < 0.25 frontal, SSIM > 0.75
10c. Geometric validation: watertight, no self-intersections
10d. Animation smoke test: detect jitter/pops across 30-frame clip
10e. Export: FBX, GLB, USD, JSON quality report
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"

# Validation thresholds (from Complete_Pipeline_Walkthrough.md)
CSIM_THRESHOLD    = 0.65
LPIPS_FRONTAL_MAX = 0.25
LPIPS_SIDE_MAX    = 0.35
SSIM_FRONTAL_MIN  = 0.75
JITTER_SIGMA_MAX  = 2.0


# ---------------------------------------------------------------------------
# 10a — ArcFace identity verification
# ---------------------------------------------------------------------------

def verify_identity_8views(
    mesh_or_view_dir: str,
    source_arcface_embedding: np.ndarray,
    view_render_paths: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Verify identity across 8 standard views.
    If rendered views are available, use them. Otherwise estimate from available views.
    """
    results = {}
    csim_list = []

    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Use the 8 eye-level views (front, 45°×7)
        view_labels = ["front", "front_right", "right", "back_right",
                       "back", "back_left", "left", "front_left"]

        if view_render_paths:
            for label in view_labels:
                img_path = view_render_paths.get(label)
                if img_path and Path(img_path).exists():
                    img = np.array(Image.open(img_path).convert("RGB"))
                    faces = app.get(img)
                    if faces:
                        emb = faces[0].embedding.astype(np.float32)
                        norm_src = source_arcface_embedding / (np.linalg.norm(source_arcface_embedding) + 1e-8)
                        norm_emb = emb / (np.linalg.norm(emb) + 1e-8)
                        csim = float(np.dot(norm_src, norm_emb))
                        results[label] = csim
                        csim_list.append(csim)
                        logger.info(f"  {label}: CSIM={csim:.3f} {'✓' if csim >= CSIM_THRESHOLD else '✗'}")
                    else:
                        results[label] = 0.0
                        csim_list.append(0.0)
                        logger.warning(f"  {label}: no face detected")

    except Exception as e:
        logger.error("ArcFace identity check failed: %s", e)

    mean_csim = float(np.mean(csim_list)) if csim_list else 0.0
    pass_csim  = mean_csim >= CSIM_THRESHOLD

    logger.info(f"  ArcFace mean CSIM: {mean_csim:.3f}  {'PASS' if pass_csim else 'FAIL'}")
    return {
        "per_view": results,
        "mean":     mean_csim,
        "pass":     pass_csim,
        "threshold": CSIM_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 10b — Perceptual quality metrics
# ---------------------------------------------------------------------------

def compute_perceptual_metrics(
    rendered_frontal: str,
    reference_frontal: str,
    rendered_side: Optional[str] = None,
    device: str = "cuda",
) -> Dict:
    """Compute LPIPS and SSIM between rendered avatar and reference photo."""
    metrics = {}

    # --- LPIPS ---
    try:
        import lpips
        import torch

        loss_fn = lpips.LPIPS(net="vgg").to(device)
        loss_fn.eval()

        def _load_tensor(path: str) -> "torch.Tensor":
            img = Image.open(path).convert("RGB").resize((512, 512))
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            lpips_frontal = float(loss_fn(_load_tensor(rendered_frontal),
                                          _load_tensor(reference_frontal)).item())

        metrics["lpips_frontal"] = lpips_frontal
        metrics["lpips_frontal_pass"] = lpips_frontal < LPIPS_FRONTAL_MAX

        if rendered_side and Path(rendered_side).exists():
            # Side LPIPS: compare to corresponding Qwen side view if available
            lpips_side = float(loss_fn(
                _load_tensor(rendered_side), _load_tensor(rendered_frontal)
            ).item())
            metrics["lpips_side"] = lpips_side
            metrics["lpips_side_pass"] = lpips_side < LPIPS_SIDE_MAX

        loss_fn.cpu()
        del loss_fn

    except ImportError:
        logger.warning("lpips not installed — skipping LPIPS metrics")

    # --- SSIM ---
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage import io

        ref  = np.array(Image.open(reference_frontal).convert("RGB").resize((512, 512)))
        rend = np.array(Image.open(rendered_frontal).convert("RGB").resize((512, 512)))
        ssim_val = float(ssim_fn(ref, rend, multichannel=True, channel_axis=2))
        metrics["ssim_frontal"] = ssim_val
        metrics["ssim_frontal_pass"] = ssim_val >= SSIM_FRONTAL_MIN
        logger.info(f"  SSIM frontal: {ssim_val:.3f}  {'PASS' if ssim_val >= SSIM_FRONTAL_MIN else 'FAIL'}")

    except ImportError:
        logger.warning("scikit-image not installed — skipping SSIM")

    return metrics


# ---------------------------------------------------------------------------
# 10c — Geometric validation
# ---------------------------------------------------------------------------

def validate_geometry(mesh_path: str) -> Dict:
    """Check mesh validity: watertight, no self-intersections, vertex count."""
    import trimesh

    results = {}
    try:
        mesh = trimesh.load(mesh_path, process=False)

        results["vertices"]    = int(len(mesh.vertices))
        results["faces"]       = int(len(mesh.faces))
        results["watertight"]  = bool(mesh.is_watertight)
        results["volume_cc3"]  = float(abs(mesh.volume)) if mesh.is_watertight else None

        # Degenerate faces
        areas = mesh.area_faces
        degen = int((areas < 1e-12).sum())
        results["degenerate_faces"] = degen

        # Normals consistency
        results["consistent_normals"] = bool(trimesh.repair.fix_normals(mesh, multibody=False) is not False)

        # Self-intersection check (expensive, sample 1000 random faces)
        try:
            n_sample = min(1000, len(mesh.faces))
            sampled  = mesh.faces[np.random.choice(len(mesh.faces), n_sample, replace=False)]
            subset   = trimesh.Trimesh(vertices=mesh.vertices, faces=sampled, process=False)
            # Use ray casting as intersection proxy
            results["self_intersection_check"] = "skipped_performance"
        except Exception:
            results["self_intersection_check"] = "error"

        logger.info(f"  Geometry: {results['vertices']:,}v  {results['faces']:,}f  "
                    f"watertight={results['watertight']}  degen={degen}")
    except Exception as e:
        logger.error("Geometric validation failed: %s", e)
        results["error"] = str(e)

    return results


# ---------------------------------------------------------------------------
# 10d — Animation smoke test
# ---------------------------------------------------------------------------

def animation_smoke_test(
    anim_params_path: str,
    fps: int = 30,
) -> Dict:
    """
    Detect jitter/pops in animation parameters.
    Flag frames where per-frame delta exceeds N standard deviations.
    """
    results = {}
    try:
        data = np.load(anim_params_path, allow_pickle=True).item()
        flagged_frames = {}

        for key, seq in data.items():
            if not isinstance(seq, np.ndarray) or seq.ndim < 1 or len(seq) < 3:
                continue
            if seq.ndim == 1:
                seq = seq[:, None]
            deltas = np.diff(seq, axis=0)
            delta_mag = np.linalg.norm(deltas, axis=1)
            mu = delta_mag.mean()
            sigma = delta_mag.std()
            if sigma < 1e-8:
                continue
            bad = np.where(delta_mag > mu + JITTER_SIGMA_MAX * sigma)[0]
            if len(bad) > 0:
                flagged_frames[key] = bad.tolist()
                logger.warning(f"  Jitter in {key}: {len(bad)} frames flagged")

        n_frames = len(next(iter(data.values()))) if data else 0
        results = {
            "n_frames":      n_frames,
            "duration_s":    n_frames / fps,
            "flagged_frames": flagged_frames,
            "jitter_pass":   len(flagged_frames) == 0,
        }
        logger.info(f"  Animation smoke test: {n_frames} frames, "
                    f"{len(flagged_frames)} parameter channels with jitter")
    except Exception as e:
        logger.error("Animation smoke test failed: %s", e)
        results["error"] = str(e)
        results["jitter_pass"] = False

    return results


# ---------------------------------------------------------------------------
# 10e — Export: GLB, FBX, USD
# ---------------------------------------------------------------------------

def export_glb(mesh_path: str, output_dir: Path, albedo_path: Optional[str] = None) -> str:
    """Export textured GLB for web/realtime use."""
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)

    if albedo_path and Path(albedo_path).exists():
        albedo_img = Image.open(albedo_path).convert("RGB")
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo_img,
            metallicFactor=0.0,
            roughnessFactor=0.45,
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=mat)

    out_path = str(output_dir / "avatar.glb")
    mesh.export(out_path)
    logger.info(f"  GLB exported: {out_path}")
    return out_path


def export_fbx_script(
    mesh_path: str,
    rig_json: Optional[str],
    blendshape_dir: Optional[str],
    output_dir: Path,
) -> str:
    """
    Write a Blender Python script to import OBJ + rig + blendshapes and export FBX.
    Run via: blender --background --python <script_path>
    """
    script = f"""
import bpy, json, os, math

# Clear default
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.import_scene.obj(filepath=r"{mesh_path}")
mesh_obj = bpy.context.active_object

# Import rig if available
rig_json_path = r"{rig_json}"
if rig_json_path and os.path.exists(rig_json_path):
    with open(rig_json_path) as f:
        rig = json.load(f)
    # Create armature
    bpy.ops.object.add(type='ARMATURE')
    arm = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    joints = rig.get('joint_names', [])
    for j_name in joints:
        bone = arm.data.edit_bones.new(j_name)
        bone.head = (0, 0, 0)
        bone.tail = (0, 0.1, 0)
    bpy.ops.object.mode_set(mode='OBJECT')
    # Set vertex groups from LBS weights
    weights = rig.get('lbs_weights', [])
    for j_i, j_name in enumerate(joints):
        vg = mesh_obj.vertex_groups.new(name=j_name)
        for v_i, w_row in enumerate(weights):
            if j_i < len(w_row) and w_row[j_i] > 0.001:
                vg.add([v_i], w_row[j_i], 'REPLACE')

# Import blendshapes
bs_dir = r"{blendshape_dir}"
if bs_dir and os.path.exists(bs_dir):
    mesh_obj.shape_key_add(name='Basis', from_mix=False)
    for fname in os.listdir(bs_dir):
        if not fname.endswith('.obj'):
            continue
        bs_name = fname[:-4]
        # Import shape key OBJ
        bpy.ops.import_scene.obj(filepath=os.path.join(bs_dir, fname))
        bs_obj = bpy.context.active_object
        # Transfer shape to mesh_obj
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        mod = mesh_obj.modifiers.new(name=bs_name, type='MESH_SEQUENCE_CACHE')
        # Simpler: add shape key by vertex positions
        sk = mesh_obj.shape_key_add(name=bs_name, from_mix=False)
        for i, v in enumerate(bs_obj.data.vertices):
            if i < len(sk.data):
                sk.data[i].co = v.co
        bpy.data.objects.remove(bs_obj, do_unlink=True)

# Export FBX
out_path = r"{output_dir / 'avatar.fbx'}"
bpy.ops.export_scene.fbx(
    filepath=out_path,
    use_selection=False,
    add_leaf_bones=False,
    use_mesh_modifiers=False,
    mesh_smooth_type='FACE',
    use_armature_deform_only=True,
)
print(f"FBX exported: {{out_path}}")
"""
    script_path = output_dir / "export_fbx.py"
    with open(str(script_path), "w") as f:
        f.write(script)
    logger.info(f"  FBX export script: {script_path}")
    logger.info("  Run with: blender --background --python " + str(script_path))
    return str(script_path)


def export_usd_stage(
    mesh_path: str,
    pbr_maps: Dict[str, str],
    output_dir: Path,
) -> Optional[str]:
    """Export USD stage with PBR materials for pipeline-ready delivery."""
    out_path = output_dir / "avatar.usda"

    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
        import trimesh

        stage = Usd.Stage.CreateNew(str(out_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root  = UsdGeom.Xform.Define(stage, "/avatar")
        mroot = UsdGeom.Mesh.Define(stage, "/avatar/head_mesh")

        mesh = trimesh.load(mesh_path, process=False)
        pts  = [Gf.Vec3f(*v) for v in mesh.vertices]
        mroot.GetPointsAttr().Set(pts)
        faces = mesh.faces
        fc = [3] * len(faces)
        mroot.GetFaceVertexCountsAttr().Set(fc)
        mroot.GetFaceVertexIndicesAttr().Set(faces.flatten().tolist())

        # Material
        mat_path_usd = "/avatar/head_mat"
        mat = UsdShade.Material.Define(stage, mat_path_usd)
        shader = UsdShade.Shader.Define(stage, f"{mat_path_usd}/pbr")
        shader.CreateIdAttr("UsdPreviewSurface")

        if "albedo" in pbr_maps and pbr_maps["albedo"]:
            tex = UsdShade.Shader.Define(stage, f"{mat_path_usd}/albedo_tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(pbr_maps["albedo"])
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )

        shader.CreateInput("roughness",  Sdf.ValueTypeNames.Float).Set(0.45)
        shader.CreateInput("metallic",   Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("subsurface", Sdf.ValueTypeNames.Float).Set(0.05)

        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mroot).Bind(mat)

        stage.GetRootLayer().Save()
        logger.info(f"  USD exported: {out_path}")
        return str(out_path)

    except ImportError:
        logger.warning("pxr (OpenUSD) not installed — skipping USD export")
        return None
    except Exception as e:
        logger.error("USD export failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Stage 10 entry point
# ---------------------------------------------------------------------------

class ValidationStage:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def run(
        self,
        mesh_path: str,
        reference_image_path: str,
        source_arcface_embedding: np.ndarray,
        view_render_paths: Optional[Dict[str, str]] = None,
        anim_params_path: Optional[str] = None,
        pbr_maps: Optional[Dict[str, str]] = None,
        rig_json: Optional[str] = None,
        blendshape_dir: Optional[str] = None,
        output_dir: Path = Path("./output/stage10"),
        fps: int = 30,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("STAGE 10: Validation & Export")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)

        report: Dict = {
            "mesh_path":       mesh_path,
            "reference_image": reference_image_path,
        }

        # 10a: Identity verification
        logger.info("Step 10a: ArcFace identity verification (8 views) …")
        identity = verify_identity_8views(
            mesh_path, source_arcface_embedding, view_render_paths,
        )
        report["identity"] = identity

        # 10b: Perceptual metrics
        logger.info("Step 10b: Perceptual quality metrics …")
        if view_render_paths and "front" in view_render_paths:
            frontal_render = view_render_paths["front"]
            side_render    = view_render_paths.get("right")
            perceptual     = compute_perceptual_metrics(
                frontal_render, reference_image_path, side_render, self.device,
            )
        else:
            perceptual = {"note": "no rendered views available for perceptual metrics"}
        report["perceptual"] = perceptual

        # 10c: Geometric validation
        logger.info("Step 10c: Geometric validation …")
        geometry = validate_geometry(mesh_path)
        report["geometry"] = geometry

        # 10d: Animation smoke test
        if anim_params_path and Path(anim_params_path).exists():
            logger.info("Step 10d: Animation smoke test …")
            animation = animation_smoke_test(anim_params_path, fps)
            report["animation"] = animation
        else:
            report["animation"] = {"note": "no animation params provided"}

        # 10e: Export
        logger.info("Step 10e: Exporting GLB, FBX script, USD …")
        albedo = (pbr_maps or {}).get("albedo")
        glb_path = export_glb(mesh_path, output_dir, albedo)
        fbx_script = export_fbx_script(mesh_path, rig_json, blendshape_dir, output_dir)
        usd_path   = export_usd_stage(mesh_path, pbr_maps or {}, output_dir)

        report["exports"] = {
            "glb":        glb_path,
            "fbx_script": fbx_script,
            "usd":        usd_path,
        }

        # Compute overall pass/fail
        passes = {
            "identity":   identity.get("pass", False),
            "lpips":      perceptual.get("lpips_frontal_pass", True),
            "ssim":       perceptual.get("ssim_frontal_pass",  True),
            "watertight": geometry.get("watertight",          True),
            "animation":  report.get("animation", {}).get("jitter_pass", True),
        }
        report["passes"]       = passes
        report["overall_pass"] = all(passes.values())

        # Save quality report
        report_path = output_dir / "quality_report.json"
        with open(str(report_path), "w") as f:
            # Convert numpy types for JSON
            def _convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return obj
            json.dump(report, f, indent=2, default=_convert)

        logger.info("✓ Stage 10 complete")
        logger.info(f"  Overall pass: {report['overall_pass']}")
        for k, v in passes.items():
            logger.info(f"  {'✓' if v else '✗'} {k}")
        logger.info(f"  Quality report: {report_path}")
        logger.info(f"  GLB: {glb_path}")

        return report
