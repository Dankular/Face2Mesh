"""
Stage 4 — Mesh Assembly, UV Unwrap, PBR Textures & Pore Detail
==============================================================
Steps:
  4a. FLAME shrinkwrap retopology (or xatlas fallback)
  4b. UV unwrap (FLAME standard UV or xatlas)
  4c. Multi-view texture baking from 24 Qwen views
  4d. PBR map generation (Albedo via Marigold IID, Normal via DSINE, Roughness)
  4e. HRN pore-level microdetail maps
  4f. Expression-dependent displacement library setup

FLAME model: /root/.cache/face_models/flame/generic_model.pkl
             Register at https://flame.is.tue.mpg.de/
"""

from __future__ import annotations
import gc
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"
FLAME_PKL  = MODELS_DIR / "flame" / "generic_model.pkl"
FLAME_UV   = MODELS_DIR / "flame" / "FLAME_UV_coor_new.npz"
DSINE_DIR  = MODELS_DIR / "dsine"
HRN_DIR    = MODELS_DIR / "hrn"


# ---------------------------------------------------------------------------
# 4a — FLAME shrinkwrap retopology
# ---------------------------------------------------------------------------

def flame_retopology(
    tsdf_mesh_path: str,
    flame_beta: np.ndarray,
    output_dir: Path,
    subdivide: int = 1,
) -> str:
    """
    Project FLAME template vertices onto the TSDF mesh surface.
    Returns path to the retopologised mesh OBJ.
    """
    out_path = output_dir / "flame_retopo.obj"

    if not FLAME_PKL.exists():
        logger.warning(
            "FLAME model not found — using xatlas fallback retopology.\n"
            "For production use, download FLAME from https://flame.is.tue.mpg.de/\n"
            "and place generic_model.pkl at %s", FLAME_PKL
        )
        return _xatlas_retopology(tsdf_mesh_path, output_dir)

    try:
        import trimesh
        import pickle

        with open(str(FLAME_PKL), "rb") as f:
            flame_model = pickle.load(f, encoding="latin1")

        # Get FLAME template vertices using beta coefficients
        shapedirs = flame_model["shapedirs"]          # (V*3, n_shape)
        v_template = flame_model["v_template"]         # (V, 3) neutral template
        faces = flame_model["f"]                       # (F, 3) triangle faces

        # Apply shape blend shapes
        n_beta = min(len(flame_beta), shapedirs.shape[-1])
        beta_pad = np.zeros(shapedirs.shape[-1])
        beta_pad[:n_beta] = flame_beta[:n_beta]

        v_shaped = v_template + shapedirs.reshape(-1, 3, shapedirs.shape[-1]) @ beta_pad
        v_shaped = v_shaped.reshape(-1, 3)

        # Subdivide for higher resolution
        flame_mesh = trimesh.Trimesh(vertices=v_shaped, faces=faces, process=False)
        for _ in range(subdivide):
            flame_mesh = flame_mesh.subdivide()

        # Load TSDF mesh
        tsdf_mesh = trimesh.load(tsdf_mesh_path, process=False)

        # Scale FLAME to match TSDF mesh bounding box
        flame_center  = flame_mesh.bounding_box.centroid
        tsdf_center   = tsdf_mesh.bounding_box.centroid
        flame_scale   = tsdf_mesh.scale / (flame_mesh.scale + 1e-8)
        flame_verts   = (flame_mesh.vertices - flame_center) * flame_scale + tsdf_center

        # Shrinkwrap: project each FLAME vertex to nearest TSDF surface point
        from trimesh.proximity import closest_point
        projected, _, _ = closest_point(tsdf_mesh, flame_verts)

        # Apply Laplacian smoothing to fix projection artifacts
        projected = _laplacian_smooth(projected, flame_mesh.faces, iterations=3)

        retopo = trimesh.Trimesh(vertices=projected, faces=flame_mesh.faces, process=False)
        retopo.export(str(out_path))

        logger.info(f"  FLAME retopology: {len(retopo.vertices):,} vertices, {len(retopo.faces):,} faces")
        return str(out_path)

    except Exception as e:
        logger.error("FLAME retopology failed: %s — falling back to xatlas", e)
        return _xatlas_retopology(tsdf_mesh_path, output_dir)


def _xatlas_retopology(tsdf_mesh_path: str, output_dir: Path) -> str:
    """Fallback: use xatlas + quadric decimation on raw TSDF mesh."""
    import trimesh
    mesh = trimesh.load(tsdf_mesh_path, process=False)

    # Decimate to ~50k faces for manageability
    target_faces = min(50000, len(mesh.faces))
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    # Fill holes
    trimesh.repair.fill_holes(mesh)
    mesh.fix_normals()

    out_path = output_dir / "mesh_retopo.obj"
    mesh.export(str(out_path))
    logger.info(f"  xatlas retopology: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return str(out_path)


def _laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Simple Laplacian smoothing."""
    v = vertices.copy()
    n = len(v)
    adj: List[List[int]] = [[] for _ in range(n)]
    for f in faces:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            adj[f[i]].append(f[j])
            adj[f[j]].append(f[i])

    for _ in range(iterations):
        new_v = v.copy()
        for i in range(n):
            nb = adj[i]
            if nb:
                new_v[i] = v[nb].mean(axis=0) * 0.5 + v[i] * 0.5
        v = new_v
    return v


# ---------------------------------------------------------------------------
# 4b — UV unwrap
# ---------------------------------------------------------------------------

def uv_unwrap(mesh_path: str, output_dir: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Generate UV coordinates for the mesh.
    Tries FLAME standard UV first; falls back to xatlas.
    Returns: (uv_mesh_path, uv_coords (V,2), uv_faces (F,3))
    """
    # Try FLAME UV layout
    if FLAME_UV.exists():
        try:
            data = np.load(str(FLAME_UV))
            uv_coords = data["vt"]
            uv_faces  = data["ft"]
            logger.info(f"  FLAME UV loaded: {len(uv_coords)} UV verts, {len(uv_faces)} UV faces")
            # The UV is already in the FLAME mesh, just copy
            import shutil
            out = output_dir / "mesh_uv.obj"
            shutil.copy(mesh_path, str(out))
            return str(out), uv_coords, uv_faces
        except Exception as e:
            logger.warning("FLAME UV load failed: %s — using xatlas", e)

    # xatlas UV parameterisation
    try:
        import xatlas
        import trimesh

        mesh   = trimesh.load(mesh_path, process=False)
        verts  = mesh.vertices.astype(np.float32)
        tris   = mesh.faces.astype(np.uint32)

        atlas  = xatlas.Atlas()
        atlas.add_mesh(verts, tris)
        atlas.generate(xatlas.PackOptions(), xatlas.ChartOptions())

        _, uv_indices, uv_coords = atlas[0]
        logger.info(f"  xatlas UV: {len(uv_coords)} UV coords")

        out_path = output_dir / "mesh_uv.obj"
        # Write OBJ with UV
        with open(str(out_path), "w") as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for uv in uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            # Build face groups mapping original indices to UV indices
            for tri_i, tri in enumerate(tris):
                ui = uv_indices[tri_i]
                f.write(f"f {tri[0]+1}/{ui[0]+1} {tri[1]+1}/{ui[1]+1} {tri[2]+1}/{ui[2]+1}\n")

        return str(out_path), uv_coords, uv_indices

    except ImportError:
        logger.error("xatlas not installed — pip install xatlas")
        return mesh_path, np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32)


# ---------------------------------------------------------------------------
# 4c — Multi-view texture baking
# ---------------------------------------------------------------------------

def bake_textures(
    mesh_path: str,
    views: Dict[str, str],
    view_angles: List[Dict],
    uv_coords: np.ndarray,
    output_dir: Path,
    texture_size: int = 2048,
) -> str:
    """
    Project Qwen view colours into UV space.
    Uses cosine-weighted blending for overlapping views.
    Returns path to albedo texture PNG.
    """
    import trimesh

    mesh  = trimesh.load(mesh_path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces,    dtype=np.int32)

    tex_rgb   = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    tex_weight = np.zeros((texture_size, texture_size, 1), dtype=np.float32)

    # Process each view
    for angle in view_angles:
        label = angle["label"]
        img_path = views.get(label)
        if img_path is None:
            continue

        view_img  = np.array(Image.open(img_path).convert("RGB").resize((512, 512))).astype(np.float32) / 255.0
        cam_dir   = _angle_to_direction(angle["azimuth"], angle["elevation"])
        face_norms = _compute_face_normals(verts, faces)

        # Weight = cosine(face_normal, camera_direction), clamped to [0, 1]
        weights  = np.clip(face_norms @ (-cam_dir), 0.0, 1.0)  # negate: cam looks inward

        # Project each face's centroid into the view image
        proj_mat = _camera_projection(angle["azimuth"], angle["elevation"], angle["distance"])
        centroids = verts[faces].mean(axis=1)  # (F, 3)

        px, py = _project_to_image(centroids, proj_mat, 512, 512)

        # Sample colour and write to UV space (approximate: use face centroid UV)
        if len(uv_coords) > 0 and len(uv_coords) == len(verts):
            uv_centroids = uv_coords[faces].mean(axis=1)  # (F, 2)
        else:
            # Fall back to flat UV from face index
            uv_centroids = np.column_stack([
                np.arange(len(faces), dtype=np.float32) / len(faces),
                np.zeros(len(faces), dtype=np.float32),
            ])

        for fi in range(len(faces)):
            w = weights[fi]
            if w < 0.05:
                continue
            x_px, y_px = int(px[fi]), int(py[fi])
            if 0 <= x_px < 512 and 0 <= y_px < 512:
                colour = view_img[y_px, x_px]
                u_tx = int(uv_centroids[fi, 0] * (texture_size - 1))
                v_tx = int((1.0 - uv_centroids[fi, 1]) * (texture_size - 1))
                u_tx = np.clip(u_tx, 0, texture_size - 1)
                v_tx = np.clip(v_tx, 0, texture_size - 1)
                tex_rgb[v_tx, u_tx]    += colour * w
                tex_weight[v_tx, u_tx] += w

    # Normalise
    valid = tex_weight[..., 0] > 0
    tex_rgb[valid] /= tex_weight[valid]

    # Inpaint holes via nearest-neighbour
    from scipy.ndimage import distance_transform_edt, label as scipy_label
    missing = ~valid
    if missing.any():
        dist, idx = distance_transform_edt(missing, return_indices=True)
        tex_rgb[missing] = tex_rgb[idx[0][missing], idx[1][missing]]

    albedo_path = output_dir / "albedo_2k.png"
    Image.fromarray((tex_rgb * 255).clip(0, 255).astype(np.uint8)).save(str(albedo_path))
    logger.info(f"  Albedo texture baked: {albedo_path} ({texture_size}×{texture_size})")
    return str(albedo_path)


def _angle_to_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ], dtype=np.float64)


def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n  = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (norms + 1e-8)


def _camera_projection(azimuth_deg: float, elevation_deg: float, distance: float):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    R_az = np.array([[np.cos(az), -np.sin(az), 0],
                     [np.sin(az),  np.cos(az), 0],
                     [0,           0,          1]])
    R_el = np.array([[1, 0,           0],
                     [0, np.cos(el), -np.sin(el)],
                     [0, np.sin(el),  np.cos(el)]])
    R = R_el @ R_az
    t = np.array([0, 0, distance])
    return (R, t)


def _project_to_image(
    points: np.ndarray, proj_mat, img_w: int, img_h: int,
    focal: float = 560.0
) -> Tuple[np.ndarray, np.ndarray]:
    R, t = proj_mat
    cam = (R @ points.T).T + t
    valid = cam[:, 2] > 0
    px = np.zeros(len(points), dtype=np.float32)
    py = np.zeros(len(points), dtype=np.float32)
    px[valid] = focal * cam[valid, 0] / cam[valid, 2] + img_w / 2
    py[valid] = focal * cam[valid, 1] / cam[valid, 2] + img_h / 2
    return px, py


# ---------------------------------------------------------------------------
# 4d — PBR map generation (Normal + Roughness + Albedo de-lit)
# ---------------------------------------------------------------------------

def generate_pbr_maps(
    albedo_path: str,
    face_image_path: str,
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, str]:
    """
    Generate normal, roughness, specular, and de-lit albedo maps.
    Uses DSINE for normals and Marigold IID for intrinsic decomposition.
    """
    results: Dict[str, str] = {"albedo": albedo_path}

    # --- Normal map via DSINE ---
    results["normal"] = _run_dsine(face_image_path, output_dir, device)

    # --- De-lit albedo + roughness via Marigold IID ---
    marigold_out = _run_marigold_iid(face_image_path, output_dir, device)
    results.update(marigold_out)

    return results


def _run_dsine(image_path: str, output_dir: Path, device: str = "cuda") -> str:
    """Run DSINE camera-intrinsics-aware normal estimation."""
    out_normal = output_dir / "normal_dsine.png"

    dsine_code = MODELS_DIR / "dsine_repo"
    if not dsine_code.exists():
        logger.info("Cloning DSINE …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/baegwangbin/DSINE.git", str(dsine_code)],
            check=True,
        )

    ckpt = DSINE_DIR / "scannet.pt"
    if not ckpt.exists():
        DSINE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading DSINE checkpoint …")
        subprocess.run(
            ["wget", "-q", "-O", str(ckpt),
             "https://huggingface.co/baegwangbin/DSINE/resolve/main/scannet.pt"],
            check=True,
        )

    try:
        if str(dsine_code) not in sys.path:
            sys.path.insert(0, str(dsine_code))

        from models.DSINE import DSINE  # type: ignore

        model = DSINE().to(device)
        ckpt_data = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(ckpt_data["model"], strict=False)
        model.eval()

        img = Image.open(image_path).convert("RGB").resize((512, 512))
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        # DSINE requires intrinsics (use default perspective)
        h, w = 512, 512
        fx = fy = 560.0
        intrinsics = torch.tensor([[fx, 0, w/2, 0, fy, h/2, 0, 0, 1]], dtype=torch.float32).to(device)
        intrinsics = intrinsics.reshape(1, 3, 3)

        with torch.no_grad():
            pred_norm = model(tensor, intrinsics=intrinsics)[-1]

        # Convert [-1,1] normal to [0,255] PNG
        norm_np = pred_norm.squeeze().permute(1, 2, 0).cpu().numpy()
        norm_png = ((norm_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(norm_png).save(str(out_normal))
        logger.info(f"  DSINE normal map: {out_normal}")

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error("DSINE failed: %s — generating flat normal map", e)
        flat = np.zeros((512, 512, 3), dtype=np.uint8)
        flat[:, :] = [128, 128, 255]  # pointing up in tangent space
        Image.fromarray(flat).save(str(out_normal))

    return str(out_normal)


def _run_marigold_iid(image_path: str, output_dir: Path, device: str = "cuda") -> Dict[str, str]:
    """Run Marigold IID for intrinsic image decomposition (albedo, roughness)."""
    out_albedo_iid = output_dir / "albedo_iid.png"
    out_roughness  = output_dir / "roughness.png"
    results = {}

    marigold_dir = MODELS_DIR / "marigold"

    try:
        from diffusers import MarigoldIntrinsicsPipeline  # type: ignore

        pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            str(marigold_dir), torch_dtype=torch.float16,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        img = Image.open(image_path).convert("RGB").resize((512, 512))
        with torch.inference_mode():
            pred = pipe(img, num_inference_steps=10)

        if hasattr(pred, "albedo"):
            alb = np.array(pred.albedo[0])
            Image.fromarray(alb).save(str(out_albedo_iid))
            results["albedo_iid"] = str(out_albedo_iid)
        if hasattr(pred, "roughness"):
            rgh = np.array(pred.roughness[0])
            if rgh.ndim == 2:
                rgh = np.stack([rgh] * 3, axis=-1)
            Image.fromarray((rgh * 255).clip(0, 255).astype(np.uint8)).save(str(out_roughness))
            results["roughness"] = str(out_roughness)

        pipe.unet.cpu()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error("Marigold IID failed: %s — using flat roughness", e)
        # Skin roughness default: ~0.45
        rgh = np.full((512, 512, 3), 114, dtype=np.uint8)
        Image.fromarray(rgh).save(str(out_roughness))
        results["roughness"] = str(out_roughness)

    return results


# ---------------------------------------------------------------------------
# 4e — HRN pore-level microdetail
# ---------------------------------------------------------------------------

def generate_hrn_microdetail(
    reference_image_path: str,
    qwen_frontal_paths: List[str],
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, str]:
    """
    Run HRN (youngLBW/HRN) to generate mid+high frequency displacement maps.
    Returns paths to deformation_map.exr and displacement_map.exr.
    """
    results: Dict[str, str] = {}
    hrn_code = MODELS_DIR / "hrn_repo"

    if not hrn_code.exists():
        logger.info("Cloning HRN repository …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/youngLBW/HRN.git", str(hrn_code)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r",
             str(hrn_code / "requirements.txt")],
            check=True,
        )

    # Download HRN pretrained weights from HF
    hrn_ckpt = HRN_DIR / "hrn_pretrained"
    if not hrn_ckpt.exists():
        HRN_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading HRN weights from HuggingFace …")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="youngLBW/HRN",
                local_dir=str(hrn_ckpt),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.error("HRN weights download failed: %s", e)
            return results

    try:
        import tempfile, shutil

        out_dir_hrn = Path(tempfile.mkdtemp())
        inp_dir_hrn = out_dir_hrn / "input"
        inp_dir_hrn.mkdir()

        # Copy reference image
        shutil.copy(reference_image_path, inp_dir_hrn / "reference.png")
        for i, p in enumerate(qwen_frontal_paths[:4]):
            shutil.copy(p, inp_dir_hrn / f"view_{i:02d}.png")

        result = subprocess.run(
            [sys.executable, str(hrn_code / "inference.py"),
             "--input_dir", str(inp_dir_hrn),
             "--output_dir", str(out_dir_hrn),
             "--checkpoint_dir", str(hrn_ckpt)],
            capture_output=True, text=True, cwd=str(hrn_code),
        )

        if result.returncode != 0:
            logger.error("HRN failed:\n%s", result.stderr[-2000:])
            return results

        # Collect outputs
        for fname in ["deformation_map.exr", "displacement_map.exr",
                      "micronormal_map_4k.exr", "microspecular_map_4k.exr"]:
            src = out_dir_hrn / fname
            if src.exists():
                dst = output_dir / fname
                shutil.copy(str(src), str(dst))
                results[fname.split(".")[0]] = str(dst)
                logger.info(f"  HRN output: {dst}")

    except Exception as e:
        logger.error("HRN error: %s", e)

    return results


# ---------------------------------------------------------------------------
# 4f — Expression displacement library
# ---------------------------------------------------------------------------

def build_expression_displacement_library(
    reference_image_path: str,
    qwen_backend,
    source_arcface: np.ndarray,
    smirk_expression: np.ndarray,
    output_dir: Path,
) -> Dict[str, str]:
    """
    Generate expression-specific deformation maps for 10 key expressions.
    Each expression view goes through HRN to produce a deformation map.
    Returns dict mapping expression_name → deformation_map path.
    """
    EXPRESSIONS = [
        ("neutral",     "neutral relaxed expression"),
        ("smile",       "wide natural smile"),
        ("frown",       "sad frown, brow furrowed"),
        ("brow_raise",  "both eyebrows raised in surprise"),
        ("brow_furrow", "both eyebrows furrowed in anger"),
        ("jaw_open",    "mouth open wide, jaw dropped"),
        ("eye_squeeze", "eyes squeezed tightly shut"),
        ("lip_pucker",  "lips puckered forward as if kissing"),
        ("nose_wrinkle","nose wrinkled in disgust"),
        ("surprise",    "wide-eyed surprised expression"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, str] = {}
    source_img = Image.open(reference_image_path).convert("RGB").resize((512, 512))

    for expr_name, expr_desc in EXPRESSIONS:
        logger.info(f"  Generating {expr_name} expression view …")
        out_view = output_dir / f"expr_{expr_name}.png"

        try:
            # Generate expression view via Qwen
            expr_angle = {"label": expr_name, "azimuth": 0, "elevation": 0, "distance": 2.5}
            # Override prompt for expression
            if hasattr(qwen_backend, "generate_view"):
                # Temporarily override label for expression prompt
                angle_copy = dict(expr_angle)
                angle_copy["expr_desc"] = expr_desc
                expr_img = qwen_backend.generate_view(source_img, angle_copy)
            else:
                expr_img = source_img

            if expr_img is None:
                expr_img = source_img

            expr_img.save(str(out_view))

            # Run HRN on expression view
            expr_disp_dir = output_dir / f"hrn_{expr_name}"
            expr_disp_dir.mkdir(exist_ok=True)
            hrn_result = generate_hrn_microdetail(
                str(out_view), [], expr_disp_dir,
            )
            if "deformation_map" in hrn_result:
                results[expr_name] = hrn_result["deformation_map"]

        except Exception as e:
            logger.warning(f"  Expression {expr_name} failed: {e}")

    logger.info(f"  Expression library: {len(results)} maps generated")
    return results


# ---------------------------------------------------------------------------
# Stage 4 entry point
# ---------------------------------------------------------------------------

class TextureStage:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def run(
        self,
        tsdf_mesh_path: str,
        views: Dict[str, str],
        view_angles: List[Dict],
        flame_beta: np.ndarray,
        source_arcface: np.ndarray,
        smirk_expression: np.ndarray,
        face_image_path: str,
        output_dir: Path,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("STAGE 4: Mesh Assembly, UV Unwrap, PBR Textures")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4a: FLAME retopology
        logger.info("Step 4a: FLAME shrinkwrap retopology …")
        retopo_mesh = flame_retopology(tsdf_mesh_path, flame_beta, output_dir)

        # 4b: UV unwrap
        logger.info("Step 4b: UV unwrap …")
        uv_mesh, uv_coords, uv_faces = uv_unwrap(retopo_mesh, output_dir)

        # 4c: Texture baking from Qwen views
        logger.info("Step 4c: Multi-view texture baking …")
        albedo_path = bake_textures(
            uv_mesh, views, view_angles, uv_coords, output_dir,
        )

        # 4d: PBR maps
        logger.info("Step 4d: PBR map generation (DSINE + Marigold IID) …")
        pbr_maps = generate_pbr_maps(albedo_path, face_image_path, output_dir, self.device)

        # 4e: HRN pore detail
        logger.info("Step 4e: HRN pore-level microdetail …")
        frontal_views = [v for k, v in views.items() if "front" in k and "hi" not in k and "lo" not in k]
        hrn_maps = generate_hrn_microdetail(
            face_image_path, frontal_views[:4], output_dir, self.device,
        )

        results = {
            "retopo_mesh":   retopo_mesh,
            "uv_mesh":       uv_mesh,
            "uv_coords":     uv_coords,
            "albedo":        albedo_path,
            "pbr_maps":      pbr_maps,
            "hrn_maps":      hrn_maps,
            "output_dir":    str(output_dir),
        }

        logger.info("✓ Stage 4 complete")
        logger.info(f"  Retopo mesh : {retopo_mesh}")
        logger.info(f"  Albedo      : {albedo_path}")
        logger.info(f"  Normal map  : {pbr_maps.get('normal', 'N/A')}")
        logger.info(f"  HRN maps    : {list(hrn_maps.keys())}")
        return results
