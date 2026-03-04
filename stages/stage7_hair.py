"""
Stage 7 — Hair Reconstruction
==============================
Option A: NeuralHaircut (strand-based, film quality) — tried first
Option B: Hair cards from BiSeNet segmentation — fast fallback
Option C: TSDF hair volume (already in mesh) — always works

For RTX 3060 12GB, NeuralHaircut may OOM (needs ~24GB). Falls back automatically.
"""

from __future__ import annotations
import gc
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"
NEURALHAIRCUT_DIR = MODELS_DIR / "neuralhaircut_repo"


# ---------------------------------------------------------------------------
# Option A — NeuralHaircut (strand-based)
# ---------------------------------------------------------------------------

def _ensure_neuralhaircut() -> bool:
    """Clone NeuralHaircut repo and install deps."""
    if not NEURALHAIRCUT_DIR.exists():
        try:
            logger.info("Cloning NeuralHaircut …")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/SamsungLabs/NeuralHaircut.git",
                 str(NEURALHAIRCUT_DIR)],
                check=True, timeout=120,
            )
            # Install minimal deps
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "torchvision", "pytorch-lightning<2", "omegaconf"],
                check=True,
            )
            return True
        except Exception as e:
            logger.warning(f"NeuralHaircut clone failed: {e}")
            return False
    return True


def run_neuralhaircut(
    reference_image_path: str,
    head_mesh_path: str,
    output_dir: Path,
    device: str = "cuda",
) -> Optional[str]:
    """
    Run NeuralHaircut strand reconstruction.
    Returns path to hair_strands.obj or None on failure.
    """
    if not _ensure_neuralhaircut():
        return None

    nh_ckpt = MODELS_DIR / "neuralhaircut" / "pretrained"
    if not nh_ckpt.exists():
        logger.info("Downloading NeuralHaircut pretrained checkpoint …")
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(
                repo_id="SamsungLabs/NeuralHaircut",
                local_dir=str(nh_ckpt),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.warning(f"NeuralHaircut weights unavailable: {e}")
            return None

    import tempfile, shutil
    out = Path(tempfile.mkdtemp())

    result = subprocess.run(
        [sys.executable, str(NEURALHAIRCUT_DIR / "scripts" / "infer_single_view.py"),
         "--image",   reference_image_path,
         "--mesh",    head_mesh_path,
         "--out_dir", str(out),
         "--checkpoint", str(nh_ckpt)],
        capture_output=True, text=True, cwd=str(NEURALHAIRCUT_DIR),
    )

    if result.returncode != 0:
        logger.warning(f"NeuralHaircut failed:\n{result.stderr[-1000:]}")
        return None

    strand_file = out / "hair_strands.obj"
    if strand_file.exists():
        dst = output_dir / "hair_strands.obj"
        shutil.copy(str(strand_file), str(dst))
        logger.info(f"  NeuralHaircut: strand file at {dst}")
        return str(dst)
    return None


# ---------------------------------------------------------------------------
# Option B — Hair cards from BiSeNet segmentation
# ---------------------------------------------------------------------------

def _write_obj(path: str, verts: np.ndarray, faces: np.ndarray, mtl: str = "hair"):
    with open(path, "w") as f:
        f.write(f"mtllib hair.mtl\nusemtl {mtl}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def build_hair_cards(
    reference_image_path: str,
    head_mesh_path: str,
    seg_mask: np.ndarray,
    output_dir: Path,
    device: str = "cuda",
) -> str:
    """
    Generate hair card strips from BiSeNet hair segmentation mask (class 17).
    Cards are placed tangentially on the head surface.
    """
    import trimesh
    from scipy.ndimage import binary_erosion

    # Extract hair region boundary
    hair_mask = (seg_mask == 17).astype(np.uint8)  # class 17 = hair
    hair_pixels = np.argwhere(hair_mask > 0)

    if len(hair_pixels) == 0:
        logger.warning("  No hair pixels in segmentation — generating default hair volume")
        return _fallback_hair_shell(head_mesh_path, output_dir)

    logger.info(f"  Hair mask: {len(hair_pixels)} pixels")

    # Load head mesh for surface placement
    mesh = trimesh.load(head_mesh_path, process=False)
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    head_center = (bounds_min + bounds_max) * 0.5

    img = Image.open(reference_image_path).convert("RGB")
    img_arr = np.array(img)
    H, W = img_arr.shape[:2]

    # Sample hair card placement points from hair mask
    n_cards = 40
    np.random.seed(42)
    sample_idx = np.random.choice(len(hair_pixels), min(n_cards, len(hair_pixels)), replace=False)
    sample_pts = hair_pixels[sample_idx]   # (n, 2) in image space

    verts_all, faces_all = [], []
    vertex_offset = 0
    card_length = (bounds_max[1] - bounds_min[1]) * 0.25
    card_width  = card_length * 0.15

    for py, px in sample_pts:
        # Convert image pixel to approximate 3D position on head surface
        norm_x = (px / W - 0.5) * (bounds_max[0] - bounds_min[0])
        norm_y = (0.5 - py / H) * (bounds_max[1] - bounds_min[1])

        # Find closest point on head mesh
        surface_pt = np.array([
            head_center[0] + norm_x,
            head_center[1] + norm_y,
            bounds_max[2] * 0.9,
        ], dtype=np.float32)

        from trimesh.proximity import closest_point
        proj, _, _ = closest_point(mesh, surface_pt[None])
        base = proj[0]

        # Hair growth direction: roughly outward from head centre + upward
        growth = base - head_center
        growth /= (np.linalg.norm(growth) + 1e-8)
        growth[1] = abs(growth[1]) + 0.3  # bias upward
        growth /= (np.linalg.norm(growth) + 1e-8)

        # Perpendicular for card width
        perp = np.cross(growth, np.array([0, 0, 1]))
        if np.linalg.norm(perp) < 1e-8:
            perp = np.cross(growth, np.array([1, 0, 0]))
        perp /= (np.linalg.norm(perp) + 1e-8)

        # Gentle curl: tip rotates slightly
        tip_offset = growth * card_length + np.cross(growth, perp) * card_length * 0.15

        b = vertex_offset
        half_w = perp * card_width * 0.5
        card_verts = [
            base - half_w,
            base + half_w,
            base + tip_offset - half_w * 0.7,
            base + tip_offset + half_w * 0.7,
        ]
        verts_all.extend(card_verts)
        faces_all.extend([[b, b+1, b+3], [b, b+3, b+2]])
        vertex_offset += 4

    verts_arr = np.array(verts_all, dtype=np.float32)
    faces_arr = np.array(faces_all, dtype=np.int32)

    out_path = str(output_dir / "hair_cards.obj")
    _write_obj(out_path, verts_arr, faces_arr)
    logger.info(f"  Hair cards: {len(faces_arr)} triangles from {n_cards} strips")
    return out_path


# ---------------------------------------------------------------------------
# Option C — TSDF hair volume (already present in mesh)
# ---------------------------------------------------------------------------

def _fallback_hair_shell(head_mesh_path: str, output_dir: Path) -> str:
    """
    Extract the hair volume from the existing TSDF head mesh by taking the
    upper portion (above face region). No modification needed — the TSDF
    mesh already contains the hair as part of the head geometry.
    """
    import trimesh, shutil

    mesh = trimesh.load(head_mesh_path, process=False)
    top_y = mesh.bounds[1][1]
    eye_y  = mesh.bounds[0][1] + (top_y - mesh.bounds[0][1]) * 0.6

    # Extract upper-head faces (hair region)
    vert_y = mesh.vertices[:, 1]
    face_mask = vert_y[mesh.faces].mean(axis=1) > eye_y
    hair_faces = mesh.faces[face_mask]

    if len(hair_faces) > 0:
        used_verts = np.unique(hair_faces)
        vmap = {v: i for i, v in enumerate(used_verts)}
        new_verts = mesh.vertices[used_verts]
        new_faces = np.array([[vmap[f[0]], vmap[f[1]], vmap[f[2]]] for f in hair_faces])
        hair_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    else:
        hair_mesh = mesh

    out_path = str(output_dir / "hair_volume.obj")
    hair_mesh.export(out_path)
    logger.info(f"  Hair volume (TSDF): {len(hair_mesh.faces)} triangles")
    return out_path


# ---------------------------------------------------------------------------
# Stage 7 entry point
# ---------------------------------------------------------------------------

class HairStage:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def run(
        self,
        reference_image_path: str,
        head_mesh_path: str,
        seg_mask: np.ndarray,
        output_dir: Path,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("STAGE 7: Hair Reconstruction")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)

        hair_result: Optional[str] = None
        method_used: str = ""

        # Option A: NeuralHaircut
        logger.info("Attempting Option A: NeuralHaircut (strand-based) …")
        try:
            hair_result = run_neuralhaircut(
                reference_image_path, head_mesh_path, output_dir, self.device,
            )
            if hair_result:
                method_used = "neuralhaircut"
                logger.info(f"  ✓ NeuralHaircut succeeded: {hair_result}")
        except Exception as e:
            logger.warning(f"  NeuralHaircut failed: {e}")

        # Option B: Hair cards
        if hair_result is None:
            logger.info("Attempting Option B: Hair cards from segmentation …")
            try:
                hair_result = build_hair_cards(
                    reference_image_path, head_mesh_path, seg_mask, output_dir, self.device,
                )
                method_used = "hair_cards"
                logger.info(f"  ✓ Hair cards built: {hair_result}")
            except Exception as e:
                logger.warning(f"  Hair cards failed: {e}")

        # Option C: TSDF volume
        if hair_result is None:
            logger.info("Using Option C: TSDF hair volume (always available) …")
            hair_result = _fallback_hair_shell(head_mesh_path, output_dir)
            method_used = "tsdf_volume"

        results = {
            "hair_mesh":   hair_result,
            "method":      method_used,
            "output_dir":  str(output_dir),
        }

        logger.info("✓ Stage 7 complete")
        logger.info(f"  Method: {method_used}")
        logger.info(f"  Hair mesh: {hair_result}")
        return results
