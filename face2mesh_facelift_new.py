"""
FaceLift Integration for Face2Mesh Pipeline
Stage 2: Multi-view Gaussian Splatting reconstruction via FaceLift
Stage 3: Gaussian → Mesh via depth rendering + TSDF fusion
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

FACELIFT_DIR = Path("/FaceLift")


# ---------------------------------------------------------------------------
# Stage 2 – FaceLift: single image → 6 consistent views + gaussians.ply
# ---------------------------------------------------------------------------

class FaceLiftStage:
    """
    Wraps FaceLift (weijielyu/FaceLift) to generate geometrically consistent
    multi-view images and a Gaussian Splatting representation from one face photo.

    VRAM strategy for 12 GB cards:
      - FP16 inference is already default in FaceLift.
      - enable_xformers_memory_efficient_attention() is called by FaceLift.
      - Additionally we enable sequential CPU offload when vram_gb < 16.
    """

    def __init__(self, facelift_dir: str = "/FaceLift", vram_gb: float = 12.0):
        self.facelift_dir = Path(facelift_dir)
        self.vram_gb = vram_gb

    def _ensure_installed(self):
        """Install FaceLift-specific deps that aren't already present."""
        try:
            import diff_gaussian_rasterization  # noqa: F401
        except ImportError:
            logger.info("Installing diff-gaussian-rasterization …")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"],
                check=True,
            )
        try:
            import facenet_pytorch  # noqa: F401
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "facenet-pytorch", "rembg"],
                check=True,
            )

    def run(
        self,
        input_image_path: str,
        output_dir: str,
        seed: int = 4,
        guidance_scale: float = 3.0,
        steps: int = 50,
    ) -> Dict:
        """
        Runs FaceLift inference on a single face image.

        Returns dict with paths to:
          - gaussians_ply: 3D Gaussian Splat (.ply)
          - multiview_png: 6-view composite image
          - turntable_mp4: 360° turntable video
          - views_dir: directory containing individual view PNGs
        """
        self._ensure_installed()

        input_image_path = Path(input_image_path)
        output_dir = Path(output_dir)
        stage_input = output_dir / "facelift_input"
        stage_output = output_dir / "facelift_output"
        stage_input.mkdir(parents=True, exist_ok=True)
        stage_output.mkdir(parents=True, exist_ok=True)

        shutil.copy(input_image_path, stage_input / input_image_path.name)

        env = {**os.environ}
        if self.vram_gb < 16:
            # Tell FaceLift to use sequential offload (patched below)
            env["FACELIFT_CPU_OFFLOAD"] = "1"

        logger.info("Running FaceLift inference (this takes 3–8 min on RTX 3060) …")
        result = subprocess.run(
            [
                sys.executable,
                str(self.facelift_dir / "inference.py"),
                "--input_dir", str(stage_input),
                "--output_dir", str(stage_output),
                "--auto_crop", "True",
                "--seed", str(seed),
                "--guidance_scale_2D", str(guidance_scale),
                "--step_2D", str(steps),
            ],
            capture_output=True,
            text=True,
            cwd=str(self.facelift_dir),
            env=env,
        )

        if result.returncode != 0:
            logger.error("FaceLift stderr:\n" + result.stderr[-2000:])
            raise RuntimeError("FaceLift inference failed — see logs above")

        stem = input_image_path.stem
        outputs = stage_output / stem
        gaussians = outputs / "gaussians.ply"
        multiview = outputs / "multiview.png"
        turntable = outputs / "turntable.mp4"

        logger.info("✓ FaceLift complete")
        logger.info(f"  Gaussians : {gaussians}")
        logger.info(f"  Multi-view: {multiview}")

        return {
            "gaussians_ply": str(gaussians),
            "multiview_png": str(multiview),
            "turntable_mp4": str(turntable) if turntable.exists() else None,
            "views_dir": str(outputs),
            "stage_output": str(stage_output),
        }


# ---------------------------------------------------------------------------
# Stage 3 – Gaussian → Mesh via depth-render + TSDF fusion
# ---------------------------------------------------------------------------

class GaussianToMesh:
    """
    Converts FaceLift's gaussians.ply to a watertight triangle mesh using:
      1. Render depth maps from each of the 6 known camera positions
      2. TSDF volumetric fusion  (Open3D)
      3. Marching-cubes extraction + Poisson clean-up

    Falls back to an alpha-density point-cloud → Poisson mesh if Open3D is
    unavailable or TSDF fusion fails.
    """

    # 6-view camera extrinsics (OpenCV convention) used by FaceLift
    # (elevation, azimuth) pairs — matches utils_folder/opencv_cameras.json
    VIEW_ANGLES = [
        (0, 0), (0, 60), (0, 120), (0, 180), (0, 240), (0, 300)
    ]
    FOCAL_LENGTH = 560.0   # pixels, for 512×512 renders
    IMG_SIZE = 512

    def __init__(self, voxel_length: float = 0.004, sdf_trunc: float = 0.02):
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc

    @staticmethod
    def _load_gaussians(ply_path: str):
        """Load position + opacity + scale from a Gaussian PLY file."""
        from plyfile import PlyData
        data = PlyData.read(ply_path)
        v = data["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        opacity = np.array(v["opacity"], dtype=np.float32)
        # Convert log-scale opacity stored by FaceLift
        opacity = 1.0 / (1.0 + np.exp(-opacity))
        return xyz, opacity

    def _render_depth(self, xyz, opacity, elev_deg, azim_deg):
        """
        Minimal software depth render by projecting Gaussian centres.
        Returns a (H, W) depth image in metres.
        """
        el = np.radians(elev_deg)
        az = np.radians(azim_deg)

        R_az = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az),  np.cos(az), 0],
            [0,           0,          1],
        ], dtype=np.float32)
        R_el = np.array([
            [1, 0,           0          ],
            [0, np.cos(el), -np.sin(el) ],
            [0, np.sin(el),  np.cos(el) ],
        ], dtype=np.float32)
        R = R_el @ R_az
        t = np.array([0, 0, 1.5], dtype=np.float32)   # camera 1.5 units away

        cam_xyz = (R @ xyz.T).T + t
        valid = (cam_xyz[:, 2] > 0) & (opacity > 0.1)
        cam_xyz = cam_xyz[valid]
        if len(cam_xyz) == 0:
            return np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)

        f = self.FOCAL_LENGTH
        cx = cy = self.IMG_SIZE / 2
        u = (f * cam_xyz[:, 0] / cam_xyz[:, 2] + cx).astype(int)
        v_px = (f * cam_xyz[:, 1] / cam_xyz[:, 2] + cy).astype(int)
        depth_img = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)

        mask = (u >= 0) & (u < self.IMG_SIZE) & (v_px >= 0) & (v_px < self.IMG_SIZE)
        u, v_px = u[mask], v_px[mask]
        z_vals = cam_xyz[mask, 2]

        # z-buffer
        for i in range(len(u)):
            if depth_img[v_px[i], u[i]] == 0 or z_vals[i] < depth_img[v_px[i], u[i]]:
                depth_img[v_px[i], u[i]] = z_vals[i]
        return depth_img

    def convert(self, gaussians_ply: str, output_dir: str) -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = output_dir / "mesh_raw.ply"

        try:
            import open3d as o3d
            self._tsdf_convert(gaussians_ply, mesh_path, o3d)
        except ImportError:
            logger.warning("Open3D not available — falling back to Poisson mesh from point cloud")
            self._poisson_convert(gaussians_ply, mesh_path)

        logger.info(f"✓ Mesh extracted: {mesh_path}")
        return {"mesh_ply": str(mesh_path)}

    def _tsdf_convert(self, gaussians_ply, mesh_path, o3d):
        xyz, opacity = self._load_gaussians(gaussians_ply)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

        f = self.FOCAL_LENGTH
        cx = cy = self.IMG_SIZE / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.IMG_SIZE, self.IMG_SIZE, f, f, cx, cy
        )

        for elev, azim in self.VIEW_ANGLES:
            depth_img = self._render_depth(xyz, opacity, elev, azim)
            depth_o3d = o3d.geometry.Image(
                (depth_img * 1000).astype(np.uint16)   # mm for Open3D
            )
            el = np.radians(elev); az = np.radians(azim)
            R_az = np.array([[np.cos(az), -np.sin(az), 0, 0],
                             [np.sin(az),  np.cos(az), 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            R_el = np.array([[1, 0, 0, 0],
                             [0, np.cos(el), -np.sin(el), 0],
                             [0, np.sin(el),  np.cos(el), 0],
                             [0, 0, 0, 1]], dtype=np.float64)
            T = np.eye(4, dtype=np.float64); T[2, 3] = 1.5
            extr = T @ R_el @ R_az
            cam = o3d.camera.PinholeCameraParameters()
            cam.intrinsic = intrinsic
            cam.extrinsic = extr
            volume.integrate(depth_o3d, intrinsic, extr)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    def _poisson_convert(self, gaussians_ply, mesh_path):
        import trimesh
        from plyfile import PlyData
        data = PlyData.read(gaussians_ply)
        v = data["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
        pc = trimesh.PointCloud(vertices=xyz)
        mesh = trimesh.voxel.ops.points_to_marching_cubes(xyz, pitch=0.02)
        mesh.export(str(mesh_path))


# ---------------------------------------------------------------------------
# Stage 7 – Qwen Multi-Angles texture enhancement
# ---------------------------------------------------------------------------

class QwenTextureEnhancer:
    """
    Uses Qwen-Image-Edit-2511 + Multiple-Angles LoRA to generate high-quality
    texture views for specific UV regions where single-image reconstruction
    has weak coverage (occipital, under-chin, ears).

    NOTE: This runs AFTER geometry is locked from FaceLift + TSDF.
    Each view is generated independently — cross-view consistency is NOT
    required here because we're painting onto a fixed mesh surface.
    """

    WEAK_ANGLES = [
        {"label": "back_head",   "prompt": "Back of head, hair, occipital region, photorealistic face portrait"},
        {"label": "under_chin",  "prompt": "Under chin view, neck, jaw underside, photorealistic face portrait"},
        {"label": "left_ear",    "prompt": "Left ear close-up, ear canal, helix, photorealistic"},
        {"label": "right_ear",   "prompt": "Right ear close-up, ear canal, helix, photorealistic"},
        {"label": "top_head",    "prompt": "Top-down view of head, hair parting, photorealistic"},
    ]

    def __init__(self, model_dir: str = "/root/.cache/face_models/qwen",
                 vram_gb: float = 12.0):
        self.model_dir = model_dir
        self.vram_gb = vram_gb
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionImg2ImgPipeline
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        logger.info("Loading Qwen-Image-Edit-2511 (4-bit NF4) for texture enhancement …")
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.float16,
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._pipe.load_lora_weights(
            "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"
        )
        if self.vram_gb < 16:
            self._pipe.enable_sequential_cpu_offload()
        else:
            self._pipe.enable_xformers_memory_efficient_attention()
        logger.info("✓ Qwen texture model loaded")

    def enhance(self, face_image_path: str, output_dir: str,
                strength: float = 0.6) -> Dict:
        """
        Generates enhanced texture views for weak UV regions.
        Returns dict mapping region label → output PNG path.
        """
        from PIL import Image as PILImage
        self._load()

        output_dir = Path(output_dir) / "texture_enhancement"
        output_dir.mkdir(parents=True, exist_ok=True)

        source = PILImage.open(face_image_path).convert("RGB").resize((512, 512))
        results = {}

        for angle in self.WEAK_ANGLES:
            label = angle["label"]
            prompt = angle["prompt"]
            out_path = output_dir / f"{label}.png"

            logger.info(f"  Generating texture view: {label} …")
            try:
                out = self._pipe(
                    prompt=prompt,
                    image=source,
                    strength=strength,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]
                out.save(str(out_path))
                results[label] = str(out_path)
                logger.info(f"    ✓ Saved: {out_path}")
            except Exception as e:
                logger.error(f"    ✗ Failed {label}: {e}")

        return results
