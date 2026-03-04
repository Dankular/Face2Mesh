"""
COMPLETE Face-to-3D Pipeline — Face2Mesh
=========================================
Film-quality single-photo → rigged 3D head avatar.

10-stage pipeline per Complete_Pipeline_Walkthrough.md:
  Stage 1  — Identity extraction       (ArcFace + MICA + SMIRK + BiSeNet)
  Stage 2  — Multi-view Gaussians      (FaceLift → gaussians.ply + 6 views)
  Stage 3  — Mesh extraction           (depth render + Depth Anything V3 + TSDF fusion)
  Stage 3b — High-detail texture views (Qwen-Image-Edit + MultiAngles LoRA → 24 views)
  Stage 4  — Mesh + textures + PBR     (FLAME retopo + UV + baking + DSINE + HRN)
  Stage 5  — Rig & blendshapes         (FLAME LBS + 50+ blendshapes + correctives)
  Stage 6  — Eyes / teeth / tongue     (layered eyes, 168-tri teeth, 700-tri tongue, eyelashes)
  Stage 7  — Hair reconstruction       (NeuralHaircut → hair cards → TSDF volume)
  Stage 8  — Lighting & materials      (DiffusionLight HDRI + IC-Light + ACEScg export)
  Stage 9  — Animation                 (Audio2Face/EMOTE/DiffPoseTalk + temporal smoothing)
  Stage 10 — Validation & export       (ArcFace CSIM + LPIPS + SSIM + GLB/FBX/USD)

FLAME NOTE: Stages 4 and 5 require the FLAME 2020 model.
  Register at: https://flame.is.tue.mpg.de/
  Place generic_model.pkl at: ~/.cache/face_models/flame/generic_model.pkl
  Without FLAME, xatlas retopology is used as fallback.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Stage implementations
from stages.stage1_identity   import IdentityExtractor
from stages.stage3_multiview  import MultiViewGenerator, VIEW_ANGLES
from stages.stage4_textures   import TextureStage
from stages.stage5_rig        import RigStage
from stages.stage6_detail     import DetailGeometryStage
from stages.stage7_hair       import HairStage
from stages.stage8_lighting   import LightingStage
from stages.stage9_animation  import AnimationStage
from stages.stage10_validation import ValidationStage

# Existing FaceLift + TSDF stages
from face2mesh_facelift import FaceLiftStage, GaussianToMesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


# ---------------------------------------------------------------------------
# Depth Anything V3 refinement (Stage 2 step 2.3)
# ---------------------------------------------------------------------------

def refine_depth_with_depth_anything(
    depth_img: np.ndarray,
    rgb_img: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run Depth Anything V2 to refine Gaussian-derived depth via affine alignment.
    Returns refined depth map (H, W).
    """
    from transformers import pipeline as hf_pipeline

    da_dir = Path.home() / ".cache" / "face_models" / "depth_anything_v2"
    model_id = str(da_dir) if (da_dir / "config.json").exists() else "depth-anything/Depth-Anything-V2-Large"

    try:
        pipe = hf_pipeline(
            "depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1,
        )
        from PIL import Image as PILImage
        pil_rgb = PILImage.fromarray(rgb_img.astype(np.uint8))
        result  = pipe(pil_rgb)
        mono_depth = np.array(result["depth"]).astype(np.float32)

        # Scale-align mono depth to Gaussian depth (affine: s * d + t)
        valid = (depth_img > 0) & (mono_depth > 0)
        if valid.sum() > 100:
            A = np.column_stack([mono_depth[valid], np.ones(valid.sum())])
            b = depth_img[valid]
            coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            refined = mono_depth * coeff[0] + coeff[1]
        else:
            refined = depth_img

        del pipe
        torch.cuda.empty_cache()
        return np.clip(refined, 0, depth_img.max() * 1.5)

    except Exception as e:
        logger.warning(f"Depth Anything V2 failed: {e} — using raw Gaussian depth")
        return depth_img


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class CompleteFaceTo3DPipeline:

    def __init__(
        self,
        device: str    = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./output_complete",
        facelift_dir: str = "/FaceLift",
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        fps: int = 30,
    ):
        self.device      = device
        self.output_dir  = Path(output_dir)
        self.facelift_dir = facelift_dir
        self.audio_path  = audio_path
        self.video_path  = video_path
        self.fps         = fps
        self.vram        = vram_gb()

        self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("=" * 70)
        logger.info("FACE2MESH — Complete Film Avatar Pipeline")
        logger.info("=" * 70)
        logger.info(f"  Device  : {device}")
        logger.info(f"  VRAM    : {self.vram:.1f} GB")
        logger.info(f"  Output  : {self.output_dir}")

        # Check FLAME availability early
        flame_pkl = Path.home() / ".cache" / "face_models" / "flame" / "generic_model.pkl"
        if not flame_pkl.exists():
            logger.warning(
                "\n" + "!" * 70 + "\n"
                "  FLAME model not found. Stages 4 and 5 will use fallback methods.\n"
                "  For film-quality output: register at https://flame.is.tue.mpg.de/\n"
                "  and place generic_model.pkl at:\n"
                f"  {flame_pkl}\n" + "!" * 70
            )

    # -----------------------------------------------------------------------
    # STAGE 1 — Identity extraction
    # -----------------------------------------------------------------------

    def stage1(self, image_path: str) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage1_identity"
        result = IdentityExtractor(self.device).run(image_path, stage_dir)
        logger.info(f"Stage 1 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 2 — FaceLift multi-view + Gaussian splatting
    # -----------------------------------------------------------------------

    def stage2(self, image_path: str) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage2_facelift"
        facelift  = FaceLiftStage(facelift_dir=self.facelift_dir, vram_gb=self.vram)
        result    = facelift.run(
            input_image_path=image_path,
            output_dir=str(stage_dir),
            seed=4,
            guidance_scale=3.0,
            steps=50,
        )
        logger.info(f"Stage 2 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 3 — Gaussian → Mesh (72-96 depth maps + TSDF)
    # -----------------------------------------------------------------------

    def stage3_mesh(self, stage2_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage3_mesh"
        g2m       = GaussianToMesh(voxel_length=0.003, sdf_trunc=0.015)
        result    = g2m.convert(
            gaussians_ply=stage2_data["gaussians_ply"],
            output_dir=str(stage_dir),
        )
        logger.info(f"Stage 3 (mesh) done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 3b — Qwen 24-view texture generation
    # -----------------------------------------------------------------------

    def stage3b_qwen_views(self, image_path: str, stage1_data: Dict,
                            stage2_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir   = self.output_dir / "stage3b_qwen_views"
        generator   = MultiViewGenerator(self.device)
        result      = generator.generate(
            image_path=image_path,
            output_dir=stage_dir,
            source_arcface=stage1_data["arcface_embedding"],
            facelift_views_dir=stage2_data.get("views_dir"),
        )
        logger.info(f"Stage 3b (24-view) done in {time.time()-t0:.1f}s")
        logger.info(f"  Mean CSIM: {result.get('mean_csim', 0):.3f}")
        return result

    # -----------------------------------------------------------------------
    # STAGE 4 — Textures & PBR
    # -----------------------------------------------------------------------

    def stage4(self, stage3_data: Dict, stage3b_data: Dict,
                stage1_data: Dict, image_path: str) -> Dict:
        t0 = time.time()
        stage_dir  = self.output_dir / "stage4_textures"
        tex_stage  = TextureStage(self.device)
        result     = tex_stage.run(
            tsdf_mesh_path=stage3_data["mesh_ply"],
            views=stage3b_data.get("views", {}),
            view_angles=VIEW_ANGLES,
            flame_beta=stage1_data.get("flame_shape_beta", np.zeros(300)),
            source_arcface=stage1_data["arcface_embedding"],
            smirk_expression=stage1_data.get("flame_expression", np.zeros(50)),
            face_image_path=image_path,
            output_dir=stage_dir,
        )
        logger.info(f"Stage 4 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 5 — Rig & blendshapes
    # -----------------------------------------------------------------------

    def stage5(self, stage4_data: Dict, stage1_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir  = self.output_dir / "stage5_rig"
        rig_stage  = RigStage(self.device)
        result     = rig_stage.run(
            retopo_mesh_path=stage4_data.get("retopo_mesh", stage4_data.get("uv_mesh", stage4_data["retopo_mesh"])),
            flame_beta=stage1_data.get("flame_shape_beta", np.zeros(300)),
            output_dir=stage_dir,
        )
        logger.info(f"Stage 5 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 6 — Eyes, teeth, tongue, detail geometry
    # -----------------------------------------------------------------------

    def stage6(self, stage3_data: Dict, stage1_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage6_detail"

        # Get head mesh bounding box
        try:
            import trimesh
            mesh = trimesh.load(stage3_data["mesh_ply"], process=False)
            bounds = np.array([mesh.bounds[0], mesh.bounds[1]])
        except Exception:
            bounds = None

        result = DetailGeometryStage().run(
            output_dir=stage_dir,
            seg_mask=stage1_data.get("seg_mask"),
            head_mesh_bounds=bounds,
        )
        logger.info(f"Stage 6 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 7 — Hair
    # -----------------------------------------------------------------------

    def stage7(self, image_path: str, stage3_data: Dict, stage1_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage7_hair"
        result    = HairStage(self.device).run(
            reference_image_path=image_path,
            head_mesh_path=stage3_data["mesh_ply"],
            seg_mask=stage1_data.get("seg_mask", np.zeros((512, 512), dtype=np.uint8)),
            output_dir=stage_dir,
        )
        logger.info(f"Stage 7 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 8 — Lighting & materials
    # -----------------------------------------------------------------------

    def stage8(self, image_path: str, stage4_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage8_lighting"
        result    = LightingStage(self.device).run(
            face_image_path=image_path,
            pbr_maps=stage4_data.get("pbr_maps", {}),
            output_dir=stage_dir,
        )
        logger.info(f"Stage 8 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 9 — Animation
    # -----------------------------------------------------------------------

    def stage9(self, stage5_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage9_animation"
        n_verts   = len(stage5_data.get("neutral_verts", np.zeros((5023, 3))))
        result    = AnimationStage(self.device, self.fps).run(
            output_dir=stage_dir,
            audio_path=self.audio_path,
            video_path=self.video_path,
            n_verts=n_verts,
        )
        logger.info(f"Stage 9 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # STAGE 10 — Validation & export
    # -----------------------------------------------------------------------

    def stage10(self, image_path: str, stage1_data: Dict, stage3_data: Dict,
                 stage3b_data: Dict, stage4_data: Dict, stage5_data: Dict,
                 stage9_data: Dict) -> Dict:
        t0 = time.time()
        stage_dir = self.output_dir / "stage10_validation"

        bs_dir = stage5_data.get("output_dir")
        if bs_dir:
            bs_dir = str(Path(bs_dir) / "blendshapes")

        result = ValidationStage(self.device).run(
            mesh_path=stage3_data["mesh_ply"],
            reference_image_path=image_path,
            source_arcface_embedding=stage1_data["arcface_embedding"],
            view_render_paths=stage3b_data.get("views", {}),
            anim_params_path=stage9_data.get("params_npy"),
            pbr_maps=stage4_data.get("pbr_maps", {}),
            rig_json=stage5_data.get("rig_json"),
            blendshape_dir=bs_dir,
            output_dir=stage_dir,
            fps=self.fps,
        )
        logger.info(f"Stage 10 done in {time.time()-t0:.1f}s")
        return result

    # -----------------------------------------------------------------------
    # Full pipeline runner
    # -----------------------------------------------------------------------

    def run(self, image_path: str) -> Dict:
        pipeline_start = time.time()
        image_path = str(Path(image_path).resolve())

        r1  = self.stage1(image_path)
        r2  = self.stage2(image_path)
        r3  = self.stage3_mesh(r2)
        r3b = self.stage3b_qwen_views(image_path, r1, r2)
        r4  = self.stage4(r3, r3b, r1, image_path)
        r5  = self.stage5(r4, r1)
        r6  = self.stage6(r3, r1)
        r7  = self.stage7(image_path, r3, r1)
        r8  = self.stage8(image_path, r4)
        r9  = self.stage9(r5)
        r10 = self.stage10(image_path, r1, r3, r3b, r4, r5, r9)

        elapsed = time.time() - pipeline_start
        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETE in {elapsed/60:.1f} min")
        logger.info(f"Overall pass: {r10.get('overall_pass', False)}")
        logger.info(f"GLB: {r10.get('exports', {}).get('glb', 'N/A')}")
        logger.info("=" * 70)

        return {
            "stage1": r1,  "stage2": r2,  "stage3": r3,  "stage3b": r3b,
            "stage4": r4,  "stage5": r5,  "stage6": r6,  "stage7":  r7,
            "stage8": r8,  "stage9": r9,  "stage10": r10,
            "elapsed_minutes": elapsed / 60,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face2Mesh — Complete Film Avatar Pipeline")
    parser.add_argument("--input",       required=True,          help="Path to face image")
    parser.add_argument("--output",      default="./output_complete")
    parser.add_argument("--facelift",    default="/FaceLift")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audio",       default=None,           help="Audio file for animation (optional)")
    parser.add_argument("--video",       default=None,           help="Actor video for retargeting (optional)")
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--stage",       type=int, default=0,    help="Run single stage (0 = full pipeline)")
    args = parser.parse_args()

    pipeline = CompleteFaceTo3DPipeline(
        device=args.device,
        output_dir=args.output,
        facelift_dir=args.facelift,
        audio_path=args.audio,
        video_path=args.video,
        fps=args.fps,
    )

    if args.stage == 0:
        pipeline.run(args.input)
    elif args.stage == 1:
        pipeline.stage1(args.input)
    elif args.stage == 2:
        pipeline.stage2(args.input)
    elif args.stage == 3:
        r2 = pipeline.stage2(args.input)
        pipeline.stage3_mesh(r2)
    else:
        logger.error("Use --stage 0 for full pipeline or 1/2/3 for individual stages")
        sys.exit(1)
