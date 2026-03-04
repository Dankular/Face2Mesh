"""
Stage 8 — Lighting, Materials & Colour Space
=============================================
8a. DiffusionLight HDRI estimation from input photo
8b. IC-Light relighting for target scene
8c. sRGB → ACEScg linear colour space conversion (CRITICAL for film)
8d. Renderer-specific material export (Arnold, RenderMan, Cycles, Unreal)
"""

from __future__ import annotations
import gc
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"
ICLIGHT_DIR = MODELS_DIR / "iclight"


# ---------------------------------------------------------------------------
# 8a — DiffusionLight HDRI estimation
# ---------------------------------------------------------------------------

def estimate_hdri(
    image_path: str,
    output_dir: Path,
    device: str = "cuda",
) -> Optional[str]:
    """
    Run DiffusionLight to estimate the HDRI environment map from the input photo.
    Returns path to estimated .hdr file or None on failure.
    """
    dl_dir = MODELS_DIR / "diffusionlight_repo"

    if not dl_dir.exists():
        import subprocess, sys
        logger.info("Cloning DiffusionLight …")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/DiffusionLight/DiffusionLight.git", str(dl_dir)],
                check=True, timeout=120,
            )
        except Exception as e:
            logger.warning(f"DiffusionLight clone failed: {e}")
            return _estimate_hdri_simple(image_path, output_dir)

    dl_ckpt = MODELS_DIR / "diffusionlight"
    if not dl_ckpt.exists():
        try:
            from huggingface_hub import snapshot_download
            logger.info("Downloading DiffusionLight checkpoint …")
            snapshot_download(
                repo_id="DiffusionLight/DiffusionLight",
                local_dir=str(dl_ckpt),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.warning(f"DiffusionLight weights failed: {e}")
            return _estimate_hdri_simple(image_path, output_dir)

    try:
        import subprocess
        out = output_dir / "environment.hdr"
        result = subprocess.run(
            ["python3", str(dl_dir / "predict_illumination.py"),
             "--input",  image_path,
             "--output", str(out),
             "--checkpoint", str(dl_ckpt)],
            capture_output=True, text=True, cwd=str(dl_dir),
            timeout=300,
        )
        if result.returncode == 0 and out.exists():
            logger.info(f"  DiffusionLight HDRI: {out}")
            return str(out)
        else:
            logger.warning(f"DiffusionLight failed: {result.stderr[-500:]}")
    except Exception as e:
        logger.warning(f"DiffusionLight error: {e}")

    return _estimate_hdri_simple(image_path, output_dir)


def _estimate_hdri_simple(image_path: str, output_dir: Path) -> str:
    """
    Simple HDRI estimation: compute dominant light direction from face shading.
    Creates a basic environment map from brightness distribution.
    """
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    # Estimate light direction from brightest face region
    brightness = img.mean(axis=2)
    brightest_y, brightest_x = np.unravel_index(brightness.argmax(), brightness.shape)
    H, W = brightness.shape

    # Convert to spherical coordinates (rough estimate)
    light_az = (brightest_x / W - 0.5) * np.pi   # -π/2 to π/2
    light_el = (0.5 - brightest_y / H) * np.pi * 0.5

    # Create a simple lat-long HDRI (512×256)
    hdri_w, hdri_h = 512, 256
    hdri = np.ones((hdri_h, hdri_w, 3), dtype=np.float32) * 0.15  # ambient base

    # Add dominant light source
    for y in range(hdri_h):
        el = (0.5 - y / hdri_h) * np.pi
        for x in range(hdri_w):
            az = (x / hdri_w - 0.5) * 2 * np.pi
            # Gaussian blob for light
            dist = np.sqrt((az - light_az) ** 2 + (el - light_el) ** 2)
            strength = np.exp(-dist * dist / 0.1) * 5.0  # 5× ambient
            col = img[brightest_y, brightest_x]
            hdri[y, x] += col * strength

    # Write as EXR (linear)
    out_path = output_dir / "environment_estimated.hdr"
    _write_hdr(str(out_path), hdri)
    logger.info(f"  Simple HDRI estimation: {out_path}")
    return str(out_path)


def _write_hdr(path: str, hdr: np.ndarray):
    """Write RGBE .hdr file."""
    try:
        import imageio
        imageio.imwrite(path, hdr.astype(np.float32))
    except Exception:
        # Fallback: write as EXR via OpenCV
        try:
            import cv2
            cv2.imwrite(path.replace(".hdr", ".exr"), cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR))
        except Exception:
            # Last resort: save as PNG (tonemapped)
            tone = (hdr / (hdr.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(tone).save(path.replace(".hdr", "_preview.png"))


# ---------------------------------------------------------------------------
# 8b — IC-Light relighting
# ---------------------------------------------------------------------------

def relight_with_iclight(
    albedo_path: str,
    normal_path: str,
    hdri_path: Optional[str],
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, str]:
    """
    Relight the face texture using IC-Light conditioned on normal map.
    Generates multiple lighting variants.
    """
    results: Dict[str, str] = {}

    try:
        # Load IC-Light as a ControlNet-based pipeline
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import torch

        iclight_model_dir = ICLIGHT_DIR
        if not (iclight_model_dir / "config.json").exists():
            logger.warning("IC-Light not found at %s — skipping relighting", iclight_model_dir)
            results["relit_neutral"] = albedo_path
            return results

        logger.info("Loading IC-Light for relighting …")
        controlnet = ControlNetModel.from_pretrained(
            str(iclight_model_dir / "controlnet"), torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            str(iclight_model_dir), controlnet=controlnet,
            torch_dtype=torch.float16, safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_xformers_memory_efficient_attention()

        albedo_img = Image.open(albedo_path).convert("RGB").resize((512, 512))
        normal_img = Image.open(normal_path).convert("RGB").resize((512, 512)) if normal_path and Path(normal_path).exists() else Image.new("RGB", (512, 512), (128, 128, 255))

        lighting_scenarios = [
            ("neutral",     "neutral studio lighting, soft diffuse light, professional portrait"),
            ("key_left",    "strong key light from upper left, professional portrait photography"),
            ("key_right",   "strong key light from upper right, professional portrait photography"),
            ("back_light",  "rim lighting from behind, dramatic portrait"),
            ("three_point", "three-point lighting, fill light, key light, rim light, studio"),
        ]

        for scenario_name, prompt in lighting_scenarios:
            try:
                out = pipe(
                    prompt=prompt,
                    image=albedo_img,
                    control_image=normal_img,
                    num_inference_steps=20,
                    guidance_scale=5.0,
                    strength=0.5,
                ).images[0]
                out_path = output_dir / f"relit_{scenario_name}.png"
                out.save(str(out_path))
                results[f"relit_{scenario_name}"] = str(out_path)
                logger.info(f"  IC-Light {scenario_name}: {out_path}")
            except Exception as e:
                logger.warning(f"  IC-Light {scenario_name} failed: {e}")

        pipe.unet.cpu()
        del pipe, controlnet
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error("IC-Light relighting failed: %s", e)
        results["relit_neutral"] = albedo_path

    return results


# ---------------------------------------------------------------------------
# 8c — Colour space conversion: sRGB → ACEScg
# ---------------------------------------------------------------------------

# sRGB primaries to ACEScg primaries transformation matrix
# sRGB → XYZ D65 → XYZ D60 → ACEScg
SRGB_TO_ACESCG = np.array([
    [0.6131,  0.3395,  0.0474],
    [0.0701,  0.9164,  0.0136],
    [0.0206,  0.1096,  0.8698],
], dtype=np.float64)


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Apply inverse sRGB OETF: linear_val = ((srgb + 0.055) / 1.055)^2.4 for srgb > 0.04045."""
    img = img.astype(np.float64)
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    )
    return linear


def convert_texture_to_acescg(
    texture_path: str,
    output_dir: Path,
    is_linear_input: bool = False,
) -> str:
    """
    Convert a texture from sRGB/linear to ACEScg colour space.
    Output is a 16-bit EXR file in linear ACEScg.
    """
    img = np.array(Image.open(texture_path).convert("RGB")).astype(np.float64) / 255.0

    if not is_linear_input:
        # Assume sRGB input from diffusion models
        img = _srgb_to_linear(img)

    # Apply colour matrix
    H, W = img.shape[:2]
    flat = img.reshape(-1, 3)
    acescg = (SRGB_TO_ACESCG @ flat.T).T.clip(0, None)
    acescg = acescg.reshape(H, W, 3)

    stem = Path(texture_path).stem
    out_path = output_dir / f"{stem}_acescg.exr"

    try:
        import OpenEXR, Imath  # type: ignore
        header = OpenEXR.Header(W, H)
        header["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        }
        exr = OpenEXR.OutputFile(str(out_path), header)
        r = acescg[:, :, 0].astype(np.float16).tobytes()
        g = acescg[:, :, 1].astype(np.float16).tobytes()
        b = acescg[:, :, 2].astype(np.float16).tobytes()
        exr.writePixels({"R": r, "G": g, "B": b})
        exr.close()
    except ImportError:
        # Fall back to imageio/cv2 EXR
        try:
            import imageio
            imageio.imwrite(str(out_path), acescg.astype(np.float32))
        except Exception:
            # PNG fallback (tonemapped, not true ACEScg)
            out_path = output_dir / f"{stem}_acescg_preview.png"
            tone = (np.power(np.clip(acescg, 0, 1), 1/2.4) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(tone).save(str(out_path))
            logger.warning("  EXR write failed — saved ACEScg preview as PNG")

    logger.info(f"  ACEScg texture: {out_path}")
    return str(out_path)


def convert_all_textures_to_acescg(
    pbr_maps: Dict[str, str],
    output_dir: Path,
) -> Dict[str, str]:
    """
    Convert all PBR maps to ACEScg.
    Colour maps (albedo): apply sRGB inverse OETF + matrix.
    Data maps (normal, roughness, displacement): verify linear, no colour matrix.
    """
    acescg_maps = {}
    colour_keys = {"albedo", "albedo_iid", "relit_neutral", "relit_key_left",
                   "relit_key_right", "relit_back_light", "relit_three_point"}

    for key, path in pbr_maps.items():
        if path is None or not Path(path).exists():
            continue
        if key in colour_keys:
            # Full sRGB → ACEScg conversion
            acescg_maps[f"{key}_acescg"] = convert_texture_to_acescg(path, output_dir, is_linear_input=False)
        else:
            # Data map: just copy (already linear or not colour-managed)
            acescg_maps[key] = path

    return acescg_maps


# ---------------------------------------------------------------------------
# 8d — Renderer-specific material export
# ---------------------------------------------------------------------------

def export_material_for_renderer(
    pbr_maps: Dict[str, str],
    acescg_maps: Dict[str, str],
    renderer: str,
    output_dir: Path,
) -> str:
    """
    Write renderer-specific material export files.
    renderer: one of "arnold", "renderman", "cycles", "unreal"
    """
    renderer_dir = output_dir / renderer
    renderer_dir.mkdir(parents=True, exist_ok=True)

    albedo   = acescg_maps.get("albedo_acescg", pbr_maps.get("albedo", ""))
    normal   = pbr_maps.get("normal",    "")
    roughness = pbr_maps.get("roughness", "")
    displacement = pbr_maps.get("displacement_map", "")

    if renderer == "arnold":
        data = {
            "shader": "aiStandardSurface",
            "base_color":     albedo,
            "normal_map":     normal,
            "roughness":      roughness,
            "sss_mode":       "randomwalk_v2",
            "sss_radius":     [0.042, 0.022, 0.018],   # R, G, B in cm (skin)
            "sss_weight":     0.3,
            "specular_ior":   1.45,
            "displacement":   displacement,
            "comment": "Arnold aiStandardSurface — all maps in ACEScg linear EXR",
        }
    elif renderer == "renderman":
        data = {
            "shader": "PxrSurface",
            "diffuse_color":  albedo,
            "bump_normal":    normal,
            "roughness":      roughness,
            "subsurface":     "burley",
            "sss_mfp":        [0.042, 0.022, 0.018],
            "sss_mfp_color":  [0.95, 0.60, 0.45],
            "spec_fresnel_mode": "ior",
            "spec_ior":       1.45,
            "displacement":   displacement,
        }
    elif renderer == "cycles":
        data = {
            "shader": "Principled BSDF",
            "Base Color":     albedo,
            "Normal":         normal,
            "Roughness":      roughness,
            "Subsurface":     0.05,
            "Subsurface Radius": [0.042, 0.022, 0.018],
            "Subsurface IOR": 1.45,
            "Specular":       0.5,
            "IOR":            1.45,
            "Displacement":   displacement,
            "note": "Enable Adaptive Subdivision + Displacement for normal/displ maps",
        }
    elif renderer == "unreal":
        data = {
            "shader": "SubsurfaceProfile",
            "Base Color":     albedo,
            "Normal":         normal,
            "Roughness":      roughness,
            "Specular":       0.5,
            "Subsurface Profile": {
                "ScatterRadius":  1.2,
                "SubsurfaceColor": [0.95, 0.60, 0.45, 1.0],
                "FalloffColor":   [0.75, 0.30, 0.15, 1.0],
            },
            "note": "Bake SSS profile as separate UE5 asset. Import maps as Linear (not sRGB) for data channels.",
        }
    else:
        data = {"maps": pbr_maps, "acescg_maps": acescg_maps}

    mat_path = renderer_dir / "material.json"
    with open(str(mat_path), "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"  {renderer} material: {mat_path}")
    return str(mat_path)


# ---------------------------------------------------------------------------
# Stage 8 entry point
# ---------------------------------------------------------------------------

class LightingStage:
    RENDERERS = ["arnold", "renderman", "cycles", "unreal"]

    def __init__(self, device: str = "cuda"):
        self.device = device

    def run(
        self,
        face_image_path: str,
        pbr_maps: Dict[str, str],
        output_dir: Path,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("STAGE 8: Lighting, Materials & ACEScg Conversion")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 8a: Estimate HDRI
        logger.info("Step 8a: DiffusionLight HDRI estimation …")
        hdri_path = estimate_hdri(face_image_path, output_dir, self.device)

        # 8b: IC-Light relighting
        logger.info("Step 8b: IC-Light relighting variants …")
        albedo  = pbr_maps.get("albedo", face_image_path)
        normal  = pbr_maps.get("normal", "")
        relit_maps = relight_with_iclight(albedo, normal, hdri_path, output_dir, self.device)
        pbr_maps.update(relit_maps)

        # 8c: Convert all colour textures to ACEScg
        logger.info("Step 8c: sRGB → ACEScg conversion …")
        acescg_maps = convert_all_textures_to_acescg(pbr_maps, output_dir)

        # 8d: Renderer-specific exports
        logger.info("Step 8d: Renderer-specific material export …")
        renderer_materials: Dict[str, str] = {}
        all_maps = {**pbr_maps, **acescg_maps}
        for renderer in self.RENDERERS:
            renderer_materials[renderer] = export_material_for_renderer(
                pbr_maps, acescg_maps, renderer, output_dir,
            )

        results = {
            "hdri":                hdri_path,
            "relit_maps":          relit_maps,
            "acescg_maps":         acescg_maps,
            "renderer_materials":  renderer_materials,
            "output_dir":          str(output_dir),
        }

        logger.info("✓ Stage 8 complete")
        logger.info(f"  HDRI: {hdri_path}")
        logger.info(f"  Relit variants: {list(relit_maps.keys())}")
        logger.info(f"  ACEScg maps: {list(acescg_maps.keys())}")
        logger.info(f"  Renderer exports: {self.RENDERERS}")
        return results
