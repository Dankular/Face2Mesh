"""
Stage 1 — Identity Extraction
=============================
ArcFace embedding + MICA FLAME shape + SMIRK expression params + BiSeNet segmentation.

FLAME model must be placed at: /root/.cache/face_models/flame/generic_model.pkl
Register and download from: https://flame.is.tue.mpg.de/
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"
FLAME_PKL = MODELS_DIR / "flame" / "generic_model.pkl"
MICA_DIR  = MODELS_DIR / "mica"
SMIRK_DIR = MODELS_DIR / "smirk"


# ---------------------------------------------------------------------------
# ArcFace
# ---------------------------------------------------------------------------

def run_arcface(image: Image.Image, ctx_id: int = 0) -> np.ndarray:
    """Extract 512-dim ArcFace identity embedding via insightface antelopev2."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="antelopev2",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    faces = app.get(np.array(image))
    if not faces:
        logger.warning("ArcFace: no face detected, returning zero embedding")
        return np.zeros(512, dtype=np.float32)
    best = max(faces, key=lambda f: f.det_score)
    logger.info(f"  ArcFace: det_score={best.det_score:.3f}")
    return best.embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# MICA — FLAME shape estimation
# ---------------------------------------------------------------------------

def _ensure_mica() -> Path:
    """Clone MICA repo and download weights if needed."""
    mica_code = MODELS_DIR / "mica_repo"
    if not mica_code.exists():
        logger.info("Cloning MICA repository …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Zielon/MICA.git", str(mica_code)],
            check=True,
        )
        # Install deps
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r",
             str(mica_code / "requirements.txt")],
            check=True,
        )
    ckpt = MICA_DIR / "mica.tar"
    if not ckpt.exists():
        MICA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MICA checkpoint (~512 MB) …")
        subprocess.run(
            ["wget", "-q", "-O", str(ckpt),
             "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c16a3be/?dl=1"],
            check=True,
        )
        subprocess.run(["tar", "-xf", str(ckpt), "-C", str(MICA_DIR)], check=True)
    return mica_code


def run_mica(image_path: str) -> Dict:
    """
    Run MICA to estimate FLAME shape parameters from a face photo.
    Returns: {"beta": np.ndarray (300,), "flame_mesh": np.ndarray (5023, 3)}
    """
    if not FLAME_PKL.exists():
        logger.error(
            "FLAME model not found at %s\n"
            "Register at https://flame.is.tue.mpg.de/ and place generic_model.pkl there.",
            FLAME_PKL,
        )
        return {"beta": np.zeros(300), "flame_mesh": None}

    try:
        mica_code = _ensure_mica()
        import tempfile, json

        out_dir = Path(tempfile.mkdtemp())
        inp_dir = out_dir / "input"
        inp_dir.mkdir()
        import shutil
        shutil.copy(image_path, inp_dir / Path(image_path).name)

        # MICA writes FLAME_parameters.npy to out_dir/<name>/
        result = subprocess.run(
            [sys.executable, str(mica_code / "demo.py"),
             "--input_dir", str(inp_dir),
             "--output_dir", str(out_dir),
             "--checkpoint", str(MICA_DIR),
             "--flame_model_path", str(FLAME_PKL)],
            capture_output=True, text=True, cwd=str(mica_code),
        )
        if result.returncode != 0:
            logger.error("MICA failed:\n%s", result.stderr[-2000:])
            return {"beta": np.zeros(300), "flame_mesh": None}

        stem = Path(image_path).stem
        param_path = out_dir / stem / "FLAME_parameters.npy"
        if param_path.exists():
            params = np.load(str(param_path), allow_pickle=True).item()
            beta = params.get("betas", np.zeros(300)).flatten()[:300]
            logger.info(f"  MICA: shape params extracted, norm={np.linalg.norm(beta):.3f}")
            return {"beta": beta, "flame_mesh": params.get("vertices", None)}
        else:
            logger.warning("MICA output not found at %s", param_path)
            return {"beta": np.zeros(300), "flame_mesh": None}

    except Exception as exc:
        logger.error("MICA error: %s", exc)
        return {"beta": np.zeros(300), "flame_mesh": None}


# ---------------------------------------------------------------------------
# SMIRK — FLAME expression + jaw estimation
# ---------------------------------------------------------------------------

def _ensure_smirk() -> Path:
    """Clone SMIRK repo and download weights."""
    smirk_code = MODELS_DIR / "smirk_repo"
    if not smirk_code.exists():
        logger.info("Cloning SMIRK repository …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/georgeretsi/smirk.git", str(smirk_code)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r",
             str(smirk_code / "requirements.txt")],
            check=True,
        )
    # Download SMIRK checkpoint from HF
    ckpt = SMIRK_DIR / "SMIRK_em1.pt"
    if not ckpt.exists():
        SMIRK_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading SMIRK checkpoint …")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="georgeretsi/smirk",
            filename="SMIRK_em1.pt",
            local_dir=str(SMIRK_DIR),
        )
    return smirk_code


def run_smirk(image_path: str, device: str = "cuda") -> Dict:
    """
    Run SMIRK to extract FLAME expression parameters from a face photo.
    Returns: {"expression": np.ndarray (50,), "jaw_pose": np.ndarray (3,),
              "eyelid": np.ndarray (2,)}
    """
    if not FLAME_PKL.exists():
        logger.warning("FLAME not found — skipping SMIRK. Set up FLAME first.")
        return {"expression": np.zeros(50), "jaw_pose": np.zeros(3), "eyelid": np.zeros(2)}

    try:
        smirk_code = _ensure_smirk()
        if str(smirk_code) not in sys.path:
            sys.path.insert(0, str(smirk_code))

        from smirk.smirk_encoder import SmirkEncoder  # noqa
        from datasets.base_dataset import get_transform  # type: ignore

        ckpt_path = SMIRK_DIR / "SMIRK_em1.pt"
        smirk = SmirkEncoder().to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        smirk.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
        smirk.eval()

        transform = get_transform(is_train=False)
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = smirk(tensor)

        expression = out.get("expression_params", torch.zeros(1, 50)).squeeze().cpu().numpy()
        jaw_pose   = out.get("jaw_params",        torch.zeros(1, 3)).squeeze().cpu().numpy()
        eyelid     = out.get("eyelid_params",     torch.zeros(1, 2)).squeeze().cpu().numpy()

        logger.info(f"  SMIRK: expression norm={np.linalg.norm(expression):.3f}  jaw={jaw_pose}")
        return {"expression": expression, "jaw_pose": jaw_pose, "eyelid": eyelid}

    except Exception as exc:
        logger.error("SMIRK error: %s", exc)
        return {"expression": np.zeros(50), "jaw_pose": np.zeros(3), "eyelid": np.zeros(2)}


# ---------------------------------------------------------------------------
# BiSeNet — face parsing / segmentation
# ---------------------------------------------------------------------------

BISENET_CLASSES = {
    0: "background", 1: "skin", 2: "l_brow", 3: "r_brow",
    4: "l_eye", 5: "r_eye", 6: "eye_g", 7: "l_ear", 8: "r_ear",
    9: "ear_r", 10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
    14: "neck", 15: "neck_l", 16: "cloth", 17: "hair", 18: "hat",
}

def run_bisenet(image: Image.Image, device: str = "cuda") -> np.ndarray:
    """
    Run BiSeNet face parsing. Returns (H, W) integer mask with class IDs.
    """
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch.nn.functional as F

    bisenet_dir = MODELS_DIR / "bisenet"
    proc  = SegformerImageProcessor.from_pretrained(str(bisenet_dir))
    model = SegformerForSemanticSegmentation.from_pretrained(str(bisenet_dir)).to(device)
    model.eval()

    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    upsampled = F.interpolate(
        logits, size=(image.height, image.width),
        mode="bilinear", align_corners=False,
    )
    mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    logger.info(f"  BiSeNet: mask shape={mask.shape}, unique classes={np.unique(mask).tolist()}")
    return mask


def get_face_region_mask(seg_mask: np.ndarray, include_classes=(1, 2, 3, 4, 5, 10, 11, 12, 13)) -> np.ndarray:
    """Return binary mask of face skin + brow + eye + nose + lip regions."""
    mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for cls in include_classes:
        mask |= (seg_mask == cls).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Stage 1 entry point
# ---------------------------------------------------------------------------

class IdentityExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def run(self, image_path: str, output_dir: Path) -> Dict:
        logger.info("=" * 60)
        logger.info("STAGE 1: Identity Extraction")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")

        # 1.1 ArcFace
        logger.info("Step 1.1: ArcFace identity embedding …")
        arcface_emb = run_arcface(image)
        np.save(str(output_dir / "arcface_embedding.npy"), arcface_emb)

        # 1.2 MICA — FLAME shape
        logger.info("Step 1.2: MICA FLAME shape estimation …")
        mica_result = run_mica(image_path)
        np.save(str(output_dir / "flame_shape_beta.npy"), mica_result["beta"])
        if mica_result["flame_mesh"] is not None:
            np.save(str(output_dir / "flame_mesh_vertices.npy"), mica_result["flame_mesh"])

        # 1.3 SMIRK — expression
        logger.info("Step 1.3: SMIRK expression estimation …")
        smirk_result = run_smirk(image_path, self.device)
        np.save(str(output_dir / "flame_expression.npy"),  smirk_result["expression"])
        np.save(str(output_dir / "flame_jaw_pose.npy"),    smirk_result["jaw_pose"])
        np.save(str(output_dir / "flame_eyelid.npy"),      smirk_result["eyelid"])

        # 1.4 BiSeNet segmentation
        logger.info("Step 1.4: BiSeNet face segmentation …")
        seg_mask = run_bisenet(image, self.device)
        np.save(str(output_dir / "seg_mask.npy"), seg_mask)

        # Save segmentation as PNG for visual inspection
        from PIL import Image as PILImage
        seg_vis = (seg_mask * (255 // max(1, seg_mask.max()))).astype(np.uint8)
        PILImage.fromarray(seg_vis).save(str(output_dir / "seg_mask.png"))

        results = {
            "arcface_embedding":  arcface_emb,
            "flame_shape_beta":   mica_result["beta"],
            "flame_expression":   smirk_result["expression"],
            "flame_jaw_pose":     smirk_result["jaw_pose"],
            "flame_eyelid":       smirk_result["eyelid"],
            "seg_mask":           seg_mask,
            "output_dir":         str(output_dir),
            "flame_mesh_vertices": mica_result.get("flame_mesh"),
        }

        logger.info("✓ Stage 1 complete")
        logger.info(f"  ArcFace embedding norm: {np.linalg.norm(arcface_emb):.3f}")
        logger.info(f"  FLAME shape (beta) norm: {np.linalg.norm(mica_result['beta']):.3f}")
        logger.info(f"  FLAME expression norm: {np.linalg.norm(smirk_result['expression']):.3f}")
        return results
