"""
Face2Mesh — Model Downloader
=============================
Downloads all required models for the complete 10-stage pipeline.

FLAME NOTE (Manual): The FLAME 2020 model requires MPI registration.
  1. Register at: https://flame.is.tue.mpg.de/
  2. Download: FLAME2020.zip
  3. Extract generic_model.pkl to: ~/.cache/face_models/flame/generic_model.pkl
  4. Download FLAME UV map (FLAME_UV_coor_new.npz) from the same page.

Disk space required: ~60-70 GB total
  - Qwen GGUF Q4_0:    11.9 GB
  - HRN:                8.0 GB
  - NeuralHaircut:     10.0 GB
  - DiffusionLight:     8.0 GB
  - EMOTE/DiffPoseTalk: 4.0 GB
  - MICA/SMIRK:         3.0 GB
  - MultiAngles LoRA:   0.1 GB
  - DSINE:              0.3 GB
  - Already downloaded: bisenet, depth_anything_v2, iclight, marigold
"""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

from typing import Optional
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def hf_download(repo_id: str, dest: Path, description: str,
                filename: Optional[str] = None) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {description} ({repo_id}) …")
    try:
        if filename:
            hf_hub_download(repo_id=repo_id, filename=filename,
                            local_dir=str(dest), local_dir_use_symlinks=False)
        else:
            snapshot_download(repo_id=repo_id, local_dir=str(dest),
                              local_dir_use_symlinks=False)
        logger.info(f"  ✓ {description}")
        return True
    except Exception as e:
        logger.error(f"  ✗ {description}: {e}")
        return False


def wget_download(url: str, dest_file: Path, description: str) -> bool:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if dest_file.exists():
        logger.info(f"  ✓ {description} (already exists)")
        return True
    logger.info(f"  Downloading {description} from {url} …")
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest_file), url],
            check=True,
        )
        logger.info(f"  ✓ {description}")
        return True
    except Exception as e:
        logger.error(f"  ✗ {description}: {e}")
        return False


def git_clone(url: str, dest: Path, description: str) -> bool:
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"  ✓ {description} (already cloned)")
        return True
    logger.info(f"  Cloning {description} from {url} …")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True, timeout=300,
        )
        logger.info(f"  ✓ {description}")
        return True
    except Exception as e:
        logger.error(f"  ✗ {description}: {e}")
        return False


def main():

    logger.info("=" * 60)
    logger.info("FACE2MESH MODEL DOWNLOADER")
    logger.info("=" * 60)
    logger.info(f"Cache: {MODELS_DIR}")
    df = subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout
    logger.info(f"Disk: {df.splitlines()[-1] if df else 'unknown'}")
    logger.info("")

    results = {}

    # ----------------------------------------------------------------
    # CRITICAL: Qwen-Image-Edit-2511 GGUF Q4_0 (Stage 3b)
    # ----------------------------------------------------------------
    logger.info("=== QWEN IMAGE EDIT (Stage 3b) ===")
    qwen_dir = MODELS_DIR / "qwen"
    qwen_dir.mkdir(exist_ok=True)

    # GGUF Q4_0 (11.9GB) — user-identified solution for 12GB VRAM
    results["qwen_gguf"] = hf_download(
        "unsloth/Qwen-Image-Edit-2511-GGUF",
        qwen_dir,
        "Qwen-Image-Edit-2511 GGUF Q4_0 (11.9GB)",
        filename="qwen-image-edit-2511-Q4_0.gguf",
    )
    # Also try mmproj (vision encoder) for llama-cpp-python
    hf_download(
        "unsloth/Qwen-Image-Edit-2511-GGUF",
        qwen_dir,
        "Qwen mmproj vision encoder",
        filename="mmproj-model-f16.gguf",
    )

    # MultiAngles LoRA (Stage 3b identity-consistent views)
    results["angles_lora"] = hf_download(
        "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        MODELS_DIR / "angles_lora",
        "MultiAngles LoRA for Qwen",
    )

    # Lightning LoRA (4-step fast inference)
    results["lightning_lora"] = hf_download(
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        MODELS_DIR / "lightning_lora",
        "Lightning LoRA for Qwen (4-step)",
    )

    # ----------------------------------------------------------------
    # SMIRK (Stage 1 — FLAME expression params)
    # ----------------------------------------------------------------
    logger.info("\n=== SMIRK (Stage 1) ===")
    smirk_dir = MODELS_DIR / "smirk"
    results["smirk_weights"] = hf_download(
        "georgeretsi/smirk",
        smirk_dir,
        "SMIRK expression estimation checkpoint (~2GB)",
    )
    results["smirk_code"] = git_clone(
        "https://github.com/georgeretsi/smirk.git",
        MODELS_DIR / "smirk_repo",
        "SMIRK code repository",
    )

    # ----------------------------------------------------------------
    # MICA (Stage 1 — FLAME shape params)
    # ----------------------------------------------------------------
    logger.info("\n=== MICA (Stage 1) ===")
    mica_dir = MODELS_DIR / "mica"
    mica_tar = mica_dir / "mica.tar"
    results["mica_weights"] = wget_download(
        "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c16a3be/?dl=1",
        mica_tar,
        "MICA checkpoint (~512MB)",
    )
    if results["mica_weights"] and mica_tar.exists():
        logger.info("  Extracting MICA …")
        try:
            subprocess.run(["tar", "-xf", str(mica_tar), "-C", str(mica_dir)], check=True)
        except Exception as e:
            logger.warning(f"  MICA tar extraction: {e}")
    results["mica_code"] = git_clone(
        "https://github.com/Zielon/MICA.git",
        MODELS_DIR / "mica_repo",
        "MICA code repository",
    )

    # ----------------------------------------------------------------
    # DSINE (Stage 4 — normal maps)
    # ----------------------------------------------------------------
    logger.info("\n=== DSINE (Stage 4 — normal maps) ===")
    dsine_dir = MODELS_DIR / "dsine"
    dsine_dir.mkdir(exist_ok=True)
    results["dsine_code"] = git_clone(
        "https://github.com/baegwangbin/DSINE.git",
        MODELS_DIR / "dsine_repo",
        "DSINE code",
    )
    results["dsine_weights"] = wget_download(
        "https://huggingface.co/baegwangbin/DSINE/resolve/main/scannet.pt",
        dsine_dir / "scannet.pt",
        "DSINE scannet checkpoint (~300MB)",
    )

    # ----------------------------------------------------------------
    # HRN (Stage 4 — pore microdetail)
    # ----------------------------------------------------------------
    logger.info("\n=== HRN (Stage 4 — pore detail) ===")
    results["hrn_code"] = git_clone(
        "https://github.com/youngLBW/HRN.git",
        MODELS_DIR / "hrn_repo",
        "HRN code",
    )
    results["hrn_weights"] = hf_download(
        "youngLBW/HRN",
        MODELS_DIR / "hrn" / "hrn_pretrained",
        "HRN pretrained weights (~8GB)",
    )

    # ----------------------------------------------------------------
    # NeuralHaircut (Stage 7)
    # ----------------------------------------------------------------
    logger.info("\n=== NeuralHaircut (Stage 7) ===")
    results["neuralhaircut_code"] = git_clone(
        "https://github.com/SamsungLabs/NeuralHaircut.git",
        MODELS_DIR / "neuralhaircut_repo",
        "NeuralHaircut code",
    )
    # Weights from HF (if available)
    nh_ckpt = MODELS_DIR / "neuralhaircut"
    nh_ckpt.mkdir(exist_ok=True)
    try:
        results["neuralhaircut_weights"] = hf_download(
            "SamsungLabs/NeuralHaircut",
            nh_ckpt,
            "NeuralHaircut weights (~10GB)",
        )
    except Exception:
        results["neuralhaircut_weights"] = False

    # ----------------------------------------------------------------
    # DiffusionLight (Stage 8)
    # ----------------------------------------------------------------
    logger.info("\n=== DiffusionLight (Stage 8) ===")
    results["diffusionlight_code"] = git_clone(
        "https://github.com/DiffusionLight/DiffusionLight.git",
        MODELS_DIR / "diffusionlight_repo",
        "DiffusionLight code",
    )
    results["diffusionlight_weights"] = hf_download(
        "DiffusionLight/DiffusionLight",
        MODELS_DIR / "diffusionlight",
        "DiffusionLight HDRI estimation (~8GB)",
    )

    # ----------------------------------------------------------------
    # EMOTE (Stage 9)
    # ----------------------------------------------------------------
    logger.info("\n=== EMOTE (Stage 9 — animation) ===")
    results["emote_code"] = git_clone(
        "https://github.com/radekd91/inferno.git",
        MODELS_DIR / "emote_repo",
        "Inferno/EMOTE code",
    )
    try:
        results["emote_weights"] = hf_download(
            "radekd91/EMOTE",
            MODELS_DIR / "emote",
            "EMOTE weights (~4GB)",
        )
    except Exception:
        results["emote_weights"] = False

    # ----------------------------------------------------------------
    # DiffPoseTalk (Stage 9 fallback)
    # ----------------------------------------------------------------
    logger.info("\n=== DiffPoseTalk (Stage 9 — animation) ===")
    results["diffposetalk_code"] = git_clone(
        "https://github.com/YoungSeng/DiffPoseTalk.git",
        MODELS_DIR / "diffposetalk_repo",
        "DiffPoseTalk code",
    )
    try:
        results["diffposetalk_weights"] = hf_download(
            "YoungSeng/DiffPoseTalk",
            MODELS_DIR / "diffposetalk",
            "DiffPoseTalk weights (~4GB)",
        )
    except Exception:
        results["diffposetalk_weights"] = False

    # ----------------------------------------------------------------
    # FLAME — MANUAL DOWNLOAD REQUIRED
    # ----------------------------------------------------------------
    logger.info("\n=== FLAME (Stages 4 & 5) — MANUAL DOWNLOAD REQUIRED ===")
    flame_dir = MODELS_DIR / "flame"
    flame_pkl = flame_dir / "generic_model.pkl"
    if flame_pkl.exists():
        logger.info("  ✓ FLAME model found")
        results["flame"] = True
    else:
        logger.warning(
            "\n" + "!" * 60 + "\n"
            "  FLAME model NOT found. This is required for:\n"
            "  - Stage 4: FLAME topology retopology\n"
            "  - Stage 5: LBS rig transfer and 50 blendshapes\n\n"
            "  To download:\n"
            "  1. Register at: https://flame.is.tue.mpg.de/\n"
            "  2. Download FLAME2020.zip\n"
            "  3. Extract and copy:\n"
            f"     cp generic_model.pkl {flame_dir}/\n"
            "  4. Also copy FLAME_UV_coor_new.npz to the same directory.\n"
            "\n  Without FLAME: xatlas retopology + generic blendshapes are used.\n"
            + "!" * 60
        )
        results["flame"] = False

    # ----------------------------------------------------------------
    # Install Python dependencies
    # ----------------------------------------------------------------
    logger.info("\n=== INSTALLING PYTHON DEPENDENCIES ===")
    deps = [
        "lpips", "pyiqa", "xatlas", "plyfile",
        "scikit-image", "scipy", "librosa", "imageio",
        "peft", "bitsandbytes",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + deps,
        check=False,
    )

    # llama-cpp-python with CUDA support (for GGUF backend)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "llama-cpp-python",
         "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"],
        check=False,
    )

    logger.info("\n")
    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    success = sum(1 for v in results.values() if v)
    total   = len(results)
    for k, v in results.items():
        logger.info(f"  {'✓' if v else '✗'} {k}")

    logger.info(f"\nSuccessful: {success}/{total}")
    if not results.get("flame"):
        logger.warning("\n⚠ FLAME not found — manual download required for full pipeline quality")
    logger.info(f"\nAll models in: {MODELS_DIR}")
    logger.info("Run pipeline: python3 pipeline_complete.py --input your_photo.jpg")


if __name__ == "__main__":
    main()
