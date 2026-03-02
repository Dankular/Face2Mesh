"""
Direct model downloader - Downloads ONLY the actual working weights from HuggingFace.
No pip installs of research repos (they don't have setup.py).
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def download_model(repo_id, local_dir, description):
    """Download a model from HuggingFace."""
    logger.info(f"Downloading {description}...")
    logger.info(f"  Repo: {repo_id}")
    logger.info(f"  Dest: {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"  SUCCESS!")
        return True
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return False


def main():
    """Download all models that are actually available on HuggingFace."""
    
    models_dir = Path.home() / ".cache" / "face_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DOWNLOADING REAL MODELS FROM HUGGINGFACE")
    logger.info("="*60)
    logger.info(f"Cache dir: {models_dir}")
    logger.info("")
    
    downloads = []
    
    # ==================================================================
    # CRITICAL MODELS (These exist and will work)
    # ==================================================================
    
    # 1. Qwen-Image-Edit-2511 (Multiview generation)
    downloads.append({
        'repo': 'Qwen/Qwen-Image-Edit-2511',
        'dir': models_dir / 'qwen',
        'desc': 'Qwen-Image-Edit-2511 (~16 GB)',
        'priority': 'CRITICAL'
    })
    
    # 2. Depth Anything V2
    downloads.append({
        'repo': 'depth-anything/Depth-Anything-V2-Large',
        'dir': models_dir / 'depth_anything_v2',
        'desc': 'Depth Anything V2 Large (~1.3 GB)',
        'priority': 'HIGH'
    })
    
    # 3. Marigold depth/normal
    downloads.append({
        'repo': 'prs-eth/marigold-v1-0',
        'dir': models_dir / 'marigold',
        'desc': 'Marigold depth (~4 GB)',
        'priority': 'MEDIUM'
    })
    
    # 4. IC-Light (relighting)
    downloads.append({
        'repo': 'lllyasviel/ic-light',
        'dir': models_dir / 'iclight',
        'desc': 'IC-Light relighting (~8 GB)',
        'priority': 'MEDIUM'
    })
    
    # 5. BiSeNet face parsing weights
    downloads.append({
        'repo': 'jonathandinu/face-parsing',
        'dir': models_dir / 'bisenet',
        'desc': 'BiSeNet face parsing (~100 MB)',
        'priority': 'HIGH'
    })
    
    # 6. EMOCA (FLAME-based expression/shape - alternative to MICA/SMIRK)
    downloads.append({
        'repo': 'radekd91/emoca',
        'dir': models_dir / 'emoca',
        'desc': 'EMOCA FLAME parameters (~2 GB)',
        'priority': 'HIGH'
    })
    
    # Execute downloads
    logger.info("Starting downloads...")
    logger.info("")
    
    success_count = 0
    total = len(downloads)
    
    for i, item in enumerate(downloads):
        logger.info(f"[{i+1}/{total}] {item['priority']} - {item['desc']}")
        if download_model(item['repo'], str(item['dir']), item['desc']):
            success_count += 1
        logger.info("")
    
    # Summary
    logger.info("="*60)
    logger.info(f"DOWNLOAD COMPLETE: {success_count}/{total} models")
    logger.info("="*60)
    logger.info("")
    logger.info("Downloaded models:")
    logger.info(f"  1. Qwen-Image-Edit-2511 -> Multiview generation")
    logger.info(f"  2. Depth Anything V2 -> Depth refinement")
    logger.info(f"  3. Marigold -> Depth/normals")
    logger.info(f"  4. IC-Light -> Relighting")
    logger.info(f"  5. BiSeNet -> Face parsing")
    logger.info(f"  6. EMOCA -> FLAME parameters")
    logger.info("")
    logger.info("MISSING MODELS (not on HuggingFace / require setup):")
    logger.info("  - FaceLift: Research code only, no pretrained weights public")
    logger.info("  - HRN: Research code only")
    logger.info("  - DSINE: Research code only")
    logger.info("  - NeuralHaircut: Research code only")
    logger.info("")
    logger.info("ALTERNATIVE APPROACH:")
    logger.info("  Instead of research models, use production-ready alternatives:")
    logger.info("  - TripoSR (HuggingFace) instead of FaceLift")
    logger.info("  - Stable Diffusion inpainting instead of custom normal maps")
    logger.info("  - FLAME template directly instead of MICA/SMIRK")
    logger.info("")
    logger.info(f"Total size: ~30 GB (instead of 90 GB)")
    logger.info("All models cached in: " + str(models_dir))


if __name__ == "__main__":
    main()
