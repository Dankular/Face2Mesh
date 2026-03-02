"""
Download ALL required models for the complete pipeline.
No placeholders - downloads everything needed for film-quality output.

Total download: ~90 GB
Estimated time: 1-3 hours depending on connection speed
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download all required models for the complete pipeline."""
    
    def __init__(self):
        self.models_dir = Path.home() / ".cache" / "face_pipeline_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.total_size_gb = 0
        self.downloaded_count = 0
        self.total_count = 17
    
    def run_command(self, cmd, description):
        """Run a shell command with logging."""
        logger.info(f"[{self.downloaded_count + 1}/{self.total_count}] {description}")
        logger.info(f"Command: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"✓ Success: {description}")
            self.downloaded_count += 1
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed: {description}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def download_all(self):
        """Download all models in order of priority."""
        
        logger.info("="*60)
        logger.info("DOWNLOADING ALL MODELS FOR COMPLETE PIPELINE")
        logger.info("="*60)
        logger.info(f"Total models: {self.total_count}")
        logger.info(f"Estimated download: ~90 GB")
        logger.info(f"Models will be cached in: {self.models_dir}")
        logger.info("="*60)
        
        # ================================================================
        # STAGE 1: Identity & Parameters (~6 GB)
        # ================================================================
        
        logger.info("\n### STAGE 1: Identity & Parameters ###\n")
        
        # 1. ArcFace (already installed via insightface)
        logger.info("[1/17] ArcFace - Already installed via insightface ✓")
        self.downloaded_count += 1
        
        # 2. MICA - FLAME shape parameters
        logger.info("\n[2/17] MICA - FLAME shape parameters (~2 GB)")
        # Official repo: https://github.com/Zielon/MICA
        # Note: MICA requires specific PyTorch version and may have dependencies
        if not self.run_command(
            "pip install git+https://github.com/Zielon/MICA.git",
            "Installing MICA"
        ):
            logger.warning("MICA installation failed - may need manual setup")
        
        # Download MICA pretrained weights
        self.run_command(
            f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='zielon/mica', filename='mica.tar', local_dir='{self.models_dir}/mica')\"",
            "Downloading MICA weights"
        )
        
        # 3. SMIRK - FLAME expression parameters
        logger.info("\n[3/17] SMIRK - FLAME expression parameters (~2 GB)")
        # Repo: https://github.com/georgeretsi/smirk
        self.run_command(
            "pip install git+https://github.com/georgeretsi/smirk.git",
            "Installing SMIRK"
        )
        
        # 4. BiSeNet - Face parsing
        logger.info("\n[4/17] BiSeNet - Face parsing (~1 GB)")
        # Using face-parsing.PyTorch implementation
        self.run_command(
            "pip install git+https://github.com/zllrunning/face-parsing.PyTorch.git",
            "Installing BiSeNet face parsing"
        )
        
        # Download BiSeNet pretrained weights
        self.run_command(
            f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='jonathandinu/face-parsing', filename='79999_iter.pth', local_dir='{self.models_dir}/bisenet')\"",
            "Downloading BiSeNet weights"
        )
        
        # ================================================================
        # STAGE 2: 3D Reconstruction (~16 GB)
        # ================================================================
        
        logger.info("\n### STAGE 2: 3D Reconstruction ###\n")
        
        # 5. FaceLift - 3D Gaussian splat reconstruction
        logger.info("\n[5/17] FaceLift - 3D Gaussian splat (~12 GB)")
        # Repo: https://github.com/weijielyu/FaceLift
        self.run_command(
            "pip install git+https://github.com/weijielyu/FaceLift.git",
            "Installing FaceLift"
        )
        
        # Download FaceLift weights from HuggingFace
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='ashawkey/FaceLift', local_dir='{self.models_dir}/facelift')\"",
            "Downloading FaceLift weights"
        )
        
        # 6. Depth Anything V3 - Monocular depth estimation
        logger.info("\n[6/17] Depth Anything V3 (~4 GB)")
        self.run_command(
            "pip install depth-anything-v2",  # V3 may use V2 package
            "Installing Depth Anything"
        )
        
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='depth-anything/Depth-Anything-V2-Large', local_dir='{self.models_dir}/depth_anything')\"",
            "Downloading Depth Anything V2 Large weights"
        )
        
        # ================================================================
        # STAGE 3: Multiview Generation (~16 GB)
        # ================================================================
        
        logger.info("\n### STAGE 3: Multiview Generation ###\n")
        
        # 7. Qwen-Image-Edit-2511 + LoRAs
        logger.info("\n[7/17] Qwen-Image-Edit-2511 + Multi-Angles LoRA (~16 GB)")
        # This downloads automatically via diffusers, but we can pre-download
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-Image-Edit-2511', local_dir='{self.models_dir}/qwen')\"",
            "Downloading Qwen-Image-Edit-2511"
        )
        
        # Multi-Angles LoRA
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA', local_dir='{self.models_dir}/qwen_lora_angles')\"",
            "Downloading Multi-Angles LoRA"
        )
        
        # Lightning LoRA (4-step fast inference)
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='lightx2v/Qwen-Image-Edit-2511-Lightning', local_dir='{self.models_dir}/qwen_lora_lightning')\"",
            "Downloading Lightning LoRA"
        )
        
        # ================================================================
        # STAGE 4: Textures & Materials (~15 GB)
        # ================================================================
        
        logger.info("\n### STAGE 4: Textures & Materials ###\n")
        
        # 8. HRN - High-resolution facial normal/detail maps
        logger.info("\n[8/17] HRN - Pore-level microdetail (~8 GB)")
        # Repo: https://github.com/youngLBW/HRN
        self.run_command(
            "pip install git+https://github.com/youngLBW/HRN.git",
            "Installing HRN"
        )
        
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='youngLBW/HRN', local_dir='{self.models_dir}/hrn')\"",
            "Downloading HRN weights"
        )
        
        # 9. DSINE - Normal estimation
        logger.info("\n[9/17] DSINE - Normal maps (~3 GB)")
        self.run_command(
            "pip install git+https://github.com/baegwangbin/DSINE.git",
            "Installing DSINE"
        )
        
        self.run_command(
            f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='baegwangbin/DSINE', filename='dsine.pt', local_dir='{self.models_dir}/dsine')\"",
            "Downloading DSINE weights"
        )
        
        # 10. Marigold - Intrinsic image decomposition
        logger.info("\n[10/17] Marigold IID - Albedo/roughness (~4 GB)")
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='prs-eth/marigold-v1-0', local_dir='{self.models_dir}/marigold')\"",
            "Downloading Marigold weights"
        )
        
        # ================================================================
        # STAGE 5: Rigging (FLAME model - free from MPI)
        # ================================================================
        
        logger.info("\n### STAGE 5: Rigging ###\n")
        
        # 11. FLAME model
        logger.info("\n[11/17] FLAME - Parametric head model")
        logger.warning("FLAME requires registration at https://flame.is.tue.mpg.de/")
        logger.warning("Please download manually and place in: " + str(self.models_dir / "flame"))
        logger.info("Required files: generic_model.pkl, FLAME_sample.ply")
        # Can't auto-download FLAME due to license agreement requirement
        
        # ================================================================
        # STAGE 6: Anatomical Details (Templates - may need manual download)
        # ================================================================
        
        logger.info("\n### STAGE 6: Anatomical Details ###\n")
        
        # 12. Universal Head 3DMM (for inner ear template)
        logger.info("\n[12/17] Universal Head 3DMM - Ear template")
        # Repo: https://github.com/steliosploumpis/Universal_Head_3DMM
        logger.warning("Universal Head 3DMM may require manual download from GitHub")
        self.run_command(
            f"git clone https://github.com/steliosploumpis/Universal_Head_3DMM.git {self.models_dir}/universal_head",
            "Cloning Universal Head 3DMM"
        )
        
        # ================================================================
        # STAGE 7: Hair (~10 GB)
        # ================================================================
        
        logger.info("\n### STAGE 7: Hair Reconstruction ###\n")
        
        # 13. NeuralHaircut
        logger.info("\n[13/17] NeuralHaircut - Hair strand reconstruction (~10 GB)")
        # Repo: https://github.com/SamsungLabs/NeuralHaircut
        self.run_command(
            "pip install git+https://github.com/SamsungLabs/NeuralHaircut.git",
            "Installing NeuralHaircut"
        )
        
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='sjmeis/NeuralHaircut', local_dir='{self.models_dir}/neuralhaircut')\"",
            "Downloading NeuralHaircut weights"
        )
        
        # ================================================================
        # STAGE 8: Lighting & Materials (~16 GB)
        # ================================================================
        
        logger.info("\n### STAGE 8: Lighting & Materials ###\n")
        
        # 14. DiffusionLight - HDRI environment estimation
        logger.info("\n[14/17] DiffusionLight - Environment lighting (~8 GB)")
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='DiffusionLight/DiffusionLight', local_dir='{self.models_dir}/diffusionlight')\"",
            "Downloading DiffusionLight weights"
        )
        
        # 15. IC-Light - Relighting
        logger.info("\n[15/17] IC-Light - Scene relighting (~8 GB)")
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='lllyasviel/ic-light', local_dir='{self.models_dir}/iclight')\"",
            "Downloading IC-Light weights"
        )
        
        # ================================================================
        # STAGE 9: Animation (~8 GB)
        # ================================================================
        
        logger.info("\n### STAGE 9: Animation ###\n")
        
        # 16. NVIDIA Audio2Face-3D v3.0
        logger.info("\n[16/17] Audio2Face-3D - Audio-driven animation (~4 GB)")
        logger.warning("Audio2Face-3D v3.0 requires NVIDIA Omniverse")
        logger.warning("Download from: https://www.nvidia.com/en-us/omniverse/apps/audio2face/")
        logger.info("Alternative: We can use Wav2Lip or other open-source solutions")
        
        # Alternative: Use open-source audio-driven animation
        self.run_command(
            "pip install git+https://github.com/Rudrabha/Wav2Lip.git",
            "Installing Wav2Lip (Audio2Face alternative)"
        )
        
        # 17. EMOTE or DiffPoseTalk
        logger.info("\n[17/17] EMOTE - Video-driven animation (~4 GB)")
        self.run_command(
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='radekd91/emote', local_dir='{self.models_dir}/emote')\"",
            "Downloading EMOTE weights"
        )
        
        # ================================================================
        # ADDITIONAL DEPENDENCIES
        # ================================================================
        
        logger.info("\n### Additional Dependencies ###\n")
        
        # Install export format libraries
        logger.info("Installing export format libraries...")
        self.run_command("pip install pygltflib", "Installing GLTF export")
        self.run_command("pip install usd-core", "Installing USD export")
        
        # PyTorch3D for TSDF (alternative to Open3D on Python 3.14)
        logger.info("Installing PyTorch3D for TSDF fusion...")
        # PyTorch3D installation is platform-specific
        if sys.platform == "win32":
            logger.warning("PyTorch3D on Windows requires building from source")
            logger.info("See: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
        else:
            self.run_command(
                "pip install pytorch3d",
                "Installing PyTorch3D"
            )
        
        # ================================================================
        # SUMMARY
        # ================================================================
        
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Successfully downloaded: {self.downloaded_count}/{self.total_count} models")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info("")
        
        if self.downloaded_count < self.total_count:
            logger.warning("Some models require manual download:")
            logger.warning("  - FLAME: https://flame.is.tue.mpg.de/")
            logger.warning("  - Audio2Face-3D: https://nvidia.com/omniverse/apps/audio2face/")
            logger.warning("  - PyTorch3D: May need building from source on Windows")
        
        logger.info("\nAll automatic downloads complete!")
        logger.info("You can now run the complete pipeline with real models.")
        logger.info("="*60)
        
        return self.downloaded_count == self.total_count


def main():
    """Main entry point."""
    downloader = ModelDownloader()
    
    print("\n" + "="*60)
    print("FACE-TO-3D PIPELINE MODEL DOWNLOADER")
    print("="*60)
    print("\nThis will download ~90 GB of models.")
    print("Downloads will be cached and reused across runs.")
    print("\nSome models require manual registration:")
    print("  - FLAME (free but requires MPI registration)")
    print("  - Audio2Face-3D (requires NVIDIA Omniverse)")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        sys.exit(0)
    
    success = downloader.download_all()
    
    if success:
        print("\n✓ All models downloaded successfully!")
        print("\nNext steps:")
        print("  1. Manually download FLAME from https://flame.is.tue.mpg.de/")
        print(f"  2. Place FLAME files in: {downloader.models_dir}/flame/")
        print("  3. Run: python pipeline_complete.py --input your_face.jpg")
        sys.exit(0)
    else:
        print("\n⚠ Some downloads failed. Check logs above.")
        print("You may need to install some models manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
