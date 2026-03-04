# -*- coding: utf-8 -*-
"""
PrereqAgent — runs BEFORE each stage.
Checks every hard requirement, auto-downloads what it can,
halts with a clear fix instruction for anything requiring human action.
"""
from __future__ import annotations
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)

FLAME_PATH  = Path.home() / ".cache" / "face_models" / "flame" / "generic_model.pkl"
QWEN_PATH   = Path.home() / ".cache" / "face_models" / "qwen_gguf" / "qwen-image-edit-2511-Q4_0.gguf"
FACELIFT_DIR = Path("/FaceLift")

QWEN_URL = (
    "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/"
    "qwen-image-edit-2511-Q4_0.gguf"
)


@dataclass
class PrereqReport:
    passed: bool
    stage: int
    missing: List[str] = field(default_factory=list)
    auto_fixed: List[str] = field(default_factory=list)
    manual_required: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"=== Stage {self.stage} Prereq {'PASS' if self.passed else 'FAIL'} ==="]
        if self.auto_fixed:
            lines.append("  Auto-fixed: " + ", ".join(self.auto_fixed))
        if self.manual_required:
            lines.append("  MANUAL ACTION REQUIRED:")
            for m in self.manual_required:
                lines.append(f"    !! {m}")
        if self.missing:
            lines.append("  Still missing: " + ", ".join(self.missing))
        return "\n".join(lines)


class PrereqAgent:
    """Check and fix prerequisites for each pipeline stage."""

    def check(self, stage: int) -> PrereqReport:
        report = PrereqReport(passed=True, stage=stage)
        checks = {
            1: self._check_stage1,
            2: self._check_stage2,
            3: self._check_stage3,
            4: self._check_stage4,
            5: self._check_stage5,
            6: self._check_stage6,
            7: self._check_stage7,
            8: self._check_stage8,
            9: self._check_stage9,
            10: self._check_stage10,
        }
        checks.get(stage, lambda r: None)(report)

        if report.manual_required or report.missing:
            report.passed = False

        logger.info(report.summary())
        return report

    # ── Stage checks ──────────────────────────────────────────────────────────

    def _check_stage1(self, r: PrereqReport):
        self._require_package(r, "insightface", pip="insightface")
        self._require_package(r, "onnxruntime", pip="onnxruntime-gpu")
        self._require_package(r, "cv2", pip="opencv-python-headless")
        # MICA / SMIRK download on first run via HF — check hub reachable
        self._require_package(r, "huggingface_hub", pip="huggingface-hub")
        self._require_cuda(r)

    def _check_stage2(self, r: PrereqReport):
        self._require_cuda(r)
        self._require_nvcc(r)

        if not FACELIFT_DIR.exists():
            r.missing.append("FaceLift repo not cloned at /FaceLift")
            r.manual_required.append(
                "Clone FaceLift: git clone https://github.com/weijielyu/FaceLift.git /FaceLift "
                "&& bash /FaceLift/setup_env.sh"
            )

        try:
            import open3d
        except ImportError:
            r.missing.append("open3d")
            r.manual_required.append(
                "open3d missing — requires Python 3.11. "
                "Install: pip install open3d>=0.18.0"
            )

        try:
            import diff_gaussian_rasterization  # noqa: F401
        except ImportError:
            r.missing.append("diff-gaussian-rasterization (CUDA extension)")
            self._try_auto_fix(
                r,
                "diff-gaussian-rasterization",
                "pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization",
            )

        self._require_package(r, "transformers", pip="transformers>=4.44.0")

    def _check_stage3(self, r: PrereqReport):
        self._require_cuda(r)
        if not QWEN_PATH.exists():
            r.missing.append(f"Qwen GGUF not found at {QWEN_PATH}")
            self._try_download_qwen(r)

        try:
            import llama_cpp  # noqa: F401
            # Verify CUDA build
            import llama_cpp.llama as _lc
            if not getattr(_lc, "LLAMA_SUPPORTS_GPU_OFFLOAD", False):
                r.missing.append("llama-cpp-python built without CUDA")
                r.manual_required.append(
                    "Rebuild: CMAKE_ARGS='-DGGML_CUDA=on' "
                    "pip install llama-cpp-python --force-reinstall --no-cache-dir"
                )
        except ImportError:
            r.missing.append("llama-cpp-python")
            r.manual_required.append(
                "Install CUDA build: "
                "CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-cache-dir"
            )

    def _check_stage4(self, r: PrereqReport):
        if not FLAME_PATH.exists():
            r.missing.append(f"FLAME model not found at {FLAME_PATH}")
            r.manual_required.append(
                "FLAME requires manual registration:\n"
                "  1. Register at https://flame.is.tue.mpg.de/\n"
                "  2. Download FLAME 2020 model\n"
                f"  3. Place generic_model.pkl at: {FLAME_PATH}"
            )
        self._require_package(r, "xatlas", pip="xatlas")
        self._require_package(r, "pymeshlab", pip="pymeshlab")

    def _check_stage5(self, r: PrereqReport):
        # Same FLAME dependency as Stage 4
        if not FLAME_PATH.exists():
            r.missing.append(f"FLAME model not found at {FLAME_PATH}")
            r.manual_required.append(
                "FLAME required — see Stage 4 instructions."
            )

    def _check_stage6(self, r: PrereqReport):
        # All procedural — only numpy/scipy needed
        self._require_package(r, "scipy", pip="scipy")
        self._require_package(r, "trimesh", pip="trimesh")

    def _check_stage7(self, r: PrereqReport):
        self._require_package(r, "trimesh", pip="trimesh")
        self._require_package(r, "scipy", pip="scipy")
        # NeuralHaircut optional — just warn
        neuralhaircut_dir = Path("/NeuralHaircut")
        if not neuralhaircut_dir.exists():
            logger.warning(
                "Stage 7: NeuralHaircut not installed — will use Hair Cards or TSDF fallback. "
                "For film-quality hair: git clone https://github.com/SamsungLabs/NeuralHaircut /NeuralHaircut"
            )

    def _check_stage8(self, r: PrereqReport):
        # DiffusionLight and IC-Light are optional — warn only
        for name, path in [("DiffusionLight", "/DiffusionLight"), ("IC-Light", "/ICLight")]:
            if not Path(path).exists():
                logger.warning(
                    f"Stage 8: {name} not installed — will use simplified fallback. "
                    f"Install for full lighting: {path}"
                )
        self._require_package(r, "imageio", pip="imageio")

    def _check_stage9(self, r: PrereqReport):
        # Need at least one animation model
        animation_models = [
            Path.home() / ".cache" / "face_models" / "emote",
            Path.home() / ".cache" / "face_models" / "diffposetalk",
            Path.home() / ".cache" / "face_models" / "audio2face",
        ]
        any_present = any(p.exists() for p in animation_models)
        if not any_present:
            logger.warning(
                "Stage 9: No animation models found. Pipeline will produce a static avatar. "
                "To enable animation, download EMOTE or DiffPoseTalk from HuggingFace."
            )

    def _check_stage10(self, r: PrereqReport):
        self._require_package(r, "pyiqa", pip="pyiqa")
        self._require_package(r, "insightface", pip="insightface")
        self._require_package(r, "skimage", pip="scikit-image")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_cuda(self, r: PrereqReport):
        if not torch.cuda.is_available():
            r.missing.append("CUDA not available")
            r.manual_required.append(
                "CUDA required. Install PyTorch with CUDA: "
                "conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia"
            )

    def _require_nvcc(self, r: PrereqReport):
        if shutil.which("nvcc") is None:
            r.missing.append("nvcc (CUDA compiler)")
            r.manual_required.append(
                "CUDA compiler required for CUDA extensions. "
                "Install: conda install -c nvidia cuda-nvcc"
            )

    def _require_package(self, r: PrereqReport, import_name: str, pip: str):
        try:
            __import__(import_name)
        except ImportError:
            r.missing.append(import_name)
            fixed = self._try_auto_fix(r, import_name, f"pip install {pip}")
            if not fixed:
                r.manual_required.append(f"Install manually: pip install {pip}")

    def _try_auto_fix(self, r: PrereqReport, name: str, cmd: str) -> bool:
        logger.info(f"Auto-installing {name}: {cmd}")
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            r.auto_fixed.append(name)
            r.missing = [m for m in r.missing if m != name]
            return True
        logger.warning(f"Auto-install failed for {name}: {result.stderr[-200:]}")
        return False

    def _try_download_qwen(self, r: PrereqReport):
        logger.info(f"Downloading Qwen GGUF (~4 GB) to {QWEN_PATH}...")
        QWEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
                filename="qwen-image-edit-2511-Q4_0.gguf",
                local_dir=str(QWEN_PATH.parent),
            )
            r.auto_fixed.append("Qwen GGUF")
            r.missing = [m for m in r.missing if "Qwen" not in m]
        except Exception as e:
            logger.warning(f"Qwen GGUF auto-download failed: {e}")
            r.manual_required.append(
                f"Download manually:\n"
                f"  wget '{QWEN_URL}' -O {QWEN_PATH}"
            )
