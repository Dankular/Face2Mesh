# -*- coding: utf-8 -*-
"""
OutputValidator — runs AFTER each stage.
Detects placeholder/fallback outputs and blocks pipeline progression.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    passed: bool
    stage: int
    checks: List[CheckResult] = field(default_factory=list)
    remediation: str = ""

    def summary(self) -> str:
        lines = [f"=== Stage {self.stage} Output {'PASS' if self.passed else 'FAIL'} ==="]
        for c in self.checks:
            icon = "OK" if c.passed else "FAIL"
            lines.append(f"  [{icon}] {c.name}: {c.detail}")
        if not self.passed and self.remediation:
            lines.append(f"\n  Remediation: {self.remediation}")
        return "\n".join(lines)


class OutputValidator:
    """Validate stage outputs are not placeholders or fallback values."""

    def validate(self, stage: int, output: Dict[str, Any]) -> ValidationReport:
        validators = {
            1: self._validate_stage1,
            2: self._validate_stage2,
            3: self._validate_stage3,
            4: self._validate_stage4,
            5: self._validate_stage5,
            6: self._validate_stage6,
            7: self._validate_stage7,
            8: self._validate_stage8,
            9: self._validate_stage9,
            10: self._validate_stage10,
        }
        report = ValidationReport(passed=True, stage=stage)
        validators.get(stage, lambda o, r: None)(output, report)
        report.passed = all(c.passed for c in report.checks)
        logger.info(report.summary())
        return report

    # ── Stage validators ─────────────────────────────────────────────────────

    def _validate_stage1(self, out: dict, r: ValidationReport):
        # ArcFace embedding: non-zero, normalised (L2 norm ~1.0)
        emb = out.get("arcface_embedding")
        if emb is None:
            r.checks.append(CheckResult("arcface_embedding", False, "key missing from output"))
        else:
            emb = np.array(emb)
            norm = float(np.linalg.norm(emb))
            is_zeros = np.all(emb == 0)
            is_uniform = float(np.std(emb)) < 0.001
            ok = not is_zeros and not is_uniform and 0.5 < norm < 2.0
            r.checks.append(CheckResult(
                "arcface_embedding",
                ok,
                f"norm={norm:.3f}, std={np.std(emb):.4f}" + (" [ZEROS — ArcFace fallback]" if is_zeros else "")
            ))

        # FLAME shape beta: at least 10 non-zero values
        beta = out.get("flame_shape_beta")
        if beta is not None:
            beta = np.array(beta)
            nonzero = int(np.sum(np.abs(beta) > 1e-4))
            ok = nonzero >= 10
            r.checks.append(CheckResult(
                "flame_shape_beta",
                ok,
                f"{nonzero}/300 non-zero values" + (" [ZEROS — MICA fallback]" if nonzero == 0 else "")
            ))

        if not r.checks[-1].passed or (len(r.checks) > 1 and not r.checks[-2].passed):
            r.remediation = (
                "Identity models missing. Ensure ArcFace, MICA, SMIRK, BiSeNet are installed. "
                "Run: python agents/orchestrator.py --dry-run to diagnose."
            )

    def _validate_stage2(self, out: dict, r: ValidationReport):
        mesh_path = out.get("mesh_ply") or out.get("mesh_path")

        if not mesh_path or not Path(str(mesh_path)).exists():
            r.checks.append(CheckResult("mesh_file", False, "no mesh file produced"))
            r.remediation = "FaceLift failed to produce a mesh. Check /FaceLift is cloned and checkpoint downloaded."
            return

        try:
            import trimesh
            mesh = trimesh.load(str(mesh_path), process=False)
            n_verts = len(mesh.vertices)
            n_faces = len(mesh.faces)

            # Real face mesh: 50k-300k verts, non-spherical bounds
            bounds = mesh.bounds
            extents = bounds[1] - bounds[0]
            aspect = max(extents) / (min(extents) + 1e-6)

            is_placeholder = n_verts < 5000 or aspect > 10.0
            r.checks.append(CheckResult(
                "mesh_geometry",
                not is_placeholder,
                f"{n_verts:,} verts, {n_faces:,} faces, extents={extents.round(3)}"
                + (" [PLACEHOLDER — too few vertices]" if n_verts < 5000 else "")
            ))

            # Check TSDF was used (not Poisson fallback)
            tsdf_flag = out.get("used_tsdf", None)
            if tsdf_flag is not None:
                r.checks.append(CheckResult(
                    "tsdf_fusion",
                    tsdf_flag,
                    "TSDF fusion used" if tsdf_flag else "[FALLBACK] Poisson reconstruction — install open3d"
                ))
        except Exception as e:
            r.checks.append(CheckResult("mesh_geometry", False, f"could not load mesh: {e}"))

        if any(not c.passed for c in r.checks):
            r.remediation = "3D reconstruction produced a placeholder. Ensure FaceLift+Open3D are correctly installed."

    def _validate_stage3(self, out: dict, r: ValidationReport):
        views = out.get("views", {})
        n_views = len(views) if isinstance(views, dict) else 0

        r.checks.append(CheckResult(
            "view_count",
            n_views >= 24,
            f"{n_views}/24 views generated" + (" [INCOMPLETE]" if n_views < 24 else "")
        ))

        csim = out.get("mean_csim", 0.0)
        r.checks.append(CheckResult(
            "identity_csim",
            csim >= 0.5,
            f"mean CSIM={csim:.3f}" + (" [TOO LOW — identity drift]" if csim < 0.5 else "")
        ))

        if any(not c.passed for c in r.checks):
            r.remediation = (
                f"Qwen multi-view generation incomplete (got {n_views}/24 views, CSIM={csim:.3f}). "
                "Verify Qwen GGUF is downloaded and llama-cpp-python has CUDA support."
            )

    def _validate_stage4(self, out: dict, r: ValidationReport):
        # Check FLAME retopo was used, not xatlas fallback
        used_flame = out.get("used_flame_retopo", False)
        r.checks.append(CheckResult(
            "flame_retopology",
            used_flame,
            "FLAME retopology applied" if used_flame
            else "[FALLBACK] xatlas generic retopology — register FLAME at flame.is.tue.mpg.de"
        ))

        # Check albedo texture is not flat/placeholder
        albedo_path = out.get("albedo_texture") or (out.get("pbr_maps") or {}).get("albedo")
        if albedo_path and Path(str(albedo_path)).exists():
            try:
                from PIL import Image
                img = np.array(Image.open(str(albedo_path)).convert("RGB"), dtype=float)
                std = float(np.std(img))
                r.checks.append(CheckResult(
                    "albedo_texture",
                    std > 15.0,
                    f"pixel std={std:.1f}" + (" [PLACEHOLDER — flat color]" if std <= 15.0 else "")
                ))
            except Exception as e:
                r.checks.append(CheckResult("albedo_texture", False, f"could not read texture: {e}"))
        else:
            r.checks.append(CheckResult("albedo_texture", False, "albedo texture file missing"))

        if any(not c.passed for c in r.checks):
            r.remediation = (
                "Stage 4 produced fallback output. "
                "If FLAME missing: register at flame.is.tue.mpg.de. "
                "If textures flat: check Qwen views from Stage 3 passed validation."
            )

    def _validate_stage5(self, out: dict, r: ValidationReport):
        blendshapes = out.get("blendshapes") or out.get("blendshape_deltas")
        if blendshapes is None:
            r.checks.append(CheckResult("blendshapes", False, "no blendshapes in output"))
            r.remediation = "Rigging failed — FLAME model required."
            return

        if isinstance(blendshapes, dict):
            n_bs = len(blendshapes)
            nonzero = sum(1 for v in blendshapes.values() if np.any(np.abs(np.array(v)) > 1e-4))
        else:
            bs_arr = np.array(blendshapes)
            n_bs = bs_arr.shape[0] if bs_arr.ndim > 1 else 1
            nonzero = int(np.sum(np.abs(bs_arr) > 1e-4))

        ok = n_bs >= 30 and nonzero >= 20
        r.checks.append(CheckResult(
            "blendshapes",
            ok,
            f"{n_bs} shapes, {nonzero} non-trivial"
            + (" [ZEROS — FLAME fallback]" if nonzero == 0 else "")
        ))
        if not ok:
            r.remediation = "Blendshapes are zeros or too few. FLAME model registration required."

    def _validate_stage6(self, out: dict, r: ValidationReport):
        required = ["eye_left", "eye_right", "teeth", "tongue", "eyelashes", "eyebrows"]
        for key in required:
            present = key in out and out[key] is not None
            r.checks.append(CheckResult(key, present, "present" if present else "MISSING"))
        if any(not c.passed for c in r.checks):
            r.remediation = "Detail geometry incomplete — Stage 6 is fully procedural, check for Python errors."

    def _validate_stage7(self, out: dict, r: ValidationReport):
        hair_path = out.get("hair_mesh")
        if not hair_path or not Path(str(hair_path)).exists():
            r.checks.append(CheckResult("hair_mesh", False, "no hair mesh file"))
            r.remediation = "Hair stage failed. Check trimesh/scipy are installed."
            return

        try:
            import trimesh
            mesh = trimesh.load(str(hair_path), process=False)
            n_verts = len(mesh.vertices)
            ok = n_verts >= 500
            method = out.get("hair_method", "unknown")
            r.checks.append(CheckResult(
                "hair_mesh",
                ok,
                f"{n_verts:,} verts via {method}"
                + (" [TOO SPARSE]" if n_verts < 500 else "")
            ))
        except Exception as e:
            r.checks.append(CheckResult("hair_mesh", False, f"could not load: {e}"))

    def _validate_stage8(self, out: dict, r: ValidationReport):
        # EXR output signals real ACEScc conversion happened
        exr_out = out.get("exr_output") or out.get("acescc_albedo")
        r.checks.append(CheckResult(
            "acescc_conversion",
            bool(exr_out and Path(str(exr_out)).exists()),
            "EXR/ACEScc output present" if exr_out else "no EXR — PNG fallback only (acceptable)"
        ))
        # Stage 8 failures are warnings, not blockers
        for c in r.checks:
            if not c.passed:
                logger.warning(f"Stage 8 non-fatal: {c.name} — {c.detail}")
                c.passed = True  # Don't block pipeline for optional lighting

    def _validate_stage9(self, out: dict, r: ValidationReport):
        params = out.get("params_npy") or out.get("animation_params")
        if params is None:
            # Static avatar is acceptable if no audio/video provided
            r.checks.append(CheckResult(
                "animation_params",
                True,
                "no animation input provided — static avatar (acceptable)"
            ))
            return

        try:
            arr = np.load(str(params)) if isinstance(params, (str, Path)) else np.array(params)
            variance = float(np.var(arr))
            ok = variance > 1e-6
            r.checks.append(CheckResult(
                "animation_params",
                ok,
                f"variance={variance:.6f}" + (" [STATIC — all frames identical]" if not ok else "")
            ))
        except Exception as e:
            r.checks.append(CheckResult("animation_params", False, f"could not read params: {e}"))

        if any(not c.passed for c in r.checks):
            r.remediation = "Animation params are static. Ensure audio/video input was provided and animation model is installed."

    def _validate_stage10(self, out: dict, r: ValidationReport):
        overall_pass = out.get("overall_pass", False)
        csim = out.get("identity_csim", 0.0)
        exports = out.get("exports", {})

        r.checks.append(CheckResult(
            "identity_csim",
            csim >= 0.5,
            f"CSIM={csim:.3f}" + (" [BELOW THRESHOLD]" if csim < 0.5 else "")
        ))

        glb = exports.get("glb")
        r.checks.append(CheckResult(
            "glb_export",
            bool(glb and Path(str(glb)).exists()),
            str(glb) if glb else "GLB export missing"
        ))

        r.checks.append(CheckResult(
            "overall_pass",
            overall_pass,
            "pipeline quality check passed" if overall_pass else "quality checks failed — review validation report"
        ))

        if any(not c.passed for c in r.checks):
            r.remediation = "Final validation failed. Check identity CSIM and exported files."
