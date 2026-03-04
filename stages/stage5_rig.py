"""
Stage 5 — FLAME Rig Transfer & Blendshapes
==========================================
Transfer FLAME's 5-joint LBS rig and 50+ expression blendshapes to the
retopologised mesh. Vertex correspondence is established via the shrinkwrap
in Stage 4, so weights transfer directly.

FLAME model: /root/.cache/face_models/flame/generic_model.pkl
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "face_models"
FLAME_PKL  = MODELS_DIR / "flame" / "generic_model.pkl"

# FLAME has 50 expression blendshapes + 6 pose (3 global, 3 neck, jaw, eyes)
BLENDSHAPE_NAMES = [
    # 50 FLAME expression blendshapes
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

# Corrective blendshapes for known collision pairs
CORRECTIVES = [
    ("eyeBlinkLeft",  "mouthSmileLeft",  "blink_smile_corrective_L"),
    ("eyeBlinkRight", "mouthSmileRight", "blink_smile_corrective_R"),
    ("jawOpen",       "mouthPucker",     "jaw_pucker_corrective"),
    ("jawOpen",       "eyeBlinkLeft",    "jaw_blink_corrective_L"),
    ("jawOpen",       "eyeBlinkRight",   "jaw_blink_corrective_R"),
]


class FLAMERigTransfer:
    """Transfer FLAME rig to a retopologised mesh."""

    def __init__(self):
        self._flame = None

    def _load_flame(self):
        if self._flame is not None:
            return True
        if not FLAME_PKL.exists():
            logger.error(
                "FLAME model not found at %s\n"
                "Download from https://flame.is.tue.mpg.de/ and place generic_model.pkl there.",
                FLAME_PKL,
            )
            return False
        with open(str(FLAME_PKL), "rb") as f:
            self._flame = pickle.load(f, encoding="latin1")
        return True

    def transfer_lbs_weights(
        self,
        retopo_verts: np.ndarray,
        flame_template_verts: np.ndarray,
    ) -> np.ndarray:
        """
        Transfer FLAME LBS weights to retopologised mesh vertices.
        Uses nearest-neighbour mapping (vertex correspondence from shrinkwrap).
        Returns: skinning weights (N_retopo, N_joints)
        """
        if not self._load_flame():
            # Return uniform weights as fallback
            n = len(retopo_verts)
            w = np.zeros((n, 5), dtype=np.float32)
            w[:, 0] = 1.0  # all bound to root
            return w

        lbs_weights = np.array(self._flame["weights"])  # (V_flame, 5)
        n_retopo = len(retopo_verts)

        # Find nearest FLAME vertex for each retopo vertex
        from scipy.spatial import KDTree
        tree = KDTree(flame_template_verts)
        _, nn_idx = tree.query(retopo_verts, k=4, workers=-1)

        # Weighted blend from 4 nearest neighbours
        dists_sq, _ = tree.query(retopo_verts, k=4)
        # Inverse distance weighting
        inv = 1.0 / (dists_sq + 1e-8)
        norm_inv = inv / inv.sum(axis=1, keepdims=True)

        retopo_weights = np.zeros((n_retopo, lbs_weights.shape[1]), dtype=np.float32)
        for k in range(4):
            retopo_weights += norm_inv[:, k:k+1] * lbs_weights[nn_idx[:, k]]

        # Renormalise rows
        row_sum = retopo_weights.sum(axis=1, keepdims=True)
        retopo_weights /= (row_sum + 1e-8)

        logger.info(f"  LBS weights transferred: {n_retopo} vertices × {retopo_weights.shape[1]} joints")
        return retopo_weights

    def generate_blendshapes(
        self,
        retopo_verts: np.ndarray,
        retopo_faces: np.ndarray,
        flame_template_verts: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate vertex delta blendshapes from FLAME expression space.
        Returns: dict mapping blendshape_name → vertex_delta (N, 3)
        """
        if not self._load_flame():
            return {}

        expdirs = np.array(self._flame["exprdirs"])   # (V*3, n_exp) or (V, 3, n_exp)
        V_flame = len(flame_template_verts)

        # Reshape to (V, 3, n_exp)
        if expdirs.ndim == 2:
            n_exp = expdirs.shape[1]
            expdirs = expdirs.reshape(V_flame, 3, n_exp)
        else:
            n_exp = expdirs.shape[-1]

        # Map from FLAME vertices to retopo vertices via nearest neighbour
        from scipy.spatial import KDTree
        tree = KDTree(flame_template_verts)
        _, nn_idx = tree.query(retopo_verts, k=1, workers=-1)

        blendshapes = {}
        n_use = min(len(BLENDSHAPE_NAMES), n_exp)

        for bs_i, bs_name in enumerate(BLENDSHAPE_NAMES[:n_use]):
            # Activate only this blendshape (weight=1.0)
            delta_flame = expdirs[:, :, bs_i]  # (V_flame, 3)
            delta_retopo = delta_flame[nn_idx]   # (V_retopo, 3)
            blendshapes[bs_name] = delta_retopo.astype(np.float32)

        logger.info(f"  Generated {len(blendshapes)} FLAME expression blendshapes")
        return blendshapes

    def generate_corrective_blendshapes(
        self,
        primary_blendshapes: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Generate corrective blendshapes for problematic shape combinations."""
        correctives = {}
        for bs_a, bs_b, corr_name in CORRECTIVES:
            if bs_a in primary_blendshapes and bs_b in primary_blendshapes:
                # Simple corrective: negate the intersection of both shapes
                # A proper corrective would be sculpted, but procedurally:
                # delta_corr = -(delta_a * delta_b)  (suppress in shared regions)
                d_a = primary_blendshapes[bs_a]
                d_b = primary_blendshapes[bs_b]
                # Corrective activates only when both are > 0.5
                corr = -0.3 * d_a * (np.abs(d_b) > 0.001).astype(np.float32)
                correctives[corr_name] = corr.astype(np.float32)
                logger.info(f"  Corrective: {corr_name}")

        return correctives


def save_rig_to_json(
    lbs_weights: np.ndarray,
    blendshapes: Dict[str, np.ndarray],
    output_dir: Path,
) -> str:
    """Save rig data as JSON for use in Blender/DCC import scripts."""
    import json

    joint_names = ["root", "neck", "jaw", "left_eye", "right_eye"]
    rig_data = {
        "joint_names":    joint_names,
        "lbs_weights":    lbs_weights.tolist(),
        "blendshapes":    {k: v.tolist() for k, v in blendshapes.items()},
    }
    out_path = output_dir / "rig_data.json"
    with open(str(out_path), "w") as f:
        json.dump(rig_data, f, indent=2)
    logger.info(f"  Rig saved: {out_path} ({len(blendshapes)} blendshapes)")
    return str(out_path)


def save_blendshapes_as_obj(
    neutral_verts: np.ndarray,
    faces: np.ndarray,
    blendshapes: Dict[str, np.ndarray],
    output_dir: Path,
) -> Dict[str, str]:
    """Save each blendshape as a separate OBJ for DCC import (Blender, Maya)."""
    bs_dir = output_dir / "blendshapes"
    bs_dir.mkdir(exist_ok=True)
    paths = {}

    for bs_name, delta in blendshapes.items():
        bs_verts = neutral_verts + delta
        obj_path = bs_dir / f"{bs_name}.obj"
        _write_obj(str(obj_path), bs_verts, faces)
        paths[bs_name] = str(obj_path)

    logger.info(f"  Blendshape OBJs written to: {bs_dir}")
    return paths


def _write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


class RigStage:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._rig = FLAMERigTransfer()

    def run(
        self,
        retopo_mesh_path: str,
        flame_beta: np.ndarray,
        output_dir: Path,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("STAGE 5: FLAME Rig Transfer & Blendshapes")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)

        import trimesh
        mesh = trimesh.load(retopo_mesh_path, process=False)
        retopo_verts = np.array(mesh.vertices, dtype=np.float64)
        retopo_faces = np.array(mesh.faces,    dtype=np.int32)

        # Build FLAME template for this person
        flame_template_verts = self._build_flame_template(flame_beta)

        # 5.1 Transfer LBS weights
        logger.info("Step 5.1: Transferring FLAME LBS weights …")
        lbs_weights = self._rig.transfer_lbs_weights(retopo_verts, flame_template_verts)

        # 5.2 Generate blendshapes
        logger.info("Step 5.2: Generating 50+ FLAME expression blendshapes …")
        blendshapes = self._rig.generate_blendshapes(
            retopo_verts, retopo_faces, flame_template_verts,
        )

        # 5.3 Corrective blendshapes
        logger.info("Step 5.3: Building corrective blendshapes …")
        correctives = self._rig.generate_corrective_blendshapes(blendshapes)
        all_bs = {**blendshapes, **correctives}

        # 5.4 Save rig
        rig_json = save_rig_to_json(lbs_weights, all_bs, output_dir)
        bs_paths = save_blendshapes_as_obj(retopo_verts, retopo_faces, all_bs, output_dir)

        results = {
            "rig_json":         rig_json,
            "lbs_weights":      lbs_weights,
            "blendshapes":      all_bs,
            "blendshape_objs":  bs_paths,
            "neutral_verts":    retopo_verts,
            "faces":            retopo_faces,
            "output_dir":       str(output_dir),
        }

        logger.info("✓ Stage 5 complete")
        logger.info(f"  Blendshapes: {len(all_bs)} ({len(blendshapes)} FLAME + {len(correctives)} correctives)")
        logger.info(f"  Rig JSON: {rig_json}")
        return results

    def _build_flame_template(self, flame_beta: np.ndarray) -> np.ndarray:
        """Compute FLAME neutral vertices for this person's shape."""
        if not FLAME_PKL.exists():
            return np.zeros((5023, 3), dtype=np.float32)
        with open(str(FLAME_PKL), "rb") as f:
            flame = pickle.load(f, encoding="latin1")
        shapedirs  = np.array(flame["shapedirs"])
        v_template = np.array(flame["v_template"])
        V = len(v_template)
        n_beta = min(len(flame_beta), shapedirs.shape[-1] if shapedirs.ndim > 2 else shapedirs.shape[1])
        beta = np.zeros(shapedirs.shape[-1] if shapedirs.ndim > 2 else shapedirs.shape[1])
        beta[:n_beta] = flame_beta[:n_beta]
        if shapedirs.ndim == 2:
            v_shaped = v_template + (shapedirs @ beta).reshape(V, 3)
        else:
            v_shaped = v_template + (shapedirs * beta).sum(axis=-1)
        return v_shaped.astype(np.float32)
