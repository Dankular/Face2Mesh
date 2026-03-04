# -*- coding: utf-8 -*-
"""
Face2Mesh Pipeline Orchestrator

Wraps pipeline_complete.py with pre/post hooks:
  - PrereqAgent  : checks requirements BEFORE each stage
  - OutputValidator: validates output is NOT a placeholder AFTER each stage

Never continues past a failed check. Every failure prints exactly what to do next.

Usage:
    python agents/orchestrator.py --input face.jpg --output ./output
    python agents/orchestrator.py --input face.jpg --dry-run
    python agents/orchestrator.py --input face.jpg --resume-from 4
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from /Face2Mesh root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.prereq_agent import PrereqAgent
from agents.output_validator import OutputValidator
from pipeline_complete import CompleteFaceTo3DPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

STAGE_NAMES = {
    1:  "Identity Extraction",
    2:  "3D Geometry (FaceLift + TSDF)",
    3:  "Texture Views (Qwen 24-view)",
    4:  "Mesh + PBR Materials",
    5:  "Rig & Blendshapes",
    6:  "Eyes / Teeth / Tongue",
    7:  "Hair Reconstruction",
    8:  "Lighting & Materials",
    9:  "Animation",
    10: "Validation & Export",
}


def _banner(stage: int, phase: str):
    name = STAGE_NAMES.get(stage, f"Stage {stage}")
    bar = "=" * 60
    logger.info(f"\n{bar}\n  STAGE {stage} — {name} [{phase}]\n{bar}")


def run_pipeline(
    image_path: str,
    output_dir: str = "./output",
    facelift_dir: str = "/FaceLift",
    resume_from: int = 1,
    dry_run: bool = False,
    audio_path: str = None,
    video_path: str = None,
):
    prereq = PrereqAgent()
    validator = OutputValidator()

    # ── Dry-run: check all prereqs, exit before any inference ─────────────────
    if dry_run:
        logger.info("DRY RUN — checking all prerequisites (no inference)")
        all_ok = True
        for stage in range(1, 11):
            _banner(stage, "PREREQ CHECK")
            report = prereq.check(stage)
            if not report.passed:
                all_ok = False
        if all_ok:
            logger.info("\n=== DRY RUN PASSED — all prerequisites satisfied ===")
        else:
            logger.error("\n=== DRY RUN FAILED — fix issues above before running ===")
            sys.exit(1)
        return

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = CompleteFaceTo3DPipeline(
        output_dir=output_dir,
        facelift_dir=facelift_dir,
        audio_path=audio_path,
        video_path=video_path,
    )

    stage_results = {}
    pipeline_start = time.time()

    # Stage runner map — each returns the output dict for that stage
    def run_stage(n: int):
        r = stage_results
        if n == 1:
            return pipeline.stage1(image_path)
        elif n == 2:
            return pipeline.stage2(image_path)
        elif n == 3:
            r3 = pipeline.stage3_mesh(r[2])
            stage_results[3] = r3
            return pipeline.stage3b_qwen_views(image_path, r[1], r[2])
        elif n == 4:
            return pipeline.stage4(stage_results[3], stage_results.get("3b", {}), r[1], image_path)
        elif n == 5:
            return pipeline.stage5(r[4], r[1])
        elif n == 6:
            return pipeline.stage6(stage_results[3], r[1])
        elif n == 7:
            return pipeline.stage7(image_path, stage_results[3], r[1])
        elif n == 8:
            return pipeline.stage8(image_path, r[4])
        elif n == 9:
            return pipeline.stage9(r[5])
        elif n == 10:
            return pipeline.stage10(image_path, r[1], stage_results[3],
                                     stage_results.get("3b", {}), r[4], r[5], r.get(9, {}))

    # ── Execute stages ────────────────────────────────────────────────────────
    for stage in range(1, 11):
        if stage < resume_from:
            logger.info(f"  Skipping Stage {stage} (resume-from={resume_from})")
            continue

        # 1. Prerequisites
        _banner(stage, "PREREQ")
        prereq_report = prereq.check(stage)
        if not prereq_report.passed:
            logger.error(
                f"\n!!! Stage {stage} BLOCKED — prerequisites not met !!!\n"
                + prereq_report.summary()
                + "\n\nFix the issues above and re-run with --resume-from "
                + str(stage)
            )
            sys.exit(1)

        # 2. Run stage
        _banner(stage, "RUNNING")
        t0 = time.time()
        try:
            # Stages 3a (mesh) and 3b (qwen) handled together
            if stage == 3:
                mesh_out = pipeline.stage3_mesh(stage_results[2])
                stage_results[3] = mesh_out
                qwen_out = pipeline.stage3b_qwen_views(image_path, stage_results[1], stage_results[2])
                output = qwen_out
                output["_mesh"] = mesh_out
            else:
                output = run_stage(stage)

            stage_results[stage] = output
            logger.info(f"  Stage {stage} ran in {time.time()-t0:.1f}s")

        except Exception as e:
            logger.error(
                f"\n!!! Stage {stage} CRASHED: {e} !!!\n"
                f"Re-run with --resume-from {stage} after fixing the error."
            )
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 3. Validate output
        _banner(stage, "VALIDATING")
        val_report = validator.validate(stage, output)
        if not val_report.passed:
            logger.error(
                f"\n!!! Stage {stage} OUTPUT INVALID — placeholder/fallback detected !!!\n"
                + val_report.summary()
                + f"\n\nFix the issues above and re-run with --resume-from {stage}"
            )
            sys.exit(1)

        _banner(stage, "COMPLETE")

    elapsed = time.time() - pipeline_start
    logger.info(
        f"\n{'='*60}\n"
        f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes\n"
        f"  Output: {output_dir}\n"
        f"{'='*60}"
    )


def main():
    parser = argparse.ArgumentParser(description="Face2Mesh Orchestrated Pipeline")
    parser.add_argument("--input",        required=True,  help="Path to input face image")
    parser.add_argument("--output",       default="./output_orchestrated")
    parser.add_argument("--facelift",     default="/FaceLift")
    parser.add_argument("--resume-from",  type=int, default=1, dest="resume_from",
                        help="Resume from this stage number (skips earlier stages)")
    parser.add_argument("--dry-run",      action="store_true", dest="dry_run",
                        help="Check all prereqs without running inference")
    parser.add_argument("--audio",        default=None, help="Audio file for Stage 9 animation")
    parser.add_argument("--video",        default=None, help="Video file for Stage 9 animation")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.input,
        output_dir=args.output,
        facelift_dir=args.facelift,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
        audio_path=args.audio,
        video_path=args.video,
    )


if __name__ == "__main__":
    main()
