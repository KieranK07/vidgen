"""Runner used when VIDGEN_DRY_RUN=1 — see fake_worker.py."""

from __future__ import annotations

import os

from .base import JobContext, RunPlan, Runner


class DryRunRunner(Runner):
    key = "dry_run"
    tool = "dry-run stub"

    def build(self, ctx: JobContext) -> RunPlan:
        kind = "video" if ctx.job_type == "video" else "image"
        out = ctx.output_dir / ("output.mp4" if kind == "video" else "output.png")
        steps = int(ctx.param("steps", 0) or (12 if kind == "video" else 8))
        argv = [
            self.python_exe(ctx.cfg),
            "-m",
            "vidgen.runners.fake_worker",
            "--output", str(out),
            "--kind", kind,
            "--steps", str(min(steps, 40)),
            "--width", str(ctx.param("width", 512)),
            "--height", str(ctx.param("height", 512)),
            "--frames", str(ctx.param("num_frames", 25)),
            "--fps", str(ctx.param("fps", 24)),
            "--seed", str(ctx.param("seed", 0)),
            "--allocate-mb", os.environ.get("VIDGEN_DRY_RUN_ALLOC_MB", "256"),
            "--step-seconds", os.environ.get("VIDGEN_DRY_RUN_STEP_SECONDS", "0.15"),
        ]
        return RunPlan(
            argv=argv,
            output_file=out,
            env={"PYTHONUNBUFFERED": "1"},
            describe=f"dry-run stub ({ctx.spec.label} {ctx.quant})",
        )
