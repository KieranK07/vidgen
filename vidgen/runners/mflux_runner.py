"""mflux runners: FLUX.1 [dev] text-to-image and FLUX.1-Fill-dev inpaint/outpaint.

    mflux-generate       --model dev -q 8 --prompt ... --steps 25 --guidance 3.5 --output out.png
    mflux-generate-fill  -q 8 --image-path src.png --masked-image-path mask.png \
                         --prompt ... --steps 25 --guidance 30 --output out.png

Mask convention (mflux / FLUX.1-Fill): white = regenerate, black = keep. The
dashboard's brush paints white, and outpainting is the same operation with the
source image padded onto a larger white-margin canvas.
"""

from __future__ import annotations

import os

from ..models import DEFAULT_DEV_GUIDANCE, DEFAULT_FILL_GUIDANCE
from .base import JobContext, RunPlan, Runner
from .mlx_video_runner import _model_cache_env


class MfluxGenerateRunner(Runner):
    key = "mflux_generate"
    tool = "mflux (FLUX.1 dev)"

    def build(self, ctx: JobContext) -> RunPlan:
        cfg = ctx.cfg
        out = ctx.output_dir / "output.png"
        binary = self.require_tool(cfg.mflux_generate_bin)
        preset = ctx.spec.presets.get(ctx.preset or "quality")
        steps = ctx.params.get("steps") or (preset.default_steps if preset else 25)
        guidance = ctx.params.get("guidance")
        guidance = DEFAULT_DEV_GUIDANCE if guidance is None else guidance

        argv = [
            binary,
            "--model",
            os.environ.get("VIDGEN_FLUX_MODEL", "dev"),
            "-q",
            str(self.quant_bits(ctx.quant)),
            "--prompt",
            str(ctx.param("prompt", "")),
            "--width",
            str(ctx.param("width", 1024)),
            "--height",
            str(ctx.param("height", 1024)),
            "--steps",
            str(steps),
            "--guidance",
            str(guidance),
            "--seed",
            str(ctx.param("seed", 0)),
            "--output",
            str(out),
        ]

        local_path = os.environ.get("VIDGEN_FLUX_DEV_PATH")
        if local_path:
            argv += ["--path", local_path]

        argv += self.extra_args("VIDGEN_MFLUX_EXTRA_ARGS")

        return RunPlan(
            argv=argv,
            output_file=out,
            env=_model_cache_env(ctx),
            describe=f"FLUX.1 [dev] {ctx.quant} / {steps} steps",
        )


class MfluxFillRunner(Runner):
    key = "mflux_fill"
    tool = "mflux (FLUX.1-Fill-dev)"

    def build(self, ctx: JobContext) -> RunPlan:
        cfg = ctx.cfg
        out = ctx.output_dir / "output.png"
        binary = self.require_tool(cfg.mflux_fill_bin)
        preset = ctx.spec.presets.get(ctx.preset or "quality")
        steps = ctx.params.get("steps") or (preset.default_steps if preset else 25)
        guidance = ctx.params.get("guidance")
        guidance = DEFAULT_FILL_GUIDANCE if guidance is None else guidance

        source = ctx.params.get("source_image")
        mask = ctx.params.get("mask_image")
        if not source or not mask:
            raise ValueError("inpaint jobs need both source_image and mask_image")
        source_path = ctx.upload_path(source)
        mask_path = ctx.upload_path(mask)

        argv = [
            binary,
            "-q",
            str(self.quant_bits(ctx.quant)),
            "--image-path",
            str(source_path),
            "--masked-image-path",
            str(mask_path),
            "--prompt",
            str(ctx.param("prompt", "")),
            "--height",
            str(ctx.param("height", 1024)),
            "--width",
            str(ctx.param("width", 1024)),
            "--steps",
            str(steps),
            "--guidance",
            str(guidance),
            "--seed",
            str(ctx.param("seed", 0)),
            "--output",
            str(out),
        ]

        local_path = os.environ.get("VIDGEN_FLUX_FILL_PATH")
        if local_path:
            argv += ["--path", local_path]

        argv += self.extra_args("VIDGEN_MFLUX_FILL_EXTRA_ARGS")

        return RunPlan(
            argv=argv,
            output_file=out,
            env=_model_cache_env(ctx),
            extra_outputs=[source_path, mask_path],
            describe=f"FLUX.1-Fill-dev {ctx.quant} / {steps} steps / guidance {guidance}",
        )
