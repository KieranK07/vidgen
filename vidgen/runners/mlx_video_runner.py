"""mlx-video runners: LTX-2.3 (default) and Wan2.2 14B (alternate).

Invocation follows the upstream CLI:

    python -m mlx_video.ltx_2.generate --pipeline dev-two-stage-hq --prompt ... -o out.mp4
    python -m mlx_video.wan_2.generate --model-dir <dir> --prompt ... --output-path out.mp4

Upstream flags change between releases. Rather than pin to one shape, every
runner accepts a VIDGEN_*_EXTRA_ARGS escape hatch and reads its module path and
quantization flag from config, so a rename upstream is a .env edit, not a patch.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import JobContext, RunPlan, Runner


class LTXRunner(Runner):
    key = "mlx_video_ltx"
    tool = "mlx-video (LTX-2)"

    def build(self, ctx: JobContext) -> RunPlan:
        cfg = ctx.cfg
        out = ctx.output_dir / "output.mp4"
        preset = ctx.spec.presets.get(ctx.preset or "quality")
        pipeline = (preset.pipeline if preset else None) or "dev-two-stage"

        argv = [
            self.python_exe(cfg),
            "-m",
            f"{cfg.mlx_video_module}.ltx_2.generate",
            "--prompt",
            str(ctx.param("prompt", "")),
            "--pipeline",
            pipeline,
            "--width",
            str(ctx.param("width", 768)),
            "--height",
            str(ctx.param("height", 512)),
            "--num-frames",
            str(ctx.param("num_frames", 97)),
            "--fps",
            str(ctx.param("fps", 24)),
            "--seed",
            str(ctx.param("seed", 0)),
            "--output",
            str(out),
        ]

        repo = os.environ.get("VIDGEN_LTX_MODEL_REPO") or ctx.spec.repo
        if repo:
            argv += ["--model-repo", repo]

        # Quantization is mandatory for a 22B model on 32GB — never omit it.
        quant_flag = os.environ.get("VIDGEN_MLX_VIDEO_QUANT_FLAG", "--quantize")
        argv += [quant_flag, str(self.quant_bits(ctx.quant))]

        cfg_scale = ctx.params.get("cfg_scale")
        if cfg_scale is None and pipeline != "distilled":
            cfg_scale = 3.0
        if cfg_scale is not None:
            argv += ["--cfg-scale", str(cfg_scale)]

        negative = ctx.params.get("negative_prompt")
        if negative:
            argv += ["--negative-prompt", str(negative)]

        source = ctx.params.get("source_image")
        if source:
            argv += ["--image", str(ctx.upload_path(source))]

        argv += self.extra_args("VIDGEN_LTX_EXTRA_ARGS")

        return RunPlan(
            argv=argv,
            output_file=out,
            env=_model_cache_env(ctx),
            describe=f"LTX-2.3 {pipeline} @ {ctx.quant}",
        )


class WanRunner(Runner):
    key = "mlx_video_wan"
    tool = "mlx-video (Wan2.2)"

    def build(self, ctx: JobContext) -> RunPlan:
        cfg = ctx.cfg
        out = ctx.output_dir / "output.mp4"
        preset = ctx.spec.presets.get(ctx.preset or "quality")
        steps = ctx.params.get("steps") or (preset.default_steps if preset else 40)

        model_dir = os.environ.get("VIDGEN_WAN_MODEL_DIR")
        if not model_dir:
            raise FileNotFoundError(
                "Wan2.2 needs a converted/quantized model directory. Set VIDGEN_WAN_MODEL_DIR "
                "to the Q5 or Q6 GGUF weights directory (see README - Model downloads)."
            )

        argv = [
            self.python_exe(cfg),
            "-m",
            f"{cfg.mlx_video_module}.wan_2.generate",
            "--model-dir",
            model_dir,
            "--prompt",
            str(ctx.param("prompt", "")),
            "--width",
            str(ctx.param("width", 768)),
            "--height",
            str(ctx.param("height", 512)),
            "--num-frames",
            str(ctx.param("num_frames", 81)),
            "--steps",
            str(steps),
            "--seed",
            str(ctx.param("seed", 0)),
            "--output-path",
            str(out),
        ]

        negative = ctx.params.get("negative_prompt") or "blurry, low quality, distorted, watermark"
        argv += ["--negative-prompt", negative]

        guide = ctx.params.get("cfg_scale")
        if guide is not None:
            argv += ["--guide-scale", str(guide)]

        source = ctx.params.get("source_image")
        if source:
            argv += ["--image", str(ctx.upload_path(source))]

        argv += self.extra_args("VIDGEN_WAN_EXTRA_ARGS")

        return RunPlan(
            argv=argv,
            output_file=out,
            env=_model_cache_env(ctx),
            describe=f"Wan2.2 14B {ctx.quant} / {steps} steps",
        )


def _model_cache_env(ctx: JobContext) -> dict[str, str]:
    cache = str(ctx.cfg.model_cache_dir)
    return {
        "HF_HOME": cache,
        "HUGGINGFACE_HUB_CACHE": str(Path(cache) / "hub"),
        # Fully local: never phone home for telemetry or update checks.
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "PYTHONUNBUFFERED": "1",
    }
