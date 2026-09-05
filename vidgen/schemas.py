"""Request/response models and job-parameter resolution."""

from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import models as registry

JobType = Literal["video", "image", "inpaint"]


class JobCreate(BaseModel):
    type: JobType
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str | None = Field(default=None, max_length=4000)

    model: str | None = None
    quant: str | None = None
    preset: str | None = None

    width: int | None = Field(default=None, ge=128, le=2048)
    height: int | None = Field(default=None, ge=128, le=2048)

    # video
    num_frames: int | None = Field(default=None, ge=9, le=257)
    duration_seconds: float | None = Field(default=None, gt=0, le=20)
    fps: int | None = Field(default=None, ge=8, le=60)
    cfg_scale: float | None = Field(default=None, ge=1.0, le=15.0)

    # image / inpaint
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=60.0)

    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    # upload ids returned by POST /uploads
    source_image: str | None = None
    mask_image: str | None = None

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _resolve(self) -> "JobCreate":
        job_type = self.type
        self.model = self.model or registry.DEFAULT_MODEL_FOR_TYPE[job_type]
        try:
            spec = registry.get_model(self.model)
        except KeyError as exc:
            raise ValueError(str(exc)) from None
        if job_type not in spec.job_types:
            raise ValueError(
                f"model '{spec.key}' does not handle '{job_type}' jobs "
                f"(it handles: {', '.join(spec.job_types)})"
            )

        self.quant = self.quant or spec.default_quant
        if self.quant not in spec.footprints:
            raise ValueError(
                f"'{self.quant}' is not a valid quantization for {spec.label} "
                f"(valid: {', '.join(spec.footprints)})"
            )

        self.preset = self.preset or spec.default_preset
        if spec.presets and self.preset not in spec.presets:
            raise ValueError(
                f"'{self.preset}' is not a valid preset for {spec.label} "
                f"(valid: {', '.join(spec.presets)})"
            )

        if job_type == "inpaint":
            if not self.source_image:
                raise ValueError("inpaint jobs require a source_image upload id")
            if not self.mask_image:
                raise ValueError("inpaint jobs require a mask_image upload id")
            self.guidance = registry.DEFAULT_FILL_GUIDANCE if self.guidance is None else self.guidance

        if job_type == "video":
            self.width = self.width or 768
            self.height = self.height or 512
            # LTX-2 and Wan both want dimensions divisible by 64.
            self.width -= self.width % 64
            self.height -= self.height % 64
            self.fps = self.fps or 24
            if self.num_frames is None:
                if self.duration_seconds:
                    self.num_frames = int(round(self.duration_seconds * self.fps))
                else:
                    self.num_frames = 97
            # LTX latent packing wants 8n+1 frames.
            self.num_frames = max(9, ((self.num_frames - 1) // 8) * 8 + 1)
            self.duration_seconds = round(self.num_frames / self.fps, 2)
        else:
            self.width = self.width or 1024
            self.height = self.height or 1024
            self.width -= self.width % 16
            self.height -= self.height % 16
            self.num_frames = 1

        preset = spec.presets.get(self.preset) if self.preset else None
        if self.steps is None and preset and preset.default_steps:
            self.steps = preset.default_steps
        if job_type == "image" and self.guidance is None:
            self.guidance = registry.DEFAULT_DEV_GUIDANCE
        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)
        return self

    def to_params(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "guidance": self.guidance,
            "seed": self.seed,
            "source_image": self.source_image,
            "mask_image": self.mask_image,
        }

    def estimated_gb(self) -> float:
        return registry.estimate_peak_gb(
            self.model,
            self.quant,
            self.preset,
            job_type=self.type,
            width=self.width or 1024,
            height=self.height or 1024,
            frames=self.num_frames or 1,
        )


class JobOut(BaseModel):
    id: str
    type: str
    status: str
    model: str
    quant: str | None = None
    preset: str | None = None
    prompt: str | None = None
    params: dict = {}
    estimated_gb: float | None = None
    peak_gb: float | None = None
    progress: float = 0
    message: str | None = None
    error: str | None = None
    output_path: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def output_url(self) -> str | None:
        return f"/outputs/{self.id}" if self.output_path else None


class UploadOut(BaseModel):
    id: str
    filename: str
    width: int | None = None
    height: int | None = None
    url: str
