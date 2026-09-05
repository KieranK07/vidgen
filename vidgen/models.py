"""Model registry: what can be run, and what it costs in unified memory.

The numbers here are *estimated peak* resident footprints — weights plus text
encoder plus VAE plus working latents — measured on a 32GB M4. They exist so
the headroom guard can refuse a job before anything is loaded. Every job also
records its *actual* observed peak in meta.json and in data/observed_footprints.json,
so you can correct the table by hand from real numbers and the estimates
sharpen with use.

Overrides live in $VIDGEN_DATA_DIR/model_footprints.json:
    {"ltx2:super-quality:q6": 23.5}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_config

JobType = str  # "video" | "image" | "inpaint"


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    # mlx-video --pipeline value (or an equivalent runner-specific knob)
    pipeline: str | None = None
    # Multiplier applied to the base weight footprint. Two-stage HQ holds a
    # larger latent through the upscale pass.
    memory_factor: float = 1.0
    default_steps: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    job_types: tuple[str, ...]
    runner: str
    # Estimated resident GB for weights + encoders + VAE, per quantization.
    footprints: dict[str, float]
    default_quant: str
    presets: dict[str, Preset] = field(default_factory=dict)
    default_preset: str | None = None
    repo: str | None = None
    supports_image_input: bool = False
    notes: str = ""

    def quant_options(self) -> list[str]:
        return list(self.footprints.keys())


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------

LTX2_PRESETS = {
    "fast": Preset(
        "fast",
        "Fast (distilled only)",
        "Distilled single pass. Fastest, lowest quality — not the default here.",
        pipeline="distilled",
        memory_factor=0.85,
    ),
    "quality": Preset(
        "quality",
        "Quality (two-stage)",
        "Guided half-res pass -> 2x latent upscale -> distilled refine.",
        pipeline="dev-two-stage",
        memory_factor=1.0,
    ),
    "super-quality": Preset(
        "super-quality",
        "Super-Quality (two-stage HQ)",
        "Same two-stage path at higher guidance and full refine. Slowest, best.",
        pipeline="dev-two-stage-hq",
        memory_factor=1.12,
    ),
}

WAN_PRESETS = {
    "quality": Preset(
        "quality",
        "Quality",
        "40 steps, dual-model high/low noise pipeline.",
        default_steps=40,
        memory_factor=1.0,
    ),
    "super-quality": Preset(
        "super-quality",
        "Super-Quality",
        "50 steps, dual-model high/low noise pipeline.",
        default_steps=50,
        memory_factor=1.05,
    ),
}

MODELS: dict[str, ModelSpec] = {
    "ltx2": ModelSpec(
        key="ltx2",
        label="LTX-2.3 (22B)",
        job_types=("video",),
        runner="mlx_video_ltx",
        # 22B: q8 will not co-exist with a full two-stage latent inside the
        # 28GB ceiling, which is exactly why the guard exists. q6 is the
        # quality-first choice that fits.
        footprints={"q8": 26.5, "q6": 19.5, "q4": 13.5},
        default_quant="q6",
        presets=LTX2_PRESETS,
        default_preset="quality",
        repo="prince-canuma/LTX-2-dev",
        supports_image_input=True,
        notes=(
            "Default engine. Quantized is required, not optional: full precision "
            "does not fit 32GB alongside the OS."
        ),
    ),
    "wan22-14b": ModelSpec(
        key="wan22-14b",
        label="Wan2.2 14B (T2V/I2V)",
        job_types=("video",),
        runner="mlx_video_wan",
        # GGUF quants. Q4 is offered but is not a default — visible quality loss.
        footprints={"q6": 19.0, "q5": 16.5, "q4": 13.5},
        default_quant="q5",
        presets=WAN_PRESETS,
        default_preset="quality",
        repo="Wan-AI/Wan2.2-T2V-A14B",
        supports_image_input=True,
        notes="Alternate engine. Q5/Q6 GGUF only for defaults; Q4 shows visible loss.",
    ),
}

# --------------------------------------------------------------------------
# Image / inpaint
# --------------------------------------------------------------------------

IMAGE_PRESETS = {
    "quality": Preset("quality", "Quality", "25 steps, dev guidance 3.5.", default_steps=25),
    "super-quality": Preset(
        "super-quality", "Super-Quality", "40 steps, dev guidance 3.5.", default_steps=40, memory_factor=1.02
    ),
}

FILL_PRESETS = {
    "quality": Preset(
        "quality",
        "Quality",
        "FLUX.1-Fill-dev documented defaults: guidance 30, 25 steps.",
        default_steps=25,
    ),
    "super-quality": Preset(
        "super-quality",
        "Super-Quality",
        "guidance 30, 40 steps.",
        default_steps=40,
        memory_factor=1.02,
    ),
}

MODELS.update(
    {
        "flux1-dev": ModelSpec(
            key="flux1-dev",
            label="FLUX.1 [dev] (12B)",
            job_types=("image",),
            runner="mflux_generate",
            # 12B transformer + T5-XXL + CLIP + VAE.
            footprints={"q8": 17.5, "q6": 14.0, "q4": 11.0},
            default_quant="q8",
            presets=IMAGE_PRESETS,
            default_preset="quality",
            repo="black-forest-labs/FLUX.1-dev",
            notes="Q8 is the default: near-indistinguishable from full precision. Q4 is not.",
        ),
        "flux1-fill-dev": ModelSpec(
            key="flux1-fill-dev",
            label="FLUX.1-Fill-dev (inpaint/outpaint)",
            job_types=("inpaint",),
            runner="mflux_fill",
            footprints={"q8": 18.5, "q6": 15.0, "q4": 12.0},
            default_quant="q8",
            presets=FILL_PRESETS,
            default_preset="quality",
            repo="black-forest-labs/FLUX.1-Fill-dev",
            supports_image_input=True,
            notes="Guidance defaults to 30 per the model card.",
        ),
    }
)

DEFAULT_MODEL_FOR_TYPE = {
    "video": "ltx2",
    "image": "flux1-dev",
    "inpaint": "flux1-fill-dev",
}

DEFAULT_FILL_GUIDANCE = 30.0
DEFAULT_DEV_GUIDANCE = 3.5


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model '{key}' (known: {', '.join(sorted(MODELS))})") from None


def models_for_type(job_type: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if job_type in m.job_types]


def _overrides() -> dict[str, float]:
    path: Path = get_config().data_dir / "model_footprints.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return {str(k): float(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def latent_overhead_gb(
    job_type: str, width: int, height: int, frames: int, preset_key: str
) -> float:
    """Working-set cost that scales with output size, on top of the weights.

    Video latents dominate: an 8x8x8-compressed latent plus the decoder's
    tiled working buffers. The two-stage path holds a half-res latent and its
    2x upscale simultaneously during the handoff, hence the multiplier.
    """
    pixels = max(width, 1) * max(height, 1)
    if job_type == "video":
        # The latent itself is small; the VAE decode activations are the cost.
        # Reference point: 768x512x97 sits around 2.0GB of latent + decode.
        base = pixels * max(frames, 1) / (768 * 512 * 97) * 2.0
        if preset_key in {"quality", "super-quality"}:
            base *= 1.35  # both latents alive across the upscale handoff
        return round(base, 2)
    # Images: latent + VAE decode buffers, small next to the weights.
    return round(pixels / (1024 * 1024) * 0.9, 2)


def estimate_peak_gb(
    model_key: str,
    quant: str,
    preset_key: str | None,
    *,
    job_type: str,
    width: int = 1024,
    height: int = 1024,
    frames: int = 1,
) -> float:
    """Estimated peak unified-memory footprint for a job, in GB."""
    spec = get_model(model_key)
    override_key = f"{model_key}:{preset_key or 'default'}:{quant}"
    overrides = _overrides()
    if override_key in overrides:
        return overrides[override_key]

    if quant not in spec.footprints:
        raise KeyError(
            f"model '{model_key}' has no '{quant}' footprint (known: {', '.join(spec.footprints)})"
        )
    base = spec.footprints[quant]
    preset = spec.presets.get(preset_key) if preset_key else None
    factor = preset.memory_factor if preset else 1.0
    overhead = latent_overhead_gb(job_type, width, height, frames, preset_key or "")
    return round(base * factor + overhead, 2)


def catalog() -> list[dict]:
    """Serializable model catalog for the dashboard."""
    out = []
    for spec in MODELS.values():
        out.append(
            {
                "key": spec.key,
                "label": spec.label,
                "job_types": list(spec.job_types),
                "quants": spec.quant_options(),
                "default_quant": spec.default_quant,
                "default_preset": spec.default_preset,
                "supports_image_input": spec.supports_image_input,
                "repo": spec.repo,
                "notes": spec.notes,
                "presets": [
                    {
                        "key": p.key,
                        "label": p.label,
                        "description": p.description,
                        "default_steps": p.default_steps,
                    }
                    for p in spec.presets.values()
                ],
                "footprints": spec.footprints,
            }
        )
    return out
