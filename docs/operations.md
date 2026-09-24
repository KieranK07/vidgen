# Operating vidgen

## Model downloads

Weights are fetched on first use into `VIDGEN_MODEL_CACHE_DIR` (default
`~/.cache/huggingface`). The first run of each model is slow; after that
everything is local.

- **LTX-2.3** (default video engine, ~22B) needs an MLX-converted repo, not the
  original weights. mlx-video's conversions live under `prince-canuma/LTX-2-*`:
  `hf download prince-canuma/LTX-2-dev`, then set `VIDGEN_LTX_MODEL_REPO`.
  Quantization is applied at load time and is required, since full precision
  does not fit alongside macOS.
- **Wan2.2** (alternate video engine) needs the 14B T2V/I2V weights, not the 5B
  or 1.3B variants, at Q5 or Q6 GGUF. Q4 has visible quality loss. Point
  `VIDGEN_WAN_MODEL_DIR` at the directory; Wan jobs fail with a clear message
  until it is set.
- **FLUX.1 [dev]** and **FLUX.1-Fill-dev** are gated repos. Accept the license
  on each model page and run `hf auth login` once. mflux resolves them from the
  HF cache; `VIDGEN_FLUX_DEV_PATH` / `VIDGEN_FLUX_FILL_PATH` override with a
  local directory.

## Fanless machines

Sustained GPU load on a fanless Mac raises the chassis temperature until macOS
lowers clocks.

- A long session slows down. The first Super-Quality LTX-2.3 render is the
  fastest; the third or fourth back-to-back render takes noticeably longer.
- For consistent timings (comparing prompts or seeds), leave a few minutes
  between runs.
- The quality-first defaults are slow by design. Queue a job and leave it.
- Close memory-heavy apps before a big render, browsers above all. A browser
  with many tabs can hold 4-8GB, which decides whether LTX-2.3 at Q6 fits under
  the 28GB ceiling or is held. The header shows how much room is left.
- If the header shows a swap warning, stop queueing and free memory. The guard
  controls only vidgen's own allocations.

## Troubleshooting

**"needs ~X GB but only Y GB is free under the ceiling"**: working as intended.
Close some apps, or lower resolution, frame count or quantization. The job
stays held and starts once there is room.

**"...which alone exceeds the 28GB ceiling"**: that combination can never run
on this machine. Step the quantization down (Q6 for LTX-2.3, not Q8) or shrink
the output.

**A runner exits with an unrecognized-argument error.** The upstream CLIs rename
flags between releases. Use the escape hatches in `.env`:
`VIDGEN_LTX_EXTRA_ARGS`, `VIDGEN_WAN_EXTRA_ARGS`, `VIDGEN_MFLUX_EXTRA_ARGS` and
`VIDGEN_MFLUX_FILL_EXTRA_ARGS` are appended verbatim, and
`VIDGEN_MLX_VIDEO_QUANT_FLAG` renames the quantization flag. See
`outputs/<job_id>/run.log` for the failure and `meta.json` for the exact
command.

**`mflux-generate: not found`**: the service is running from a different
interpreter than the one mflux was installed into. Set `VIDGEN_PYTHON` to the
venv's python, or `VIDGEN_MFLUX_GENERATE` to the binary's absolute path.

**Jobs marked "Service restarted while this job was running."**: expected after
a crash or restart mid-render. Renders are not resumable; resubmit.

**Gated-repo 401s from Hugging Face**: accept the license for FLUX.1-dev and
FLUX.1-Fill-dev on their model pages, then `hf auth login`.

## Footprint corrections

Observed peaks accumulate in `data/observed_footprints.json`. If real numbers
drift from the built-in estimates, write corrections into
`data/model_footprints.json` (`{"ltx2:super-quality:q6": 23.5}`) and the guard
uses those instead.
