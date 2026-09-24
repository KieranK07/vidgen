# vidgen

![Queue view showing a job held with a plain-language memory reason ("needs ~11.2GB but only 8.9GB is free under the 28GB ceiling... Over by 3.4GB"), alongside queued and completed jobs and the model manager log](docs/img/dashboard-held-job.png)

A local job queue and dashboard for LTX-2.3, Wan2.2 and FLUX.1 on a 32GB Apple
Silicon Mac, with admission control that keeps the machine from swapping.
Text-to-video, image-to-video, text-to-image and inpaint/outpaint, queued and
browsable from one page. No cloud calls, no telemetry, no accounts; the only
network traffic is the first model download from Hugging Face.

```
  Video   LTX-2.3 (22B) via mlx-video, Quality / Super-Quality two-stage
          Wan2.2 14B via mlx-video, Q5/Q6 GGUF (alternate engine)
  Image   FLUX.1 [dev] (12B) via mflux, Q8
  Fill    FLUX.1-Fill-dev via mflux, guidance 30 / 25 steps
```

## How it works

On Apple Silicon the GPU allocates from the same 32GB that macOS uses. Going
over does not raise an out-of-memory error; the Mac starts swapping and a
20-minute render takes an hour. vidgen decides whether a job may start at all.

- **One model resident at a time.** The queue is sequential, and the model
  manager unloads the current model before checking whether the next one fits.
  An LTX-2.3 to FLUX.1 switch never holds both.
- **Runners are child processes.** Dropping references in a long-lived Python
  process does not reliably return Metal memory. When a child process exits,
  every buffer is freed, and a wedged render can be killed without taking the
  service down.
- **Headroom guard at 28GB.** Before a load, vidgen reads unified-memory state
  (`psutil`, else `sysctl` + `vm_stat`) and compares `in use + estimated peak +
  1GB margin` against the ceiling. Estimates come from a per-model, per-quant,
  per-preset table plus a term that scales with resolution and frame count.

Each job gets one of three outcomes:

- Fits now: load and run.
- Does not fit right now: **held** with a reason ("Over by 3.2GB, close memory-heavy apps")
  and re-checked on a timer.
- Can never fit on this machine (LTX-2.3 at Q8, say): rejected at submit time
  with a suggestion.

If the memory probe fails, the guard refuses rather than loading blind.
Defaults are the highest quality that fits: Q8 for FLUX, Q6 for LTX-2.3, Q5 for
Wan2.2.

## Install

Requires an Apple Silicon Mac on macOS 14+, Python 3.11+, and about 120GB of
disk for all four model families. `ffmpeg` is optional (dry-run stub only).

```bash
./scripts/setup.sh                 # .venv, service, test deps, both model stacks
WITH_MODELS=0 ./scripts/setup.sh   # service and test deps only
```

`setup.sh` copies `.env.example` to `.env` if it is missing. `.env` sets the
model cache, output directory, port and every memory threshold. Variables
already set in the shell take precedence over `.env`.

Weights download on first use. Per-model setup (MLX-converted LTX repo, Wan
GGUF directory, gated FLUX repos) is in [docs/operations.md](docs/operations.md).

## Run

```bash
./scripts/vidgen.sh start      # also: stop | restart | status | logs
./scripts/vidgen.sh install    # launchd agent at login; uninstall to remove
```

Then open http://127.0.0.1:8817.

### Without weights

```bash
VIDGEN_DRY_RUN=1 ./scripts/vidgen.sh restart
```

Dry-run swaps in a stub renderer that runs as a subprocess, allocates memory,
reports progress and writes an output file, so the queue, guard, gallery and
mask editor work end to end.

By default the stub is admitted at its own small footprint, so jobs never hold.
To see the held path, have the guard charge the real model's estimate:

```bash
VIDGEN_DRY_RUN=1 VIDGEN_DRY_RUN_REAL_ESTIMATES=1 ./scripts/vidgen.sh restart
```

A default LTX-2.3 video job is then held whenever less than its estimated peak
is free under the ceiling. `VIDGEN_DRY_RUN_ALLOC_MB` sets how much memory the
stub itself allocates.

## Using it

Tabs for Video, Image, Inpaint/Outpaint and Gallery, plus a Queue view. The
header shows the resident model, memory used against the ceiling, and a swap
warning. Frame counts snap to 8n+1 and dimensions to multiples of 64.

| Preset | LTX-2.3 pipeline | What it does |
| --- | --- | --- |
| Fast | `distilled` | Single distilled pass, for comparison. |
| **Quality** | `dev-two-stage` | Half-res pass, 2x latent upscale, distilled refine. Default. |
| Super-Quality | `dev-two-stage-hq` | Same path at higher guidance and full refine. |

The inpaint editor paints a mask (white regenerates, black keeps). Outpainting
sets margins and pre-fills the new area into the mask.

## API

![FastAPI /docs endpoint list for the vidgen HTTP API](docs/img/api-docs.png)

```
POST   /jobs               queue a job
GET    /jobs               list jobs         ?status=&type=&model=&limit=
GET    /jobs/{id}          one job, with meta.json inlined when finished
DELETE /jobs/{id}          cancel if active, delete (and purge files) if finished
POST   /uploads            multipart image upload -> {id}, for i2v / inpaint
GET    /outputs/{id}/file  the render        ?download=true
GET    /outputs/{id}/meta.json
GET    /api/status         resident model, memory, queue counts, event log
GET    /api/models         model + preset + quantization catalog
POST   /api/estimate       what a job would cost, and whether it fits right now
```

```bash
curl -X POST localhost:8817/jobs -H 'Content-Type: application/json' -d '{
  "type": "video",
  "prompt": "A lighthouse in heavy fog, slow push-in, anamorphic",
  "model": "ltx2", "preset": "super-quality", "quant": "q6",
  "width": 768, "height": 512, "duration_seconds": 4, "fps": 24, "seed": 7
}'
```

`type` is `video`, `image` or `inpaint`. Image-to-video adds a `source_image`
upload id; inpaint adds `source_image` and `mask_image`. Each finished job
writes `outputs/<job_id>/` with the render, `meta.json` (parameters, exact
command, estimated and observed memory peak) and `run.log`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers the headroom guard and single-model invariant against a faked
memory probe, and drives the API against the stub renderer. It needs no GPU and
no weights.

## Limits

- Tested on one machine, an M4 Air with 32GB. On a larger Mac, raise
  `VIDGEN_MEMORY_CEILING_GB`.
- Footprint estimates are hand-measured. Observed peaks are logged for
  correction; see [docs/operations.md](docs/operations.md).
- The Wan2.2 runner is tested for argument construction but has seen far less
  use than the LTX-2.3 and FLUX paths.
- Upstream CLIs rename flags between releases; each runner has an extra-args
  escape hatch in `.env`.
- Renders are not resumable. A restart mid-job loses the job.

Operating notes for fanless Macs and troubleshooting are in
[docs/operations.md](docs/operations.md).

## Layout

```
vidgen/
  config.py          env-driven configuration
  memory.py          unified-memory probing (psutil -> vm_stat -> /proc)
  models.py          model/preset/quant registry + peak-footprint estimator
  model_manager.py   single-active-model policy, headroom guard, subprocess exec
  db.py              SQLite job store (WAL)
  worker.py          single-worker sequential queue
  schemas.py         request validation + parameter resolution
  runners/           mlx-video (LTX, Wan) and mflux (generate, fill) adapters
  static/index.html  the dashboard
scripts/             setup.sh, vidgen.sh
tests/               memory-guard and API tests
docs/operations.md   model setup, fanless notes, troubleshooting
```

## License

MIT. See [LICENSE](LICENSE).
