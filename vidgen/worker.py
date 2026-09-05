"""Single-worker sequential job queue.

One worker thread, one job at a time. This is not a throughput compromise —
it is the direct consequence of the single-active-model policy. Two concurrent
renders on 32GB of unified memory means two models resident, which means swap,
which on a fanless M4 is worse than serial by a wide margin.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from . import db
from .config import get_config
from .memory import snapshot
from .model_manager import MemoryGuardError, get_manager
from .models import get_model
from .runners import JobContext, get_runner, parse_progress

log = logging.getLogger("vidgen.worker")

_LOG_TAIL_LINES = 40


class Worker:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._cancel_requested: set[str] = set()
        self._lock = threading.Lock()
        self.current_job_id: str | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vidgen-worker", daemon=True)
        self._thread.start()
        log.info("worker started")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        get_manager().cancel_running()
        if self._thread:
            self._thread.join(timeout=timeout)
        log.info("worker stopped")

    def notify(self) -> None:
        self._wake.set()

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            self._cancel_requested.add(job_id)
            is_current = self.current_job_id == job_id
        if is_current:
            get_manager().cancel_running()
        return is_current

    def _cancel_pending(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel_requested

    def _clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancel_requested.discard(job_id)

    # -- main loop --------------------------------------------------------

    def _loop(self) -> None:
        cfg = get_config()
        db.init_db()
        while not self._stop.is_set():
            job = db.next_queued()
            if job is None:
                self._wake.wait(timeout=2.0)
                self._wake.clear()
                continue
            if self._cancel_pending(job["id"]):
                db.update_job(
                    job["id"], status="cancelled", finished_at=time.time(), message="Cancelled before start."
                )
                self._clear_cancel(job["id"])
                continue
            try:
                self._run_job(job)
            except Exception:  # pragma: no cover - defensive
                log.exception("worker crashed handling job %s", job["id"])
                db.update_job(
                    job["id"],
                    status="failed",
                    error="Internal worker error; see the service log.",
                    finished_at=time.time(),
                )
            # If the last job was held for memory, back off before retrying.
            latest = db.get_job(job["id"])
            if latest and latest["status"] == "held":
                self._wake.wait(timeout=cfg.memory_hold_retry_seconds)
                self._wake.clear()

    # -- one job ----------------------------------------------------------

    def _run_job(self, job: dict[str, Any]) -> None:
        cfg = get_config()
        manager = get_manager()
        job_id = job["id"]
        params = job["params"]
        spec = get_model(job["model"])

        out_dir = cfg.output_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"

        ctx = JobContext(
            job_id=job_id,
            job_type=job["type"],
            spec=spec,
            quant=job["quant"] or spec.default_quant,
            preset=job["preset"],
            params=params,
            output_dir=out_dir,
            log_path=log_path,
            cfg=cfg,
        )

        try:
            runner = get_runner(spec.runner, dry_run=cfg.dry_run)
            plan = runner.build(ctx)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            db.update_job(
                job_id, status="failed", error=str(exc), finished_at=time.time(), started_at=time.time()
            )
            return

        estimated = float(job["estimated_gb"] or 0.0)
        if cfg.dry_run:
            # The stub renderer does not load weights, so charging it the real
            # model's footprint would hold every job forever. Submit-time
            # admission control still uses the real estimate.
            alloc_gb = float(os.environ.get("VIDGEN_DRY_RUN_ALLOC_MB", "256")) / 1024
            estimated = round(alloc_gb + 0.5, 2)
        started = time.time()

        with self._lock:
            self.current_job_id = job_id

        try:
            with manager.activate(
                model_key=spec.key,
                label=spec.label,
                quant=ctx.quant,
                preset=ctx.preset,
                estimated_gb=estimated,
                job_id=job_id,
            ) as active:
                db.update_job(
                    job_id,
                    status="running",
                    started_at=started,
                    progress=0.0,
                    message=plan.describe,
                    error=None,
                    log_path=str(log_path),
                    hold_attempts=0,
                )
                mem_at_start = snapshot()
                last_write = 0.0

                def on_line(line: str) -> None:
                    nonlocal last_write
                    frac = parse_progress(line)
                    now = time.time()
                    if frac is not None and now - last_write > 1.0:
                        last_write = now
                        db.update_job(job_id, progress=round(frac, 4), message=line[:200])

                code, peak = active.run(
                    plan.argv,
                    env=plan.env,
                    log_path=log_path,
                    on_line=on_line,
                    timeout=cfg.job_timeout_seconds,
                )
        except MemoryGuardError as exc:
            attempts = int(job["hold_attempts"] or 0) + 1
            if attempts >= cfg.memory_hold_max_attempts:
                db.update_job(
                    job_id,
                    status="failed",
                    error=f"Held {attempts} times without enough free memory. {exc}",
                    finished_at=time.time(),
                    hold_attempts=attempts,
                )
            else:
                db.update_job(
                    job_id,
                    status="held",
                    message=str(exc),
                    hold_attempts=attempts,
                    progress=0.0,
                )
            return
        except TimeoutError as exc:
            db.update_job(
                job_id,
                status="failed",
                error=str(exc),
                finished_at=time.time(),
            )
            return
        finally:
            with self._lock:
                self.current_job_id = None

        finished = time.time()

        if self._cancel_pending(job_id):
            self._clear_cancel(job_id)
            db.update_job(
                job_id, status="cancelled", finished_at=finished, message="Cancelled while running."
            )
            return

        if code != 0 or not plan.output_file.exists():
            tail = _tail(log_path)
            reason = (
                f"{runner.tool} exited with code {code}."
                if code != 0
                else f"{runner.tool} exited cleanly but produced no output file."
            )
            db.update_job(
                job_id,
                status="failed",
                error=f"{reason}\n\n{tail}".strip(),
                finished_at=finished,
                peak_gb=round(peak, 2),
            )
            return

        meta = {
            "job_id": job_id,
            "type": job["type"],
            "prompt": job["prompt"],
            "negative_prompt": params.get("negative_prompt"),
            "model": spec.key,
            "model_label": spec.label,
            "model_repo": spec.repo,
            "preset": ctx.preset,
            "quant": ctx.quant,
            "seed": params.get("seed"),
            "width": params.get("width"),
            "height": params.get("height"),
            "num_frames": params.get("num_frames"),
            "fps": params.get("fps"),
            "duration_seconds": params.get("duration_seconds"),
            "steps": params.get("steps"),
            "guidance": params.get("guidance"),
            "cfg_scale": params.get("cfg_scale"),
            "command": plan.argv,
            "dry_run": cfg.dry_run,
            "memory": {
                "estimated_peak_gb": round(estimated, 2),
                "observed_peak_rss_gb": round(peak, 2),
                "ceiling_gb": cfg.memory_ceiling_gb,
                "system_used_gb_at_start": round(mem_at_start.used_gb, 2),
                "system_available_gb_at_start": round(mem_at_start.available_gb, 2),
                "swap_used_gb_at_end": round(snapshot().swap_used_gb, 2),
                "probe": mem_at_start.source,
            },
            "timestamps": {
                "created_at": job["created_at"],
                "started_at": started,
                "finished_at": finished,
                "duration_seconds": round(finished - started, 1),
            },
            "output": plan.output_file.name,
        }

        # Keep the inpaint inputs beside the result so the before/after view
        # still works after the uploads directory is cleaned.
        if job["type"] == "inpaint":
            for key, name in (("source_image", "source.png"), ("mask_image", "mask.png")):
                upload_id = params.get(key)
                if not upload_id:
                    continue
                try:
                    shutil.copy2(ctx.upload_path(upload_id), out_dir / name)
                    meta[key] = name
                except (OSError, ValueError, FileNotFoundError):
                    pass

        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        db.update_job(
            job_id,
            status="completed",
            progress=1.0,
            message="Done.",
            output_path=str(plan.output_file),
            peak_gb=round(peak, 2),
            finished_at=finished,
            error=None,
        )
        _record_observed_footprint(spec.key, ctx.preset, ctx.quant, peak)


def _tail(path: Path, lines: int = _LOG_TAIL_LINES) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _record_observed_footprint(model_key: str, preset: str | None, quant: str, peak_gb: float) -> None:
    """Feed real peaks back into the estimator so the guard sharpens with use."""
    if peak_gb <= 0:
        return
    cfg = get_config()
    path = cfg.data_dir / "observed_footprints.json"
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    key = f"{model_key}:{preset or 'default'}:{quant}"
    entry = data.get(key, {"samples": 0, "max_gb": 0.0})
    entry["samples"] = int(entry.get("samples", 0)) + 1
    entry["max_gb"] = round(max(float(entry.get("max_gb", 0.0)), peak_gb), 2)
    entry["last_gb"] = round(peak_gb, 2)
    data[key] = entry
    try:
        path.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


_worker: Worker | None = None


def get_worker() -> Worker:
    global _worker
    if _worker is None:
        _worker = Worker()
    return _worker


def reset_worker() -> None:
    global _worker
    if _worker is not None:
        _worker.stop(timeout=2)
    _worker = None
