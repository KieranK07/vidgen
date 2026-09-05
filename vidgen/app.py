"""FastAPI application: job API + dashboard.

Everything is local. There are no outbound calls in this process, and the
runners are launched with telemetry env vars disabled.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import db
from . import models as registry
from .config import get_config
from .model_manager import get_manager
from .schemas import JobCreate
from .worker import get_worker

log = logging.getLogger("vidgen.api")

ALLOWED_UPLOAD_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    logging.basicConfig(
        level=os.environ.get("VIDGEN_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    cfg.ensure_dirs()
    db.init_db()
    interrupted = list(db.reset_interrupted())
    if interrupted:
        log.warning("marked %d interrupted job(s) as failed: %s", len(interrupted), interrupted)
    worker = get_worker()
    worker.start()
    log.info(
        "vidgen ready on http://%s:%s  outputs=%s  cache=%s  ceiling=%.0fGB%s",
        cfg.host, cfg.port, cfg.output_dir, cfg.model_cache_dir, cfg.memory_ceiling_gb,
        "  [DRY RUN]" if cfg.dry_run else "",
    )
    try:
        yield
    finally:
        worker.stop()
        db.close_local()


app = FastAPI(title="vidgen", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _job_out(row: dict) -> dict:
    row = dict(row)
    row["output_url"] = f"/outputs/{row['id']}/file" if row.get("output_path") else None
    row["meta_url"] = f"/outputs/{row['id']}/meta.json" if row.get("output_path") else None
    row.pop("log_path", None)
    return row


def _safe_output_file(job_id: str, name: str) -> Path:
    cfg = get_config()
    base = (cfg.output_dir / job_id).resolve()
    if cfg.output_dir.resolve() not in base.parents:
        raise HTTPException(400, "bad job id")
    target = (base / name).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(400, "bad path")
    if not target.is_file():
        raise HTTPException(404, "not found")
    return target


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


@app.get("/api/status")
def status() -> dict:
    cfg = get_config()
    mgr = get_manager().status()
    counts = db.counts_by_status()
    return {
        **mgr,
        "queue": {
            "queued": counts.get("queued", 0),
            "held": counts.get("held", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "cancelled": counts.get("cancelled", 0),
        },
        "current_job_id": get_worker().current_job_id,
        "dry_run": cfg.dry_run,
        "output_dir": str(cfg.output_dir),
        "model_cache_dir": str(cfg.model_cache_dir),
        "events": get_manager().events(limit=25),
        "server_time": time.time(),
    }


@app.get("/api/models")
def models() -> dict:
    return {
        "models": registry.catalog(),
        "defaults": registry.DEFAULT_MODEL_FOR_TYPE,
        "fill_guidance": registry.DEFAULT_FILL_GUIDANCE,
    }


@app.post("/api/estimate")
def estimate(payload: JobCreate) -> dict:
    required = payload.estimated_gb()
    decision = get_manager().check_headroom(
        required, label=registry.get_model(payload.model).label
    )
    return {
        "estimated_gb": required,
        "fits_now": decision.ok,
        "detail": decision.as_dict(),
        "resolved": payload.to_params(),
    }


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------


@app.post("/uploads")
async def upload(file: UploadFile = File(...)) -> dict:
    cfg = get_config()
    ext = ALLOWED_UPLOAD_TYPES.get((file.content_type or "").lower())
    if ext is None:
        raise HTTPException(
            415, f"unsupported upload type {file.content_type!r}; use PNG, JPEG or WebP"
        )
    upload_id = f"{uuid.uuid4().hex}{ext}"
    dest = cfg.upload_dir / upload_id
    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "upload exceeds 40MB")
            fh.write(chunk)
    width = height = None
    try:
        from PIL import Image  # type: ignore

        with Image.open(dest) as img:
            width, height = img.size
    except Exception:
        pass
    return {
        "id": upload_id,
        "filename": file.filename,
        "width": width,
        "height": height,
        "url": f"/uploads/{upload_id}",
    }


@app.get("/uploads/{upload_id}")
def get_upload(upload_id: str) -> FileResponse:
    cfg = get_config()
    path = (cfg.upload_dir / upload_id).resolve()
    if cfg.upload_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


@app.post("/jobs", status_code=201)
def create_job(payload: JobCreate) -> dict:
    spec = registry.get_model(payload.model)
    required = payload.estimated_gb()

    # Refuse impossible jobs at submit time rather than queueing something that
    # can never run. Jobs that merely do not fit *right now* are still accepted;
    # the worker holds them until memory frees up.
    cfg = get_config()
    if required + cfg.memory_safety_margin_gb > cfg.memory_ceiling_gb:
        raise HTTPException(
            422,
            detail=(
                f"{spec.label} at {payload.quant}"
                + (f"/{payload.preset}" if payload.preset else "")
                + f" for this size needs ~{required:.1f}GB, over the "
                f"{cfg.memory_ceiling_gb:.0f}GB ceiling. Lower the resolution or frame count, "
                f"or step down one quantization level."
            ),
        )

    job_id = db.create_job(
        job_type=payload.type,
        model=payload.model,
        quant=payload.quant,
        preset=payload.preset,
        prompt=payload.prompt,
        params=payload.to_params(),
        estimated_gb=required,
    )
    get_worker().notify()
    row = db.get_job(job_id)
    assert row is not None
    return _job_out(row)


@app.get("/jobs")
def list_jobs(
    status: str | None = Query(default=None),
    type: str | None = Query(default=None),
    model: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    rows = db.list_jobs(status=status, job_type=type, model=model, limit=limit)
    return {"jobs": [_job_out(r) for r in rows]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    out = _job_out(row)
    meta_path = get_config().output_dir / job_id / "meta.json"
    if meta_path.is_file():
        try:
            out["meta"] = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    return out


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, purge: bool = Query(default=True)) -> dict:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")

    worker = get_worker()
    if row["status"] in ("queued", "held", "running"):
        was_running = worker.request_cancel(job_id)
        db.update_job(
            job_id,
            status="cancelled",
            finished_at=time.time(),
            message="Cancelled by user.",
        )
        return {"id": job_id, "cancelled": True, "was_running": was_running, "deleted": False}

    if purge:
        shutil.rmtree(get_config().output_dir / job_id, ignore_errors=True)
    db.delete_job(job_id)
    return {"id": job_id, "cancelled": False, "deleted": True, "purged": purge}


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


@app.get("/outputs/{job_id}/file")
def output_file(job_id: str, download: bool = Query(default=False)) -> FileResponse:
    row = db.get_job(job_id)
    if row is None or not row.get("output_path"):
        raise HTTPException(404, "no output for that job")
    path = Path(row["output_path"])
    if not path.is_file():
        raise HTTPException(404, "output file is missing on disk")
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    suffix = path.suffix
    return FileResponse(
        path,
        media_type=media,
        filename=f"vidgen-{job_id}{suffix}" if download else None,
    )


@app.get("/outputs/{job_id}/{name}")
def output_asset(job_id: str, name: str) -> FileResponse:
    return FileResponse(_safe_output_file(job_id, name))


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_model=None)
def index():
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return RedirectResponse("/docs")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=422, content={"detail": str(exc)})
