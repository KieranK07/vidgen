from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(env):
    from vidgen.app import app

    with TestClient(app) as c:
        yield c


def wait_for(client, job_id, statuses, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/jobs/{job_id}").json()
        if last["status"] in statuses:
            return last
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} stuck in {last and last['status']}: {last}")


def png_bytes() -> bytes:
    from vidgen.runners.fake_worker import write_png
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.png"
        write_png(p, 64, 64, 3)
        return p.read_bytes()


def test_status_and_models(client):
    st = client.get("/api/status").json()
    assert st["resident_model"] is None
    assert st["ceiling_gb"] == 28.0
    assert st["dry_run"] is True

    models = client.get("/api/models").json()
    keys = {m["key"] for m in models["models"]}
    assert {"ltx2", "wan22-14b", "flux1-dev", "flux1-fill-dev"} <= keys
    assert models["defaults"]["video"] == "ltx2"
    assert models["fill_guidance"] == 30.0


def test_video_defaults_are_quality_not_fast(client):
    r = client.post("/jobs", json={"type": "video", "prompt": "a lighthouse in fog"})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["model"] == "ltx2"
    assert job["preset"] == "quality"
    assert job["quant"] == "q6"
    assert job["params"]["num_frames"] % 8 == 1
    assert job["params"]["width"] % 64 == 0


def test_image_job_completes_and_writes_meta(client):
    r = client.post(
        "/jobs",
        json={"type": "image", "prompt": "a brass telescope", "width": 512, "height": 512, "seed": 11},
    )
    job_id = r.json()["id"]
    done = wait_for(client, job_id, {"completed", "failed"})
    assert done["status"] == "completed", done.get("error")
    assert done["meta"]["model"] == "flux1-dev"
    assert done["meta"]["quant"] == "q8"
    assert done["meta"]["seed"] == 11
    assert done["meta"]["memory"]["ceiling_gb"] == 28.0
    assert done["meta"]["memory"]["observed_peak_rss_gb"] > 0
    assert done["meta"]["timestamps"]["duration_seconds"] >= 0

    out = client.get(f"/outputs/{job_id}/file")
    assert out.status_code == 200
    assert out.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_job_over_ceiling_is_rejected_at_submit(client):
    r = client.post(
        "/jobs",
        json={
            "type": "video",
            "prompt": "impossible",
            "quant": "q8",
            "preset": "super-quality",
            "width": 1280,
            "height": 704,
            "num_frames": 193,
        },
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "ceiling" in detail and "quantization" in detail


def test_invalid_model_for_type(client):
    r = client.post("/jobs", json={"type": "image", "prompt": "x", "model": "ltx2"})
    assert r.status_code == 422


def test_inpaint_requires_uploads(client):
    r = client.post("/jobs", json={"type": "inpaint", "prompt": "a red door"})
    assert r.status_code == 422


def test_inpaint_roundtrip(client):
    data = png_bytes()
    up1 = client.post("/uploads", files={"file": ("src.png", io.BytesIO(data), "image/png")}).json()
    up2 = client.post("/uploads", files={"file": ("mask.png", io.BytesIO(data), "image/png")}).json()

    r = client.post(
        "/jobs",
        json={
            "type": "inpaint",
            "prompt": "a red door",
            "source_image": up1["id"],
            "mask_image": up2["id"],
            "width": 512,
            "height": 512,
        },
    )
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["model"] == "flux1-fill-dev"
    assert job["params"]["guidance"] == 30.0
    assert job["params"]["steps"] == 25

    done = wait_for(client, job["id"], {"completed", "failed"})
    assert done["status"] == "completed", done.get("error")
    assert client.get(f"/outputs/{job['id']}/source.png").status_code == 200
    assert client.get(f"/outputs/{job['id']}/mask.png").status_code == 200


def test_upload_rejects_bad_type(client):
    r = client.post("/uploads", files={"file": ("x.txt", io.BytesIO(b"nope"), "text/plain")})
    assert r.status_code == 415


def test_list_filter_and_delete(client):
    r = client.post("/jobs", json={"type": "image", "prompt": "a fig", "width": 256, "height": 256})
    job_id = r.json()["id"]
    wait_for(client, job_id, {"completed", "failed"})

    assert len(client.get("/jobs", params={"type": "image"}).json()["jobs"]) == 1
    assert len(client.get("/jobs", params={"type": "video"}).json()["jobs"]) == 0
    assert len(client.get("/jobs", params={"model": "flux1-dev"}).json()["jobs"]) == 1

    d = client.delete(f"/jobs/{job_id}").json()
    assert d["deleted"] is True
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_delete_queued_job_cancels(client, monkeypatch):
    monkeypatch.setenv("VIDGEN_DRY_RUN_STEP_SECONDS", "0.5")
    ids = [
        client.post("/jobs", json={"type": "image", "prompt": f"p{i}", "steps": 20}).json()["id"]
        for i in range(3)
    ]
    last = ids[-1]
    d = client.delete(f"/jobs/{last}").json()
    assert d["cancelled"] is True
    assert client.get(f"/jobs/{last}").json()["status"] == "cancelled"


def test_path_traversal_blocked(client):
    r = client.post("/jobs", json={"type": "image", "prompt": "x", "width": 256, "height": 256})
    job_id = r.json()["id"]
    wait_for(client, job_id, {"completed", "failed"})
    assert client.get(f"/outputs/{job_id}/../../../etc/passwd").status_code in (400, 404)


def test_estimate_endpoint(client):
    r = client.post(
        "/api/estimate",
        json={"type": "video", "prompt": "x", "quant": "q6", "preset": "super-quality"},
    )
    body = r.json()
    assert body["estimated_gb"] > 15
    assert "fits_now" in body
    assert body["resolved"]["num_frames"] % 8 == 1
