from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated config + DB + fresh singletons for each test."""
    for key in list(os.environ):
        if key.startswith("VIDGEN_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VIDGEN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIDGEN_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VIDGEN_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VIDGEN_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("VIDGEN_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VIDGEN_ENV_FILE", str(tmp_path / "nonexistent.env"))
    monkeypatch.setenv("VIDGEN_DRY_RUN", "1")
    monkeypatch.setenv("VIDGEN_DRY_RUN_STEP_SECONDS", "0.01")
    monkeypatch.setenv("VIDGEN_DRY_RUN_ALLOC_MB", "16")
    monkeypatch.setenv("VIDGEN_MEMORY_HOLD_RETRY_SECONDS", "1")

    from vidgen import config, db, model_manager, worker

    config.reset_config()
    db.close_local()
    model_manager.reset_manager()
    worker.reset_worker()
    cfg = config.get_config()
    db.init_db()
    yield cfg
    worker.reset_worker()
    model_manager.reset_manager()
    db.close_local()
    config.reset_config()
