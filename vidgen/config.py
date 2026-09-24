"""Configuration for vidgen.

Everything is driven by environment variables (optionally loaded from a .env
file next to the project root). No hardcoded machine-specific paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency)."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(Path(os.environ.get("VIDGEN_ENV_FILE", PROJECT_ROOT / ".env")))


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser().resolve()


def _float(env: str, default: float) -> float:
    try:
        return float(os.environ[env])
    except (KeyError, ValueError):
        return default


def _int(env: str, default: int) -> int:
    try:
        return int(os.environ[env])
    except (KeyError, ValueError):
        return default


def _bool(env: str, default: bool) -> bool:
    val = os.environ.get(env)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # --- paths -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _path("VIDGEN_DATA_DIR", str(PROJECT_ROOT / "data")))
    output_dir: Path = field(default_factory=lambda: _path("VIDGEN_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
    upload_dir: Path = field(default_factory=lambda: _path("VIDGEN_UPLOAD_DIR", str(PROJECT_ROOT / "uploads")))
    model_cache_dir: Path = field(
        default_factory=lambda: _path("VIDGEN_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
    )
    log_dir: Path = field(default_factory=lambda: _path("VIDGEN_LOG_DIR", str(PROJECT_ROOT / "logs")))

    # --- server ----------------------------------------------------------
    host: str = field(default_factory=lambda: os.environ.get("VIDGEN_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("VIDGEN_PORT", 8817))

    # --- memory policy ---------------------------------------------------
    # Hard ceiling on total system allocation. On a 32GB unified-memory Mac we
    # keep ~4GB for macOS + background processes. A job whose estimated peak
    # would cross this line is refused/held rather than allowed to swap.
    total_memory_gb: float = field(default_factory=lambda: _float("VIDGEN_TOTAL_MEMORY_GB", 32.0))
    memory_ceiling_gb: float = field(default_factory=lambda: _float("VIDGEN_MEMORY_CEILING_GB", 28.0))
    # Extra slack subtracted from measured availability to absorb transient
    # allocation spikes inside a runner.
    memory_safety_margin_gb: float = field(default_factory=lambda: _float("VIDGEN_MEMORY_SAFETY_MARGIN_GB", 1.0))
    # When a job cannot fit right now, hold it in the queue this many seconds
    # before re-checking, instead of failing it outright.
    memory_hold_retry_seconds: int = field(default_factory=lambda: _int("VIDGEN_MEMORY_HOLD_RETRY_SECONDS", 30))
    # After this many consecutive holds the job is failed with a clear message.
    memory_hold_max_attempts: int = field(default_factory=lambda: _int("VIDGEN_MEMORY_HOLD_MAX_ATTEMPTS", 20))

    # --- execution -------------------------------------------------------
    # Runners execute in a child process so that unloading a model is a
    # guaranteed OS-level free of every Metal buffer, not a best-effort
    # Python-side drop. See docs in model_manager.py.
    python_executable: str = field(default_factory=lambda: os.environ.get("VIDGEN_PYTHON", ""))
    job_timeout_seconds: int = field(default_factory=lambda: _int("VIDGEN_JOB_TIMEOUT_SECONDS", 60 * 90))
    dry_run: bool = field(default_factory=lambda: _bool("VIDGEN_DRY_RUN", False))
    # In dry-run mode, admit the stub using the real model's estimated peak
    # rather than the stub's own small allocation, so the hold path shows up.
    dry_run_real_estimates: bool = field(default_factory=lambda: _bool("VIDGEN_DRY_RUN_REAL_ESTIMATES", False))

    # --- model tooling ---------------------------------------------------
    mlx_video_module: str = field(default_factory=lambda: os.environ.get("VIDGEN_MLX_VIDEO_MODULE", "mlx_video"))
    mflux_generate_bin: str = field(default_factory=lambda: os.environ.get("VIDGEN_MFLUX_GENERATE", "mflux-generate"))
    mflux_fill_bin: str = field(default_factory=lambda: os.environ.get("VIDGEN_MFLUX_FILL", "mflux-generate-fill"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vidgen.sqlite3"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.output_dir, self.upload_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_dirs()
    return _config


def reset_config() -> None:
    """Test hook — forces re-read of the environment."""
    global _config
    _config = None
