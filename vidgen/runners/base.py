"""Runner interface.

A runner turns a job row into a command line. It never loads a model itself —
the model manager owns that, and executes the command in a child process so
that "unloaded" means the OS has the memory back.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..models import ModelSpec


@dataclass
class JobContext:
    job_id: str
    job_type: str
    spec: ModelSpec
    quant: str
    preset: str | None
    params: dict[str, Any]
    output_dir: Path
    log_path: Path
    cfg: Config

    def param(self, key: str, default: Any = None) -> Any:
        value = self.params.get(key, default)
        return default if value is None else value

    def upload_path(self, upload_id: str) -> Path:
        path = (self.cfg.upload_dir / upload_id).resolve()
        if self.cfg.upload_dir not in path.parents:
            raise ValueError(f"upload id escapes the upload directory: {upload_id!r}")
        if not path.is_file():
            raise FileNotFoundError(f"upload {upload_id!r} no longer exists on disk")
        return path


@dataclass
class RunPlan:
    argv: list[str]
    output_file: Path
    env: dict[str, str] = field(default_factory=dict)
    # Extra files the runner is expected to produce (e.g. a last frame).
    extra_outputs: list[Path] = field(default_factory=list)
    describe: str = ""


class Runner:
    key: str = ""
    #: human-readable name of the external tool, for error messages
    tool: str = ""

    def build(self, ctx: JobContext) -> RunPlan:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def quant_bits(quant: str) -> int:
        m = re.fullmatch(r"q(\d+)", quant.strip().lower())
        if not m:
            raise ValueError(f"unrecognised quantization {quant!r}; expected q4/q5/q6/q8")
        return int(m.group(1))

    @staticmethod
    def python_exe(cfg: Config) -> str:
        return cfg.python_executable or sys.executable

    @staticmethod
    def extra_args(env_var: str) -> list[str]:
        """Escape hatch: append arbitrary flags without editing code.

        These upstream CLIs move fast; VIDGEN_LTX_EXTRA_ARGS etc. let you adapt
        to a flag rename without waiting on this project.
        """
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return []
        import shlex

        return shlex.split(raw)

    @staticmethod
    def require_tool(name: str) -> str:
        found = shutil.which(name)
        if not found:
            raise FileNotFoundError(
                f"'{name}' is not on PATH. Install it (see README) or point at it with the "
                f"matching VIDGEN_* environment variable."
            )
        return found


#: line -> progress fraction (0..1) or None
ProgressParser = Callable[[str], float | None]

_PCT = re.compile(r"(\d{1,3})\s?%")
_STEP = re.compile(r"(?:step|frame|it)[^0-9]{0,4}(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_BAR = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def parse_progress(line: str) -> float | None:
    """Best-effort progress from tqdm-ish runner output."""
    m = _STEP.search(line) or _BAR.search(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return max(0.0, min(done / total, 1.0))
    m = _PCT.search(line)
    if m:
        return max(0.0, min(int(m.group(1)) / 100, 1.0))
    return None
