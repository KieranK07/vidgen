"""Runner registry."""

from __future__ import annotations

from .base import JobContext, RunPlan, Runner, parse_progress
from .dry_run_runner import DryRunRunner
from .mflux_runner import MfluxFillRunner, MfluxGenerateRunner
from .mlx_video_runner import LTXRunner, WanRunner

_RUNNERS: dict[str, Runner] = {
    r.key: r
    for r in (LTXRunner(), WanRunner(), MfluxGenerateRunner(), MfluxFillRunner())
}

_DRY_RUN = DryRunRunner()


def get_runner(key: str, *, dry_run: bool = False) -> Runner:
    if dry_run:
        return _DRY_RUN
    try:
        return _RUNNERS[key]
    except KeyError:
        raise KeyError(f"no runner registered for '{key}'") from None


__all__ = [
    "JobContext",
    "RunPlan",
    "Runner",
    "get_runner",
    "parse_progress",
]
