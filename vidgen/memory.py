"""Unified-memory probing.

On Apple Silicon the GPU and CPU share one pool, so "how much VRAM is left" is
just "how much unified memory is left". Once that pool is exhausted macOS
starts compressing and swapping, and Metal allocations begin to stall — on a
fanless M4 Air that turns a 20-minute render into an hour. We therefore treat
crossing the ceiling as a failure to *prevent*, not a slowdown to tolerate.

Probing strategy (macOS first, portable fallbacks for dev/CI):
  1. psutil, if installed — best cross-platform answer.
  2. sysctl + vm_stat — no dependencies, macOS native.
  3. /proc/meminfo — Linux, used by the test suite and CI.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

GB = 1024 ** 3


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float
    available_gb: float
    used_gb: float
    swap_used_gb: float
    source: str

    def as_dict(self) -> dict:
        return {
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "swap_used_gb": round(self.swap_used_gb, 2),
            "source": self.source,
        }


def _sysctl_int(name: str) -> int | None:
    if not shutil.which("sysctl"):
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        return int(out)
    except (subprocess.SubprocessError, ValueError):
        return None


def _macos_swap_used_gb() -> float:
    if not shutil.which("sysctl"):
        return 0.0
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except subprocess.SubprocessError:
        return 0.0
    m = re.search(r"used\s*=\s*([\d.]+)([MGK])", out)
    if not m:
        return 0.0
    value, unit = float(m.group(1)), m.group(2)
    return {"K": value / (1024 * 1024), "M": value / 1024, "G": value}.get(unit, 0.0)


def _probe_vm_stat() -> MemorySnapshot | None:
    total_bytes = _sysctl_int("hw.memsize")
    if total_bytes is None or not shutil.which("vm_stat"):
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=True).stdout
    except subprocess.SubprocessError:
        return None

    page_size = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page_size = int(m.group(1))

    stats: dict[str, int] = {}
    for line in out.splitlines():
        km = re.match(r'"?([^":]+)"?:\s+(\d+)\.', line)
        if km:
            stats[km.group(1).strip()] = int(km.group(2))

    def pages(*names: str) -> int:
        return sum(stats.get(n, 0) for n in names)

    # macOS reclaims free + speculative + purgeable + clean file-backed pages
    # under pressure; those are what a new allocation can actually take.
    available_pages = pages(
        "Pages free", "Pages speculative", "Pages purgeable", "File-backed pages"
    )
    available_bytes = available_pages * page_size
    total = total_bytes / GB
    available = min(available_bytes / GB, total)
    return MemorySnapshot(
        total_gb=total,
        available_gb=available,
        used_gb=total - available,
        swap_used_gb=_macos_swap_used_gb(),
        source="vm_stat",
    )


def _probe_psutil() -> MemorySnapshot | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    try:
        swap_used = psutil.swap_memory().used / GB
    except Exception:  # pragma: no cover - platform dependent
        swap_used = 0.0
    return MemorySnapshot(
        total_gb=vm.total / GB,
        available_gb=vm.available / GB,
        used_gb=(vm.total - vm.available) / GB,
        swap_used_gb=swap_used,
        source="psutil",
    )


def _probe_proc_meminfo() -> MemorySnapshot | None:
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    info[key] = int(parts[0]) * 1024
    except OSError:
        return None
    total = info.get("MemTotal", 0) / GB
    available = info.get("MemAvailable", info.get("MemFree", 0)) / GB
    swap_used = (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / GB
    if total <= 0:
        return None
    return MemorySnapshot(
        total_gb=total,
        available_gb=available,
        used_gb=total - available,
        swap_used_gb=max(swap_used, 0.0),
        source="proc_meminfo",
    )


def snapshot() -> MemorySnapshot:
    probes = (
        (_probe_psutil, True),
        (_probe_vm_stat, platform.system() == "Darwin"),
        (_probe_proc_meminfo, True),
    )
    for probe, enabled in probes:
        if not enabled:
            continue
        snap = probe()
        if snap is not None:
            return snap
    # Last resort: report a conservative unknown state rather than lying about
    # having headroom.
    return MemorySnapshot(0.0, 0.0, 0.0, 0.0, "unavailable")


def process_rss_gb(pid: int) -> float:
    """Resident set size of a child runner process, in GB. 0.0 if unknown."""
    try:
        import psutil  # type: ignore

        return psutil.Process(pid).memory_info().rss / GB
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        return int(out) * 1024 / GB
    except (subprocess.SubprocessError, ValueError):
        return 0.0
