"""Single-active-model policy and the unified-memory headroom guard.

Design note — why runners are child processes
---------------------------------------------
"Unload the model" inside a long-lived Python process means dropping references
and hoping MLX's allocator returns the Metal buffers. In practice a fragmented
32GB pool does not come all the way back, and the next load starts from a worse
baseline. Running each job in a child process makes unloading an OS-level
guarantee: the process exits, every buffer is freed, the pool is clean.

So `ModelManager` owns the *policy* (who is resident, may the next thing load)
and the runner subprocess is the *mechanism*. The two invariants it enforces:

  1. At most one model is resident at any moment. A switch from LTX-2.3 to
     FLUX.1 [dev] unloads LTX first and waits for the memory to actually come
     back before FLUX is allowed to start — the two never overlap.
  2. No load may push projected total allocation past the ceiling (~28GB of
     32GB). If it would, the job is held or refused. Swapping unified memory
     is treated as a failure mode to prevent, not a slowdown to absorb.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .config import get_config
from .memory import MemorySnapshot, process_rss_gb, snapshot

log = logging.getLogger("vidgen.models")


class MemoryGuardError(RuntimeError):
    """Raised when a job cannot fit under the memory ceiling."""

    def __init__(self, message: str, *, projected_gb: float, snap: MemorySnapshot):
        super().__init__(message)
        self.projected_gb = projected_gb
        self.snapshot = snap


@dataclass
class ResidentModel:
    model_key: str
    label: str
    quant: str
    preset: str | None
    estimated_gb: float
    job_id: str | None
    loaded_at: float
    pid: int | None = None
    peak_rss_gb: float = 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["resident_seconds"] = round(time.time() - self.loaded_at, 1)
        return d


@dataclass
class HeadroomDecision:
    ok: bool
    reason: str
    required_gb: float
    projected_total_gb: float
    ceiling_gb: float
    snapshot: MemorySnapshot = field(default_factory=snapshot)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "required_gb": round(self.required_gb, 2),
            "projected_total_gb": round(self.projected_total_gb, 2),
            "ceiling_gb": self.ceiling_gb,
            "memory": self.snapshot.as_dict(),
        }


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resident: ResidentModel | None = None
        self._proc: subprocess.Popen | None = None
        self._events: deque[dict] = deque(maxlen=200)
        self._cfg = get_config()

    # -- introspection ----------------------------------------------------

    @property
    def resident(self) -> ResidentModel | None:
        with self._lock:
            return self._resident

    def events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._events)[-limit:]

    def status(self) -> dict:
        snap = snapshot()
        with self._lock:
            res = self._resident.as_dict() if self._resident else None
        return {
            "resident_model": res,
            "memory": snap.as_dict(),
            "ceiling_gb": self._cfg.memory_ceiling_gb,
            "safety_margin_gb": self._cfg.memory_safety_margin_gb,
            "headroom_gb": round(max(self._cfg.memory_ceiling_gb - snap.used_gb, 0.0), 2),
            "swapping": snap.swap_used_gb > 1.0,
        }

    def _emit(self, kind: str, **fields) -> None:
        event = {"ts": time.time(), "event": kind, **fields}
        self._events.append(event)
        log.info("%s %s", kind, fields)

    # -- policy -----------------------------------------------------------

    def check_headroom(self, required_gb: float, *, label: str = "") -> HeadroomDecision:
        """Can `required_gb` be allocated right now without crossing the ceiling?

        Assumes any currently-resident model has already been unloaded — call
        `unload()` first when switching. That ordering is deliberate: it is what
        makes an immediate LTX-2.3 -> FLUX.1 switch safe.
        """
        cfg = self._cfg
        snap = snapshot()
        margin = cfg.memory_safety_margin_gb
        projected = snap.used_gb + required_gb + margin

        if snap.source == "unavailable":
            return HeadroomDecision(
                ok=False,
                reason="Could not read unified memory state; refusing to load blind.",
                required_gb=required_gb,
                projected_total_gb=projected,
                ceiling_gb=cfg.memory_ceiling_gb,
                snapshot=snap,
            )

        if required_gb + margin > cfg.memory_ceiling_gb:
            return HeadroomDecision(
                ok=False,
                reason=(
                    f"{label or 'This job'} needs ~{required_gb:.1f}GB, which alone exceeds the "
                    f"{cfg.memory_ceiling_gb:.0f}GB ceiling. Pick a smaller quantization, a lower "
                    f"resolution, or fewer frames."
                ),
                required_gb=required_gb,
                projected_total_gb=projected,
                ceiling_gb=cfg.memory_ceiling_gb,
                snapshot=snap,
            )

        if projected > cfg.memory_ceiling_gb:
            over = projected - cfg.memory_ceiling_gb
            return HeadroomDecision(
                ok=False,
                reason=(
                    f"{label or 'This job'} needs ~{required_gb:.1f}GB but only "
                    f"{max(cfg.memory_ceiling_gb - snap.used_gb, 0.0):.1f}GB is free under the "
                    f"{cfg.memory_ceiling_gb:.0f}GB ceiling ({snap.used_gb:.1f}GB in use by the system). "
                    f"Over by {over:.1f}GB — close memory-heavy apps (browser tabs especially) and it "
                    f"will start on the next check."
                ),
                required_gb=required_gb,
                projected_total_gb=projected,
                ceiling_gb=cfg.memory_ceiling_gb,
                snapshot=snap,
            )

        if required_gb + margin > snap.available_gb:
            return HeadroomDecision(
                ok=False,
                reason=(
                    f"{label or 'This job'} needs ~{required_gb:.1f}GB but macOS reports only "
                    f"{snap.available_gb:.1f}GB reclaimable right now. Holding rather than swapping."
                ),
                required_gb=required_gb,
                projected_total_gb=projected,
                ceiling_gb=cfg.memory_ceiling_gb,
                snapshot=snap,
            )

        return HeadroomDecision(
            ok=True,
            reason="ok",
            required_gb=required_gb,
            projected_total_gb=projected,
            ceiling_gb=cfg.memory_ceiling_gb,
            snapshot=snap,
        )

    def unload(self, *, reason: str = "switch") -> None:
        """Free the resident model and wait for the memory to actually return."""
        with self._lock:
            resident = self._resident
            proc = self._proc
            if resident is None and proc is None:
                return
            self._resident = None
            self._proc = None

        if proc is not None and proc.poll() is None:
            self._terminate(proc)

        before = snapshot().used_gb
        self._wait_for_release(before)
        if resident is not None:
            self._emit(
                "model_unloaded",
                model=resident.model_key,
                quant=resident.quant,
                preset=resident.preset,
                reason=reason,
                peak_rss_gb=round(resident.peak_rss_gb, 2),
                used_gb_after=round(snapshot().used_gb, 2),
            )

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
            else:  # pragma: no cover
                proc.terminate()
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:  # pragma: no cover
                    proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        except ProcessLookupError:  # pragma: no cover
            pass

    @staticmethod
    def _wait_for_release(used_before_gb: float, timeout: float = 20.0) -> None:
        """Poll until the OS has actually given the memory back (or we give up).

        Without this, a fast job-to-job switch can run its headroom check
        against a stale snapshot that still counts the model we just killed.
        """
        deadline = time.time() + timeout
        last = used_before_gb
        while time.time() < deadline:
            time.sleep(0.4)
            now = snapshot().used_gb
            if now <= last - 0.05:
                last = now
                continue
            return

    # -- execution --------------------------------------------------------

    @contextmanager
    def activate(
        self,
        *,
        model_key: str,
        label: str,
        quant: str,
        preset: str | None,
        estimated_gb: float,
        job_id: str | None = None,
    ) -> Iterator["ActiveModel"]:
        """Make a model the single resident model, or raise MemoryGuardError.

        Unloads whatever is resident first — unconditionally, even for the same
        model key — so two models are never alive at the same instant.
        """
        with self._lock:
            if self._resident is not None:
                self.unload(reason=f"making room for {model_key}")

            decision = self.check_headroom(estimated_gb, label=label)
            if not decision.ok:
                self._emit(
                    "load_refused",
                    model=model_key,
                    quant=quant,
                    preset=preset,
                    required_gb=round(estimated_gb, 2),
                    reason=decision.reason,
                )
                raise MemoryGuardError(
                    decision.reason, projected_gb=decision.projected_total_gb, snap=decision.snapshot
                )

            resident = ResidentModel(
                model_key=model_key,
                label=label,
                quant=quant,
                preset=preset,
                estimated_gb=estimated_gb,
                job_id=job_id,
                loaded_at=time.time(),
            )
            self._resident = resident
            self._emit(
                "model_loading",
                model=model_key,
                quant=quant,
                preset=preset,
                job_id=job_id,
                estimated_gb=round(estimated_gb, 2),
                free_under_ceiling_gb=round(
                    decision.ceiling_gb - decision.snapshot.used_gb, 2
                ),
            )

        try:
            yield ActiveModel(self, resident)
        finally:
            self.unload(reason="job finished")

    def run_subprocess(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        on_line: Callable[[str], None] | None = None,
        timeout: int | None = None,
    ) -> tuple[int, float]:
        """Run a runner command, sampling its RSS. Returns (exit_code, peak_gb)."""
        cfg = self._cfg
        timeout = timeout or cfg.job_timeout_seconds
        full_env = {**os.environ, **(env or {})}
        log_fh = open(log_path, "a", buffering=1) if log_path else None

        popen_kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "cwd": str(cwd) if cwd else None,
            "env": full_env,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(list(argv), **popen_kwargs)
        with self._lock:
            self._proc = proc
            if self._resident:
                self._resident.pid = proc.pid

        peak = 0.0
        stop_sampler = threading.Event()

        def sample() -> None:
            nonlocal peak
            first = True
            while True:
                # Sample immediately so short jobs are measured too, then settle
                # into a low-frequency poll.
                if not first and stop_sampler.wait(1.5):
                    break
                if first:
                    time.sleep(0.2)
                    first = False
                rss = process_rss_gb(proc.pid)
                if rss > peak:
                    peak = rss
                    with self._lock:
                        if self._resident and self._resident.pid == proc.pid:
                            self._resident.peak_rss_gb = rss
                snap = snapshot()
                if snap.swap_used_gb > 2.0:
                    self._emit(
                        "swap_pressure",
                        swap_used_gb=round(snap.swap_used_gb, 2),
                        note="unified memory is swapping; output will be slow",
                    )

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()

        deadline = time.time() + timeout
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if log_fh:
                    log_fh.write(line + "\n")
                if on_line:
                    on_line(line)
                if time.time() > deadline:
                    self._terminate(proc)
                    raise TimeoutError(f"runner exceeded {timeout}s")
            code = proc.wait(timeout=max(deadline - time.time(), 1))
        except TimeoutError:
            raise
        except subprocess.TimeoutExpired:
            self._terminate(proc)
            raise TimeoutError(f"runner exceeded {timeout}s")
        finally:
            stop_sampler.set()
            sampler.join(timeout=3)
            if log_fh:
                log_fh.close()
            with self._lock:
                if self._proc is proc:
                    self._proc = None

        return code, peak

    def cancel_running(self) -> bool:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        self._emit("job_cancelled", pid=proc.pid)
        self._terminate(proc)
        return True


@dataclass
class ActiveModel:
    manager: ModelManager
    resident: ResidentModel

    def run(self, argv: Sequence[str], **kwargs) -> tuple[int, float]:
        return self.manager.run_subprocess(argv, **kwargs)


_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelManager()
        return _manager


def reset_manager() -> None:
    """Test hook."""
    global _manager
    with _manager_lock:
        _manager = None
