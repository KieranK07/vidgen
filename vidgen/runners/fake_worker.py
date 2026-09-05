"""Stand-in renderer used when VIDGEN_DRY_RUN=1.

Runs as a real child process, really allocates memory, really emits progress
lines and really writes an output file — so the queue, the memory guard, the
RSS sampler and the dashboard can all be exercised end to end without the
model weights being present. Nothing here touches MLX.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path


def write_png(path: Path, width: int, height: int, seed: int) -> None:
    width = max(16, min(width, 1024))
    height = max(16, min(height, 1024))
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            r = (x * 255 // width) ^ (seed & 0xFF)
            g = (y * 255 // height) ^ ((seed >> 8) & 0xFF)
            b = ((x + y) * 255 // (width + height)) ^ ((seed >> 16) & 0xFF)
            rows += bytes((r & 0xFF, g & 0xFF, b & 0xFF))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(rows), 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def write_stub_video(path: Path, frames: int, fps: int) -> None:
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        path.write_bytes(b"VIDGEN_DRY_RUN_STUB\n")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "frame.png"
        write_png(src, 512, 288, 7)
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-loop", "1", "-i", str(src),
                "-t", str(max(frames / max(fps, 1), 1)),
                "-r", str(fps), "-pix_fmt", "yuv420p", str(path),
            ],
            check=False,
        )
    if not path.exists() or path.stat().st_size == 0:
        path.write_bytes(b"VIDGEN_DRY_RUN_STUB\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--kind", default="image", choices=["image", "video"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allocate-mb", type=int, default=256)
    ap.add_argument("--step-seconds", type=float, default=0.15)
    args = ap.parse_args()

    print(f"[dry-run] pretending to load model, allocating {args.allocate_mb}MB", flush=True)
    ballast = bytearray(max(args.allocate_mb, 1) * 1024 * 1024)
    for i in range(0, len(ballast), 4096):  # touch pages so RSS is real
        ballast[i] = 1

    for step in range(1, args.steps + 1):
        time.sleep(args.step_seconds)
        print(f"step {step}/{args.steps}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "video":
        write_stub_video(out, args.frames, args.fps)
    else:
        write_png(out, args.width, args.height, args.seed)
    print(f"[dry-run] wrote {out}", flush=True)
    del ballast
    return 0


if __name__ == "__main__":
    sys.exit(main())
