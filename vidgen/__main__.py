"""`python -m vidgen` — start the service."""

from __future__ import annotations

import argparse

import uvicorn

from .config import get_config


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(prog="vidgen", description="Local generation queue + dashboard")
    ap.add_argument("--host", default=cfg.host)
    ap.add_argument("--port", type=int, default=cfg.port)
    ap.add_argument("--reload", action="store_true", help="dev auto-reload")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    uvicorn.run(
        "vidgen.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        # One worker process. The queue is deliberately serial.
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
