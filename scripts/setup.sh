#!/usr/bin/env bash
# One-time setup: virtualenv, service deps, optional model stacks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "warning: vidgen targets Apple Silicon macOS. Continuing anyway." >&2
fi

PY="${PYTHON:-python3}"
echo "==> creating .venv with $PY"
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

echo "==> installing the service"
pip install -e .

if [ "${WITH_MODELS:-1}" = "1" ]; then
  echo "==> installing mlx-video (LTX-2 / Wan2.2)"
  pip install "git+https://github.com/Blaizzy/mlx-video.git"
  echo "==> installing mflux (FLUX.1)"
  pip install mflux
else
  echo "==> skipping model stacks (WITH_MODELS=0)"
fi

[ -f .env ] || { cp .env.example .env; echo "==> wrote .env from .env.example"; }

cat <<'DONE'

Setup complete.

  ./scripts/vidgen.sh start      # run it now
  ./scripts/vidgen.sh install    # run at login (launchd)

Weights are downloaded lazily on first use of each model — the first LTX-2.3
or FLUX run will take a while and needs disk space. See README "Model downloads".
DONE
