#!/usr/bin/env bash
# Build the image and run the README quickstart INSIDE the clean image. This is
# the real clean-environment gate that replaces the venv stand-in in
# routing:verify. Requires a working Docker daemon.
#
# Exit codes: 0 = gate passed; 2 = docker unavailable (caller may fall back to the
# venv stand-in); other non-zero = the gate FAILED.
set -euo pipefail

IMAGE="${IMAGE:-routing-eval:smoke}"
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "container_smoke: docker not installed -- container gate cannot run here." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "container_smoke: docker daemon not reachable." >&2
  exit 2
fi

echo "==> building $IMAGE"
docker build -t "$IMAGE" .

echo "==> entrypoint responds"
docker run --rm "$IMAGE" --help >/dev/null && echo "    entrypoint OK"

echo "==> quickstart inside the clean image (install -> run -> frontier)"
docker run --rm --entrypoint sh "$IMAGE" -c '
  routing-eval run --dataset standin --n 60 --out /tmp/r.json &&
  routing-eval frontier --records /tmp/r.json --accuracy-threshold 0.80 | tail -3 &&
  echo "    container quickstart OK"'

echo "==> container smoke OK"
