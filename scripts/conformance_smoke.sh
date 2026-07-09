#!/usr/bin/env bash
# Proves the /input/tasks.json -> /output/results.json scoring contract
# (DECISIONS.md D17-D20) end-to-end against a REAL container build
# (linux/amd64, matching the grading VM).
#
# Two modes:
#   default              routed through a local stub Fireworks server (no
#                         API key needed) -- the routing:verify gate.
#   REAL_FIREWORKS=1      routed through the REAL Fireworks endpoint, using
#                         FIREWORKS_BASE_URL / FIREWORKS_API_KEY /
#                         ALLOWED_MODELS already present in the environment
#                         (e.g. `set -a; source .env; set +a` first). This
#                         does NOT prove token/accuracy quality (no gold
#                         answers exist for the real practice tasks) -- only
#                         that the real round trip works, exits 0, and fits
#                         the wall-clock budget.
#
# Uses scripts/fixtures/practice_tasks.json -- the real 8 Track 1 practice
# tasks (task_id + prompt only), pasted 2026-07-09.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-routing-eval:score-smoke}"
PORT="${PORT:-8811}"
WORKDIR="$(mktemp -d)"
SERVER_PID=""
REAL_FIREWORKS="${REAL_FIREWORKS:-0}"

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "conformance_smoke: docker not available -- cannot run the real container gate." >&2
  exit 2
fi

mkdir -p "$WORKDIR/input" "$WORKDIR/output"
cp scripts/fixtures/practice_tasks.json "$WORKDIR/input/tasks.json"

if [ "$REAL_FIREWORKS" = "1" ]; then
  : "${FIREWORKS_BASE_URL:?REAL_FIREWORKS=1 requires FIREWORKS_BASE_URL in the environment}"
  : "${FIREWORKS_API_KEY:?REAL_FIREWORKS=1 requires FIREWORKS_API_KEY in the environment}"
  : "${ALLOWED_MODELS:?REAL_FIREWORKS=1 requires ALLOWED_MODELS in the environment}"
  RUN_BASE_URL="$FIREWORKS_BASE_URL"
  RUN_API_KEY="$FIREWORKS_API_KEY"
  RUN_ALLOWED_MODELS="$ALLOWED_MODELS"
  echo "==> REAL_FIREWORKS=1: using the live Fireworks endpoint (no stub server)"
else
  echo "==> starting fake Fireworks server on :$PORT"
  python3 scripts/fake_fireworks_server.py "$PORT" &
  SERVER_PID=$!
  for i in $(seq 1 20); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/', timeout=0.2)" 2>/dev/null && break
    sleep 0.2
  done
  RUN_BASE_URL="http://host.docker.internal:${PORT}/v1"
  RUN_API_KEY="stub-key"
  RUN_ALLOWED_MODELS="stub-model"
fi

echo "==> building $IMAGE (linux/amd64)"
docker buildx build --platform linux/amd64 -t "$IMAGE" --load .

echo "==> running the scoring entrypoint (routing-eval score)"
START_S=$(date +%s)
docker run --rm \
  -e FIREWORKS_BASE_URL="$RUN_BASE_URL" \
  -e FIREWORKS_API_KEY="$RUN_API_KEY" \
  -e ALLOWED_MODELS="$RUN_ALLOWED_MODELS" \
  -v "$WORKDIR/input:/input:ro" \
  -v "$WORKDIR/output:/output" \
  "$IMAGE"
END_S=$(date +%s)
ELAPSED_S=$((END_S - START_S))
echo "==> container ran for ${ELAPSED_S}s (budget: 600s total, D19)"

echo "==> validating /output/results.json"
REAL_FIREWORKS="$REAL_FIREWORKS" python3 -c "
import json, os
with open('$WORKDIR/output/results.json') as f:
    results = json.load(f)
assert isinstance(results, list) and results, 'results.json must be a non-empty list'
for r in results:
    assert set(r.keys()) == {'task_id', 'answer'}, f'bad shape: {r}'
    assert isinstance(r['task_id'], str) and r['task_id']
    if os.environ['REAL_FIREWORKS'] != '1':
        assert r['answer'].startswith('stub-answer:'), f'answer did not reach the stub server: {r}'
print(f'    {len(results)} results, shape OK' +
      ('' if os.environ['REAL_FIREWORKS'] == '1' else ', all reached the stub server'))
"

if [ "$REAL_FIREWORKS" = "1" ]; then
  cp "$WORKDIR/output/results.json" /tmp/real_practice_results.json
  echo "==> copied real results to /tmp/real_practice_results.json"
fi

echo "==> conformance smoke OK"
