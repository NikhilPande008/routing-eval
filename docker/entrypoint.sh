#!/bin/sh
# Container entrypoint (2026-07-11, local tier): start the bundled llama.cpp
# server if a model is bundled, wait for it to become healthy, then exec the
# routing-eval CLI. Fail-safe by construction: ANY problem bringing the local
# server up leaves LOCAL_BASE_URL unset, and policy.py treats local=None as
# "every local-tier category escalates to Fireworks" -- the pre-local-tier
# behavior, which costs tokens but never accuracy.
set -u

GGUF="${LOCAL_GGUF_PATH:-/app/models/local.gguf}"
LLAMA_BIN="${LLAMA_SERVER_BIN:-/app/llama/llama-server}"
PORT="${LOCAL_PORT:-8080}"

if [ -f "$GGUF" ] && [ -x "$LLAMA_BIN" ]; then
    # Thread count: a cgroup cpu quota (e.g. docker run --cpus 2) does NOT
    # change nproc -- nproc reports the HOST's CPUs, and oversubscribing a
    # 2-cpu quota with 12 threads thrashes. Prefer LOCAL_THREADS, then the
    # cgroup v2 quota, then nproc.
    THREADS="${LOCAL_THREADS:-}"
    if [ -z "$THREADS" ] && [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r _quota _period < /sys/fs/cgroup/cpu.max || true
        case "$_quota" in
            ''|max|*[!0-9]*) ;;
            *) [ -n "$_period" ] && [ "$_period" -gt 0 ] && THREADS=$(( (_quota + _period - 1) / _period )) ;;
        esac
    fi
    [ -z "$THREADS" ] && THREADS="$(nproc 2>/dev/null || echo 2)"
    LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA_BIN" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
        --threads "$THREADS" --ctx-size "${LOCAL_CTX:-3072}" \
        --parallel 1 >/tmp/llama-server.log 2>&1 &
    SRV_PID=$!
    # /health returns 503 while the model loads, 200 when ready. 90s budget:
    # only a FAILED start ever burns all of it, and 90s still leaves ample
    # room in the 10-minute total (D19) for the all-remote fallback path.
    tries=0
    until python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)" 2>/dev/null; do
        tries=$((tries+1))
        if [ "$tries" -ge 90 ] || ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "entrypoint: llama-server not healthy after ${tries} tries -- continuing remote-only" >&2
            tail -5 /tmp/llama-server.log >&2 || true
            SRV_PID=""
            break
        fi
        sleep 1
    done
    # Speed probe (D48 redesign, PLAN-TOKEN-OPT.md 5.0): the grading VM's CPU
    # class is unknowable offline (the D45 lesson). Time one tiny completion;
    # if even 4 tokens take longer than LOCAL_PROBE_MAX_S (default 6s), the
    # hardware is too slow for local answers to beat their per-request
    # timeouts -- disable the local tier for this run entirely, which is
    # exactly the proven remote-only configuration (costs tokens, never
    # accuracy).
    if [ -n "$SRV_PID" ]; then
        PROBE_S=$(python3 - <<'PY'
import json, time, urllib.request
body = json.dumps({"model": "probe", "messages": [{"role": "user", "content": "Say OK"}],
                   "max_tokens": 4, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    urllib.request.urlopen(req, timeout=20)
    print(f"{time.time()-t0:.1f}")
except Exception:
    print("999")
PY
        )
        # D52: limit relaxed 6s -> 10s. The old 6s limit was calibrated on
        # this machine's emulation and may have been disabling the tier on
        # marginally slower grading hardware; the batch-level time governor
        # (PolicyRouter.local_budget_s) now bounds total spend, so a slower
        # probe no longer risks the 10-minute budget -- it just means fewer
        # tasks fit inside the local budget before it exhausts.
        echo "entrypoint: local speed probe ${PROBE_S}s (limit ${LOCAL_PROBE_MAX_S:-10}s)" >&2
        SLOWEST=$(printf '%s\n' "$PROBE_S" "${LOCAL_PROBE_MAX_S:-10}" | sort -g | tail -1)
        if [ "$SLOWEST" != "${LOCAL_PROBE_MAX_S:-10}" ]; then
            echo "entrypoint: hardware too slow -- local tier DISABLED, remote-only run" >&2
            SRV_PID=""
        fi
    fi
    if [ -n "$SRV_PID" ]; then
        export LOCAL_BASE_URL="http://127.0.0.1:${PORT}/v1"
        export LOCAL_MODEL="${LOCAL_MODEL:-bundled-local}"
        # 256: knowledge explanations and math working (D52) need more room
        # than the short-category 128; a cap only costs anything if used.
        export LOCAL_MAX_TOKENS="${LOCAL_MAX_TOKENS:-256}"
        echo "entrypoint: local model ready at ${LOCAL_BASE_URL}" >&2
    fi
else
    echo "entrypoint: no bundled local model -- remote-only" >&2
fi

exec routing-eval "$@"
