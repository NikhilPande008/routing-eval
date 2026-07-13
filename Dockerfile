# Routing agent container.
#
# Runtime footprint is the base Python image plus this package -- ZERO third-party
# runtime dependencies. setuptools/wheel are pulled only at BUILD time to build the
# package; nothing third-party is installed for runtime.
#
# ENTRYPOINT is the CLI. Default CMD is the scoring path (`score`, per the
# official Track 1 guide, DECISIONS.md D20): reads /input/tasks.json, answers
# each task with one Fireworks call (env: FIREWORKS_BASE_URL /
# FIREWORKS_API_KEY / ALLOWED_MODELS), writes /output/results.json, exits 0.
# Override CMD (e.g. `run`, `frontier`, `--help`) for the eval/calibration CLI.
#
# Pin by digest for full reproducibility if desired:
#   FROM python:3.12-slim@sha256:<digest>
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: the ONLY system library the llama.cpp ubuntu-x64 release binaries
# need that python:3.12-slim doesn't already carry (verified by ldd, D44).
# A C runtime lib, not a Python dependency: the package stays zero-runtime-deps.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Local tier, second attempt (D48 ladder, PLAN-TOKEN-OPT.md): bundled
# llama.cpp CPU server + Qwen2.5-1.5B (2x faster than D45's failed 3B on
# the same throttled measurement). The D45-lesson redesign lives in
# docker/entrypoint.sh: a startup speed probe DISABLES the local tier on
# slow hardware (pure remote-only run -- costs tokens, never accuracy),
# per-request timeouts are 10s (was 25s), and generations are capped via
# LOCAL_MAX_TOKENS. Local answers are scored at ZERO token cost (D26);
# every validation failure escalates to Fireworks on the category's
# original live-validated template. These layers go FIRST: they are ~1.1GB
# and never change -- code edits below must not invalidate them.
COPY docker/llama /app/llama
COPY docker/models /app/models

# Copy only what the package needs to build + run (see .dockerignore).
COPY pyproject.toml README.md ./
COPY routing_eval ./routing_eval

# Install the package itself. No runtime deps are declared, so this pulls only the
# build backend, not third-party runtime libraries. The import check fails the
# build early if anything is broken.
RUN pip install . && python -c "import routing_eval; print('build import OK')"

# Create the scoring mount points so an evaluator dry-run without bind mounts
# still exits cleanly. The container intentionally runs as root: on Linux
# graders, /output is commonly a root-owned bind mount, and a non-root user can
# crash while writing /output/results.json even when the entrypoint is correct.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /input /output \
    && chown -R appuser:appuser /input /output

COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/llama/llama-server

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["score"]
